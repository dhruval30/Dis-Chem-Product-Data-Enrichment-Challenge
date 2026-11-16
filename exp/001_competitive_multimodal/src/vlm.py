from __future__ import annotations

import gc
import json
import platform
import os
import traceback
from pathlib import Path

from .common import ROOT, BudgetExceeded, digest, durable_event, file_hash, read_json, versions
from .image_processing import load_image


class _ObjectPairs(list):
    pass


def parse_json(text, *, recover_observations=False, parser_events=None):
    text = text.strip()
    def reject_constant(value):
        raise ValueError('Non-finite JSON constant: ' + value)
    decoder = json.JSONDecoder(object_pairs_hook=_ObjectPairs, parse_constant=reject_constant)

    def convert(value, path=(), printed_ok=False):
        if isinstance(value, _ObjectPairs):
            keys = [k for k, _ in value]
            if len(keys) != len(set(keys)):
                raise ValueError('Duplicate JSON keys at ' + '/'.join(map(str, path)))
            if len(path) == 2 and path[0] == 'images':
                printed = dict(value).get('printed', [])
                printed_ok = bool(printed) and all(isinstance(p, _ObjectPairs) and len(p) == 2 and
                    {k for k, _ in p} == {'kind', 'text'} and all(isinstance(v, str) for _, v in p) for p in printed)
            return {k: convert(v, path + (k,), printed_ok) for k, v in value}
        if isinstance(value, list):
            if recover_observations and len(path) == 3 and path[0] == 'images' and path[2] == 'observations':
                expanded = []
                for item in value:
                    keys = [k for k, _ in item] if isinstance(item, _ObjectPairs) else []
                    # Recover only complete, ordered triples. Never choose between competing values.
                    if len(keys) > 3 and keys == ['attribute', 'value', 'evidence'] * (len(keys) // 3):
                        expanded.extend(_ObjectPairs(item[i:i + 3]) for i in range(0, len(item), 3))
                        if parser_events is not None:
                            parser_events.append({'kind': 'observation_objects_recovered',
                                'image_index': path[1], 'count': len(keys) // 3})
                    elif keys and len(keys) != len(set(keys)) and printed_ok:
                        if parser_events is not None:
                            parser_events.append({'kind': 'ambiguous_observation_discarded', 'image_index': path[1],
                                                  'duplicate_keys': sorted({k for k in keys if keys.count(k) > 1})})
                    else:
                        expanded.append(item)
                value = expanded
            return [convert(v, path + (i,), printed_ok) for i, v in enumerate(value)]
        return value
    # Recover one complete object; never repair missing content or competing objects.
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    text = text.removeprefix('\ufeff').strip()
    try:
        value = decoder.decode(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, _ObjectPairs):
        return convert(value)
    if value is not None:
        raise ValueError('Model response must contain an object')
    found = []
    offset = 0
    while offset < len(text):
        start = text.find('{', offset)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text[start:])
        except ValueError as error:
            raise ValueError('Incomplete or malformed JSON object') from error
        if not isinstance(value, _ObjectPairs):
            raise ValueError('Model response must contain an object')
        found.append(value)
        offset = start + end
    if len(found) != 1:
        raise ValueError('Expected exactly one complete JSON object')
    return convert(found[0])


def model_manifest(path):
    path = Path(path)
    if not (path / 'config.json').is_file() or not list(path.glob('*.safetensors')):
        raise ValueError('Download the MLX checkpoint locally before inference')
    # Hash all checkpoint files once per invocation. No implicit remote loading.
    return {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob('*')) if p.is_file()}


def check_processor(path):
    """Exercise the actual processor dependency path without loading weights or generating."""
    import torch
    import torchvision
    from mlx_vlm.utils import load_processor
    path = Path(path).resolve()
    if not (path / 'config.json').is_file():
        raise ValueError('Local checkpoint config is missing')
    processor = load_processor(path, local_files_only=True)
    return {'status': 'loaded', 'processor_class': type(processor).__name__,
            'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'generation_performed': False, 'weights_loaded': False}


def limit_processor_pixels(processor, cfg):
    maximum = cfg.get('processor_max_pixels')
    if maximum is None:
        return {}
    image_processor = getattr(processor, 'image_processor', None)
    if image_processor is None:
        raise ValueError('Processor exposes no image processor; safe pixel limits cannot be verified')
    applied = {}
    for key, value in (('max_pixels', maximum), ('min_pixels', cfg.get('processor_min_pixels'))):
        if value is not None and hasattr(image_processor, key):
            setattr(image_processor, key, value)
            applied[key] = value
    size = getattr(image_processor, 'size', None)
    if isinstance(size, dict) and 'longest_edge' in size:
        size = dict(size)
        size['longest_edge'] = maximum
        if 'shortest_edge' in size and cfg.get('processor_min_pixels') is not None:
            size['shortest_edge'] = cfg['processor_min_pixels']
        image_processor.size = size
        applied['size'] = size
    if 'max_pixels' not in applied and 'size' not in applied:
        raise ValueError('This image processor does not expose supported pixel controls')
    return applied


class VLM:
    def __init__(self, cfg, cache, budget, manifest, seed, diagnostics_dir=None):
        self.cfg, self.cache, self.budget, self.manifest = cfg, cache, budget, manifest
        self.seed = seed
        self.model = self.processor = self.model_config = None
        self.record_key = None
        self.record_call_limit = 0
        self.loaded = False
        self.generation_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.diagnostics_dir = Path(diagnostics_dir) if diagnostics_dir else cache.root / 'vlm_diagnostics'
        self.record_id = None
        self.active_stage = None
        self.attempt_scope = None

    def log(self, event, stage=None, **metadata):
        durable_event(self.diagnostics_dir / 'vlm.jsonl', event,
            stage=stage or self.active_stage, record_id=self.record_id, record_key=self.record_key,
            profile=self.cfg.get('profile', 'competitive'), attempt_scope=self.attempt_scope, **metadata)

    def save_prompt(self, stage, text, kind='assembled'):
        key = digest([self.record_key, stage, kind, text])
        path = self.diagnostics_dir / 'prompts' / (key + '.txt')
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open('w', encoding='utf-8') as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        self.log('prompt_saved', stage=stage, prompt_kind=kind, prompt_path=str(path),
                 prompt_hash=digest(text), prompt_chars=len(text))
        return path

    def load(self):
        if self.model is not None:
            self.log('model_load_reused')
            return
        self.log('model_load_start', model_path=self.cfg['path'], memory_limit_gb=self.cfg['memory_limit_gb'])
        if self.cfg['backend'] != 'mlx_vlm':
            raise ValueError('Only the documented mlx_vlm backend is implemented')
        if platform.system() != 'Darwin' or platform.machine() != 'arm64':
            raise RuntimeError('MLX backend requires native Apple Silicon Python')
        import mlx.core as mx
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        mx.random.seed(self.seed)
        if hasattr(mx, 'set_memory_limit'):
            mx.set_memory_limit(int(self.cfg['memory_limit_gb'] * 1024**3))
        config = read_json(Path(self.cfg['path']) / 'config.json')
        quant = config.get('quantization', config.get('quantization_config', {}))
        if quant.get('bits') != self.cfg['quantisation_bits']:
            raise ValueError('Checkpoint quantisation differs from configuration')
        # Publish loaded state only after weights, processor and configuration all succeed.
        model, processor = load(self.cfg['path'], local_files_only=True)
        model_config = load_config(self.cfg['path'], local_files_only=True)
        self.log('processor_limits_start')
        applied = limit_processor_pixels(processor, self.cfg)
        self.log('processor_limits_complete', applied=applied)
        self.model, self.processor, self.model_config = model, processor, model_config
        self.loaded = True
        self.budget.check()
        self.log('model_load_complete', processor_class=type(processor).__name__)

    def ask(self, stage, prompt, paths=()):
        self.active_stage = stage
        self.log('request_start', image_count=len(paths), prompt_chars=len(prompt))
        try:
            return self._ask(stage, prompt, paths)
        except BaseException as error:
            self.log('request_failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            self.log('cleanup_start')
            try:
                if self.loaded:
                    import mlx.core as mx
                    mx.clear_cache()
                self.log('cleanup_complete')
            except BaseException as error:
                self.log('cleanup_failed', error=str(error), traceback=traceback.format_exc())
                raise

    def _ask(self, stage, prompt, paths=()):
        if len(paths) > self.cfg['max_images']:
            raise ValueError('Image count exceeds the selected profile limit')
        if len(prompt) > self.cfg['max_input_chars']:
            raise ValueError('Prompt exceeds input character budget; no silent evidence truncation')
        key = digest([stage, prompt, [file_hash(p) for p in paths], self.cfg, self.manifest, versions(), self.seed,
                      file_hash(ROOT / 'src/vlm.py'), file_hash(ROOT / 'src/image_processing.py'),
                      file_hash(ROOT / 'src/visual.py')])
        old = self.cache.get('vlm', key)
        if old is not None:
            if stage in ('image_extraction', 'image_verification'):
                from .visual import validate_response
                validate_response(old['parsed'])
            self.log('cache_hit', call_key=key)
            return old['parsed']
        try:
            self.load()
        except Exception:
            self.failure_count += 1
            self.log('model_load_failed', traceback=traceback.format_exc())
            raise
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        import mlx.core as mx
        images = []
        for index, path in enumerate(paths):
            self.log('image_loading_start', image_index=index, image_path=str(path))
            from PIL import Image
            with Image.open(path) as original:
                original_size = list(original.size)
            self.log('image_loading_complete', image_index=index, original_size=original_size)
            self.log('image_preprocessing_start', image_index=index, max_side=self.cfg['image_max_side'])
            image = load_image(path, self.cfg['image_max_side'])
            images.append(image)
            self.log('image_preprocessing_complete', image_index=index, prepared_size=list(image.size))
        self.log('chat_template_start', image_count=len(images))
        formatted = apply_chat_template(self.processor, self.model_config, prompt, num_images=len(images))
        self.save_prompt(stage, formatted, kind='chat_formatted')
        self.log('chat_template_complete', formatted_chars=len(formatted))
        cap = self.cfg.get('max_prompt_tokens')
        if cap is not None:
            tokenizer = getattr(self.processor, 'tokenizer', self.processor)
            prompt_tokens = len(tokenizer.encode(formatted, add_special_tokens=False))
            self.log('prompt_token_count', prompt_tokens=prompt_tokens, maximum=cap)
            if prompt_tokens > cap:
                raise ValueError('Formatted text prompt exceeds the selected token limit')
        attempt_key = digest([key, self.attempt_scope]) if self.attempt_scope else key
        last = None
        for _ in range(1 + self.budget.cfg['retry_count']):
            self.budget.check()
            self.cache.attempts('vlm', attempt_key, 1 + self.budget.cfg['retry_count'])
            if self.record_key:
                record_attempt = digest([self.record_key, self.attempt_scope]) if self.attempt_scope else self.record_key
                self.cache.attempts('record_vlm', record_attempt, self.record_call_limit)
            try:
                mx.random.seed(self.seed + int(key[:8], 16) % 100000)
                self.budget.state['calls'] += 1
                self.generation_count += 1
                self.budget.save()
                self.log('vlm_call_start', call_key=key, attempt_key=attempt_key,
                         image_count=len(images), prepared_sizes=[list(x.size) for x in images],
                         max_tokens=self.cfg.get('stage_max_tokens', {}).get(stage, self.cfg['max_tokens']),
                         actual_generation_calls=self.budget.state['calls'])
                result = generate(self.model, self.processor, formatted, image=images or None,
                                  max_tokens=self.cfg.get('stage_max_tokens', {}).get(stage, self.cfg['max_tokens']),
                                  temperature=self.cfg['temperature'], verbose=False)
                self.log('vlm_call_complete', call_key=key)
                raw = result if isinstance(result, str) else result.text
                self.cache.put('vlm_raw', key, {'text': raw})
                self.log('parsing_start', call_key=key, output_chars=len(raw))
                parser_events = []
                parsed = parse_json(raw, recover_observations=stage in ('image_extraction', 'image_verification'),
                                    parser_events=parser_events)
                for event in parser_events:
                    self.log('json_structure_recovered', call_key=key, recovery=event)
                if stage in ('image_extraction', 'image_verification'):
                    from .visual import validate_response
                    validate_response(parsed)
                self.log('parsing_complete', call_key=key)
                self.cache.put('vlm', key, {'text': raw, 'parsed': parsed, 'parser_events': parser_events})
                if hasattr(mx, 'get_peak_memory'):
                    self.budget.state['mlx_peak_bytes'] = max(
                        self.budget.state.get('mlx_peak_bytes', 0), mx.get_peak_memory())
                self.budget.save()
                self.budget.check()
                self.success_count += 1
                return parsed
            except BudgetExceeded:
                raise
            except Exception as error:
                self.failure_count += 1
                self.log('vlm_attempt_failed', call_key=key, error=str(error), traceback=traceback.format_exc())
                last = error
                self.cache.put('vlm_errors', key, {'error': str(error)})
                self.budget.save()
        raise RuntimeError(f'{stage} failed after allowed attempts: {last}')

    def close(self):
        self.log('model_unload_start')
        self.model = self.processor = None
        gc.collect()
        import mlx.core as mx
        mx.clear_cache()
        self.log('model_unload_complete')
