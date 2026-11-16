from __future__ import annotations

import copy
import fcntl
import json
import platform
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from .common import ROOT, Budget, BudgetExceeded, Cache, digest, file_hash, read_json, versions, write_json
from .data import image_mapping, load_inputs, plain, mapping_path
from .enrichment import finish, should_use_vlm
from .evidence import Evidence, Fact, Source
from .ocr import OCR
from .retrieval import Catalogue, duplicate_signatures, identity
from .schema import Schema
from .text_extraction import extract, extract_nutrition_text
from .vlm import VLM, model_manifest
from .verification import canonical as canonical_text


def restore(payload):
    evidence = Evidence()
    evidence.sources = {x['id']: Source(**x) for x in payload['sources']}
    evidence.facts = [Fact(**x) for x in payload['facts']]
    evidence.issues = payload['issues']
    return evidence


def reuse_ocr_cache(cache, path, cfg, metadata, previous):
    components = ('src/ocr.py', 'src/image_processing.py')
    packages = ('numpy', 'Pillow', 'rapidocr-onnxruntime', 'onnxruntime', 'opencv-python', 'PyYAML')
    if (previous.get('config', {}).get('ocr') != cfg['ocr'] or
        any(previous.get('source', {}).get(n) != metadata['source'].get(n) for n in components) or
        any(previous.get('packages', {}).get(n) != metadata['packages'].get(n) for n in packages)):
        return False
    image_hash = metadata['images'].get(str(path))
    if not image_hash or previous.get('images', {}).get(str(path)) != image_hash:
        return False
    def key(packages):
        return digest([image_hash, cfg['ocr'], packages,
                       metadata['source']['src/ocr.py'], metadata['source']['src/image_processing.py']])
    current_key = key(metadata['packages'])
    if cache.get('ocr', current_key) is not None:
        return True
    cached = cache.get('ocr', key(previous['packages']))
    if cached is None or cached.get('image_hash') != image_hash:
        return False
    cache.put('ocr', current_key, cached)
    return True


def archive_degraded(records_dir, run_dir, rows):
    for row in rows:
        record_key = digest(row['record_id'])
        previous = records_dir / (record_key + '.json')
        if previous.exists() and read_json(previous).get('degraded', False):
            history = run_dir / 'attempt_history' / record_key
            history.mkdir(parents=True, exist_ok=True)
            previous.replace(history / f'{time.time_ns()}.json')


def initialise(cfg, need_model=False):
    start = time.monotonic()
    data_dir = Path(cfg['data_dir'])
    schema = Schema(data_dir / 'submission_format.json')
    inputs = load_inputs(data_dir)
    mapping, warnings = image_mapping(data_dir, Path(cfg['image_dir']))
    for split in ('sample', 'labels'):
        if list(inputs[split][0]) != schema.columns:
            raise ValueError(f'{split} headers differ from official schema')
    inputs_hash = {p.name: file_hash(p) for p in sorted(data_dir.iterdir()) if p.is_file() and not p.name.startswith('.')}
    mapping_source = mapping_path(data_dir, Path(cfg['image_dir']))
    if mapping_source.is_relative_to(data_dir):
        inputs_hash[str(mapping_source.relative_to(data_dir))] = file_hash(mapping_source)
    source_hash = {str(p.relative_to(ROOT)): file_hash(p) for directory in ('src', 'prompts')
                   for p in sorted((ROOT / directory).rglob('*')) if p.is_file() and p.suffix in ('.py', '.txt')}
    source_hash['config_schema'] = file_hash(ROOT / 'config.yaml')
    for filename in ('run.py', 'evaluate.py', 'validate.py', 'requirements.txt'):
        source_hash[filename] = file_hash(ROOT / filename)
    # Shared by records, and included in every run identity; detect replacement images.
    image_hashes = {str(p): file_hash(p) for p in sorted({p for paths in mapping.values() for p in paths})}
    manifest = model_manifest(cfg['model']['path']) if need_model else {}
    metadata = {'inputs': inputs_hash, 'images': image_hashes, 'source': source_hash,
                'mapping_file': {'path': str(mapping_source), 'hash': file_hash(mapping_source)},
                'packages': versions(), 'python': sys.version, 'platform': platform.platform(),
                'model_files': manifest, 'config': cfg,
                'initialisation_seconds': time.monotonic() - start}
    return schema, inputs, mapping, warnings, metadata


def preflight(cfg, check_model_processor=False):
    schema, inputs, mapping, warnings, meta = initialise(cfg)
    from PIL import Image
    unreadable = 0
    for paths in mapping.values():
        for path in paths:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                unreadable += 1
    output = Path(cfg['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free / 1024**3
    report = {'rows': {k: len(v) for k, v in inputs.items()},
              'columns': {k: list(v[0]) for k, v in inputs.items()},
              'images_per_record': dict(Counter(len(mapping.get(r['record_id'], [])) for r in inputs['test'])),
              'missing_mappings': len(warnings), 'unreadable_images': unreadable,
              'free_disk_gb': free, 'schema_columns': schema.columns,
              'sample_order_differs': [r['article_id'] for r in inputs['sample']] != [r['record_id'] for r in inputs['test']],
              'model_present': (Path(cfg['model']['path']) / 'config.json').exists(),
              'packages': versions(), 'input_hashes': meta['inputs']}
    test_paths = [p for row in inputs['test'] for p in mapping.get(row['record_id'], [])]
    covered = sum(bool(mapping.get(row['record_id'])) for row in inputs['test'])
    report['image_mapping_source'] = str(mapping_path(Path(cfg['data_dir']), Path(cfg['image_dir'])))
    report['test_image_coverage'] = {'products_with_images': covered,
        'products_without_images': len(inputs['test']) - covered,
        'image_entries': len(test_paths), 'unique_image_files': len(set(test_paths))}
    if cfg['policy']['retrieval']:
        catalogue = Catalogue(inputs['reference'], cfg['retrieval'])
        relations, approved = Counter(), 0
        for row in inputs['test']:
            candidates = catalogue.find(row)
            relations.update(c['relation'] for c in candidates)
            evidence = extract(row)
            catalogue.enrich(evidence, candidates)
            approved += any(s.kind == 'family_shared' for s in evidence.sources.values())
        report['retrieval'] = {'candidate_relations': dict(relations), 'records_with_approved_shared_facts': approved}
    write_json(Path(cfg['report_dir']) / 'preflight.json', report)
    if check_model_processor:
        from .vlm import check_processor
        try:
            report['processor_check'] = check_processor(cfg['model']['path'])
        except Exception as error:
            report['processor_check'] = {'status': 'failed', 'error': str(error)}
            write_json(Path(cfg['report_dir']) / 'preflight.json', report)
            raise RuntimeError('Processor integration check failed; see preflight.json') from error
        write_json(Path(cfg['report_dir']) / 'preflight.json', report)
    if free < cfg['budget']['min_disk_gb']:
        raise RuntimeError('Insufficient disk headroom')
    print(json.dumps(report, indent=2))


def representative(rows, mapping, limit, seed):
    rng = random.Random(seed)
    groups = {}
    for row in rows:
        key = (row['category_group'], bool(mapping.get(row['record_id'])),
               bool(row.get('Product Description')))
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    keys = list(groups)
    rng.shuffle(keys)
    selected = []
    while len(selected) < limit and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
    return selected


def pseudo_rows(inputs, mapping, cfg):
    selected = representative(inputs['reference'], mapping, cfg['validation']['pseudo_size'], cfg['seed'])
    patterns = representative(inputs['test'], mapping, len(selected), cfg['seed'] + 1)
    masked, gold, excluded_ids, signatures = [], {}, set(), set()
    for index, original in enumerate(selected):
        row = copy.deepcopy(original)
        pattern = patterns[index % len(patterns)]
        # Entire source channels are removed, not just the parsed target attribute.
        # This deliberately errs towards avoiding leakage in duplicated free text.
        for key in ('Short Description', 'Product Description'):
            if not pattern.get(key):
                row['Short Description'] = row['Product Description'] = ''
        for key in ('Specifications', 'Nutritional Information'):
            if pattern.get(key, '') in ('', '{}') or '[MASKED]' in pattern.get(key, ''):
                row[key] = '[MASKED]' if key == 'Specifications' else '{}'
                # Prevent a withheld declaration from surviving in a parallel text field.
                row['Short Description'] = row['Product Description'] = ''
        row['do_not_predict_fields'] = pattern.get('do_not_predict_fields', '[]')
        gold[row['record_id']] = original
        masked.append(row)
        excluded_ids.add(original['record_id'])
        signatures.update(duplicate_signatures(original))
    # Exclude all sampled targets and duplicates from every validation query.
    return masked, gold, excluded_ids, signatures


def run(cfg, split, limit=0, resume=False, benchmark=False, retry_degraded=False, recovery_dir=None, record_ids=None):
    started = time.monotonic()
    needs_model = cfg['policy']['vlm'] and cfg['policy']['vlm_selection'] != 'off'
    schema, inputs, mapping, warnings, metadata = initialise(cfg, needs_model)
    excluded_ids, signatures, gold = set(), set(), {}
    if split == 'pseudo':
        rows, gold, excluded_ids, signatures = pseudo_rows(inputs, mapping, cfg)
        if not cfg['validation']['pseudo_images']:
            mapping = {}
        else:
            # Remove any reference whose image bytes duplicate a held-out image.
            heldout_hashes = {metadata['images'][str(p)] for r in rows for p in mapping.get(r['record_id'], [])}
            excluded_ids.update(r['record_id'] for r in inputs['reference']
                                if any(metadata['images'][str(p)] in heldout_hashes
                                       for p in mapping.get(r['record_id'], [])))
    else:
        rows = inputs[split]
        if split == 'template':
            excluded_ids.update(r['record_id'] for r in rows)
            for row in rows:
                signatures.update(duplicate_signatures(row))
    all_rows = rows
    if cfg['policy'].get('require_image', False):
        rows = [row for row in rows if mapping.get(row['record_id'])]
    if limit:
        rows = representative(rows, mapping, limit, cfg['seed'])
    if recovery_dir:
        if not retry_degraded:
            raise ValueError('Recovery requires --retry-degraded')
        recovery_path = Path(recovery_dir).resolve()
        if not recovery_path.is_relative_to(Path(cfg['output_dir']).resolve() / 'runs'):
            raise ValueError('Recovery source must be inside experiment runs')
        recovery_manifest = read_json(recovery_path / 'manifest.json')
        if recovery_manifest['split'] != split:
            raise ValueError('Recovery split differs from requested split')
        source_ids = {read_json(p)['record_id'] for folder in ('records', 'status')
                      for p in (recovery_path / folder).glob('*.json') if 'record_id' in read_json(p)}
        if not source_ids:
            source_ids = set(recovery_manifest.get('requested_ids', []))
        if cfg['policy'].get('require_image', False):
            source_ids = {record_id for record_id in source_ids if mapping.get(record_id)}
        rows = [row for row in all_rows if row['record_id'] in source_ids]
        if not rows or {row['record_id'] for row in rows} != source_ids:
            raise ValueError('Recovery records do not match current inputs')
        if limit and len(rows) > limit and not record_ids:
            raise ValueError('Recovery population exceeds --limit; increase it to avoid silently dropping records')
    if record_ids:
        selected = set(record_ids)
        population = {row['record_id']: row for row in all_rows}
        if not selected <= population.keys():
            raise ValueError('Unknown --record-id value')
        if cfg['policy'].get('require_image', False) and any(not mapping.get(record_id) for record_id in selected):
            raise ValueError('The selected profile requires a mapped image for every --record-id')
        rows = [row for row in all_rows if row['record_id'] in selected]
        if limit and len(rows) > limit:
            raise ValueError('Selected record IDs exceed --limit')
    # Hash image bytes and inputs, but exclude initialisation duration from run identity.
    stable = {k: v for k, v in metadata.items() if k != 'initialisation_seconds'}
    run_id = digest([stable, split, sorted(excluded_ids)])[:20]
    run_dir = Path(cfg['output_dir']) / 'runs' / split / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (Path(cfg['output_dir']) / 'run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another experiment process is using the shared cache')
    try:
        return _run_locked(cfg, split, rows, all_rows, inputs, mapping, warnings, metadata,
                           run_dir, schema, gold, excluded_ids, signatures, resume, benchmark, started,
                           retry_degraded, recovery_dir)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _run_locked(cfg, split, rows, all_rows, inputs, mapping, warnings, metadata,
                run_dir, schema, gold, excluded_ids, signatures, resume, benchmark, started,
                retry_degraded=False, recovery_dir=None):
    if (run_dir / 'manifest.json').exists() and not resume:
        raise ValueError('Run already exists; pass --resume to reuse completed work')
    cache = Cache(Path(cfg['output_dir']) / 'cache')
    if shutil.disk_usage(run_dir).free < cfg['budget']['min_disk_gb'] * 1024**3:
        raise RuntimeError('Insufficient disk headroom')
    budget = Budget(cfg['budget'], run_dir / 'budget.json')
    # Include input preparation and model fingerprinting in cumulative accounting.
    budget.base += time.monotonic() - started
    manifest = {'split': split, 'metadata': metadata, 'expected_ids': [r['record_id'] for r in all_rows],
                'requested_ids': [r['record_id'] for r in rows], 'complete': False}
    if retry_degraded:
        manifest['manual_retry_scope'] = str(time.time_ns())
    write_json(run_dir / 'manifest.json', manifest)
    write_json(run_dir / 'input_warnings.json', warnings)
    if gold:
        write_json(run_dir / 'pseudo_gold.json', gold)
        write_json(run_dir / 'pseudo_inputs.json', {r['record_id']: r for r in all_rows})
    catalogue = Catalogue(inputs['reference'], cfg['retrieval']) if cfg['policy']['retrieval'] else None
    ocr = OCR(cfg['ocr'], cache, budget, cfg['policy']['max_ocr_images'])
    cfg_with_schema = {**cfg, '_schema': schema}
    write_json(run_dir / 'config.json', cfg)
    records_dir = run_dir / 'records'
    prep_dir = run_dir / 'prepared'
    records_dir.mkdir(exist_ok=True)
    prep_dir.mkdir(exist_ok=True)
    recovery_metadata = None
    if recovery_dir:
        recovery = Path(recovery_dir).resolve()
        if not recovery.is_relative_to(Path(cfg['output_dir']).resolve() / 'runs'):
            raise ValueError('Recovery source must be an existing experiment run')
        old_manifest = read_json(recovery / 'manifest.json')
        if old_manifest['split'] != split:
            raise ValueError('Recovery split differs from requested split')
        recovery_metadata = old_manifest['metadata']
        # Data/model changes still receive a new run identity. Never overwrite the source run.
        if recovery != run_dir:
            component_names = ('src/data.py', 'src/text_extraction.py', 'src/retrieval.py',
                               'src/ocr.py', 'src/image_processing.py')
            compatible = (recovery_metadata['inputs'] == metadata['inputs'] and
                recovery_metadata['config']['ocr'] == cfg['ocr'] and
                recovery_metadata['config']['retrieval'] == cfg['retrieval'] and
                recovery_metadata['config']['policy']['ocr'] == cfg['policy']['ocr'] and
                recovery_metadata['config']['policy']['retrieval'] == cfg['policy']['retrieval'] and
                all(recovery_metadata['source'].get(n) == metadata['source'].get(n) for n in component_names))
            if compatible:
                for row in rows:
                    record_key = digest(row['record_id'])
                    previous = recovery / 'prepared' / (record_key + '.json')
                    target = prep_dir / previous.name
                    if previous.is_file() and not target.exists():
                        old_prep = read_json(previous)
                        if all(metadata['images'].get(p) == recovery_metadata['images'].get(p)
                               for p in old_prep.get('paths', [])):
                            shutil.copy2(previous, target)
                    previous_record = recovery / 'records' / (record_key + '.json')
                    target_record = records_dir / previous_record.name
                    if previous_record.is_file() and not target_record.exists():
                        result = read_json(previous_record)
                        if not result.get('degraded', False):
                            from .submission import validate_traces
                            if not validate_traces(result, row, schema, cfg['verification']):
                                result['reused_from'] = str(previous_record)
                                write_json(target_record, result)
            manifest['recovery_source'] = str(recovery)
            write_json(run_dir / 'manifest.json', manifest)
    if retry_degraded:
        archive_degraded(records_dir, run_dir, rows)
    def key(row):
        return digest(row['record_id'])
    def status(row, value):
        write_json(run_dir / 'status' / (key(row) + '.json'), value)
    def log(event):
        with (run_dir / 'events.jsonl').open('a') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
    try:
        budget.check()
        # OCR is loaded only for the preparation pass, then released before MLX.
        for row in rows:
            record_key = key(row)
            if (records_dir / (record_key + '.json')).exists() or (prep_dir / (record_key + '.json')).exists():
                continue
            budget.check()
            status(row, {'state': 'preparing', 'record_id': row['record_id']})
            t0 = time.monotonic()
            evidence = extract(row)
            candidates = catalogue.find(row, excluded_ids, signatures) if catalogue else []
            if catalogue:
                catalogue.enrich(evidence, candidates)
            paths = mapping.get(row['record_id'], [])
            readable = []
            for path in paths:
                try:
                    from PIL import Image
                    with Image.open(path) as image:
                        image.verify()
                    readable.append(path)
                    if cfg['policy']['ocr']:
                        if recovery_metadata:
                            reuse_ocr_cache(cache, path, cfg, metadata, recovery_metadata)
                        result = None
                        for attempt in range(1 + cfg['budget']['retry_count']):
                            try:
                                result = ocr.run(path)
                                break
                            except BudgetExceeded:
                                raise
                            except Exception:
                                if attempt == cfg['budget']['retry_count']:
                                    raise
                        if result.get('skipped'):
                            evidence.issues.append({'kind': 'ocr_stage_limit', 'image': str(path)})
                        ambiguous = sum(line.get('ambiguous', False) for line in result['raw'])
                        if ambiguous:
                            evidence.issues.append({'kind': 'ambiguous_ocr', 'image': str(path), 'lines': ambiguous})
                        for line in result['lines']:
                            evidence.source('ocr', str(path), line['text'], line['confidence'])
                        for block_index, block in enumerate(result.get('blocks', [])):
                            source = evidence.source('ocr', str(path) + f'/block/{block_index}',
                                block['text'], block['confidence'], metadata={'lines': block['lines']})
                            extract_nutrition_text(evidence, source)
                            # Parse only literal labelled values; keep facts bound to the original block.
                            parsed = extract({'Specifications': block['text'], 'Product Description': block['text']}, kind='ocr')
                            for fact in parsed.facts:
                                original = parsed.sources[fact.source_id]
                                if canonical_text(original.text) in canonical_text(block['text']):
                                    evidence.add(fact.field, fact.path, fact.value, source,
                                                 original.text, confidence=1., method='ocr_labelled')
                except BudgetExceeded:
                    raise
                except Exception as error:
                    # Readable images with failed OCR remain usable by the vision stage.
                    evidence.issues.append({'kind': 'image_or_ocr_error', 'image': str(path), 'error': str(error)})
            prepared = {'evidence': evidence.payload(), 'paths': [str(p) for p in readable],
                        'retrieval': candidates, 'prepare_seconds': time.monotonic() - t0}
            write_json(prep_dir / (record_key + '.json'), prepared)
            status(row, {'state': 'prepared', 'record_id': row['record_id']})
        ocr.close()
        vlm = VLM(cfg['model'], cache, budget, metadata['model_files'], cfg['seed'],
                  diagnostics_dir=run_dir / 'diagnostics') if cfg['policy']['vlm'] else None
        if vlm is not None:
            vlm.attempt_scope = manifest.get('manual_retry_scope')
        used_model = sum(read_json(p).get('model_attempted', read_json(p).get('used_model', False)) for p in records_dir.glob('*.json'))
        # Sparse records with images receive the first opportunities under a stage cap.
        priority = sorted(rows, key=lambda r: (bool(r.get('Product Description')), not bool(mapping.get(r['record_id'])), r['record_id']))
        for row in priority:
            record_key = key(row)
            destination = records_dir / (record_key + '.json')
            if destination.exists():
                continue
            budget.check()
            prep = read_json(prep_dir / (record_key + '.json'))
            evidence = restore(prep['evidence'])
            paths = [Path(p) for p in prep['paths']][:cfg['model']['max_images']]
            use_model = (used_model < cfg['policy']['max_vlm_records'] and
                         should_use_vlm(row, evidence, paths, cfg))
            t0 = time.monotonic()
            status(row, {'state': 'inferring', 'record_id': row['record_id']})
            try:
                if vlm is not None:
                    vlm.record_id = row['record_id']
                    vlm.record_key = digest([str(run_dir), row['record_id']])
                    vlm.record_call_limit = cfg['policy']['max_calls_per_record']
                result = finish(row, evidence, paths, prep['retrieval'], vlm, cfg_with_schema, use_model)
                result.update({'record_id': row['record_id'],
                               'seconds': time.monotonic() - t0 + prep['prepare_seconds']})
                budget.check()
                write_json(destination, result)
                used_model += bool(use_model)
                state = 'degraded' if result['degraded'] else 'complete'
                status(row, {'state': state, 'record_id': row['record_id'],
                             'model_attempted': result['model_attempted'], 'model_loaded': result['model_loaded'],
                             'model_succeeded': result['model_succeeded'], 'model_failed': result['model_failed']})
                log({'event': 'record_' + state, 'record_id': row['record_id'], 'seconds': result['seconds'],
                     'used_model': result['used_model'], 'model_loaded': result['model_loaded'],
                     'rejected': len(result['rejected'])})
            except BudgetExceeded:
                raise
            except Exception as error:
                status(row, {'state': 'failed', 'record_id': row['record_id'], 'error': str(error)})
                log({'event': 'record_failed', 'record_id': row['record_id'], 'error': str(error)})
            finally:
                budget.save()
        if vlm is not None and vlm.model is not None:
            vlm.close()
        completed = {read_json(p)['record_id'] for p in records_dir.glob('*.json')}
        manifest['processing_complete'] = completed == set(manifest['expected_ids'])
        manifest['completed_count'] = len(completed)
        manifest['degraded_count'] = sum(bool(read_json(p).get('degraded')) for p in records_dir.glob('*.json'))
        manifest['complete'] = manifest['processing_complete'] and manifest['degraded_count'] == 0
        write_json(run_dir / 'manifest.json', manifest)
        write_json(Path(cfg['report_dir']) / f'latest_{split}.json', {'run_dir': str(run_dir)})
        if benchmark:
            elapsed = time.monotonic() - started
            preparation = sum(read_json(p)['prepare_seconds'] for p in prep_dir.glob('*.json'))
            record_seconds = sum(read_json(p)['seconds'] for p in records_dir.glob('*.json'))
            report = {'sample_size': len(rows), 'completed': len(completed),
                      'wall_seconds_this_invocation': elapsed, 'budget': budget.state,
                      'extrapolated_seconds': elapsed / max(len(rows), 1) * len(all_rows),
                      'warning': 'Extrapolation includes cache effects and is not a compliance claim.',
                      'model_records': used_model, 'total_population': len(all_rows),
                      'preparation_seconds_including_cached_records': preparation,
                      'enrichment_seconds_including_cached_records': max(0, record_seconds - preparation),
                      'initialisation_seconds_this_invocation': metadata['initialisation_seconds']}
            write_json(run_dir / 'benchmark.json', report)
        print(json.dumps({'run_dir': str(run_dir), 'completed': len(completed),
                          'expected': len(all_rows), 'complete': manifest['complete']}))
        return run_dir
    finally:
        ocr.close()
        if 'vlm' in locals() and vlm is not None and vlm.model is not None:
            vlm.close()
        budget.save()
