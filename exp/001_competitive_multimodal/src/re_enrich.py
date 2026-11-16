from __future__ import annotations

import copy
import json
from pathlib import Path

from .common import ROOT, digest, file_hash, read_json, write_json
from .data import csv_rows
from .enrichment import finish, add_visual
from .content import route_evidence
from .pipeline import restore
from .schema import Schema
from .text_extraction import extract
from .vlm import parse_json


def saved_visuals(source, manifest, output, selected):
    origin = Path(manifest.get('re_enrichment_visual_source', manifest.get('re_enrichment_source', str(source)))).resolve()
    if not origin.is_relative_to(output):
        raise ValueError('Saved visual diagnostics escape experiment outputs')
    diagnostics = origin / 'diagnostics/vlm.jsonl'
    responses = {}
    if diagnostics.exists():
        for line in diagnostics.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if (event.get('record_id') in selected and event.get('stage') == 'image_extraction' and
                    event.get('event') in ('parsing_start', 'parsing_complete') and event.get('call_key')):
                key = event['call_key']
                if not isinstance(key, str) or len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
                    raise ValueError('Invalid saved model call key')
                path = output / 'cache/vlm_raw' / (key + '.json')
                if path.is_file():
                    responses[event['record_id']] = path
    return origin, responses


def re_enrich(source_dir, cfg, record_ids=()):
    """Rebuild fields from saved evidence without loading OCR, images or any model."""
    source = Path(source_dir).resolve()
    output = Path(cfg['output_dir']).resolve()
    if not source.is_relative_to(output):
        raise ValueError('Re-enrichment source must be inside experiment outputs')
    original = read_json(source / 'manifest.json')
    mapping = original['metadata'].get('mapping_file')
    if mapping and (not Path(mapping['path']).is_file() or file_hash(Path(mapping['path'])) != mapping['hash']):
        raise ValueError('Image mapping changed since saved evidence was collected')
    schema = Schema(Path(cfg['data_dir']) / 'submission_format.json')
    data_dir = Path(cfg['data_dir'])
    for name, expected in original['metadata']['inputs'].items():
        if name.startswith('.'):
            continue
        if not (data_dir / name).is_file() or file_hash(data_dir / name) != expected:
            raise ValueError(f'Input changed since saved evidence was collected: {name}')
    split = original['split']
    if split == 'pseudo':
        rows = read_json(source / 'pseudo_inputs.json')
    else:
        filename = 'Test_Masked.csv' if split == 'test' else 'template_input.csv'
        rows = {r['record_id']: r for r in csv_rows(data_dir / filename, 'record_id')}
    saved = {read_json(p)['record_id']: p for p in (source / 'records').glob('*.json')}
    selected = set(record_ids) if record_ids else set(saved)
    if not selected or not selected <= saved.keys() or not selected <= rows.keys():
        raise ValueError('Requested re-enrichment records are unavailable')
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for folder in ('src', 'prompts')
              for p in sorted((ROOT / folder).rglob('*')) if p.suffix in ('.py', '.txt')}
    hashes['config_schema'] = file_hash(ROOT / 'config.yaml')
    for name in ('run.py', 'evaluate.py', 'validate.py', 'requirements.txt'):
        hashes[name] = file_hash(ROOT / name)
    snapshot = {key: file_hash(saved[key]) for key in sorted(selected)}
    origin, visual_raw = saved_visuals(source, original, output, selected)
    dependencies = {str(path): file_hash(path) for path in visual_raw.values()}
    for record_id in selected:
        path = source / 'prepared' / (digest(record_id) + '.json')
        if path.is_file():
            dependencies[str(path)] = file_hash(path)
    fingerprint = digest([str(source), snapshot, dependencies, cfg, hashes])[:20]
    destination = output / 're_enriched' / fingerprint
    destination.mkdir(parents=True, exist_ok=True)
    config_with_schema = {**cfg, '_schema': schema}
    degraded = 0
    for record_id in sorted(selected):
        old = read_json(saved[record_id])
        prepared = source / 'prepared' / (digest(record_id) + '.json')
        prepared_data = read_json(prepared) if prepared.is_file() else {}
        replay = {'status': 'unavailable', 'parser_events': []}
        can_replay = record_id in visual_raw and bool(prepared_data.get('paths')) and 'evidence' in prepared_data
        evidence = restore(prepared_data['evidence'] if can_replay else old['evidence'])
        # Rebuild derived text with current routing rules, retaining untouched raw OCR/VLM sources.
        removed = {s.id for s in evidence.sources.values() if s.kind in ('grounded_text', 'record')}
        evidence.sources = {k: v for k, v in evidence.sources.items() if k not in removed}
        evidence.facts = [f for f in evidence.facts if f.source_id not in removed]
        evidence.absorb(extract(rows[record_id]))
        if can_replay:
            replay['raw_response'] = str(visual_raw[record_id])
            try:
                response = parse_json(read_json(visual_raw[record_id])['text'], recover_observations=True,
                                      parser_events=replay['parser_events'])
                if response.get('schema_version') != 'visual-evidence-v2':
                    raise ValueError('Saved visual response is not visual-evidence-v2')
                route_evidence(rows[record_id], evidence, cfg)
                source_config = original['metadata'].get('config', cfg)
                count = source_config.get('model', cfg['model'])['max_images']
                paths = [Path(p) for p in prepared_data['paths'][:count]]
                add_visual(response, paths, evidence, cfg['verification'], rows[record_id])
                replay.update({'status': 'replayed', 'parsed': response})
            except (ValueError, TypeError, KeyError) as error:
                replay.update({'status': 'failed', 'error': str(error)})
                evidence.issues.append({'kind': 'saved_visual_replay_failed', 'error': str(error)})
        result = finish(rows[record_id], evidence, [], old.get('retrieval', []), None, config_with_schema, False)
        failures = {stage for stage, state in old.get('stage_status', {}).items() if state == 'failed'}
        resolved_image_failure = replay['status'] == 'replayed' and failures == {'image_extraction'}
        result['degraded'] = bool(old.get('degraded')) and not resolved_image_failure or replay['status'] == 'failed'
        result['recovered_stages'] = ['image_extraction'] if resolved_image_failure else []
        result['extraction_stage_status'] = old.get('stage_status', {})
        degraded += result['degraded']
        result.update({'record_id': record_id, 'seconds': 0., 'extraction_source': str(saved[record_id]),
                       're_enriched_without_inference': True, 'visual_replay': replay})
        write_json(destination / 'records' / (digest(record_id) + '.json'), result)
        if prepared.is_file():
            target = destination / 'prepared' / prepared.name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(prepared.read_bytes())
        write_json(destination / 'status' / (digest(record_id) + '.json'),
                   {'state': 'degraded' if result['degraded'] else 'complete', 'record_id': record_id})
    manifest = copy.deepcopy(original)
    manifest['metadata']['source'] = hashes
    manifest['metadata']['config'] = cfg
    manifest['metadata']['inputs'] = {k: v for k, v in original['metadata']['inputs'].items() if not k.startswith('.')}
    manifest.update({'requested_ids': sorted(selected), 'completed_count': len(selected),
                     'processing_complete': selected == set(original['expected_ids']),
                     'complete': selected == set(original['expected_ids']) and degraded == 0, 'degraded_count': degraded,
                     're_enrichment_source': str(source), 're_enrichment_input_hashes': snapshot,
                     're_enrichment_visual_source': str(origin), 'replay_dependencies': dependencies})
    write_json(destination / 'manifest.json', manifest)
    write_json(destination / 'config.json', cfg)
    write_json(destination / 'budget.json', {'generation_calls': 0, 'ocr_images': 0,
        'note': 'Postprocessing only. Original extraction runtime and cost are in the source run.',
        'original_extraction_budget': read_json(source / 'budget.json')})
    if split == 'pseudo':
        for name in ('pseudo_inputs.json', 'pseudo_gold.json'):
            (destination / name).write_bytes((source / name).read_bytes())
    write_json(Path(cfg['report_dir']) / f'latest_{split}.json', {'run_dir': str(destination)})
    print(destination)
    return destination
