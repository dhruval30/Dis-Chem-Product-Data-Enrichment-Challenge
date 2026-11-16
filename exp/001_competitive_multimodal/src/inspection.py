from __future__ import annotations

import json
import time
from pathlib import Path

from .common import digest, read_json, write_json
from .data import csv_rows


def export_inspection(run_dir, record_id, cfg):
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_relative_to(Path(cfg['output_dir']).resolve()):
        raise ValueError('Inspection run must be inside experiment outputs')
    key = digest(record_id)
    destination = run_dir / 'inspections' / f'{time.time_ns()}_{key[:12]}'
    destination.mkdir(parents=True, exist_ok=False)
    manifest = read_json(run_dir / 'manifest.json')
    prepared_path = run_dir / 'prepared' / (key + '.json')
    record_path = run_dir / 'records' / (key + '.json')
    status_path = run_dir / 'status' / (key + '.json')
    prepared = read_json(prepared_path) if prepared_path.exists() else {}
    result = read_json(record_path) if record_path.exists() else {}
    evidence = result.get('evidence', prepared.get('evidence', {}))
    filename = {'test': 'Test_Masked.csv', 'template': 'template_input.csv'}.get(manifest['split'])
    if filename:
        rows = csv_rows(Path(cfg['data_dir']) / filename)
        write_json(destination / 'input.json', next(row for row in rows if row['record_id'] == record_id))
    elif (run_dir / 'pseudo_inputs.json').exists():
        write_json(destination / 'input.json', read_json(run_dir / 'pseudo_inputs.json')[record_id])
    paths = prepared.get('paths', [])
    write_json(destination / 'images.json', {'original_paths': paths,
        'vlm_selected_paths': paths[:cfg['model']['max_images']], 'images_are_references_not_copies': True})
    write_json(destination / 'ocr_text.json', [s for s in evidence.get('sources', []) if s['kind'] == 'ocr'])
    write_json(destination / 'visual_evidence.json', [s for s in evidence.get('sources', [])
        if s['kind'] in ('visual_printed', 'visual_observation', 'visual')])
    write_json(destination / 'accepted.json', result.get('accepted_facts', []))
    write_json(destination / 'rejected.json', {'facts_and_sentences': result.get('rejected', []),
        'issues': evidence.get('issues', []),
        'ineligible_sources': [s for s in evidence.get('sources', []) if not s.get('eligible')]})
    write_json(destination / 'enriched_fields.json', result.get('values', {}))
    write_json(destination / 'description_trace.json', result.get('description_trace', {}))
    write_json(destination / 'product_content.json', [s for s in evidence.get('sources', []) if s['kind'] == 'grounded_text'])
    write_json(destination / 'visual_replay.json', result.get('visual_replay', {}))
    if status_path.exists():
        (destination / 'status.json').write_bytes(status_path.read_bytes())
    diagnostics_run = Path(manifest.get('re_enrichment_visual_source', manifest.get('re_enrichment_source', str(run_dir)))).resolve()
    diagnostics = run_dir / 'diagnostics/vlm.jsonl'
    if not diagnostics.exists() and diagnostics_run.is_relative_to(Path(cfg['output_dir']).resolve()):
        diagnostics = diagnostics_run / 'diagnostics/vlm.jsonl'
    events = []
    if diagnostics.exists():
        for line in diagnostics.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('record_id') == record_id:
                events.append(event)
    write_json(destination / 'diagnostics.json', events)
    for event in events:
        if event.get('event') == 'prompt_saved' and event.get('prompt_path'):
            path = Path(event['prompt_path']).resolve()
            if (path.is_relative_to(run_dir / 'diagnostics/prompts') or
                path.is_relative_to(diagnostics_run / 'diagnostics/prompts')) and path.is_file():
                target = destination / 'assembled_prompts' / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
    cache = Path(cfg['output_dir']) / 'cache'
    parsed = []
    for call_key in sorted({event['call_key'] for event in events if event.get('call_key')}):
        for stage in ('vlm_raw', 'vlm', 'vlm_errors'):
            path = cache / stage / (call_key + '.json')
            if path.exists():
                target = destination / 'model_responses' / stage / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
                if stage == 'vlm':
                    value = read_json(path).get('parsed', {})
                    if value.get('schema_version') == 'visual-evidence-v2':
                        parsed.append(value)
    write_json(destination / 'parsed_visual_responses.json', parsed)
    hashes = {manifest['metadata']['images'].get(path) for path in paths} - {None}
    for path in (cache / 'ocr').glob('*.json'):
        if read_json(path).get('image_hash') in hashes:
            target = destination / 'ocr_raw' / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    (destination / 'README.md').write_text(
        '# Visual inspection export\n\n'
        'images.json references the original files without opening or copying them. '
        'ocr_text.json and ocr_raw/ preserve existing OCR evidence. '
        'model_responses/ contains existing raw, parsed and error cache files; no response is synthesised. '
        'parsed_visual_responses.json contains v2 image outputs. visual_evidence.json records identity, '
        'reliability and provenance. accepted.json, rejected.json, description_trace.json and '
        'enriched_fields.json show downstream use and rejection reasons. '
        'A missing model response may mean loading failed before generation; inspect diagnostics.json and status.json.\n')
    print(json.dumps({'inspection_dir': str(destination), 'has_record_result': bool(result),
                      'parsed_visual_calls': len(parsed)}, indent=2))
    return destination
