from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

from .common import file_hash, read_json, write_json
from .data import blocked, csv_rows, load_inputs
from .pipeline import restore
from .schema import Schema, TEXT
from .verification import verify, verify_sentence


def validate_file(path, cfg, input_rows=None, require_order=True):
    schema = Schema(Path(cfg['data_dir']) / 'submission_format.json')
    inputs = load_inputs(Path(cfg['data_dir']))
    expected = input_rows if input_rows is not None else inputs['test']
    by_id = {r['record_id']: r for r in expected}
    order = ([r['record_id'] for r in expected] if input_rows is not None else
             [r['article_id'] for r in inputs['sample']])
    errors, ids = [], []
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != schema.columns:
            errors.append({'kind': 'wrong_columns', 'found': reader.fieldnames})
        for number, row in enumerate(reader, 2):
            article_id = row.get('article_id', '')
            ids.append(article_id)
            if None in row or any(v is None for v in row.values()):
                errors.append({'line': number, 'kind': 'wrong_row_width'})
                continue
            if article_id not in by_id:
                errors.append({'line': number, 'kind': 'unexpected_id'})
                continue
            try:
                values = schema.decode(row)
                for error in schema.errors(values, blocked(by_id[article_id])):
                    errors.append({'line': number, 'kind': 'schema', 'error': error})
                for field in blocked(by_id[article_id]) & schema.empty.keys():
                    if row[field] != '':
                        errors.append({'line': number, 'kind': 'nonblank_blocked_field', 'field': field})
            except (ValueError, KeyError, TypeError) as error:
                errors.append({'line': number, 'kind': 'decode_error', 'error': str(error)})
    if len(ids) != len(set(ids)):
        errors.append({'kind': 'duplicate_ids'})
    if set(ids) != set(by_id):
        errors.append({'kind': 'id_set_mismatch', 'missing': len(set(by_id) - set(ids)),
                       'extra': len(set(ids) - set(by_id))})
    if require_order and ids != order:
        errors.append({'kind': 'wrong_order'})
    return errors


def validate_traces(result, row, schema, verification):
    evidence = restore(result['evidence'])
    verified, _, _ = verify(evidence, schema, verification, blocked(row))
    errors = schema.errors(result['values'], blocked(row))
    for field in schema.empty:
        if field in TEXT:
            continue
        if verified[field] != result['values'][field]:
            errors.append(f'{field}: saved value differs from verified evidence')
    if result['values']['online_name'] != verified['online_name']:
        errors.append('online_name differs from verified identity')
    for field in ('short_description', 'long_description'):
        trace = result.get('description_trace', {}).get(field, [])
        if ' '.join(item['text'] for item in trace) != result['values'][field]:
            errors.append(f'{field}: missing sentence trace')
        for sentence in trace:
            reason = verify_sentence(sentence['text'], sentence['source_ids'], evidence, verification)
            if reason:
                errors.append(f'{field}: {reason}')
    long_text = result['values']['long_description']
    if long_text and not long_text.casefold().startswith(row['Product Name'].casefold()):
        errors.append('long_description: missing full product-name opening')
    if re.search(r'[•\n]', result['values']['short_description']):
        errors.append('short_description: bullets or line breaks')
    return errors


def write_submission(run_dir, cfg):
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_relative_to(Path(cfg['output_dir'])):
        raise ValueError('Run directory must be inside experiment outputs')
    manifest = read_json(run_dir / 'manifest.json')
    if manifest['split'] != 'test' or manifest.get('complete') is not True:
        raise ValueError('Refusing submission: full test run is incomplete')
    inputs = load_inputs(Path(cfg['data_dir']))
    schema = Schema(Path(cfg['data_dir']) / 'submission_format.json')
    for name, expected_hash in manifest['metadata']['inputs'].items():
        if file_hash(Path(cfg['data_dir']) / name) != expected_hash:
            raise ValueError(f'Input changed since inference: {name}')
    mapping = manifest['metadata'].get('mapping_file')
    if mapping and (not Path(mapping['path']).is_file() or file_hash(Path(mapping['path'])) != mapping['hash']):
        raise ValueError('Image mapping changed since inference')
    saved_cfg = read_json(run_dir / 'config.json')
    for name, expected_hash in manifest['metadata']['source'].items():
        actual = (Path(__file__).resolve().parents[1] / ('config.yaml' if name == 'config_schema' else name))
        if not actual.is_file() or file_hash(actual) != expected_hash:
            raise ValueError(f'Implementation changed since inference: {name}')
    records = [read_json(p) for p in (run_dir / 'records').glob('*.json')]
    records_by_id = {r['record_id']: r for r in records}
    test_by_id = {r['record_id']: r for r in inputs['test']}
    if len(records) != len(records_by_id) or set(records_by_id) != set(test_by_id):
        raise ValueError('Missing, duplicate or extra processed records')
    image_hashes = manifest['metadata']['images']
    used_images = set()
    for result in records:
        for source in result['evidence']['sources']:
            if source['kind'] in ('ocr', 'visual', 'visual_printed', 'visual_observation', 'grounded_text'):
                location = source.get('metadata', {}).get('image_path') or source['location'].split('/block/', 1)[0]
                if location in image_hashes:
                    used_images.add(location)
    for image in sorted(used_images):
        if not Path(image).is_file() or file_hash(Path(image)) != image_hashes[image]:
            raise ValueError('A supplied image changed since inference')
    errors = []
    for record_id, result in records_by_id.items():
        for error in validate_traces(result, test_by_id[record_id], schema, saved_cfg['verification']):
            errors.append({'record_id': record_id, 'error': error})
    write_json(run_dir / 'release_validation.json', {'errors': errors,
        'degraded_records': sum(bool(r.get('degraded')) for r in records),
        'visual_sources': sum(s['kind'] == 'visual' for r in records for s in r['evidence']['sources'])})
    if errors:
        raise ValueError('Evidence validation failed; inspect release_validation.json')
    destination = run_dir / 'submission.csv'
    temp = run_dir / 'submission.csv.pending'
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=schema.columns)
        writer.writeheader()
        for sample in inputs['sample']:
            record_id = sample['article_id']
            values = records_by_id[record_id]['values']
            row = {'article_id': record_id}
            for field in schema.empty:
                value = values[field]
                row[field] = '' if field in blocked(test_by_id[record_id]) or value == schema.empty[field] else (
                    value if field in TEXT else json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
            writer.writerow(row)
    errors = validate_file(temp, cfg)
    if errors:
        write_json(run_dir / 'release_validation.json', {'errors': errors})
        raise ValueError('Submission validation failed; pending file has not been released')
    os.replace(temp, destination)
    print(destination)
    return destination
