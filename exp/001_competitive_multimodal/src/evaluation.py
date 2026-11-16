from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .common import read_json, write_json
from .data import blocked, load_inputs, plain
from .pipeline import restore
from .schema import Schema, TEXT, leaves
from .text_extraction import extract
from .verification import canonical, verify


def atoms(value, path=()):
    result = []
    if isinstance(value, dict):
        if set(value) == {'attribute', 'value'}:
            return [(path, canonical(value['attribute']), canonical(value['value']))]
        for key, child in value.items():
            result.extend(atoms(child, path + (key,)))
    elif isinstance(value, list):
        for child in value:
            result.extend(atoms(child, path))
    elif value != '' and value is not None:
        result.append((path, canonical(value)))
    return result


def counts(prediction, reference):
    p, r = Counter(atoms(prediction)), Counter(atoms(reference))
    tp = sum((p & r).values())
    return tp, sum(p.values()) - tp, sum(r.values()) - tp


def metrics(counter):
    tp, fp, fn = counter['tp'], counter['fp'], counter['fn']
    return {**counter, 'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def evaluate_run(run_dir, cfg):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / 'manifest.json')
    schema = Schema(Path(cfg['data_dir']) / 'submission_format.json')
    inputs = load_inputs(Path(cfg['data_dir']))
    saved = read_json(run_dir / 'config.json')
    records = [read_json(p) for p in (run_dir / 'records').glob('*.json')]
    split = manifest['split']
    if split == 'template':
        targets = {r['article_id']: schema.decode(r) for r in inputs['labels']}
        source_rows = {r['record_id']: r for r in inputs['template']}
    elif split == 'pseudo':
        gold = read_json(run_dir / 'pseudo_gold.json')
        source_rows = read_json(run_dir / 'pseudo_inputs.json')
        targets = {}
        for key, row in gold.items():
            original_evidence = extract(row)
            values, _, _ = verify(original_evidence, schema, saved['verification'], blocked(source_rows[key]))
            values['short_description'] = plain(row.get('Short Description', ''))
            values['long_description'] = plain(row.get('Product Description', ''))
            targets[key] = values
    else:
        targets = {}
        source_rows = {r['record_id']: r for r in inputs['test']}
    overall, categories = defaultdict(Counter), defaultdict(lambda: defaultdict(Counter))
    coverage, errors, rejected, issues, qualitative, recoverable = Counter(), Counter(), Counter(), Counter(), [], defaultdict(Counter)
    invalid_rows = 0
    from .submission import validate_traces
    for record in records:
        record_id = record['record_id']
        row = source_rows[record_id]
        category = row['category_group']
        values = record['values']
        validation_errors = validate_traces(record, row, schema, saved['verification'])
        invalid_rows += bool(validation_errors)
        errors.update(validation_errors)
        rejected.update(x['reason'] for x in record['rejected'])
        issues.update(x['kind'] for x in record['evidence']['issues'])
        target = targets.get(record_id)
        observed = ' '.join(s['text'] for s in record['evidence']['sources']
                            if s['eligible'] and s['kind'] in ('record', 'ocr'))
        for field in schema.empty:
            coverage[field] += bool(list(leaves(values[field])))
            if target is None:
                continue
            # Official judge excludes product/field pairs with no reference information.
            reference_atoms = atoms(target[field])
            if not reference_atoms or field in TEXT or field in blocked(row):
                continue
            tp, fp, fn = counts(values[field], target[field])
            for counter in (overall[field], categories[category][field]):
                counter.update({'tp': tp, 'fp': fp, 'fn': fn, 'evaluated_products': 1})
            for path, scalar in leaves(target[field]):
                status = 'text_observable' if canonical(scalar) in canonical(observed) else 'not_text_observable'
                recoverable[field][status] += 1
        if target is not None:
            # Lexical diagnostics are explicitly not the hidden semantic score.
            diagnostics = {}
            for field in TEXT:
                predicted_words = set(canonical(values[field]).split())
                reference_words = set(canonical(target[field]).split())
                diagnostics[field] = {'prediction': values[field], 'reference': target[field],
                                      'token_jaccard': len(predicted_words & reference_words) / max(1, len(predicted_words | reference_words)),
                                      'exact_normalised': canonical(values[field]) == canonical(target[field])}
            qualitative.append({'record_id': record_id, 'category': category, 'fields': diagnostics})
    report = {'split': split, 'completed': len(records), 'expected': len(manifest['expected_ids']),
              'degraded_records': sum(bool(r.get('degraded')) for r in records),
              'model_status': {key: sum(r.get(key) is True for r in records) for key in
                  ('model_attempted', 'model_loaded', 'model_succeeded', 'model_failed')},
              'source_fallback_sentences': sum(s.get('mode') == 'source' for r in records
                  for sentences in r.get('description_trace', {}).values() for s in sentences),
              'stage_status': dict(Counter(f'{stage}:{status}' for r in records
                  for stage, status in r.get('stage_status', {}).items())),
              'invalid_output_rate': invalid_rows / max(len(records), 1),
              'coverage': {f: coverage[f] / max(len(records), 1) for f in schema.empty},
              'structured_lexical_metrics': {f: metrics(c) for f, c in overall.items()},
              'by_category': {c: {f: metrics(x) for f, x in fields.items()} for c, fields in categories.items()},
              'recoverability_diagnostics': dict(recoverable), 'schema_and_trace_errors': dict(errors),
              'rejected_candidate_reasons': dict(rejected), 'evidence_issues': dict(issues),
              'runtime': read_json(run_dir / 'budget.json'),
              'limitations': ['No local semantic judge score or overall leaderboard estimate.',
                             'Structured comparison uses normalised lexical values and counts all unmatched predictions as FP.',
                             'Pseudo targets are deterministic silver extractions, not human labels.',
                             'Not text observable does not prove irrecoverable; images may contain the answer.',
                             'Schema validation and lexical trace checks do not prove semantic entailment.']}
    write_json(run_dir / 'evaluation.json', report)
    write_json(run_dir / 'qualitative_review.json', qualitative)
    print(json.dumps(report, indent=2))
    return report


def compare_runs(run_dirs, destination):
    reports = []
    for directory in run_dirs:
        directory = Path(directory)
        report = read_json(directory / 'evaluation.json')
        cfg = read_json(directory / 'config.json')
        manifest = read_json(directory / 'manifest.json')
        reports.append({'run_dir': str(directory), 'policy': cfg['policy'],
                        'ids': sorted(read_json(p)['record_id'] for p in (directory / 'records').glob('*.json')),
                        'expected_ids': sorted(manifest['expected_ids']), 'seed': cfg['seed'],
                        'pseudo_inputs': read_json(directory / 'pseudo_inputs.json') if (directory / 'pseudo_inputs.json').exists() else None,
                        'report': report})
    if len({json.dumps(r['ids']) for r in reports}) != 1:
        raise ValueError('Ablation runs do not share the same validation population')
    if len({json.dumps([r['expected_ids'], r['seed'], r['pseudo_inputs']], sort_keys=True) for r in reports}) != 1:
        raise ValueError('Ablation runs differ in seed, target population or masked inputs')
    write_json(destination, reports)
