from __future__ import annotations

import argparse
from pathlib import Path

from src.common import ROOT, config, read_json


def parser():
    p = argparse.ArgumentParser(description='Experiment 001: evidence-grounded multimodal enrichment')
    p.add_argument('command', choices=['preflight', 'infer', 'inspect', 're-enrich', 'benchmark', 'evaluate', 'submission', 'validate', 'compare'])
    p.add_argument('--config', default=str(ROOT / 'config.yaml'))
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    p.add_argument('--seed', type=int)
    p.add_argument('--split', choices=['test', 'template', 'pseudo'], default='test')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--record-id', action='append', default=[], help='Select an exact input record ID; may be repeated')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-degraded', action='store_true', help='Retry degraded records; optionally reuse --run-dir as the source run')
    p.add_argument('--check-processor', action='store_true', help='Preflight only: initialise the local processor without weights or generation')
    p.add_argument('--run-dir')
    p.add_argument('--file')
    p.add_argument('--runs', nargs='+')
    return p


def main():
    args = parser().parse_args()
    overrides = args.set + ([f'seed={args.seed}'] if args.seed is not None else [])
    cfg = config(args.config, overrides)
    if args.limit < 0:
        raise ValueError('--limit must be nonnegative')
    if args.command == 'preflight':
        from src.pipeline import preflight
        preflight(cfg, check_model_processor=args.check_processor)
    elif args.command == 're-enrich':
        if not args.run_dir:
            raise ValueError('re-enrich requires --run-dir pointing to saved evidence')
        from src.re_enrich import re_enrich
        directory = re_enrich(args.run_dir, cfg, args.record_id)
        from src.inspection import export_inspection
        for record_id in args.record_id:
            export_inspection(directory, record_id, cfg)
    elif args.command in ('infer', 'benchmark', 'inspect'):
        from src.pipeline import run
        if args.command == 'benchmark' and not args.limit:
            raise ValueError('Benchmark requires an explicit --limit')
        if args.run_dir and not args.retry_degraded:
            raise ValueError('Inference --run-dir requires --retry-degraded; it selects recovery input, not output')
        if args.command == 'inspect' and len(args.record_id) != 1:
            raise ValueError('inspect requires exactly one --record-id')
        if args.command == 'inspect':
            cfg['policy']['vlm'] = True
            cfg['policy']['vlm_selection'] = 'all'
        run_dir = run(cfg, args.split, 1 if args.command == 'inspect' else args.limit, args.resume or args.retry_degraded,
                      args.command == 'benchmark', retry_degraded=args.retry_degraded,
                      recovery_dir=args.run_dir, record_ids=args.record_id)
        if args.command == 'inspect':
            from src.inspection import export_inspection
            export_inspection(run_dir, args.record_id[0], cfg)
        elif not args.limit and not args.record_id and not read_json(run_dir / 'manifest.json')['complete']:
            raise RuntimeError('Incomplete or degraded run: inspect status/events and use --retry-degraded for failed model stages')
    elif args.command == 'validate':
        from src.submission import validate_file
        path = (Path(args.file) if args.file else
                Path(read_json(Path(cfg['report_dir']) / f'latest_{args.split}.json')['run_dir']) / 'submission.csv')
        errors = validate_file(path, cfg)
        for error in errors:
            print(error)
        if errors:
            raise SystemExit(1)
        print('Submission schema, IDs, ordering and blank-output rules passed')
    elif args.command == 'compare':
        from src.evaluation import compare_runs
        if not args.runs:
            raise ValueError('--runs is required')
        compare_runs(args.runs, Path(cfg['report_dir']) / 'ablation_comparison.json')
    else:
        directory = (Path(args.run_dir) if args.run_dir else
                     Path(read_json(Path(cfg['report_dir']) / f'latest_{args.split}.json')['run_dir']))
        if args.command == 'evaluate':
            from src.evaluation import evaluate_run
            evaluate_run(directory, cfg)
        else:
            from src.submission import write_submission
            write_submission(directory, cfg)


if __name__ == '__main__':
    main()
