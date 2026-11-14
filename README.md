# Dis-Chem-Product-Data-Enrichment-Challenge

Competition workspace for the Zindi Dis-Chem Product Data Enrichment Challenge: enrich masked retail product records into ten customer-facing content fields using only the supplied data and images. Scored by an LLM judge on semantic quality for three text fields and F1 over structured values. Phase 1 runs 13 November to 14 December 2026.

- Challenge details: [`challenge-info/`](challenge-info/)
- Content standard: [`challenge-info/guardrails.md`](challenge-info/guardrails.md)
- Workflow and conventions: [`AGENTS.md`](AGENTS.md)
- Experiment history: [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md)
- Public leaderboard snapshot: [`lb.md`](lb.md)
- Data: `dataset/` (competition files; add them locally)

## Constraints

- Ten output fields, 1,157 test records, CSV with JSON cells for structured fields.
- Three hours end to end, US$20 all-in, one GPU up to 24 GB VRAM, one retry.
- Evidence only from the supplied record and images. No browsing, no external data, no invented facts.
- 2 submissions per day, 20 total.

## Quick start

```bash
source venv-dischem/bin/activate
python exp/001_descriptive_name/run.py
```

Each experiment lives in `exp/<id>_<name>/` and writes its outputs to the matching `results/<id>_<name>/` directory.
