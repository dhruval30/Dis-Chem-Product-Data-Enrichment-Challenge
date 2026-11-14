# Dis-Chem Product Data Enrichment Challenge

Competition workspace. The official pages are mirrored in `challenge-info/`. Fill in the competition-specific facts there before modelling; the sections below are the working conventions that stay the same across experiments.

## Layout

- `challenge-info/` holds the competition pages and content standard: `about.md`, `data.md`, `eval.md`, `rules.md`, `important_note.md`, `additional_datasets.md`, `guardrails.md`. Read `important_note.md` and `guardrails.md` first; site rules often contradict each other in places.
- `dataset/` holds the raw competition files. Treat it as a read-only input; never edit files in place. The official guardrail PDF lives here; `challenge-info/guardrails.md` is the working summary.
- `eda/` contains exploratory-data-analysis scripts and `eda_report.md`.
- `eda/figures/` contains EDA figures.
- `eda/csvs/` contains EDA CSV outputs.
- `exp/` contains standalone experiments. Each experiment uses a zero-padded numeric prefix and a descriptive name, for example `exp/001_descriptive_name/`. It must run without depending on another experiment or its results.
- `results/` mirrors experiments, for example `results/001_descriptive_name/`, and contains all outputs for its corresponding experiment.
- `EXPERIMENT_LEDGER.md` records every experiment.
- `lb.md` is a snapshot of the public leaderboard. Re-scrape it to refresh.
- `venv-dischem/` is the project virtual environment. It is gitignored.

## Experiment ledger

- Maintain `EXPERIMENT_LEDGER.md` in the repository root. Add an entry for every experiment with its identifier, short description, local score and leaderboard score.
- When asked to commit an experiment, commit its code, matching results and ledger entry together.
- An experiment commit message contains the experiment description and its local and leaderboard scores.

## How to talk to me

Match my usual chat style: lowercase, casual, direct, a little colloquial, but technically sharp. No corporate voice, unnecessary enthusiasm, emojis or em dashes. Don't sound like a press release.

Keep replies short by default. Say what changed, why it matters and anything that needs my decision. Go into detail when the issue genuinely calls for it. Don't pad an answer with generic explanations or constantly reassure me.

Example tone (a reference, not a template):

> yeah, the join is multiplying rows because a user has multiple interaction records per item. i'll fix the aggregation grain and rerun the checks.

## Documentation style

Documentation should be professional, grammatically correct, straightforward and human. Follow the direct style of `README.md`, with proper grammar for project documentation.

- Use plain English and concrete technical details.
- Prefer short paragraphs and descriptive headings.
- Explain what something does, why it matters and how to use it. Avoid marketing fluff and inflated claims.
- No filler such as "leveraging cutting-edge technologies", "seamlessly revolutionizing" or "in today's rapidly evolving landscape".
- Don't make documentation artificially long or over-polished. Give enough detail to make the project understandable and reproducible, then stop.
- Don't present simulated results as real-world business outcomes. Be precise about what was measured and what is only proposed.
- No assistant credits, generated-by footers or co-author attribution to coding tools.

## Git and commits

**Never commit without asking first.**

Work in meaningful, reviewable chunks. After completing a coherent change:

1. Summarize what changed and any checks you ran.
2. Ask whether to commit that chunk.
3. Commit only after an explicit yes. If the answer is no, continue without committing.
4. Stage only the files relevant to that chunk. Don't bundle unrelated changes, overwrite my work or make one massive end-of-session commit.

Use short, meaningful, lowercase commit messages that sound like something I'd write. Be specific about the change, not the tool used to make it.

Good examples: `add source record dedup`, `fix user-item mapping joins`, `add validation split retry state`.

Avoid vague messages like `updates`, bloated commit messages, assistant names, co-author footers or any generated-by attribution. Don't amend, force-push or change history unless I explicitly ask.

## Working conventions

- Set the random seed everywhere. Re-running an experiment must reproduce its reported score.
- Validate in a way that mirrors the leaderboard setup. Split by the grouping or time structure the competition uses, and never let validation rows leak information from the scored set or from the future.
- Use only information that would be available at prediction time for the target row.
- Every emitted fact must trace to the supplied record or its images. Do not browse, and do not fill gaps from model memory or general knowledge. An accurate omission beats a plausible invention.
- Never write `N/A`, `none`, `unknown` or a default. Leave unsupported fields blank so the evaluator applies the schema's empty value.
- Respect the Phase 1 caps in the design: three hours end to end, US$20 all-in, 24 GB VRAM, one retry.
- Prefer small, reproducible scripts per experiment (`run.py` with a seed argument). Write all outputs, logs and submission files under `results/<experiment>/`.
- Pin package versions in each experiment's `requirements.txt`. Use the shared `requirements.txt` only for packages needed across the project.
- Keep notebooks for exploration only. Any result that goes into the ledger must be reproducible from a script.
- Keep raw inputs read-only. Never edit `dataset/` in place.
- Check available disk before writing large artifacts and clean up scratch files when done.

## Shared feature cache

- Reuse compatible engineered matrices from the Git-ignored `cache/<fingerprint>/` directory; do not create full copies per experiment or recipe.
- Cache keys depend on input data and schema, feature definitions and ordering, precision and preprocessing versions, never experiment IDs or model hyperparameters.
- Store new engineered features separately from reusable base matrices.
- Preserve existing artifacts and caches. Check for active users before moving a cache, and use a same-filesystem `mv`, never a copy. Do not delete or migrate files in use.
- Model and LLM calls are expensive against the time and cost caps. Cache parsed inputs, prompts and responses keyed by record hash plus prompt and model version, and reuse them across experiments instead of re-calling.

## Competition specifics

Source of truth is `challenge-info/`. Re-read `important_note.md` and `guardrails.md` before changing the extraction or submission setup.

- **Task:** enrich masked Dis-Chem catalogue records into ten product-content fields using only the supplied record, its images and the supplied guardrails.
- **Metric:** LLM-judged score in [0, 1]. `online_name`, `short_description` and `long_description` are scored semantically (1.00, 0.75, 0.50, 0.25, 0.00). Structured fields use per-value matching and `F1 = 2*TP / (2*TP + FP + FN)`. Overall score is the equal-weight mean of the evaluated fields.
- **Deadline:** Phase 1 runs 13 November to 14 December 2026. Phase 2 runs 25 December 2026 to 15 January 2027 for the top five. Exact closing times are on Zindi.
- **Submission format:** CSV with `article_id` plus ten output columns, 1,157 rows, header row. Structured cells are JSON. `article_id` must equal the input `record_id` exactly.
- **Leaderboard split:** public and private splits are minority and majority of the test set. The rules state both 30/70 and 20/80, so treat the public score as noisy.
- **Submission limits:** 2 per day, 20 total, 2 selected for judging.
- **Data:** `Test_Masked.csv` is 1,157 records and `unlabelled_data.csv` is 13,864 records with unmasked descriptions. `template.csv` has 14 reference outputs. 840 test records and 10,014 unlabelled records have at least one image.
- **Data handling:** read-only, `dataset/` gitignored, no raw rows or identifiers in docs. Licence is CC-BY-SA 4.0. Only the provided datasets may be used; no external data.
- **Validation:** there is no labelled validation set. Check schema and JSON validity, empty-value conformance against `do_not_predict_fields`, evidence traceability, and a small reviewed sample. Do not tune against the 14 template rows.
- **External data:** not permitted. Openly available pretrained models are allowed as tools but must not supply unsupported product facts.
- **Compute:** Python only, open-source packages, one GPU up to 24 GB VRAM, US$20 all-in, three hours end to end.
- **Code review:** top 10 are asked for code within 48 hours and must reproduce the score. Keep the pipeline scripted, seeded and pinned.
