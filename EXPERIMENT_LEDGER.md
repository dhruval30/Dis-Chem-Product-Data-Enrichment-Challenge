# Experiment Ledger

One row per experiment. Local score is the validation score from our own split; LB score is the public leaderboard score for that submission. Note the leaderboard window or data release when comparing scores, because the public split can change between submissions.

Add the entry in the same commit as the experiment code and its results.

| Experiment | Description | Local score | LB score |
| --- | --- | --- | --- |
| 001_competitive_multimodal | Evidence-grounded catalogue enrichment with deterministic extraction, RapidOCR, local Qwen3-VL, retrieval, verification and M2-safe inference | not scored | not submitted |
| 002_vllm_single_pass | Single-pass VLM (Qwen3.5-9B FP8 via vLLM) with deterministic spec, nutrition and name parsing, recall-oriented merge and a silver validation set | pending | pending |
