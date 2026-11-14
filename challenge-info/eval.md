# Evaluation

> Source: Zindi competition Evaluation page, supplied 2026-11-14.

## Metric

Phase 1 is scored by an LLM judge that compares each prediction with reference product content. Exact wording is not required when the meaning is equivalent. Higher is better; the score is in [0, 1].

### Text fields

`online_name`, `short_description` and `long_description` receive a semantic quality score per product, judged on correctness, completeness and whether the material meaning of the reference is preserved.

| Score | Meaning |
|---:|---|
| 1.00 | Correct and complete |
| 0.75 | Mostly correct, with minor omissions or differences |
| 0.50 | Partially correct |
| 0.25 | Limited correct information |
| 0.00 | Unusable or materially different |

Equivalent wording can receive full credit. Missing or incorrect information lowers the score.

### Structured fields

The remaining fields contain lists or structured objects. The judge compares each non-empty value with the reference by meaning, including units, scope and basis, and considers values within the JSON structure rather than the whole cell as text. The reference is the source of truth; the judge does not consult product images or other sources.

Each reference value can be matched once:

- A matched reference value is a true positive (TP).
- A reference value with no matching prediction is a false negative (FN).
- A prediction value that does not match is an extra. The judge classifies each extra as duplicate, acceptable or unacceptable. Duplicates and harmless compatible details are not penalised; incorrect, conflicting or otherwise unacceptable extras are false positives (FP).

The structured field score is:

```
F1 = 2 * TP / (2 * TP + FP + FN)
```

This balances correct information against missing information and unacceptable additions.

### Overall score

- Each text field score is averaged across the products where that field has reference information.
- For each structured field, TP, FP and FN are combined across evaluated products before computing F1.
- The overall score is the arithmetic mean of the evaluated field scores, with equal weight per field.
- A field without reference information for a product is not evaluated for that product, and fields with no evaluable reference information are excluded from the average.

Scoring a submission takes roughly 30 to 40 minutes.

## Public and private leaderboard

Zindi maintains a public leaderboard during the challenge and a private leaderboard released at close. The public split is a minority of the test set and the private split is the remainder, so public movements are a noisy guide.

The supplied rules state `about 30%` public / `70%` private in one place and `approximately 20%` public / `80%` private in the detailed section. Treat the exact proportion as unconfirmed and assume the public score is noisy.

## Submission file

A CSV with the columns in `submission_format.json`, one row per test `record_id` and a header row. `article_id` must equal the input `record_id` exactly. Structured cells contain JSON; blank cells are read as the field's empty value.

The file is submitted through Zindi, not Kaggle. Use `SampleSubmission.csv` for the column order and `template.csv` for value examples.

## Phase 1 constraints

- End-to-end inference: 3 hours maximum.
- All-in inference cost: US$20 maximum.
- One GPU with at most 24 GB VRAM.
- At most one retry per failed call, counted inside the time and cost caps.

## Validation notes

There is no labelled validation set. The 14 template records are the only examples with reference outputs, so local scoring is limited to:

1. Schema and JSON validity against `submission_format.json`.
2. Field coverage and empty-value conformance, including `do_not_predict_fields`.
3. Evidence checks: every extracted fact traceable to the supplied record or image.
4. A manual or LLM-judge review of a small sample, written to a report, not used to overfit the 14 templates.
