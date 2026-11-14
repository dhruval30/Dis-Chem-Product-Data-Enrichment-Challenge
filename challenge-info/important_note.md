# Important Notes

Compiled 2026-11-14 from the Zindi Overview, Data, Evaluation and Rules pages plus the supplied product enrichment guardrails. Read this before modelling.

## Priority notes

- **Evidence is restricted to the supplied material.** The guardrails prohibit competitor or retailer pages, reviews, forums, search snippets, generic web pages, AI-generated answers and hidden SEO fields as evidence. The rules also say only the provided datasets may be used. Do not browse for facts and do not fill gaps from general knowledge.
- **An accurate omission beats a plausible invention.** If a fact cannot be verified from the record or its images, leave the field blank. The judge penalises incorrect and unsupported extras.
- **The public/private split is ambiguous.** One section says 30/70, another says 20/80. Treat the public score as noisy and avoid decisions based on small movements.
- **`[MASKED]` is not empty and empty is not "none".** Do not write `N/A`, `none`, `unknown` or a default. Blank cells take the field's empty value shape.
- **`do_not_predict_fields` includes names outside the output schema** (`category_specific_fields`, `medicine_supplement_fields`, `cleaning_and_care`, `warranty`). Only the ten submission fields matter; leave a field blank when it is listed and it is a submission column.

## Things that affect modelling

- **Ten output fields, three of them free text.** `online_name`, `short_description` and `long_description` are judged semantically. Be correct and complete rather than verbose, and do not let the short and long descriptions duplicate each other.
- **Structured fields need exact units and bases.** Preserve the printed basis for nutrition, the labelled axis for dimensions, and the source unit for weight. Do not infer a missing unit or relabel length as depth.
- **No inferring safety or efficacy.** Never turn source silence into advice on children, pregnancy, interactions or side effects. Keep qualifiers such as "may help" intact.
- **Similar products differ in pack size, flavour, shade, weight and dimensions.** Confirm the exact variant before transferring any detail from another record.
- **Unlabelled data is context, not answers.** Use it to learn catalogue patterns and field conventions; it has no reference outputs.
- **Only 14 reference examples.** The template is small enough to overfit. Use it to learn the schema and house style, then generalise.

## Things that affect data handling

- **Raw files stay out of Git.** `dataset/` is gitignored, including the 18k images. Do not force-add them.
- **No raw rows or identifiers in docs.** Keep examples and reports aggregate-level.
- **The licence is CC-BY-SA 4.0.** Sharing is permitted under that licence, but the repository still treats the catalogue as competition material.
- **Treat `record_id` as text.** Preserve it exactly when writing `article_id`.

## Things that affect the final submission

- **Format:** CSV with `article_id` plus the ten output columns. Structured cells contain JSON.
- **Row count:** exactly 1,157 rows, one per test record, plus the header.
- **Constraints:** three hours end to end, US$20 all-in, 24 GB VRAM, one retry.
- **Budget:** 2 submissions per day and 20 total. Submit deliberately.
- **Deadline:** Phase 1 closes 14 December 2026. Phase 2 runs 25 December 2026 to 15 January 2027 for the top five.
- **Code review:** the top 10 are asked for code and must reproduce the score within 48 hours. Keep the pipeline scripted, seeded and pinned.

## Open questions

- Exact Phase 1 closing time and timezone.
- Whether image content may be used as evidence for `online_name` and descriptions, or only for validation. The guardrails say images in scope must show the matching product and variant without implying unsupported facts.
- Whether the LLM judge penalises an `online_name` that differs from the supplied `Product Name` only in capitalisation or punctuation.
