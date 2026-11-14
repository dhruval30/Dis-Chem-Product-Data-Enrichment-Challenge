# Additional Datasets

> Source: Zindi competition Rules page, supplied 2026-11-14.

## Policy

The challenge rules are restrictive: use only the datasets provided for the challenge. External data is not permitted, even if it is public and free. Openly available pretrained models are allowed as tools, but they must not be used to fetch or inject facts that are not supported by the supplied records.

The product enrichment guardrails reinforce this. Browsing, competitor and retailer pages, reviews, forums, search snippets, generic web pages, AI-generated answers and hidden SEO fields are not acceptable evidence.

## Current external sources

None. No external data will be used.

## Candidates to evaluate

Only openly available pretrained models and packages are candidates. For each, record the licence, availability and role.

| Field | Value |
|---|---|
| Source | |
| Link | |
| Licence | |
| Size | |
| Why it might help | |
| Leakage risk | |
| Available at prediction time | |
| Decision | |

## Rules of thumb

- Prefer local open-weight models that fit within 24 GB VRAM and the US$20 cost cap.
- No paid APIs, credit-card trials or gated models.
- No AutoML tooling.
- A pretrained model may extract and structure facts from the supplied record, but it must not supply product facts from its own memory.
- Declare every model and package used in the write-up and the experiment ledger.
