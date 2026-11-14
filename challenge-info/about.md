# About

> Source: Zindi competition Overview page, supplied 2026-11-14. Competition: Dis-Chem Product Data Enrichment Challenge, hosted by Zindi with Dis-Chem and Bigly Labs.

## Overview

Dis-Chem is a South African pharmacy and health retailer with tens of thousands of products across health, baby, beauty, food and household. Product information is spread across several systems and is often incomplete or inconsistent: names are abbreviated, descriptions missing, attributes disagree between sources, and ingredients, usage instructions or warnings are not structured.

The task is to build an end-to-end AI solution that enriches, validates and standardises product catalogue records. Each input record contains partial product information, structured catalogue fields and links to available product images. Some information is intentionally removed or masked to simulate real-world gaps.

The solution must recover missing information, extract attributes, place them in a predefined structured format and ensure every fact is supported by an acceptable source. Generating plausible descriptions is not the goal: accuracy, completeness, consistency and traceability matter, and unsupported or invented content is penalised.

## How it works

### Phase 1: Online submission (13 November 2026 to 14 December 2026)

Competitors enrich the masked test records and submit a CSV in the format defined by `submission_format.json`. Submissions are scored automatically with an LLM judge that compares predictions against reference product content. The top 10 on the private leaderboard are asked for their code, and five competitors are selected to proceed to Phase 2.

Phase 1 constraints:

- End-to-end inference time: 3 hours maximum.
- All-in inference cost: US$20 maximum.
- One GPU with at most 24 GB VRAM.
- At most one retry for a failed call, included in the time and cost caps.

### Phase 2: End-to-end system (25 December 2026 to 15 January 2027)

Selected competitors build a full production system for more detailed enrichment. The client constraints are announced when Phase 2 launches, along with its evaluation criteria and ranking procedure.

## Prizes

| Place | Prize |
|---|---:|
| 1st | US$2,500 |
| 2nd | US$1,500 |
| 3rd | US$1,000 |

There are 5,000 Zindi points available for the challenge.

## Timeline

All dates 2026.

| Event | Date |
|---|---|
| Phase 1 opens; submissions open | 13 November |
| Phase 1 closes; submissions close | 14 December |
| Phase 2 opens | 25 December |
| Phase 2 closes | 15 January 2027 |

The exact closing time and timezone for each phase are given on the Zindi challenge page.

## Eligibility and teams

- Open to all, individual or teams of up to four.
- Each team member must have their own Zindi account, and multiple accounts are not permitted.
- No sharing of code or information between accounts outside a team.
- No additional team members may be added in the final 5 days of the challenge.
- Organisers: Dis-Chem and Bigly Labs, administered by Zindi.
