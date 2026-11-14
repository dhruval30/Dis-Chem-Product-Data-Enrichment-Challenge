# Product Enrichment Guardrails

> Source: `dataset/product_enrichment_guardrail.pdf`, supplied 2026-11-14. This is a working summary; the PDF controls. The guards are about content, while the submission schema controls field names and format.

## 1. Evidence comes first

Match the exact product, barcode and variant before using a source. Use current packaging and labels, official product inserts, approved brand material and approved product records that match the item. For regulated products, rely on exact-product approved wording and supplied regulatory information.

- Do not use competitor or retailer pages, reviews, forums, search snippets, generic web pages, AI-generated answers or hidden SEO/schema fields as evidence.
- Do not browse or use general knowledge to fill a gap. A source must support the exact meaning, qualifier, strength, amount, unit and variant.
- If approved sources disagree, omit the disputed fact and flag it for review. Do not choose a source by guesswork.

Order of authority: mandatory legal or pack wording, then category-specific requirements, then this content standard, then the task's field and format rules.

## 2. Write precisely; do not overclaim

Use clear, neutral, original customer-facing language. Prefer short, direct sentences and consistent UK English. Preserve the source's certainty, conditions, quantities and units. Do not copy long passages or merely swap a few words.

- Never infer safety, suitability, compatibility, efficacy, performance or a free-from status from a product name or from missing information.
- Keep every qualifier: "may help" is not "will". A claim about one ingredient is not automatically a claim about the whole product.
- Do not add guarantees, superiority, clinical proof, disease treatment or health outcomes unless an approved exact-product source supports the same claim at the same strength.

## 3. Extra care for health and safety

- **Medicines and supplements:** reproduce dosage, warnings, contraindications and regulated wording exactly. Keep descriptive copy factual; never turn source silence into advice about children, pregnancy, interactions or side effects.
- **Food and allergens:** retain ingredient order and every declared allergen and trace qualifier. Never imply "allergen-free" unless exact approved wording permits it.
- **Baby, medical and safety products:** do not promise safety or risk reduction beyond approved product evidence.

## 4. Field rules

- **Product name:** approved brand and product name with material variant, size or quantity. Keep the exact model; do not invent, abbreviate or truncate identifying details.
- **Short description:** concise summary of what the product is or does. No bullets, and do not repeat it as the opening of the long description.
- **Long description:** start with the full product name and what it is or is for. Add only useful supported details; do not duplicate dedicated structured fields.
- **Ingredients:** preserve source order and wording. Distinguish a complete ingredient declaration from named ingredients; retain "may contain" and allergen qualifiers exactly.
- **Nutrition:** preserve the printed basis (per 100 g/ml and per serving), serving size and units. Do not convert between bases or fill blanks by guessing.
- **Specifications, weight and dimensions:** record explicit values only. Do not infer a missing unit, axis, or net/gross weight. Do not relabel length as depth; use height x width x depth only where the source labels support it.
- **Variants:** keep shade, flavour, fragrance, size and other variant facts separate. Reuse shared copy only when it is true for every variant.
- **Use and storage:** preserve supplied method, dose, frequency, temperature, warnings and storage conditions. Do not invent directions or compatibility.

## 5. When information is missing

Leave unsupported optional fields blank or omit them. Do not write "N/A", "none", "unknown" or a default. If a required fact conflicts, fails to match the product or variant, or cannot be verified, stop and flag it for review instead of guessing.

## 6. Before release

- Does every value match the exact product and variant?
- Can every claim be traced to an approved source with its qualifiers intact?
- Are numbers, units, allergens and nutrition bases correct?
- Do the descriptions agree with the structured fields, and does the output follow the required schema?
- If images are in scope, do they show the matching product and variant without implying unsupported facts?

Final rule: an accurate omission is better than a plausible invention.
