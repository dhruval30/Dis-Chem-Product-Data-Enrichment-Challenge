# Data

> Source: Zindi competition Data page, supplied 2026-11-14. Files are downloaded from the challenge Data tab into `../dataset/`.

## About

The dataset contains retail product records from several Dis-Chem product groups. Each record is one product with partial source information: a name, optional descriptions, optional nutritional information, optional specifications and catalogue classification fields. Some records include downloadable product images.

Information is intentionally missing or masked. The marker `[MASKED]` appears in source text fields for some records, and other cells are simply blank. Missing is not the same as "none": an absent fact must not be guessed.

The data is released under CC-BY-SA 4.0 and may be used for commercial, non-commercial, research and education purposes. Redistribution is permitted under that licence, but the repository still keeps the raw files out of Git for size and hygiene.

## Files

| File | Size | Description |
|---|---:|---|
| `Test_Masked.csv` | 1.0 MB | 1,157 records to enrich. `record_id` becomes `article_id` in the submission. Some source fields are blank or contain `[MASKED]`. |
| `unlabelled_data.csv` | 18 MB | 13,864 records with unmasked descriptions and specifications. No expected outputs. Use for catalogue understanding and approach development, not as an answer source. |
| `template_input.csv` | 15.2 KB | 14 example input records, in the test input format. |
| `template.csv` | 17 KB | Expected outputs for the 14 template input records. Match on `record_id` to `article_id`. Learning examples only; never include them in the submission. |
| `SampleSubmission.csv` | 21.4 KB | Format example with the required columns and empty cells. Not a source of product facts. |
| `submission_format.json` | 2.8 KB | Required output columns, JSON field shapes and empty values. |
| `columns.docx` | 8.8 KB | Describes each input and output column. |
| `product_enrichment_guardrail.pdf` | 21 KB | Content standard for evidence, wording, safety, field placement and missing information. Read before modelling. |
| `product_images.zip` | 2.3 GB | Downloadable product images for some records. Extracted locally to `product_images/`. |
| `image_mapping.xlsx` | 343 KB | Maps `record_id` to image filenames. A record with several images has several rows; an empty `img_filename` means no image. |

`product_images/image_mapping.csv` is the extracted copy of the mapping used locally. `product_images/images/` holds 18,145 JPG files.

## Input columns

| Column | Notes |
|---|---|
| `record_id` | Record identifier. Treat as text and copy exactly into `article_id`. |
| `category_group` | Product group, used as category context. Not an output. |
| `Product Name` | Supplied product name. The main identity signal. |
| `Short Description` | Source short description. May be blank or `[MASKED]`. |
| `Product Description` | Source long description. May be blank or `[MASKED]`. |
| `Nutritional Information` | JSON text, often `{}`. Empty object means no nutrition data supplied. |
| `Specifications` | Table, free text, blank or `[MASKED]`. |
| `T_DEPARTMENT` | Catalogue department classification. |
| `T_CATEGORY` | Catalogue category classification. |
| `T_BMC` | Catalogue BMC value. Use as provided. |
| `T_BRAND` | Catalogue brand. |
| `T_ARTICLE_CATEGORY` | Catalogue article category. |
| `T_WEIGHT_UNIT` | Unit only. A unit by itself is not a product weight. |
| `VOLUME` | Supplied volume value. Do not treat it as a dimension. |
| `T_VOLUME_UNIT` | Unit for `VOLUME`. |
| `do_not_predict_fields` | JSON array of output fields that do not apply to the record. Leave those cells blank; the evaluator applies the field's empty value. |

## Output schema

The submission has 11 columns: `article_id` plus the ten enriched fields. Structured fields are JSON encoded inside CSV cells.

| Field | Type | Shape |
|---|---|---|
| `article_id` | string | Input `record_id`. |
| `online_name` | string | Customer-facing product name with material variant, size or quantity. |
| `short_description` | string | Concise description. No bullets, no repetition of the long description opening. |
| `long_description` | string | Starts with the full product name and what it is or is for. |
| `ingredient_information` | array of strings | Ingredient names in source order. |
| `storage_instructions` | array of strings | Storage and use conditions from the source. |
| `variant_attributes` | object | `shade`, `flavour`, `fragrance`, `other`, each an array of strings. |
| `product_specifications` | object | `pack_size` (`value`, `unit`), `application_area` (array), `packaging_type` (string), `packaging_material` (array), `other` (array of `{attribute, value}`). |
| `weight` | object | `value`, `unit`. |
| `dimensions` | object | `height`, `width`, `depth`, `unit`. |
| `nutritional_information` | object | `serving_size` (`value`, `unit`), `per_100g` and `per_serving` (objects of nutrient names to numeric values). |

Empty values per field are defined in `submission_format.json`; a blank CSV cell is interpreted as that empty value. Do not write `N/A`, `none` or `unknown`.

`do_not_predict_fields` also contains names outside the output schema, such as `category_specific_fields`, `medicine_supplement_fields`, `cleaning_and_care` and `warranty`. Those are not submission columns and can be ignored.

## Measured schema

Measured locally from the supplied files on 2026-11-14.

- `Test_Masked.csv`: 1,157 rows, 16 columns. `record_id` is an integer in the file but must be emitted as the same text value.
- `unlabelled_data.csv`: 13,864 rows, 15 columns. Same input columns as test, without `do_not_predict_fields`.
- `template.csv`: 14 rows, 11 output columns.

Missing and masked values in the test file:

| Column | Blank | `[MASKED]` |
|---|---:|---:|
| `Short Description` | 476 | 0 |
| `Product Description` | 476 | 0 |
| `Specifications` | 964 | 193 |
| `Nutritional Information` | 0 | 23 |
| `VOLUME` | 180 | n/a |
| `T_VOLUME_UNIT` | 175 | n/a |

Non-empty nutritional JSON appears in 23 test records and 451 unlabelled records; the rest are `{}`.

Product groups in test: BEAUTY 115, SPORTS RELATED 108, NUTRITION CATEGORY 103, VITAMINS / HERBAL 98, PERSONAL CARE 96, PERFUMERY 96, BABY FOOD AND FEEDING 85, BABY TOYS 84, COSMETICS 82, BABY HARDWARE 62, BABY CLOTHING 55, BABY HYGIENE 41, BABY HOME AND SAFETY 41, HAIR SALON PROFESSIONAL 23, DIAPERS AND WIPES 18, BABY HEALTH 18, HOME SCENTS 17, MATERNITY 12, plus one each for BEAUTY EXPENSE, BABY FMCG and BEAUTY SALON EXPENSE.

Images exist for 840 of the 1,157 test records and 10,014 of the 13,864 unlabelled records. Every test `record_id` appears in the image mapping, but some rows have an empty filename.

## Data handling

- Raw data is read-only. Never edit files in `dataset/` in place.
- `dataset/` is gitignored; do not force-add the images or CSVs.
- Do not paste raw rows or identifiers into documentation or commit messages.
- The guardrails forbid using competitor pages, reviews, forums, search snippets, generic web pages or AI-generated answers as evidence. Catalogued facts must come from the supplied record, the supplied images where the task allows them, or approved product material.
