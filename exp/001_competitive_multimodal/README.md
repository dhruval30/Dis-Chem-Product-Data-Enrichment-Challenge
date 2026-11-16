# Experiment 001: Competitive Multimodal Product Enrichment

This experiment combines exact extraction, packaging OCR, local vision-language inference and constrained catalogue retrieval. The hypothesis is that recovering explicit facts and writing focused descriptions will improve both structured-field recall and semantic content quality without adding unsupported product information. No score, runtime or hardware fit has been measured. This is an implementation awaiting execution and review, not a validated competitive result.

## Competition interpretation

The implementation follows the supplied `submission_format.json`, column definitions, guardrail PDF and mirrored challenge pages. The official evaluation averages evaluated fields equally. Structured fields aggregate TP, FP and FN across eligible products before F1; empty references are excluded. The three text fields use the hidden semantic judge. Local lexical diagnostics do not reproduce that judge or its distinction between acceptable and unacceptable extras.

The guardrail PDF controls evidence handling even when the 14 template outputs contain facts unavailable from their masked inputs. Missing units, nutrition inequalities and unsupported axes are not guessed to imitate those examples. `VOLUME` can describe cubic packaging volume. It is never used as weight or physical dimensions. `T_WEIGHT_UNIT` alone supplies no weight. Fields in `do_not_predict_fields` are blank CSV cells, including when an extraction could populate them. Names outside the ten submission fields are ignored.

Only supplied data is used for product facts. Exact catalogue matches require full normalised name, brand and category agreement. Family matching additionally uses the quantity-stripped name, catalogue hierarchy and identity-token overlap. Fuzzy similarity alone never permits copying a fact. Ordinary family descriptions remain non-transferable context. Shared structured attributes need agreement across at least two distinct named products without contrary family declarations. Packaging type/material also need direct target-text corroboration. Application area can be reused for the same fully named product differing only in size. No family route transfers ingredients, nutrition, warnings, variants, quantities, weights or dimensions. Every approved family fact is restricted to its recorded field, path and value. This narrow size-invariance assumption needs sample review; disable it with `retrieval.allow_family_transfer=false` if inappropriate for a category.

The repository's `challenge-info/eval.md`, `rules.md` and `important_note.md` explicitly state three hours and US$20. They are mirrored summaries, and the user reports that the currently accessible official Phase 1 page does not establish these local caps. The discrepancy is unresolved; this experiment does not treat them as confirmed requirements. `budget.max_seconds` and `budget.max_cost_usd` default to `null`. Configure operational limits explicitly, for example `--set budget.max_seconds=10800 --set budget.max_cost_usd=20`, if you want those ceilings. Memory monitoring, local cost estimates, cumulative timing and one failed-call retry remain active. Native calls cannot be preempted mid-call, and nothing here establishes runtime or cost compliance.

## Architecture

### Product-first content routing

The latest quality revision replaces caption-based fallbacks with `src/content.py`. Existing saved inspections showed that good packaging OCR was present but never reached safe-mode copy: the old fallback read only CSV descriptions and VLM observations, rejected supported benefit verbs indiscriminately, and promoted the first decorative observation into the short description. Appearance captions were also inserted as arbitrary visible_* specifications. Those were routing/selection failures, not evidence that OCR or the local model could not read the product.

The content router now corroborates OCR product identity using brand and product-name anchors, independently from the VLM's identity status. `association_basis` records the supplied mapping, `identity_basis` records visible corroboration, and `vlm_identity_status` retains the model's separate assessment. Unidentified or mismatched packaging cannot silently establish health claims. Derived grounded_text sources retain exact parent IDs/quotes, permitted text normalisation, actual OCR confidence and a content role. Tick prefixes, wrapped lines, ampersands and missing spacing can be normalised; numbers, units and qualifiers are not invented or converted.

Supported product descriptions, benefits, label declarations and explicit usage wording take priority over appearance. A statement containing "helps" or "may help" is not automatically rejected if the supplied exact-product evidence supports it. Hype, guarantees and unsupported treatment claims remain excluded. Label claims retain attribution. Short copy uses a concise identity span or an existing supported summary; long copy starts with the full name and adds distinct useful evidence. Fallback selection no longer rewards longer generated text merely for being longer. Artwork, logos and box illustrations remain available for inspection but do not become product copy or specifications. Actual garment/accessory appearance can support one restrained design sentence. Visual observations do not automatically become structured specification entries.

The official ten-field schema and applicability rules are unchanged. Printed quantities remain distinct from weight/dimensions and photo counts. Ingredients are not inferred from a product name, benefit or named ingredient. Nutrition basis and values remain source-exact; additional declared fat labels are normalised to consistent unit-bearing keys. The M2-safe profile still performs only its bounded image stage, but its CPU-only content routing now uses the already collected OCR instead of discarding it.

### Re-enrich existing inspections without model calls

Start with the existing three-record evidence run. This command loads saved evidence and applies the current routing, verification and copy composition. It does not load model weights, initialise OCR, run generation or repeat retrieval:

```sh
python -m pytest exp/001_competitive_multimodal/tests -q
python exp/001_competitive_multimodal/run.py re-enrich \
  --config exp/001_competitive_multimodal/profiles/m2_safe.yaml \
  --run-dir exp/001_competitive_multimodal/outputs/runs/test/9fb917cf3e17fdc9d7ba \
  --record-id 310290 --record-id 1464 --record-id 1537
python exp/001_competitive_multimodal/run.py evaluate \
  --config exp/001_competitive_multimodal/profiles/m2_safe.yaml --split test
```

Outputs go to `outputs/re_enriched/<fingerprint>/`; original runs and caches are untouched. Each selected record receives a new inspection export with `product_content.json`, accepted/rejected evidence, source-linked copy, fields and original raw-response diagnostics where available. The fingerprint includes source-record bytes, current code and effective configuration. Changed input files are refused. Re-enrichment preserves a degraded source run's degraded status and retains original extraction budget information; zero new calls does not mean the original extraction was free. Its result cannot contain information the original extraction missed.

After reviewing these outputs, inspect a fresh small cohort with the safe profile, then run template/pseudo validation and the existing ablations. Move to the competitive eight-record path before considering full inference. For one full competitive run, explicitly choose:

```sh
python -u exp/001_competitive_multimodal/run.py infer \
  --config exp/001_competitive_multimodal/config.yaml --split test --resume --retry-degraded
python exp/001_competitive_multimodal/run.py evaluate --split test
python exp/001_competitive_multimodal/run.py submission --split test
python exp/001_competitive_multimodal/validate.py --split test
```

Do not release until the full run is complete, evidence review and schema checks pass, degraded records are resolved and measured memory/runtime are acceptable. Submission remains a separate manual step and nothing uploads to Zindi. Routing refinements reuse raw OCR automatically when preprocessing/settings/dependencies are unchanged. Image-prompt changes intentionally invalidate incompatible VLM requests; use re-enrich for cost-free comparison of postprocessing changes instead of forcing new extraction. No revised test, inference or performance evaluation has been run by the implementer.

### Visual evidence v2

The image stage now returns two evidence groups: exact printed text and directly observable appearance. Identity is `confirmed`, `unverifiable` or `mismatch`. A mapped, unbranded product photograph can supply appearance evidence with unverifiable identity; it does not prove the manufacturer's identity or commercial attributes. Explicit conflicting printed identity or incompatible product form quarantines the image and its OCR evidence. Missing identification alone does not quarantine OCR.

`src/visual.py` validates the versioned response schema and applies rule-based reliability labels. New visual sources do not carry model-invented confidence probabilities. Sources retain image path/index, identity basis, visible support, printed kind and any independently agreeing OCR source. Printed identity needs brand/name anchors; partial identification is downgraded to unverifiable. High-risk declarations, warnings, nutrition, directions and regulated content require OCR corroboration. Ordinary exact-product printed claims remain attributed label statements. Ingredient highlights go to labelled specifications, never to a complete ingredient list.

Appearance can describe form, colours, patterns, decoration, shape and visible design. Materials, tactile/performance properties, measurements, suitability and efficacy are rejected as appearance inferences. Selected appearance features can populate explicitly named `visible_*` entries in specifications.other. Apparent colours cannot establish official shades. Visible counts are used only in photo-attributed descriptive copy and cannot establish pack quantity. Printed variants need explicit variant labels for structured placement. The supplied submission schema is unchanged.

Accepted observations reach structured extraction and both description paths, including deterministic visual-copy fallbacks when text generation is disabled or fails. A sparse short description can combine the supplied name with a directly observed feature. Long copy begins with the supplied name and can add a few appearance details or explicitly attributed label claims. Observation source eligibility is rechecked at release; mapping alone never authorises arbitrary claims. These filters reduce risk but do not prove a VLM observation is visually correct.

OCR still runs first at its separate configurable high resolution. OCR context is associated with the selected image index, not pooled indiscriminately across images. Matching OCR/VLM facts are deduplicated downstream. New extraction uses one evidence pass; uncorroborated high-risk details remain review-only instead of requiring repeated GPU reads. Legacy extraction handling remains for existing fixtures, but v2 prompts, schema configuration and content-keyed caches prevent silent reuse of legacy image responses.

### Manual record inspection

The compact prompt now puts JSON metadata inside an explicit INPUT_CONTEXT block first and places the complete output contract after it. It explicitly forbids echoing Product Name/T_BRAND as output keys and requires visual-evidence-v2, one image object and the printed/observation structures. Textless images are asked for appearance observations with unverifiable identity. The previous compact prompt ended with the target metadata JSON, a plausible echo cue; this is a prompt-level explanation, not an experimentally proven cause. Chat-template and image-attachment APIs are unchanged.

The safe output remains capped at 256 tokens. The compact contract limits output to two short observations and one short printed block, omitting optional identity strings. These bounds are intended to leave room for complete JSON; actual token usage and response adherence remain untested. Strict schema validation still rejects a metadata echo instead of fabricating missing evidence fields. A changed assembled prompt and the existing content-keyed cache logic prevent reuse of the old request.

Both the assembled instruction string and the chat-formatted prompt are saved under `diagnostics/prompts/`, with fsynced files and prompt_saved events. Inspection exports copy them into `assembled_prompts/`. Raw model responses remain in model_responses/vlm_raw/ even when parsing or schema validation fails. Compare those files to inspect exactly what was sent and returned.

Use this command to inspect the textless record already checked on the M2 Air:

```sh
python -u exp/001_competitive_multimodal/run.py inspect \
  --config exp/001_competitive_multimodal/profiles/m2_safe.yaml \
  --split test --record-id 310290 --resume --retry-degraded
```

Replace the ID to inspect another product. Inspection forces VLM selection for that record while retaining the selected profile's image, token and stage limits. It reuses compatible caches and writes a timestamped directory beneath the current run's `inspections/`. It exports original image-path references, raw OCR caches, aligned OCR text, raw/parsed VLM cache files, visual sources with provenance, accepted/rejected facts and reasons, description traces, final enriched fields, status and diagnostics. It does not open files, launch a UI, generate a submission or overwrite historical run results. A processor failure can still produce an inspection export showing missing model responses and preserved errors.

M2-safe inspection keeps one image, a 512 px bound and 256 output tokens. Its compact prompt covers printed identity/quantity plus appearance, rather than all tiny label panels; structured/prose model stages remain disabled, but verified visual evidence can still supply deterministic fields and grounded copy. Competitive inspection uses `--config exp/001_competitive_multimodal/config.yaml` and may process more images and longer label text, with the existing larger memory/runtime risk. Products without mapped images remain supported by the competitive profile; the image-only smoke profile deliberately requires an image.

Manual checks after reviewing this revision:

```sh
python -m pytest exp/001_competitive_multimodal/tests -q
python -u exp/001_competitive_multimodal/run.py infer \
  --config exp/001_competitive_multimodal/profiles/m2_safe.yaml \
  --split test --record-id 310290 --limit 1 --resume --retry-degraded
python -u exp/001_competitive_multimodal/run.py infer \
  --config exp/001_competitive_multimodal/profiles/m2_safe.yaml \
  --split test --limit 3 --resume --retry-degraded
python exp/001_competitive_multimodal/run.py evaluate \
  --config exp/001_competitive_multimodal/profiles/m2_safe.yaml --split test
```

Only when you choose the competitive path:

```sh
python -u exp/001_competitive_multimodal/run.py infer \
  --config exp/001_competitive_multimodal/config.yaml \
  --split test --limit 8 --resume --retry-degraded
```

No revised prompt, schema, inspection command or new regression test has been executed by the implementer. The user's earlier successful safe call establishes compatibility of the previous profile, not correctness or sustained stability of these revised outputs. The safe image count, image dimensions, generation caps and sequential execution are preserved.

1. Load all five CSVs as strings, the Excel mapping and official JSON schema. Validate unique identifiers and input joins. Preserve test IDs exactly, including leading zeroes. Convert numeric Excel mapping IDs only at that mapping boundary.
2. Extract readable specification cells, labelled declarations, quantities and source nutrition. Retain partial tables around masks. Keep each fact's source, quote, destination, value, confidence and method. Raw source text remains available for the model to organise.
3. Retrieve exact, family and related candidates with brand/category constraints and RapidFuzz. Exclude validation targets and duplicates before retrieval. Transfer only exact or narrowly approved shared facts; retain related descriptions as context that cannot be cited as evidence.
4. OCR each unique image once. Apply EXIF orientation, bounded resizing, optional mild contrast, and rotation fallback. Preserve raw boxes, text, scores and rotation, plus accepted lines and combined blocks. Numbers are not spell-corrected.
5. Release OCR before loading one 4-bit Qwen3-VL-4B model through MLX-VLM. Prioritise sparse records with images. Adaptive inference targets missing copy, unextracted explicitly labelled attributes and images with insufficient OCR; rich source copy is retained when generation adds little. The all policy explicitly enables every model stage where applicable.
6. Verify quotes, source eligibility, field shapes, numbers, units, labels, confidence and conflicts. Omit disputed values and retain review reasons. Preserve ingredient declarations in order rather than mixing alternative lists. Accept prose sentence by sentence with source links and conservative lexical checks.
7. Save each completed record atomically. Submission release rechecks saved evidence, requires full test completion, joins to the sample ID order and validates the CSV before renaming the pending file.

The code uses separate packaging, structured and prose stages, plus a fresh visual read when high-risk label blocks need confirmation. Every stage fails independently. A failed or malformed model response leaves deterministic facts and supported source-description fallbacks available. Degraded records and stage failures are reported rather than hidden. Category context and stricter regulated-wording checks apply to medicines, supplements and baby health products.

## Files

```text
config.yaml                Paths, model, adaptive policy, thresholds and budgets
requirements.txt           Pinned direct dependencies and core MLX runtime
run.py                     Preflight, inference, benchmark, evaluation and release CLI
validate.py / evaluate.py   Convenience entry points
src/data.py                CSV and Excel inputs, masks and applicability
src/schema.py              Official format checks and strict nested validation
src/text_extraction.py     Deterministic quantities, tables, declarations and nutrition
src/image_processing.py    Image orientation and bounded preprocessing
src/ocr.py                 RapidOCR, rotations and reusable image cache
src/vlm.py                 Local MLX backend, strict JSON parsing and retry accounting
src/retrieval.py            Catalogue candidate selection and leakage exclusions
src/evidence.py             Source and fact provenance
src/enrichment.py           Adaptive vision, field extraction and prose generation
src/verification.py         Grounding, conflicts and claim checks
src/pipeline.py             Two-pass execution, resume and manifests
src/submission.py           Evidence audit, sample-order join and release validation
src/evaluation.py           Template, pseudo and ablation reporting
src/common.py               Configuration, hashes, cache, versions and budgets
prompts/                    Versioned extraction and generation instructions
tests/                      Unexecuted regression tests
outputs/                    Run artifacts and shared cache, ignored by Git
reports/                    Aggregate preflight and run pointers, ignored by Git
```

Everything generated stays inside this experiment. Models and original inputs are outside it. No root documentation, experiment ledger, datasets or other experiments are modified.

## Environment and model setup

All commands below are for manual execution from the repository root. Use the existing environment; no new virtual environment is required. Native arm64 Python 3.10–3.12 is required for this pinned stack. MLX-VLM 0.3.4 requires OpenCV 4.12+, whose wheel requires NumPy 2 on these Python versions. Those versions are pinned together. Transitive dependencies are resolved at installation; freeze the resolved environment for final reproduction.

```sh
source venv-dis-chem-product-enrichment/bin/activate
python -m pip install -r exp/001_competitive_multimodal/requirements.txt
python -m pip check
python -m pip freeze > exp/001_competitive_multimodal/reports/environment.lock.txt
df -h .
hf download mlx-community/Qwen3-VL-4B-Instruct-4bit --local-dir models/Qwen3-VL-4B-Instruct-4bit
```

For the final reproducible run, add `--revision` with the exact Hugging Face commit you selected to the download command. The runner hashes every downloaded checkpoint file, so replacement files invalidate run and model caches. Keep several GB of free disk for the model, package installation and evidence cache, beyond the configured 2 GB output headroom. The checkpoint download size and peak inference memory must be checked on your machine.

The [model card](https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit) identifies this as an Apache-2.0 MLX conversion made using MLX-VLM 0.3.4. The [tagged Qwen3-VL implementation](https://github.com/Blaizzy/mlx-vlm/tree/v0.3.4/mlx_vlm/models/qwen3_vl) exists in that runtime. The backend uses its documented `load`, `load_config`, `apply_chat_template` and `generate` interfaces. This is interface-level compatibility evidence, not a successful local test. No alternative backend is silently substituted. A fallback checkpoint must be compatible with this runtime and selected explicitly through `model.path` and `model.quantisation_bits`.

RapidOCR uses the Apache-2.0 `rapidocr-onnxruntime` 1.4.4 package and its packaged OCR assets. MLX-VLM is MIT and MLX is MIT. Check all installed package/model licences for your final write-up; the supplied catalogue licence remains CC-BY-SA 4.0. No paid API is used. The implementation is ordinary source modules called by `run.py`, not an installed custom package.

After download, run inference offline so that missing assets fail instead of being fetched:

```sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

## Paths and configuration

Paths in `config.yaml` are resolved relative to this experiment directory, independent of the shell working directory. Defaults read `../../dataset`, prefer `dataset/image_mapping.csv` when present and otherwise use `image_mapping.xlsx`. Blank filenames are ignored; duplicate entries are deduplicated and missing files reported. Nested paths are preserved under `dataset/product_images`; dataset-relative `product_images/...` filenames are also supported without allowing path escape. Preflight reports the selected mapping source, actual test products with/without existing images, deduplicated entries and unique files. It does not assume the expected 840/317 product split or 1,362 entries. The default model remains `../../models/Qwen3-VL-4B-Instruct-4bit`. Output columns and empty shapes come from the official schema.

Override existing keys with repeated `--set key=value`. Boolean and list values use YAML syntax. Paths to output/report directories must resolve within the experiment. Model batch size is deliberately restricted to one; increasing it fails explicitly. Image count, image resolution, output tokens, model path, quantisation, temperature, stage caps, thresholds and source policies are configurable.

```sh
python exp/001_competitive_multimodal/run.py preflight
python exp/001_competitive_multimodal/run.py preflight --check-processor
python -m pytest exp/001_competitive_multimodal/tests -q
```

Preflight reads all inputs, hashes supplied files/images, validates joins and verifies image readability. It writes aggregate counts, versions, disk headroom and missing-image counts to `reports/preflight.json`. It performs no OCR or VLM inference. It can take longer than a CSV-only check because the complete image collection is read. The tests use synthetic records plus the official local schema, never model inference.

`--check-processor` additionally imports PyTorch/Torchvision and loads the local checkpoint processor through MLX-VLM's processor loader. It exercises Qwen3-VL's AutoVideoProcessor dependency path without loading model weights or generating tokens. Failure is saved in `preflight.json` and exits with an error. The pinned Apple Silicon pair is `torch==2.8.0` and `torchvision==0.23.0`. They support processor initialisation; the actual language/vision model still runs through MLX. Their versions are included in run and model-cache fingerprints. A successful processor check does not prove that model-weight loading or generation works.

## First sample and labelled-example checks

### M2 Air safe smoke test

The competitive defaults remain in `config.yaml`. `profiles/m2_safe.yaml` inherits them and overrides only the local debugging controls: one image per call, 512 px maximum input side, bounded processor pixel area, a compact identity-only prompt, 256 generated tokens, sequential records and an 8 GiB MLX allocation limit. It disables structured/prose generation and the second visual read, leaving OCR, deterministic extraction and retrieval intact. Automatic call retries are disabled. This is a conservative debugging profile, not a full-quality submission configuration, and has not been tested or benchmarked.

Run one image-bearing record first. VLM use is forced by the profile. The selection is seeded; it excludes products with no mapped image so this command actually exercises vision:

```sh
python -u exp/001_competitive_multimodal/run.py infer \
  --config exp/001_competitive_multimodal/profiles/m2_safe.yaml \
  --split test --limit 1 --resume --retry-degraded
```

For a known product, add `--record-id RECORD_ID`. IDs are matched exactly, and the safe profile rejects an ID with no mapped image. After inspecting the one-record artifacts, expand to three:

```sh
python -u exp/001_competitive_multimodal/run.py infer \
  --config exp/001_competitive_multimodal/profiles/m2_safe.yaml \
  --split test --limit 3 --resume --retry-degraded
python exp/001_competitive_multimodal/run.py evaluate \
  --config exp/001_competitive_multimodal/profiles/m2_safe.yaml --split test
```

Each run writes `diagnostics/vlm.jsonl`. Every diagnostic line is flushed and fsynced. Events bracket prompt assembly, model loading/reuse, processor-limit setup, image loading, PIL preprocessing, chat templating, the native VLM call, parsing and cleanup. Metadata includes record/stage, profile, image sizes/count, token limits, call identity and manual retry scope. Python exceptions retain full tracebacks. Native Metal aborts may bypass Python exception handlers and cleanup; the final durable event identifies the last entered boundary. The library's tensor preprocessing, vision encoding, text prefill and decoding occur inside its `generate` call and are bracketed together, not individually timed. Text-token counting does not include the image tokens later expanded by the processor.

Capture native stderr as well if needed by appending `> exp/001_competitive_multimodal/reports/m2_safe_console.log 2>&1` to the inference command. This console file is separate from the durable JSONL diagnostics. Avoid rerunning with the same filename if you need to preserve an earlier console log.

After a native abort, the unfinished record has no completed record JSON; prepared evidence and OCR caches remain. `--retry-degraded` creates a recorded manual attempt scope so an interrupted request can be retried without deleting its attempt history. Automatic retries remain bounded by `budget.retry_count`; explicit manual retries are separate attempts, remain in diagnostics, and contribute to cumulative time/call/cost accounting. Cached successful calls are reused. For recovery from another fingerprint, supply `--run-dir PATH_TO_INTERRUPTED_RUN`; recovery also recognises status entries for records that never reached a result JSON. A supplied `--record-id` can restrict recovery to one product.

The safe pixel controls must be supported by the installed image processor; otherwise loading fails explicitly before generation. Lower resolution, shorter context and fewer tokens reduce workload but cannot guarantee avoidance of a Metal watchdog timeout. Native failures, processor compatibility and thermal behaviour still need manual verification. The regular competitive path below keeps the stronger extraction and generation stages.

```sh
python exp/001_competitive_multimodal/run.py infer --split test --limit 8
python exp/001_competitive_multimodal/run.py evaluate --split test
python exp/001_competitive_multimodal/run.py infer --split template
python exp/001_competitive_multimodal/evaluate.py --split template
```

Sampling is seeded and balanced across category, image availability and description availability. Eight rows are an integration check, not a useful timing estimate for every category. Review the sample's per-record evidence and rejected candidates. Inspect `qualitative_review.json` for template prose, missing fields and exact normalised name agreement. Do not tune model settings to maximise these 14 examples.

Test evaluation reports coverage, invalid-output rate, rejected candidate reasons and runtime without pretending that test labels exist. The template report includes structured lexical precision/recall/F1 and prose token diagnostics. Every missing reference value counts as FN; every unmatched prediction counts as FP locally, although the official judge may accept a harmless extra. Ingredient declaration segmentation and semantically equivalent wording can make lexical scores pessimistic.

## Benchmark before full inference

```sh
python exp/001_competitive_multimodal/run.py benchmark --split test --limit 48 --resume
python exp/001_competitive_multimodal/run.py evaluate --split test
```

`benchmark.json` records elapsed time, completed/model record counts, preparation/enrichment time, checkpoint/input initialisation time, process peak RSS, MLX peak allocation when available and a population extrapolation. Initialisation includes hashing all supplied images and checkpoint files. This fixed overhead is multiplied in the simple extrapolation, so read the stage totals as well. Existing caches make later measurements warmer; label them accordingly. The previous eight rows can be reused, and the budget remains cumulative for this run identity. No hard runtime-fit claim follows from the extrapolation.

The runtime and memory monitors check boundaries between stages/calls. They do not interrupt the native OCR/MLX kernel mid-call. RSS and MLX allocations overlap on unified memory and must not be added together. Close memory-heavy applications, connect power and inspect swap/thermal behaviour during the benchmark. If necessary, lower `model.max_images`, `model.image_max_side` or `model.max_tokens`, or reduce `policy.max_vlm_records` after reviewing which records lose useful evidence. A changed configuration creates a new run identity.

## Pseudo-masked validation

```sh
python exp/001_competitive_multimodal/run.py infer --split pseudo
python exp/001_competitive_multimodal/run.py evaluate --split pseudo
```

The pseudo split selects reference records reproducibly, then applies source-channel masking guided by test missingness. A blank/partially masked specification or nutrition channel is withheld as a whole. Descriptions are also removed when withholding a structured channel to avoid the same declaration surviving in parallel prose. This is deliberately coarser and often harder than the actual partial-mask distribution. The product name, classification and supplied images remain legitimate input evidence; a quantity already present in a product name is recoverable information, not a reconstructed secret label.

The complete original reference row is stored separately as `pseudo_gold.json` for evaluation only and never passed to extraction or prompts. All selected target IDs, full normalised identities, duplicate description signatures and matching image hashes are excluded from the retrieval pool. No reference template labels are used for pseudo inference. Disable images with `--set validation.pseudo_images=false` for a separate text-only pseudo task.

Pseudo targets are deterministic silver extractions from original records. They are not labelled enriched outputs. Their score checks supported reconstruction and regressions, not leaderboard accuracy. Per-field text-observability counts distinguish explicitly observable values from values absent from remaining record/OCR text. Values marked not text observable may still be visible in images; this is not a proof of impossibility. Category reports and qualitative review help find where the proxy fails.

## Ablations

Use the same seed and pseudo size for every comparison. These commands create distinct configuration identities inside this one experiment and share compatible OCR/model-call cache entries. Run each only when you intend to spend its compute budget.

```sh
python exp/001_competitive_multimodal/run.py infer --split pseudo --set policy.ocr=false
python exp/001_competitive_multimodal/run.py evaluate --split pseudo
python exp/001_competitive_multimodal/run.py infer --split pseudo --set policy.vlm=false
python exp/001_competitive_multimodal/run.py evaluate --split pseudo
python exp/001_competitive_multimodal/run.py infer --split pseudo --set policy.retrieval=false
python exp/001_competitive_multimodal/run.py evaluate --split pseudo
python exp/001_competitive_multimodal/run.py compare --runs PATH_TO_FULL_PSEUDO_RUN PATH_TO_NO_OCR_RUN PATH_TO_NO_VLM_RUN PATH_TO_NO_RETRIEVAL_RUN
```

Replace the four path arguments with the directories printed by their inference commands. `compare` requires the same expected ID population and writes `reports/ablation_comparison.json`. Evaluate immediately after each run or use an explicit `--run-dir`: latest pointers are overwritten by later runs of the same split. With VLM disabled, deterministic extraction, OCR parsing and supported source descriptions remain available. A product-name fallback is used only when useful description evidence is absent.

## Full inference, release and validation

```sh
python exp/001_competitive_multimodal/run.py infer --split test --resume
python exp/001_competitive_multimodal/run.py evaluate --split test
python exp/001_competitive_multimodal/run.py submission --split test
python exp/001_competitive_multimodal/validate.py --split test
```

Submission generation prints the exact CSV path. Validation without `--file` uses the latest run pointer. You can instead pass that path directly to `validate.py --file PATH`. The validator reports all detected CSV errors and returns a nonzero exit code on invalid data. It checks column order, row width, exact ID set, sample ordering, duplicate IDs, JSON, nested types, placeholders, finite numbers and blocked blank cells. The standalone CSV validator cannot prove evidence grounding without a run artifact; the release command performs that additional trace audit.

No command submits to Zindi. Full inference never automatically generates a submission. Submission release fails if any record is missing or failed, if inputs changed, or if saved structured/prose outputs fail their evidence checks. Unsupported fields can remain empty in a completed record. A configured OCR/model stage limit is an explicit policy choice and is logged; it is not a failed record.

## Artifacts and resume

Every run is under `outputs/runs/<split>/<fingerprint>/`. It contains `manifest.json`, `config.json`, `budget.json`, structured `events.jsonl`, per-record `status/`, prepared evidence in `prepared/`, completed outputs and provenance in `records/`, and evaluation/benchmark/release reports when requested. Raw/parsed model responses and full OCR results live in `outputs/cache/`. These contain catalogue material and must stay out of Git. Root-level `reports/latest_<split>.json` points to the latest finished invocation, even if some records failed.

Use the identical command plus `--resume` after interruption. Completed records and prepared OCR evidence are reused. Cache keys include input bytes, model files, prompt contents, preprocessing source, relevant configuration and package versions. Full run identities also include all implementation sources and input hashes. Changes make a new run; old artifacts are retained. A lock prevents simultaneous experiment writers. Cumulative budgets and call attempt ledgers survive process restarts. A failed cached call receives at most one retry in total, not one retry on every resume. Do not delete the attempt ledger to retry the same call again.

Degraded records retain deterministic results but are marked `degraded`, not `complete`. A full run with degraded records is not ready for submission. Each record separately reports `model_attempted`, `model_loaded`, `model_succeeded`, `model_failed` and actual generation-call count. `used_model` means a stage succeeded, not merely that the selection policy requested inference. Cached successful model output can be used without loading a model in that invocation; those two statuses intentionally differ.

Use `--retry-degraded --resume` to retry degraded records under an unchanged run identity. Their previous outputs move to `attempt_history/`; prepared evidence and OCR caches remain. After installing dependencies or editing code, fingerprints change. Recover the exact earlier cohort by supplying its directory explicitly:

```sh
python exp/001_competitive_multimodal/run.py infer --split test --limit 8 \
  --set policy.vlm_selection=all --retry-degraded \
  --run-dir exp/001_competitive_multimodal/outputs/runs/test/b15831c4e89a426cdcbd/
python exp/001_competitive_multimodal/run.py evaluate --split test
```

`--run-dir` here selects recovery input; the current code/environment writes to its current fingerprint directory and leaves the original run unchanged. The cohort is selected by the source record IDs, not resampled. Compatible prepared evidence is copied only after input/configuration/component/image checks. Compatible successful records must pass the current evidence audit before reuse. Existing OCR entries can be reused across a Torch-only environment change when OCR code, image bytes, settings and relevant dependency versions match. Changed extraction code causes cheap deterministic preparation to run again; changed OCR preprocessing or dependencies requires fresh OCR. Exhausted generation retry ledgers are not reset. The missing-processor failure occurred before generation and did not consume generation attempts.

`policy.max_calls_per_record` counts physical uncached VLM requests, including retries; its default eight permits four stages with one retry each. Stage token limits are under `model.stage_max_tokens`. Successful cached stages do not consume another request. Adaptive selection looks for missing copy with recoverable evidence, unextracted explicitly labelled fields, or images with no OCR. An existing description alone does not trigger inference. Use `--set policy.vlm_selection=all` for full inference. `policy.max_vlm_records` and `policy.max_ocr_images` remain operational controls. Cost estimates depend on `budget.local_cost_per_hour_usd`; the zero default is not an all-in cost claim.

## Evidence policy and known limitations

### Reviewing revised product content without inference

The scale-readiness revision recognises count suffixes such as `X 70`, `30's` and `pcs`, validates variant values before placement, and routes scent-labelled data to fragrance for home scents/perfumery. It strips trailing amounts before deriving a scent from a name. Supplied inline ingredient declarations and explicit storage directions under usage headings are extracted without generating missing content.

Existing OCR detections are expanded into individual spans with their original boxes and confidence. Storage columns can be reconstructed geometrically without rerunning OCR. A nutrition-basis amount is not treated as pack contents. Mapped panels with matching brand but insufficient formulation identity may supply an explicit, non-invented storage instruction; they cannot supply nutrition through that exception. Formulation disagreements quarantine image evidence. Numerical visual claims require OCR agreement on labels, values and units even when the model classifies them as ordinary claims. Uncertain net quantities remain review evidence.

An ambiguous observation can be discarded while independent, complete printed objects are retained. The original raw response and discard event are preserved. Replay also considers raw responses from failed parsing stages. A previously failed image stage is marked recovered only if current parsing and import succeed; unrelated failed stages and unrecoverable responses remain degraded. Missing mandatory identity fields or malformed structure are not fabricated.

### Full-population M2 configuration

`profiles/m2_safe.yaml` remains the forced-image debugging profile. `profiles/m2_full.yaml` inherits its one-image, 512 px, 256-token, sequential limits but includes products without images and selects VLM work adaptively. It keeps model-based field/prose generation disabled. It is a full-population baseline, not a claim of leaderboard quality or runtime compliance.

```sh
python exp/001_competitive_multimodal/run.py preflight --config exp/001_competitive_multimodal/profiles/m2_full.yaml
python exp/001_competitive_multimodal/run.py infer --config exp/001_competitive_multimodal/profiles/m2_full.yaml --split test --limit 30 --resume --retry-degraded
python exp/001_competitive_multimodal/run.py evaluate --config exp/001_competitive_multimodal/profiles/m2_full.yaml --split test
# After reviewing the sample, manually choose to process the complete population:
caffeinate -di python -u exp/001_competitive_multimodal/run.py infer --config exp/001_competitive_multimodal/profiles/m2_full.yaml --split test --resume --retry-degraded
python exp/001_competitive_multimodal/run.py evaluate --config exp/001_competitive_multimodal/profiles/m2_full.yaml --split test
python exp/001_competitive_multimodal/run.py submission --config exp/001_competitive_multimodal/profiles/m2_full.yaml --split test
python exp/001_competitive_multimodal/run.py validate --config exp/001_competitive_multimodal/profiles/m2_full.yaml --split test
```

Image mappings prefer the CSV in the dataset root, then the CSV in the configured image directory, then Excel. The actual mapping file is fingerprinted and checked at release. Derived OCR evidence retains the original image path for auditing. Existing run artifacts, raw OCR caches and model responses are preserved. A malformed Olli-type response can still require a new manually initiated call; replay preserves usable OCR fields but does not pretend the failed response was complete. Native Metal timeouts remain possible, and `caffeinate` only prevents sleep.

Run `re-enrich` against a saved run to apply current deterministic extraction, quantity rules, evidence routing and description composition. It rebuilds supplied-record facts from the unchanged inputs, preserves saved OCR and visual sources, and writes a new directory under `outputs/re_enriched/`. It does not load a model or rerun OCR. Historical run files remain unchanged.

```sh
python -m pytest exp/001_competitive_multimodal/tests -q
python exp/001_competitive_multimodal/run.py re-enrich \
  --config exp/001_competitive_multimodal/profiles/m2_safe.yaml \
  --run-dir PATH_TO_SAVED_RUN
python exp/001_competitive_multimodal/run.py evaluate \
  --config exp/001_competitive_multimodal/profiles/m2_safe.yaml --split test
```

The compact image prompt specifies the output contract without a filled example. Literal prompt placeholders are rejected before parsed-response caching; saved placeholder observations are quarantined during re-enrichment. New prompt contents and the visual validator implementation invalidate incompatible VLM caches. Raw responses remain available for inspection. Re-enrichment cannot recover printed facts omitted by the original model response; those require a new extraction call.

When original prepared evidence and raw model responses are available, re-enrichment now starts from pre-vision OCR and reparses those responses under current rules. This can remove a false historical identity decision and recover observations lost by the former JSON decoder without generation. `visual_replay` in each record reports replay status, the raw-response path, parsed output and any structural recovery. Missing raw/prepared artifacts are reported as unavailable; ambiguous responses produce degraded results. Prepared and raw-response hashes participate in the replay fingerprint. Inspection exports include `visual_replay.json` and reference the original inference diagnostics.

The JSON parser rejects duplicate object keys. The only supported recovery expands complete repeated `attribute`, `value`, `evidence` triples inside an image's observation array into separate objects. It preserves their order and text, logs the recovery and then applies the unchanged visual schema. Repeated printed values, identity keys, incomplete triples and non-finite constants are rejected. Missing content and uncertain numbers are never repaired.

Printed amounts are compared as strength, whole-pack quantities or individual components. Equivalent count wording does not create a mismatch, and components cannot replace a declared set quantity. An uncorroborated VLM quantity conflict stays in review rather than overriding supplied contents. A literal printed amount can be routed even when the model calls it a specification. Ordinary quantities can use exact OCR/VLM agreement with OCR confidence at least 0.90 and confirmed image identity; regulated categories retain the 0.95 requirement. No serving or measurement conversions are performed.

Generic claims and highlighted ingredients are descriptive evidence rather than automatic specifications. Slogans and repeated identity wording remain context only. Highlighted ingredients never become a complete ingredient declaration. Appearance descriptions use actual product colours alongside other useful evidence, reject inferred pearl/diamond materials, and keep packaging illustrations out of product copy. Multi-clause source sentences are separated before deduplication so repeated fit claims need not be copied in full.

Pack extraction distinguishes capsule/tablet strength from contents and preserves same-unit component expressions without calculating totals. Mixed-unit or otherwise ambiguous components remain unpopulated. Explicit name sizes, assorted designations, fragrance names after candle/diffuser identifiers, and supported anatomical application areas have deterministic routing. Coded OCR shades require agreement with the supplied name. Reliable OCR spans can recover labelled quantities and printed styling features without relying on a whole panel's average confidence.

Description fallbacks preserve identifying sizes and amounts, use bounded renderings of supplied skin-type/finish specifications, and prefer actual product features over packaging appearance. Source marketing phrases, section headings and repeated claims are filtered. Fragrance-note wording stays descriptive and never becomes an ingredient declaration. Supporting explanations of visual observations are not independently promoted into customer-facing claims. These rules are conservative heuristics, not a semantic correctness guarantee.

The M2-safe profile still uses one image, a 512 px maximum side, 256 output tokens, sequential records and no model-based field/prose generation. Full competitive settings remain unchanged and have not been made safe against the reported Metal watchdog timeout by this content revision. Review saved-evidence changes first, then manually test the revised image prompt on one product before repeating a small cohort.

Uncertain VLM identity does not revoke OCR evidence supplied through the record's image mapping. Clear printed brand, declared variant or quantity contradictions quarantine that image's evidence and are logged separately. A missing front-facing name on a secondary image remains uncertainty, not proof of another product. Sparse records receive name-based copy without catalogue-category filler. Speculative promises are removed and logged; source claims requiring review are excluded from automatic fallback copy. Generated prose citing record descriptions is also filtered for those unreviewed claims. Structured lists are deduplicated by normalised value while preserving the first declared spelling; ingredient declaration order remains intact. A redundant pack-size entry in specifications.other is removed when the dedicated pack-size field contains the same value.

- Visual-evidence-v2 uses rule-based reliability rather than self-reported numerical confidence. Identity can be supported by visible brand/product wording or matching OCR anchors from the same image; explicit contradictions quarantine the evidence. Printed facts retain exact quotes, and high-risk declarations require OCR corroboration. Unverifiable mapped images can support directly observable appearance, not commercial specifications. Legacy visual-response handling retains its older confidence gates for compatibility. Manually review medical, safety and numerical claims.
- OCR preserves per-orientation spatial rows and panel blocks. Block confidence is descriptive; factual verification uses the confidence of cited lines, or the exact nutrient row and header. Weak unrelated lines do not lower a reliable cell's confidence. Numeric nutrition extraction requires explicit units, an ordered basis header and an unambiguous row-to-column association. Ambiguous tables and inequalities remain omitted.
- Supported input descriptions are sentence-traced fallbacks for disabled, failed, incomplete or rejected generation. Obvious hype and dedicated structured sections are excluded from fallback copy. Useful source details omitted by generated long copy can be retained. Long copy receives a full-name opening and short copy has no bullets. This preserves source information without assuming every original claim is independently verified.
- Lexical quote and sentence checks are real filters, not proof of semantic entailment. A source can itself contain conflicting copy; a cited number can still belong to the wrong product claim. Review disagreements, medical wording and nutrient tables. Generated claims never become independent sources of evidence.
- Explicit per-100ml panels, inequalities such as `<0.1`, unlabelled axes and dimensions without units are retained as source/review evidence but cannot be faithfully represented in the schema's numeric/per-100g or labelled-axis slots. They are omitted rather than converted.
- OCR is not trained for every language, curved label or tiny nutrition panel. Rotation fallback is conditional and does not inspect all orientations once sufficient text is found. Raw OCR coordinates refer to the resized, rotated image. No inferred number correction is performed.
- The model uses bounded multi-image input. OCR processes every mapped image within its stage cap, but vision sees only the first configured number. Selecting informative panels is a candidate improvement after reviewing actual results.
- Prompts over `model.max_input_chars` fail their stage explicitly. Generation falls back to existing verified facts and source copy. JSON parsing accepts a single complete object inside wrappers but never repairs digits, missing values or competing objects. Failed stage requests retain their persistent retry allowance across restarts.
- Matching nutrition rows from ambiguous OCR tables remains conservative. Deterministic JSON nutrition has exact cell associations; model nutrition must demonstrate a matching cell/label/basis. No serving conversions are calculated.
- Model and OCR integrations, package resolution, tests, end-to-end runtime and inference quality have not been executed or verified locally. There is no measured evidence that this configuration beats 0.653 or fits three hours on an M2 Air.

## Troubleshooting

For a Metal command-buffer timeout, begin with the M2 safe profile above. Do not assume that successful CLI generation proves the larger experiment prompts or multi-image calls are safe. Read the final `diagnostics/vlm.jsonl` events and preserve console stderr. A missing `vlm_call_complete` after `vlm_call_start` localises the failure to the native generate boundary, but does not distinguish vision encoding, prefill and decoding. A processor-limit error is a Python integration failure and should remain a degraded result rather than a native crash.

If MLX fails to import, confirm Python is arm64 and the macOS/Metal requirements for the pinned MLX wheel are met. If the model architecture fails, verify the downloaded checkpoint is the MLX Qwen3-VL conversion and the environment contains MLX-VLM 0.3.4. Do not rename a Transformers checkpoint to make it appear compatible.

If installation reports NumPy/OpenCV conflicts, use this experiment's pinned requirements rather than the unpinned project requirements. Preserve a resolved lock after a successful install. OCR and model assets must already be present before offline inference; missing assets should fail.

If a run is incomplete, inspect `status/` and `events.jsonl` before resuming. Invalid JSON, oversized prompts, exhausted attempts, memory pressure and budget exhaustion have different remedies. Changing prompts, policy or checkpoint starts a new reproducible run while retaining compatible shared cache entries. Budget exhaustion cannot be bypassed by restarting the same run. If the benchmark already predicts a limit breach, revise the policy before spending the full test budget.
