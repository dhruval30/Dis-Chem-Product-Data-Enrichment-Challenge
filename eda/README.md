# EDA

Exploratory analysis of the competition data.

- `run_eda.py` is the main entry point once the dataset is in place.
- `eda_report.md` holds the written findings.
- `figures/` holds plots and `csvs/` holds table outputs.

EDA must not fit competition models or create submissions. Keep `dataset/` read-only.

Run the reproducible analysis from the repository root:

```sh
venv-dis-chem-product-enrichment/bin/python eda/run_eda.py --seed 42 --out-dir eda
```

Dependencies are pinned in `requirements.txt`. Each run should record seed, package versions, input file sizes and modification times in a metadata file next to the outputs.
