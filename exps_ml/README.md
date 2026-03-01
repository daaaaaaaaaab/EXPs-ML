# Exposure + Demographics ML Benchmark (Minimal, Reproducible)

This repository benchmarks several tree-based classifiers across multiple disease endpoints using:
- Disease-specific exposure pools (`configs/exposures_by_disease.csv`)
- Demographics covariates (one-hot encoded)
- LightGBM + SHAP ranking to select top exposure features
- SMOTE + StandardScaler inside each CV fold
- Grid search (once per disease, based on Exposure+Demo)

## Files
- `src/train_models.py` — main training script
- `configs/exposures_by_disease.csv` — mapping: `disease, exposure` (one exposure per row)
- `requirements.txt` — Python dependencies
- `imputed_data_sample.csv` — small synthetic example (100 rows) to verify the pipeline

## Data placement (real analysis)
Do **not** upload your real `imputed_data.csv` to GitHub. Instead:

1. Put your real dataset at the repository root:
   - `./imputed_data.csv`

2. Keep column names consistent with:
   - Disease columns: `Yes` / `No`
   - Exposures: must match `configs/exposures_by_disease.csv`
   - Demographics: script expects `age, gender, race, education, PIR, eGFR` by default

## Install
```bash
pip install -r requirements.txt
```

## Run (with sample)
```bash
python src/train_models.py --data imputed_data_sample.csv
```

## Run (with your real data)
```bash
python src/train_models.py --data imputed_data.csv
```

Outputs will be written to `outputs/`:
- `metrics_<disease>.csv`
- `metrics_all_diseases.csv`
- `summary_mean_metrics.csv`
- `auc_boxplot_<Model>.png`
