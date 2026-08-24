# STLF-ENTSO-2026

**Short-Term Load Forecasting** for Austrian electricity demand using
ENTSO-E transparency data and Open-Meteo weather features.

A clean, reproducible, leakage-aware research pipeline for STLF
experiments with LSTM / BiLSTM models and probabilistic (quantile)
forecasting.

---

## Pipeline Overview

```
Raw Data  (df_combined_AT.csv)
   ↓
Chronological Split  (80% train / 10% val / 10% test)
   ↓
Train-only Preprocessing  (Yeo-Johnson + MinMaxScaler)
   ↓
Window Generation  (lookback=24h, horizon=24h)
   ↓
LSTM / BiLSTM  (deterministic or quantile output)
   ↓
Point / Quantile Forecast
   ↓
Inverse Transformation  (back to MW scale)
   ↓
Evaluation  (MAE, RMSE, MAPE, sMAPE, Pinball, PICP, MPIW)
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | ENTSO-E Transparency + Open-Meteo |
| Country | Austria (AT) |
| File | `df_combined_AT.csv` |
| Period | 2015-01-01 → 2020-09-30 |
| Frequency | Hourly |
| Rows | 50,400 |
| Target | `AT_load_actual_entsoe_transparency` (MW) |
| Weather | `temperature_2m (°C)`, `rain (mm)`, `relative_humidity_2m (%)`, `sunshine_duration (s)` |

---

## Project Structure

```
stlf-entso-2026/
│
├── data/
│   ├── raw/                        # Raw dataset placeholder
│   └── processed/                  # Intermediate data
│
├── src/
│   ├── data/
│   │   ├── load_data.py            # CSV/Parquet loading & validation
│   │   ├── preprocessing.py        # Chronological split, scalers
│   │   └── windowing.py            # Sliding-window generation
│   │
│   ├── models/
│   │   ├── lstm.py                 # Unidirectional LSTM
│   │   ├── bilstm.py              # Bidirectional LSTM
│   │   └── probabilistic.py       # Quantile BiLSTM (P10/P50/P90)
│   │
│   ├── training/
│   │   └── trainer.py              # Training with EarlyStopping
│   │
│   ├── evaluation/
│   │   ├── point_metrics.py        # MAE, RMSE, MAPE, sMAPE, baselines
│   │   └── probabilistic_metrics.py # Pinball, PICP, MPIW, IS
│   │
│   └── utils/
│       └── seed.py                 # Reproducibility seeds
│
├── experiments/
│   ├── 01_data_validation.ipynb    # Dataset quality report
│   ├── 02_deterministic_baseline.ipynb  # BiLSTM univariate baseline
│   ├── 03_feature_ablation.ipynb   # Univariate vs multivariate
│   ├── 04_horizon_analysis.ipynb   # 6h/12h/24h/48h/72h comparison
│   └── 05_probabilistic.ipynb      # Quantile forecasting
│
├── configs/
│   └── baseline.yaml               # All experiment parameters
│
├── results/
│   ├── deterministic/              # Point forecast outputs
│   ├── probabilistic/              # Quantile forecast outputs
│   └── leakage_audit.md            # Systematic leakage audit
│
└── README.md
```

---

## Methodology

### Data Split
Strictly chronological: **80% train → 10% validation → 10% test**.
No shuffling. No random sampling.

### Preprocessing
1. **Yeo-Johnson** transformation on skewed weather columns (fitted on train only).
2. **MinMaxScaler** on all features (fitted on train only).
3. **Separate MinMaxScaler** on the target column for proper inverse transformation.

### Windowing
- **Lookback:** 24 hours (input context)
- **Horizon:** 24 hours (forecast target)
- Evaluation windows may use the preceding split's last observations as input context (this is NOT leakage — see `results/leakage_audit.md`).

### Models
- **LSTM:** Standard unidirectional LSTM → Dense
- **BiLSTM:** Bidirectional LSTM → Dense
- **Quantile BiLSTM:** BiLSTM → Dense → Reshape to (horizon, 3) with pinball loss

### Training
- **EarlyStopping** on validation loss (patience=15)
- `shuffle=False` (temporal ordering preserved)
- `restore_best_weights=True`
- Maximum 200 epochs, batch size 32

### Evaluation
**Point metrics** (computed on original MW scale after inverse transform):
- MAE, RMSE, MAPE, sMAPE
- Horizon-wise breakdown (h=1…24)
- Naive baselines: persistence, daily (t-24), weekly (t-168)

**Probabilistic metrics:**
- Pinball loss (P10, P50, P90)
- PICP (90% nominal coverage)
- MPIW (interval sharpness)
- Interval Score

---

## Leakage Prevention

| Rule | Implementation |
|------|---------------|
| Preprocessors fitted on train only | `fit_preprocessor(train_df, ...)` |
| No training on val/test targets | `model.fit(X_train, y_train, validation_data=...)` |
| No test-based early stopping | `monitor="val_loss"` |
| No test-based hyperparameter tuning | All params fixed in `configs/baseline.yaml` |
| No random time-series shuffling | `chronological_split()` + `shuffle=False` |
| Metrics on original scale | `inverse_y()` before all metric functions |

See `results/leakage_audit.md` for the full audit.

---

## Experiment Order

1. **01_data_validation** — Verify dataset integrity
2. **02_deterministic_baseline** — Clean BiLSTM baseline + naive baselines
3. **03_feature_ablation** — Univariate vs. multivariate
4. **04_horizon_analysis** — Multi-horizon comparison
5. **05_probabilistic** — Quantile forecasting with calibration analysis

---

## Running on Kaggle

1. Clone the repository (auto-handled by notebook Cell 1).
2. Add `df_combined_AT.csv` as a Kaggle Dataset mounted at `/kaggle/input/stlf-entso-2026/`.
3. Run notebooks in order from `experiments/`.

All notebooks are designed for **Kernel → Restart & Run All**.

---

## Legacy Experiments

Original experimental notebooks (`experiment_pipeline.ipynb`,
`revised_experiment_pipeline.ipynb`, `experiment_pipeline_yeojohnson.ipynb`,
etc.) are preserved in the repository root as historical reference material.
They are **not** part of the reconstructed pipeline.
