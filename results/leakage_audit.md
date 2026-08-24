# Leakage Audit — STLF-ENTSO-2026

This document systematically examines every potential source of data
leakage in the reconstructed pipeline.

**Audit date:** 2026-08-24  
**Audited codebase:** `src/` modules + `experiments/` notebooks

---

## 1. Scaler / Transformer Fitting

| Check | Result | Leakage? | Prevention |
|-------|--------|----------|------------|
| `MinMaxScaler` for features | Fitted on `train_df` only in `fit_preprocessor()` | ✅ No | `feature_scaler.fit(train_copy[feature_cols])` — see `src/data/preprocessing.py` line where `fit()` is called exclusively on train data |
| `MinMaxScaler` for target | Fitted on `train_df` only, **separate** scaler | ✅ No | `target_scaler.fit(train_copy[[target_col]])` — dedicated scaler prevents cross-contamination with feature scaling |
| `PowerTransformer` (Yeo-Johnson) | Fitted on `train_df` only | ✅ No | `power_transformer.fit(train_df[skewed_cols])` |
| Scaler applied to val/test | Uses `transform()` only (never `fit()`) | ✅ No | `transform_data()` calls `.transform()`, never `.fit()` or `.fit_transform()` |

---

## 2. Chronological Splitting

| Check | Result | Leakage? | Prevention |
|-------|--------|----------|------------|
| Split order | TRAIN → VAL → TEST (chronological) | ✅ No | `chronological_split()` uses `iloc[:train_end]`, `iloc[train_end:val_end]`, `iloc[val_end:]` |
| Random shuffling of split | **Not used** | ✅ No | No `sklearn.model_selection.train_test_split()` or random sampling anywhere in `src/data/` |
| Temporal ordering preserved | Index sorted ascending before split | ✅ No | `load_dataset()` calls `df.sort_index()` |

---

## 3. Windowing / Target Leakage

| Check | Result | Leakage? | Prevention |
|-------|--------|----------|------------|
| Training windows | All inputs and targets from train period | ✅ No | `create_train_windows()` operates only on train arrays |
| Validation windows — targets | All targets from validation period | ✅ No | `create_evaluation_windows()` only keeps windows where `i >= context_len`, ensuring targets are from eval period |
| Validation windows — inputs | May use last `lookback` train observations as context | ✅ No | Using **historical observations** as input context is standard rolling-origin evaluation, NOT leakage |
| Test windows — targets | All targets from test period | ✅ No | Same mechanism as validation |
| Test windows — inputs | May use last `lookback` validation observations | ✅ No | Same reasoning — historical context is legitimate |
| Future target values as model input | Target column is included in features but only past values enter the input window | ✅ No | Window construction: `X[i-lookback:i]` (past), `y[i:i+horizon]` (future). No overlap. |

---

## 4. Training Protocol

| Check | Result | Leakage? | Prevention |
|-------|--------|----------|------------|
| Early stopping monitor | `val_loss` | ✅ No | `EarlyStopping(monitor="val_loss")` in `trainer.py` |
| Early stopping on test loss | **Not done** | ✅ No | Test data never passed to `model.fit()` |
| Training on validation targets | **Not done** | ✅ No | `model.fit(X_train, y_train, validation_data=(X_val, y_val))` — validation used only for monitoring |
| Training on test targets | **Not done** | ✅ No | Test data accessed only during final prediction |
| Shuffle during training | `shuffle=False` | ✅ No | Explicit `shuffle=False` in `train_model()` |
| Hyperparameter tuning on test | **Not done** | ✅ No | All hyperparameters fixed in `configs/baseline.yaml` before seeing test results |

---

## 5. Metric Computation

| Check | Result | Leakage? | Prevention |
|-------|--------|----------|------------|
| Metrics on scaled values | **Not done** (fixed from old pipeline) | ✅ No | `inverse_y()` is called before all metric functions |
| Metrics on original MW scale | ✅ Yes | ✅ No | `compute_all_metrics()` receives inverse-transformed arrays |
| Test metrics influence model selection | **Not done** | ✅ No | Model architecture and hyperparameters are fixed before test evaluation |

---

## 6. Feature Engineering

| Check | Result | Leakage? | Prevention |
|-------|--------|----------|------------|
| Target-derived features using future values | **None created** | ✅ No | No lag/lead features are engineered; raw columns used directly |
| Future weather variables as input | Weather features are **concurrent** (same timestamp as load) | ⚠️ Note | In a real deployment, weather at time *t* would need to be a forecast. For research with historical data, using observed weather is standard practice. This is documented but not considered leakage in the research context. |
| Accidental use of test-period statistics | **Not done** | ✅ No | No global statistics (mean, std) computed across full dataset |

---

## 7. Reproducibility

| Check | Result |
|-------|--------|
| Python random seed | ✅ Set via `set_seed(42)` |
| NumPy random seed | ✅ Set via `set_seed(42)` |
| TensorFlow random seed | ✅ Set via `set_seed(42)` |
| `PYTHONHASHSEED` | ✅ Set via `set_seed(42)` |
| GPU determinism | ⚠️ Not guaranteed (documented in `seed.py`) |

---

## Summary

| Category | Status |
|----------|--------|
| Preprocessing leakage | ✅ Clean |
| Temporal split integrity | ✅ Clean |
| Windowing leakage | ✅ Clean |
| Training protocol | ✅ Clean |
| Metric computation | ✅ Clean |
| Feature engineering | ✅ Clean (with weather note) |
| Reproducibility | ✅ Seeds set (GPU non-determinism documented) |

**Conclusion:** The reconstructed pipeline contains **no identified data leakage**.
All preprocessing is fitted exclusively on training data, all evaluation uses
properly inverse-transformed predictions, and no test information influences
model training or selection.
