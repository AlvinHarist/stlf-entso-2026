"""
Preprocessing pipeline with strict leakage prevention.

All transformers (Yeo-Johnson, MinMaxScaler) are fitted **exclusively on
the training set**.  The fitted transformers are then applied to
validation and test sets via ``transform_data()``.

A **separate scaler** is maintained for the target column so that
predictions can be inverse-transformed back to the original MW scale
before computing final evaluation metrics.

Chronological Split
-------------------
The dataset is split *strictly in temporal order*:

    TRAIN  →  VALIDATION  →  TEST

No shuffling.  No random sampling.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer


# ──────────────────────────────────────────────────────────────────────
# Chronological splitting
# ──────────────────────────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-series DataFrame chronologically.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by its DatetimeIndex (ascending).
    train_ratio : float
        Fraction of data for training (default 0.8).
    val_ratio : float
        Fraction of data for validation (default 0.1).
        Test ratio is ``1 - train_ratio - val_ratio``.

    Returns
    -------
    train_df, val_df, test_df : tuple of pd.DataFrame
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"[split] Train : {len(train_df):>6d} rows  "
          f"({train_df.index.min()} --> {train_df.index.max()})")
    print(f"[split] Val   : {len(val_df):>6d} rows  "
          f"({val_df.index.min()} --> {val_df.index.max()})")
    print(f"[split] Test  : {len(test_df):>6d} rows  "
          f"({test_df.index.min()} --> {test_df.index.max()})")

    return train_df, val_df, test_df


# ──────────────────────────────────────────────────────────────────────
# Preprocessor fitting  (TRAIN ONLY)
# ──────────────────────────────────────────────────────────────────────

def fit_preprocessor(
    train_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    skewed_cols: Optional[List[str]] = None,
    use_yeojohnson: bool = True,
) -> Dict[str, Any]:
    """Fit all preprocessing transformers on the *training set only*.

    Transformations applied (in order):
    1. **Yeo-Johnson** on skewed weather columns (optional).
    2. **MinMaxScaler** on all feature columns.
    3. **Separate MinMaxScaler** on the target column.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training portion of the dataset.
    target_col : str
        Name of the target column (e.g. ``AT_load_actual_entsoe_transparency``).
    feature_cols : list of str
        All feature column names to include (may or may not include target).
    skewed_cols : list of str or None
        Columns to apply Yeo-Johnson to.  If None and *use_yeojohnson* is
        True, defaults to ``['rain (mm)', 'sunshine_duration (s)']``.
    use_yeojohnson : bool
        Whether to apply Yeo-Johnson transformation.

    Returns
    -------
    dict
        Preprocessor dictionary with keys:
        - ``target_col``, ``feature_cols``, ``skewed_cols``
        - ``power_transformer`` (or None)
        - ``feature_scaler`` (MinMaxScaler for features)
        - ``target_scaler``  (MinMaxScaler for target only)
    """
    # Default skewed columns -------------------------------------------
    if skewed_cols is None and use_yeojohnson:
        skewed_cols = [
            c for c in ["rain (mm)", "sunshine_duration (s)"]
            if c in feature_cols
        ]

    # 1. Yeo-Johnson (fit on train) ------------------------------------
    power_transformer = None
    if use_yeojohnson and skewed_cols:
        power_transformer = PowerTransformer(method="yeo-johnson")
        power_transformer.fit(train_df[skewed_cols])

    # 2. Apply Yeo-Johnson to train copy so scaler sees transformed data
    train_copy = train_df.copy()
    if power_transformer is not None:
        train_copy[skewed_cols] = power_transformer.transform(
            train_copy[skewed_cols]
        )

    # 3. MinMaxScaler for features (fit on train) ----------------------
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train_copy[feature_cols].values)

    # 4. Separate MinMaxScaler for target (fit on train) ---------------
    target_scaler = MinMaxScaler()
    target_scaler.fit(train_copy[[target_col]].values)

    preprocessor = {
        "target_col": target_col,
        "feature_cols": feature_cols,
        "skewed_cols": skewed_cols if skewed_cols else [],
        "use_yeojohnson": use_yeojohnson,
        "power_transformer": power_transformer,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }

    print("[fit_preprocessor] Fitted on TRAIN data only.")
    if power_transformer is not None:
        print(f"  Yeo-Johnson applied to: {skewed_cols}")
    print(f"  Feature scaler fitted on {len(feature_cols)} columns.")
    print(f"  Target scaler fitted on: {target_col}")

    return preprocessor


# ──────────────────────────────────────────────────────────────────────
# Transformation
# ──────────────────────────────────────────────────────────────────────

def transform_data(
    df: pd.DataFrame,
    preprocessor: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the *already-fitted* preprocessor to any split.

    Parameters
    ----------
    df : pd.DataFrame
        Train, validation, or test DataFrame.
    preprocessor : dict
        Output of ``fit_preprocessor()``.

    Returns
    -------
    X_scaled : np.ndarray, shape (n_samples, n_features)
        Scaled feature matrix.
    y_scaled : np.ndarray, shape (n_samples,)
        Scaled target vector.
    """
    data = df.copy()
    feature_cols = preprocessor["feature_cols"]
    target_col = preprocessor["target_col"]
    skewed_cols = preprocessor["skewed_cols"]
    pt = preprocessor["power_transformer"]

    # 1. Yeo-Johnson ---------------------------------------------------
    if pt is not None and skewed_cols:
        data[skewed_cols] = pt.transform(data[skewed_cols])

    # 2. Scale features ------------------------------------------------
    X_scaled = preprocessor["feature_scaler"].transform(
        data[feature_cols].values
    )

    # 3. Scale target --------------------------------------------------
    y_scaled = preprocessor["target_scaler"].transform(
        data[[target_col]].values
    ).ravel()

    return X_scaled, y_scaled


# ──────────────────────────────────────────────────────────────────────
# Inverse transformation  (target only)
# ──────────────────────────────────────────────────────────────────────

def inverse_y(
    scaled_y: np.ndarray,
    preprocessor: Dict[str, Any],
) -> np.ndarray:
    """Inverse-transform scaled target values back to the original scale.

    This is **critical** for computing meaningful evaluation metrics
    (MAE in MW, MAPE in %, etc.).

    Parameters
    ----------
    scaled_y : np.ndarray
        Scaled target values.  Can be 1-D ``(n,)`` or 2-D ``(n, horizon)``.
    preprocessor : dict
        Output of ``fit_preprocessor()``.

    Returns
    -------
    np.ndarray
        Values in the original scale (e.g. MW).
    """
    scaler = preprocessor["target_scaler"]
    original_shape = scaled_y.shape

    # Scaler expects 2-D input
    flat = scaled_y.reshape(-1, 1)
    result = scaler.inverse_transform(flat).ravel()

    return result.reshape(original_shape)


# ──────────────────────────────────────────────────────────────────────
# Feature engineering  (leakage-safe)
# ──────────────────────────────────────────────────────────────────────

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical temporal features derived purely from the DatetimeIndex.

    No future information is used — each row's features depend only on
    its own timestamp.  The resulting values are already in [-1, 1], so
    MinMaxScaler will compress them only slightly.

    Columns added
    -------------
    hour_sin, hour_cos       sin/cos(2π·hour/24)
    dow_sin,  dow_cos        sin/cos(2π·day_of_week/7)
    month_sin, month_cos     sin/cos(2π·month/12)

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``pd.DatetimeIndex``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with six new columns appended.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("create_temporal_features requires a DatetimeIndex.")

    out = df.copy()
    idx = df.index

    out["hour_sin"]   = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"]   = np.cos(2 * np.pi * idx.hour / 24)
    out["dow_sin"]    = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"]    = np.cos(2 * np.pi * idx.dayofweek / 7)
    out["month_sin"]  = np.sin(2 * np.pi * idx.month / 12)
    out["month_cos"]  = np.cos(2 * np.pi * idx.month / 12)

    return out


def create_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add seasonal lag features for the target column.

    **Must be called on the FULL DataFrame BEFORE ``chronological_split``.**
    ``pandas.Series.shift(lag)`` is strictly backwards — it moves the
    series *forward in index* by ``lag`` steps, meaning
    ``lag_k[t] = target[t-k]``.  No future values are accessed.

    The first ``max(lags)`` rows will be NaN and are dropped before
    returning.  Callers should be aware that the resulting DataFrame
    starts at index ``max(lags)``, not 0.

    Columns added
    -------------
    ``lag_{k}`` for each ``k`` in *lags*, where
    ``lag_{k}[t] = target[t-k]``.

    Parameters
    ----------
    df : pd.DataFrame
        Full chronological dataset with a DatetimeIndex.
    target_col : str
        Name of the load column to lag.
    lags : list of int, optional
        Lag values in hours.  Defaults to ``[24, 168]`` (1-day, 1-week).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with lag columns added and NaN rows at the head
        removed.
    """
    if lags is None:
        lags = [24, 168]

    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)

    # Drop rows where any lag is NaN (at the start of the series)
    n_before = len(out)
    out = out.dropna(subset=[f"lag_{lag}" for lag in lags])
    n_dropped = n_before - len(out)
    print(f"[create_lag_features] Dropped {n_dropped} leading NaN rows "
          f"(max lag = {max(lags)}).  Remaining: {len(out)} rows.")

    return out


TEMPORAL_FEATURE_COLS = [
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
]

LAG_FEATURE_COLS_DEFAULT = ["lag_24", "lag_168"]

