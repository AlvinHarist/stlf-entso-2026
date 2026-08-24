"""
Point forecast evaluation metrics.

All metrics expect inputs in the **original scale** (e.g. MW).  Apply
``inverse_y()`` to scaled predictions before calling these functions.

Provides:
- MAE, RMSE, MAPE, sMAPE  (aggregate over all samples and horizons)
- Horizon-wise metrics     (per-step h=1…H)
- Naive baseline computation  (persistence, daily, weekly)
- ``compute_all_metrics``   (convenience wrapper)
"""

from typing import Dict, List, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Scalar metrics
# ──────────────────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%).

    Entries where ``y_true == 0`` are excluded to avoid division by zero.
    """
    y_true, y_pred = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%).

    Uses the formula:  200 * |y - ŷ| / (|y| + |ŷ|).
    Entries where both y and ŷ are zero are excluded.
    """
    y_true, y_pred = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


# ──────────────────────────────────────────────────────────────────────
# Horizon-wise metrics
# ──────────────────────────────────────────────────────────────────────

def horizon_wise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, List[float]]:
    """Compute MAE, RMSE, MAPE, sMAPE for each forecast step h=1…H.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples, horizon)
    y_pred : np.ndarray, shape (n_samples, horizon)

    Returns
    -------
    dict
        Keys: ``mae``, ``rmse``, ``mape``, ``smape``.
        Each value is a list of length *horizon*.
    """
    H = y_true.shape[1]
    result = {"mae": [], "rmse": [], "mape": [], "smape": []}

    for h in range(H):
        yt = y_true[:, h]
        yp = y_pred[:, h]
        result["mae"].append(mae(yt, yp))
        result["rmse"].append(rmse(yt, yp))
        result["mape"].append(mape(yt, yp))
        result["smape"].append(smape(yt, yp))

    return result


# ──────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute all aggregate point-forecast metrics.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Can be 1-D or 2-D.  If 2-D, metrics are computed over the
        flattened arrays (all samples × all horizon steps).

    Returns
    -------
    dict
        ``{"MAE": …, "RMSE": …, "MAPE": …, "sMAPE": …}``
    """
    yt = y_true.ravel()
    yp = y_pred.ravel()
    return {
        "MAE": mae(yt, yp),
        "RMSE": rmse(yt, yp),
        "MAPE": mape(yt, yp),
        "sMAPE": smape(yt, yp),
    }


# ──────────────────────────────────────────────────────────────────────
# Naive baselines
# ──────────────────────────────────────────────────────────────────────

def compute_naive_baselines(
    y_test_actual: np.ndarray,
    full_series: np.ndarray,
    test_start_idx: int,
    horizon: int,
) -> Dict[str, Dict[str, float]]:
    """Compute naive baseline metrics on the same test windows.

    Three baselines:
    1. **Persistence** (last-value naive): ŷ(t+h) = y(t) for all h.
    2. **Daily naive**: ŷ(t+h) = y(t + h − 24).
    3. **Weekly naive**: ŷ(t+h) = y(t + h − 168).

    Parameters
    ----------
    y_test_actual : np.ndarray, shape (n_windows, horizon)
        Actual (inverse-transformed) test targets.
    full_series : np.ndarray, shape (N,)
        The complete target series in original scale (all splits
        concatenated), used to look up lagged values.
    test_start_idx : int
        Index into *full_series* where the test set begins.
    horizon : int
        Forecast horizon.

    Returns
    -------
    dict
        ``{"persistence": {metrics}, "daily_naive": {metrics}, "weekly_naive": {metrics}}``
    """
    n_windows = y_test_actual.shape[0]
    results = {}

    for name, lag in [("persistence", 0), ("daily_naive", 24), ("weekly_naive", 168)]:
        preds = []
        valid_actuals = []
        for w in range(n_windows):
            # The forecast origin for window w is at
            # full_series index = test_start_idx + w (the first target value).
            origin = test_start_idx + w

            if name == "persistence":
                # Repeat the last observed value for all horizon steps
                last_val = full_series[origin - 1]
                pred = np.full(horizon, last_val)
            else:
                # Use the value from *lag* hours before each target step
                pred = []
                for h in range(horizon):
                    target_idx = origin + h
                    lagged_idx = target_idx - lag
                    if lagged_idx >= 0:
                        pred.append(full_series[lagged_idx])
                    else:
                        pred.append(np.nan)
                pred = np.array(pred)

            if not np.any(np.isnan(pred)):
                preds.append(pred)
                valid_actuals.append(y_test_actual[w])

        if len(preds) > 0:
            preds = np.array(preds)
            valid_actuals = np.array(valid_actuals)
            results[name] = compute_all_metrics(valid_actuals, preds)
        else:
            results[name] = {"MAE": float("nan"), "RMSE": float("nan"),
                             "MAPE": float("nan"), "sMAPE": float("nan")}

    return results
