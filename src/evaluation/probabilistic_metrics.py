"""
Probabilistic forecast evaluation metrics.

All metrics expect inputs in the **original scale** (e.g. MW).

Metrics implemented:
- **Pinball loss** (quantile loss) for individual quantiles
- **PICP** — Prediction Interval Coverage Probability
- **MPIW** — Mean Prediction Interval Width
- **Interval Score** — combined calibration + sharpness score
"""

from typing import Dict

import numpy as np


def pinball_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> float:
    """Compute pinball (quantile) loss.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted quantile values.
    quantile : float
        Quantile level in (0, 1).

    Returns
    -------
    float
        Mean pinball loss.
    """
    error = y_true - y_pred
    loss = np.where(error >= 0, quantile * error, (quantile - 1.0) * error)
    return float(np.mean(loss))


def picp(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Prediction Interval Coverage Probability.

    The fraction of true values that fall within [lower, upper].

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    lower, upper : np.ndarray
        Lower and upper bounds of the prediction interval.

    Returns
    -------
    float
        Coverage probability in [0, 1].
    """
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def mpiw(
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Mean Prediction Interval Width.

    Measures sharpness — narrower intervals are preferred as long as
    coverage remains adequate.

    Parameters
    ----------
    lower, upper : np.ndarray
        Lower and upper bounds of the prediction interval.

    Returns
    -------
    float
        Mean interval width.
    """
    return float(np.mean(upper - lower))


def interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Interval Score (Gneiting & Raftery, 2007).

    Combines calibration and sharpness into a single score.  Lower is
    better.

    For a ``(1 - alpha)`` prediction interval:
    - Penalises width of the interval.
    - Penalises under-coverage with factor ``2/alpha``.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    lower, upper : np.ndarray
        Lower and upper bounds of the prediction interval.
    alpha : float
        Nominal miscoverage rate.  For a 90 % PI, ``alpha = 0.1``.

    Returns
    -------
    float
        Mean interval score.
    """
    width = upper - lower
    penalty_lower = (2.0 / alpha) * np.maximum(lower - y_true, 0)
    penalty_upper = (2.0 / alpha) * np.maximum(y_true - upper, 0)
    score = width + penalty_lower + penalty_upper
    return float(np.mean(score))


def compute_all_probabilistic_metrics(
    y_true: np.ndarray,
    pred_p10: np.ndarray,
    pred_p50: np.ndarray,
    pred_p90: np.ndarray,
) -> Dict[str, float]:
    """Compute all probabilistic metrics for P10/P50/P90 forecasts.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values (original scale).
    pred_p10, pred_p50, pred_p90 : np.ndarray
        Predicted quantiles (original scale).

    Returns
    -------
    dict
        Dictionary with all probabilistic evaluation metrics.
    """
    yt = y_true.ravel()
    p10 = pred_p10.ravel()
    p50 = pred_p50.ravel()
    p90 = pred_p90.ravel()

    return {
        "pinball_p10": pinball_score(yt, p10, 0.10),
        "pinball_p50": pinball_score(yt, p50, 0.50),
        "pinball_p90": pinball_score(yt, p90, 0.90),
        "pinball_mean": (
            pinball_score(yt, p10, 0.10)
            + pinball_score(yt, p50, 0.50)
            + pinball_score(yt, p90, 0.90)
        ) / 3.0,
        "PICP_90": picp(yt, p10, p90),
        "MPIW_90": mpiw(p10, p90),
        "interval_score_90": interval_score(yt, p10, p90, alpha=0.10),
    }
