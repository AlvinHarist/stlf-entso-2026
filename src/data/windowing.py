"""
Sliding-window generation for supervised time-series forecasting.

Indexing Convention
-------------------
Given arrays ``X`` (features) and ``y`` (target), both of length *N*,
a single sample is constructed as:

    input  = X[i - lookback : i]      →  shape (lookback, n_features)
    output = y[i : i + horizon]       →  shape (horizon,)

where *i* ranges from ``lookback`` to ``N - horizon`` (inclusive start).

The **forecast origin** is at index ``i - 1`` (the last observed value in
the input window).  The model is asked to predict the *next* ``horizon``
values starting from index ``i``.

Leakage Prevention
------------------
Two factory functions are provided:

``create_train_windows``
    All target values must fall within the training period.

``create_evaluation_windows``
    The first window's *input context* may use the last ``lookback``
    observations from the preceding split (e.g. training data).  This is
    legitimate: using **historical observations** as input context is NOT
    leakage.  The key rule is that **target values** from one split must
    never appear as targets in another split.
"""

from typing import Tuple

import numpy as np


def create_windows(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int = 24,
    horizon: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window samples from aligned feature / target arrays.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (already scaled).
    y : np.ndarray, shape (n_samples,)
        Target vector (already scaled).
    lookback : int
        Number of past time steps used as model input.
    horizon : int
        Number of future time steps to predict.

    Returns
    -------
    X_win : np.ndarray, shape (n_windows, lookback, n_features)
    y_win : np.ndarray, shape (n_windows, horizon)
    """
    X_win, y_win = [], []
    n = len(X)

    for i in range(lookback, n - horizon + 1):
        X_win.append(X[i - lookback : i])   # input window
        y_win.append(y[i : i + horizon])     # target window

    return np.array(X_win), np.array(y_win)


def create_train_windows(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lookback: int = 24,
    horizon: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create windows for **training**.

    All input observations *and* all target values originate from the
    training set, ensuring zero leakage.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Scaled training features and targets.
    lookback, horizon : int
        Window sizes.

    Returns
    -------
    X_win, y_win : tuple of np.ndarray
    """
    return create_windows(X_train, y_train, lookback, horizon)


def create_evaluation_windows(
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    X_context: np.ndarray,
    y_context: np.ndarray,
    lookback: int = 24,
    horizon: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create windows for **validation or test** evaluation.

    The input context for the first few windows may include observations
    from the preceding split.  For example, when creating validation
    windows, the last ``lookback`` training observations provide the
    historical context needed to forecast the first validation targets.

    *** This is NOT leakage. ***
    Using *historical input observations* from a preceding period is the
    standard practice in rolling-origin evaluation.  Leakage would occur
    only if *future target values* were used as model inputs.

    Parameters
    ----------
    X_eval, y_eval : np.ndarray
        Scaled features and target for the evaluation split
        (validation or test).
    X_context, y_context : np.ndarray
        Scaled features and target from the *preceding* split,
        providing historical context.  Typically the last ``lookback``
        rows are sufficient, but more can be passed safely.
    lookback, horizon : int
        Window sizes.

    Returns
    -------
    X_win, y_win : tuple of np.ndarray

    Notes
    -----
    Internally, the context and evaluation arrays are concatenated along
    axis 0.  Windows are then created starting from the point where the
    first **target** value falls within the evaluation period.  This
    guarantees that:
      - All target values in ``y_win`` belong to the evaluation period.
      - Input windows may use observations from the context period.
    """
    # Concatenate context + evaluation data
    X_full = np.concatenate([X_context, X_eval], axis=0)
    y_full = np.concatenate([y_context, y_eval], axis=0)

    context_len = len(X_context)

    X_win, y_win = [], []
    n = len(X_full)

    for i in range(lookback, n - horizon + 1):
        # The target window spans [i, i+horizon).
        # We only keep windows whose *entire* target falls in the eval period.
        if i >= context_len:
            X_win.append(X_full[i - lookback : i])
            y_win.append(y_full[i : i + horizon])

    return np.array(X_win), np.array(y_win)
