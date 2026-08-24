"""
Probabilistic (quantile) BiLSTM model.

Outputs simultaneous predictions for multiple quantiles using the
multi-quantile **pinball loss** (also known as quantile loss).

Architecture
------------
    Input (batch, lookback, n_features)
        ↓
    Bidirectional(LSTM(units))
        ↓
    Dropout(dropout)
        ↓
    Dense(horizon × n_quantiles)
        ↓
    Reshape(horizon, n_quantiles)  →  Output (batch, horizon, n_quantiles)

Default quantiles: 0.10, 0.50, 0.90  →  P10, P50, P90.
"""

from typing import List

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


# ──────────────────────────────────────────────────────────────────────
# Pinball (quantile) loss
# ──────────────────────────────────────────────────────────────────────

def multi_quantile_pinball_loss(quantiles: List[float]):
    """Create a combined pinball loss for multiple quantiles.

    The loss is the **sum** of individual pinball losses across all
    requested quantiles.

    Parameters
    ----------
    quantiles : list of float
        Quantile levels, e.g. ``[0.1, 0.5, 0.9]``.

    Returns
    -------
    callable
        A Keras-compatible loss function with signature ``(y_true, y_pred)``.
        ``y_pred`` is expected to have shape ``(batch, horizon, n_quantiles)``.
        ``y_true`` is broadcast from ``(batch, horizon)`` to match.
    """
    quantiles_tf = tf.constant(quantiles, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        # y_true: (batch, horizon) → expand to (batch, horizon, 1)
        y_true = tf.expand_dims(y_true, axis=-1)

        error = y_true - y_pred   # (batch, horizon, n_quantiles)
        loss = tf.maximum(quantiles_tf * error,
                          (quantiles_tf - 1.0) * error)
        return tf.reduce_mean(loss)

    loss_fn.__name__ = "multi_quantile_pinball_loss"
    return loss_fn


# ──────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────

def build_quantile_bilstm(
    lookback: int,
    n_features: int,
    horizon: int,
    quantiles: List[float] | None = None,
    units: int = 64,
    dropout: float = 0.0,
    learning_rate: float = 0.001,
) -> models.Model:
    """Build a BiLSTM that outputs quantile forecasts.

    Parameters
    ----------
    lookback : int
        Number of input time steps.
    n_features : int
        Number of input features per time step.
    horizon : int
        Number of future time steps to predict.
    quantiles : list of float or None
        Quantile levels.  Defaults to ``[0.1, 0.5, 0.9]``.
    units : int
        Number of LSTM hidden units (per direction).
    dropout : float
        Dropout rate after the BiLSTM layer.
    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled model.  Output shape: ``(batch, horizon, n_quantiles)``.
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    n_q = len(quantiles)

    inp = layers.Input(shape=(lookback, n_features), name="input")
    x = layers.Bidirectional(
        layers.LSTM(units, return_sequences=False),
        name="bilstm",
    )(inp)
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout")(x)
    x = layers.Dense(horizon * n_q, name="dense_flat")(x)
    out = layers.Reshape((horizon, n_q), name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="QuantileBiLSTM")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=multi_quantile_pinball_loss(quantiles),
    )
    return model
