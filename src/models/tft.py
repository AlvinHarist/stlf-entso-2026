"""
Simplified Temporal Fusion Transformer (TFT) for deterministic multi-step forecasting.

Reference
---------
Lim et al. (2021) "Temporal Fusion Transformers for Interpretable Multi-horizon
Time Series Forecasting" (International Journal of Forecasting).

Feature Mapping for This Dataset
---------------------------------
The original TFT distinguishes three input types:
  1. Static covariates    — entity-level, constant over time (e.g., region ID)
  2. Observed past inputs — only available up to the current time (e.g., load, weather)
  3. Known future inputs  — available for future horizons  (e.g., time-of-day)

Mapping to the ENTSO-E Austria STLF dataset:

    STATIC:          None (single-region dataset — no entity embedding needed)

    OBSERVED PAST:   ALL feature columns used by the experiment:
                     • load (target)
                     • weather: temperature_2m, rain, relative_humidity_2m,
                                sunshine_duration
                     • lags: lag_24, lag_168
                     NOTE: weather is OBSERVED, not forecast. Using it as
                     known-future would constitute leakage.

    KNOWN FUTURE:    Temporal cyclical features ONLY:
                     • hour_sin, hour_cos, dow_sin, dow_cos,
                       month_sin, month_cos
                     These are deterministic functions of the calendar and are
                     LEGITIMATELY KNOWN for any future horizon.
                     Available only in experiments E2, E4, E5.
                     In E0, E1, E3 (no temporal features in feature_cols),
                     the known_future pathway is skipped and the TFT
                     degrades to a simpler LSTM-attention architecture.

⚠️  DATA LEAKAGE WARNING documented:
    Do NOT provide observed weather as known_future. The dataset does not
    contain actual weather forecasts — only reanalysis/observed data.

Architecture (simplified TFT for this framework)
-------------------------------------------------
    ┌──────────────────────────────────────────────────┐
    │  Past encoder (Observed + Known-past)            │
    │    → Stacked LSTM (n_lstm_layers)                │
    │    → Gate & Norm                                 │
    ├──────────────────────────────────────────────────┤
    │  Future encoder (Known-future only)              │
    │    → Dense embedding per future step             │
    │    → Gate & Norm                                 │
    ├──────────────────────────────────────────────────┤
    │  Temporal Self-Attention (on past encoder output)│
    │    → MultiHeadAttention → Add+Norm               │
    ├──────────────────────────────────────────────────┤
    │  Position-wise FFN → Add+Norm                    │
    └──────────────────────────────────────────────────┘
    → Global Avg Pool → Dense(horizon)
    Output (batch, horizon)

This is a SIMPLIFIED TFT: it preserves the key elements (LSTM encoder,
gating, self-attention) but omits variable selection networks and
static covariate conditioning for clarity and fair comparison.

Hyperparameters (baseline)
--------------------------
    hidden_dim       : 64
    n_heads          : 4
    n_lstm_layers    : 2
    dropout          : 0.1
    learning_rate    : 0.001
"""

from typing import List

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Temporal cyclical feature names (known-future when calendar is given)
TFT_KNOWN_FUTURE_COLS = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
]


# ──────────────────────────────────────────────────────────────────────
# Gated Residual Network (GRN)
# ──────────────────────────────────────────────────────────────────────

class GatedResidualNetwork(layers.Layer):
    """GRN: core building block of TFT.

    Applies two-layer FC with ELU activation, gating, and residual connection.
    """

    def __init__(self, hidden_dim: int, output_dim: int,
                 dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(hidden_dim, activation="elu")
        self.fc2 = layers.Dense(output_dim)
        self.gate = layers.Dense(output_dim, activation="sigmoid")
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(dropout)
        self.skip = layers.Dense(output_dim, use_bias=False)

    def call(self, x, training=False):
        residual = self.skip(x)
        h = self.fc1(x)
        h = self.drop(h, training=training)
        h = self.fc2(h)
        g = self.gate(h)
        return self.norm(residual + g * h)


# ──────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────

def build_tft(
    lookback: int,
    n_features: int,
    horizon: int,
    feature_cols: List[str],
    hidden_dim: int = 64,
    n_heads: int = 4,
    n_lstm_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
) -> models.Model:
    """Build and compile a Simplified TFT for multi-step forecasting.

    Parameters
    ----------
    lookback : int
        Number of input time steps (past sequence length).
    n_features : int
        Number of input features per time step.
    horizon : int
        Number of output time steps (forecast horizon).
    feature_cols : list of str
        Column names of the features in the experiment.  Used to identify
        which subset are known-future temporal features.
    hidden_dim : int
        Hidden dimension throughout the model.
    n_heads : int
        Number of attention heads.
    n_lstm_layers : int
        Number of stacked LSTM layers in the past encoder.
    dropout : float
        Dropout rate.
    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model with MSE loss and Adam optimizer.

    Notes
    -----
    Known-future features are identified by their column name appearing in
    TFT_KNOWN_FUTURE_COLS.  Their indices in feature_cols are extracted and
    used to slice the input tensor.  If no temporal features are present
    (e.g., E0, E1, E3), the known-future pathway is skipped.
    """
    # Identify indices of known-future columns
    future_idx = [
        i for i, col in enumerate(feature_cols)
        if col in TFT_KNOWN_FUTURE_COLS
    ]
    n_future = len(future_idx)
    n_past = n_features  # all features are observed-past

    has_future = n_future > 0

    inp = layers.Input(shape=(lookback, n_features), name="input")

    # ── Past encoder: Stacked LSTM ─────────────────────────────────────
    x_past = inp
    for i in range(n_lstm_layers):
        return_seq = True  # always return sequences
        x_past = layers.LSTM(
            hidden_dim,
            return_sequences=return_seq,
            name=f"past_lstm_{i}",
        )(x_past)
        if dropout > 0:
            x_past = layers.Dropout(dropout, name=f"past_lstm_drop_{i}")(x_past)

    # GRN gating on past encoder output
    x_past = GatedResidualNetwork(
        hidden_dim=hidden_dim, output_dim=hidden_dim,
        dropout=dropout, name="past_grn"
    )(x_past)
    # x_past: (batch, lookback, hidden_dim)

    # ── Known-future encoder ──────────────────────────────────────────
    if has_future:
        # Extract known-future channels from input
        future_indices_tensor = tf.constant(future_idx, dtype=tf.int32)

        def extract_future(t):
            return tf.gather(t, future_indices_tensor, axis=2)

        x_future_raw = layers.Lambda(
            extract_future, name="extract_future_features"
        )(inp)
        # x_future_raw: (batch, lookback, n_future)

        # Embed future features to hidden_dim
        x_future = layers.Dense(hidden_dim, activation="elu",
                                 name="future_embed")(x_future_raw)
        x_future = GatedResidualNetwork(
            hidden_dim=hidden_dim, output_dim=hidden_dim,
            dropout=dropout, name="future_grn"
        )(x_future)
        # x_future: (batch, lookback, hidden_dim)

        # Combine past + future via addition
        x_combined = layers.Add(name="past_future_add")([x_past, x_future])
    else:
        x_combined = x_past

    # ── Temporal Self-Attention ────────────────────────────────────────
    attn_out = layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=hidden_dim // n_heads,
        dropout=dropout,
        name="temporal_self_attn",
    )(x_combined, x_combined)
    if dropout > 0:
        attn_out = layers.Dropout(dropout, name="attn_drop")(attn_out)
    x = layers.LayerNormalization(epsilon=1e-6, name="attn_norm")(
        x_combined + attn_out
    )

    # ── Position-wise FFN ─────────────────────────────────────────────
    ffn_out = layers.Dense(hidden_dim * 2, activation="relu", name="ffn_dense1")(x)
    ffn_out = layers.Dense(hidden_dim, name="ffn_dense2")(ffn_out)
    if dropout > 0:
        ffn_out = layers.Dropout(dropout, name="ffn_drop")(ffn_out)
    x = layers.LayerNormalization(epsilon=1e-6, name="ffn_norm")(x + ffn_out)

    # ── Output head ───────────────────────────────────────────────────
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    out = layers.Dense(horizon, name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="TFT")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model
