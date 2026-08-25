"""
Vanilla Transformer encoder for deterministic multi-step forecasting.

Architecture
------------
    Input (batch, lookback, n_features)
        ↓
    Linear projection  →  d_model
        ↓
    Sinusoidal positional encoding
        ↓
    N × TransformerEncoderBlock
        (MultiHeadAttention → Add+Norm → FFN → Add+Norm)
        ↓
    Global Average Pooling  (batch, d_model)
        ↓
    Dense(horizon)  →  Output (batch, horizon)

Design Notes
------------
- Encoder-only (no decoder): the sequence representation is pooled and
  then projected directly to the 24-step forecast horizon.
- Sinusoidal positional encoding matches the sequence length at build time.
- Dropout is applied inside attention and FFN sub-layers.
- All hyperparameters are documented explicitly for fair comparison.

Hyperparameters (baseline)
--------------------------
    d_model          : 64
    n_heads          : 4
    n_layers         : 2
    dff              : 128  (feedforward inner dimension)
    dropout          : 0.1
    learning_rate    : 0.001
"""

import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


# ──────────────────────────────────────────────────────────────────────
# Positional Encoding
# ──────────────────────────────────────────────────────────────────────

def _sinusoidal_positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    """Pre-compute sinusoidal positional encoding.

    Parameters
    ----------
    seq_len : int
        Length of the input sequence.
    d_model : int
        Model dimension (must be even).

    Returns
    -------
    tf.Tensor, shape (1, seq_len, d_model)
    """
    positions = np.arange(seq_len)[:, None]          # (seq_len, 1)
    dims = np.arange(d_model)[None, :]               # (1, d_model)
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[None, :, :], tf.float32)   # (1, seq_len, d_model)


# ──────────────────────────────────────────────────────────────────────
# Transformer Encoder Block
# ──────────────────────────────────────────────────────────────────────

class TransformerEncoderBlock(layers.Layer):
    """Single Transformer encoder block: MHA + FFN with residuals."""

    def __init__(self, d_model: int, n_heads: int, dff: int,
                 dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout
        )
        self.ffn_dense1 = layers.Dense(dff, activation="relu")
        self.ffn_dense2 = layers.Dense(d_model)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        # Multi-head self-attention
        attn_out = self.mha(x, x, training=training)
        attn_out = self.dropout1(attn_out, training=training)
        x = self.norm1(x + attn_out)

        # Feed-forward network
        ffn_out = self.ffn_dense1(x)
        ffn_out = self.ffn_dense2(ffn_out)
        ffn_out = self.dropout2(ffn_out, training=training)
        x = self.norm2(x + ffn_out)
        return x


# ──────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────

def build_transformer(
    lookback: int,
    n_features: int,
    horizon: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    dff: int = 128,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
) -> models.Model:
    """Build and compile a Transformer encoder for multi-step forecasting.

    Parameters
    ----------
    lookback : int
        Number of input time steps.
    n_features : int
        Number of input features per time step.
    horizon : int
        Number of output time steps (forecast horizon).
    d_model : int
        Model/embedding dimension.
    n_heads : int
        Number of attention heads.  Must divide d_model evenly.
    n_layers : int
        Number of Transformer encoder blocks.
    dff : int
        Inner dimension of the feed-forward sublayer.
    dropout : float
        Dropout rate applied inside attention and FFN.
    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model with MSE loss and Adam optimizer.
    """
    # Pre-compute positional encoding (constant tensor)
    pos_enc = _sinusoidal_positional_encoding(lookback, d_model)

    inp = layers.Input(shape=(lookback, n_features), name="input")

    # Project to d_model
    x = layers.Dense(d_model, name="input_projection")(inp)

    # Add positional encoding (broadcast over batch)
    x = x + pos_enc

    if dropout > 0:
        x = layers.Dropout(dropout, name="input_dropout")(x)

    # Stack of encoder blocks
    for i in range(n_layers):
        x = TransformerEncoderBlock(
            d_model=d_model, n_heads=n_heads, dff=dff,
            dropout=dropout, name=f"encoder_block_{i}"
        )(x)

    # Pool over time dimension
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # Output projection
    out = layers.Dense(horizon, name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="Transformer")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model
