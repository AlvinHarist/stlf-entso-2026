"""
PatchTST for deterministic multi-step forecasting.

Reference
---------
Nie et al. (2023) "A Time Series is Worth 64 Words: Long-term Forecasting with
Transformers" (ICLR 2023).

Architecture
------------
    Input (batch, lookback, n_features)
        ↓  [channel-independent: process each feature independently]
    Patching: divide each channel into non-overlapping (or strided) patches
        → (batch * n_features, n_patches, patch_len)
        ↓
    Linear patch embedding  →  d_model
        ↓
    Learned positional embedding
        ↓
    N × TransformerEncoderBlock
        ↓
    Flatten  →  (batch * n_features, n_patches * d_model)
        ↓
    Dense(horizon)  →  (batch * n_features, horizon)
        ↓
    Aggregate across features  →  (batch, horizon)
        [Mean of per-channel forecasts]
        ↓
    Output (batch, 24)

Key Design Principles
---------------------
1. Channel-Independence: each feature channel is processed independently by
   the same Transformer. This avoids cross-channel attention leakage and
   improves generalization.
2. Patching converts continuous time-steps into "tokens", allowing the
   Transformer to see coarser temporal patterns without quadratic cost.
3. The final forecast is the MEAN of per-channel forecasts — this is a
   simple aggregation strategy that works well empirically and maintains
   the fair comparison constraint (no channel gets privileged treatment).

Patch Length Adaptation
-----------------------
The standard PatchTST uses patch_len=16, stride=8 for 96-step inputs.
For the SHORT sequences in E0–E4 (lookback=24), we use:
    patch_len = 4, stride = 2   →  n_patches = (24 - 4) // 2 + 1 = 11
For the LONGER sequence in E5 (lookback=168):
    patch_len = 12, stride = 8  →  n_patches = (168 - 12) // 8 + 1 = 20

Hyperparameters (baseline)
--------------------------
    patch_len        : 4  (lookback=24) / 12 (lookback=168)
    stride           : 2  (lookback=24) / 8  (lookback=168)
    d_model          : 64
    n_heads          : 4
    n_layers         : 2
    dff              : 128
    dropout          : 0.1
    learning_rate    : 0.001
"""

import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


# ──────────────────────────────────────────────────────────────────────
# Transformer Encoder Block (reused from transformer.py, self-contained)
# ──────────────────────────────────────────────────────────────────────

class _TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, n_heads, dff, dropout, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout
        )
        self.ffn1 = layers.Dense(dff, activation="relu")
        self.ffn2 = layers.Dense(d_model)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.mha(x, x, training=training)
        attn_out = self.drop1(attn_out, training=training)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn2(self.ffn1(x))
        ffn_out = self.drop2(ffn_out, training=training)
        return self.norm2(x + ffn_out)


# ──────────────────────────────────────────────────────────────────────
# PatchTST channel processing block (for a single channel)
# ──────────────────────────────────────────────────────────────────────

def _auto_patch_params(lookback: int) -> tuple:
    """Choose patch_len and stride automatically based on lookback."""
    if lookback <= 48:
        return 4, 2
    elif lookback <= 96:
        return 8, 4
    else:
        return 12, 8


# ──────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────

def build_patchtst(
    lookback: int,
    n_features: int,
    horizon: int,
    patch_len: int | None = None,
    stride: int | None = None,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    dff: int = 128,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
) -> models.Model:
    """Build and compile PatchTST for multi-step forecasting.

    Parameters
    ----------
    lookback : int
        Number of input time steps.
    n_features : int
        Number of input features per time step.
    horizon : int
        Number of output time steps (forecast horizon).
    patch_len : int or None
        Patch length.  If None, auto-selected based on lookback:
        4 for lookback ≤ 48, 8 for ≤ 96, 12 otherwise.
    stride : int or None
        Patch stride.  If None, auto-selected: patch_len // 2.
    d_model : int
        Embedding dimension per patch token.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of Transformer encoder blocks.
    dff : int
        Inner dimension of the FFN sublayer.
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
    Channel-independent design: each of the n_features channels is processed
    independently by the shared Transformer.  The per-channel forecasts are
    averaged to produce the final (batch, horizon) output.
    """
    if patch_len is None or stride is None:
        patch_len, stride = _auto_patch_params(lookback)

    # Calculate number of patches
    n_patches = (lookback - patch_len) // stride + 1
    if n_patches < 1:
        raise ValueError(
            f"patch_len={patch_len} and stride={stride} produce "
            f"n_patches={n_patches} < 1 for lookback={lookback}. "
            "Reduce patch_len or stride."
        )

    inp = layers.Input(shape=(lookback, n_features), name="input")

    # ── Channel-independent processing ────────────────────────────────
    # Transpose to (batch, n_features, lookback) for easier channel ops
    x = layers.Permute((2, 1), name="channel_first")(inp)  # (B, C, T)

    # Reshape to (B*C, T, 1) to process each channel separately
    B_dyn = tf.shape(inp)[0]
    x_bc = layers.Reshape((n_features, lookback))(x)   # (B, C, T)

    # We'll use a shared Dense for patch embedding
    # First, create patches via extracting strided windows
    # We do this with a Conv1D per channel applied on (B*C, T)

    # Merge batch and channel dims: (B*C, T)
    x_merged = layers.Reshape((n_features * lookback,), name="merge_bc")(
        layers.Permute((2, 1))(inp)
    )  # NOT what we want — let's do it properly with tf.reshape

    # Proper approach: use Lambda to reshape and process
    # Input: (B, T, C) → process as (B*C, T, 1)

    def extract_patches(t):
        # t: (B, T, C)
        B_ = tf.shape(t)[0]
        T_ = lookback
        C_ = n_features
        # → (B*C, T)
        t_perm = tf.transpose(t, [0, 2, 1])        # (B, C, T)
        t_flat = tf.reshape(t_perm, (B_ * C_, T_))  # (B*C, T)
        # Extract patches: sliding window with given stride
        patches = tf.signal.frame(t_flat, frame_length=patch_len,
                                  frame_step=stride)  # (B*C, n_patches, patch_len)
        return patches

    patches = layers.Lambda(extract_patches, name="patching")(inp)
    # patches: (B*C, n_patches, patch_len)

    # Patch embedding: project each patch to d_model
    x = layers.Dense(d_model, name="patch_embedding")(patches)
    # x: (B*C, n_patches, d_model)

    # Learned positional embedding
    pos_embed = layers.Embedding(input_dim=n_patches, output_dim=d_model,
                                 name="pos_embedding")
    pos_idx = tf.range(n_patches)
    x = x + pos_embed(pos_idx)  # broadcast over batch

    if dropout > 0:
        x = layers.Dropout(dropout, name="input_dropout")(x)

    # Transformer encoder blocks (shared weights across channels)
    for i in range(n_layers):
        x = _TransformerEncoderBlock(
            d_model=d_model, n_heads=n_heads, dff=dff,
            dropout=dropout, name=f"encoder_block_{i}"
        )(x)
    # x: (B*C, n_patches, d_model)

    # Flatten patch dimension
    x = layers.Flatten(name="flatten")(x)
    # x: (B*C, n_patches * d_model)

    # Per-channel forecast
    x = layers.Dense(horizon, name="channel_output")(x)
    # x: (B*C, horizon)

    # Reshape back to (B, C, horizon) and average over channels
    def reshape_and_mean(t):
        B_ = tf.shape(inp)[0]
        t_bc = tf.reshape(t, (B_, n_features, horizon))   # (B, C, H)
        return tf.reduce_mean(t_bc, axis=1)               # (B, H)

    out = layers.Lambda(reshape_and_mean, name="channel_mean")(x)
    # out: (B, horizon)

    model = models.Model(inputs=inp, outputs=out, name="PatchTST")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model
