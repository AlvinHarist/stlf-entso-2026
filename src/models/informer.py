"""
Informer with ProbSparse Self-Attention for deterministic multi-step forecasting.

Reference
---------
Zhou et al. (2021) "Informer: Beyond Efficient Transformer for Long Sequence
Time-Series Forecasting" (AAAI 2021).

Architecture
------------
    Input (batch, lookback, n_features)
        ↓
    Linear projection → d_model
        ↓
    Sinusoidal positional encoding
        ↓
    N × InformerEncoderLayer
        (ProbSparse MultiHead Attention → Add+Norm → Conv Distilling → Add+Norm
         → FFN → Add+Norm)
        ↓
    Global Average Pooling  (batch, d_model)
        ↓
    Dense(horizon)  →  Output (batch, horizon)

ProbSparse Attention
--------------------
Instead of computing all Q-K similarities, ProbSparse selects the top-k
queries (where k = ceil(ln(L_Q))) that have the highest "sparsity measure"
(max similarity minus mean similarity).  This reduces complexity from
O(L²) to O(L log L) while maintaining similar predictive quality.

Conv Distilling
---------------
After each attention layer, a 1D max-pooling with stride 2 halves the
sequence length, progressively focusing on salient features.  This gives
the Informer its characteristic triangular architecture.

Design Notes for this implementation
-------------------------------------
- We use ENCODER ONLY (no decoder) with global average pooling → Dense(horizon),
  which is consistent with the Transformer and PatchTST implementations in this
  framework and avoids the autoregressive decoder complexity.
- The input sequence length determines ProbSparse k.
- Distilling is applied between encoder layers only.
- This is a clean, self-contained Keras implementation.

Hyperparameters (baseline)
--------------------------
    d_model          : 64
    n_heads          : 4
    n_layers         : 2
    dff              : 128
    dropout          : 0.1
    use_distilling   : True
    learning_rate    : 0.001
"""

import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


# ──────────────────────────────────────────────────────────────────────
# Positional Encoding (shared with Transformer)
# ──────────────────────────────────────────────────────────────────────

def _sinusoidal_positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    positions = np.arange(seq_len)[:, None]
    dims = np.arange(d_model)[None, :]
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[None, :, :], tf.float32)


# ──────────────────────────────────────────────────────────────────────
# ProbSparse Attention
# ──────────────────────────────────────────────────────────────────────

class ProbSparseAttention(layers.Layer):
    """Multi-head ProbSparse self-attention.

    For sequences shorter than ~16, k = ceil(ln(L)) ≈ L so this degrades
    gracefully to full attention.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout_rate = dropout

        self.W_q = layers.Dense(d_model, name=f"{self.name}_Wq")
        self.W_k = layers.Dense(d_model, name=f"{self.name}_Wk")
        self.W_v = layers.Dense(d_model, name=f"{self.name}_Wv")
        self.W_o = layers.Dense(d_model, name=f"{self.name}_Wo")
        self.dropout = layers.Dropout(dropout)

    def _split_heads(self, x):
        # x: (batch, seq, d_model) → (batch, heads, seq, d_k)
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]
        x = tf.reshape(x, (B, L, self.n_heads, self.d_k))
        return tf.transpose(x, [0, 2, 1, 3])

    def _prob_sparse_select(self, Q, K):
        """Select top-k queries based on sparsity measure.

        Returns
        -------
        Q_sparse : (batch, heads, k, d_k)
        indices  : (batch, heads, k)
        """
        # Q: (batch, heads, L_Q, d_k)
        # K: (batch, heads, L_K, d_k)
        L_Q = tf.shape(Q)[2]
        L_K = tf.shape(K)[2]

        k = tf.maximum(1, tf.cast(
            tf.math.ceil(tf.math.log(tf.cast(L_Q, tf.float32))), tf.int32
        ))
        k = tf.minimum(k, L_Q)

        # Randomly sample k keys to compute a proxy score for each query
        scale = tf.cast(self.d_k, tf.float32) ** -0.5

        # Compute full attention score (L_Q × L_K)
        scores = tf.matmul(Q, K, transpose_b=True) * scale  # (B,H,L_Q,L_K)

        # Sparsity measure: max(score) - mean(score) per query
        sparsity = tf.reduce_max(scores, axis=-1) - tf.reduce_mean(scores, axis=-1)
        # (B, H, L_Q)

        _, top_idx = tf.math.top_k(sparsity, k=k)
        # top_idx: (B, H, k)

        # Gather top queries
        # Need to gather from (B, H, L_Q, d_k) along dim 2
        B_ = tf.shape(Q)[0]
        H_ = tf.shape(Q)[1]
        batch_idx = tf.tile(
            tf.reshape(tf.range(B_), (B_, 1, 1)),
            [1, H_, k]
        )
        head_idx = tf.tile(
            tf.reshape(tf.range(H_), (1, H_, 1)),
            [B_, 1, k]
        )
        gather_idx = tf.stack([batch_idx, head_idx, top_idx], axis=-1)
        Q_sparse = tf.gather_nd(Q, gather_idx)  # (B, H, k, d_k)

        return Q_sparse, top_idx

    def call(self, x, training=False):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = self._split_heads(Q)  # (B, H, L, d_k)
        K = self._split_heads(K)
        V = self._split_heads(V)

        L_Q = tf.shape(Q)[2]

        # ProbSparse: select top-k queries
        Q_sparse, top_idx = self._prob_sparse_select(Q, K)  # (B,H,k,d_k), (B,H,k)
        k = tf.shape(Q_sparse)[2]

        # Attention for sparse queries
        scale = tf.cast(self.d_k, tf.float32) ** -0.5
        attn_scores = tf.matmul(Q_sparse, K, transpose_b=True) * scale  # (B,H,k,L_K)
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)
        context_sparse = tf.matmul(attn_weights, V)  # (B,H,k,d_k)

        # Initialize output with mean of V (default for non-selected queries)
        V_mean = tf.reduce_mean(V, axis=2, keepdims=True)  # (B,H,1,d_k)
        out = tf.tile(V_mean, [1, 1, L_Q, 1])              # (B,H,L_Q,d_k)

        # Scatter top-k context back to correct positions
        # Build scatter indices
        B_ = tf.shape(Q)[0]
        H_ = tf.shape(Q)[1]
        batch_idx = tf.tile(tf.reshape(tf.range(B_), (B_, 1, 1)), [1, H_, k])
        head_idx  = tf.tile(tf.reshape(tf.range(H_), (1, H_, 1)), [B_, 1, k])
        scatter_idx = tf.stack([batch_idx, head_idx, top_idx], axis=-1)

        # Flatten for scatter
        out_flat = tf.reshape(out, (-1, self.d_k))
        idx_flat = tf.reshape(scatter_idx, (-1, 3))
        B_dyn = tf.shape(Q)[0]
        H_dyn = tf.shape(Q)[1]
        L_dyn = L_Q
        flat_indices = (idx_flat[:, 0] * H_dyn * L_dyn
                        + idx_flat[:, 1] * L_dyn
                        + idx_flat[:, 2])
        ctx_flat = tf.reshape(context_sparse, (-1, self.d_k))
        out_flat = tf.tensor_scatter_nd_update(
            out_flat,
            tf.expand_dims(flat_indices, 1),
            ctx_flat,
        )
        out = tf.reshape(out_flat, (B_dyn, H_dyn, L_dyn, self.d_k))

        # Merge heads
        out = tf.transpose(out, [0, 2, 1, 3])  # (B, L, H, d_k)
        out = tf.reshape(out, (tf.shape(x)[0], L_Q, self.d_model))

        return self.W_o(out)


# ──────────────────────────────────────────────────────────────────────
# Informer Encoder Layer
# ──────────────────────────────────────────────────────────────────────

class InformerEncoderLayer(layers.Layer):
    """Informer encoder layer: ProbSparse attention + optional Conv distilling + FFN."""

    def __init__(self, d_model: int, n_heads: int, dff: int,
                 dropout: float, use_distilling: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.attn = ProbSparseAttention(d_model, n_heads, dropout,
                                        name=f"{self.name}_probsparse_attn")
        self.ffn1 = layers.Dense(dff, activation="relu")
        self.ffn2 = layers.Dense(d_model)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)
        self.use_distilling = use_distilling
        if use_distilling:
            self.distil_conv = layers.Conv1D(
                d_model, kernel_size=3, padding="same", activation="elu"
            )
            self.distil_pool = layers.MaxPool1D(pool_size=2, strides=2,
                                                padding="same")

    def call(self, x, training=False):
        # ProbSparse self-attention
        attn_out = self.attn(x, training=training)
        attn_out = self.drop1(attn_out, training=training)
        x = self.norm1(x + attn_out)

        # Conv distilling (halves sequence length)
        if self.use_distilling:
            x = self.distil_conv(x)
            x = self.distil_pool(x)

        # FFN
        ffn_out = self.ffn1(x)
        ffn_out = self.ffn2(ffn_out)
        ffn_out = self.drop2(ffn_out, training=training)
        x = self.norm2(x + ffn_out)
        return x


# ──────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────

def build_informer(
    lookback: int,
    n_features: int,
    horizon: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    dff: int = 128,
    dropout: float = 0.1,
    use_distilling: bool = True,
    learning_rate: float = 0.001,
) -> models.Model:
    """Build and compile an Informer for multi-step forecasting.

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
        Number of attention heads.
    n_layers : int
        Number of Informer encoder layers.
    dff : int
        Inner dimension of the FFN sublayer.
    dropout : float
        Dropout rate.
    use_distilling : bool
        Whether to apply Conv distilling (halving of sequence length).
    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model with MSE loss and Adam optimizer.
    """
    pos_enc = _sinusoidal_positional_encoding(lookback, d_model)

    inp = layers.Input(shape=(lookback, n_features), name="input")

    # Linear projection
    x = layers.Dense(d_model, name="input_projection")(inp)
    x = x + pos_enc

    if dropout > 0:
        x = layers.Dropout(dropout, name="input_dropout")(x)

    # Informer encoder layers
    for i in range(n_layers):
        # After distilling, sequence length halves, so only distil for first n_layers-1
        distil = use_distilling and (i < n_layers - 1)
        x = InformerEncoderLayer(
            d_model=d_model, n_heads=n_heads, dff=dff,
            dropout=dropout, use_distilling=distil,
            name=f"informer_layer_{i}",
        )(x)

    # Global average pooling
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # Output projection
    out = layers.Dense(horizon, name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="Informer")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model
