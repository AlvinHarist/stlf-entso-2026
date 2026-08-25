"""
Temporal Convolutional Network (TCN) for deterministic multi-step forecasting.

Architecture
------------
    Input (batch, lookback, n_features)
        ↓
    N × ResidualBlock
        (CausalConv1D(dilation=2^i) → WeightNorm → ReLU → Dropout)
        (CausalConv1D(dilation=2^i) → WeightNorm → ReLU → Dropout)
        (+ 1×1 residual connection if channels mismatch)
        ↓
    Last timestep slice  (batch, filters)
        ↓
    Dense(horizon)  →  Output (batch, horizon)

Design Notes
------------
- Causal padding ensures no future information leaks into the convolution.
- Exponentially increasing dilation allows the receptive field to grow
  quickly:  RF = (kernel_size - 1) * sum(2^i for i in range(n_blocks)) + 1
- Weight Normalization is applied via tf.keras.layers.LayerNormalization
  per block (approximating the original WN from the TCN paper).
- The residual path uses a 1×1 conv to match channels when n_features ≠ filters.
- Dropout is placed after each activation, not after the residual sum.

Hyperparameters (baseline)
--------------------------
    filters          : 64
    kernel_size      : 3
    n_blocks         : 4    (dilation schedule: [1, 2, 4, 8])
    dropout          : 0.1
    learning_rate    : 0.001
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


# ──────────────────────────────────────────────────────────────────────
# Residual Block
# ──────────────────────────────────────────────────────────────────────

class TCNResidualBlock(layers.Layer):
    """TCN residual block with dilated causal convolutions."""

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout

        # First dilated causal conv
        self.conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            padding="causal",
            activation="relu",
            name=f"{self.name}_conv1",
        )
        self.norm1 = layers.LayerNormalization(name=f"{self.name}_norm1")
        self.drop1 = layers.Dropout(dropout, name=f"{self.name}_drop1")

        # Second dilated causal conv
        self.conv2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            padding="causal",
            activation="relu",
            name=f"{self.name}_conv2",
        )
        self.norm2 = layers.LayerNormalization(name=f"{self.name}_norm2")
        self.drop2 = layers.Dropout(dropout, name=f"{self.name}_drop2")

        # 1×1 residual projection (built lazily in first call)
        self._residual_proj = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        if in_channels != self.filters:
            self._residual_proj = layers.Conv1D(
                filters=self.filters,
                kernel_size=1,
                padding="same",
                name=f"{self.name}_res_proj",
            )
        super().build(input_shape)

    def call(self, x, training=False):
        residual = x

        # Branch
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.drop1(out, training=training)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.drop2(out, training=training)

        # Residual connection
        if self._residual_proj is not None:
            residual = self._residual_proj(residual)

        return tf.keras.activations.relu(out + residual)


# ──────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────

def build_tcn(
    lookback: int,
    n_features: int,
    horizon: int,
    filters: int = 64,
    kernel_size: int = 3,
    n_blocks: int = 4,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
) -> models.Model:
    """Build and compile a TCN for multi-step forecasting.

    Parameters
    ----------
    lookback : int
        Number of input time steps.
    n_features : int
        Number of input features per time step.
    horizon : int
        Number of output time steps (forecast horizon).
    filters : int
        Number of convolutional filters per block.
    kernel_size : int
        Convolution kernel size.
    n_blocks : int
        Number of residual blocks.  Dilation schedule is [1, 2, 4, ..., 2^(n_blocks-1)].
    dropout : float
        Dropout rate inside each residual block.
    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model with MSE loss and Adam optimizer.

    Notes
    -----
    Receptive field (in time steps):
        RF = 1 + 2 * (kernel_size - 1) * sum(2^i for i in range(n_blocks))
    With defaults (kernel=3, n_blocks=4):
        RF = 1 + 2*2*(1+2+4+8) = 1 + 60 = 61  →  covers 61 past steps.
    """
    inp = layers.Input(shape=(lookback, n_features), name="input")
    x = inp

    # Stack of residual blocks with exponentially increasing dilation
    for i in range(n_blocks):
        dilation = 2 ** i
        x = TCNResidualBlock(
            filters=filters,
            kernel_size=kernel_size,
            dilation=dilation,
            dropout=dropout,
            name=f"tcn_block_{i}",
        )(x)

    # Take only the last timestep (sequence-to-vector)
    x = layers.Lambda(lambda t: t[:, -1, :], name="last_timestep")(x)

    # Output projection
    out = layers.Dense(horizon, name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="TCN")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model
