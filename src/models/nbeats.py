"""
N-BEATS Multivariate Variant (NBEATS-MV) for deterministic multi-step forecasting.

IMPORTANT LABEL: This implementation is labeled **NBEATS-MV** in all result tables.
The canonical N-BEATS (Oreshkin et al., 2019) is a univariate model that takes a
1-D historical load sequence and produces doubly-residual forecasts through
trend + seasonality stacks.  This implementation extends the architecture to
accept multivariate input (load + weather + temporal + lag features) by
flattening the (lookback × n_features) input before each block's FC stack.
The key N-BEATS design elements preserved are:
    1. Doubly-residual stacking: each block produces a backcast (subtracted from
       the input) and a forecast (summed into the final output).
    2. Block types: Generic, Trend, and Seasonality.
    3. No recurrent layers — purely feedforward.

For E0 (load-only, n_features=1), NBEATS-MV is equivalent to canonical N-BEATS.

Architecture
------------
    Input (batch, lookback, n_features)
        ↓  flatten  →  (batch, lookback * n_features)
        ↓
    Stack_1 (Trend blocks)
        Block_1_1: FC → backcast + forecast
        Block_1_2: ...
        Block_1_k: ...
    Stack_2 (Seasonality blocks)
        Block_2_1: ...
        ...
        ↓
    Sum of all block forecasts  →  (batch, horizon)

Hyperparameters (baseline)
--------------------------
    n_stacks            : 2   (1 trend + 1 seasonality)
    n_blocks_per_stack  : 3
    hidden_units        : 256
    layers_per_block    : 4
    learning_rate       : 0.001
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


# ──────────────────────────────────────────────────────────────────────
# Basis functions for Trend and Seasonality blocks
# ──────────────────────────────────────────────────────────────────────

def _trend_basis(lookback: int, horizon: int, degree: int = 3) -> tuple:
    """Polynomial basis vectors for trend block.

    Returns
    -------
    backcast_basis : np.ndarray (degree+1, lookback)
    forecast_basis : np.ndarray (degree+1, horizon)
    """
    t_back = np.linspace(0, 1, lookback)
    t_fore = np.linspace(0, 1, horizon)
    backcast_basis = np.stack([t_back ** d for d in range(degree + 1)])
    forecast_basis = np.stack([t_fore ** d for d in range(degree + 1)])
    return (backcast_basis.astype(np.float32),
            forecast_basis.astype(np.float32))


def _seasonality_basis(lookback: int, horizon: int) -> tuple:
    """Fourier basis vectors for seasonality block.

    Uses the minimum of lookback and horizon harmonics.
    """
    n_harmonics = max(1, min(lookback // 2, horizon // 2, 24))
    t_back = np.arange(lookback) / lookback
    t_fore = np.arange(horizon) / horizon

    cos_back = np.stack([np.cos(2 * np.pi * h * t_back) for h in range(1, n_harmonics + 1)])
    sin_back = np.stack([np.sin(2 * np.pi * h * t_back) for h in range(1, n_harmonics + 1)])
    cos_fore = np.stack([np.cos(2 * np.pi * h * t_fore) for h in range(1, n_harmonics + 1)])
    sin_fore = np.stack([np.sin(2 * np.pi * h * t_fore) for h in range(1, n_harmonics + 1)])

    backcast_basis = np.concatenate([cos_back, sin_back], axis=0).astype(np.float32)
    forecast_basis = np.concatenate([cos_fore, sin_fore], axis=0).astype(np.float32)
    return backcast_basis, forecast_basis


# ──────────────────────────────────────────────────────────────────────
# N-BEATS Block
# ──────────────────────────────────────────────────────────────────────

class NBEATSBlock(layers.Layer):
    """Single N-BEATS block (generic, trend, or seasonality)."""

    def __init__(
        self,
        input_size: int,
        horizon: int,
        hidden_units: int,
        n_layers: int,
        block_type: str,      # 'generic' | 'trend' | 'seasonality'
        lookback: int,        # needed for basis construction
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.horizon = horizon
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.block_type = block_type
        self.lookback = lookback

        # Shared FC stack
        self.fc_stack = [
            layers.Dense(hidden_units, activation="relu",
                         name=f"{self.name}_fc{i}")
            for i in range(n_layers)
        ]

        if block_type == "generic":
            self.theta_backcast = layers.Dense(input_size, name=f"{self.name}_theta_back")
            self.theta_forecast = layers.Dense(horizon, name=f"{self.name}_theta_fore")
            self._backcast_basis = None
            self._forecast_basis = None

        elif block_type == "trend":
            degree = 3
            b_basis, f_basis = _trend_basis(lookback, horizon, degree)
            self._backcast_basis = tf.constant(b_basis)  # (degree+1, lookback)
            self._forecast_basis = tf.constant(f_basis)  # (degree+1, horizon)
            self.theta_coef = layers.Dense(degree + 1, name=f"{self.name}_theta")

        elif block_type == "seasonality":
            b_basis, f_basis = _seasonality_basis(lookback, horizon)
            n_harmonics_x2 = b_basis.shape[0]
            self._backcast_basis = tf.constant(b_basis)
            self._forecast_basis = tf.constant(f_basis)
            self.theta_coef = layers.Dense(n_harmonics_x2, name=f"{self.name}_theta")

        else:
            raise ValueError(f"Unknown block_type: {block_type}")

    def call(self, x):
        # x: (batch, input_size)
        h = x
        for fc in self.fc_stack:
            h = fc(h)

        if self.block_type == "generic":
            backcast = self.theta_backcast(h)   # (batch, input_size)
            forecast = self.theta_forecast(h)   # (batch, horizon)
        else:
            # Basis expansion
            theta = self.theta_coef(h)          # (batch, n_basis)
            # backcast: (batch, n_basis) @ (n_basis, lookback_mv) — only first lookback cols
            backcast = theta @ self._backcast_basis  # (batch, lookback)
            # Pad/crop to match input_size (input_size = lookback * n_features)
            # For multi-feature case, only the load channel is reconstructed;
            # the residual on non-load channels is left at zero.
            n_features = self.input_size // self.lookback
            if n_features > 1:
                # Expand backcast across feature dimension by tiling
                backcast = tf.tile(
                    tf.expand_dims(backcast, axis=-1), [1, 1, n_features]
                )
                backcast = tf.reshape(backcast, (-1, self.input_size))
            forecast = theta @ self._forecast_basis   # (batch, horizon)

        return backcast, forecast


# ──────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────

def build_nbeats(
    lookback: int,
    n_features: int,
    horizon: int,
    n_stacks: int = 2,
    n_blocks_per_stack: int = 3,
    hidden_units: int = 256,
    layers_per_block: int = 4,
    learning_rate: float = 0.001,
) -> models.Model:
    """Build and compile NBEATS-MV for multi-step forecasting.

    Parameters
    ----------
    lookback : int
        Number of input time steps.
    n_features : int
        Number of input features per time step.
    horizon : int
        Number of output time steps (forecast horizon).
    n_stacks : int
        Number of stacks.  Stack 0 = Trend, Stack 1 = Seasonality,
        additional stacks = Generic.
    n_blocks_per_stack : int
        Number of blocks within each stack.
    hidden_units : int
        Width of each Dense layer in the FC stack.
    layers_per_block : int
        Number of Dense layers in each block's FC stack.
    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model with MSE loss and Adam optimizer.

    Notes
    -----
    This model is labeled **NBEATS-MV** (multivariate variant) to distinguish
    it from the canonical univariate N-BEATS.  See module docstring for details.
    """
    input_size = lookback * n_features

    # Determine block types per stack
    stack_types = []
    for s in range(n_stacks):
        if s == 0:
            stack_types.append("trend")
        elif s == 1:
            stack_types.append("seasonality")
        else:
            stack_types.append("generic")

    inp = layers.Input(shape=(lookback, n_features), name="input")
    x_flat = layers.Reshape((input_size,), name="flatten")(inp)

    residual = x_flat
    forecast_total = tf.zeros_like(
        layers.Dense(horizon, use_bias=False, trainable=False, name="_zero_init")(x_flat)
    )

    all_block_forecasts = []

    block_idx = 0
    for s, btype in enumerate(stack_types):
        for b in range(n_blocks_per_stack):
            block = NBEATSBlock(
                input_size=input_size,
                horizon=horizon,
                hidden_units=hidden_units,
                n_layers=layers_per_block,
                block_type=btype,
                lookback=lookback,
                name=f"nbeats_stack{s}_block{b}",
            )
            backcast, forecast = block(residual)

            # Doubly-residual: subtract backcast from residual
            residual = residual - backcast
            all_block_forecasts.append(forecast)
            block_idx += 1

    # Sum all block forecasts
    if len(all_block_forecasts) == 1:
        out = all_block_forecasts[0]
    else:
        out = layers.Add(name="forecast_sum")(all_block_forecasts)

    model = models.Model(inputs=inp, outputs=out, name="NBEATS_MV")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model
