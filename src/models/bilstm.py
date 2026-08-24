"""
Bidirectional LSTM model for deterministic point forecasting.

Architecture
------------
    Input (batch, lookback, n_features)
        ↓
    Bidirectional(LSTM(units))
        ↓
    Dropout(dropout)
        ↓
    Dense(horizon)  →  Output (batch, horizon)
"""

from tensorflow.keras import layers, models, optimizers


def build_bilstm(
    lookback: int,
    n_features: int,
    horizon: int,
    units: int = 64,
    dropout: float = 0.0,
    learning_rate: float = 0.001,
) -> models.Model:
    """Build and compile a Bidirectional LSTM for multi-step forecasting.

    Parameters
    ----------
    lookback : int
        Number of input time steps.
    n_features : int
        Number of input features per time step.
    horizon : int
        Number of output time steps (forecast horizon).
    units : int
        Number of LSTM hidden units (per direction).
    dropout : float
        Dropout rate applied after the BiLSTM layer (0 = no dropout).
    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model with MSE loss and Adam optimizer.
    """
    inp = layers.Input(shape=(lookback, n_features), name="input")
    x = layers.Bidirectional(
        layers.LSTM(units, return_sequences=False),
        name="bilstm",
    )(inp)
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(horizon, name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="BiLSTM")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model
