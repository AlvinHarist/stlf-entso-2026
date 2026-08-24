"""
Model training utilities.

Provides a single ``train_model`` function that handles:
- EarlyStopping on *validation loss* (never test loss)
- ``shuffle=False`` to preserve temporal ordering within batches
- ``restore_best_weights=True``
- Recording of training history and best epoch
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    batch_size: int = 32,
    patience: int = 15,
    save_dir: Optional[str | Path] = None,
    verbose: int = 1,
) -> Dict[str, Any]:
    """Train a Keras model with early stopping on validation loss.

    LEAKAGE SAFEGUARDS:
    - ``validation_data`` is the *validation* set, never the test set.
    - ``shuffle=False`` preserves temporal ordering of training batches.
    - Early stopping monitors ``val_loss``, not test loss.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled Keras model.
    X_train, y_train : np.ndarray
        Training features and targets (windowed).
    X_val, y_val : np.ndarray
        Validation features and targets (windowed).
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Mini-batch size.
    patience : int
        Early-stopping patience (epochs without improvement).
    save_dir : str or Path, optional
        If provided, the best model is saved here as ``best_model.keras``.
    verbose : int
        Keras verbosity level.

    Returns
    -------
    dict
        - ``history``: training history dict (loss, val_loss per epoch)
        - ``best_epoch``: epoch with the best validation loss (1-indexed)
        - ``best_val_loss``: best validation loss achieved
        - ``model``: trained model reference
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,        # preserve temporal ordering
        callbacks=callbacks,
        verbose=verbose,
    )

    # Determine best epoch (EarlyStopping restores best weights)
    val_losses = history.history["val_loss"]
    best_epoch = int(np.argmin(val_losses)) + 1  # 1-indexed
    best_val_loss = float(min(val_losses))

    # Optionally save model
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "best_model.keras"
        model.save(model_path)
        print(f"[trainer] Model saved to {model_path}")

    print(f"[trainer] Best epoch: {best_epoch} | "
          f"Best val_loss: {best_val_loss:.6f}")

    return {
        "history": history.history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "model": model,
    }
