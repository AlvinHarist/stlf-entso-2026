"""
Reproducibility utilities.

Sets seeds for Python random, NumPy, and TensorFlow to ensure
reproducible experiments. Note that perfect GPU determinism is NOT
guaranteed by TensorFlow unless additional environment variables
(TF_DETERMINISTIC_OPS=1, TF_CUDNN_DETERMINISTIC=1) are set, and even
then some operations may remain non-deterministic.
"""

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to use across all random number generators.

    Notes
    -----
    - Python's built-in ``random`` module is seeded.
    - NumPy's global RNG is seeded.
    - TensorFlow's global seed is set.
    - ``PYTHONHASHSEED`` is set for hash reproducibility.
    - GPU determinism is NOT guaranteed without additional TF config.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Import TensorFlow lazily so the module can be imported without TF.
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
