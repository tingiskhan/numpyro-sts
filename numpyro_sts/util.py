from typing import Any, Generator

import jax.numpy as np


def cast_to_tensor(*x: Any) -> Generator[np.ndarray, None, None]:
    return (xt if isinstance(xt, np.ndarray) else np.array(xt) for xt in x)


def build_selector(std: np.ndarray) -> np.ndarray:
    """
    Builds a selector matrix based on the boolean mask indicating zeros.

    Args:
        std: Standard deviation.

    Returns:
        An :class:`np.ndarray`.
    """

    eye = np.eye(std.shape[0])
    indices = np.argwhere(std == 0.0)

    return np.delete(eye, indices, axis=-1, assume_unique_indices=True)