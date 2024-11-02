import jax.numpy as np
from numpyro.distributions.util import promote_shapes
from jax.typing import ArrayLike

from .base import LinearTimeseries


class LocalLinearTrend(LinearTimeseries):
    """
    Implements a local linear trend model.

    Args:
        std: Standard deviation of random walks.
        initial_value: Initial value of random walks.
    """

    def __init__(self, std: ArrayLike, initial_value: ArrayLike, drift: ArrayLike = None, **kwargs):
        if drift is None:
            drift = np.zeros_like(initial_value)

        std, initial_value, drift = promote_shapes(std, initial_value, drift, shape=(2,))
        matrix = np.array([[1.0, 1.0], [0.0, 1.0]])

        super().__init__(drift, matrix, std, initial_value, **kwargs)
