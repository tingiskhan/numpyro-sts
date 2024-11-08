import jax.numpy as jnp
import numpy as np
from numpyro.distributions.util import promote_shapes
from jax.typing import ArrayLike

from .base import LinearTimeseries


class SmoothLocalLinearTrend(LinearTimeseries):
    """
    Implements a smooth local linear trend model.

    Args:
        std: Standard deviation of random walk.
        initial_value: Initial value of random walks.
    """

    def __init__(self, n: int, std: ArrayLike, initial_value: ArrayLike, **kwargs):
        (initial_value,) = promote_shapes(initial_value, shape=(2,))

        (std,) = promote_shapes(std, shape=(1,))
        std = jnp.concatenate([jnp.zeros_like(std), std], axis=-1)

        matrix = jnp.array([[1.0, 1.0], [0.0, 1.0]])
        offset = jnp.zeros_like(initial_value)

        column_mask = np.array([False, True])

        super().__init__(n, offset, matrix, std, initial_value, column_mask=column_mask, **kwargs)
