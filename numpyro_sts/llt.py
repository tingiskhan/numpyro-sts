import jax.numpy as np
from numpyro.distributions.util import promote_shapes
from jax.typing import ArrayLike

from numpyro.distributions import TransformedDistribution, Normal

from .base import LinearTimeseries


class LocalLinearTrend(TransformedDistribution):
    """
    Implements a local linear trend model.

    Args:
        std: Standard deviation of random walks.
        initial_value: Initial value of random walks.
    """

    def __init__(self, num_steps: int, std: ArrayLike, initial_value: ArrayLike, drift: ArrayLike = None, **kwargs):
        if drift is None:
            drift = np.zeros_like(initial_value)

        std, initial_value, drift = promote_shapes(std, initial_value, drift, shape=(2,))
        matrix = np.array([[1.0, 1.0], [0.0, 1.0]])

        transform = LinearTimeseries(drift, matrix, std, initial_value)
        base_dist = Normal().expand((num_steps, 2))

        super().__init__(base_dist, transform)
