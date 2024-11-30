import jax.numpy as np
from jax.typing import ArrayLike

from numpyro.distributions import TransformedDistribution, Normal

from .base import LinearTimeseries
from .util import cast_to_tensor


class RandomWalk(TransformedDistribution):
    """
    Defines a 1D Random Walk model with Gaussian increments.

    Args:
        num_steps: see :class:`BaseLinearTimeseries`.
        std: Standard deviation of random increments.
        initial_value: Initial value of the process.
        drift: Drift of process. Defaults to 0.
    """

    def __init__(self, num_steps: int, std, initial_value, drift: ArrayLike = 0.0):
        arrays = cast_to_tensor(drift, std, initial_value)

        arrays = np.broadcast_arrays(*arrays)
        offset, std, initial_value = (np.expand_dims(a, axis=-1) for a in arrays)

        matrix = np.ones((1, 1))

        transform = LinearTimeseries(offset, matrix, std, initial_value)
        base_dist = Normal().expand((num_steps, 1))

        super().__init__(base_dist, transform)
