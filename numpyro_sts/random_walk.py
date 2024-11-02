import jax.numpy as np
from jax.typing import ArrayLike

from .base import LinearTimeseries
from .util import cast_to_tensor


class RandomWalk(LinearTimeseries):
    """
    Defines a 1D Random Walk model with Gaussian increments.

    Args:
        n: see :class:`BaseLinearTimeseries`.
        std: Standard deviation of random increments.
        initial_value: Initial value of the process.
        drift: Drift of process. Defaults to 0.
    """

    def __init__(self, std, initial_value, drift: ArrayLike = 0.0, **kwargs):
        arrays = cast_to_tensor(drift, std, initial_value)

        arrays = np.broadcast_arrays(*arrays)
        offset, std, initial_value = (np.expand_dims(a, axis=-1) for a in arrays)

        matrix = np.ones((1, 1))

        super().__init__(offset, matrix, std, initial_value, **kwargs)
