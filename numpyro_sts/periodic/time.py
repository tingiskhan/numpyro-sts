from numpy.typing import ArrayLike
import jax.numpy as jnp
import numpy as np

from ..base import LinearTimeseries
from ..util import cast_to_tensor


class TimeSeasonal(LinearTimeseries):
    r"""
    Represents a periodic series of the form:

    .. math::
        \gamma_{t + 1} = \sum_{i = 1}^{s - 1} \gamma_{t + 1 - j} + \eps_{t + 1}

    Args:
        num_seasons: Number of seasons to include.

    """

    def __init__(self, num_seasons: int, initial_value: ArrayLike, **kwargs):
        top = -jnp.ones([1, num_seasons - 1])
        bottom = jnp.eye(num_seasons - 2, num_seasons - 1)

        matrix = jnp.concatenate([top, bottom], axis=-2)
        offset = jnp.zeros_like(top).squeeze(-2)

        initial_value, = cast_to_tensor(initial_value)
        std = np.zeros_like(initial_value)

        super().__init__(offset, matrix, std, initial_value, **kwargs)
