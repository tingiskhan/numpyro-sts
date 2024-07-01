from numpy.typing import ArrayLike
import jax.numpy as jnp

from ..base import LinearTimeseries
from ..util import cast_to_tensor


class SeasonalSeries(LinearTimeseries):
    r"""
    Represents a periodic component of the form:

    .. math::
        \gamma_{t + 1} = \sum_{i = 1}^{s - 1} \gamma_{t + 1 - j} + \eps_{t + 1}

    Args:
        num_seasons: Number of seasons to include.

    """

    def __init__(self, n: int, num_seasons: int, std: ArrayLike, initial_value: ArrayLike):
        top = -jnp.ones([1, num_seasons - 1])
        bottom = jnp.eye(num_seasons - 2, num_seasons - 1)

        matrix = jnp.concatenate([top, bottom], axis=-2)
        offset = jnp.ones_like(top).squeeze(-2)

        mask = jnp.eye(num_seasons - 1, 1).squeeze(-1).astype(jnp.bool_)

        std, initial_value = cast_to_tensor(std, initial_value)
        std = std * mask

        super().__init__(n, offset, matrix, std, initial_value, mask=mask)
