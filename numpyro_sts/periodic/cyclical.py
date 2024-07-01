import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..base import LinearTimeseries
from ..util import cast_to_tensor


class Cyclical(LinearTimeseries):
    """
    Represents a periodic component by means of a trigonometric series TODO: math
    """

    def __init__(self, n: int, periodicity: int, std: ArrayLike, initial_value: ArrayLike):
        offset = jnp.zeros(2)
        lamda, offset = cast_to_tensor(periodicity, offset)

        cos_lamda = jnp.cos(lamda)
        sin_lamda = jnp.sin(lamda)

        matrix = jnp.array([
            [cos_lamda, sin_lamda],
            [-sin_lamda, cos_lamda]
        ])

        std = jnp.full_like(offset, std)

        super().__init__(n, offset, matrix, std, initial_value)
