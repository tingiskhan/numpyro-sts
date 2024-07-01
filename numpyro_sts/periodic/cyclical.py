import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..base import LinearTimeseries
from ..util import cast_to_tensor


class Cyclical(LinearTimeseries):
    """
    Represents a periodic component by means of a trigonometric series TODO: math

    Args:
        periodicity: Periodicity of component.
    """

    def __init__(self, n: int, periodicity: int, std: ArrayLike, initial_value: ArrayLike):
        lamda, = cast_to_tensor(periodicity)

        cos_lamda = jnp.cos(lamda)
        sin_lamda = jnp.sin(lamda)

        top = jnp.stack([cos_lamda, sin_lamda], axis=-1)
        bottom = jnp.stack([-sin_lamda, cos_lamda], axis=-1)
        matrix = jnp.stack([top, bottom], axis=-2)

        offset = jnp.zeros(2)
        std = jnp.full_like(offset, std)

        super().__init__(n, offset, matrix, std, initial_value)
