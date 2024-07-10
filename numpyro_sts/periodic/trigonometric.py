from jax.scipy.linalg import block_diag
from numpy.typing import ArrayLike
import jax.numpy as jnp

from ..base import LinearTimeseries


class TrigonometricSeasonal(LinearTimeseries):
    """
    Represents a seasonal component by means of a trigonometric series TODO: math

    Args:
        periodicity: Periodicity of component.
    """

    def __init__(self, n: int, num_seasons: int, std: ArrayLike, initial_value: ArrayLike, **kwargs):
        half_seasons = num_seasons // 2
        lamda = 2.0 * jnp.pi * jnp.arange(1, half_seasons + 1) / num_seasons

        cos_lamda = jnp.cos(lamda)
        sin_lamda = jnp.sin(lamda)

        top = jnp.stack([cos_lamda, sin_lamda], axis=-1)
        bottom = jnp.stack([-sin_lamda, cos_lamda], axis=-1)
        matrix = jnp.stack([top, bottom], axis=-2)

        split_matrix = jnp.split(matrix, half_seasons, axis=0)
        matrix = block_diag(*(m.squeeze(-3) for m in split_matrix))

        offset = jnp.zeros(2 * half_seasons)
        std = jnp.full_like(offset, std)

        super().__init__(n, offset, matrix, std, initial_value, **kwargs)
