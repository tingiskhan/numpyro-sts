from math import ceil

import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..base import LinearTimeseries


class TrigonometricSeasonality(LinearTimeseries):
    """
    Represents a periodic component by means of a trigonometric series TODO: math
    """

    def __init__(self, n: int, num_seasons: int, std: ArrayLike, initial_value: ArrayLike):
        half_seasons = ceil(num_seasons / 2)

        lamdas = 2 * jnp.pi * jnp.arange(1, half_seasons) / num_seasons


        super().__init__(n, offset, matrix, std, initial_value)

