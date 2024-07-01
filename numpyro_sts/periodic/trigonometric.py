from math import ceil

import jax.numpy as jnp
from numpy.typing import ArrayLike

from .cyclical import Cyclical


class TrigonometricSeasonality(Cyclical):
    """
    Represents a periodic component by means of a trigonometric series TODO: math

    Args:
        num_seasons: Number of seasons to use.
        initial_value: Initial value. Must be of size ``ceil(num_seasons / 2) x 2``.
    """

    def __init__(self, n: int, num_seasons: int, std: ArrayLike, initial_value: ArrayLike):
        half_seasons = ceil(num_seasons / 2)
        lamdas = 2 * jnp.pi * jnp.arange(1, half_seasons + 1) / num_seasons

        super().__init__(n, lamdas, std, initial_value)
