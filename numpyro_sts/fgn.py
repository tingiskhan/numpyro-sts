from functools import reduce
from operator import mul

import jax.numpy as np
from jax.random import PRNGKey, normal
from numpyro.distributions import Distribution, constraints


def sample_stationary_gaussian(key: PRNGKey, covariances: np.ndarray, num_samples: int = None) -> np.ndarray:
    r"""
    Samples an arbitrary stationary Gaussian process via `Davies and Harte`_ .

    .. _`Davies and Harte`: https://hagerpa.github.io/talks/excersize_sheet_sampling_of_fBm.pdf
    Args:
        key: Random key to use.
        covariances: Covariance of time.
        num_samples: Number of samples to produce.

    Returns:
        Returns an array of size :math:`T \times N`.
    """

    flipped_covariances = np.flip(covariances, axis=-1)
    gammas = np.concatenate([covariances, flipped_covariances[1:-1]], axis=-1)

    unconstrained_lambdas = np.fft.fft(gammas).real
    lambdas = np.where(unconstrained_lambdas < 0.0, 0.0, unconstrained_lambdas)

    shape = covariances.shape[:-1] + (2 * (covariances.shape[-1] - 1),)
    if num_samples:
        shape = (num_samples,) + shape

    gaussians = normal(key, shape=shape)

    sqrt_diag = lambdas ** 0.5
    intermediary = sqrt_diag * np.fft.ifft(gaussians).real

    return np.fft.fft(intermediary, n=covariances.shape[-1]).real


def sample_fgn(key: PRNGKey, h: float, num_timesteps: int, **kwargs) -> np.ndarray:
    r"""
    Samples a fractional Gaussian process parametrized via the Hurst exponent.

    Args:
        key: See :meth:`sample_stationary_gaussian`.
        h: Hurst exponent.
        num_timesteps: Size of time grid.

    Returns:
        Returns sampled fractional Gaussian noise.
    """

    grid = np.arange(0, num_timesteps)

    two_h = 2.0 * h
    cov = 0.5 * ((grid + 1) ** two_h + np.abs(grid - 1) ** two_h) - np.abs(grid) ** two_h

    return sample_stationary_gaussian(key, cov, **kwargs)


class FractionalGaussianNoise(Distribution):
    """
    Implements the fractional Gaussian.

    Args:
        n: Number of timesteps.
        h: Hurst exponent.

    """

    arg_constraints = {
        "h": constraints.open_interval(0.0, 1.0),
        "n": constraints.integer_greater_than(0),
    }

    support = constraints.real

    pytree_aux_fields = ("n",)
    pytree_data_fields = ("h",)

    def __init__(self, n: int, h: float):
        self.h = np.array(h)
        self.n = n

        event_shape = (n,)
        batch_shape = self.h.shape

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, key, sample_shape=()):
        m = None
        if sample_shape:
            m = reduce(mul, sample_shape)

        samples = sample_fgn(key, self.h, self.n, num_samples=m)

        return samples.reshape(sample_shape + self.batch_shape + self.event_shape)
