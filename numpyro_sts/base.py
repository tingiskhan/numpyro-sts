import warnings
from functools import cached_property, reduce
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax import vmap, lax
from jax.random import normal, PRNGKey
from numpyro.contrib.control_flow import scan
from numpyro.distributions import Distribution, Normal, constraints, MultivariateNormal
from numpyro.distributions.transforms import RecursiveLinearTransform
from numpyro.distributions.util import validate_sample
from numpyro.util import is_prng_key
from jax.typing import ArrayLike
import jax.scipy.linalg as linalg


def _broadcast_and_reshape(x: jnp.ndarray, shape, dim: int) -> jnp.ndarray:
    last_dims = x.shape[dim:]
    return jnp.broadcast_to(x, shape + last_dims).reshape((-1,) + last_dims)


def _loc_transition(state, offset, matrix) -> jnp.ndarray:
    return offset + (matrix @ state[..., None]).reshape(state.shape)


def _sample_shocks(
    key: PRNGKey, event_shape: Tuple[int, ...], batch_shape: Tuple[int, ...], selector: jnp.ndarray
) -> jnp.ndarray:
    shock_shape = event_shape[:-1] + selector.shape[-1:]

    flat_shape = () if not batch_shape else (reduce(lambda u, v: u * v, batch_shape),)
    samples = normal(key, shape=flat_shape + shock_shape)

    fun = jnp.matmul
    if batch_shape:
        selector = jnp.broadcast_to(selector, samples.shape[:1] + selector.shape)
        fun = vmap(fun)

    rotated_samples = fun(selector, samples[..., None]).squeeze(-1)

    return rotated_samples.reshape(batch_shape + event_shape)


def _verify_parameters(offset, matrix, std, initial_value, std_is_matrix):
    ndim = matrix.shape[-1]

    assert initial_value.ndim >= 1
    assert matrix.ndim >= 2 and matrix.shape[-2] == matrix.shape[-1] == ndim
    assert initial_value.shape[-1] == matrix.shape[-1]

    assert offset.shape[-1] == initial_value.shape[-1]

    if std_is_matrix:
        assert std.ndim >= 2 and std.shape[-1] == std.shape[-2] == ndim


class LinearTimeseries(RecursiveLinearTransform):
    r"""
    Defines a base model for linear stochastic models with Gaussian increments.

    Args:
        offset: Constant offset in transition equation. Of size :math:`[batch size] \times dimension`.
        matrix: Matrix of linear combination of states. Of size :math:`[batch size] \times dimension \times dimension`.
        std: Standard deviation of innovations. Of size :math:`[batch size] \times dimension`.
        initial_value: Initial value of the time series. Of size :math:`[batch size] \times dimension`.
    """

    def __init__(
        self,
        offset: ArrayLike,
        matrix: ArrayLike,
        std: ArrayLike,
        initial_value: ArrayLike,
        *,
        std_is_matrix: bool = False,
        mask: np.ndarray = None,
    ):
        _verify_parameters(offset, matrix, std, initial_value, std_is_matrix)
        self._std_is_matrix = std_is_matrix

        batch_shape = jnp.broadcast_shapes(
            offset.shape[:-1], matrix.shape[:-2], std.shape[: -(1 + int(self._std_is_matrix))], initial_value.shape[:-1]
        )

        parameter_shape = batch_shape + initial_value.shape[-1:]

        self.offset = jnp.broadcast_to(offset, parameter_shape)
        self.initial_value = jnp.broadcast_to(initial_value, parameter_shape)
        self.matrix = jnp.broadcast_to(matrix, parameter_shape + initial_value.shape[-1:])

        std_shape = parameter_shape if not self._std_is_matrix else parameter_shape + initial_value.shape[-1:]
        self.std = jnp.broadcast_to(std, std_shape)

        self.mask = (mask if mask is not None else np.ones(self.matrix.shape[-1])).astype(bool)
        self.selector = np.eye(self.matrix.shape[-1])[:, mask]

        super().__init__(transition_matrix=matrix)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.einsum("...j,kj->...k", jnp.moveaxis(x, -2, 0), self.selector)

        if not self._std_is_matrix:
            x *= self.std

        def f(y_t, x_tp1):
            y_t = jnp.einsum("...ij,...j->...i", self.transition_matrix, y_t) + x_tp1 + self.offset
            return y_t, y_t

        _, y = lax.scan(f, self.initial_value, x)
        return jnp.moveaxis(y, 0, -2)

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        # Move the time axis to the first position so we can scan over it in reverse.
        y = jnp.moveaxis(y, -2, 0)

        def f(y_t, prev):
            x_tp1 = y_t - jnp.einsum("...ij,...j->...i", self.transition_matrix, prev) - self.offset

            if not self._std_is_matrix:
                x_tp1 /= self.std

            return prev, x_tp1

        _, x = lax.scan(f, y[-1], jnp.roll(y, 1, axis=0).at[0].set(self.initial_value), reverse=True)
        return jnp.moveaxis(x, 0, -2)

    def log_abs_det_jacobian(self, x: jnp.ndarray, y: jnp.ndarray, intermediates=None):
        return jnp.zeros_like(x, shape=x.shape[:-2])

    def tree_flatten(self):
        params = (self.transition_matrix, self.offset, self.initial_value, self.std)
        param_names = ("transition_matrix", "offset", "initial_value", "std")
        aux_data = {"std_is_matrix": self._std_is_matrix, "mask": self.mask}

        return params, (param_names, aux_data)

    def __eq__(self, other):
        raise NotImplementedError()

    def union(self, other: "LinearTimeseries") -> "LinearTimeseries":
        """
        Combines self with other series to create a joint series.

        Args:
            other: Other series to combine.

        Returns:
            Returns a new instance of :class:`LinearTimeseries`.
        """

        matrix = linalg.block_diag(self.matrix, other.matrix)
        offset = jnp.concatenate([self.offset, other.offset], axis=-1)
        initial_value = jnp.concatenate([self.initial_value, other.initial_value], axis=-1)

        # TODO: fix other ones as well
        if any([self._std_is_matrix, other._std_is_matrix]):
            raise NotImplementedError("Do not handle matrix!")

        std = jnp.concatenate([self.std, other.std], axis=-1)
        mask = np.concatenate([self.mask, other.mask], axis=-1)

        model = LinearTimeseries(offset, matrix, std, initial_value, std_is_matrix=False, mask=mask)

        return model

    def predict(self, value: jnp.ndarray) -> "LinearTimeseries":
        """
        Creates a "prediction" instance of self.

        Args:
            value: New start value.

        Returns:
            Returns new instance of :class:`LinearTimeseries`.
        """

        future_model = LinearTimeseries(
            self.offset,
            self.matrix,
            self.std,
            value,
            std_is_matrix=self._std_is_matrix,
            mask=self.mask,
        )

        return future_model

    def deterministic(self) -> "LinearTimeseries":
        """
        Constructs a deterministic version of the series.

        Notes:
            Only use deterministic models in conjunction with non-deterministic models.

        Returns:
            Instance of :class:`LinearTimeseries`.
        """

        model = LinearTimeseries(
            self.offset,
            self.matrix,
            self.std,
            self.initial_value,
            std_is_matrix=self._std_is_matrix,
            mask=np.zeros_like(self.mask),
        )

        return model