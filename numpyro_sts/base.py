import warnings
from functools import cached_property, reduce
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.random import normal, PRNGKey
from numpyro.contrib.control_flow import scan
from numpyro.distributions import Distribution, Normal, constraints, MultivariateNormal
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


class LinearTimeseries(Distribution):
    r"""
    Defines a base model for linear stochastic models with Gaussian increments.

    Args:
        offset: Constant offset in transition equation. Of size :math:`[batch size] \times dimension`.
        matrix: Matrix of linear combination of states. Of size :math:`[batch size] \times dimension \times dimension`.
        std: Standard deviation of innovations. Of size :math:`[batch size] \times dimension`.
        initial_value: Initial value of the time series. Of size :math:`[batch size] \times dimension`.
        column_mask: Mask for constructing the "selector" matrix.
    """

    pytree_data_fields = ("offset", "matrix", "std", "initial_value")
    pytree_aux_fields = ("n", "_std_is_matrix", "column_mask", "selector")

    support = constraints.real_matrix
    has_enumerate_support = False

    arg_constraints = {
        "offset": constraints.real_vector,
        "matrix": constraints.real_matrix,
        "std": constraints.real_vector,
        "initial_value": constraints.real_vector,
        "n": constraints.positive_integer,
    }

    def __init__(
        self,
        n: int,
        offset: ArrayLike,
        matrix: ArrayLike,
        std: ArrayLike,
        initial_value: ArrayLike,
        *,
        std_is_matrix: bool = False,
        column_mask: np.ndarray = None,
        validate_args=None,
        **kwargs,
    ):
        if "mask" in kwargs:
            warnings.warn("'mask' is deprecated in favor of 'column_mask'", DeprecationWarning)
            column_mask = kwargs.pop("mask")

        _verify_parameters(offset, matrix, std, initial_value, std_is_matrix)
        times = jnp.arange(n)

        self._std_is_matrix = std_is_matrix

        event_shape = times.shape + initial_value.shape[-1:]
        batch_shape = jnp.broadcast_shapes(
            offset.shape[:-1], matrix.shape[:-2], std.shape[: -(1 + int(self._std_is_matrix))], initial_value.shape[:-1]
        )

        parameter_shape = batch_shape + initial_value.shape[-1:]

        self.n = n
        self.offset = jnp.broadcast_to(offset, parameter_shape)
        self.initial_value = jnp.broadcast_to(initial_value, parameter_shape)
        self.matrix = jnp.broadcast_to(matrix, parameter_shape + initial_value.shape[-1:])

        std_shape = parameter_shape if not self._std_is_matrix else parameter_shape + initial_value.shape[-1:]
        self.std = jnp.broadcast_to(std, std_shape)

        super().__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)

        if column_mask is None:
            column_mask = np.ones(self.event_shape[-1], dtype=np.bool_)

        self.column_mask = column_mask
        self.selector = np.eye(self.event_shape[-1])[..., self.column_mask]

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)

        def body(state, eps_tp1):
            x_tp1 = self.sample_from_shock(state, eps_tp1)
            return x_tp1, x_tp1

        def scan_fn(init, noise):
            return scan(body, init, noise)

        batch_shape = sample_shape + self.batch_shape

        shocks = _sample_shocks(key, self.event_shape, batch_shape, self.selector)
        inits = jnp.broadcast_to(self.initial_value, sample_shape + self.initial_value.shape)

        batch_dim = len(batch_shape)
        if batch_dim:
            shocks = jnp.moveaxis(shocks, -2, 0)
            _, samples = scan_fn(inits, shocks)

            return jnp.moveaxis(samples, 0, -2)

        return scan_fn(inits, shocks)[-1]

    @validate_sample
    def log_prob(self, value):
        # NB: very similar to numpyro's implementation of EulerMaruyama
        sample_shape = jnp.broadcast_shapes(value.shape[: -self.event_dim], self.batch_shape)
        value = jnp.broadcast_to(value, sample_shape + self.event_shape)

        initial_value = jnp.expand_dims(self.initial_value, -2)
        initial_value = jnp.broadcast_to(initial_value, sample_shape + initial_value.shape[-2:])

        stacked = jnp.concatenate([initial_value, value], axis=-2)
        loc_fun = _loc_transition

        offset = self.offset
        matrix = self.matrix
        std = self.std

        selector = self.selector
        inverse_fun = jnp.matmul

        if sample_shape:
            stacked = stacked.reshape((-1,) + stacked.shape[-2:])

            offset = _broadcast_and_reshape(offset, sample_shape, -1)
            matrix = _broadcast_and_reshape(matrix, sample_shape, -2)
            std = _broadcast_and_reshape(std, sample_shape, -2 if self._std_is_matrix else -1)

            selector = jnp.broadcast_to(selector, stacked.shape[:1] + selector.shape)

            loc_fun = vmap(_loc_transition)
            inverse_fun = vmap(inverse_fun)

        x_tm1 = stacked[..., :-1, :]
        x_t = stacked[..., 1:, :]

        loc = loc_fun(x_tm1, offset, matrix)

        transposed_selector = selector.swapaxes(-1, -2)
        loc = inverse_fun(transposed_selector, loc[..., None]).squeeze(-1)
        x_t = inverse_fun(transposed_selector, x_t[..., None]).squeeze(-1)

        if not self._std_is_matrix:
            std = inverse_fun(transposed_selector, std[..., None]).swapaxes(-1, -2)
            dist = Normal(loc, std).to_event(2)
        else:
            std = inverse_fun(transposed_selector, std)
            std = jnp.expand_dims(std, -3)

            dist = MultivariateNormal(loc, scale_tril=std).to_event(1)

        return dist.log_prob(x_t).reshape(sample_shape)

    def sample_from_shock(self, x_t, eps_t: jnp.ndarray) -> jnp.ndarray:
        """
        Samples the distribution conditioned on shocks.

        Args:
            x_t: Current state.
            eps_t: The shocks to use.

        Returns:
            Returns sampled process.
        """

        loc = _loc_transition(x_t, self.offset, self.matrix)

        if not self._std_is_matrix:
            return loc + self.std * eps_t

        return loc + (self.std @ eps_t[..., None]).squeeze(-1)

    def union(self, other: "LinearTimeseries") -> "LinearTimeseries":
        """
        Combines self with other series to create a joint series.

        Args:
            other: Other series to combine.

        Returns:
            Returns a new instance of :class:`LinearTimeseries`.
        """

        assert self.n == other.n, "Number of steps do not match!"
        batch_shape = jnp.broadcast_shapes(self.batch_shape, other.batch_shape)

        assert not batch_shape, "Currently does not support batch shapes!"

        matrix = linalg.block_diag(self.matrix, other.matrix)
        offset = jnp.concatenate([self.offset, other.offset], axis=-1)
        initial_value = jnp.concatenate([self.initial_value, other.initial_value], axis=-1)

        # TODO: fix other ones as well
        std = jnp.concatenate([self.std, other.std], axis=-1)
        mask = np.concatenate([self.column_mask, other.column_mask], axis=-1)

        model = LinearTimeseries(self.n, offset, matrix, std, initial_value, column_mask=mask, std_is_matrix=False)

        return model

    def deterministic(self) -> "LinearTimeseries":
        """
        Constructs a deterministic version of the timeseries.

        Returns:
            A :class:`LinearTimeseries` with column mask set to False.
        """

        model = LinearTimeseries(
            self.n,
            self.offset,
            self.matrix,
            self.std,
            self.initial_value,
            column_mask=np.zeros_like(self.column_mask),
        )

        return model

    def predict(self, n: int, value: jnp.ndarray) -> "LinearTimeseries":
        """
        Creates a "prediction" instance of self.

        Args:
            n: Number of future predictions.
            value: New start value.

        Returns:
            Returns new instance of :class:`LinearTimeseries`.
        """

        future_model = LinearTimeseries(
            n,
            self.offset,
            self.matrix,
            self.std,
            value,
            std_is_matrix=self._std_is_matrix,
            column_mask=self.column_mask,
        )

        return future_model
