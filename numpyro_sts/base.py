from numbers import Number
from typing import Union

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.random import normal
from numpyro.contrib.control_flow import scan
from numpyro.distributions import Distribution, Normal, constraints, MultivariateNormal
from numpyro.distributions.util import validate_sample
from numpyro.util import is_prng_key


ArrayLike = Union[jnp.ndarray, Number, np.ndarray]


def _broadcast_and_reshape(x: jnp.ndarray, shape, dim: int) -> jnp.ndarray:
    last_dims = x.shape[dim:]
    return jnp.broadcast_to(x, shape + last_dims).reshape((-1,) + last_dims)


def _loc_transition(state, offset, matrix) -> jnp.ndarray:
    return offset + (matrix @ state[..., None]).reshape(state.shape)


# TODO: Make it so that if you pass mask, the shock sampling is handled automatically
class LinearTimeseries(Distribution):
    r"""
    Defines a base model for linear stochastic models with Gaussian increments.

    Args:
        offset: Constant offset in transition equation. Of size :math:`[batch size] \times dimension`.
        matrix: Matrix of linear combination of states. Of size :math:`[batch size] \times dimension \times dimension`.
        std: Standard deviation of innovations. Of size :math:`[batch size] \times dimension`.
        initial_value: Initial value of the time series. Of size :math:`[batch size] \times dimension`.
        mask: Mask for removing specific columns from calculation.
    """

    pytree_data_fields = ("offset", "matrix", "std", "initial_value", "mask")
    pytree_aux_fields = ("n", "_sample_shape", "_column_mask", "_std_is_matrix")

    support = constraints.real_matrix
    has_enumerate_support = False

    arg_constraints = {
        "offset": constraints.real_vector,
        "matrix": constraints.real_matrix,
        "std": constraints.real_vector,
        "initial_value": constraints.real_vector,
        "n": constraints.positive_integer,
    }

    @staticmethod
    def _verify_parameters(offset, matrix, std, initial_value, std_is_matrix):
        ndim = matrix.shape[-1]

        assert initial_value.ndim >= 1
        assert matrix.ndim >= 2 and matrix.shape[-2] == matrix.shape[-1] == ndim

        if std_is_matrix:
            assert std.ndim >= 2 and std.shape[-1] == std.shape[-2] == ndim

    def __init__(
        self,
        n: int,
        offset: ArrayLike,
        matrix: ArrayLike,
        std: ArrayLike,
        initial_value: ArrayLike,
        *,
        std_is_matrix: bool = False,
        mask: ArrayLike = None,
        validate_args=None,
    ):
        self._verify_parameters(offset, matrix, std, initial_value, std_is_matrix)
        times = jnp.arange(n)

        self._std_is_matrix = std_is_matrix

        event_shape = times.shape + initial_value.shape[-1:]
        batch_shape = jnp.broadcast_shapes(
            offset.shape[:-1], matrix.shape[:-2], std.shape[:-(1 + int(self._std_is_matrix))], initial_value.shape[:-1]
        )

        parameter_shape = batch_shape + initial_value.shape[-1:]

        self.n = n
        self.offset = jnp.broadcast_to(offset, parameter_shape)
        self.initial_value = jnp.broadcast_to(initial_value, parameter_shape)
        self.matrix = jnp.broadcast_to(matrix, parameter_shape + initial_value.shape[-1:])

        std_shape = parameter_shape if not self._std_is_matrix else parameter_shape + initial_value.shape[-1:]
        self.std = jnp.broadcast_to(std, std_shape)

        cols_to_sample = event_shape[-1]
        if mask is not None:
            assert mask.shape == event_shape[-1:], "Shapes not congruent!"
            cols_to_sample = mask.sum(axis=-1)

        self._column_mask = mask
        self._sample_shape = times.shape + (cols_to_sample,)

        super().__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)

    def _sample_shocks(self, key, batch_shape) -> jnp.ndarray:
        samples = normal(key, shape=batch_shape + self._sample_shape)

        if self._column_mask is None:
            return samples

        result = jnp.zeros(batch_shape + self.event_shape, dtype=samples.dtype)
        return result.at[..., self._column_mask].set(samples)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)

        batch_shape = sample_shape + self.batch_shape

        def body(state, xs):
            (eps_tp1,) = xs
            x_tp1 = self.sample_from_shock(state, eps_tp1)

            return x_tp1, x_tp1

        def scan_fn(init, noise):
            return scan(body, init, (noise,))

        eps = self._sample_shocks(key, batch_shape)
        inits = jnp.broadcast_to(self.initial_value, sample_shape + self.initial_value.shape)

        batch_dim = len(batch_shape)
        if batch_dim:
            eps = jnp.moveaxis(eps, -2, 0)
            _, samples = scan_fn(inits, eps)

            return jnp.moveaxis(samples, 0, -2)

        return scan_fn(inits, eps)[-1]

    @validate_sample
    def log_prob(self, value):
        # TODO: Consider passing initial distribution instead of value as this is kinda tricky...
        # NB: Note that sending initial distribution will also be tricky for an AR process as well...
        sample_shape = jnp.broadcast_shapes(value.shape[: -self.event_dim], self.batch_shape)
        value = jnp.broadcast_to(value, sample_shape + self.event_shape)

        initial_value = jnp.expand_dims(self.initial_value, -2)
        initial_value = jnp.broadcast_to(initial_value, sample_shape + initial_value.shape[-2:])

        stacked = jnp.concatenate([initial_value, value], axis=-2)

        if sample_shape:
            stacked_reshape = stacked.reshape((-1,) + stacked.shape[-2:])
            x_tm1 = stacked_reshape[:, :-1]

            offset = _broadcast_and_reshape(self.offset, sample_shape, -1)
            matrix = _broadcast_and_reshape(self.matrix, sample_shape, -2)

            loc = vmap(_loc_transition)(x_tm1, offset, matrix).reshape(sample_shape + self.event_shape)
        else:
            x_tm1 = stacked[:-1]
            loc = _loc_transition(x_tm1, self.offset, self.matrix)

        x_t = stacked[..., 1:, :]

        std = self.std
        if not self._std_is_matrix:
            std = jnp.expand_dims(std, -2)

        if self._column_mask is not None:
            loc = loc[..., self._column_mask]
            std = std[..., self._column_mask]

            if self._std_is_matrix:
                std = std[..., self._column_mask, :]

            x_t = x_t[..., self._column_mask]

        # NB: Could also use event shapes
        if not self._std_is_matrix:
            dist = Normal(loc, std).to_event(1)
        else:
            dist = MultivariateNormal(loc, scale_tril=std)

        return dist.log_prob(x_t).sum(axis=-1)

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
