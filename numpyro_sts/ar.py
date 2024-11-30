import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro.distributions import TransformedDistribution, Normal

from .base import LinearTimeseries
from .util import cast_to_tensor


class AutoRegressive(TransformedDistribution):
    """
    Implements an auto regressive process.
    """

    def __init__(
        self,
        num_steps: int,
        phi: ArrayLike,
        std: ArrayLike,
        mu: ArrayLike = None,
        initial_value: ArrayLike = None,
    ):
        std, phi, mu = cast_to_tensor(std, phi, mu if mu is not None else jnp.zeros_like(std))

        batch_shape = jnp.broadcast_shapes(std.shape, phi.shape[:-1])

        phi = jnp.reshape(phi, batch_shape + (1, 1))
        std = jnp.reshape(std, batch_shape + (1,))
        mu = jnp.reshape(mu, batch_shape + (1,))

        offset = mu * (1.0 - phi.squeeze(-1))

        init = jnp.reshape(initial_value if initial_value is not None else jnp.zeros(1), batch_shape + (1,))
        transform = LinearTimeseries(offset, phi, std, init)

        base_dist = Normal().expand((num_steps, 1))
        super().__init__(base_dist, transform)
