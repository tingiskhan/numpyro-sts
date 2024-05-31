import jax.numpy as jnp
import numpy as np

from .base import ArrayLike, LinearTimeseries
from .util import cast_to_tensor


class AutoRegressive(LinearTimeseries):
    """
    Implements an auto regressive process.
    """

    def __init__(
        self,
        n: int,
        phi: ArrayLike,
        std: ArrayLike,
        order: int,
        mu: ArrayLike = None,
        initial_value: ArrayLike = None,
        **kwargs,
    ):
        std, phi, mu = cast_to_tensor(std, phi, mu if mu is not None else jnp.zeros_like(std))

        batch_shape = jnp.broadcast_shapes(std.shape, phi.shape[:-1])

        phi = jnp.reshape(phi, batch_shape + (1, order))
        std = jnp.reshape(std, batch_shape + (1,))
        mu = jnp.reshape(mu, batch_shape + (1,))

        if order > 1:
            phi = jnp.concatenate([phi, jnp.eye(order - 1, order)], axis=-2)

            offset = mu * (1.0 - phi.sum(axis=-1))

            zeros = jnp.zeros(batch_shape + (order - 1,))
            offset = jnp.concatenate([offset, zeros], axis=-1)
            std = jnp.concatenate([std, zeros], axis=-1)
        else:
            offset = mu * (1.0 - phi.squeeze(-1))

        init = jnp.reshape(initial_value if initial_value is not None else jnp.zeros(order), batch_shape + (order,))

        mask = np.array([True] + (order - 1) * [False], dtype=jnp.bool_)

        super().__init__(n, offset, phi, std, init, mask=mask, **kwargs)
