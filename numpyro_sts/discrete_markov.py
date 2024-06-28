from jax.typing import ArrayLike
import jax.numpy as jnp
import jax.random as jrnd
from numpyro.contrib.control_flow import scan
from numpyro.distributions import Distribution, constraints, Categorical
import jax.scipy as jsc

from .util import cast_to_tensor


def _find_stationary(p: ArrayLike, as_logit: bool = False) -> jnp.ndarray:
    """
    Finds the stationary distribution of a Markov chain.

    Args:
        p: Transition matrix.
        as_logit: Whether to return logit version.
    """

    n = p.shape[0]

    eye = jnp.eye(n)
    a_prime = (eye - p).swapaxes(-1, -2)

    sum_row = jnp.ones([1, n])
    a = jnp.concatenate([a_prime, sum_row], axis=0)

    b_prime = jnp.zeros(n)
    b = jnp.concatenate([b_prime, jnp.ones(1)], axis=0)

    probs = jnp.linalg.lstsq(a, b)[0]

    if not as_logit:
        return probs

    return jsc.special.logit(probs)


class DiscreteMarkovChain(Distribution):
    """
    Implements a Discrete Markov Chain.

    Args:
        n: Number of time samples to sample.
        transition_matrix: Transition matrix at each time step.
    """

    pytree_aux_fields = ("n",)
    pytree_data_fields = ("transition_matrix", "initial_value")

    has_enumerate_support = True
    support = constraints.positive_integer

    arg_constraints = {
        "n": constraints.positive_integer,
        "transition_matrix": constraints.greater_than_eq(0.0),
    }

    # TODO: set event shape etc.
    def __init__(self, n: int, transition_matrix: ArrayLike, inital_value: ArrayLike, *, validate_args: bool = False):
        assert transition_matrix.shape[0] == transition_matrix.shape[1], "The transition matrix must be square!"

        self.n = n
        self.transition_matrix, self.initial_value = cast_to_tensor(transition_matrix, inital_value)

        batch_shape = self.transition_matrix.shape[:-2]
        super().__init__(batch_shape=batch_shape, event_shape=(self.n,), validate_args=validate_args)

    @staticmethod
    def get_stationary_distribution(transition_matrix: ArrayLike, as_logit: bool = False) -> jnp.ndarray:
        """
        Gets the stationary distribution for a given transition matrix.

        Args:
            transition_matrix: Transition matrix of the Markov chain.
            as_logit: Whether to return probabilities on logit form or not.

        Returns:
            Returns a vector of stationary probabilities.
        """

        return _find_stationary(transition_matrix, as_logit=as_logit)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        as_logits = jsc.special.logit(self.transition_matrix)

        def body_fn(state_t, _):
            x_t, key_t = state_t
            transition_probabilities = as_logits[..., x_t, :]

            key_tp1, _ = jrnd.split(key_t)
            x_tp1 = jrnd.categorical(key_tp1, transition_probabilities, shape=shape)

            return (x_tp1, key_tp1), x_tp1

        initial_value = jnp.broadcast_to(self.initial_value, sample_shape)
        _, x = scan(body_fn, (initial_value, key), jnp.arange(self.n))

        return jnp.moveaxis(x, 0, -1)

    def log_prob(self, value):
        sample_shape = jnp.broadcast_shapes(value.shape[: -self.event_dim], self.batch_shape)

        initial_value = jnp.broadcast_to(self.initial_value, sample_shape)
        value = jnp.concatenate([initial_value[..., None], value], axis=-1)

        x_tm1 = value[..., :-1]
        x_t = value[..., 1:]

        transition_probabilities = jnp.broadcast_to(
            self.transition_matrix, sample_shape + self.transition_matrix.shape[-2:]
        )

        # TODO: might have to use take along dim?
        selected_transition_probabilities = jnp.take_along_axis(transition_probabilities, x_tm1[..., None], axis=-2)

        return Categorical(probs=selected_transition_probabilities).to_event(self.event_dim).log_prob(x_t)

    def enumerate_support(self, expand=True):
        values = jnp.arange(self.transition_matrix.shape[-1])[..., None]

        shape = jnp.broadcast_shapes(values.shape, self.event_shape)
        values = jnp.broadcast_to(values, shape)

        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)

        return values