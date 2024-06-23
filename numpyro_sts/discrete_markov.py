from jax.typing import ArrayLike
import jax.numpy as jnp
from numpyro.distributions import Distribution, constraints, Categorical


def _find_stationary(p: ArrayLike) -> jnp.ndarray:
    """
    Finds the stationary distribution of a Markov chain.

    Args:
        p: Transition matrix.
    """

    n = p.shape[0]

    eye = jnp.eye(n)
    a_prime = (eye - p).swapaxes(-1, -2)

    sum_row = jnp.ones([1, n])
    a = jnp.concatenate([a_prime, sum_row], axis=0)

    b_prime = jnp.zeros(n)
    b = jnp.concatenate([b_prime, jnp.ones(1)], axis=0)

    return jnp.linalg.lstsq(a, b)[0]


class DiscreteMarkovChain(Distribution):
    """
    Implements a Discrete Markov Chain.

    Args:
        n: Number of time samples to sample.
        transition_matrix: Transition matrix at each time step.
    """

    pytree_aux_fields = ("n",)
    pytree_data_fields = ("transition_matrix", "stationary_distribution")

    has_enumerate_support = True
    arg_constraints = {
        "n": constraints.positive_integer,
        "transition_matrix": constraints.greater_than_eq(0.0),
    }

    def __init__(self, n: int, transition_matrix: ArrayLike, *, validate_args: bool = False):
        super().__init__(validate_args=validate_args)

        assert transition_matrix.shape[0] == transition_matrix.shape[1], "The transition matrix must be square!"

        self.n = n
        self.transition_matrix = transition_matrix
        self.stationary_distribution = _find_stationary(self.transition_matrix)

    def sample(self, key, sample_shape=()):
        initial_state = Categorical(probs=self.stationary_distribution).sample(key, sample_shape=sample_shape)

        return initial_state


    def log_prob(self, value):
        pass
