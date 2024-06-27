import numpy as np
import numpyro
from jax._src.random import PRNGKey
from numpyro.distributions import Dirichlet, Categorical, HalfNormal, Normal
import jax.numpy as jnp
from numpyro.infer import Predictive, NUTS, MCMC

from numpyro_sts import DiscreteMarkovChain


def markov_switching_regime(n: int, x: np.ndarray = None):
    # markov
    with numpyro.plate("num_states", 2):
        transition_matrix = numpyro.sample("transition_matrix", Dirichlet(np.ones(2)))

    initial_dist = DiscreteMarkovChain.get_stationary_distribution(transition_matrix)
    initial_value = numpyro.sample("v_0", Categorical(initial_dist))

    markov_model = DiscreteMarkovChain(n, transition_matrix, initial_value)
    state = numpyro.sample("x", markov_model, infer={"enumerate": "parallel"})

    # obs
    good = numpyro.sample("good", HalfNormal())
    bad = -numpyro.sample("bad", HalfNormal())

    mu = jnp.stack([bad, good])
    sigma = numpyro.sample("sigma", HalfNormal())

    with numpyro.plate("num_time", n):
        y = numpyro.sample("y", Normal(mu[state], sigma), obs=x)



def test_markov():
    predictive = Predictive(markov_switching_regime, num_samples=1)
    key = PRNGKey(0)
    samples = predictive(key, 500)

    y = np.array(samples["y"].squeeze(0))

    kernel = NUTS(markov_switching_regime)
    mcmc = MCMC(kernel, num_samples=1_000, num_warmup=2_000)
    mcmc.run(key, y.shape[0], y)

    # TODO: test values