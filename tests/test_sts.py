import numpyro
import pytest as pt
from jax.random import PRNGKey
import numpy as np
from numpyro.distributions import HalfNormal, Normal, TransformedDistribution
from numpyro.infer import MCMC, NUTS

from numpyro_sts import RandomWalk, LocalLinearTrend, AutoRegressive, LinearTimeseries, SmoothLocalLinearTrend, periodic


numpyro.set_platform("cpu")


def models():
    yield SmoothLocalLinearTrend(0.05, np.zeros(2))
    yield RandomWalk(0.05, 0.0)
    yield AutoRegressive(0.95, 0.05, 1, 0.1)


N = 100


@pt.mark.parametrize("transform", models())
@pt.mark.parametrize("shape", [(), (10,)])
def test_models(transform, shape):
    key = PRNGKey(123)

    dist = TransformedDistribution(
        Normal().expand((100, 1)),
        transform,
    )

    samples = dist.sample(key)

    # NB: this should be fetched from the predictive distribution rather...
    assert samples["std"].std() <= 1.0


def test_models_numpyro_context():
    key = PRNGKey(123)

    true_model = AutoRegressive(500, 0.99, 0.05, 1)
    y = true_model.sample(key)

    def numpyro_model(n, y_):
        phi = numpyro.sample("phi", Normal())
        std = numpyro.sample("std", HalfNormal())

        x = numpyro.sample("x", AutoRegressive(n, phi, std, order=1), obs=y_)

    kernel = NUTS(numpyro_model)
    mcmc = MCMC(kernel, num_samples=500, num_warmup=1_000)
    mcmc.run(key, y.shape[0], y_=y)

    samples = mcmc.get_samples()
    quantiles = np.quantile(samples["std"], [0.001, 0.999])

    assert (quantiles[0] <= true_model.std <= quantiles[1]).all()