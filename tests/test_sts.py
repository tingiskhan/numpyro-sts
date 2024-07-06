import numpyro
import pytest as pt
from jax.random import PRNGKey
import numpy as np
from numpyro.distributions import HalfNormal, Normal
from numpyro.infer import MCMC, NUTS

from numpyro_sts import RandomWalk, LocalLinearTrend, AutoRegressive, LinearTimeseries, SmoothLocalLinearTrend, periodic


numpyro.set_platform("cpu")


def models(n):
    for b in [(), (5, 10)]:
        yield RandomWalk(n, 0.05, 0.0, validate_args=True).expand(b)
        yield LocalLinearTrend(n, np.array([0.05, 1e-3]), np.zeros(2), validate_args=True).expand(b)
        yield AutoRegressive(n, 0.99, 0.05, 1, validate_args=True).expand(b)
        yield AutoRegressive(n, np.array([0.99, -0.5]), 0.05, 2, validate_args=True).expand(b)
        yield AutoRegressive(n, np.array([0.99, -0.5]), 0.05, 2, 0.5).expand(b)
        yield periodic.TimeSeasonal(n, 5, 0.05, np.zeros(4)).expand(b)
        yield periodic.Cyclical(n, 2.0 * np.pi / (n // 2), 0.05, np.zeros(2)).expand(b)

        mat = np.array([
            [0.95, -0.05],
            [1.0, 0.0]
        ])
        std_mat = np.array([0.05, 0.0])
        offset = np.zeros_like(std_mat)
        mask = std_mat != 0.0

        yield LinearTimeseries(n, offset, mat, std_mat, np.zeros_like(std_mat), column_mask=mask).expand(b)
        yield SmoothLocalLinearTrend(n, 0.01, np.zeros(2)).expand(b)

        mat = np.eye(2)
        std_mat = 0.05 * np.eye(2)
        offset = np.zeros(2)

        yield LinearTimeseries(n, offset, mat, std_mat, offset, std_is_matrix=True).expand(b)

    yield RandomWalk(n, np.full(10, 0.05), 0.0, validate_args=True)


N = 100


@pt.mark.parametrize("model", models(N))
@pt.mark.parametrize("shape", [(), (10,)])
def test_models(model, shape):
    key = PRNGKey(123)

    samples = model.sample(key, shape)
    assert samples.shape == shape + model.batch_shape + model.event_shape

    log_prob = model.log_prob(samples)
    assert log_prob.shape == shape + model.batch_shape


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



@pt.mark.parametrize("shape", [(), (10,)])
def test_constant_model(shape):
    mat = np.eye(2)
    offset = np.ones(2)
    initial_value = np.zeros_like(offset)
    mask = np.zeros_like(offset, dtype=bool)

    model = LinearTimeseries(N, offset, mat, np.zeros_like(offset), initial_value, column_mask=mask)

    key = PRNGKey(123)
    samples = model.sample(key, shape)

    assert (samples[..., -1, :] == N).all()

    log_prob = model.log_prob(samples)
    assert (log_prob == 0.0).all()
