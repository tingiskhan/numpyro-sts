from numpyro.distributions import TransformedDistribution, Normal

from numpyro_sts import AutoRegressive, periodic, LocalLinearTrend
import numpy as np
import jax.random as jrnd


def test_union():
    n = 100

    llt = LocalLinearTrend(np.array([0.05, 1e-3]), np.zeros(2))
    ar = AutoRegressive(0.99, 0.05, 1)
    seasonal = periodic.TimeSeasonal(5, 0.05, np.zeros(4)).deterministic()

    combined = llt.union(ar).union(seasonal)

    key = jrnd.PRNGKey(123)

    base = Normal().expand((n, 3))
    samples = base.sample(key)

    transformed = combined(samples)
    inverse = combined._inverse(transformed)

    assert transformed.shape == (n, 7)
    assert inverse.shape == base.batch_shape

    assert np.allclose(samples, inverse, rtol=1e-5, atol=1e-6)

    distribution = TransformedDistribution(base, combined)

    log_prob = distribution.log_prob(transformed)
    base_prob = base.to_event(2).log_prob(samples)

    assert np.allclose(log_prob, base_prob)
