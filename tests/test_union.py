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

    distribution = TransformedDistribution(
        Normal().expand((100, 3)),
        combined,
    )

    samples = distribution.sample(key)
    log_prob = distribution.log_prob(samples)

    assert samples.shape == combined.event_shape
    assert combined.event_shape[-1] == 7
