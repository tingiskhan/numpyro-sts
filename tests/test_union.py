from numpyro_sts import AutoRegressive, periodic, LocalLinearTrend
import numpy as np
import jax.random as jrnd


def test_union():
    n = 100

    llt = LocalLinearTrend(n, np.array([0.05, 1e-3]), np.zeros(2))
    ar = AutoRegressive(n, 0.99, 0.05, 1)
    seasonal = periodic.TimeSeasonal(n, 5, 0.05, np.zeros(4))

    combined = llt.union(ar).union(seasonal)

    key = jrnd.PRNGKey(123)
    samples = combined.sample(key)

    log_prob = combined.log_prob(samples)

    assert samples.shape == combined.event_shape
    assert combined.event_shape[-1] == 7


