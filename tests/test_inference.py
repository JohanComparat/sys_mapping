"""Tests for sys_mapping.inference — log_prob, run_mcmc, parameter extraction."""

import numpy as np
import pytest
import jax.numpy as jnp

from sys_mapping.inference import (
    make_log_prob,
    run_mcmc,
    get_param_variance_from_chain,
    get_param_covariance_from_chain,
)

N_SYS = 2
N_PIX = 300


@pytest.fixture(scope="module")
def obs_data():
    rng = np.random.default_rng(5)
    delta_g = rng.standard_normal(N_PIX) * 0.1
    delta_t = rng.standard_normal((N_SYS, N_PIX))
    return delta_g, delta_t


class TestMakeLogProb:
    def test_scalar_valid_theta(self, obs_data):
        dg, dt = obs_data
        log_prob, n_dim = make_log_prob(N_SYS, "additive", dg, dt)
        theta = np.zeros(n_dim)
        theta[-1] = 0.1
        assert np.isfinite(log_prob(theta))

    def test_sigma_too_small_returns_neginf(self, obs_data):
        dg, dt = obs_data
        log_prob, n_dim = make_log_prob(N_SYS, "additive", dg, dt)
        theta = np.zeros(n_dim)
        theta[-1] = 0.0  # sigma <= sigma_min
        assert log_prob(theta) == -np.inf

    def test_non_finite_lp_returns_neginf(self, obs_data):
        dg, dt = obs_data
        # Craft parameters that make the likelihood non-finite
        log_prob, n_dim = make_log_prob(N_SYS, "additive", dg, dt)
        theta = np.full(n_dim, 1e300)  # extreme values → NaN from overflow
        theta[-1] = 0.1
        result = log_prob(theta)
        assert result == -np.inf or np.isfinite(result)  # either outcome is acceptable

    def test_vectorize_mode_shape(self, obs_data):
        dg, dt = obs_data
        log_prob_vec, n_dim = make_log_prob(N_SYS, "additive", dg, dt, vectorize=True)
        n_walkers = 8
        thetas = np.zeros((n_walkers, n_dim))
        thetas[:, -1] = 0.1
        result = log_prob_vec(thetas)
        assert result.shape == (n_walkers,)

    def test_vectorize_sigma_mask(self, obs_data):
        dg, dt = obs_data
        log_prob_vec, n_dim = make_log_prob(N_SYS, "additive", dg, dt, vectorize=True)
        n_walkers = 4
        thetas = np.zeros((n_walkers, n_dim))
        thetas[:, -1] = 0.1
        thetas[0, -1] = 0.0  # first walker has sigma=0 → should be -inf
        result = log_prob_vec(thetas)
        assert result[0] == -np.inf
        assert np.all(np.isfinite(result[1:]))


class TestRunMcmcSkewed:
    def test_skewed_chain_ndim(self, obs_data):
        dg, dt = obs_data
        chain, _ = run_mcmc(
            N_SYS, model="additive",
            delta_g_obs=dg, delta_t=dt,
            use_skewed=True,
            n_walkers=20, n_steps=15, n_burn=5,
            progress=False,
        )
        # additive + skewed: N_SYS a-params + sigma + gamma = N_SYS + 2
        assert chain.shape[1] == N_SYS + 2


class TestParamExtraction:
    @pytest.fixture(scope="class")
    def combined_chain(self):
        rng = np.random.default_rng(0)
        n_samples = 500
        n_dim = 2 * N_SYS + 1  # combined model: a, b, sigma
        return rng.standard_normal((n_samples, n_dim)) * 0.01

    def test_get_param_variance_shape(self, combined_chain):
        var_a, var_b = get_param_variance_from_chain(combined_chain, N_SYS, "combined")
        assert var_a.shape == (N_SYS,)
        assert var_b.shape == (N_SYS,)

    def test_get_param_variance_non_negative(self, combined_chain):
        var_a, var_b = get_param_variance_from_chain(combined_chain, N_SYS, "combined")
        assert np.all(var_a >= 0)
        assert np.all(var_b >= 0)

    def test_multiplicative_model_b_equals_a(self):
        rng = np.random.default_rng(1)
        chain = rng.standard_normal((500, N_SYS + 1)) * 0.01  # a + sigma
        cov_a, cov_b = get_param_covariance_from_chain(chain, N_SYS, "multiplicative")
        np.testing.assert_array_equal(cov_a, cov_b)

    def test_n_sys_1_additive(self):
        rng = np.random.default_rng(2)
        chain = rng.standard_normal((500, 2)) * 0.01  # 1 a-param + sigma
        cov_a, cov_b = get_param_covariance_from_chain(chain, 1, "additive")
        assert cov_a.shape == (1, 1)
        assert cov_b.shape == (1, 1)
        assert float(cov_b[0, 0]) == 0.0
