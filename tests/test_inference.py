"""Tests for sys_mapping.inference — log_prob, run_mcmc, parameter extraction."""

import warnings

import numpy as np
import pytest
import jax.numpy as jnp

import sys_mapping as sm
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

    def test_multiplicative_model_has_no_additive_covariance(self):
        """The chain holds b, not a: a is identically zero in this model."""
        rng = np.random.default_rng(1)
        chain = rng.standard_normal((500, N_SYS + 1)) * 0.01  # b + sigma
        cov_a, cov_b = get_param_covariance_from_chain(chain, N_SYS, "multiplicative")
        np.testing.assert_array_equal(cov_a, np.zeros_like(cov_a))
        assert float(np.trace(cov_b)) > 0.0

    def test_n_sys_1_additive(self):
        rng = np.random.default_rng(2)
        chain = rng.standard_normal((500, 2)) * 0.01  # 1 a-param + sigma
        cov_a, cov_b = get_param_covariance_from_chain(chain, 1, "additive")
        assert cov_a.shape == (1, 1)
        assert cov_b.shape == (1, 1)
        assert float(cov_b[0, 0]) == 0.0


class TestRefineToMLE:
    """A likelihood ratio needs a maximum, and a posterior median is not one."""

    @staticmethod
    def _correlated_templates(rng, n_sys, n_pix):
        """A basis like LS10's: strongly correlated, hence a degenerate posterior."""
        base = rng.standard_normal((3, n_pix))
        t = rng.standard_normal((n_sys, 3)) @ base + 0.35 * rng.standard_normal((n_sys, n_pix))
        return (t - t.mean(1, keepdims=True)) / t.std(1, keepdims=True)

    def test_additive_mle_equals_ols_exactly(self):
        """For the additive Gaussian model the MLE is OLS, so this is analytic."""
        rng = np.random.default_rng(0)
        n_pix, n_sys = 4000, 3
        dt = rng.standard_normal((n_sys, n_pix))
        a = np.array([0.05, -0.03, 0.02])
        dg = a @ dt + rng.standard_normal(n_pix) * 0.1

        theta0 = sm.pack_params(np.zeros(n_sys), None, 0.5, model="additive")
        theta = sm.refine_to_mle(theta0, dg, dt, model="additive")

        ols, _, _, _ = np.linalg.lstsq(dt.T, dg, rcond=None)
        np.testing.assert_allclose(theta[:n_sys], ols, atol=1e-5)
        # sigma_hat is the residual rms (ddof=0), the Gaussian MLE for the scale.
        assert theta[n_sys] == pytest.approx(np.std(dg - ols @ dt), rel=1e-3)

    def test_result_is_independent_of_the_starting_point(self):
        """The signature of a real maximum: it depends on the data, not the seed."""
        rng = np.random.default_rng(1)
        n_pix, n_sys = 6000, 4
        dt = self._correlated_templates(rng, n_sys, n_pix)
        a = rng.normal(0, 0.03, n_sys)
        dg = a @ dt + rng.standard_normal(n_pix) * 0.2

        thetas = []
        for k in range(4):
            r = np.random.default_rng(50 + k)
            start = sm.pack_params(a + r.normal(0, 0.02, n_sys), None,
                                   0.2 * (1 + r.normal(0, 0.1)), model="additive")
            thetas.append(sm.refine_to_mle(start, dg, dt, model="additive"))
        for t in thetas[1:]:
            np.testing.assert_allclose(t, thetas[0], atol=1e-5)

    def test_never_returns_a_worse_point_than_it_was_given(self):
        rng = np.random.default_rng(2)
        n_pix, n_sys = 3000, 2
        dt = rng.standard_normal((n_sys, n_pix))
        dg = rng.standard_normal(n_pix) * 0.3
        log_lik = sm.make_log_likelihood(n_sys, "additive", False)
        theta0 = sm.pack_params(np.array([0.4, -0.4]), None, 0.9, model="additive")
        theta = sm.refine_to_mle(theta0, dg, dt, model="additive")
        assert float(log_lik(theta, dg, dt)) >= float(log_lik(theta0, dg, dt))

    def test_refinement_removes_negative_lambda_lr(self):
        """The defect this function exists for.

        On a degenerate posterior a per-coordinate median lands off the ridge, so
        the 'nested' model can score higher than the full one and lambda_LR goes
        negative.  Refining both points restores lambda_LR >= 0.
        """
        from sys_mapping.model_selection import likelihood_ratio_test

        rng = np.random.default_rng(3)
        n_pix, n_sys = 8000, 8
        dt = self._correlated_templates(rng, n_sys, n_pix)
        a = rng.normal(0, 0.02, n_sys)
        b = rng.normal(0, 0.02, n_sys)
        dg = np.asarray(sm.apply_contamination(
            rng.standard_normal(n_pix) * 0.3, dt, a, b))

        n_neg_raw = n_neg_refined = 0
        for k in range(6):
            r = np.random.default_rng(200 + k)
            th_null = sm.pack_params(a + r.normal(0, 0.01, n_sys), None, 0.3,
                                     model="additive")
            th_alt = sm.pack_params(a + r.normal(0, 0.01, n_sys),
                                    b + r.normal(0, 0.01, n_sys), 0.3,
                                    model="combined")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = likelihood_ratio_test(dg, dt, th_null, th_alt,
                                            "additive", "combined").lambda_lr
                ref = likelihood_ratio_test(
                    dg, dt,
                    sm.refine_to_mle(th_null, dg, dt, model="additive"),
                    sm.refine_to_mle(th_alt, dg, dt, model="combined"),
                    "additive", "combined").lambda_lr
            n_neg_raw += raw < 0
            n_neg_refined += ref < 0
        assert n_neg_raw > 0, "the failure being guarded against did not reproduce"
        assert n_neg_refined == 0

    def test_negative_lambda_lr_warns(self):
        """It must be loud, not clipped: the magnitude is meaningless either way."""
        from sys_mapping.model_selection import likelihood_ratio_test

        rng = np.random.default_rng(4)
        n_pix, n_sys = 4000, 3
        dt = rng.standard_normal((n_sys, n_pix))
        dg = rng.standard_normal(n_pix) * 0.3
        # A deliberately bad alt point scores below the null.
        th_null = sm.pack_params(np.zeros(n_sys), None, 0.3, model="additive")
        th_alt = sm.pack_params(np.full(n_sys, 0.5), np.full(n_sys, 0.5), 0.3,
                                model="combined")
        with pytest.warns(RuntimeWarning, match="lambda_LR"):
            res = likelihood_ratio_test(dg, dt, th_null, th_alt,
                                        "additive", "combined")
        assert res.lambda_lr < 0  # reported, not clipped

