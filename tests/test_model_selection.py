"""Tests for model_selection.py: likelihood ratio test."""

import numpy as np
import pytest

from sys_mapping.model_selection import likelihood_ratio_test
from sys_mapping.contamination import pack_params


N_PIX = 300
N_SYS = 3


def _make_data(seed=0):
    rng = np.random.default_rng(seed)
    delta_t = rng.standard_normal((N_SYS, N_PIX))
    delta_t -= delta_t.mean(axis=1, keepdims=True)
    delta_g_obs = rng.standard_normal(N_PIX) * 0.1
    return delta_g_obs, delta_t


class TestLikelihoodRatioTest:
    def test_identical_params_lambda_zero(self):
        """When null and alt are at the same point (embedded), λ_LR >= 0."""
        dg, dt = _make_data()
        sigma = 0.1
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)

        # Additive null embedded in combined alt: b=0 in combined
        theta_null = pack_params(a, None, sigma, model="additive")
        theta_alt = pack_params(a, b, sigma, model="combined")

        result = likelihood_ratio_test(dg, dt, theta_null, theta_alt, "additive", "combined")
        assert result.lambda_lr >= -1e-6  # should be >= 0

    def test_better_fit_gives_positive_lambda(self):
        """Combined model at true params should have higher likelihood than additive null."""
        rng = np.random.default_rng(1)
        a_true = np.array([0.15, -0.1, 0.12])
        b_true = np.array([0.2, -0.15, 0.1])
        sigma = 0.05
        delta_t = rng.standard_normal((N_SYS, N_PIX))
        delta_t -= delta_t.mean(axis=1, keepdims=True)

        from sys_mapping.contamination import apply_contamination
        import jax.numpy as jnp
        dg_clean = rng.standard_normal(N_PIX) * 0.02
        dg_obs = np.asarray(
            apply_contamination(
                jnp.asarray(dg_clean), jnp.asarray(delta_t),
                jnp.asarray(a_true), jnp.asarray(b_true),
            )
        )

        theta_null = pack_params(a_true, None, sigma, model="additive")
        theta_alt = pack_params(a_true, b_true, sigma, model="combined")

        result = likelihood_ratio_test(dg_obs, delta_t, theta_null, theta_alt, "additive", "combined")
        assert result.lambda_lr > 0

    def test_degrees_of_freedom_additive_vs_combined(self):
        """Combined has N_SYS extra params over additive, so r = N_SYS."""
        dg, dt = _make_data()
        sigma = 0.1
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)
        theta_null = pack_params(a, None, sigma, model="additive")
        theta_alt = pack_params(a, b, sigma, model="combined")

        result = likelihood_ratio_test(dg, dt, theta_null, theta_alt, "additive", "combined")
        assert result.n_dof == N_SYS

    def test_degrees_of_freedom_mult_vs_combined(self):
        """Combined has N_SYS extra params over multiplicative, so r = N_SYS."""
        dg, dt = _make_data()
        sigma = 0.1
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)
        theta_null = pack_params(a, a, sigma, model="multiplicative")
        theta_alt = pack_params(a, b, sigma, model="combined")

        result = likelihood_ratio_test(dg, dt, theta_null, theta_alt, "multiplicative", "combined")
        assert result.n_dof == N_SYS

    def test_invalid_direction_raises(self):
        """Null more complex than alt should raise ValueError."""
        dg, dt = _make_data()
        sigma = 0.1
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)
        theta_null = pack_params(a, b, sigma, model="combined")
        theta_alt = pack_params(a, None, sigma, model="additive")

        with pytest.raises(ValueError):
            likelihood_ratio_test(dg, dt, theta_null, theta_alt, "combined", "additive")

    def test_p_value_range(self):
        dg, dt = _make_data()
        sigma = 0.1
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)
        theta_null = pack_params(a, None, sigma, model="additive")
        theta_alt = pack_params(a, b, sigma, model="combined")

        result = likelihood_ratio_test(dg, dt, theta_null, theta_alt, "additive", "combined")
        assert 0.0 <= result.p_value <= 1.0

    def test_reject_null_field_consistent(self):
        dg, dt = _make_data()
        sigma = 0.1
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)
        theta_null = pack_params(a, None, sigma, model="additive")
        theta_alt = pack_params(a, b, sigma, model="combined")

        result = likelihood_ratio_test(dg, dt, theta_null, theta_alt, "additive", "combined",
                                       significance=0.05)
        assert result.reject_null == (result.p_value < 0.05)
