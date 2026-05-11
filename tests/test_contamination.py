"""Tests for contamination.py: forward model, inversion, two-point correction."""

import numpy as np
import pytest
import jax.numpy as jnp

from sys_mapping.contamination import (
    apply_contamination,
    invert_contamination,
    compute_two_point_correction,
    pack_params,
    unpack_params,
    n_free_params,
)


class TestNFreeParams:
    def test_additive(self):
        assert n_free_params(5, "additive") == 5

    def test_multiplicative(self):
        assert n_free_params(5, "multiplicative") == 5

    def test_combined(self):
        assert n_free_params(5, "combined") == 10

    def test_invalid_model(self):
        with pytest.raises(ValueError):
            n_free_params(5, "unknown")


class TestPackUnpack:
    def test_roundtrip_combined(self):
        n = 3
        a = np.array([0.1, -0.2, 0.3])
        b = np.array([0.05, 0.0, -0.1])
        sigma = 0.15
        theta = pack_params(a, b, sigma, model="combined")
        a2, b2, s2, g2 = unpack_params(jnp.asarray(theta), n, "combined", use_skewed=False)
        np.testing.assert_allclose(np.asarray(a2), a, atol=1e-12)
        np.testing.assert_allclose(np.asarray(b2), b, atol=1e-12)
        np.testing.assert_allclose(float(s2), sigma, atol=1e-12)
        assert g2 is None

    def test_roundtrip_additive(self):
        n = 3
        a = np.array([0.1, -0.2, 0.3])
        sigma = 0.1
        theta = pack_params(a, None, sigma, model="additive")
        a2, b2, s2, _ = unpack_params(jnp.asarray(theta), n, "additive")
        np.testing.assert_allclose(np.asarray(a2), a, atol=1e-12)
        np.testing.assert_allclose(np.asarray(b2), np.zeros(n), atol=1e-12)

    def test_roundtrip_multiplicative(self):
        n = 3
        a = np.array([0.1, -0.2, 0.3])
        sigma = 0.1
        theta = pack_params(a, a, sigma, model="multiplicative")
        a2, b2, _, _ = unpack_params(jnp.asarray(theta), n, "multiplicative")
        np.testing.assert_allclose(np.asarray(a2), a, atol=1e-12)
        np.testing.assert_allclose(np.asarray(b2), a, atol=1e-12)

    def test_skewed_gamma_included(self):
        n = 2
        a = np.array([0.1, 0.2])
        b = np.array([0.0, 0.0])
        theta = pack_params(a, b, 0.1, gamma=1.5, model="combined")
        _, _, s, g = unpack_params(jnp.asarray(theta), n, "combined", use_skewed=True)
        np.testing.assert_allclose(float(g), 1.5, atol=1e-12)


class TestApplyInvertRoundtrip:
    def test_roundtrip_combined(self, delta_g_clean, delta_t, true_a, true_b):
        dg = jnp.asarray(delta_g_clean)
        dt = jnp.asarray(delta_t)
        a = jnp.asarray(true_a)
        b = jnp.asarray(true_b)

        obs = apply_contamination(dg, dt, a, b)
        recovered = invert_contamination(obs, dt, a, b)
        np.testing.assert_allclose(np.asarray(recovered), delta_g_clean, atol=1e-10)

    def test_zero_params_identity(self, delta_g_clean, delta_t):
        n_sys = delta_t.shape[0]
        dg = jnp.asarray(delta_g_clean)
        dt = jnp.asarray(delta_t)
        a = jnp.zeros(n_sys)
        b = jnp.zeros(n_sys)

        obs = apply_contamination(dg, dt, a, b)
        np.testing.assert_allclose(np.asarray(obs), delta_g_clean, atol=1e-12)

    def test_additive_only(self, delta_g_clean, delta_t, true_a):
        n_sys = delta_t.shape[0]
        dg = jnp.asarray(delta_g_clean)
        dt = jnp.asarray(delta_t)
        a = jnp.asarray(true_a)
        b = jnp.zeros(n_sys)

        obs = apply_contamination(dg, dt, a, b)
        recovered = invert_contamination(obs, dt, a, b)
        np.testing.assert_allclose(np.asarray(recovered), delta_g_clean, atol=1e-10)

    def test_multiplicative_only(self, delta_g_clean, delta_t, true_b):
        n_sys = delta_t.shape[0]
        dg = jnp.asarray(delta_g_clean)
        dt = jnp.asarray(delta_t)
        a = jnp.zeros(n_sys)
        b = jnp.asarray(true_b)

        obs = apply_contamination(dg, dt, a, b)
        recovered = invert_contamination(obs, dt, a, b)
        np.testing.assert_allclose(np.asarray(recovered), delta_g_clean, atol=1e-10)


class TestTwoPointCorrection:
    def test_zero_params_no_correction(self):
        n_bins = 10
        n_sys = 3
        w_obs = np.random.default_rng(0).standard_normal(n_bins)
        a_sq = np.zeros(n_sys)
        b_sq = np.zeros(n_sys)
        tcorr = np.ones((n_sys, n_bins))

        w_corr = compute_two_point_correction(
            jnp.asarray(w_obs), jnp.asarray(a_sq), jnp.asarray(b_sq), jnp.asarray(tcorr)
        )
        np.testing.assert_allclose(np.asarray(w_corr), w_obs, atol=1e-12)

    def test_additive_bias_subtracted(self):
        n_bins = 5
        n_sys = 1
        w_obs = np.ones(n_bins)
        a_sq = np.array([0.1])
        b_sq = np.zeros(n_sys)
        tcorr = np.ones((n_sys, n_bins))

        w_corr = compute_two_point_correction(
            jnp.asarray(w_obs), jnp.asarray(a_sq), jnp.asarray(b_sq), jnp.asarray(tcorr)
        )
        np.testing.assert_allclose(np.asarray(w_corr), np.full(n_bins, 0.9), atol=1e-12)

    def test_multiplicative_bias_divided(self):
        n_bins = 5
        n_sys = 1
        w_obs = np.ones(n_bins)
        a_sq = np.zeros(n_sys)
        b_sq = np.array([0.5])
        tcorr = np.ones((n_sys, n_bins))

        w_corr = compute_two_point_correction(
            jnp.asarray(w_obs), jnp.asarray(a_sq), jnp.asarray(b_sq), jnp.asarray(tcorr)
        )
        np.testing.assert_allclose(np.asarray(w_corr), np.full(n_bins, 1.0 / 1.5), atol=1e-12)
