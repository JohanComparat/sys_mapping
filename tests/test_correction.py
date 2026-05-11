"""Tests for correction.py: debiasing, template rotation, two-point correction."""

import numpy as np
import pytest

from sys_mapping.correction import (
    debias_params,
    rotate_templates,
    transform_params_from_rotated,
    correct_two_point_function,
)


class TestDebiasParams:
    def test_zero_variance_no_change(self):
        a = np.array([0.1, -0.2, 0.3])
        b = np.array([0.05, 0.1, -0.1])
        var_a = np.zeros(3)
        var_b = np.zeros(3)
        a_sq, b_sq = debias_params(a, b, var_a, var_b)
        np.testing.assert_allclose(a_sq, a**2, atol=1e-12)
        np.testing.assert_allclose(b_sq, b**2, atol=1e-12)

    def test_large_variance_clips_to_zero(self):
        a = np.array([0.1])
        b = np.array([0.1])
        var_a = np.array([1.0])  # larger than a^2
        var_b = np.array([1.0])
        a_sq, b_sq = debias_params(a, b, var_a, var_b)
        assert a_sq[0] == 0.0
        assert b_sq[0] == 0.0

    def test_partial_debiasing(self):
        a = np.array([0.4])
        b = np.array([0.5])
        var_a = np.array([0.1])  # a^2 = 0.16, debiased = 0.06
        var_b = np.array([0.2])  # b^2 = 0.25, debiased = 0.05
        a_sq, b_sq = debias_params(a, b, var_a, var_b)
        np.testing.assert_allclose(a_sq[0], 0.16 - 0.1, atol=1e-12)
        np.testing.assert_allclose(b_sq[0], 0.25 - 0.2, atol=1e-12)

    def test_non_negative_output(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 0.1, 10)
        var_a = rng.uniform(0, 0.05, 10)
        a_sq, _ = debias_params(a, a, var_a, var_a)
        assert np.all(a_sq >= 0)


class TestRotateTemplates:
    def test_orthogonality_of_rotated_templates(self):
        rng = np.random.default_rng(1)
        n_sys, n_pix = 4, 500
        delta_t = rng.standard_normal((n_sys, n_pix))
        delta_t_rot, R, eigvals = rotate_templates(delta_t)

        # Covariance of rotated templates should be diagonal
        cov_rot = delta_t_rot @ delta_t_rot.T / n_pix
        off_diag = cov_rot - np.diag(np.diag(cov_rot))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-10)

    def test_rotation_matrix_is_orthogonal(self):
        rng = np.random.default_rng(2)
        delta_t = rng.standard_normal((3, 300))
        _, R, _ = rotate_templates(delta_t)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_variance_preserved(self):
        """Total variance is preserved under orthogonal rotation."""
        rng = np.random.default_rng(3)
        delta_t = rng.standard_normal((3, 500))
        delta_t_rot, _, eigvals = rotate_templates(delta_t)
        var_orig = np.sum(np.var(delta_t, axis=1))
        var_rot = np.sum(np.var(delta_t_rot, axis=1))
        np.testing.assert_allclose(var_rot, var_orig, rtol=1e-10)

    def test_eigenvalues_descending(self):
        rng = np.random.default_rng(4)
        delta_t = rng.standard_normal((5, 300))
        _, _, eigvals = rotate_templates(delta_t)
        assert np.all(np.diff(eigvals) <= 0), "Eigenvalues not in descending order"

    def test_roundtrip_params(self):
        rng = np.random.default_rng(5)
        n_sys, n_pix = 3, 400
        delta_t = rng.standard_normal((n_sys, n_pix))
        a_orig = rng.standard_normal(n_sys) * 0.1
        b_orig = rng.standard_normal(n_sys) * 0.05

        delta_t_rot, R, _ = rotate_templates(delta_t)
        a_rot = R @ a_orig
        b_rot = R @ b_orig
        a_back, b_back = transform_params_from_rotated(a_rot, b_rot, R)

        np.testing.assert_allclose(a_back, a_orig, atol=1e-12)
        np.testing.assert_allclose(b_back, b_orig, atol=1e-12)


class TestCorrectTwoPoint:
    def test_zero_params_identity(self):
        n_bins, n_sys = 8, 3
        rng = np.random.default_rng(0)
        w_obs = rng.standard_normal(n_bins)
        a_hat = np.zeros(n_sys)
        b_hat = np.zeros(n_sys)
        var_a = np.zeros(n_sys)
        var_b = np.zeros(n_sys)
        tcorr = np.ones((n_sys, n_bins))

        w_corr = correct_two_point_function(w_obs, a_hat, b_hat, var_a, var_b, tcorr)
        np.testing.assert_allclose(w_corr, w_obs, atol=1e-12)

    def test_additive_subtracted(self):
        n_bins, n_sys = 5, 1
        w_obs = np.ones(n_bins)
        a_hat = np.array([0.5])   # a^2 = 0.25
        b_hat = np.zeros(1)
        var_a = np.zeros(1)
        var_b = np.zeros(1)
        tcorr = np.ones((1, n_bins))

        w_corr = correct_two_point_function(w_obs, a_hat, b_hat, var_a, var_b, tcorr)
        np.testing.assert_allclose(w_corr, np.full(n_bins, 1 - 0.25), atol=1e-12)

    def test_debiasing_affects_output(self):
        """With variance = a^2, debiased a_sq = 0, so no correction applied."""
        n_bins, n_sys = 5, 1
        a_hat = np.array([0.5])
        b_hat = np.zeros(1)
        var_a = np.array([0.25])  # equals a^2 → debias to 0
        var_b = np.zeros(1)
        w_obs = np.ones(n_bins)
        tcorr = np.ones((1, n_bins))

        w_corr = correct_two_point_function(w_obs, a_hat, b_hat, var_a, var_b, tcorr)
        np.testing.assert_allclose(w_corr, w_obs, atol=1e-12)


class TestCorrectPowerSpectrumHarmonic:
    def test_output_shape(self):
        from sys_mapping.correction import correct_power_spectrum_harmonic

        n_ell, n_sys = 20, 2
        ell = np.arange(n_ell, dtype=float)
        cl_obs = np.ones(n_ell) * 1e-4
        alpha = np.array([0.1, -0.05])
        t_cls = np.ones((n_sys, n_ell)) * 5e-5
        cl_corr = correct_power_spectrum_harmonic(cl_obs, ell, n_sys, alpha, t_cls)
        assert cl_corr.shape == (n_ell,)

    def test_zero_alpha_applies_harmonic_bias_only(self):
        from sys_mapping.correction import correct_power_spectrum_harmonic
        from sys_mapping.power_spectrum import harmonic_bias

        n_ell, n_sys = 10, 2
        ell = np.arange(n_ell, dtype=float)
        cl_obs = np.ones(n_ell) * 1e-4
        alpha = np.zeros(n_sys)
        t_cls = np.ones((n_sys, n_ell)) * 5e-5
        cl_corr = correct_power_spectrum_harmonic(cl_obs, ell, n_sys, alpha, t_cls)
        np.testing.assert_allclose(cl_corr, cl_obs - harmonic_bias(n_sys, ell), rtol=1e-10)
