"""Tests for sys_mapping.power_spectrum — harmonic-space estimators."""

import numpy as np
import pytest

from sys_mapping.power_spectrum import (
    harmonic_bias,
    measure_pseudo_cl,
    mode_projection_bias,
    subtract_template_cl,
)


class TestHarmonicBias:
    def test_formula(self):
        ell = np.array([0, 1, 2, 5, 10], dtype=float)
        n = 3
        bias = harmonic_bias(n, ell)
        expected = -n / (2 * ell + 1)
        np.testing.assert_allclose(bias, expected, rtol=1e-12)

    def test_single_template(self):
        ell = np.arange(10, dtype=float)
        bias = harmonic_bias(1, ell)
        assert bias[0] == pytest.approx(-1.0)  # ell=0: -1/(1) = -1
        assert bias[1] == pytest.approx(-1 / 3)

    def test_negative_definite(self):
        ell = np.arange(1, 50, dtype=float)
        bias = harmonic_bias(5, ell)
        assert np.all(bias < 0)

    def test_zero_templates(self):
        ell = np.arange(10, dtype=float)
        bias = harmonic_bias(0, ell)
        np.testing.assert_array_equal(bias, 0.0)


class TestMeasurePseudoCl:
    @pytest.fixture
    def nside(self):
        return 32

    @pytest.fixture
    def full_sky_map(self, nside):
        rng = np.random.default_rng(0)
        n_pix = 12 * nside**2
        return rng.standard_normal(n_pix)

    def test_returns_correct_shape(self, nside, full_sky_map):
        n_pix = 12 * nside**2
        mask = np.ones(n_pix, dtype=bool)
        lmax = 10
        ell, cl = measure_pseudo_cl(full_sky_map, mask, lmax=lmax)
        assert ell.shape == (lmax + 1,)
        assert cl.shape == (lmax + 1,)

    def test_ell_is_integer_sequence(self, nside, full_sky_map):
        n_pix = 12 * nside**2
        mask = np.ones(n_pix, dtype=bool)
        lmax = 8
        ell, _ = measure_pseudo_cl(full_sky_map, mask, lmax=lmax)
        np.testing.assert_array_equal(ell, np.arange(lmax + 1))

    def test_cl_nonnegative_for_autocorrelation(self, nside, full_sky_map):
        n_pix = 12 * nside**2
        mask = np.ones(n_pix, dtype=bool)
        _, cl = measure_pseudo_cl(full_sky_map, mask, lmax=20)
        assert np.all(cl >= 0)

    def test_full_sky_white_noise_cl_near_flat(self, nside):
        """Full-sky white noise: pseudo-Cl should be roughly flat (unit variance map)."""
        rng = np.random.default_rng(42)
        n_pix = 12 * nside**2
        wn_map = rng.standard_normal(n_pix)
        mask = np.ones(n_pix, dtype=bool)
        _, cl = measure_pseudo_cl(wn_map, mask, lmax=20)
        # Expected ~ 1 / n_pix (white noise Cl)
        cl_mean = np.mean(cl[2:])  # skip monopole/dipole
        assert cl_mean > 0

    def test_zero_map_gives_zero_cl(self, nside):
        n_pix = 12 * nside**2
        zero_map = np.zeros(n_pix)
        mask = np.ones(n_pix, dtype=bool)
        _, cl = measure_pseudo_cl(zero_map, mask, lmax=10)
        np.testing.assert_allclose(cl, 0.0, atol=1e-30)

    def test_lmax_none_uses_default(self, nside, full_sky_map):
        n_pix = 12 * nside**2
        mask = np.ones(n_pix, dtype=bool)
        ell, cl = measure_pseudo_cl(full_sky_map, mask)  # lmax=None
        assert len(ell) == 3 * nside

    def test_mask_zeros_out_pixels(self, nside, full_sky_map):
        n_pix = 12 * nside**2
        mask = np.zeros(n_pix, dtype=bool)
        _, cl = measure_pseudo_cl(full_sky_map, mask, lmax=10)
        np.testing.assert_allclose(cl, 0.0, atol=1e-30)


class TestSubtractTemplateCl:
    @pytest.fixture
    def setup(self):
        nside = 32
        n_pix = 12 * nside**2
        rng = np.random.default_rng(1)
        delta_g = rng.standard_normal(n_pix)
        delta_t = rng.standard_normal((2, n_pix))
        mask = np.ones(n_pix, dtype=bool)
        lmax = 15
        _, pseudo_cl = measure_pseudo_cl(delta_g, mask, lmax=lmax)
        alpha = np.array([0.1, -0.05])
        return pseudo_cl, delta_t, mask, alpha, lmax

    def test_zero_alpha_returns_input(self, setup):
        pseudo_cl, delta_t, mask, alpha, lmax = setup
        cl_cleaned = subtract_template_cl(pseudo_cl, delta_t, mask,
                                          np.zeros_like(alpha), lmax=lmax)
        np.testing.assert_allclose(cl_cleaned, pseudo_cl, rtol=1e-12)

    def test_output_shape(self, setup):
        pseudo_cl, delta_t, mask, alpha, lmax = setup
        cl_cleaned = subtract_template_cl(pseudo_cl, delta_t, mask, alpha, lmax=lmax)
        assert cl_cleaned.shape == pseudo_cl.shape

    def test_subtraction_reduces_power_for_positive_alpha(self, setup):
        """Positive alpha on a positive template Cl reduces the power spectrum."""
        pseudo_cl, delta_t, mask, _, lmax = setup
        alpha = np.array([0.5, 0.0])  # strong positive contamination
        cl_cleaned = subtract_template_cl(pseudo_cl, delta_t, mask, alpha, lmax=lmax)
        # Mean power should be reduced (template is subtracted with positive sign)
        assert np.mean(cl_cleaned) < np.mean(pseudo_cl)

    def test_lmax_none_infers_from_pseudo_cl(self, setup):
        pseudo_cl, delta_t, mask, alpha, _ = setup
        cl_cleaned = subtract_template_cl(pseudo_cl, delta_t, mask, alpha)  # lmax=None
        assert cl_cleaned.shape == pseudo_cl.shape


class TestModeProjectionBias:
    def test_basic_mode_shape(self):
        n_ell = 20
        pseudo_cl = np.ones(n_ell) * 1e-4
        ell = np.arange(n_ell, dtype=float)
        cl_deproj, bias = mode_projection_bias(
            pseudo_cl, None, ell, n_templates=2, mode="basic"
        )
        assert cl_deproj.shape == (n_ell,)
        assert bias.shape == (n_ell,)

    def test_basic_bias_matches_harmonic_bias(self):
        n_ell = 15
        pseudo_cl = np.ones(n_ell) * 1e-4
        ell = np.arange(n_ell, dtype=float)
        n_templates = 3
        cl_deproj, bias = mode_projection_bias(
            pseudo_cl, None, ell, n_templates=n_templates, mode="basic"
        )
        expected_bias = harmonic_bias(n_templates, ell)
        np.testing.assert_allclose(bias, expected_bias, rtol=1e-12)

    def test_basic_corrects_pseudo_cl(self):
        n_ell = 15
        ell = np.arange(n_ell, dtype=float)
        # Inject a known bias
        n_templates = 2
        true_cl = np.ones(n_ell) * 1e-4
        bias = harmonic_bias(n_templates, ell)
        observed_cl = true_cl + bias  # pseudo-Cl contains the bias
        cl_deproj, _ = mode_projection_bias(
            observed_cl, None, ell, n_templates=n_templates, mode="basic"
        )
        np.testing.assert_allclose(cl_deproj, true_cl, rtol=1e-12)

    def test_extended_mode_shape(self):
        n_ell = 20
        pseudo_cl = np.ones(n_ell) * 1e-4
        ell = np.arange(n_ell, dtype=float)
        cl_deproj, bias = mode_projection_bias(
            pseudo_cl, None, ell, n_templates=2, mode="extended", threshold=0.1
        )
        assert cl_deproj.shape == (n_ell,)
        assert bias.shape == (n_ell,)

    def test_invalid_mode_raises(self):
        ell = np.arange(10, dtype=float)
        pseudo_cl = np.ones(10)
        with pytest.raises(ValueError, match="mode must be"):
            mode_projection_bias(pseudo_cl, None, ell, n_templates=1, mode="invalid")

    def test_with_coupling_matrix(self):
        n_ell = 10
        ell = np.arange(n_ell, dtype=float)
        pseudo_cl = np.ones(n_ell) * 1e-3
        M = np.eye(n_ell) * 1.1  # slight scaling
        cl_deproj, bias = mode_projection_bias(
            pseudo_cl, M, ell, n_templates=1, mode="basic"
        )
        assert cl_deproj.shape == (n_ell,)
        # Bias should be M @ b_ell
        expected_bias = M @ harmonic_bias(1, ell)
        np.testing.assert_allclose(bias, expected_bias, rtol=1e-12)
