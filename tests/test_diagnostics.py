"""Tests for sys_mapping.diagnostics — null tests, SNR ranking, footprint masking."""

import numpy as np
import pytest

from sys_mapping.diagnostics import (
    footprint_mask_diagnostics,
    null_test_cross_correlations,
    snr_template_ranking,
)


class TestNullTestCrossCorrelations:
    def test_output_keys(self):
        rng = np.random.default_rng(0)
        weights = rng.standard_normal(500) + 1.0
        delta_t = rng.standard_normal((3, 500))
        result = null_test_cross_correlations(weights, delta_t, n_bootstrap=20, seed=0)
        assert "correlations" in result
        assert "p_values" in result

    def test_output_shapes(self):
        rng = np.random.default_rng(1)
        n_sys = 4
        weights = rng.standard_normal(300) + 1.0
        delta_t = rng.standard_normal((n_sys, 300))
        result = null_test_cross_correlations(weights, delta_t, n_bootstrap=20, seed=0)
        assert result["correlations"].shape == (n_sys,)
        assert result["p_values"].shape == (n_sys,)

    def test_p_values_in_unit_interval(self):
        rng = np.random.default_rng(2)
        weights = rng.standard_normal(400) + 1.0
        delta_t = rng.standard_normal((3, 400))
        result = null_test_cross_correlations(weights, delta_t, n_bootstrap=30, seed=0)
        assert np.all(result["p_values"] >= 0)
        assert np.all(result["p_values"] <= 1)

    def test_uncorrelated_gives_small_correlations(self):
        """Random weights and templates: correlations should be near zero."""
        rng = np.random.default_rng(42)
        n_pix = 1000
        weights = rng.standard_normal(n_pix) + 1.0
        delta_t = rng.standard_normal((5, n_pix))
        result = null_test_cross_correlations(weights, delta_t, n_bootstrap=50, seed=0)
        assert np.all(np.abs(result["correlations"]) < 0.3)

    def test_fully_correlated_gives_high_correlation(self):
        """Weights proportional to a template: |r| should be near 1."""
        rng = np.random.default_rng(3)
        n_pix = 500
        t0 = rng.standard_normal(n_pix)
        delta_t = np.vstack([t0, rng.standard_normal((2, n_pix))])
        weights = 1.0 + 0.5 * t0  # exact linear relationship
        result = null_test_cross_correlations(weights, delta_t, n_bootstrap=20, seed=0)
        assert abs(result["correlations"][0]) > 0.7

    def test_constant_weights_give_zero_correlations(self):
        rng = np.random.default_rng(4)
        weights = np.ones(300)
        delta_t = rng.standard_normal((2, 300))
        result = null_test_cross_correlations(weights, delta_t, n_bootstrap=10, seed=0)
        np.testing.assert_allclose(result["correlations"], 0.0, atol=1e-10)


class TestSnrTemplateRanking:
    @pytest.fixture
    def strong_contamination_data(self):
        rng = np.random.default_rng(10)
        n_pix, n_sys = 3000, 4
        delta_t = rng.standard_normal((n_sys, n_pix))
        # Template 2 has 10× stronger contamination
        amplitudes = np.array([0.05, 0.04, 0.5, 0.03])
        delta_g = amplitudes @ delta_t + rng.standard_normal(n_pix) * 0.2
        return delta_g, delta_t

    def test_shape_template_method(self, strong_contamination_data):
        delta_g, delta_t = strong_contamination_data
        snr = snr_template_ranking(delta_g, delta_t, method="template")
        assert snr.shape == (4,)

    def test_shape_data_method(self, strong_contamination_data):
        delta_g, delta_t = strong_contamination_data
        snr = snr_template_ranking(delta_g, delta_t, method="data")
        assert snr.shape == (4,)

    def test_shape_peak_method(self, strong_contamination_data):
        pytest.importorskip("healpy")
        import healpy as hp
        n_pix = 12 * 16**2  # nside=16
        rng = np.random.default_rng(11)
        n_sys = 3
        delta_t = rng.standard_normal((n_sys, n_pix))
        amplitudes = np.array([0.05, 0.5, 0.02])
        delta_g = amplitudes @ delta_t + rng.standard_normal(n_pix) * 0.3
        snr = snr_template_ranking(delta_g, delta_t, method="peak")
        assert snr.shape == (n_sys,)

    def test_strongest_template_ranks_first_data_method(self, strong_contamination_data):
        delta_g, delta_t = strong_contamination_data
        snr = snr_template_ranking(delta_g, delta_t, method="data")
        assert np.argmax(snr) == 2

    def test_strongest_template_ranks_first_template_method(self, strong_contamination_data):
        delta_g, delta_t = strong_contamination_data
        snr = snr_template_ranking(delta_g, delta_t, method="template")
        assert np.argmax(snr) == 2

    def test_nonneg_data_method(self, strong_contamination_data):
        delta_g, delta_t = strong_contamination_data
        snr = snr_template_ranking(delta_g, delta_t, method="data")
        assert np.all(snr >= 0)

    def test_nonneg_template_method(self, strong_contamination_data):
        delta_g, delta_t = strong_contamination_data
        snr = snr_template_ranking(delta_g, delta_t, method="template")
        assert np.all(snr >= 0)

    def test_zero_norm_gives_zero_snr(self):
        n_pix, n_sys = 200, 2
        delta_g = np.zeros(n_pix)  # zero-norm galaxy field → snr[i] = 0
        delta_t = np.random.default_rng(0).standard_normal((n_sys, n_pix))
        snr = snr_template_ranking(delta_g, delta_t, method="data")
        np.testing.assert_array_equal(snr, 0.0)

    def test_invalid_method_raises(self, strong_contamination_data):
        delta_g, delta_t = strong_contamination_data
        with pytest.raises(ValueError, match="method must be"):
            snr_template_ranking(delta_g, delta_t, method="unknown")


class TestFootprintMaskDiagnostics:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(20)
        n_pix, n_sys = 4000, 3
        delta_t = rng.standard_normal((n_sys, n_pix))
        delta_g = 0.1 * delta_t[0] + rng.standard_normal(n_pix) * 0.5
        good = np.ones(n_pix, dtype=bool)
        mask_fractions = np.array([0.0, 0.05, 0.10, 0.15])
        return delta_g, delta_t, good, mask_fractions

    def test_output_keys(self, data):
        delta_g, delta_t, good, fracs = data
        result = footprint_mask_diagnostics(delta_g, delta_t, fracs, good)
        assert "alpha_hat" in result
        assert "scatter" in result

    def test_output_shapes(self, data):
        delta_g, delta_t, good, fracs = data
        result = footprint_mask_diagnostics(delta_g, delta_t, fracs, good)
        assert result["alpha_hat"].shape == (4, 3)
        assert result["scatter"].shape == (3,)

    def test_scatter_nonneg(self, data):
        delta_g, delta_t, good, fracs = data
        result = footprint_mask_diagnostics(delta_g, delta_t, fracs, good)
        assert np.all(result["scatter"] >= 0)

    def test_zero_fraction_matches_ols(self, data):
        """Masking fraction=0 should reproduce plain OLS."""
        delta_g, delta_t, good, _ = data
        fracs = np.array([0.0])
        result = footprint_mask_diagnostics(delta_g, delta_t, fracs, good)
        alpha_expected, *_ = np.linalg.lstsq(delta_t.T, delta_g, rcond=None)
        np.testing.assert_allclose(result["alpha_hat"][0], alpha_expected, rtol=1e-8)

    def test_stable_system_has_low_scatter(self):
        """System with no systematic signal: masking barely changes amplitudes."""
        rng = np.random.default_rng(30)
        n_pix, n_sys = 5000, 2
        delta_t = rng.standard_normal((n_sys, n_pix))
        delta_g = rng.standard_normal(n_pix) * 0.1  # pure noise
        good = np.ones(n_pix, dtype=bool)
        fracs = np.linspace(0, 0.3, 7)
        result = footprint_mask_diagnostics(delta_g, delta_t, fracs, good)
        # Scatter should be small when there's no real signal
        assert np.all(result["scatter"] < 0.1)
