"""Tests for sys_mapping.diagnostics — null tests, SNR ranking, footprint masking."""

import numpy as np
import pytest

from sys_mapping.diagnostics import (
    footprint_mask_diagnostics,
    isd_template_significance,
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


class TestSnrTemplateRankingIsd:
    """Tests for snr_template_ranking with method='isd'."""

    @pytest.fixture
    def contaminated_data(self):
        rng = np.random.default_rng(50)
        n_pix, n_sys = 4000, 4
        delta_t = rng.standard_normal((n_sys, n_pix))
        # Strong contamination from template 2 only
        amplitudes = np.array([0.02, 0.02, 0.5, 0.02])
        delta_g = amplitudes @ delta_t + rng.standard_normal(n_pix) * 0.05
        return delta_g, delta_t

    def test_isd_shape_nonneg(self, contaminated_data):
        delta_g, delta_t = contaminated_data
        snr = snr_template_ranking(delta_g, delta_t, method="isd")
        assert snr.shape == (4,)
        assert np.all(snr >= 0)

    def test_isd_injected_template_ranks_first(self, contaminated_data):
        delta_g, delta_t = contaminated_data
        snr = snr_template_ranking(delta_g, delta_t, method="isd")
        assert np.argmax(snr) == 2

    def test_isd_pure_noise_small_delta_chisq(self):
        rng = np.random.default_rng(51)
        n_pix, n_sys = 3000, 3
        delta_t = rng.standard_normal((n_sys, n_pix))
        delta_g = rng.standard_normal(n_pix) * 0.1
        snr = snr_template_ranking(delta_g, delta_t, method="isd")
        # Pure noise: Δχ² should be small (well below a signal-level value)
        assert np.all(snr < 50.0)

    def test_isd_poly_order_3(self, contaminated_data):
        delta_g, delta_t = contaminated_data
        snr = snr_template_ranking(delta_g, delta_t, method="isd", poly_order=3)
        assert snr.shape == (4,)
        assert np.all(snr >= 0)

    def test_isd_with_fracdet(self, contaminated_data):
        delta_g, delta_t = contaminated_data
        n_pix = delta_t.shape[1]
        rng = np.random.default_rng(52)
        fracdet = rng.uniform(0.5, 1.0, n_pix)
        snr = snr_template_ranking(delta_g, delta_t, method="isd", fracdet=fracdet)
        assert snr.shape == (4,)
        assert np.all(np.isfinite(snr))
        assert np.argmax(snr) == 2

    def test_isd_custom_n_bins(self, contaminated_data):
        delta_g, delta_t = contaminated_data
        snr = snr_template_ranking(delta_g, delta_t, method="isd", n_bins=20)
        assert snr.shape == (4,)
        assert np.all(snr >= 0)

    def test_isd_invalid_method_still_raises(self):
        rng = np.random.default_rng(0)
        delta_t = rng.standard_normal((2, 100))
        delta_g = rng.standard_normal(100)
        with pytest.raises(ValueError, match="method must be"):
            snr_template_ranking(delta_g, delta_t, method="bad_method")


class TestIsdTemplateSignificance:
    """Tests for isd_template_significance (GLASS mock-based p-values)."""

    pytest.importorskip("glass")

    @pytest.fixture
    def setup(self):
        pytest.importorskip("glass")
        nside = 16
        import healpy as hp
        n_full = hp.nside2npix(nside)
        rng = np.random.default_rng(60)
        n_pix = n_full  # full sky as footprint
        n_sys = 3
        delta_t = rng.standard_normal((n_sys, n_pix))
        good = np.ones(n_full, dtype=bool)
        # Inject contamination from template 1
        delta_g = 0.4 * delta_t[1] + rng.standard_normal(n_pix) * 0.1
        z_edges = np.array([0.0, 0.5])
        nz = np.array([500.0])
        return delta_g, delta_t, good, nside, 500, z_edges, nz

    def test_returns_expected_keys(self, setup):
        delta_g, delta_t, good, nside, n_total, z_edges, nz = setup
        result = isd_template_significance(
            delta_g, delta_t, good, nside, n_total, z_edges, nz, n_mocks=3, seed=0,
        )
        assert set(result.keys()) == {"delta_chi2", "p_values", "delta_chi2_mocks"}

    def test_shapes(self, setup):
        delta_g, delta_t, good, nside, n_total, z_edges, nz = setup
        n_mocks = 4
        result = isd_template_significance(
            delta_g, delta_t, good, nside, n_total, z_edges, nz, n_mocks=n_mocks, seed=0,
        )
        n_sys = delta_t.shape[0]
        assert result["delta_chi2"].shape == (n_sys,)
        assert result["p_values"].shape == (n_sys,)
        assert result["delta_chi2_mocks"].shape == (n_mocks, n_sys)

    def test_p_values_range(self, setup):
        delta_g, delta_t, good, nside, n_total, z_edges, nz = setup
        result = isd_template_significance(
            delta_g, delta_t, good, nside, n_total, z_edges, nz, n_mocks=5, seed=0,
        )
        assert np.all(result["p_values"] > 0)
        assert np.all(result["p_values"] <= 1)

    def test_delta_chi2_nonneg(self, setup):
        delta_g, delta_t, good, nside, n_total, z_edges, nz = setup
        result = isd_template_significance(
            delta_g, delta_t, good, nside, n_total, z_edges, nz, n_mocks=3, seed=0,
        )
        assert np.all(result["delta_chi2"] >= 0)
        assert np.all(result["delta_chi2_mocks"] >= 0)

    def test_injected_template_has_smallest_pvalue(self, setup):
        delta_g, delta_t, good, nside, n_total, z_edges, nz = setup
        result = isd_template_significance(
            delta_g, delta_t, good, nside, n_total, z_edges, nz, n_mocks=20, seed=0,
        )
        # Template 1 was injected — it should have the largest Δχ² and smallest p-value
        assert np.argmax(result["delta_chi2"]) == 1
        assert np.argmin(result["p_values"]) == 1
