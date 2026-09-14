"""Tests for sys_mapping.diagnostics — null tests, SNR ranking, footprint masking."""

import numpy as np
import pytest
import sys_mapping as sm

from sys_mapping.diagnostics import (
    footprint_mask_diagnostics,
    isd_template_significance,
    null_test_cross_correlations,
    snr_template_ranking,
)

# These tests exercise GLASS mock mechanics on a parametric field chosen on purpose,
# not a calibrated null, so the library's "not matched to any sample" warning is
# expected here and would only bury real ones.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*not matched to any sample.*:UserWarning")


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


class TestResidualTemplateCorrelation:
    """The residual test correlates the corrected density, and is calibrated on mocks.

    The weight-template correlation it replaces as a goodness-of-fit is invariant to
    the size of the fitted amplitude, so it cannot tell a large residual from a
    negligible one.  This one must respond to what was left behind, and must hold its
    size on a spatially correlated field where a white-noise calibration would not.
    """

    @staticmethod
    def _clustered(rng, n_sys=3, n_null=60, nside=32):
        import healpy as hp

        npix = hp.nside2npix(nside)
        _, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True)
        good = np.abs(lat) > 30

        def field(fwhm, amp=1.0):
            m = hp.smoothing(rng.standard_normal(npix), fwhm=np.radians(fwhm))[good]
            return amp * (m - m.mean()) / m.std()

        t = np.array([field(f) for f in (6, 12, 20)[:n_sys]])
        null = np.array([field(8, 0.3) for _ in range(n_null)])
        return field, t, null

    def test_leave_one_out_variance_matches_brute_force(self):
        rng = np.random.default_rng(3)
        t = rng.standard_normal((3, 800))
        null = rng.standard_normal((25, 800))
        out = sm.residual_template_correlation_test(rng.standard_normal(800), t, null)

        def r_of(d):
            return np.array([np.corrcoef(d, ti)[0, 1] for ti in t])

        R = np.array([r_of(m) for m in null])
        brute = np.array([
            np.sum(R[k] ** 2 / np.delete(R, k, axis=0).var(axis=0, ddof=1))
            for k in range(len(R))
        ])
        np.testing.assert_allclose(out["null_chi2"], brute, rtol=1e-8)
        np.testing.assert_allclose(out["null_std"] ** 2, R.var(axis=0, ddof=1), rtol=1e-8)

    @pytest.mark.slow
    def test_holds_its_size_on_a_clustered_field(self):
        rng = np.random.default_rng(1)
        field, t, null = self._clustered(rng)
        ps = np.array([
            sm.residual_template_correlation_test(field(8, 0.3), t, null)["p_value"]
            for _ in range(150)
        ])
        # Nominal 5 %; 150 draws give a binomial sd of about 1.8 %.
        assert 0.0 <= np.mean(ps < 0.05) <= 0.11

    def test_clustering_inflates_the_variance_far_beyond_white_noise(self):
        rng = np.random.default_rng(2)
        field, t, null = self._clustered(rng)
        out = sm.residual_template_correlation_test(field(8, 0.3), t, null)
        white = 1.0 / t.shape[1]
        assert np.median(out["null_std"] ** 2) > 5 * white

    def test_responds_to_the_size_of_the_residual(self):
        # The property the weight-template correlation lacks.
        rng = np.random.default_rng(4)
        field, t, null = self._clustered(rng)
        base = field(8, 0.3)
        chi = [sm.residual_template_correlation_test(base + a * t[0], t, null)["chi2"]
               for a in (0.0, 0.05, 0.1, 0.2)]
        assert all(b > a for a, b in zip(chi, chi[1:]))
        assert sm.residual_template_correlation_test(base + 0.2 * t[0], t, null)[
            "p_value"] < 0.05

    def test_weight_correlation_is_blind_where_this_is_not(self):
        rng = np.random.default_rng(5)
        t = rng.standard_normal((3, 20000))
        clus = rng.standard_normal(20000) * 0.3
        null = rng.standard_normal((40, 20000)) * 0.3
        r_w, chi = [], []
        for a0 in (1e-1, 1e-4):
            a_hat = np.array([0.8 * a0, 0.0, 0.0])
            d_obs = clus + np.array([a0, 0, 0]) @ t
            w = 1.0 / (1.0 + a_hat @ t)
            r_w.append(np.max(np.abs(sm.null_test_cross_correlations(w, t, 10)["correlations"])))
            chi.append(sm.residual_template_correlation_test(d_obs - a_hat @ t, t, null)["chi2"])
        assert abs(r_w[0] - r_w[1]) < 0.01          # weight correlation: indistinguishable
        assert chi[0] > 10 * chi[1]                 # residual test: separates them

    @pytest.mark.parametrize("bad", ["corr", "null_pix", "too_few"])
    def test_rejects_malformed_input(self, bad):
        rng = np.random.default_rng(0)
        t = rng.standard_normal((3, 500))
        d = rng.standard_normal(500)
        null = rng.standard_normal((20, 500))
        if bad == "corr":
            d = d[:400]
        elif bad == "null_pix":
            null = null[:, :400]
        else:
            null = null[:4]
        with pytest.raises(ValueError):
            sm.residual_template_correlation_test(d, t, null)


class TestResidualTestRequirements:
    """The two ways a residual test silently goes inert, pinned so neither recurs."""

    @staticmethod
    def _setup(seed=1):
        import healpy as hp

        rng = np.random.default_rng(seed)
        npix = hp.nside2npix(32)
        _, lat = hp.pix2ang(32, np.arange(npix), lonlat=True)
        good = np.abs(lat) > 30

        def field(fwhm, amp=1.0):
            m = hp.smoothing(rng.standard_normal(npix), fwhm=np.radians(fwhm))[good]
            return amp * (m - m.mean()) / m.std()

        T = np.array([field(f) for f in (6, 12, 20)])
        return field, T

    @staticmethod
    def _ols(d, T):
        return d - np.linalg.lstsq(T.T, d, rcond=None)[0] @ T

    def test_a_regression_zeroes_the_correlation_with_its_own_templates(self):
        field, T = self._setup()
        d = self._ols(field(8, 0.3), T)
        null = np.array([self._ols(field(8, 0.3), T) for _ in range(20)])
        out = sm.residual_template_correlation_test(d, T, null)
        # Rounding error, in data and null alike: nothing to test.
        assert np.all(np.abs(out["correlations"]) < 1e-10)
        assert np.all(out["null_std"] < 1e-10)

    def test_uncorrected_nulls_pin_a_fitted_template_at_p_one(self):
        # The corrected field's correlation with a template it fitted is rounding
        # error, while an uncorrected realisation keeps its chance correlation, so
        # the data beats every realisation whatever residual it carries.
        field, T = self._setup(seed=2)
        raw_null = np.array([field(8, 0.3) for _ in range(40)])
        for resid in (0.0, 0.3):
            d = self._ols(field(8, 0.3) + resid * T[0], T)
            out = sm.residual_template_correlation_test(d, T, raw_null)
            assert out["p_value"] == 1.0

    @pytest.mark.slow
    def test_a_held_out_template_with_matched_nulls_is_calibrated_and_has_power(self):
        field, T = self._setup(seed=3)
        others = T[1:]
        rates = {}
        for resid in (0.0, 0.08):
            ps = []
            for _ in range(60):
                null = np.array([self._ols(field(8, 0.3), others) for _ in range(30)])
                d = self._ols(field(8, 0.3) + resid * T[0], others)
                ps.append(sm.residual_template_correlation_test(d, T[[0]], null)["p_value"])
            rates[resid] = float(np.mean(np.asarray(ps) < 0.05))
        assert rates[0.0] <= 0.13          # nominal 5 %, 60 draws
        assert rates[0.08] >= 0.8


class TestCalibratedTemplateSignificance:
    """Detection significance that holds its size on a spatially correlated field.

    The independent-pixel significance fires on most clean realisations because its
    error is too small and because the reported number is a maximum over templates.
    """

    @staticmethod
    def _setup(seed=3, n_sys=6, n_null=300):
        import healpy as hp

        rng = np.random.default_rng(seed)
        npix = hp.nside2npix(32)
        _, lat = hp.pix2ang(32, np.arange(npix), lonlat=True)
        good = np.abs(lat) > 30

        def field(fwhm, amp=1.0):
            m = hp.smoothing(rng.standard_normal(npix), fwhm=np.radians(fwhm))[good]
            return amp * (m - m.mean()) / m.std()

        T = sm.standardise_on_footprint(np.array([field(f) for f in np.linspace(5, 25, n_sys)]))
        null = np.array([field(8, 0.3) for _ in range(n_null)])
        return field, T, null

    def test_leave_one_out_scatter_matches_brute_force(self):
        rng = np.random.default_rng(1)
        t = rng.standard_normal((3, 600))
        null = rng.standard_normal((30, 600))
        out = sm.calibrated_template_significance(rng.standard_normal(600), t, null)
        proj = np.linalg.pinv(t.T)
        A = null @ proj.T
        brute = []
        for k in range(len(A)):
            sd = np.delete(A, k, axis=0).std(axis=0, ddof=1)
            brute.append(np.max(np.abs(A[k]) / sd))
        np.testing.assert_allclose(out["null_max_significance"], brute, rtol=1e-8)

    @pytest.mark.slow
    def test_reproduces_the_false_positive_rate_and_removes_it(self):
        # 400 realisations put the p floor, 1/401, below the 0.0027 threshold; at 300
        # the floor is 1/301 and the family-wise count could never exceed zero.
        field, T, null = self._setup(n_null=400)
        n_sys = T.shape[0]
        iid = fw = fw05 = 0
        trials = 300
        for _ in range(trials):
            d = field(8, 0.3)
            a = np.linalg.lstsq(T.T, d, rcond=None)[0]
            r = d - a @ T
            cov = r.var() * len(d) / (len(d) - n_sys) * np.linalg.inv(T @ T.T)
            iid += np.any(np.abs(a) / np.sqrt(np.diag(cov)) > 3)
            fwp = sm.calibrated_template_significance(d, T, null)["family_wise_p"]
            fw += fwp <= 0.0027
            fw05 += fwp <= 0.05
        assert 1.0 / (1 + len(null)) < 0.0027
        assert iid / trials > 0.5            # the defect: most clean fields "detect"
        assert fw / trials <= 0.02           # nominal 0.27 %; 300 draws
        assert 0.01 <= fw05 / trials <= 0.10  # nominal 5 %; binomial sd 1.3 %

    def test_per_template_thresholds_are_crossed_by_some_template_too_often(self):
        # Multiplicity: calibrating the error alone leaves the family-wise rate high.
        field, T, null = self._setup(seed=4)
        crossed = np.mean([
            np.any(sm.calibrated_template_significance(field(8, 0.3), T, null)["p_values"] < 0.05)
            for _ in range(120)
        ])
        assert crossed > 0.15                # about 1 - 0.95^6 = 26 %

    def test_detects_a_real_systematic_on_the_right_template(self):
        field, T, null = self._setup(seed=5)
        out = sm.calibrated_template_significance(field(8, 0.3) + 0.15 * T[2], T, null)
        assert out["family_wise_p"] < 0.01
        assert int(np.argmax(out["significance"])) == 2

    @pytest.mark.parametrize("bad", ["obs", "null_pix", "too_few"])
    def test_rejects_malformed_input(self, bad):
        rng = np.random.default_rng(0)
        t = rng.standard_normal((3, 400))
        d = rng.standard_normal(400)
        null = rng.standard_normal((20, 400))
        if bad == "obs":
            d = d[:300]
        elif bad == "null_pix":
            null = null[:, :300]
        else:
            null = null[:4]
        with pytest.raises(ValueError):
            sm.calibrated_template_significance(d, t, null)
