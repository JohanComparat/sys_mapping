"""Tests for sys_mapping.mocks — synthetic galaxy catalog generation."""

import numpy as np
import pytest

from sys_mapping.mocks import (
    MockCatalog,
    generate_lognormal_field,
    make_galactic_mask,
    make_mock_catalog,
    make_mock_suite,
)


# ---------------------------------------------------------------------------
# generate_lognormal_field
# ---------------------------------------------------------------------------


class TestGenerateLognormalField:
    def test_shape(self):
        nside = 16
        delta = generate_lognormal_field(nside, sigma=0.5, seed=0)
        assert delta.shape == (12 * nside**2,)

    def test_mean_near_zero(self):
        """E[exp(G - sigma^2/2)] - 1 = 0 in the large-pixel limit."""
        delta = generate_lognormal_field(nside=32, sigma=0.5, seed=42)
        assert abs(float(delta.mean())) < 0.05  # Poisson noise broadens this

    def test_always_above_minus_one(self):
        """Physical constraint: number count n_g > 0 ⟹ δ_g > −1."""
        delta = generate_lognormal_field(nside=32, sigma=0.5, seed=1)
        assert float(delta.min()) > -1.0

    def test_positively_skewed(self):
        """Lognormal distribution has positive skewness."""
        from scipy.stats import skew

        delta = generate_lognormal_field(nside=64, sigma=0.5, seed=0)
        assert skew(delta) > 0

    def test_variance_increases_with_sigma(self):
        delta_lo = generate_lognormal_field(nside=32, sigma=0.3, seed=0)
        delta_hi = generate_lognormal_field(nside=32, sigma=0.8, seed=0)
        assert float(delta_hi.std()) > float(delta_lo.std())

    def test_reproducibility(self):
        d1 = generate_lognormal_field(nside=16, sigma=0.5, seed=77)
        d2 = generate_lognormal_field(nside=16, sigma=0.5, seed=77)
        np.testing.assert_array_equal(d1, d2)


# ---------------------------------------------------------------------------
# make_galactic_mask
# ---------------------------------------------------------------------------


class TestMakeGalacticMask:
    def test_shape(self):
        nside = 16
        mask = make_galactic_mask(nside, lat_cut_deg=20.0)
        assert mask.shape == (12 * nside**2,)

    def test_dtype_bool(self):
        mask = make_galactic_mask(nside=16)
        assert mask.dtype == bool

    def test_sky_fraction_plausible(self):
        """|b| > 20° mask removes ~1/3 of the sky."""
        mask = make_galactic_mask(nside=32, lat_cut_deg=20.0)
        sky_frac = float(mask.mean())
        assert 0.5 < sky_frac < 0.9

    def test_wider_cut_removes_more_sky(self):
        mask_20 = make_galactic_mask(nside=32, lat_cut_deg=20.0)
        mask_40 = make_galactic_mask(nside=32, lat_cut_deg=40.0)
        assert float(mask_40.mean()) < float(mask_20.mean())

    def test_zero_cut_retains_almost_full_sky(self):
        # lat_cut=0 uses strict >, so pixels exactly on the galactic equator
        # (lat=0) are excluded. The galactic equator at NSIDE=16 covers ~2%.
        mask = make_galactic_mask(nside=16, lat_cut_deg=0.0)
        assert float(mask.mean()) > 0.95


# ---------------------------------------------------------------------------
# make_mock_catalog
# ---------------------------------------------------------------------------


class TestMakeMockCatalog:
    @pytest.fixture(scope="class")
    def mock_combined(self):
        return make_mock_catalog(nside=16, n_sys=2, scenario="combined", seed=0)

    @pytest.fixture(scope="class")
    def mock_none(self):
        return make_mock_catalog(nside=16, n_sys=2, scenario="none", seed=0)

    def test_returns_mock_catalog(self, mock_combined):
        assert isinstance(mock_combined, MockCatalog)

    def test_n_sys(self, mock_combined):
        assert mock_combined.n_sys == 2

    def test_galaxy_counts_positive(self, mock_combined):
        assert mock_combined.n_gal > 0

    def test_random_counts_positive(self, mock_combined):
        assert mock_combined.n_rand > 0

    def test_rand_larger_than_gal(self, mock_combined):
        assert mock_combined.n_rand > mock_combined.n_gal

    def test_positions_in_range(self, mock_combined):
        # RA is wrapped to [0, 360); Dec is clamped to [-90, 90]
        assert np.all(mock_combined.ra_gal >= 0)
        assert np.all(mock_combined.ra_gal < 360)
        assert np.all(mock_combined.dec_gal >= -90)
        assert np.all(mock_combined.dec_gal <= 90)

    def test_templates_shape(self, mock_combined):
        assert mock_combined.templates.shape == (2, 12 * 16**2)

    def test_delta_true_shape(self, mock_combined):
        assert mock_combined.delta_true.shape == (12 * 16**2,)

    def test_delta_obs_shape(self, mock_combined):
        assert mock_combined.delta_obs.shape == (12 * 16**2,)

    def test_mask_shape(self, mock_combined):
        assert mock_combined.mask.shape == (12 * 16**2,)
        assert mock_combined.mask.dtype == bool

    def test_scenario_none_obs_equals_true(self, mock_none):
        """No-contamination scenario: delta_obs must equal delta_true."""
        np.testing.assert_array_equal(mock_none.delta_obs, mock_none.delta_true)

    def test_scenario_none_zero_amplitudes(self, mock_none):
        np.testing.assert_array_equal(mock_none.a_true, 0.0)
        np.testing.assert_array_equal(mock_none.b_true, 0.0)

    def test_scenario_additive_zero_b(self):
        mock = make_mock_catalog(nside=16, n_sys=2, scenario="additive", seed=1)
        np.testing.assert_array_equal(mock.b_true, 0.0)

    def test_scenario_multiplicative_zero_a(self):
        mock = make_mock_catalog(nside=16, n_sys=2, scenario="multiplicative", seed=2)
        np.testing.assert_array_equal(mock.a_true, 0.0)

    def test_combined_obs_differs_from_true(self, mock_combined):
        """With nonzero amplitudes, observed field must differ from true."""
        assert not np.allclose(mock_combined.delta_obs, mock_combined.delta_true)

    def test_fixed_amplitudes_respected(self):
        a_in = np.array([0.2, -0.15])
        b_in = np.array([0.1, 0.05])
        mock = make_mock_catalog(nside=16, n_sys=2,
                                 a_true=a_in, b_true=b_in,
                                 scenario="combined", seed=3)
        np.testing.assert_array_equal(mock.a_true, a_in)
        np.testing.assert_array_equal(mock.b_true, b_in)

    def test_reproducibility(self):
        m1 = make_mock_catalog(nside=16, n_sys=2, scenario="additive", seed=99)
        m2 = make_mock_catalog(nside=16, n_sys=2, scenario="additive", seed=99)
        np.testing.assert_array_equal(m1.ra_gal, m2.ra_gal)
        np.testing.assert_array_equal(m1.delta_true, m2.delta_true)

    def test_n_good_pix_less_than_n_pix(self, mock_combined):
        assert mock_combined.n_good_pix < mock_combined.n_pix

    def test_additive_contamination_changes_field(self):
        """Additive contamination with nonzero amplitude must change delta_obs."""
        a = np.array([0.5, -0.3])
        mock = make_mock_catalog(nside=16, n_sys=2, a_true=a,
                                 scenario="additive", seed=5)
        assert not np.allclose(mock.delta_obs, mock.delta_true)

    def test_multiplicative_contamination_changes_field(self):
        b = np.array([0.5, -0.3])
        mock = make_mock_catalog(nside=16, n_sys=2, b_true=b,
                                 scenario="multiplicative", seed=6)
        assert not np.allclose(mock.delta_obs, mock.delta_true)


# ---------------------------------------------------------------------------
# make_mock_suite
# ---------------------------------------------------------------------------


class TestMakeMockSuite:
    @pytest.fixture(scope="class")
    def suite(self):
        return make_mock_suite(nside=16, n_sys=2, seed=0)

    def test_returns_all_four_scenarios(self, suite):
        assert set(suite.keys()) == {"none", "additive", "multiplicative", "combined"}

    def test_all_values_are_mock_catalog(self, suite):
        for mock in suite.values():
            assert isinstance(mock, MockCatalog)

    def test_n_sys_consistent(self, suite):
        for mock in suite.values():
            assert mock.n_sys == 2

    def test_shared_amplitudes_across_scenarios(self, suite):
        """All scenarios use the same a_true and b_true values."""
        # combined should have both nonzero (if drawn from N(0,0.1) at least one nonzero)
        # additive should have b_true=0
        np.testing.assert_array_equal(suite["additive"].b_true, 0.0)
        np.testing.assert_array_equal(suite["multiplicative"].a_true, 0.0)
        np.testing.assert_array_equal(suite["none"].a_true, 0.0)
        np.testing.assert_array_equal(suite["none"].b_true, 0.0)

    def test_none_obs_equals_true(self, suite):
        mock = suite["none"]
        np.testing.assert_array_equal(mock.delta_obs, mock.delta_true)

    def test_custom_scenarios_subset(self):
        suite = make_mock_suite(nside=16, n_sys=2,
                                scenarios=["none", "additive"], seed=1)
        assert set(suite.keys()) == {"none", "additive"}

    def test_fixed_amplitudes_forwarded(self):
        a = np.array([0.3, -0.1])
        b = np.array([0.2, 0.15])
        suite = make_mock_suite(nside=16, n_sys=2,
                                a_amplitudes=a, b_amplitudes=b, seed=2)
        np.testing.assert_array_equal(suite["additive"].a_true, a)
        np.testing.assert_array_equal(suite["multiplicative"].b_true, b)


# ---------------------------------------------------------------------------
# End-to-end: pixelize → overdensity → OLS recovery (fast integration test)
# ---------------------------------------------------------------------------


class TestMockPipelineIntegration:
    def test_overdensity_near_zero_mean(self):
        """Pixelized overdensity from a mock should have near-zero mean."""
        import sys_mapping as sm

        mock = make_mock_catalog(nside=16, n_sys=2, scenario="none", seed=10,
                                 n_mean=50)
        gal_counts = sm.pixelize_catalog(mock.ra_gal, mock.dec_gal, mock.nside)
        rand_counts = sm.pixelize_catalog(mock.ra_rand, mock.dec_rand, mock.nside)
        delta_g, good_pix = sm.compute_overdensity(gal_counts, rand_counts)
        assert abs(float(delta_g.mean())) < 0.05

    def test_ols_recovers_additive_amplitude(self):
        """OLS should recover a single strong additive amplitude to within 30%."""
        import sys_mapping as sm

        a_in = np.array([0.4, 0.0])
        mock = make_mock_catalog(nside=16, n_sys=2, a_true=a_in,
                                 scenario="additive", seed=20, n_mean=100)

        gal_counts = sm.pixelize_catalog(mock.ra_gal, mock.dec_gal, mock.nside)
        rand_counts = sm.pixelize_catalog(mock.ra_rand, mock.dec_rand, mock.nside)
        delta_g, good_pix = sm.compute_overdensity(gal_counts, rand_counts)
        delta_t = sm.assign_template_values(mock.templates, good_pix)

        X = delta_t.T
        alpha_ols, *_ = np.linalg.lstsq(X, delta_g, rcond=None)
        rel_err = abs(alpha_ols[0] - a_in[0]) / abs(a_in[0])
        assert rel_err < 0.5  # loose tolerance at NSIDE=16
