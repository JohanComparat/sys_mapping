"""Tests for sys_mapping.utils — two-point measurement utilities."""

import warnings

import numpy as np
import pytest

import sys_mapping as sm
from sys_mapping.utils import (
    measure_two_point_function,
    measure_two_point_function_corrfunc,
    measure_kk_correlation_treecorr,
    measure_kk_covariance_treecorr,
    measure_kk_correlation_corrfunc,
)

try:
    import Corrfunc  # noqa: F401
    _corrfunc_available = True
except ImportError:
    _corrfunc_available = False

corrfunc_available = pytest.mark.skipif(
    not _corrfunc_available, reason="Corrfunc not installed"
)

N_GAL = 500
N_RAND = 1000


@pytest.fixture(scope="module")
def sky_patch():
    rng = np.random.default_rng(42)
    ra_g = rng.uniform(30.0, 60.0, N_GAL)
    dec_g = rng.uniform(-10.0, 10.0, N_GAL)
    ra_r = rng.uniform(30.0, 60.0, N_RAND)
    dec_r = rng.uniform(-10.0, 10.0, N_RAND)
    return ra_g, dec_g, ra_r, dec_r


@pytest.fixture(scope="module")
def kk_catalog():
    rng = np.random.default_rng(7)
    n = 300
    ra = rng.uniform(30.0, 60.0, n)
    dec = rng.uniform(-10.0, 10.0, n)
    k = rng.standard_normal(n)
    return ra, dec, k


class TestMeasureTwoPointTreecorr:
    def test_output_shapes(self, sky_patch):
        ra_g, dec_g, ra_r, dec_r = sky_patch
        theta, w = measure_two_point_function(
            ra_g, dec_g, ra_r, dec_r,
            min_sep=1.0, max_sep=500.0, nbins=5, sep_units="arcmin",
        )
        assert theta.shape == (5,)
        assert w.shape == (5,)

    def test_theta_increasing(self, sky_patch):
        ra_g, dec_g, ra_r, dec_r = sky_patch
        theta, _ = measure_two_point_function(
            ra_g, dec_g, ra_r, dec_r,
            min_sep=1.0, max_sep=500.0, nbins=5, sep_units="arcmin",
        )
        assert np.all(np.diff(theta) > 0)


@corrfunc_available
class TestMeasureTwoPointCorrfunc:
    def test_output_shapes(self, sky_patch):
        ra_g, dec_g, ra_r, dec_r = sky_patch
        theta, w = measure_two_point_function_corrfunc(
            ra_g, dec_g, ra_r, dec_r,
            min_sep=1.0, max_sep=500.0, nbins=5, sep_units="arcmin",
        )
        assert theta.shape == (5,)
        assert w.shape == (5,)

    def test_theta_increasing(self, sky_patch):
        ra_g, dec_g, ra_r, dec_r = sky_patch
        theta, _ = measure_two_point_function_corrfunc(
            ra_g, dec_g, ra_r, dec_r,
            min_sep=1.0, max_sep=500.0, nbins=5, sep_units="arcmin",
        )
        assert np.all(np.diff(theta) > 0)


class TestMeasureKKTreecorr:
    def test_auto_correlation_shape(self, kk_catalog):
        ra, dec, k = kk_catalog
        theta, xi = measure_kk_correlation_treecorr(
            ra, dec, k, min_sep=60, max_sep=1000, nbins=5,
        )
        assert theta.shape == (5,)
        assert xi.shape == (5,)

    def test_cross_correlation_shape(self, kk_catalog):
        ra, dec, k = kk_catalog
        theta, xi = measure_kk_correlation_treecorr(
            ra, dec, k, ra2=ra, dec2=dec, k2=k,
            min_sep=60, max_sep=1000, nbins=5,
        )
        assert theta.shape == (5,)
        assert xi.shape == (5,)


@pytest.fixture(scope="module")
def kk_field():
    """Denser scalar field with a wider area, for the patch jack-knife."""
    rng = np.random.default_rng(11)
    n = 2000
    ra = rng.uniform(20.0, 70.0, n)
    dec = rng.uniform(-15.0, 15.0, n)
    k = rng.standard_normal(n)
    return ra, dec, k


class TestMeasureKKCovarianceTreecorr:
    def test_shapes_and_symmetry(self, kk_field):
        ra, dec, k = kk_field
        theta, xi, cov, centers = measure_kk_covariance_treecorr(
            ra, dec, k, npatch=16, min_sep=6, max_sep=60, nbins=5,
        )
        assert theta.shape == (5,)
        assert xi.shape == (5,)
        assert cov.shape == (5, 5)
        assert centers.shape == (16, 3)
        np.testing.assert_allclose(cov, cov.T, atol=1e-14)
        # jack-knife variances are positive
        assert np.all(np.diag(cov) > 0)

    def test_patch_centers_reuse(self, kk_field):
        """Passing the returned centres back reproduces the same geometry (and xi)."""
        ra, dec, k = kk_field
        theta1, xi1, _, centers = measure_kk_covariance_treecorr(
            ra, dec, k, npatch=16, min_sep=6, max_sep=60, nbins=5,
        )
        theta2, xi2, cov2, centers2 = measure_kk_covariance_treecorr(
            ra, dec, k, patch_centers=centers, min_sep=6, max_sep=60, nbins=5,
        )
        assert centers2.shape == centers.shape
        np.testing.assert_allclose(xi2, xi1, rtol=1e-10, atol=1e-12)
        assert cov2.shape == (5, 5)


@corrfunc_available
class TestMeasureKKCorrfunc:
    def test_auto_correlation_shape(self, kk_catalog):
        ra, dec, k = kk_catalog
        theta, xi = measure_kk_correlation_corrfunc(
            ra, dec, k, min_sep=60, max_sep=1000, nbins=5,
        )
        assert theta.shape == (5,)
        assert xi.shape == (5,)

    def test_cross_correlation_shape(self, kk_catalog):
        ra, dec, k = kk_catalog
        theta, xi = measure_kk_correlation_corrfunc(
            ra, dec, k, ra2=ra, dec2=dec, k2=k,
            min_sep=60, max_sep=1000, nbins=5,
        )
        assert theta.shape == (5,)
        assert xi.shape == (5,)


class TestUnstandardisedBasisGuard:
    """A basis standardised somewhere other than where it is used is not standardised.

    Survey-property maps are normalised over their own valid region, which is
    larger than any one sample's footprint.  Restricted to the footprint their
    variances drift, and every quantity read in "standardised units" -- the
    amplitudes, the condition number, the template auto-correlations the
    two-point correction subtracts -- drifts with them.
    """

    def test_rescaled_template_warns(self):
        rng = np.random.default_rng(0)
        delta_t = rng.standard_normal((3, 5000))
        delta_t -= delta_t.mean(axis=1, keepdims=True)
        delta_t[2] *= 4.0
        with pytest.warns(RuntimeWarning, match="not standardised over the pixels"):
            sm.compute_covariance_matrix(delta_t)

    def test_standardised_basis_is_silent(self):
        rng = np.random.default_rng(1)
        delta_t = rng.standard_normal((3, 5000))
        delta_t -= delta_t.mean(axis=1, keepdims=True)
        delta_t /= delta_t.std(axis=1, keepdims=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            sm.compute_covariance_matrix(delta_t)

    def test_zero_mean_tolerance_scales_with_sample_size(self):
        # A centred draw has a sample mean of order 1/sqrt(n_pix); a fixed
        # relative threshold would fire on it every time and train the reader
        # to ignore the warning.
        rng = np.random.default_rng(2)
        for n_pix in (500, 5_000, 50_000):
            delta_t = rng.standard_normal((3, n_pix))
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                sm.compute_covariance_matrix(delta_t)

    def test_a_real_offset_still_warns(self):
        rng = np.random.default_rng(3)
        delta_t = rng.standard_normal((3, 5000)) + 0.2
        with pytest.warns(RuntimeWarning, match="zero mean"):
            sm.compute_covariance_matrix(delta_t)


@pytest.mark.slow
class TestTemplateCorrelationSupport:
    """xi_i(theta) must have support below the pixel scale.

    A template is constant within a pixel, so its correct auto-correlation at
    sub-pixel separations is its variance.  Measured on the pixel centres it
    comes back as exactly zero instead, because no pair of distinct pixels is
    that close, and the two-point correction then subtracts nothing over the
    range carrying most of the signal.
    """

    NSIDE = 32

    @staticmethod
    def _setup(nside=32, n_gal=120_000, seed=0):
        import healpy as hp

        rng = np.random.default_rng(seed)
        npix = hp.nside2npix(nside)
        _, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True)
        idx = np.where(np.abs(lat) > 30)[0]
        smooth = hp.smoothing(rng.standard_normal(npix), fwhm=np.radians(6))
        t = smooth[idx]
        t = (t - t.mean()) / t.std()

        # Galaxies drawn uniformly on the sphere and kept where they land in the
        # footprint, so they populate each pixel's area rather than its centre.
        slot = np.full(npix, -1, dtype=np.int64)
        slot[idx] = np.arange(idx.size)
        lon_r = rng.uniform(0.0, 360.0, 6 * n_gal)
        lat_r = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, 6 * n_gal)))
        sl = slot[hp.ang2pix(nside, lon_r, lat_r, lonlat=True)]
        keep = sl >= 0
        lon_r, lat_r, sl = lon_r[keep][:n_gal], lat_r[keep][:n_gal], sl[keep][:n_gal]

        p_lon, p_lat = hp.pix2ang(nside, idx, lonlat=True)
        return t, (p_lon, p_lat), (lon_r, lat_r, t[sl]), nside

    @staticmethod
    def _kk(lon, lat, k):
        treecorr = pytest.importorskip("treecorr")
        cat = treecorr.Catalog(ra=lon, dec=lat, ra_units="degrees",
                               dec_units="degrees", k=k)
        corr = treecorr.KKCorrelation(min_sep=0.5, max_sep=300.0, nbins=30,
                                      sep_units="arcmin", bin_slop=0.01)
        corr.process(cat)
        return corr.rnom, corr.xi

    def test_pixel_grid_has_no_support_below_the_pixel_scale(self):
        import healpy as hp

        t, (p_lon, p_lat), _, nside = self._setup(self.NSIDE)
        theta, xi = self._kk(p_lon, p_lat, t)
        # The zero boundary sits at the nearest-neighbour spacing between pixel
        # centres, which is below nside2resol (a mean spacing); compare well
        # inside it rather than at the boundary bin.
        sub = theta < 0.7 * hp.nside2resol(nside, arcmin=True)
        assert sub.sum() > 15, "test needs bins below the pixel scale"
        assert np.all(xi[sub] == 0.0)
        assert np.sum(xi == 0.0) >= 20

    def test_galaxy_measurement_recovers_the_variance_there(self):
        import healpy as hp

        t, _, (g_lon, g_lat, g_k), nside = self._setup(self.NSIDE)
        theta, xi = self._kk(g_lon, g_lat, g_k)
        sub = theta < 0.7 * hp.nside2resol(nside, arcmin=True)
        # Non-zero everywhere, and equal to the variance below the pixel scale.
        assert np.sum(xi == 0.0) == 0
        assert np.allclose(xi[sub] / float(np.var(t)), 1.0, rtol=0.25)

    def test_the_two_agree_above_the_pixel_scale(self):
        import healpy as hp

        t, (p_lon, p_lat), (g_lon, g_lat, g_k), nside = self._setup(self.NSIDE)
        theta, xi_pix = self._kk(p_lon, p_lat, t)
        _, xi_gal = self._kk(g_lon, g_lat, g_k)
        over = (theta > 1.5 * hp.nside2resol(nside, arcmin=True)) & (xi_pix > 0.05)
        assert over.sum() > 2
        assert np.allclose(xi_gal[over], xi_pix[over], rtol=0.2)


class TestTemplateCorrelationMatrix:
    """The full xi_ij(theta), cross terms included, from galaxy-carried values."""

    @staticmethod
    def _field(n=40000, seed=0):
        rng = np.random.default_rng(seed)
        ra = rng.uniform(0, 60, n)
        dec = np.degrees(np.arcsin(rng.uniform(-0.3, 0.6, n)))
        a = np.sin(np.radians(ra) * 6)
        k = np.vstack([a, 0.7 * a + 0.3 * np.cos(np.radians(dec) * 9),
                       rng.standard_normal(n) * 0.2])
        return ra, dec, k

    KW = dict(min_sep=1.0, max_sep=200.0, nbins=12)

    def test_symmetric_with_autos_on_the_diagonal(self):
        treecorr = pytest.importorskip("treecorr")
        ra, dec, k = self._field()
        _, xi = sm.template_correlation_matrix(ra, dec, k, **self.KW)
        assert xi.shape == (3, 3, 12)
        np.testing.assert_allclose(xi, xi.transpose(1, 0, 2), atol=1e-12)
        for i in range(3):
            _, auto = sm.measure_kk_correlation_treecorr(ra, dec, k[i], **self.KW)
            np.testing.assert_allclose(xi[i, i], auto, atol=1e-12)

    def test_cross_terms_are_not_negligible_for_correlated_templates(self):
        # The situation the auto-only correction gets wrong: two templates sharing a
        # large-scale mode.  Their cross correlation is of the size of the autos.
        pytest.importorskip("treecorr")
        ra, dec, k = self._field()
        _, xi = sm.template_correlation_matrix(ra, dec, k, **self.KW)
        assert np.max(np.abs(xi[0, 1])) > 0.5 * np.max(np.abs(xi[1, 1]))
        # ...and an independent noise template carries essentially none.
        assert np.max(np.abs(xi[0, 2])) < 0.05 * np.max(np.abs(xi[0, 0]))

    def test_subsampling_preserves_the_matrix(self):
        pytest.importorskip("treecorr")
        ra, dec, k = self._field(n=80000)
        _, full = sm.template_correlation_matrix(ra, dec, k, **self.KW)
        _, sub = sm.template_correlation_matrix(ra, dec, k, max_points=20000, **self.KW)
        big = np.abs(full) > 0.05
        np.testing.assert_allclose(sub[big], full[big], rtol=0.15)

    def test_the_matrix_feeds_the_correction(self):
        # A 3-D matrix must reach the full-sum branch and change the answer when the
        # templates are correlated, or the cross terms were never used.
        pytest.importorskip("treecorr")
        ra, dec, k = self._field()
        _, xi = sm.template_correlation_matrix(ra, dec, k, **self.KW)
        w_obs = np.linspace(0.5, 0.01, 12)
        a = np.array([0.05, 0.05, 0.0])
        z = np.zeros(3)
        full = sm.correct_two_point_function(w_obs, a, z, z, z, xi)
        auto = sm.correct_two_point_function(
            w_obs, a, z, z, z, np.stack([xi[i, i] for i in range(3)]))
        assert not np.allclose(full, auto)
