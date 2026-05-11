"""Tests for sys_mapping.utils — two-point measurement utilities."""

import numpy as np
import pytest

from sys_mapping.utils import (
    measure_two_point_function,
    measure_two_point_function_corrfunc,
    measure_kk_correlation_treecorr,
    measure_kk_correlation_corrfunc,
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
