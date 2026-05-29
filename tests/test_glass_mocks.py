"""Tests for sys_mapping.glass_mocks — GLASS-based full-sky mock generation."""

from __future__ import annotations

import os

import numpy as np
import pytest

UCHUU_DATA = os.path.expanduser(
    "~/data/Uchuu/FullSky/mock_catalogues/"
    "MOCK_VLIM_ANY_10.65_Mstar_12.0_0.05_z_0.26_N_0923373/"
    "MOCK_VLIM_ANY_10.65_Mstar_12.0_0.05_z_0.26_N_0923373_DATA.fits"
)
UCHUU_RAND = UCHUU_DATA.replace("_DATA.fits", "_RAND.fits")


# ── measure_nz ─────────────────────────────────────────────────────────────────


class TestMeasureNz:
    def test_output_shapes(self):
        from sys_mapping.glass_mocks import measure_nz

        rng = np.random.default_rng(0)
        z = rng.uniform(0.05, 0.26, 1000)
        edges, nz = measure_nz(z, 0.05, 0.26, n_bins=10)
        assert edges.shape == (11,)
        assert nz.shape == (10,)

    def test_edges_span_range(self):
        from sys_mapping.glass_mocks import measure_nz

        rng = np.random.default_rng(1)
        z = rng.uniform(0.05, 0.26, 1000)
        edges, _ = measure_nz(z, 0.05, 0.26, n_bins=15)
        assert np.isclose(edges[0], 0.05)
        assert np.isclose(edges[-1], 0.26)

    def test_counts_sum_leq_total(self):
        from sys_mapping.glass_mocks import measure_nz

        rng = np.random.default_rng(2)
        z = rng.uniform(0.0, 0.30, 500)
        edges, nz = measure_nz(z, 0.05, 0.26, n_bins=5)
        # Galaxies outside [z_min, z_max] are excluded
        assert nz.sum() <= 500
        assert nz.sum() >= 0

    def test_non_negative_counts(self):
        from sys_mapping.glass_mocks import measure_nz

        rng = np.random.default_rng(3)
        z = rng.uniform(0.05, 0.26, 2000)
        _, nz = measure_nz(z, 0.05, 0.26, n_bins=10)
        assert np.all(nz >= 0)


# ── generate_glass_fullsky_mock ────────────────────────────────────────────────


class TestGenerateGlassFullskyMock:
    """Use a small nside=16 and n_total=2000 for speed."""

    @pytest.fixture(scope="class")
    def mock(self):
        from sys_mapping.glass_mocks import generate_glass_fullsky_mock, measure_nz

        rng = np.random.default_rng(42)
        z_ref = rng.uniform(0.05, 0.26, 5000)
        z_edges, nz = measure_nz(z_ref, 0.05, 0.26, n_bins=5)
        return generate_glass_fullsky_mock(nside=16, n_total=2000, z_edges=z_edges, nz=nz, seed=0)

    def test_has_required_keys(self, mock):
        for key in ("ra", "dec", "z", "ra_rand", "dec_rand", "n_total", "nside", "seed"):
            assert key in mock, f"Missing key: {key}"

    def test_galaxies_nonempty(self, mock):
        assert len(mock["ra"]) > 0

    def test_ra_range(self, mock):
        assert mock["ra"].min() >= 0.0
        assert mock["ra"].max() < 360.0

    def test_dec_range(self, mock):
        assert mock["dec"].min() >= -90.0
        assert mock["dec"].max() <= 90.0

    def test_z_range(self, mock):
        # Redshifts should be within [z_min, z_max] = [0.05, 0.26]
        assert mock["z"].min() >= 0.04
        assert mock["z"].max() <= 0.27

    def test_rand_ra_range(self, mock):
        assert mock["ra_rand"].min() >= 0.0
        assert mock["ra_rand"].max() <= 360.0

    def test_rand_dec_range(self, mock):
        assert mock["dec_rand"].min() >= -90.0
        assert mock["dec_rand"].max() <= 90.0

    def test_reproducibility(self):
        from sys_mapping.glass_mocks import generate_glass_fullsky_mock, measure_nz

        rng = np.random.default_rng(99)
        z_ref = rng.uniform(0.05, 0.26, 2000)
        z_edges, nz = measure_nz(z_ref, 0.05, 0.26, n_bins=4)
        m1 = generate_glass_fullsky_mock(nside=8, n_total=500, z_edges=z_edges, nz=nz, seed=77)
        m2 = generate_glass_fullsky_mock(nside=8, n_total=500, z_edges=z_edges, nz=nz, seed=77)
        np.testing.assert_array_equal(m1["ra"], m2["ra"])
        np.testing.assert_array_equal(m1["dec"], m2["dec"])

    def test_different_seeds_differ(self):
        from sys_mapping.glass_mocks import generate_glass_fullsky_mock, measure_nz

        rng = np.random.default_rng(11)
        z_ref = rng.uniform(0.05, 0.26, 2000)
        z_edges, nz = measure_nz(z_ref, 0.05, 0.26, n_bins=4)
        m1 = generate_glass_fullsky_mock(nside=8, n_total=500, z_edges=z_edges, nz=nz, seed=1)
        m2 = generate_glass_fullsky_mock(nside=8, n_total=500, z_edges=z_edges, nz=nz, seed=2)
        assert not np.array_equal(m1["ra"], m2["ra"])

    def test_density_approx_target(self, mock):
        # Actual count should be within 20% of target
        target = 2000
        actual = len(mock["ra"])
        assert abs(actual - target) / target < 0.25

    def test_rand_factor(self):
        from sys_mapping.glass_mocks import generate_glass_fullsky_mock, measure_nz

        rng = np.random.default_rng(5)
        z_ref = rng.uniform(0.05, 0.26, 1000)
        z_edges, nz = measure_nz(z_ref, 0.05, 0.26, n_bins=3)
        m = generate_glass_fullsky_mock(
            nside=8, n_total=500, z_edges=z_edges, nz=nz, rand_factor=5, seed=0
        )
        # Randoms should be ~5× galaxies
        ratio = len(m["ra_rand"]) / m["n_total"]
        assert 4 <= ratio <= 6


# ── load_uchuu_mock (skip if data absent) ─────────────────────────────────────


@pytest.mark.skipif(not os.path.exists(UCHUU_DATA), reason="Uchuu data not available")
class TestLoadUchuuMock:
    @pytest.fixture(scope="class")
    def uchuu(self):
        from sys_mapping.simulation import load_uchuu_mock

        return load_uchuu_mock(UCHUU_DATA, UCHUU_RAND)

    def test_keys_present(self, uchuu):
        for key in ("ra", "dec", "z", "ra_rand", "dec_rand"):
            assert key in uchuu

    def test_galaxy_count(self, uchuu):
        assert len(uchuu["ra"]) == 923_373

    def test_ra_range(self, uchuu):
        assert uchuu["ra"].min() >= 0.0
        assert uchuu["ra"].max() <= 360.0

    def test_dec_range(self, uchuu):
        assert uchuu["dec"].min() >= -90.0
        assert uchuu["dec"].max() <= 90.0

    def test_redshift_range(self, uchuu):
        assert uchuu["z"].min() >= 0.05
        assert uchuu["z"].max() <= 0.27

    def test_nz_from_uchuu(self, uchuu):
        from sys_mapping.glass_mocks import measure_nz

        z_edges, nz = measure_nz(uchuu["z"], 0.05, 0.26, n_bins=20)
        assert nz.sum() <= len(uchuu["z"])
        assert nz.sum() > 0
        assert z_edges.shape == (21,)
        assert nz.shape == (20,)
