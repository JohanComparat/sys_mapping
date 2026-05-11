"""Tests for maps.py: HEALPix template generation and catalog pixelization."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import healpy as hp

from sys_mapping.maps import (
    systematic_power_spectrum,
    generate_systematic_map,
    generate_systematic_maps,
    load_real_template,
    load_real_templates,
    pixelize_catalog,
    compute_overdensity,
    assign_template_values,
)

# Paths to real data files — tests requiring these are skipped if absent.
_SYST_DIR = Path("~/data/legacysurvey/dr10/systematics").expanduser()
_GAIA_PATH = _SYST_DIR / "GAIA_nstar_faint_NSIDE_00064.fits"
_LS10_PATH = _SYST_DIR / "LS10_GALDEPTH_Z_NSIDE_0064.fits"
_HAS_REAL_DATA = _GAIA_PATH.exists() and _LS10_PATH.exists()
real_data = pytest.mark.skipif(not _HAS_REAL_DATA, reason="LS10/GAIA FITS files not found")

NSIDE = 32  # small for fast tests


class TestPowerSpectrum:
    def test_all_families_sum_to_unit_variance(self):
        for family in range(5):
            cl = systematic_power_spectrum(NSIDE, family)
            lmax = 3 * NSIDE - 1
            ell = np.arange(lmax + 1, dtype=float)
            var = np.sum((2 * ell + 1) / (4 * np.pi) * cl)
            np.testing.assert_allclose(var, 1.0, rtol=1e-10, err_msg=f"family={family}")

    def test_monopole_zero(self):
        for family in range(5):
            cl = systematic_power_spectrum(NSIDE, family)
            assert cl[0] == 0.0

    def test_invalid_family(self):
        with pytest.raises(ValueError):
            systematic_power_spectrum(NSIDE, 99)

    def test_shapes(self):
        for family in range(5):
            cl = systematic_power_spectrum(NSIDE, family)
            assert cl.shape == (3 * NSIDE,)


class TestGenerateMap:
    def test_unit_variance(self):
        for family in range(5):
            m = generate_systematic_map(NSIDE, family, seed=family)
            np.testing.assert_allclose(np.std(m), 1.0, atol=0.05, err_msg=f"family={family}")

    def test_zero_mean(self):
        for family in range(5):
            m = generate_systematic_map(NSIDE, family, seed=family)
            np.testing.assert_allclose(np.mean(m), 0.0, atol=1e-10, err_msg=f"family={family}")

    def test_correct_npix(self):
        m = generate_systematic_map(NSIDE, 0, seed=0)
        assert m.shape == (hp.nside2npix(NSIDE),)

    def test_reproducible_with_seed(self):
        m1 = generate_systematic_map(NSIDE, 0, seed=42)
        m2 = generate_systematic_map(NSIDE, 0, seed=42)
        np.testing.assert_array_equal(m1, m2)


class TestGenerateMaps:
    def test_default_five_families(self):
        maps = generate_systematic_maps(NSIDE, seed=0)
        assert maps.shape[0] == 5
        assert maps.shape[1] == hp.nside2npix(NSIDE)

    def test_custom_families(self):
        maps = generate_systematic_maps(NSIDE, families=[0, 2], seed=0)
        assert maps.shape[0] == 2

    def test_each_map_unit_variance(self):
        maps = generate_systematic_maps(NSIDE, seed=0)
        for i in range(maps.shape[0]):
            np.testing.assert_allclose(np.std(maps[i]), 1.0, atol=0.05, err_msg=f"map {i}")


class TestPixelizeCatalog:
    def test_total_count_preserved(self):
        rng = np.random.default_rng(0)
        n_gal = 1000
        ra = rng.uniform(0, 360, n_gal)
        dec = rng.uniform(-60, 60, n_gal)
        counts = pixelize_catalog(ra, dec, NSIDE)
        assert int(np.sum(counts)) == n_gal

    def test_output_shape(self):
        rng = np.random.default_rng(1)
        ra = rng.uniform(0, 360, 500)
        dec = rng.uniform(-30, 30, 500)
        counts = pixelize_catalog(ra, dec, NSIDE)
        assert counts.shape == (hp.nside2npix(NSIDE),)

    def test_weighted_counts_sum(self):
        rng = np.random.default_rng(2)
        n_gal = 500
        ra = rng.uniform(0, 360, n_gal)
        dec = rng.uniform(-30, 30, n_gal)
        weights = rng.uniform(0.5, 1.5, n_gal)
        counts = pixelize_catalog(ra, dec, NSIDE, weights=weights)
        np.testing.assert_allclose(np.sum(counts), np.sum(weights), rtol=1e-10)


class TestComputeOverdensity:
    def test_mean_approximately_zero(self):
        rng = np.random.default_rng(3)
        n_pix = hp.nside2npix(NSIDE)
        # Uniform distribution
        gal = rng.poisson(10, n_pix).astype(float)
        rand = rng.poisson(100, n_pix).astype(float)
        delta_g, mask = compute_overdensity(gal, rand)
        np.testing.assert_allclose(np.mean(delta_g), 0.0, atol=0.1)

    def test_masking_removes_low_randoms(self):
        n_pix = 100
        gal = np.ones(n_pix)
        rand = np.ones(n_pix)
        rand[:10] = 0.0  # should be masked
        _, mask = compute_overdensity(gal, rand, min_random_fraction=0.1)
        assert not np.any(mask[:10])
        assert np.all(mask[10:])


class TestAssignTemplateValues:
    def test_shape(self):
        n_pix = hp.nside2npix(NSIDE)
        n_sys = 4
        templates = np.random.default_rng(0).standard_normal((n_sys, n_pix))
        mask = np.zeros(n_pix, dtype=bool)
        mask[:50] = True
        result = assign_template_values(templates, mask)
        assert result.shape == (n_sys, 50)


# ── Tests for load_real_template (synthetic FITS fixture) ──────────────────

def _write_mock_fits(path: Path, column: str, data: np.ndarray):
    """Write a minimal BinTableHDU FITS file for testing."""
    from astropy.io import fits as afits
    from astropy.table import Table
    t = Table({column: data})
    t.write(str(path), format="fits", overwrite=True)


class TestLoadRealTemplateSynthetic:
    """Tests using a temporary synthetic FITS file — no real data required."""

    def test_normalisation_full_sky(self, tmp_path):
        """Full-sky map: template mean=0 and std=1 over all pixels."""
        nside = 32
        n_pix = hp.nside2npix(nside)
        rng = np.random.default_rng(0)
        raw = rng.uniform(1.0, 5.0, n_pix)  # all positive, full sky
        p = tmp_path / "test.fits"
        _write_mock_fits(p, "my_col", raw)

        t, mask = load_real_template(p, "my_col")
        assert t.shape == (n_pix,)
        assert mask.all()
        np.testing.assert_allclose(t[mask].mean(), 0.0, atol=1e-10)
        np.testing.assert_allclose(t[mask].std(), 1.0, atol=1e-10)

    def test_outside_footprint_set_to_zero(self, tmp_path):
        """Pixels with value <= valid_min are set to 0 in the template."""
        nside = 32
        n_pix = hp.nside2npix(nside)
        raw = np.ones(n_pix)
        raw[:100] = 0.0            # 100 outside-footprint pixels
        p = tmp_path / "test.fits"
        _write_mock_fits(p, "depth", raw)

        t, mask = load_real_template(p, "depth", valid_min=0.0)
        assert not mask[:100].any()
        assert mask[100:].all()
        np.testing.assert_array_equal(t[:100], 0.0)

    def test_valid_mask_correct(self, tmp_path):
        """valid_mask correctly identifies pixels above valid_min."""
        nside = 16
        n_pix = hp.nside2npix(nside)
        raw = np.full(n_pix, 2.0)
        raw[::3] = 0.0             # every 3rd pixel is zero
        p = tmp_path / "test.fits"
        _write_mock_fits(p, "col", raw)

        _, mask = load_real_template(p, "col", valid_min=0.0)
        expected = raw > 0.0
        np.testing.assert_array_equal(mask, expected)

    def test_invalid_column_raises(self, tmp_path):
        nside = 16
        raw = np.ones(hp.nside2npix(nside))
        p = tmp_path / "test.fits"
        _write_mock_fits(p, "col_a", raw)
        with pytest.raises(ValueError, match="not found"):
            load_real_template(p, "col_b")

    def test_ud_grade_applied(self, tmp_path):
        """Requesting a different NSIDE triggers ud_grade."""
        n_pix_128 = hp.nside2npix(128)
        raw = np.ones(n_pix_128)
        p = tmp_path / "test.fits"
        _write_mock_fits(p, "col", raw)

        t, mask = load_real_template(p, "col", nside=64)
        assert t.shape == (hp.nside2npix(64),)


# ── Tests for load_real_template using real GAIA / LS10 files ──────────────

class TestLoadRealTemplateGAIA:
    @real_data
    def test_shape(self):
        t, mask = load_real_template(_GAIA_PATH, "nstar_faint")
        assert t.shape == (hp.nside2npix(64),)

    @real_data
    def test_full_sky_gaia(self):
        """GAIA nstar_faint covers the full sky — valid_mask should be all True."""
        _, mask = load_real_template(_GAIA_PATH, "nstar_faint")
        assert mask.all(), "GAIA nstar_faint expected to be full-sky"

    @real_data
    def test_mean_zero_std_one_over_valid(self):
        t, mask = load_real_template(_GAIA_PATH, "nstar_faint")
        np.testing.assert_allclose(t[mask].mean(), 0.0, atol=1e-10)
        np.testing.assert_allclose(t[mask].std(), 1.0, atol=1e-10)


class TestLoadRealTemplateLS10Depth:
    @real_data
    def test_shape(self):
        t, mask = load_real_template(_LS10_PATH, "GALDEPTH_Z")
        assert t.shape == (hp.nside2npix(64),)

    @real_data
    def test_partial_footprint(self):
        """LS10 depth map covers only ~46% of the sky."""
        _, mask = load_real_template(_LS10_PATH, "GALDEPTH_Z")
        frac = mask.sum() / len(mask)
        assert 0.30 < frac < 0.65, f"unexpected footprint fraction: {frac:.2f}"

    @real_data
    def test_outside_footprint_is_zero(self):
        t, mask = load_real_template(_LS10_PATH, "GALDEPTH_Z")
        np.testing.assert_array_equal(t[~mask], 0.0)

    @real_data
    def test_mean_zero_std_one_over_valid(self):
        t, mask = load_real_template(_LS10_PATH, "GALDEPTH_Z")
        np.testing.assert_allclose(t[mask].mean(), 0.0, atol=1e-10)
        np.testing.assert_allclose(t[mask].std(), 1.0, atol=1e-10)


class TestLoadRealTemplatesMocked:
    """Tests for load_real_templates using mocked load_real_template — no real data needed."""

    def test_shape_and_names(self, tmp_path):
        from unittest.mock import patch
        nside = 32
        n_pix = hp.nside2npix(nside)
        rng = np.random.default_rng(0)
        fake_t = rng.standard_normal(n_pix)
        fake_mask = np.ones(n_pix, dtype=bool)

        with patch("sys_mapping.maps.load_real_template", return_value=(fake_t, fake_mask)):
            templates, names, mask = load_real_templates(nside, tmp_path)

        assert templates.shape == (2, n_pix)
        assert names == ["GAIA_nstar_faint", "LS10_GALDEPTH_Z"]

    def test_mask_is_intersection(self, tmp_path):
        from unittest.mock import patch
        nside = 32
        n_pix = hp.nside2npix(nside)
        mask_gaia = np.ones(n_pix, dtype=bool)
        mask_depth = np.ones(n_pix, dtype=bool)
        mask_depth[:10] = False  # depth misses first 10 pixels

        fake_t = np.zeros(n_pix)
        side_effects = [(fake_t, mask_gaia), (fake_t, mask_depth)]

        with patch("sys_mapping.maps.load_real_template", side_effect=side_effects):
            _, _, mask = load_real_templates(nside, tmp_path)

        assert not mask[:10].any()
        assert mask[10:].all()


class TestLoadRealTemplates:
    @real_data
    def test_shape(self):
        templates, names, mask = load_real_templates(64, _SYST_DIR)
        assert templates.shape == (2, hp.nside2npix(64))

    @real_data
    def test_names(self):
        _, names, _ = load_real_templates(64, _SYST_DIR)
        assert names == ["GAIA_nstar_faint", "LS10_GALDEPTH_Z"]

    @real_data
    def test_valid_mask_is_intersection(self):
        templates, _, mask = load_real_templates(64, _SYST_DIR)
        # LS10 depth is not full-sky, so intersection < total pixels
        assert mask.sum() < hp.nside2npix(64)
        # But must cover at least 20% of sky
        assert mask.sum() > 0.20 * hp.nside2npix(64)
