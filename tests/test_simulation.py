"""Tests for sys_mapping.simulation — contamination injection, FITS I/O, recovery."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

SYST_DIR = os.path.expanduser("~/data/legacysurvey/dr10/systematics/")
SYST_DIR_AVAILABLE = os.path.isdir(os.path.join(SYST_DIR, "0032"))

UCHUU_DATA = os.path.expanduser(
    "~/data/Uchuu/FullSky/mock_catalogues/"
    "MOCK_VLIM_ANY_10.65_Mstar_12.0_0.05_z_0.26_N_0923373/"
    "MOCK_VLIM_ANY_10.65_Mstar_12.0_0.05_z_0.26_N_0923373_DATA.fits"
)
UCHUU_RAND = UCHUU_DATA.replace("_DATA.fits", "_RAND.fits")


# ── Shared synthetic fixture ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_catalog():
    """Small full-sky synthetic catalog for fast unit tests (nside=32)."""
    rng = np.random.default_rng(0)
    n_gal = 5_000
    n_rand = 50_000
    ra = rng.uniform(0, 360, n_gal)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n_gal)))
    z = rng.uniform(0.05, 0.26, n_gal)
    ra_rand = rng.uniform(0, 360, n_rand)
    dec_rand = np.degrees(np.arcsin(rng.uniform(-1, 1, n_rand)))
    return {"ra": ra, "dec": dec, "z": z, "ra_rand": ra_rand, "dec_rand": dec_rand}


@pytest.fixture(scope="module")
def synthetic_templates():
    """Small set of (3, 12*32²) zero-mean unit-variance templates."""
    from sys_mapping.maps import generate_systematic_maps

    return generate_systematic_maps(nside=32, seed=7)  # (5, 12288)


# ── make_contamination_grid ────────────────────────────────────────────────────


class TestMakeContaminationGrid:
    def test_returns_nine_configs(self):
        from sys_mapping.simulation import make_contamination_grid

        cfgs = make_contamination_grid(n_sys=3)
        assert len(cfgs) == 9

    def test_level_variety(self):
        from sys_mapping.simulation import make_contamination_grid

        cfgs = make_contamination_grid(n_sys=2)
        levels = {c.level for c in cfgs}
        assert levels == {"low", "medium", "high"}

    def test_scenario_variety(self):
        from sys_mapping.simulation import make_contamination_grid

        cfgs = make_contamination_grid(n_sys=2)
        scenarios = {c.scenario for c in cfgs}
        assert scenarios == {"additive", "multiplicative", "combined"}

    def test_additive_b_is_zero(self):
        from sys_mapping.simulation import make_contamination_grid

        cfgs = make_contamination_grid(n_sys=3)
        for c in cfgs:
            if c.scenario == "additive":
                np.testing.assert_array_equal(c.b_true, np.zeros(3))

    def test_multiplicative_a_is_zero(self):
        from sys_mapping.simulation import make_contamination_grid

        cfgs = make_contamination_grid(n_sys=3)
        for c in cfgs:
            if c.scenario == "multiplicative":
                np.testing.assert_array_equal(c.a_true, np.zeros(3))

    def test_amplitude_magnitudes(self):
        from sys_mapping.simulation import LEVELS, make_contamination_grid

        cfgs = make_contamination_grid(n_sys=4, seed=0)
        for c in cfgs:
            expected_mag = LEVELS[c.level]
            for arr in (c.a_true, c.b_true):
                for v in arr:
                    if v != 0.0:
                        assert np.isclose(abs(v), expected_mag), f"Expected |{v}|={expected_mag}"

    def test_reproducible_with_seed(self):
        from sys_mapping.simulation import make_contamination_grid

        g1 = make_contamination_grid(n_sys=3, seed=42)
        g2 = make_contamination_grid(n_sys=3, seed=42)
        for c1, c2 in zip(g1, g2):
            np.testing.assert_array_equal(c1.a_true, c2.a_true)
            np.testing.assert_array_equal(c1.b_true, c2.b_true)


# ── inject_systematics ─────────────────────────────────────────────────────────


class TestInjectSystematics:
    def test_output_shape(self, synthetic_catalog, synthetic_templates):
        from sys_mapping.simulation import inject_systematics

        ra, dec = synthetic_catalog["ra"], synthetic_catalog["dec"]
        ra_r, dec_r = synthetic_catalog["ra_rand"], synthetic_catalog["dec_rand"]
        n_sys = synthetic_templates.shape[0]
        a = np.zeros(n_sys)
        b = np.zeros(n_sys)
        w = inject_systematics(ra, dec, ra_r, dec_r, synthetic_templates, a, b, nside=32)
        assert w.shape == (len(ra),)

    def test_zero_contamination_weights_near_unity(self, synthetic_catalog, synthetic_templates):
        from sys_mapping.simulation import inject_systematics

        ra, dec = synthetic_catalog["ra"], synthetic_catalog["dec"]
        ra_r, dec_r = synthetic_catalog["ra_rand"], synthetic_catalog["dec_rand"]
        n_sys = synthetic_templates.shape[0]
        a = np.zeros(n_sys)
        b = np.zeros(n_sys)
        w = inject_systematics(ra, dec, ra_r, dec_r, synthetic_templates, a, b, nside=32)
        # With no contamination, weights should be ~1 (within shot-noise scatter)
        np.testing.assert_allclose(w, 1.0, atol=0.5)

    def test_weights_all_positive(self, synthetic_catalog, synthetic_templates):
        from sys_mapping.simulation import inject_systematics

        ra, dec = synthetic_catalog["ra"], synthetic_catalog["dec"]
        ra_r, dec_r = synthetic_catalog["ra_rand"], synthetic_catalog["dec_rand"]
        n_sys = synthetic_templates.shape[0]
        a = np.ones(n_sys) * 0.05
        b = np.zeros(n_sys)
        w = inject_systematics(ra, dec, ra_r, dec_r, synthetic_templates, a, b, nside=32)
        assert np.all(w > 0)

    def test_nonzero_contamination_alters_weights(self, synthetic_catalog, synthetic_templates):
        from sys_mapping.simulation import inject_systematics

        ra, dec = synthetic_catalog["ra"], synthetic_catalog["dec"]
        ra_r, dec_r = synthetic_catalog["ra_rand"], synthetic_catalog["dec_rand"]
        n_sys = synthetic_templates.shape[0]
        a_zero = np.zeros(n_sys)
        a_nonzero = np.ones(n_sys) * 0.10
        w_clean = inject_systematics(ra, dec, ra_r, dec_r, synthetic_templates, a_zero, a_zero, nside=32)
        w_cont = inject_systematics(ra, dec, ra_r, dec_r, synthetic_templates, a_nonzero, a_zero, nside=32)
        # Weights should differ between clean and contaminated
        assert not np.allclose(w_clean, w_cont)


# ── save / load round-trip ─────────────────────────────────────────────────────


class TestSaveLoadRoundtrip:
    def test_roundtrip(self, synthetic_catalog):
        from sys_mapping.simulation import (
            ContaminationConfig,
            load_simulation_catalog,
            save_simulation_catalog,
        )

        ra, dec, z = (
            synthetic_catalog["ra"][:100],
            synthetic_catalog["dec"][:100],
            synthetic_catalog["z"][:100],
        )
        w = np.ones(100, dtype=float) * 1.23
        config = ContaminationConfig(
            level="medium",
            scenario="additive",
            a_true=np.array([0.05, -0.05, 0.05]),
            b_true=np.zeros(3),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_sim.fits")
            save_simulation_catalog(ra, dec, z, w, config, path)
            ra2, dec2, z2, w2, cfg2 = load_simulation_catalog(path)

        np.testing.assert_allclose(ra2, ra.astype(np.float32), rtol=1e-5)
        np.testing.assert_allclose(dec2, dec.astype(np.float32), rtol=1e-5)
        np.testing.assert_allclose(z2, z.astype(np.float32), rtol=1e-5)
        np.testing.assert_allclose(w2, w.astype(np.float32), rtol=1e-5)
        assert cfg2.level == "medium"
        assert cfg2.scenario == "additive"
        np.testing.assert_allclose(cfg2.a_true, config.a_true, rtol=1e-5)

    def test_parent_dir_created(self, synthetic_catalog):
        from sys_mapping.simulation import (
            ContaminationConfig,
            save_simulation_catalog,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "dir", "sim.fits")
            config = ContaminationConfig("low", "multiplicative", np.zeros(2), np.ones(2) * 0.02)
            save_simulation_catalog(
                synthetic_catalog["ra"][:10],
                synthetic_catalog["dec"][:10],
                synthetic_catalog["z"][:10],
                np.ones(10),
                config,
                path,
            )
            assert os.path.exists(path)


# ── load_systematic_maps (skip if data absent) ─────────────────────────────────


@pytest.mark.skipif(not SYST_DIR_AVAILABLE, reason="LSDR10 systematic maps not available")
class TestLoadSystematicMaps:
    def test_shape_nside32(self):
        from sys_mapping.simulation import load_systematic_maps

        templates, names, footprint = load_systematic_maps(SYST_DIR, nside=32)
        assert templates.shape == (5, 12 * 32**2)
        assert len(names) == 5
        assert footprint.shape == (12 * 32**2,)
        assert footprint.dtype == bool

    def test_maps_finite(self):
        from sys_mapping.simulation import load_systematic_maps

        templates, _, _fp = load_systematic_maps(SYST_DIR, nside=32)
        assert np.all(np.isfinite(templates))

    def test_footprint_is_subset(self):
        from sys_mapping.simulation import load_systematic_maps

        _, _, footprint = load_systematic_maps(SYST_DIR, nside=32)
        # Footprint must be a proper subset: not full-sky, not empty
        assert footprint.any()
        assert not footprint.all()

    def test_subset_selection(self):
        from sys_mapping.simulation import load_systematic_maps

        templates, names, footprint = load_systematic_maps(
            SYST_DIR, nside=32, map_names=["LS10_EBV", "LS10_GALDEPTH_Z"]
        )
        assert templates.shape == (2, 12 * 32**2)
        assert names == ["LS10_EBV", "LS10_GALDEPTH_Z"]
        assert footprint.shape == (12 * 32**2,)


# ── apply_footprint_mask ──────────────────────────────────────────────────────


class TestApplyFootprintMask:
    def test_filters_galaxies_and_randoms(self):
        import healpy as hp
        from sys_mapping.simulation import apply_footprint_mask

        nside = 16
        n_pix = hp.nside2npix(nside)
        rng = np.random.default_rng(42)

        # Build a footprint that covers only the northern hemisphere (dec > 0)
        theta_pix, _ = hp.pix2ang(nside, np.arange(n_pix))
        footprint = theta_pix < np.pi / 2  # dec > 0

        n_gal = 2000
        catalog = {
            "ra": rng.uniform(0, 360, n_gal),
            "dec": rng.uniform(-90, 90, n_gal),
            "z": rng.uniform(0.05, 0.26, n_gal),
            "ra_rand": rng.uniform(0, 360, 10 * n_gal),
            "dec_rand": rng.uniform(-90, 90, 10 * n_gal),
            "n_total": n_gal,
        }

        filtered = apply_footprint_mask(catalog, footprint, nside=nside)

        # All returned positions must be in footprint pixels
        theta_g = np.radians(90 - filtered["dec"])
        phi_g = np.radians(filtered["ra"])
        pix_g = hp.ang2pix(nside, theta_g, phi_g)
        assert footprint[pix_g].all()

        theta_r = np.radians(90 - filtered["dec_rand"])
        phi_r = np.radians(filtered["ra_rand"])
        pix_r = hp.ang2pix(nside, theta_r, phi_r)
        assert footprint[pix_r].all()

        # n_total updated
        assert filtered["n_total"] == len(filtered["ra"])

        # Roughly half the galaxies remain (northern hemisphere ≈ 50%)
        assert 0.3 * n_gal < filtered["n_total"] < 0.7 * n_gal


# ── run_wtheta_recovery (requires treecorr, uses synthetic data) ───────────────


class TestWthetaRecoveryKeys:
    """Fast smoke test on synthetic catalog + synthetic templates."""

    def test_result_has_required_keys(self, synthetic_catalog, synthetic_templates):
        from sys_mapping.simulation import ContaminationConfig, run_wtheta_recovery

        n_sys = synthetic_templates.shape[0]
        config = ContaminationConfig(
            level="medium",
            scenario="additive",
            a_true=np.ones(n_sys) * 0.05,
            b_true=np.zeros(n_sys),
        )
        result = run_wtheta_recovery(
            synthetic_catalog,
            synthetic_templates,
            template_names=[f"t{i}" for i in range(n_sys)],
            config=config,
            nside=32,
            methods=["OLS"],
            min_sep=0.5,
            max_sep=5.0,
            nbins=5,
        )
        required = {"theta", "w_true", "w_contaminated", "w_recovered_OLS", "params_OLS", "config"}
        assert required.issubset(result.keys())

    def test_theta_is_sorted(self, synthetic_catalog, synthetic_templates):
        from sys_mapping.simulation import ContaminationConfig, run_wtheta_recovery

        n_sys = synthetic_templates.shape[0]
        config = ContaminationConfig(
            level="low",
            scenario="additive",
            a_true=np.ones(n_sys) * 0.02,
            b_true=np.zeros(n_sys),
        )
        result = run_wtheta_recovery(
            synthetic_catalog,
            synthetic_templates,
            template_names=[f"t{i}" for i in range(n_sys)],
            config=config,
            nside=32,
            methods=["OLS"],
            min_sep=0.5,
            max_sep=5.0,
            nbins=5,
        )
        theta = result["theta"]
        assert np.all(theta[1:] > theta[:-1])

    def test_no_contamination_consistent(self, synthetic_catalog, synthetic_templates):
        """With zero contamination, w_true should equal w_contaminated."""
        from sys_mapping.simulation import ContaminationConfig, run_wtheta_recovery

        n_sys = synthetic_templates.shape[0]
        config = ContaminationConfig(
            level="low",
            scenario="additive",
            a_true=np.zeros(n_sys),
            b_true=np.zeros(n_sys),
        )
        result = run_wtheta_recovery(
            synthetic_catalog,
            synthetic_templates,
            template_names=[f"t{i}" for i in range(n_sys)],
            config=config,
            nside=32,
            methods=["OLS"],
            min_sep=0.5,
            max_sep=5.0,
            nbins=5,
        )
        # With no contamination, contaminated should equal truth (within float precision)
        np.testing.assert_allclose(result["w_contaminated"], result["w_true"], rtol=1e-4)
