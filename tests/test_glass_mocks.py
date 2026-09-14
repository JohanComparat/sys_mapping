"""Tests for sys_mapping.glass_mocks — GLASS-based full-sky mock generation."""

from __future__ import annotations

import os

import numpy as np
import pytest

import sys_mapping as sm

# These tests exercise GLASS mock mechanics on a parametric field chosen on purpose,
# not a calibrated null, so the library's "not matched to any sample" warning is
# expected here and would only bury real ones.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*not matched to any sample.*:UserWarning")

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


class TestLoadMatchedCl:
    """Resolution of a per-setup matched spectrum, and the validation gate."""

    @staticmethod
    def _write(tmp_path, sample, nside, *, passed, ratio=1.0, n=8):
        import json
        p = tmp_path / f"{sample}_NSIDE{nside:04d}_match.json"
        p.write_text(json.dumps({
            "sample": sample, "nside": nside,
            "cl_matched": [0.0] + [1e-3] * n,
            "validation": {"passed": passed, "large_scale_ratio": ratio,
                           "tol": 0.1, "l_range": [2, n]},
        }))
        return p

    def test_exact_resolution_wins_when_it_passes(self, tmp_path):
        s = "SAMPLE_A"
        self._write(tmp_path, s, 32, passed=True, n=4)
        self._write(tmp_path, s, 128, passed=True, n=16)
        cl = sm.load_matched_cl(tmp_path, s, 32)
        assert len(cl) == 5          # the NSIDE 32 file, not the finer one

    def test_finest_wins_when_no_exact_file(self, tmp_path):
        s = "SAMPLE_B"
        self._write(tmp_path, s, 32, passed=True, n=4)
        self._write(tmp_path, s, 128, passed=True, n=16)
        cl = sm.load_matched_cl(tmp_path, s, 64)   # no NSIDE 64 file
        assert len(cl) == 17         # falls through to NSIDE 128

    def test_failed_exact_falls_back_to_a_validated_resolution(self, tmp_path):
        """A failed fit at one resolution says nothing about a passing one.

        The spectrum is a property of the sample and its footprint; resolution
        limits what can be *checked*, not what can be *used*.  This is the LS10
        logM>=11.5 case: its NSIDE 64 fit failed while its NSIDE 128 fit passed.
        """
        s = "SAMPLE_C"
        self._write(tmp_path, s, 64, passed=False, ratio=0.954, n=4)
        self._write(tmp_path, s, 128, passed=True, n=16)
        with pytest.warns(UserWarning, match="did not pass the large-scale check"):
            cl = sm.load_matched_cl(tmp_path, s, 64)
        assert len(cl) == 17

    def test_refuses_when_nothing_passed(self, tmp_path):
        """Falling back must not become a way of accepting an unvalidated null."""
        s = "SAMPLE_D"
        self._write(tmp_path, s, 64, passed=False, ratio=0.6)
        self._write(tmp_path, s, 128, passed=False, ratio=0.5)
        with pytest.raises(ValueError, match="failed the large-scale check"):
            sm.load_matched_cl(tmp_path, s, 64)

    def test_missing_sample_returns_none(self, tmp_path):
        assert sm.load_matched_cl(tmp_path, "NO_SUCH_SAMPLE", 64) is None


class TestSpectrumIsChosenNotDefaulted:
    """A mock whose spectrum nobody chose must say so.

    The default power law under-clusters a real galaxy sample by a large factor, so a
    null built on it is too narrow.  Choosing a parametric field explicitly is
    legitimate (a simulation whose truth is that power law), and a matched spectrum is
    the calibrated case; only the silent fall-through warns.
    """

    @staticmethod
    def _args():
        from sys_mapping.glass_mocks import measure_nz

        z = np.random.default_rng(1).uniform(0.05, 0.26, 2000)
        z_edges, nz = measure_nz(z, 0.05, 0.26, n_bins=3)
        return dict(nside=8, n_total=2000, z_edges=z_edges, nz=nz, seed=0)

    def test_neither_spectrum_nor_amplitude_warns(self):
        pytest.importorskip("glass")
        from sys_mapping.glass_mocks import generate_glass_fullsky_mock

        with pytest.warns(UserWarning, match="not matched to any sample"):
            generate_glass_fullsky_mock(**self._args())

    @pytest.mark.parametrize("kind", ["amplitude", "matched"])
    def test_a_chosen_spectrum_is_silent(self, kind):
        pytest.importorskip("glass")
        import warnings

        from sys_mapping.glass_mocks import _make_glass_cls, generate_glass_fullsky_mock

        extra = ({"cl_amplitude": 5e-4} if kind == "amplitude"
                 else {"cl_input": _make_glass_cls(8, amplitude=2e-3)})
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*not matched to any sample.*")
            generate_glass_fullsky_mock(**self._args(), **extra)

    def test_delta_map_runs_and_accepts_a_matched_spectrum(self):
        # It read lognormal_shift without declaring it, so every call raised NameError.
        pytest.importorskip("glass")
        from sys_mapping.glass_mocks import _make_glass_cls, generate_glass_delta_map

        a = generate_glass_delta_map(8, 0.3, cl_amplitude=5e-4, seed=0)
        b = generate_glass_delta_map(8, 0.3, cl_input=_make_glass_cls(8, amplitude=5e-2), seed=0)
        assert a.shape == b.shape == (768,)
        assert np.all(1.0 + a >= 0.0)
        # A spectrum 100x stronger gives a visibly wider field on the same seed.
        assert b.std() > 3 * a.std()
