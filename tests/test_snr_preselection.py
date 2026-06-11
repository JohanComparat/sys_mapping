"""Integration tests for the two-stage SNR template pre-selection pipeline.

Builds a GLASS full-sky universe, injects known contamination from two
synthetic systematic templates, and verifies that every ranking method
identifies those templates correctly.
"""

import numpy as np
import pytest

glass = pytest.importorskip("glass")

import healpy as hp  # noqa: E402 — only needed after glass is confirmed

from sys_mapping.diagnostics import isd_template_significance, snr_template_ranking
from sys_mapping.glass_mocks import generate_glass_fullsky_mock
from sys_mapping.maps import (
    assign_template_values,
    compute_overdensity,
    generate_systematic_maps,
    pixelize_catalog,
)
from sys_mapping.model_selection import greedy_forward_select, snr_preselect
from sys_mapping.simulation import LEVELS

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
NSIDE = 16
N_TOTAL = 50_000     # large enough for reliable detection at a=0.02 (low contamination)
N_TEMPLATES = 8
CONTAM_IDX = [2, 5]          # the two injected contaminant templates
Z_EDGES = np.array([0.0, 0.5])
NZ = np.array([1000.0])
FAMILIES = [0, 1, 2, 3, 4, 0, 1, 2]   # 8 maps; families 2 and 0 appear twice


# ---------------------------------------------------------------------------
# Module-level fixtures (run once for the entire test file)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def universe():
    """Full-sky GLASS universe + 8 synthetic systematic maps.

    Returns
    -------
    delta_g_clean : (n_good,) ndarray
        Clean galaxy overdensity on the footprint.
    delta_t : (8, n_good) ndarray
        Systematic template values restricted to the footprint.
    good : (n_full,) bool ndarray
        Footprint mask.
    delta_t_full : (8, n_full) ndarray
        Template maps on the full HEALPix sphere (for peak-method tests).
    """
    cat = generate_glass_fullsky_mock(NSIDE, N_TOTAL, Z_EDGES, NZ, seed=0)
    delta_t_full = generate_systematic_maps(NSIDE, families=FAMILIES, seed=1)
    n_gal = pixelize_catalog(cat["ra"], cat["dec"], NSIDE)
    n_rand = pixelize_catalog(cat["ra_rand"], cat["dec_rand"], NSIDE)
    delta_g_clean, good = compute_overdensity(n_gal, n_rand)
    delta_t = assign_template_values(delta_t_full, good)
    return delta_g_clean, delta_t, good, delta_t_full


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _contaminate(delta_g_clean, delta_t, amplitude):
    """Additive contamination from CONTAM_IDX templates at given amplitude."""
    delta_g = delta_g_clean.copy()
    for i in CONTAM_IDX:
        delta_g = delta_g + amplitude * delta_t[i]
    return delta_g


# ---------------------------------------------------------------------------
# Tests: SNR ranking — all methods × all levels
# ---------------------------------------------------------------------------

class TestSnrRankingAllMethods:
    """All ranking methods detect the two injected templates at every level."""

    @pytest.mark.parametrize("method", ["data", "template", "isd"])
    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_contaminants_rank_in_top_3(self, universe, method, level):
        delta_g_clean, delta_t, good, _ = universe
        delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS[level])
        result = snr_preselect(delta_g_obs, delta_t, method=method, n_top=3)
        top3 = set(result.selected_indices[:3])
        assert 2 in top3 and 5 in top3, (
            f"Templates {CONTAM_IDX} expected in top-3 "
            f"(method={method}, level={level}); got {sorted(top3)}"
        )

    @pytest.mark.parametrize("method", ["data", "template", "isd"])
    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_snr_values_nonneg(self, universe, method, level):
        delta_g_clean, delta_t, good, _ = universe
        delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS[level])
        result = snr_preselect(delta_g_obs, delta_t, method=method)
        assert np.all(result.snr_values >= 0), (
            f"Negative SNR values for method={method}, level={level}"
        )

    def test_ranking_monotone_with_amplitude_data(self, universe):
        """SNR of injected template 2 increases with contamination amplitude."""
        delta_g_clean, delta_t, good, _ = universe
        snr_levels = []
        for level in ["low", "medium", "high"]:
            delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS[level])
            result = snr_preselect(delta_g_obs, delta_t, method="data")
            snr_levels.append(float(result.snr_values[CONTAM_IDX[0]]))
        assert snr_levels[0] < snr_levels[1] < snr_levels[2], (
            f"Expected monotone increase but got SNR values: {snr_levels}"
        )

    def test_ranking_monotone_with_amplitude_isd(self, universe):
        """ISD Δχ² of injected template 2 increases with contamination amplitude."""
        delta_g_clean, delta_t, good, _ = universe
        dchi2_levels = []
        for level in ["low", "medium", "high"]:
            delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS[level])
            result = snr_preselect(delta_g_obs, delta_t, method="isd")
            dchi2_levels.append(float(result.snr_values[CONTAM_IDX[0]]))
        assert dchi2_levels[0] < dchi2_levels[1] < dchi2_levels[2], (
            f"Expected monotone increase in ISD Δχ² but got: {dchi2_levels}"
        )

    def test_selected_indices_valid_range(self, universe):
        delta_g_clean, delta_t, good, _ = universe
        delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS["medium"])
        result = snr_preselect(delta_g_obs, delta_t, method="data")
        assert all(0 <= i < N_TEMPLATES for i in result.selected_indices)

    def test_no_duplicate_indices(self, universe):
        delta_g_clean, delta_t, good, _ = universe
        delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS["medium"])
        result = snr_preselect(delta_g_obs, delta_t, method="data")
        assert len(result.selected_indices) == len(set(result.selected_indices))


# ---------------------------------------------------------------------------
# Tests: full two-stage pipeline and ISD significance
# ---------------------------------------------------------------------------

class TestSnrPreselectionPipeline:
    """End-to-end pipeline: SNR pre-selection followed by greedy decontamination."""

    def test_snr_preselect_top4_contains_both_injected(self, universe):
        """Top-4 SNR pre-selection must contain templates 2 and 5."""
        delta_g_clean, delta_t, good, _ = universe
        delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS["medium"])
        pre = snr_preselect(delta_g_obs, delta_t, method="data", n_top=4)
        assert 2 in pre.selected_indices, (
            f"Template 2 missing from top-4: {pre.selected_indices}"
        )
        assert 5 in pre.selected_indices, (
            f"Template 5 missing from top-4: {pre.selected_indices}"
        )

    def test_full_pipeline_snr_then_greedy(self, universe):
        """Pre-selection top-4 → greedy selects at least one template."""
        delta_g_clean, delta_t, good, _ = universe
        delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS["medium"])
        pre = snr_preselect(delta_g_obs, delta_t, method="data", n_top=4)
        delta_t_sub = delta_t[pre.selected_indices]
        greedy = greedy_forward_select(delta_g_obs, delta_t_sub, p_threshold=0.05)
        assert len(greedy.selected_indices) >= 1, (
            "Greedy forward selection on the pre-selected subset found nothing"
        )

    def test_isd_significance_pvalues_small_for_injected(self, universe):
        """ISD mock-based p-values for injected templates are < 0.2 at high level."""
        delta_g_clean, delta_t, good, _ = universe
        delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS["high"])
        result = isd_template_significance(
            delta_g_obs, delta_t, good, NSIDE, N_TOTAL, Z_EDGES, NZ,
            n_mocks=20, seed=42,
        )
        p = result["p_values"]
        assert p[2] < 0.2, f"p_value[2] = {p[2]:.4f} — expected < 0.2"
        assert p[5] < 0.2, f"p_value[5] = {p[5]:.4f} — expected < 0.2"

    def test_isd_significance_injected_vs_noise(self, universe):
        """Mean p-value of noise templates > mean p-value of injected templates."""
        delta_g_clean, delta_t, good, _ = universe
        delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS["high"])
        result = isd_template_significance(
            delta_g_obs, delta_t, good, NSIDE, N_TOTAL, Z_EDGES, NZ,
            n_mocks=20, seed=42,
        )
        p = result["p_values"]
        noise_idx = [i for i in range(N_TEMPLATES) if i not in CONTAM_IDX]
        mean_p_noise = float(np.mean(p[noise_idx]))
        mean_p_injected = float(np.mean(p[list(CONTAM_IDX)]))
        assert mean_p_noise > mean_p_injected, (
            f"mean p (noise) = {mean_p_noise:.4f}, "
            f"mean p (injected) = {mean_p_injected:.4f}"
        )

    def test_isd_result_shapes(self, universe):
        delta_g_clean, delta_t, good, _ = universe
        delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS["high"])
        n_mocks = 8
        result = isd_template_significance(
            delta_g_obs, delta_t, good, NSIDE, N_TOTAL, Z_EDGES, NZ,
            n_mocks=n_mocks, seed=0,
        )
        assert result["delta_chi2"].shape == (N_TEMPLATES,)
        assert result["p_values"].shape == (N_TEMPLATES,)
        assert result["delta_chi2_mocks"].shape == (n_mocks, N_TEMPLATES)

    def test_isd_result_nonneg(self, universe):
        delta_g_clean, delta_t, good, _ = universe
        delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS["high"])
        result = isd_template_significance(
            delta_g_obs, delta_t, good, NSIDE, N_TOTAL, Z_EDGES, NZ,
            n_mocks=5, seed=0,
        )
        assert np.all(result["delta_chi2"] >= 0)
        assert np.all(result["delta_chi2_mocks"] >= 0)

    def test_n_top_controls_subset_size(self, universe):
        delta_g_clean, delta_t, good, _ = universe
        delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS["medium"])
        for k in [1, 3, 5, 8]:
            result = snr_preselect(delta_g_obs, delta_t, method="data", n_top=k)
            assert len(result.selected_indices) == k, (
                f"Expected {k} indices, got {len(result.selected_indices)}"
            )

    def test_snr_preselect_result_is_sorted(self, universe):
        """selected_indices are in descending SNR order."""
        delta_g_clean, delta_t, good, _ = universe
        delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS["medium"])
        result = snr_preselect(delta_g_obs, delta_t, method="data")
        snr_order = [result.snr_values[i] for i in result.selected_indices]
        assert snr_order == sorted(snr_order, reverse=True)


# ---------------------------------------------------------------------------
# Tests: cross-method agreement
# ---------------------------------------------------------------------------

class TestMethodComparison:
    """Cross-method agreement at high contamination level."""

    def test_all_snr_methods_rank_contaminant2_in_top2(self, universe):
        """At high level, all SNR methods rank template 2 in position 0 or 1."""
        delta_g_clean, delta_t, good, _ = universe
        delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS["high"])
        for method in ["data", "template", "isd"]:
            result = snr_preselect(delta_g_obs, delta_t, method=method)
            top2 = result.selected_indices[:2]
            assert 2 in top2, (
                f"Template 2 not in top-2 for method={method}: {top2}"
            )

    def test_data_and_template_methods_agree_on_top2(self, universe):
        """At high level, data and template methods return identical top-2 sets."""
        delta_g_clean, delta_t, good, _ = universe
        delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS["high"])
        r_data = snr_preselect(delta_g_obs, delta_t, method="data")
        r_tmpl = snr_preselect(delta_g_obs, delta_t, method="template")
        assert set(r_data.selected_indices[:2]) == set(r_tmpl.selected_indices[:2])

    def test_isd_snr_contaminants_larger_than_noise(self, universe):
        """ISD Δχ² for injected templates > all noise templates at high level."""
        delta_g_clean, delta_t, good, _ = universe
        delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS["high"])
        result = snr_preselect(delta_g_obs, delta_t, method="isd")
        snr = result.snr_values
        noise_idx = [i for i in range(N_TEMPLATES) if i not in CONTAM_IDX]
        for ci in CONTAM_IDX:
            for ni in noise_idx:
                assert snr[ci] > snr[ni], (
                    f"SNR[contam={ci}]={snr[ci]:.2f} <= SNR[noise={ni}]={snr[ni]:.2f}"
                )

    def test_peak_method_shape(self, universe):
        """Smoke test: peak method runs on full-sky arrays and returns right shape."""
        delta_g_clean, delta_t, good, delta_t_full = universe
        n_full = hp.nside2npix(NSIDE)
        delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS["high"])
        # Embed footprint overdensity back into full-sky (zeros outside)
        delta_g_obs_full = np.zeros(n_full)
        delta_g_obs_full[good] = delta_g_obs
        snr = snr_template_ranking(delta_g_obs_full, delta_t_full, method="peak")
        assert snr.shape == (N_TEMPLATES,)
        assert np.all(snr >= 0)

    def test_combined_snr_min_and_n_top_filter(self, universe):
        """snr_min AND n_top together produce a subset ≤ n_top with all SNR ≥ snr_min."""
        delta_g_clean, delta_t, good, _ = universe
        delta_g_obs = _contaminate(delta_g_clean, delta_t, LEVELS["medium"])
        result = snr_preselect(delta_g_obs, delta_t, method="data",
                               snr_min=0.0, n_top=4)
        assert len(result.selected_indices) <= 4
        for idx in result.selected_indices:
            assert result.snr_values[idx] >= 0.0
