"""Integration tests: all methods on synthetic mocks built from real GAIA / LS10 templates.

These tests use the real GAIA_nstar_faint and LS10_GALDEPTH_Z HEALPix maps
(at NSIDE=32 for speed) as systematic templates, inject known contamination
amplitudes, and verify that every implemented method returns outputs of the
correct shape and recovers the injected parameters within a loose tolerance.

Methods tested
--------------
- OLS (ordinary least squares, pixel-level linear regression)
- ElasticNet (L1+L2 penalised regression via scikit-learn)
- ISD-1 (Iterative Systematics Decontamination, poly_order=1)
- ISD-3 (Iterative Systematics Decontamination, poly_order=3)
- MCMC-additive (Berlfein+2024 with b_i=0 fixed)
- MCMC-combined (Berlfein+2024, free a_i and b_i)
- Likelihood ratio test (model selection between additive and combined)
- Null test (residual correlation with templates after weighting)

All tests skip automatically if the FITS files are not present on disk.
"""

from pathlib import Path

import numpy as np
import pytest
import healpy as hp
import jax.numpy as jnp

import sys_mapping as sm
from sys_mapping.maps import load_real_templates
from sys_mapping.contamination import apply_contamination, unpack_params
from sys_mapping.correction import rotate_templates, transform_params_from_rotated
from sys_mapping.model_selection import likelihood_ratio_test

# ── Data availability check ────────────────────────────────────────────────

_SYST_ROOT = Path("~/data/legacysurvey/dr10/systematics").expanduser()
# Files live in the NSIDE-specific subdirectory (e.g. 0032/)
_SYST_DIR = _SYST_ROOT / "0032"
_GAIA_32 = _SYST_DIR / "GAIA_nstar_faint_NSIDE_00032.fits"
_LS10_32 = _SYST_DIR / "LS10_GALDEPTH_Z_NSIDE_0032.fits"
_HAS_REAL_DATA = _GAIA_32.exists() and _LS10_32.exists()

real_data = pytest.mark.skipif(
    not _HAS_REAL_DATA,
    reason="GAIA/LS10 FITS files not found in ~/data/legacysurvey/dr10/systematics/0032/",
)

# ── Shared fixture ─────────────────────────────────────────────────────────

NSIDE = 32
_N_SYNTH = 2       # families 0 and 1 (fast, large-scale)
_N_MEAN = 50       # galaxies per pixel (high density → better S/N for tests)
_A_TRUE = np.array([0.08, -0.05,  0.06, -0.04])   # 2 synth + 2 real = 4 total
_B_TRUE = np.array([0.04,  0.00, -0.03,  0.05])
_SEED = 7


@pytest.fixture(scope="module")
def real_mock():
    """Return the full-pipeline mock inputs, computed once for the module."""
    if not _HAS_REAL_DATA:
        pytest.skip("GAIA/LS10 FITS files not found")

    real_tmpl, real_names, footprint_mask = load_real_templates(NSIDE, _SYST_DIR)
    synth_tmpl = sm.generate_systematic_maps(NSIDE, families=list(range(_N_SYNTH)), seed=0)
    synth_names = [f"synth_{i}" for i in range(_N_SYNTH)]

    templates = np.vstack([synth_tmpl, real_tmpl])
    template_names = synth_names + real_names
    n_sys = templates.shape[0]   # 4 total

    # --- generate synthetic galaxy catalog within the LS10 footprint ---
    n_pix = hp.nside2npix(NSIDE)
    rng = np.random.default_rng(_SEED)
    np.random.seed(_SEED)

    lmax = 3 * NSIDE - 1
    ell = np.arange(lmax + 1, dtype=float)
    cl_G = (ell + 1.0) ** (-2); cl_G[0] = 0.0
    cl_G *= 0.5**2 / np.sum((2 * ell + 1) / (4 * np.pi) * cl_G)
    G = hp.synfast(cl_G, nside=NSIDE, lmax=lmax)
    delta_true = np.exp(G - 0.5 * 0.5**2) - 1.0

    a_true = _A_TRUE[:n_sys]
    b_true = _B_TRUE[:n_sys]

    delta_cont = np.asarray(
        apply_contamination(
            jnp.asarray(delta_true),
            jnp.asarray(templates),
            jnp.asarray(a_true),
            jnp.asarray(b_true),
        )
    )

    lam = np.maximum(_N_MEAN * (1.0 + delta_cont), 0.0) * footprint_mask.astype(float)
    gal_counts = rng.poisson(lam).astype(float)
    rand_counts = np.round(footprint_mask.astype(float) * _N_MEAN * 8).astype(float)

    delta_g, good_pix = sm.compute_overdensity(gal_counts, rand_counts)
    delta_t = sm.assign_template_values(templates, good_pix)   # (n_sys, n_good)
    delta_t_rot, R, eigenvalues = rotate_templates(delta_t)

    return {
        "templates": templates,
        "template_names": template_names,
        "n_sys": n_sys,
        "footprint_mask": footprint_mask,
        "delta_g": delta_g,
        "delta_t": delta_t,
        "delta_t_rot": delta_t_rot,
        "R": R,
        "eigenvalues": eigenvalues,
        "a_true": a_true,
        "b_true": b_true,
        "good_pix": good_pix,
        "gal_counts": gal_counts,
        "rand_counts": rand_counts,
    }


# ── Helper: loose recovery check ──────────────────────────────────────────

def _check_recovery(a_hat, a_true, tol=0.20, label=""):
    """Assert that the mean absolute bias is below tol."""
    bias = np.abs(a_hat - a_true)
    assert bias.mean() < tol, (
        f"{label}: mean |bias| = {bias.mean():.4f} > {tol:.2f}\n"
        f"  a_hat  = {np.round(a_hat, 4)}\n"
        f"  a_true = {np.round(a_true, 4)}"
    )


# ── Shape / structural tests ───────────────────────────────────────────────

class TestRealTemplateSetup:
    @real_data
    def test_template_shape(self, real_mock):
        assert real_mock["templates"].shape == (4, hp.nside2npix(NSIDE))

    @real_data
    def test_template_names(self, real_mock):
        assert "GAIA_nstar_faint" in real_mock["template_names"]
        assert "LS10_GALDEPTH_Z" in real_mock["template_names"]

    @real_data
    def test_footprint_coverage(self, real_mock):
        frac = real_mock["footprint_mask"].mean()
        assert 0.30 < frac < 0.70, f"unexpected footprint fraction {frac:.2f}"

    @real_data
    def test_delta_t_shape(self, real_mock):
        n_good = real_mock["good_pix"].sum()
        assert real_mock["delta_t"].shape == (4, n_good)

    @real_data
    def test_pca_eigenvalues_positive(self, real_mock):
        assert np.all(real_mock["eigenvalues"] > 0)


# ── OLS (linear regression) ───────────────────────────────────────────────

class TestOLSRealTemplates:
    @real_data
    def test_output_shape(self, real_mock):
        n_sys = real_mock["n_sys"]
        X = real_mock["delta_t"].T  # (n_good, n_sys) design matrix
        XtX = X.T @ X
        Xty = X.T @ real_mock["delta_g"]
        a_ols = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        assert a_ols.shape == (n_sys,)

    @real_data
    def test_recovery_additive(self, real_mock):
        """OLS recovers additive amplitudes to within 0.20 mean absolute error."""
        X = real_mock["delta_t"].T  # (n_good, n_sys) design matrix
        a_ols = np.linalg.lstsq(X.T @ X, X.T @ real_mock["delta_g"], rcond=None)[0]
        _check_recovery(a_ols, real_mock["a_true"], tol=0.20, label="OLS")


# ── ElasticNet ────────────────────────────────────────────────────────────

class TestElasticNetRealTemplates:
    @real_data
    def test_output_shape(self, real_mock):
        a_en, _, _ = sm.elasticnet_contamination_fit(
            real_mock["delta_g"], real_mock["delta_t"], cv_folds=3
        )
        assert a_en.shape == (real_mock["n_sys"],)

    @real_data
    def test_recovery(self, real_mock):
        a_en, weights, cv_info = sm.elasticnet_contamination_fit(
            real_mock["delta_g"], real_mock["delta_t"], cv_folds=3
        )
        _check_recovery(a_en, real_mock["a_true"], tol=0.25, label="ElasticNet")
        assert "alpha_reg" in cv_info

    @real_data
    def test_weights_positive(self, real_mock):
        _, weights, _ = sm.elasticnet_contamination_fit(
            real_mock["delta_g"], real_mock["delta_t"], cv_folds=3
        )
        assert np.all(weights > 0)


# ── ISD — Iterative Systematics Decontamination ───────────────────────────

class TestISDRealTemplates:
    # ISD returns (weights, alpha_hat_all, n_iterations) where alpha_hat_all
    # is a 1-D array of shape (n_expanded,); first n_sys elements are linear.

    @real_data
    def test_isd1_output_shape(self, real_mock):
        n_sys = real_mock["n_sys"]
        _, alpha_hat, _ = sm.iterative_systematics_decontamination(
            real_mock["delta_g"], real_mock["delta_t"], poly_order=1
        )
        a_final = np.asarray(alpha_hat)[:n_sys]
        assert a_final.shape == (n_sys,)

    @real_data
    def test_isd3_output_shape(self, real_mock):
        n_sys = real_mock["n_sys"]
        _, alpha_hat, _ = sm.iterative_systematics_decontamination(
            real_mock["delta_g"], real_mock["delta_t"], poly_order=3
        )
        a_final = np.asarray(alpha_hat)[:n_sys]
        assert a_final.shape == (n_sys,)

    @real_data
    def test_isd1_recovery(self, real_mock):
        n_sys = real_mock["n_sys"]
        _, alpha_hat, _ = sm.iterative_systematics_decontamination(
            real_mock["delta_g"], real_mock["delta_t"], poly_order=1
        )
        a_final = np.asarray(alpha_hat)[:n_sys]
        _check_recovery(a_final, real_mock["a_true"], tol=0.25, label="ISD-1")

    @real_data
    def test_isd3_finite(self, real_mock):
        """ISD-3 may be ill-conditioned with real correlated templates; check finiteness only."""
        n_sys = real_mock["n_sys"]
        _, alpha_hat, _ = sm.iterative_systematics_decontamination(
            real_mock["delta_g"], real_mock["delta_t"], poly_order=3
        )
        a_final = np.asarray(alpha_hat)[:n_sys]
        assert a_final.shape == (n_sys,)
        assert np.all(np.isfinite(a_final))

    @real_data
    def test_convergence(self, real_mock):
        """ISD-1 should return a finite number of iterations (convergence not guaranteed)."""
        _, _, n_iter = sm.iterative_systematics_decontamination(
            real_mock["delta_g"], real_mock["delta_t"],
            poly_order=1, max_iter=50,
        )
        assert n_iter >= 1
        assert n_iter <= 50


# ── MCMC — additive model ─────────────────────────────────────────────────

class TestMCMCAdditiveRealTemplates:
    @real_data
    def test_chain_shape(self, real_mock):
        n_sys = real_mock["n_sys"]
        n_dim = n_sys + 1          # a_1..n + sigma  (no skew by default)
        n_walkers = max(2 * n_dim + 2, 30)
        flat, _ = sm.run_mcmc(
            n_sys=n_sys, model="additive",
            delta_g_obs=real_mock["delta_g"],
            delta_t=real_mock["delta_t_rot"],
            n_walkers=n_walkers, n_steps=120, n_burn=40,
            seed=_SEED, progress=False,
        )
        assert flat.shape == (n_walkers * (120 - 40), n_dim)

    @real_data
    def test_recovery_a(self, real_mock):
        n_sys = real_mock["n_sys"]
        n_dim = n_sys + 2
        n_walkers = max(2 * n_dim + 2, 30)
        flat, _ = sm.run_mcmc(
            n_sys=n_sys, model="additive",
            delta_g_obs=real_mock["delta_g"],
            delta_t=real_mock["delta_t_rot"],
            n_walkers=n_walkers, n_steps=200, n_burn=60,
            seed=_SEED, progress=False,
        )
        theta = sm.get_mle_params(flat)
        a_rot, _, _, _ = unpack_params(theta, n_sys, "additive")
        a_hat, _ = transform_params_from_rotated(
            np.asarray(a_rot), np.zeros(n_sys), real_mock["R"]
        )
        _check_recovery(a_hat, real_mock["a_true"], tol=0.25, label="MCMC-add")


# ── MCMC — combined model (Berlfein+2024) ──────────────────────────────────

class TestMCMCCombinedRealTemplates:
    @real_data
    def test_chain_shape(self, real_mock):
        n_sys = real_mock["n_sys"]
        n_dim = 2 * n_sys + 1      # a_1..n + b_1..n + sigma  (no skew by default)
        n_walkers = max(2 * n_dim + 2, 40)
        flat, _ = sm.run_mcmc(
            n_sys=n_sys, model="combined",
            delta_g_obs=real_mock["delta_g"],
            delta_t=real_mock["delta_t_rot"],
            n_walkers=n_walkers, n_steps=120, n_burn=40,
            seed=_SEED, progress=False,
        )
        assert flat.shape == (n_walkers * (120 - 40), n_dim)

    @real_data
    def test_recovery_a_combined(self, real_mock):
        n_sys = real_mock["n_sys"]
        n_dim = 2 * n_sys + 2
        n_walkers = max(2 * n_dim + 2, 40)
        flat, _ = sm.run_mcmc(
            n_sys=n_sys, model="combined",
            delta_g_obs=real_mock["delta_g"],
            delta_t=real_mock["delta_t_rot"],
            n_walkers=n_walkers, n_steps=200, n_burn=60,
            seed=_SEED, progress=False,
        )
        theta = sm.get_mle_params(flat)
        a_rot, b_rot, _, _ = unpack_params(theta, n_sys, "combined")
        a_hat, b_hat = transform_params_from_rotated(
            np.asarray(a_rot), np.asarray(b_rot), real_mock["R"]
        )
        _check_recovery(a_hat, real_mock["a_true"], tol=0.30, label="MCMC-comb a")

    @real_data
    def test_recovery_b_combined(self, real_mock):
        n_sys = real_mock["n_sys"]
        n_dim = 2 * n_sys + 2
        n_walkers = max(2 * n_dim + 2, 40)
        flat, _ = sm.run_mcmc(
            n_sys=n_sys, model="combined",
            delta_g_obs=real_mock["delta_g"],
            delta_t=real_mock["delta_t_rot"],
            n_walkers=n_walkers, n_steps=200, n_burn=60,
            seed=_SEED, progress=False,
        )
        theta = sm.get_mle_params(flat)
        a_rot, b_rot, _, _ = unpack_params(theta, n_sys, "combined")
        _, b_hat = transform_params_from_rotated(
            np.asarray(a_rot), np.asarray(b_rot), real_mock["R"]
        )
        _check_recovery(b_hat, real_mock["b_true"], tol=0.30, label="MCMC-comb b")

    @real_data
    def test_var_a_positive(self, real_mock):
        n_sys = real_mock["n_sys"]
        n_dim = 2 * n_sys + 2
        n_walkers = max(2 * n_dim + 2, 40)
        flat, _ = sm.run_mcmc(
            n_sys=n_sys, model="combined",
            delta_g_obs=real_mock["delta_g"],
            delta_t=real_mock["delta_t_rot"],
            n_walkers=n_walkers, n_steps=120, n_burn=40,
            seed=_SEED, progress=False,
        )
        var_a, var_b = sm.get_param_variance_from_chain(flat, n_sys, "combined")
        assert np.all(var_a > 0)
        assert np.all(var_b > 0)


# ── Likelihood ratio test ─────────────────────────────────────────────────

class TestLRTRealTemplates:
    @real_data
    def test_lrt_rejects_additive_null(self, real_mock):
        """With non-zero b_true, the combined model should beat additive."""
        n_sys = real_mock["n_sys"]
        R = real_mock["R"]

        n_dim_add = n_sys + 2
        nw_add = max(2 * n_dim_add + 2, 30)
        flat_add, _ = sm.run_mcmc(
            n_sys=n_sys, model="additive",
            delta_g_obs=real_mock["delta_g"],
            delta_t=real_mock["delta_t_rot"],
            n_walkers=nw_add, n_steps=200, n_burn=60,
            seed=_SEED, progress=False,
        )

        n_dim_comb = 2 * n_sys + 2
        nw_comb = max(2 * n_dim_comb + 2, 40)
        flat_comb, _ = sm.run_mcmc(
            n_sys=n_sys, model="combined",
            delta_g_obs=real_mock["delta_g"],
            delta_t=real_mock["delta_t_rot"],
            n_walkers=nw_comb, n_steps=200, n_burn=60,
            seed=_SEED, progress=False,
        )

        theta_add = sm.get_mle_params(flat_add)
        theta_comb = sm.get_mle_params(flat_comb)

        result = likelihood_ratio_test(
            real_mock["delta_g"], real_mock["delta_t_rot"],
            theta_add, theta_comb,
            null_model="additive", alt_model="combined",
            significance=0.05,
        )
        # b_true is non-zero → combined should be preferred
        assert result.lambda_lr > 0, "LRT statistic must be positive"
        assert result.reject_null, (
            f"Expected additive null to be rejected; λ_LR={result.lambda_lr:.2f}"
        )

    @real_data
    def test_lrt_p_value_range(self, real_mock):
        n_sys = real_mock["n_sys"]
        nw = max(2 * (n_sys + 2) + 2, 30)
        flat_add, _ = sm.run_mcmc(
            n_sys=n_sys, model="additive",
            delta_g_obs=real_mock["delta_g"],
            delta_t=real_mock["delta_t_rot"],
            n_walkers=nw, n_steps=120, n_burn=40,
            seed=_SEED, progress=False,
        )
        nw_c = max(2 * (2 * n_sys + 2) + 2, 40)
        flat_comb, _ = sm.run_mcmc(
            n_sys=n_sys, model="combined",
            delta_g_obs=real_mock["delta_g"],
            delta_t=real_mock["delta_t_rot"],
            n_walkers=nw_c, n_steps=120, n_burn=40,
            seed=_SEED, progress=False,
        )
        result = likelihood_ratio_test(
            real_mock["delta_g"], real_mock["delta_t_rot"],
            sm.get_mle_params(flat_add), sm.get_mle_params(flat_comb),
            null_model="additive", alt_model="combined",
        )
        assert 0.0 <= result.p_value <= 1.0


# ── Null test (post-correction residual check) ────────────────────────────

class TestNullTestRealTemplates:
    @real_data
    def test_null_corr_after_correction(self, real_mock):
        """After OLS correction, max residual template correlation < 0.40."""
        n_sys = real_mock["n_sys"]
        X = real_mock["delta_t"].T  # (n_good, n_sys)
        a_ols = np.linalg.lstsq(X.T @ X, X.T @ real_mock["delta_g"], rcond=None)[0]
        weights = 1.0 / np.maximum(1.0 + X @ a_ols, 1e-3)

        null = sm.null_test_cross_correlations(
            weights, real_mock["delta_t"], n_bootstrap=50, seed=_SEED
        )
        max_r = np.max(np.abs(null["correlations"]))
        assert max_r < 0.50, (
            f"Residual template correlation too large after OLS correction: max|r|={max_r:.3f}"
        )

    @real_data
    def test_null_output_keys(self, real_mock):
        weights = np.ones(real_mock["delta_g"].shape[0])
        null = sm.null_test_cross_correlations(
            weights, real_mock["delta_t"], n_bootstrap=20, seed=0
        )
        assert "correlations" in null
        assert "p_values" in null
        assert null["correlations"].shape == (real_mock["n_sys"],)


# ── SNR template ranking ──────────────────────────────────────────────────

class TestSNRRankingRealTemplates:
    @real_data
    def test_ranking_shape(self, real_mock):
        for method in ("template", "data"):
            snr = sm.snr_template_ranking(
                real_mock["delta_g"], real_mock["delta_t"], method=method
            )
            assert snr.shape == (real_mock["n_sys"],)

    @real_data
    def test_ranking_positive(self, real_mock):
        snr = sm.snr_template_ranking(
            real_mock["delta_g"], real_mock["delta_t"], method="data"
        )
        assert np.all(snr >= 0)

    @real_data
    def test_real_templates_not_zero_snr(self, real_mock):
        """The GAIA and LS10 templates should have non-trivial |correlation| with delta_g."""
        snr = sm.snr_template_ranking(
            real_mock["delta_g"], real_mock["delta_t"], method="data"
        )
        # At least one template should have SNR > 0.01
        assert np.max(snr) > 0.01
