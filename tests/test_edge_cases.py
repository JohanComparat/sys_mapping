"""Guards, warnings and rarely taken options of the public API.

Each test pins one branch the main test modules do not reach: an input check, a
documented fallback, an optional argument, or the error a missing optional dependency
raises.  Inputs are tiny so the module runs in seconds.
"""
from __future__ import annotations

import builtins
import json
import sys
import warnings

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest

import sys_mapping as sm
from sys_mapping import contamination, correction, covariance, diagnostics, glass_mocks, maps
from sys_mapping import mocks, model_selection, nuts, regression, simulation, utils
from sys_mapping.bootstrap import _assign_patches
from sys_mapping.contamination import TemplateResponse, evaluate_response
from sys_mapping.inference import refine_to_mle, run_additive_analytic

RNG = np.random.default_rng(11)


@pytest.fixture
def small_field():
    n_pix, n_sys = 400, 3
    t = RNG.standard_normal((n_sys, n_pix))
    t = (t - t.mean(1, keepdims=True)) / t.std(1, keepdims=True)
    g = 0.05 * t[0] + RNG.standard_normal(n_pix) * 0.1
    return g, t


def _block_import(monkeypatch, name):
    """Make ``import name`` (and its submodules) raise ImportError."""
    real_import = builtins.__import__

    def fake(mod, *args, **kwargs):
        if mod == name or mod.startswith(name + "."):
            raise ImportError(f"No module named {mod!r}")
        return real_import(mod, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        monkeypatch.delitem(sys.modules, key, raising=False)


# ── missing optional dependencies ────────────────────────────────────────────────

@pytest.mark.parametrize("call", [
    lambda: utils.measure_two_point_function(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)),
    lambda: utils.measure_cross_two_point_function(*[np.zeros(3)] * 6),
    lambda: utils.measure_kk_correlation_treecorr(np.zeros(3), np.zeros(3), np.zeros(3)),
    lambda: utils.measure_kk_covariance_treecorr(np.zeros(3), np.zeros(3), np.zeros(3)),
], ids=["2pt", "cross", "kk", "kk_cov"])
def test_treecorr_missing_is_named(monkeypatch, call):
    _block_import(monkeypatch, "treecorr")
    with pytest.raises(ImportError, match="treecorr is required"):
        call()


@pytest.mark.parametrize("call", [
    lambda: utils.measure_two_point_function_corrfunc(*[np.zeros(3)] * 4),
    lambda: utils.measure_kk_correlation_corrfunc(np.zeros(3), np.zeros(3), np.zeros(3)),
], ids=["2pt", "kk"])
def test_corrfunc_missing_is_named(monkeypatch, call):
    _block_import(monkeypatch, "Corrfunc")
    with pytest.raises(ImportError, match="Corrfunc is required"):
        call()


def test_sklearn_missing_is_named(monkeypatch, small_field):
    g, t = small_field
    _block_import(monkeypatch, "sklearn")
    with pytest.raises(ImportError, match="scikit-learn is required"):
        regression.elasticnet_contamination_fit(g, t)


# ── utils ────────────────────────────────────────────────────────────────────────

def test_cross_two_point_reuses_counts_and_weights():
    pytest.importorskip("treecorr")
    rng = np.random.default_rng(0)
    ra1, dec1 = rng.uniform(30, 60, 2000), rng.uniform(-10, 10, 2000)
    ra2, dec2 = rng.uniform(30, 60, 2000), rng.uniform(-10, 10, 2000)
    ra_r, dec_r = rng.uniform(30, 60, 8000), rng.uniform(-10, 10, 8000)
    kw = dict(min_sep=5.0, max_sep=100.0, nbins=5, bin_slop=0.1)
    theta, w, var, dr, rr = utils.measure_cross_two_point_function(
        ra1, dec1, ra2, dec2, ra_r, dec_r, w1=np.ones(2000), w2=np.ones(2000), **kw)
    theta2, w2, _, _, _ = utils.measure_cross_two_point_function(
        ra1, dec1, ra2, dec2, ra_r, dec_r, dr=dr, rr=rr, **kw)
    np.testing.assert_allclose(w2, w, rtol=1e-10)
    assert np.all(np.isfinite(var))


def test_template_correlation_matrix_checks_shapes():
    with pytest.raises(ValueError, match="must have shape"):
        utils.template_correlation_matrix(np.zeros(4), np.zeros(5), np.zeros((2, 4)))


# ── glass_mocks ──────────────────────────────────────────────────────────────────

def test_sanitise_cl_pads_and_floors():
    out = glass_mocks.sanitise_cl(np.array([5.0, 1.0, -0.5, 0.2]), lmax=7)
    assert out.shape == (8,) and out[0] == 0.0
    assert np.all(out[1:] > 0)                       # negative entry floored
    assert np.all(out[4:] == out[3])                 # held at the last measured value
    np.testing.assert_array_equal(glass_mocks.sanitise_cl(np.zeros(3), lmax=2), np.zeros(3))


def _write_match(path, cl, validation="absent"):
    d = {"cl_matched": list(cl)}
    if validation != "absent":
        d["validation"] = validation
    path.write_text(json.dumps(d))
    return path


class TestLoadMatchedClBranches:
    def test_direct_file_and_missing_file(self, tmp_path):
        f = _write_match(tmp_path / "x_match.json", [0.0, 1e-3, 1e-3],
                         {"passed": True, "large_scale_ratio": 1.0})
        assert len(sm.load_matched_cl(f)) == 3
        assert sm.load_matched_cl(tmp_path / "missing.json") is None

    def test_directory_needs_sample(self, tmp_path):
        with pytest.raises(ValueError, match="sample is needed"):
            sm.load_matched_cl(tmp_path)

    def test_empty_spectrum_returns_none(self, tmp_path):
        f = _write_match(tmp_path / "e_match.json", [], {"passed": True})
        assert sm.load_matched_cl(f) is None

    def test_unvalidated_file(self, tmp_path):
        f = _write_match(tmp_path / "u_match.json", [0.0, 1e-3])
        with pytest.raises(ValueError, match="no validation block"):
            sm.load_matched_cl(f)
        assert len(sm.load_matched_cl(f, require_validated=False)) == 2

    def test_failed_file_warns_when_not_required(self, tmp_path):
        f = _write_match(tmp_path / "f_match.json", [0.0, 1e-3],
                         {"passed": False, "large_scale_ratio": 0.5, "tol": 0.1})
        with pytest.warns(UserWarning, match="failed the large-scale check"):
            assert len(sm.load_matched_cl(f, require_validated=False)) == 2

    def test_unreadable_candidate_counts_as_unvalidated(self, tmp_path):
        (tmp_path / "S_NSIDE0032_match.json").write_text("{not json")
        _write_match(tmp_path / "S_NSIDE0064_match.json", [0.0, 1e-3, 2e-3],
                     {"passed": True, "large_scale_ratio": 1.0})
        assert len(sm.load_matched_cl(tmp_path, "S", 32)) == 3

    def test_no_nside_takes_the_finest(self, tmp_path):
        for ns, n in ((32, 4), (64, 8)):
            _write_match(tmp_path / f"S_NSIDE{ns:04d}_match.json", [0.0] + [1e-3] * n,
                         {"passed": True, "large_scale_ratio": 1.0})
        assert len(sm.load_matched_cl(tmp_path, "S")) == 9


def test_sample_positions_from_empty_visibility():
    pytest.importorskip("glass")
    nside = 8
    delta = np.zeros(hp.nside2npix(nside))
    ra, dec = glass_mocks.sample_positions_from_delta(delta, 1.0, vis=np.zeros_like(delta), seed=0)
    assert ra.size == 0 and dec.size == 0


# ── simulation ───────────────────────────────────────────────────────────────────

def test_contamination_config_properties():
    lin = simulation.ContaminationConfig(level="low", scenario="additive",
                                         a_true=np.array([0.1, 0.0, 0.2]), b_true=np.zeros(3))
    assert lin.n_sys == 3 and lin.contaminated == (0, 2)
    resp = TemplateResponse("tanh", 1.0, 0.05)
    nl = simulation.ContaminationConfig(level="low", scenario="additive",
                                        a_true=np.zeros(3), b_true=np.zeros(3),
                                        responses=(None, resp, None))
    assert nl.contaminated == (1,)


def test_load_uchuu_mock_reads_fits(tmp_path):
    from astropy.io import fits
    rng = np.random.default_rng(0)
    gal = fits.BinTableHDU.from_columns([
        fits.Column(name="RA", format="D", array=rng.uniform(0, 90, 50)),
        fits.Column(name="DEC", format="D", array=rng.uniform(0, 45, 50)),
        fits.Column(name="redshift_S", format="D", array=rng.uniform(0.1, 0.3, 50))])
    ran = fits.BinTableHDU.from_columns([
        fits.Column(name="RA", format="D", array=rng.uniform(0, 90, 200)),
        fits.Column(name="DEC", format="D", array=rng.uniform(0, 45, 200))])
    gal.writeto(tmp_path / "m_DATA.fits")
    ran.writeto(tmp_path / "m_RAND.fits")
    cat = simulation.load_uchuu_mock(tmp_path / "m_DATA.fits", tmp_path / "m_RAND.fits")
    assert cat["ra"].shape == (50,) and cat["ra_rand"].shape == (200,)
    assert "z" in cat


def test_load_systematic_maps_names_missing_file(tmp_path):
    (tmp_path / "0016").mkdir()
    with pytest.raises(FileNotFoundError, match="Systematic map not found"):
        simulation.load_systematic_maps(tmp_path, 16)


def test_apply_footprint_mask_without_optional_keys():
    nside = 8
    fp = np.zeros(hp.nside2npix(nside), bool)
    fp[: fp.size // 2] = True
    rng = np.random.default_rng(1)
    cat = {"ra": rng.uniform(0, 360, 100), "dec": rng.uniform(-90, 90, 100),
           "ra_rand": rng.uniform(0, 360, 300), "dec_rand": rng.uniform(-90, 90, 300)}
    out = simulation.apply_footprint_mask(cat, fp, nside)
    assert "z" not in out and "n_total" not in out
    assert out["ra"].size < 100


# ── likelihood, inference, nuts ─────────────────────────────────────────────────

def test_skewed_gls_likelihood_is_refused():
    with pytest.raises(ValueError):
        sm.make_log_likelihood(2, "combined", use_skewed=True, precision=object())


def test_analytic_posterior_without_templates(small_field):
    g, _ = small_field
    chain, sampler = run_additive_analytic(0, delta_g_obs=g, delta_t=np.zeros((0, g.size)),
                                           n_samples=2000, seed=0)
    assert chain.shape == (2000, 1)
    assert abs(np.median(chain) - np.std(g)) < 0.02 * np.std(g) + 1e-3


def test_refine_to_mle_warns_when_start_is_already_the_maximum(small_field):
    g, t = small_field
    a, *_ = np.linalg.lstsq(t.T, g, rcond=None)
    sigma = np.sqrt(np.mean((g - a @ t) ** 2))
    theta0 = sm.pack_params(a, None, sigma, model="additive")
    with pytest.warns(RuntimeWarning, match="did not improve"):
        out = refine_to_mle(theta0, g, t, model="additive", max_iter=5)
    np.testing.assert_array_equal(out, theta0)


def test_refine_to_mle_multiplicative_model(small_field):
    g, t = small_field
    theta0 = sm.pack_params(None, np.full(3, 0.01), 0.2, model="multiplicative")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = refine_to_mle(theta0, g, t, model="multiplicative", max_iter=50)
    assert out.shape == theta0.shape and np.all(np.isfinite(out))


def test_logdensity_priors_lower_the_density(small_field):
    g, t = small_field
    flat, n_dim, _ = nuts.build_logdensity(3, "combined", g, t)
    pri, _, _ = nuts.build_logdensity(3, "combined", g, t, prior_scale_a=0.01, prior_scale_b=0.01)
    u = jnp.asarray(np.r_[np.full(6, 0.1), np.log(0.1)])
    assert float(pri(u)) < float(flat(u))


def test_run_nuts_single_skewed_chain(small_field):
    g, t = small_field
    chain, sampler = nuts.run_nuts(3, model="additive", delta_g_obs=g, delta_t=t,
                                   use_skewed=True, n_chains=1, n_warmup=50, n_samples=40, seed=0)
    assert chain.shape == (40, 5)
    assert np.isnan(sampler.rhat)


# ── regression ───────────────────────────────────────────────────────────────────

def test_isd_refuses_a_step_that_inverts_the_field():
    rng = np.random.default_rng(3)
    n_pix = 4000
    t = rng.standard_normal((1, n_pix))
    g = -0.9 * t[0] + rng.standard_normal(n_pix) * 0.05
    with pytest.warns(UserWarning, match="step refused and template retired"):
        res = regression.iterative_systematics_decontamination(
            g, t, poly_order=1, chi2_68=1.0, bad_pixel_frac=1e-4, w_max=1.5)
    assert res.stopped_on == "exhausted"


def test_run_decontamination_with_ranking_preselection(small_field):
    g, t = small_field
    res = regression.run_decontamination("OLS", g, t, preselect=True,
                                         preselect_method="template", preselect_n_top=2)
    assert len(res["preselect_indices"]) == 2
    assert res["a_hat"].shape == (2,)


# ── correction, covariance ───────────────────────────────────────────────────────

def test_monte_carlo_two_point_correction_with_full_matrix():
    n_sys, n_bins = 2, 4
    xi = np.abs(np.random.default_rng(0).standard_normal((n_sys, n_sys, n_bins))) * 0.1
    xi = 0.5 * (xi + xi.transpose(1, 0, 2))
    w, cov = correction.correct_two_point_function(
        np.linspace(1, 0.1, n_bins), np.array([0.05, 0.02]), np.array([0.01, 0.0]),
        np.full(n_sys, 1e-4), np.full(n_sys, 1e-4), xi,
        return_cov=True, cov_a=1e-4 * np.eye(n_sys), cov_b=1e-4 * np.eye(n_sys),
        n_mc=200, random_state=0)
    assert w.shape == (n_bins,) and cov.shape == (n_bins, n_bins)


def test_overcorrection_bias_single_mock_has_zero_scatter(small_field):
    g, t = small_field
    est = lambda field: np.array([np.mean(field ** 2)])
    bias, scatter = correction.estimate_overcorrection_bias(g[None, :], t, est, method="OLS",
                                                            return_scatter=True)
    assert bias.shape == (1,) and np.all(scatter == 0)


@pytest.mark.parametrize("shapes", [((3,), (5, 3)), ((2, 3), (5,)), ((2, 3), (5, 4))])
def test_mock_sandwich_covariance_checks_shapes(shapes):
    t_shape, f_shape = shapes
    with pytest.raises(ValueError):
        covariance.mock_sandwich_covariance(np.zeros(t_shape), np.zeros(f_shape))


# ── diagnostics ──────────────────────────────────────────────────────────────────

def test_ranking_rejects_unknown_binning(small_field):
    g, t = small_field
    with pytest.raises(ValueError, match="binning must be"):
        sm.snr_template_ranking(g, t, method="isd", binning="nope")
    with pytest.raises(ValueError, match="binning must be"):
        diagnostics.isd_marginal_fit(g, t, binning="nope")


def test_vet_templates_with_a_single_patch_falls_back(small_field):
    g, t = small_field
    with pytest.warns(UserWarning, match="fewer than two patches"):
        out = diagnostics.vet_templates_against_tracer(t, g, patch_ids=np.zeros(g.size, int))
    assert np.all(np.isfinite(out["sigma"]))


def test_vet_templates_on_too_few_pixels_returns_zero():
    t = np.array([[0.0, 1.0]])
    out = diagnostics.vet_templates_against_tracer(t, np.array([1.0, 2.0]))
    assert np.all(out["rho"] == 0)


def test_footprint_diagnostics_warns_on_ignored_argument(small_field):
    g, t = small_field
    with pytest.warns(DeprecationWarning, match="ignores good_pixels"):
        diagnostics.footprint_mask_diagnostics(g, t, np.array([0.0, 0.1]),
                                               good_pixels=np.ones(g.size, bool))


# ── contamination, mocks, model selection, maps, bootstrap ─────────────────────

def test_two_point_correction_rejects_wrong_rank():
    with pytest.raises(ValueError, match="template_correlations"):
        sm.compute_two_point_correction(jnp.ones(3), jnp.ones(2), jnp.ones(2), jnp.ones((2, 2, 2, 3)))


def test_response_edge_cases():
    t = np.linspace(-1, 1, 11)
    linear = t * (0.1 / np.sqrt(np.mean(t ** 2)))      # a zero shape reduces to linear
    assert np.allclose(evaluate_response(TemplateResponse("tanh", 0.0, 0.1), t), linear)
    assert np.allclose(evaluate_response(TemplateResponse("exp", 0.0, 0.1), t), linear)
    assert np.all(evaluate_response(TemplateResponse("threshold", 5.0, 0.1), t) == 0)
    with pytest.raises(ValueError, match="unknown response kind"):
        evaluate_response(TemplateResponse("cubic-ish", 1.0, 0.1), t)


def test_galactic_mask_from_celestial_coordinates():
    m_gal = mocks.make_galactic_mask(16, 20.0, coord_in="G")
    m_cel = mocks.make_galactic_mask(16, 20.0, coord_in="C")
    assert m_cel.dtype == bool and not np.array_equal(m_gal, m_cel)
    assert abs(m_gal.mean() - m_cel.mean()) < 0.02


def test_unseeded_generators_run():
    assert np.isfinite(mocks.generate_lognormal_field(8, 0.3)).all()
    assert np.isfinite(maps.generate_systematic_map(8, family=1)).all()


def test_mock_catalog_with_explicit_families():
    m = mocks.make_mock_catalog(8, 2, scenario="additive", template_families=[3, 4], seed=1)
    assert m.n_sys == 2


def test_lrt_null_distribution_needs_2d_mocks(small_field):
    g, t = small_field
    with pytest.raises(ValueError, match="must be 2-D"):
        model_selection.lrt_null_distribution(g, t, lambda *a, **k: None)


def test_load_real_template_regrades_and_handles_empty_footprint(tmp_path):
    from astropy.io import fits
    nside = 8
    raw = np.arange(hp.nside2npix(nside), dtype=float)
    fits.BinTableHDU.from_columns([fits.Column(name="v", format="D", array=raw)]).writeto(
        tmp_path / "t.fits")
    t, mask = maps.load_real_template(tmp_path / "t.fits", "v", nside=4)
    assert t.shape == (hp.nside2npix(4),)
    t0, mask0 = maps.load_real_template(tmp_path / "t.fits", "v", valid_min=1e9)
    assert not mask0.any() and np.all(t0 == 0)


def test_assign_patches_with_explicit_coarse_resolution():
    good = np.ones(hp.nside2npix(8), bool)
    ids = _assign_patches(good, nside=8, n_patches=12, nside_patch=1)
    assert len(np.unique(ids)) == 12


# ── run_decontamination: ISD pre-selection and correlated-noise paths ───────────

@pytest.mark.filterwarnings("ignore:.*not matched to any sample.*:UserWarning")
def test_isd_preselection_feeds_the_isd_threshold():
    pytest.importorskip("glass")
    nside = 8
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(5)
    t = rng.standard_normal((3, npix))
    t = (t - t.mean(1, keepdims=True)) / t.std(1, keepdims=True)
    g = 0.3 * t[1] + rng.standard_normal(npix) * 0.2
    res = regression.run_decontamination(
        "ISD-1", g, t, preselect=True, preselect_method="isd", preselect_p_threshold=0.5,
        preselect_n_mocks=4, preselect_cl_amplitude=5e-4, good_pixels=np.ones(npix, bool),
        n_total_footprint=20000, z_edges=np.array([0.1, 0.3]), nz=np.array([1.0]), nside=nside)
    assert res["preselect_isd"] is not None
    kept = res["preselect_indices"]
    assert 1 <= len(kept) <= 3 and res["a_hat"].shape == (len(kept),)
    assert np.all(np.isfinite(res["weights"]))


def test_pixel_precision_forces_nuts(small_field):
    g, t = small_field
    prec = covariance.build_lowrank_precision(RNG.standard_normal((g.size, 20)) * 0.1, 0.01,
                                              n_modes=3)
    res = regression.run_decontamination("MCMC-add", g, t, sampler="analytic",
                                         pixel_precision=prec, n_chains=1,
                                         nuts_n_warmup=30, nuts_n_samples=30)
    assert res["sampler_backend"] == "nuts"


# ── compiled functions are shared, not rebuilt per call ─────────────────────────

def test_likelihood_and_refinement_are_compiled_once():
    from sys_mapping.inference import _neg_log_lik_value_and_grad
    assert sm.make_log_likelihood(4, "additive") is sm.make_log_likelihood(4, "additive")
    assert sm.make_log_likelihood(4, "additive") is not sm.make_log_likelihood(4, "combined")
    assert _neg_log_lik_value_and_grad(4, "combined", False) is \
        _neg_log_lik_value_and_grad(4, "combined", False)


def test_nuts_fits_of_same_shape_share_one_runner(small_field):
    g, t = small_field
    before = len(nuts._RUNNER_CACHE)
    for seed in (0, 1):
        nuts.run_nuts(3, model="additive", delta_g_obs=g + 0.01 * seed, delta_t=t,
                      n_chains=1, n_warmup=17, n_samples=13, seed=seed)
    assert len(nuts._RUNNER_CACHE) == before + 1
