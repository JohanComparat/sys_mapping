"""Tests that the JAX kernels reproduce independent NumPy reference implementations.

The package runs these statistics on JAX only; the NumPy loops below are the reference
they are checked against, kept here rather than as production fallbacks.

Covers:
  - ISD Δχ² significance for poly_order > 1 and/or fracdet weights
    (JAX ``_one_isd_poly`` vmap vs ``_numpy_isd_delta_chi2``), and ``isd_marginal_fit``,
  - the null-test cross-correlation JAX path vs ``_numpy_null_test``,
  - the opt-in ``backend="jax"`` polynomial-OLS reweighting loop vs
    ``backend="numpy"`` (the v1.2 method formerly named ISD),
  - reproducibility of parallel GLASS mock generation (n_jobs) vs serial.
"""

import numpy as np
import pytest

from sys_mapping.diagnostics import (isd_marginal_fit, null_test_cross_correlations,
                                     snr_template_ranking)
from sys_mapping.regression import polynomial_ols_decontamination as poly_ols

# These tests exercise GLASS mock mechanics on a parametric field chosen on purpose,
# not a calibrated null, so the library's "not matched to any sample" warning is
# expected here and would only bury real ones.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*not matched to any sample.*:UserWarning")


def _bins(t_i, n_bins, quantile):
    t_min, t_max = float(t_i.min()), float(t_i.max())
    if quantile:
        edges = np.quantile(t_i, np.linspace(0.0, 1.0, n_bins + 1))
        return np.clip(np.searchsorted(edges[1:-1], t_i, side="right"), 0, n_bins - 1)
    return np.clip(np.floor((t_i - t_min) / (t_max - t_min) * n_bins).astype(int),
                   0, n_bins - 1)


def _numpy_isd_fit(delta_g_obs, delta_t, n_bins=10, poly_order=1, fracdet=None,
                   quantile=False):
    """Reference ISD marginal fit: (delta_chi2, ascending coeffs, t_range) per template."""
    n_sys, n_pix = delta_t.shape
    w = np.ones(n_pix) if fracdet is None else np.asarray(fracdet, float)
    g_bar = float(np.dot(w, delta_g_obs) / np.sum(w))
    dchi2 = np.zeros(n_sys)
    coeffs = np.zeros((n_sys, poly_order + 1))
    t_range = np.zeros((n_sys, 2))
    for i, t_i in enumerate(delta_t):
        if t_i.min() >= t_i.max():
            continue
        bin_idx = _bins(t_i, n_bins, quantile)
        s_b, n_b, sig_b = [], [], []
        for b in range(n_bins):
            m = bin_idx == b
            if not np.any(m) or np.sum(w[m]) < 1e-30:
                continue
            std_g = float(np.std(delta_g_obs[m]))
            if std_g < 1e-10:
                continue
            s_b.append(float(np.dot(w[m], t_i[m]) / np.sum(w[m])))
            n_b.append(float(np.dot(w[m], delta_g_obs[m]) / np.sum(w[m])))
            sig_b.append(std_g / np.sqrt(int(np.sum(m))))
        if len(s_b) < 2:
            continue
        s_arr, n_arr, sig_arr = np.array(s_b), np.array(n_b), np.array(sig_b)
        inv_s2 = 1.0 / sig_arr ** 2
        chi2_null = float(np.dot((n_arr - g_bar) ** 2, inv_s2))
        eff_order = min(poly_order, len(s_arr) - 1)
        c_desc = np.polyfit(s_arr, n_arr, eff_order, w=1.0 / sig_arr)
        chi2_model = float(np.dot((n_arr - np.polyval(c_desc, s_arr)) ** 2, inv_s2))
        dchi2[i] = max(chi2_null - chi2_model, 0.0)
        coeffs[i, : eff_order + 1] = c_desc[::-1]
        t_range[i] = (s_arr.min(), s_arr.max())
    return dchi2, coeffs, t_range


def _numpy_null_test(weights, delta_t, n_bootstrap, seed):
    """Reference weight-template correlations with permutation p-values."""
    rng = np.random.default_rng(seed)
    w_c = weights - weights.mean()
    w_norm = np.sqrt(np.sum(w_c ** 2))
    corr = np.empty(delta_t.shape[0])
    p = np.empty(delta_t.shape[0])
    for i, t_i in enumerate(delta_t):
        t_c = t_i - t_i.mean()
        t_norm = np.sqrt(np.sum(t_c ** 2))
        corr[i] = np.sum(w_c * t_c) / (w_norm * t_norm)
        perm = [abs(np.sum(rng.permutation(w_c) * t_c) / (w_norm * t_norm))
                for _ in range(n_bootstrap)]
        p[i] = (np.sum(np.array(perm) >= abs(corr[i])) + 1) / (n_bootstrap + 1)
    return {"correlations": corr, "p_values": p}


@pytest.fixture(scope="module")
def field():
    rng = np.random.default_rng(21)
    n_pix, n_sys = 5000, 5
    dt = rng.standard_normal((n_sys, n_pix))
    dt -= dt.mean(1, keepdims=True)
    dt /= dt.std(1, keepdims=True)
    dg = 0.3 * dt[2] - 0.2 * dt[0] ** 2 + rng.standard_normal(n_pix) * 0.3
    return dg, dt


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("use_fracdet", [False, True])
def test_isd_significance_jax_matches_numpy(field, order, use_fracdet):
    dg, dt = field
    fracdet = None
    if use_fracdet:
        fracdet = np.random.default_rng(0).uniform(0.5, 1.0, dg.shape[0])
    jax_snr = snr_template_ranking(dg, dt, method="isd", n_bins=10,
                                   poly_order=order, fracdet=fracdet)
    np_snr = _numpy_isd_fit(dg, dt, n_bins=10, poly_order=order, fracdet=fracdet)[0]
    assert np.allclose(jax_snr, np_snr, rtol=1e-6, atol=1e-6)
    # identical template ranking
    assert list(np.argsort(-jax_snr)) == list(np.argsort(-np_snr))


def test_null_test_correlations_jax_matches_numpy(field):
    dg, dt = field
    rng = np.random.default_rng(1)
    weights = 1.0 + 0.1 * dt[1] + rng.standard_normal(dg.shape[0]) * 0.05
    r_jax = null_test_cross_correlations(weights, dt, n_bootstrap=200, seed=0)
    r_np = _numpy_null_test(weights, dt, n_bootstrap=200, seed=0)
    # correlations are deterministic -> must match exactly
    assert np.allclose(r_jax["correlations"], r_np["correlations"], atol=1e-12)
    # permutation p-values are Monte-Carlo -> statistically close
    assert np.max(np.abs(r_jax["p_values"] - r_np["p_values"])) < 0.1


@pytest.mark.parametrize("quantile", [False, True])
@pytest.mark.parametrize("order", [1, 3])
def test_isd_marginal_fit_matches_numpy(field, order, quantile):
    dg, dt = field
    binning = "quantile" if quantile else "width"
    dchi2, coeffs, t_range = isd_marginal_fit(dg, dt, n_bins=10, poly_order=order,
                                              binning=binning)
    r_dchi2, r_coeffs, r_range = _numpy_isd_fit(dg, dt, n_bins=10, poly_order=order,
                                                quantile=quantile)
    assert np.allclose(dchi2, r_dchi2, rtol=1e-6, atol=1e-6)
    assert np.allclose(coeffs, r_coeffs, rtol=1e-5, atol=1e-8)
    assert np.allclose(t_range, r_range, atol=1e-12)


@pytest.mark.parametrize("method", ["template", "data"])
def test_ranking_matches_numpy(field, method):
    dg, dt = field
    snr = snr_template_ranking(dg, dt, method=method)
    ref = np.empty(dt.shape[0])
    for i, t_i in enumerate(dt):
        if method == "template":
            denom = np.dot(t_i, t_i) + 1e-30
            alpha = np.dot(t_i, dg) / denom
            ref[i] = abs(alpha) / np.sqrt(np.mean((dg - alpha * t_i) ** 2) / denom + 1e-30)
        else:
            g_c, t_c = dg - dg.mean(), t_i - t_i.mean()
            ref[i] = abs(np.sum(g_c * t_c) / (np.linalg.norm(g_c) * np.linalg.norm(t_c)))
    assert np.allclose(snr, ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("order,lam", [(1, 0.0), (2, 0.0), (3, 1e-3)])
def test_poly_ols_backend_jax_matches_numpy(order, lam):
    rng = np.random.default_rng(4)
    n_pix, n_sys = 12000, 4
    dt = rng.standard_normal((n_sys, n_pix))
    dt -= dt.mean(1, keepdims=True)
    dt /= dt.std(1, keepdims=True)
    dg = np.array([0.15, -0.08, 0.04, 0.02]) @ dt + rng.standard_normal(n_pix) * 0.4
    w_np, a_np, it_np = poly_ols(dg, dt, poly_order=order, max_iter=30, tol=1e-6,
                            lambda_poly=lam, backend="numpy")
    w_jx, a_jx, it_jx = poly_ols(dg, dt, poly_order=order, max_iter=30, tol=1e-6,
                            lambda_poly=lam, backend="jax")
    assert it_np == it_jx
    assert np.allclose(w_np, w_jx, atol=1e-10)
    assert np.allclose(a_np, a_jx, atol=1e-10)


def test_mock_parallel_matches_serial():
    glass = pytest.importorskip("glass")
    from sys_mapping import isd_template_significance

    rng = np.random.default_rng(0)
    nside = 16
    npix = 12 * nside ** 2
    good = np.ones(npix, dtype=bool)
    dt = rng.standard_normal((3, npix))
    dt -= dt.mean(1, keepdims=True)
    dt /= dt.std(1, keepdims=True)
    dg = 0.4 * dt[1] + rng.standard_normal(npix) * 0.1
    z_edges = np.array([0.1, 0.3, 0.5])
    nz = np.array([500.0, 400.0])
    kw = dict(good_pixels=good, nside=nside, n_total=0, n_total_footprint=5000,
              z_edges=z_edges, nz=nz, n_mocks=6, seed=0, rand_factor=2)
    r1 = isd_template_significance(dg, dt, n_jobs=1, **kw)
    r2 = isd_template_significance(dg, dt, n_jobs=2, **kw)
    # same per-mock seeds -> bit-identical regardless of n_jobs
    assert np.array_equal(r1["delta_chi2_mocks"], r2["delta_chi2_mocks"])
    assert np.array_equal(r1["p_values"], r2["p_values"])
