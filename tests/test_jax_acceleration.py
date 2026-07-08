"""Tests that the new JAX-accelerated paths reproduce their NumPy references.

Covers:
  - ISD Δχ² significance for poly_order > 1 and/or fracdet weights
    (JAX ``_one_isd_poly`` vmap vs the NumPy fallback in snr_template_ranking),
  - the null-test cross-correlation JAX path vs the NumPy loop,
  - the opt-in ``backend="jax"`` ISD reweighting loop vs ``backend="numpy"``,
  - reproducibility of parallel GLASS mock generation (n_jobs) vs serial.
"""

import numpy as np
import pytest

import sys_mapping.diagnostics as D
from sys_mapping.diagnostics import snr_template_ranking, null_test_cross_correlations
from sys_mapping.regression import iterative_systematics_decontamination as isd


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
    assert D._JAX_AVAILABLE, "JAX expected in the test environment"
    jax_snr = snr_template_ranking(dg, dt, method="isd", n_bins=10,
                                   poly_order=order, fracdet=fracdet)
    D._JAX_AVAILABLE = False
    try:
        np_snr = snr_template_ranking(dg, dt, method="isd", n_bins=10,
                                      poly_order=order, fracdet=fracdet)
    finally:
        D._JAX_AVAILABLE = True
    assert np.allclose(jax_snr, np_snr, rtol=1e-6, atol=1e-6)
    # identical template ranking
    assert list(np.argsort(-jax_snr)) == list(np.argsort(-np_snr))


def test_null_test_correlations_jax_matches_numpy(field):
    dg, dt = field
    rng = np.random.default_rng(1)
    weights = 1.0 + 0.1 * dt[1] + rng.standard_normal(dg.shape[0]) * 0.05
    r_jax = null_test_cross_correlations(weights, dt, n_bootstrap=200, seed=0)
    D._JAX_AVAILABLE = False
    try:
        r_np = null_test_cross_correlations(weights, dt, n_bootstrap=200, seed=0)
    finally:
        D._JAX_AVAILABLE = True
    # correlations are deterministic -> must match exactly
    assert np.allclose(r_jax["correlations"], r_np["correlations"], atol=1e-12)
    # permutation p-values are Monte-Carlo -> statistically close
    assert np.max(np.abs(r_jax["p_values"] - r_np["p_values"])) < 0.1


@pytest.mark.parametrize("order,lam", [(1, 0.0), (2, 0.0), (3, 1e-3)])
def test_isd_backend_jax_matches_numpy(order, lam):
    rng = np.random.default_rng(4)
    n_pix, n_sys = 12000, 4
    dt = rng.standard_normal((n_sys, n_pix))
    dt -= dt.mean(1, keepdims=True)
    dt /= dt.std(1, keepdims=True)
    dg = np.array([0.15, -0.08, 0.04, 0.02]) @ dt + rng.standard_normal(n_pix) * 0.4
    w_np, a_np, it_np = isd(dg, dt, poly_order=order, max_iter=30, tol=1e-6,
                            lambda_poly=lam, backend="numpy")
    w_jx, a_jx, it_jx = isd(dg, dt, poly_order=order, max_iter=30, tol=1e-6,
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
