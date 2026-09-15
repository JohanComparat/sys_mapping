"""Which public numeric functions survive ``jax.jit``, ``jax.vmap`` and ``jax.grad``.

Each case wraps one public function as ``f(x)`` of a single float array and states the
transforms that make sense for it.  A transform passes when it runs on traced inputs and
reproduces the eager result.  Functions written against NumPy fail under tracing; those are
listed in ``KNOWN_FAILURES`` and marked strict ``xfail``, so porting one to JAX turns its
entry into an unexpected pass that must be removed from the list.  The passing fraction is
the package's JAX-transformability metric (``docs/coverage.rst``).
"""
from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sys_mapping as sm

RNG = np.random.default_rng(3)
N_PIX, N_SYS, N_BINS = 64, 3, 5
T = RNG.standard_normal((N_SYS, N_PIX))
DG = RNG.standard_normal(N_PIX) * 0.2
A = np.array([0.02, -0.01, 0.03])
B = np.array([0.01, 0.00, -0.02])
XI = np.abs(RNG.standard_normal((N_SYS, N_BINS))) * 0.1
XI3 = np.abs(RNG.standard_normal((N_SYS, N_SYS, N_BINS))) * 0.1
NULL = RNG.standard_normal((12, N_PIX)) * 0.2
W_OBS = np.linspace(1.0, 0.1, N_BINS)


def _loglik_additive(theta):
    return sm.make_log_likelihood(N_SYS, "additive")(theta, jnp.asarray(DG), jnp.asarray(T))


# name -> (callable of one array, example input, transforms that make sense)
CASES = {
    "make_log_likelihood": (_loglik_additive, np.r_[A, 0.2], ("jit", "vmap", "grad")),
    "apply_contamination": (lambda a: sm.apply_contamination(DG, T, a, B), A, ("jit", "vmap", "grad")),
    "invert_contamination": (lambda a: sm.invert_contamination(DG, T, a, B), A, ("jit", "vmap", "grad")),
    "compute_two_point_correction": (
        lambda w: sm.compute_two_point_correction(w, A**2, B**2, XI), W_OBS, ("jit", "vmap", "grad")),
    "compute_two_point_correction[matrix]": (
        lambda w: sm.compute_two_point_correction(w, np.outer(A, A), np.outer(B, B), XI3),
        W_OBS, ("jit", "vmap", "grad")),
    "debias_params": (lambda a: sm.debias_params(a, B, 1e-4 * np.ones(N_SYS), 1e-4 * np.ones(N_SYS))[0],
                      A, ("jit", "vmap", "grad")),
    "debias_params_matrix": (lambda a: sm.debias_params_matrix(a, B, 1e-4 * np.eye(N_SYS),
                                                               1e-4 * np.eye(N_SYS))[0],
                             A, ("jit", "vmap", "grad")),
    "correct_two_point_function": (
        lambda w: sm.correct_two_point_function(w, A, B, 1e-4 * np.ones(N_SYS), 1e-4 * np.ones(N_SYS), XI),
        W_OBS, ("jit", "vmap", "grad")),
    "rotate_templates": (lambda t: sm.rotate_templates(t)[0], T, ("jit", "vmap")),
    "transform_params_from_rotated": (
        lambda a: sm.transform_params_from_rotated(a, B, np.eye(N_SYS))[0], A, ("jit", "vmap", "grad")),
    "sample_covariance": (lambda x: sm.sample_covariance(x), NULL[:, :N_BINS], ("jit", "vmap", "grad")),
    "mock_sandwich_covariance": (lambda x: sm.mock_sandwich_covariance(T, x), NULL, ("jit", "vmap", "grad")),
    "calibrated_template_significance": (
        lambda d: sm.calibrated_template_significance(d, T, NULL)["significance"], DG, ("jit", "vmap")),
    "residual_template_correlation_test": (
        lambda d: sm.residual_template_correlation_test(d, T, NULL)["chi2"], DG, ("jit", "vmap")),
    "snr_template_ranking": (lambda d: sm.snr_template_ranking(d, T, method="template"), DG, ("jit", "vmap")),
    "standardise_on_footprint": (lambda t: sm.standardise_on_footprint(t), T, ("jit", "vmap", "grad")),
    "posterior_median_params": (lambda c: sm.posterior_median_params(c), NULL[:, :4], ("jit", "vmap")),
}

# Written against NumPy (or with data-dependent shapes / Python control flow on values).
# Each entry is a (case, transform) that is known not to trace.  Remove an entry when the
# function is ported; strict xfail makes a stale entry fail the suite.
_NUMPY_ON_TRACER = {
    "correct_two_point_function": "np.asarray on w_obs (correction.py)",
    "rotate_templates": "np.linalg.eigh on the second moment (correction.py)",
    "sample_covariance": "np.cov (covariance.py)",
    "mock_sandwich_covariance": "np.linalg on the mock fields (covariance.py)",
    "calibrated_template_significance": "np.linalg.pinv and input validation (diagnostics.py)",
    "residual_template_correlation_test": "np.asarray and input validation (diagnostics.py)",
    "snr_template_ranking": "np.asarray before the JAX kernel (diagnostics.py)",
    "posterior_median_params": "np.median (inference.py)",
}
KNOWN_FAILURES: dict[tuple[str, str], str] = {
    (name, tr): f"written against NumPy: {why}"
    for name, why in _NUMPY_ON_TRACER.items()
    for tr in CASES[name][2]
}


def _params():
    for name, (_, _, transforms) in CASES.items():
        for tr in transforms:
            marks = []
            reason = KNOWN_FAILURES.get((name, tr))
            if reason:
                marks.append(pytest.mark.xfail(strict=True, reason=reason))
            yield pytest.param(name, tr, id=f"{name}-{tr}", marks=marks)


def _scalar(y):
    return jnp.sum(jnp.asarray(y) ** 2)


@pytest.mark.parametrize("name,transform", list(_params()))
def test_transform(name, transform):
    fn, x, _ = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eager = np.asarray(fn(np.asarray(x)))
        if transform == "jit":
            out = np.asarray(jax.jit(fn)(jnp.asarray(x)))
            np.testing.assert_allclose(out, eager, rtol=1e-6, atol=1e-10)
        elif transform == "vmap":
            batch = jnp.stack([jnp.asarray(x), jnp.asarray(x) * 1.01])
            out = np.asarray(jax.vmap(fn)(batch))
            np.testing.assert_allclose(out[0], eager, rtol=1e-6, atol=1e-10)
        elif transform == "grad":
            g = np.asarray(jax.grad(lambda z: _scalar(fn(z)))(jnp.asarray(x, dtype=float)))
            assert g.shape == np.shape(x) and np.all(np.isfinite(g))
