"""Tests for likelihood.py: Gaussian and skewed-Gaussian log-likelihoods."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from sys_mapping.likelihood import make_log_likelihood
from sys_mapping.contamination import pack_params


N_PIX = 200
N_SYS = 3


def _make_data(seed=42):
    rng = np.random.default_rng(seed)
    delta_t = rng.standard_normal((N_SYS, N_PIX))
    delta_t -= delta_t.mean(axis=1, keepdims=True)
    delta_g_obs = rng.standard_normal(N_PIX) * 0.1
    return delta_g_obs, delta_t


class TestGaussianLikelihood:
    def test_zero_params_equals_normal_logpdf(self):
        """When a=b=0, log_likelihood = Gaussian log-density of delta_g_obs."""
        from scipy.stats import norm

        dg, dt = _make_data()
        sigma = float(np.std(dg))
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)
        theta = pack_params(a, b, sigma, model="combined")

        log_lik = make_log_likelihood(N_SYS, "combined", use_skewed=False)
        ll = float(log_lik(jnp.asarray(theta), jnp.asarray(dg), jnp.asarray(dt)))

        expected = float(np.sum(norm.logpdf(dg, loc=0, scale=sigma)))
        np.testing.assert_allclose(ll, expected, rtol=1e-6)

    def test_wrong_sigma_lower_likelihood(self):
        """Mis-specified sigma should give a lower likelihood."""
        dg, dt = _make_data()
        sigma_true = float(np.std(dg))
        sigma_bad = sigma_true * 3.0
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)

        theta_good = pack_params(a, b, sigma_true, model="combined")
        theta_bad = pack_params(a, b, sigma_bad, model="combined")

        log_lik = make_log_likelihood(N_SYS, "combined")
        ll_good = float(log_lik(jnp.asarray(theta_good), jnp.asarray(dg), jnp.asarray(dt)))
        ll_bad = float(log_lik(jnp.asarray(theta_bad), jnp.asarray(dg), jnp.asarray(dt)))
        assert ll_good > ll_bad

    def test_all_models_return_finite(self):
        dg, dt = _make_data()
        sigma = 0.1
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)

        for model in ("additive", "multiplicative", "combined"):
            theta = pack_params(a, b, sigma, model=model)
            log_lik = make_log_likelihood(N_SYS, model)
            ll = float(log_lik(jnp.asarray(theta), jnp.asarray(dg), jnp.asarray(dt)))
            assert np.isfinite(ll), f"log_likelihood not finite for model={model}"


class TestSkewedLikelihood:
    def test_gamma_zero_equals_gaussian(self):
        """Setting gamma=0 should recover the Gaussian log-likelihood."""
        dg, dt = _make_data()
        sigma = 0.1
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)

        theta_gauss = pack_params(a, b, sigma, model="combined")
        theta_skewed = pack_params(a, b, sigma, gamma=0.0, model="combined")

        log_lik_gauss = make_log_likelihood(N_SYS, "combined", use_skewed=False)
        log_lik_skewed = make_log_likelihood(N_SYS, "combined", use_skewed=True)

        ll_g = float(log_lik_gauss(jnp.asarray(theta_gauss), jnp.asarray(dg), jnp.asarray(dt)))
        ll_s = float(log_lik_skewed(jnp.asarray(theta_skewed), jnp.asarray(dg), jnp.asarray(dt)))
        np.testing.assert_allclose(ll_s, ll_g, atol=1e-6)

    def test_nonzero_gamma_returns_finite(self):
        dg, dt = _make_data()
        sigma = 0.1
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)
        theta = pack_params(a, b, sigma, gamma=1.5, model="combined")
        log_lik = make_log_likelihood(N_SYS, "combined", use_skewed=True)
        ll = float(log_lik(jnp.asarray(theta), jnp.asarray(dg), jnp.asarray(dt)))
        assert np.isfinite(ll)


class TestGradient:
    def test_gradient_is_finite(self):
        """JAX grad of log_likelihood w.r.t. theta must be finite."""
        dg, dt = _make_data()
        sigma = 0.1
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)
        theta = jnp.asarray(pack_params(a, b, sigma, model="combined"))

        log_lik = make_log_likelihood(N_SYS, "combined")
        grad_fn = jax.grad(lambda t: log_lik(t, jnp.asarray(dg), jnp.asarray(dt)))
        grad = grad_fn(theta)
        assert np.all(np.isfinite(np.asarray(grad))), "Gradient contains non-finite values"

    def test_gradient_zero_at_mle(self):
        """Gradient of the additive Gaussian log-likelihood is zero at the OLS MLE.

        For the additive model (b=0):
          a_MLE = (dt @ dt.T)^{-1} @ (dt @ dg)   [OLS solution]
          σ_MLE = std(residual)                    [biased MLE]
        At this point the score is identically zero.
        """
        rng = np.random.default_rng(99)
        n_pix = 2000
        dg = rng.normal(0, 0.1, n_pix)
        dt = rng.standard_normal((N_SYS, n_pix))
        dt -= dt.mean(axis=1, keepdims=True)

        # Analytical MLE for additive model
        C = dt @ dt.T                          # (N_SYS, N_SYS)
        a_mle = np.linalg.solve(C, dt @ dg)   # OLS: ∂lnL/∂a = 0

        residual = dg - a_mle @ dt
        # ∂lnL/∂σ = -N/σ + Σr²/σ³ = 0  →  σ² = mean(r²)  (NOT std, which centres first)
        sigma_mle = float(np.sqrt(np.mean(residual**2)))

        theta = jnp.asarray(pack_params(a_mle, None, sigma_mle, model="additive"))
        log_lik = make_log_likelihood(N_SYS, "additive")
        grad_fn = jax.grad(lambda t: log_lik(t, jnp.asarray(dg), jnp.asarray(dt)))
        grad = np.asarray(grad_fn(theta))

        np.testing.assert_allclose(grad[:N_SYS], 0.0, atol=1e-5)   # ∂lnL/∂a = 0
        np.testing.assert_allclose(grad[N_SYS], 0.0, atol=1e-5)    # ∂lnL/∂σ = 0


class TestGLSPrecision:
    """Correlated-noise (GLS) likelihood: precision replaces σ²I with σ²R."""

    def _dense_R(self, mock, shot):
        m = mock.shape[1]
        delta = mock - mock.mean(1, keepdims=True)
        u_raw = delta / np.sqrt(m - 1.0)
        shot = np.broadcast_to(np.asarray(shot, float), (mock.shape[0],))
        d_raw = shot + np.einsum("ij,ij->i", u_raw, u_raw)
        s = 1.0 / np.sqrt(d_raw)
        return np.diag(shot * s**2) + (u_raw * s[:, None]) @ (u_raw * s[:, None]).T

    def _gls_ll_dense(self, dg, dt, a, b, sigma, R):
        """Reference combined-model GLS log-likelihood using a dense R."""
        mult = b @ dt
        add = a @ dt
        r = (dg - add) / (1.0 + mult)
        n = dg.size
        quad = float(r @ np.linalg.solve(R, r))
        logdet = float(np.linalg.slogdet(R)[1])
        log_gauss = -0.5 * n * np.log(2 * np.pi * sigma**2) - 0.5 * logdet - 0.5 / sigma**2 * quad
        log_jac = float(np.sum(np.log(np.abs(1.0 + mult))))
        return log_gauss - log_jac

    def test_precision_none_matches_white(self):
        """precision=None is byte-for-byte the existing white likelihood."""
        from sys_mapping.covariance import build_lowrank_precision
        dg, dt = _make_data()
        theta = pack_params(np.zeros(N_SYS), np.zeros(N_SYS), 0.1, model="combined")
        white = make_log_likelihood(N_SYS, "combined")
        also_white = make_log_likelihood(N_SYS, "combined", precision=None)
        v0 = float(white(jnp.asarray(theta), jnp.asarray(dg), jnp.asarray(dt)))
        v1 = float(also_white(jnp.asarray(theta), jnp.asarray(dg), jnp.asarray(dt)))
        np.testing.assert_allclose(v0, v1, rtol=0, atol=0)

    def test_identity_R_equals_white(self):
        """R ≈ I (negligible modes) reproduces the white likelihood."""
        from sys_mapping.covariance import build_lowrank_precision
        rng = np.random.default_rng(11)
        dg, dt = _make_data()
        tiny = rng.standard_normal((N_PIX, 30)) * 1e-6
        P = build_lowrank_precision(tiny, 1.0)
        a = rng.standard_normal(N_SYS) * 0.05
        b = rng.standard_normal(N_SYS) * 0.05
        theta = pack_params(a, b, 0.1, model="combined")
        white = make_log_likelihood(N_SYS, "combined")
        gls = make_log_likelihood(N_SYS, "combined", precision=P)
        v0 = float(white(jnp.asarray(theta), jnp.asarray(dg), jnp.asarray(dt)))
        v1 = float(gls(jnp.asarray(theta), jnp.asarray(dg), jnp.asarray(dt)))
        np.testing.assert_allclose(v1, v0, atol=1e-6)

    def test_correlated_R_matches_dense_reference(self):
        """GLS likelihood equals the dense-R hand computation for a nontrivial R and theta."""
        from sys_mapping.covariance import build_lowrank_precision
        rng = np.random.default_rng(12)
        dg, dt = _make_data()
        mock = rng.standard_normal((N_PIX, 40)) * 0.3
        shot = 0.05
        P = build_lowrank_precision(mock, shot)
        R = self._dense_R(mock, shot)
        a = rng.standard_normal(N_SYS) * 0.05
        b = rng.standard_normal(N_SYS) * 0.05
        sigma = 0.12
        theta = pack_params(a, b, sigma, model="combined")
        gls = make_log_likelihood(N_SYS, "combined", precision=P)
        got = float(gls(jnp.asarray(theta), jnp.asarray(dg), jnp.asarray(dt)))
        want = self._gls_ll_dense(dg, dt, a, b, sigma, R)
        np.testing.assert_allclose(got, want, rtol=1e-8)

    def test_gls_gradient_finite(self):
        from sys_mapping.covariance import build_lowrank_precision
        rng = np.random.default_rng(13)
        dg, dt = _make_data()
        mock = rng.standard_normal((N_PIX, 30)) * 0.3
        P = build_lowrank_precision(mock, 0.05)
        theta = jnp.asarray(pack_params(np.zeros(N_SYS), np.zeros(N_SYS), 0.1, model="combined"))
        gls = make_log_likelihood(N_SYS, "combined", precision=P)
        grad = jax.grad(lambda t: gls(t, jnp.asarray(dg), jnp.asarray(dt)))(theta)
        assert np.all(np.isfinite(np.asarray(grad)))
