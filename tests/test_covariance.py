"""Tests for sys_mapping.covariance — low-rank + diagonal precision operator (Woodbury).

The operator supplies R^{-1} and ln|R| for the correlated-noise (GLS) likelihood without ever
forming the N x N matrix.  These tests pin it against a dense reference on small problems.
"""

import numpy as np
import pytest
import jax.numpy as jnp

from sys_mapping.covariance import (
    build_lowrank_precision,
    LowRankPrecision,
    mock_sandwich_covariance,
    sample_covariance,
    hartlap_factor,
    build_harmonic_precision,
)


N_PIX = 250
N_MOCK = 40


def _dense_R(mock_fields, shot):
    """Rebuild the dense unit-diagonal R = D + U U^T the same way build_lowrank_precision does."""
    fields = np.asarray(mock_fields, float)
    m = fields.shape[1]
    delta = fields - fields.mean(1, keepdims=True)
    u_raw = delta / np.sqrt(m - 1.0)
    shot = np.broadcast_to(np.asarray(shot, float), (fields.shape[0],))
    d_raw = shot + np.einsum("ij,ij->i", u_raw, u_raw)
    s = 1.0 / np.sqrt(d_raw)
    u = u_raw * s[:, None]
    return np.diag(shot * s**2) + u @ u.T


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(1)
    mock = rng.standard_normal((N_PIX, N_MOCK)) * 0.3
    shot = 0.05
    P = build_lowrank_precision(mock, shot)
    R = _dense_R(mock, shot)
    return mock, shot, P, R


class TestWoodbury:
    def test_unit_diagonal(self, problem):
        _, _, _, R = problem
        np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-10)

    def test_apply_matches_dense_inverse(self, problem):
        _, _, P, R = problem
        rng = np.random.default_rng(2)
        v = rng.standard_normal(N_PIX)
        got = np.asarray(P.apply(jnp.asarray(v)))
        want = np.linalg.solve(R, v)
        np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10)

    def test_apply_matrix_columns(self, problem):
        _, _, P, R = problem
        rng = np.random.default_rng(3)
        V = rng.standard_normal((N_PIX, 4))
        got = np.asarray(P.apply(jnp.asarray(V)))
        want = np.linalg.solve(R, V)
        np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10)

    def test_quadratic_form(self, problem):
        _, _, P, R = problem
        rng = np.random.default_rng(4)
        v = rng.standard_normal(N_PIX)
        got = float(P.quad(jnp.asarray(v)))
        want = float(v @ np.linalg.solve(R, v))
        np.testing.assert_allclose(got, want, rtol=1e-8)

    def test_logdet_matches_dense(self, problem):
        _, _, P, R = problem
        want = float(np.linalg.slogdet(R)[1])
        np.testing.assert_allclose(P.logdet, want, rtol=1e-8, atol=1e-8)

    def test_n_modes_truncation(self):
        rng = np.random.default_rng(5)
        mock = rng.standard_normal((N_PIX, N_MOCK)) * 0.3
        P = build_lowrank_precision(mock, 0.05, n_modes=10)
        assert P.n_modes == 10
        # Still a valid precision: unit-diagonal R, symmetric-positive quad form.
        rng2 = np.random.default_rng(6)
        v = rng2.standard_normal(N_PIX)
        assert float(P.quad(jnp.asarray(v))) > 0


class TestValidation:
    def test_too_few_mocks(self):
        with pytest.raises(ValueError, match=">= 2"):
            build_lowrank_precision(np.zeros((10, 1)), 0.1)

    def test_nonpositive_shot(self):
        rng = np.random.default_rng(7)
        with pytest.raises(ValueError, match="positive"):
            build_lowrank_precision(rng.standard_normal((10, 5)), 0.0)

    def test_wrong_ndim(self):
        with pytest.raises(ValueError, match="2-D"):
            build_lowrank_precision(np.zeros(10), 0.1)


class TestMockSandwich:
    def test_matches_dense_reference(self):
        """Sandwich equals (TT^T)^-1 (T C T^T) (TT^T)^-1 with C the empirical field covariance."""
        rng = np.random.default_rng(20)
        n_sys, n_pix, n_mock = 4, 300, 60
        T = rng.standard_normal((n_sys, n_pix))
        fields = rng.standard_normal((n_mock, n_pix)) * 0.2
        got = mock_sandwich_covariance(T, fields)
        # dense reference
        Minv = np.linalg.inv(T @ T.T)
        C = np.cov(fields, rowvar=False, ddof=1)          # (n_pix, n_pix)
        want = Minv @ (T @ C @ T.T) @ Minv
        np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10)

    def test_symmetric_psd(self):
        rng = np.random.default_rng(21)
        T = rng.standard_normal((3, 200))
        fields = rng.standard_normal((50, 200)) * 0.2
        cov = mock_sandwich_covariance(T, fields)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(cov) >= -1e-10)

    def test_shape_and_validation(self):
        rng = np.random.default_rng(22)
        T = rng.standard_normal((3, 100))
        assert mock_sandwich_covariance(T, rng.standard_normal((10, 100))).shape == (3, 3)
        with pytest.raises(ValueError, match="pixel dimension mismatch"):
            mock_sandwich_covariance(T, rng.standard_normal((10, 99)))
        with pytest.raises(ValueError, match=">= 2"):
            mock_sandwich_covariance(T, rng.standard_normal((1, 100)))


class TestSampleCovariance:
    def test_matches_numpy_cov(self):
        """Unbiased (ddof=1) sample covariance of an estimator ensemble."""
        rng = np.random.default_rng(30)
        samples = rng.standard_normal((80, 6)) * 0.1
        got = sample_covariance(samples)
        want = np.cov(samples, rowvar=False, ddof=1)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)

    def test_symmetric_psd(self):
        rng = np.random.default_rng(31)
        cov = sample_covariance(rng.standard_normal((50, 5)))
        np.testing.assert_allclose(cov, cov.T, atol=1e-14)
        assert np.all(np.linalg.eigvalsh(cov) >= -1e-12)

    def test_shape_and_validation(self):
        rng = np.random.default_rng(32)
        assert sample_covariance(rng.standard_normal((40, 7))).shape == (7, 7)
        with pytest.raises(ValueError, match="2-D"):
            sample_covariance(np.zeros(10))
        with pytest.raises(ValueError, match=">= 2"):
            sample_covariance(np.zeros((1, 7)))


class TestHartlapFactor:
    def test_value(self):
        # (n_real - n_bins - 2) / (n_real - 1)
        assert hartlap_factor(100, 20) == (100 - 20 - 2) / (100 - 1)

    def test_in_unit_interval(self):
        f = hartlap_factor(300, 20)
        assert 0.0 < f < 1.0

    def test_requires_enough_realizations(self):
        with pytest.raises(ValueError, match="n_real > n_bins \\+ 2"):
            hartlap_factor(21, 20)  # 21 <= 20 + 2


class TestHarmonicStub:
    def test_not_implemented(self):
        with pytest.raises(NotImplementedError, match="mock_sandwich_covariance"):
            build_harmonic_precision()
