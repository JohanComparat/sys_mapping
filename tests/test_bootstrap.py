"""Tests for sys_mapping.bootstrap — block bootstrap and jackknife."""

import numpy as np
import pytest

from sys_mapping.bootstrap import _assign_patches, block_bootstrap_variance, jackknife_covariance

NSIDE = 16
N_PIX = 12 * NSIDE ** 2


@pytest.fixture(scope="module")
def mock_data():
    rng = np.random.default_rng(42)
    good = np.ones(N_PIX, dtype=bool)
    dg = rng.standard_normal(N_PIX) * 0.1
    dt = rng.standard_normal((3, N_PIX))
    return good, dg, dt


def _mean_estimator(dg, dt):
    return np.array([np.mean(dg)])


class TestAssignPatches:
    def test_all_pixels_assigned(self):
        good = np.ones(N_PIX, dtype=bool)
        ids = _assign_patches(good, NSIDE, 8)
        assert len(ids) == N_PIX

    def test_partial_mask(self):
        good = np.ones(N_PIX, dtype=bool)
        good[:100] = False
        ids = _assign_patches(good, NSIDE, 8)
        assert len(ids) == good.sum()


class TestBlockBootstrap:
    def test_shape(self, mock_data):
        good, dg, dt = mock_data
        var = block_bootstrap_variance(
            dg, dt, good, NSIDE, _mean_estimator, n_bootstrap=20, n_patches=8, seed=0,
        )
        assert var.shape == (1,)

    def test_non_negative(self, mock_data):
        good, dg, dt = mock_data
        var = block_bootstrap_variance(
            dg, dt, good, NSIDE, _mean_estimator, n_bootstrap=20, n_patches=8, seed=1,
        )
        assert float(var[0]) >= 0.0


class TestJackknife:
    def test_shape(self, mock_data):
        good, dg, dt = mock_data
        cov = jackknife_covariance(dg, dt, good, NSIDE, _mean_estimator, n_patches=8)
        assert cov.shape == (1, 1)

    def test_non_negative_diagonal(self, mock_data):
        good, dg, dt = mock_data
        cov = jackknife_covariance(dg, dt, good, NSIDE, _mean_estimator, n_patches=8)
        assert float(cov[0, 0]) >= 0.0

    def test_small_patches_warning(self, mock_data):
        good, dg, dt = mock_data
        with pytest.warns(UserWarning, match="n_patches=3 is small"):
            jackknife_covariance(dg, dt, good, NSIDE, _mean_estimator, n_patches=3)
