"""Tests for the DES Y6 features added alongside the ISD rewrite.

Covers:
  - spatially compact cross-validation folds for ElasticNet,
  - inverse-variance pixel weights,
  - template vetting against an external LSS tracer,
  - w(theta) over-correction debias and method-marginalised covariance.
"""

import numpy as np
import pytest

import sys_mapping as sm
from sys_mapping.covariance import method_marginalised_covariance
from sys_mapping.correction import (
    estimate_overcorrection_bias,
    debias_two_point_function,
)


class TestSpatialCrossValidation:
    def test_patch_labels_cover_every_pixel_once(self):
        good = np.ones(12 * 16 ** 2, dtype=bool)
        ids = sm.assign_spatial_patches(good, nside=16, n_patches=48)
        assert len(ids) == int(good.sum())
        assert ids.min() == 0
        assert np.unique(ids).size >= 2

    def test_patches_are_spatially_compact(self):
        """Pixels sharing a patch are closer together than random pairs."""
        import healpy as hp

        nside = 16
        good = np.ones(hp.nside2npix(nside), dtype=bool)
        ids = sm.assign_spatial_patches(good, nside=nside, n_patches=48)
        vecs = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))).T

        rng = np.random.default_rng(0)
        same, diff = [], []
        for _ in range(2000):
            i, j = rng.integers(0, len(ids), 2)
            d = float(np.dot(vecs[i], vecs[j]))
            (same if ids[i] == ids[j] else diff).append(d)
        # Higher dot product = smaller angular separation.
        assert np.mean(same) > np.mean(diff)

    def test_group_cv_runs_and_is_flagged(self):
        pytest.importorskip("sklearn")
        rng = np.random.default_rng(3)
        n_pix, n_sys = 4000, 3
        delta_t = rng.standard_normal((n_sys, n_pix))
        delta_g = 0.1 * delta_t[0] + rng.standard_normal(n_pix) * 0.3
        ids = np.repeat(np.arange(20), n_pix // 20)
        a_hat, w, info = sm.elasticnet_contamination_fit(
            delta_g, delta_t, patch_ids=ids)
        assert a_hat.shape == (n_sys,)
        assert info["cv_spatial"] is True
        a_ref, _, info_ref = sm.elasticnet_contamination_fit(delta_g, delta_t)
        assert info_ref["cv_spatial"] is False

    def test_single_patch_falls_back_with_warning(self):
        pytest.importorskip("sklearn")
        rng = np.random.default_rng(4)
        delta_t = rng.standard_normal((2, 2000))
        delta_g = rng.standard_normal(2000) * 0.3
        with pytest.warns(UserWarning, match="fewer than two patches"):
            _, _, info = sm.elasticnet_contamination_fit(
                delta_g, delta_t, patch_ids=np.zeros(2000, dtype=int))
        assert info["cv_spatial"] is False


class TestInverseVariancePixelWeights:
    def test_normalised_finite_and_monotonic(self):
        n_gal = np.array([10.0, 20.0, 0.0])
        n_ran = np.array([100.0, 100.0, 100.0])
        w = sm.inverse_variance_pixel_weights(n_gal, n_ran)
        assert np.isclose(w.mean(), 1.0)
        assert np.all(np.isfinite(w)) and np.all(w > 0)
        # More galaxies -> more Poisson variance -> less weight.
        assert w[1] < w[0] < w[2]

    def test_partial_coverage_is_downweighted(self):
        n_gal = np.array([10.0, 10.0])
        n_ran = np.array([100.0, 25.0])
        w = sm.inverse_variance_pixel_weights(n_gal, n_ran)
        assert w[1] < w[0]

    def test_empty_pixel_stays_finite(self):
        w = sm.inverse_variance_pixel_weights(np.zeros(5), np.full(5, 100.0))
        assert np.all(np.isfinite(w))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            sm.inverse_variance_pixel_weights(np.zeros(3), np.zeros(4))


class TestTemplateVetting:
    @pytest.fixture
    def maps(self):
        rng = np.random.default_rng(0)
        n_pix = 5000
        tracer = rng.standard_normal(n_pix)
        clean = rng.standard_normal(n_pix)
        dirty = 0.5 * tracer + 0.5 * rng.standard_normal(n_pix)
        return np.vstack([clean, dirty]), tracer

    def test_flags_the_contaminated_template(self, maps):
        delta_t, tracer = maps
        out = sm.vet_templates_against_tracer(delta_t, tracer)
        assert out["significance"][1] > 10 * out["significance"][0]
        assert abs(out["rho"][0]) < 0.1
        assert out["rho"][1] > 0.5

    def test_jackknife_errors_when_patches_given(self, maps):
        delta_t, tracer = maps
        ids = np.repeat(np.arange(50), tracer.size // 50)
        out = sm.vet_templates_against_tracer(delta_t, tracer, patch_ids=ids)
        assert out["jackknife"] is True
        assert np.all(out["sigma"] > 0)

    def test_rank_correlation_survives_a_monotonic_distortion(self, maps):
        """Spearman is invariant under a monotonic remap of a template."""
        delta_t, tracer = maps
        out = sm.vet_templates_against_tracer(delta_t, tracer)
        warped = delta_t.copy()
        warped[1] = np.exp(warped[1])
        out_w = sm.vet_templates_against_tracer(warped, tracer)
        np.testing.assert_allclose(out_w["rho"], out["rho"], atol=1e-12)

    def test_shape_mismatch_raises(self, maps):
        delta_t, tracer = maps
        with pytest.raises(ValueError, match="expected"):
            sm.vet_templates_against_tracer(delta_t, tracer[:-1])


class TestOvercorrectionDebias:
    def test_bias_is_negative_on_clean_mocks(self):
        """Weighting a contamination-free field removes power, not nothing."""
        rng = np.random.default_rng(11)
        n_pix, n_sys, n_mock = 3000, 6, 12
        delta_t = rng.standard_normal((n_sys, n_pix))
        delta_t -= delta_t.mean(1, keepdims=True)
        delta_t /= delta_t.std(1, keepdims=True)
        mocks = rng.standard_normal((n_mock, n_pix)) * 0.3

        def w_est(field):
            return np.array([float(np.var(field))])

        b_add, scatter = estimate_overcorrection_bias(
            mocks, delta_t, w_est, method="OLS", return_scatter=True)
        assert b_add.shape == (1,)
        assert scatter.shape == (1,)
        # OLS on 6 templates always removes some variance from a clean field.
        assert b_add[0] < 0

    def test_debias_subtracts(self):
        w = np.array([0.10, 0.05])
        b = np.array([-0.002, -0.001])
        np.testing.assert_allclose(debias_two_point_function(w, b), [0.102, 0.051])

    def test_debias_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            debias_two_point_function(np.zeros(3), np.zeros(2))

    def test_pixel_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="same footprint"):
            estimate_overcorrection_bias(
                np.zeros((2, 10)), np.zeros((2, 11)), lambda f: np.zeros(1))


class TestMethodMarginalisedCovariance:
    def test_adds_rank_one_term_and_stays_psd(self):
        rng = np.random.default_rng(5)
        n = 6
        a = rng.standard_normal((n, n))
        cov = a @ a.T + np.eye(n)
        w_a = rng.standard_normal(n) * 0.01
        w_b = w_a + rng.standard_normal(n) * 0.002
        out = method_marginalised_covariance(cov, w_a, w_b)
        assert np.allclose(out, out.T)
        assert np.all(np.linalg.eigvalsh(out) > 0)
        np.testing.assert_allclose(out - cov, np.outer(w_a - w_b, w_a - w_b))

    def test_block_diagonal_term(self):
        cov = np.eye(4)
        w_a = np.array([1.0, 2.0, 3.0, 4.0])
        w_b = np.zeros(4)
        out = method_marginalised_covariance(cov, w_a, w_b, block_sizes=[2, 2])
        # Cross-block terms are untouched.
        assert out[0, 2] == 0.0 and out[1, 3] == 0.0
        # Within-block terms are the outer product.
        assert np.isclose(out[0, 1], 2.0)

    def test_bad_shapes_raise(self):
        with pytest.raises(ValueError, match="expected"):
            method_marginalised_covariance(np.eye(3), np.zeros(4), np.zeros(4))
        with pytest.raises(ValueError, match="block_sizes"):
            method_marginalised_covariance(
                np.eye(4), np.zeros(4), np.zeros(4), block_sizes=[2, 3])
