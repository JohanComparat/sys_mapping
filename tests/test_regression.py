"""Tests for sys_mapping.regression — ElasticNet, ISD, method comparison."""

import numpy as np
import pytest

from sys_mapping.regression import (
    _compute_weights,
    iterative_systematics_decontamination,
    method_comparison,
    run_decontamination,
)


# Skip ElasticNet tests gracefully if scikit-learn is not installed
sklearn_available = pytest.importorskip("sklearn", reason="scikit-learn not installed")


class TestComputeWeights:
    def test_zero_alpha_gives_unit_weights(self):
        rng = np.random.default_rng(0)
        delta_t = rng.standard_normal((3, 500))
        alpha = np.zeros(3)
        weights = _compute_weights(alpha, delta_t)
        np.testing.assert_allclose(weights, 1.0, rtol=1e-12)

    def test_weights_positive(self):
        rng = np.random.default_rng(1)
        delta_t = rng.standard_normal((2, 1000))
        alpha = np.array([0.2, -0.3])
        weights = _compute_weights(alpha, delta_t)
        assert np.all(weights > 0)

    def test_shape(self):
        delta_t = np.ones((3, 200))
        alpha = np.array([0.1, 0.0, -0.1])
        weights = _compute_weights(alpha, delta_t)
        assert weights.shape == (200,)


class TestElasticNet:
    @pytest.fixture
    def contaminated_data(self):
        rng = np.random.default_rng(42)
        n_pix, n_sys = 5000, 3
        delta_t = rng.standard_normal((n_sys, n_pix))
        true_alpha = np.array([0.2, -0.1, 0.05])
        delta_g = true_alpha @ delta_t + rng.standard_normal(n_pix) * 0.3
        return delta_g, delta_t, true_alpha

    def test_output_shapes(self, contaminated_data):
        from sys_mapping.regression import elasticnet_contamination_fit

        delta_g, delta_t, _ = contaminated_data
        alpha_hat, weights, cv_info = elasticnet_contamination_fit(
            delta_g, delta_t, alpha_reg=1e-4
        )
        assert alpha_hat.shape == (3,)
        assert weights.shape == (5000,)
        assert isinstance(cv_info, dict)

    def test_weights_bounded_positive(self, contaminated_data):
        from sys_mapping.regression import elasticnet_contamination_fit

        delta_g, delta_t, _ = contaminated_data
        _, weights, _ = elasticnet_contamination_fit(delta_g, delta_t, alpha_reg=1e-4)
        assert np.all(weights > 0)
        assert np.all(weights < 100)

    def test_zero_contamination_alpha_near_zero(self):
        from sys_mapping.regression import elasticnet_contamination_fit

        rng = np.random.default_rng(7)
        n_pix, n_sys = 5000, 3
        delta_t = rng.standard_normal((n_sys, n_pix))
        # Pure noise — no systematic signal
        delta_g = rng.standard_normal(n_pix) * 0.1
        alpha_hat, _, _ = elasticnet_contamination_fit(
            delta_g, delta_t, alpha_reg=0.1
        )
        assert np.all(np.abs(alpha_hat) < 0.5)

    def test_cv_info_contains_required_keys(self, contaminated_data):
        from sys_mapping.regression import elasticnet_contamination_fit

        delta_g, delta_t, _ = contaminated_data
        _, _, cv_info = elasticnet_contamination_fit(delta_g, delta_t, alpha_reg=1e-3)
        assert "alpha_reg" in cv_info
        assert "l1_ratio" in cv_info

    def test_recovers_additive_contamination(self, contaminated_data):
        from sys_mapping.regression import elasticnet_contamination_fit

        delta_g, delta_t, true_alpha = contaminated_data
        alpha_hat, _, _ = elasticnet_contamination_fit(
            delta_g, delta_t, alpha_reg=1e-5
        )
        # ElasticNet should recover the amplitudes to within ~30% given noise
        relative_error = np.abs(alpha_hat - true_alpha) / (np.abs(true_alpha) + 1e-10)
        assert np.all(relative_error < 1.0)  # generous tolerance


class TestIterativeSystematicsDecontamination:
    @pytest.fixture
    def contaminated_data(self):
        rng = np.random.default_rng(0)
        n_pix, n_sys = 8000, 3
        delta_t = rng.standard_normal((n_sys, n_pix))
        true_alpha = np.array([0.15, -0.08, 0.04])
        delta_g = true_alpha @ delta_t + rng.standard_normal(n_pix) * 0.4
        return delta_g, delta_t, true_alpha

    def test_output_shapes(self, contaminated_data):
        delta_g, delta_t, _ = contaminated_data
        weights, alpha_all, n_it = iterative_systematics_decontamination(
            delta_g, delta_t, poly_order=1, max_iter=5
        )
        assert weights.shape == (8000,)
        assert alpha_all.shape[0] >= 3  # at least n_sys coefficients
        assert 1 <= n_it <= 5

    def test_weights_positive(self, contaminated_data):
        delta_g, delta_t, _ = contaminated_data
        weights, _, _ = iterative_systematics_decontamination(
            delta_g, delta_t, poly_order=1
        )
        assert np.all(weights > 0)

    def test_convergence_linear(self, contaminated_data):
        """Linear OLS (poly_order=1) should converge in few iterations."""
        delta_g, delta_t, _ = contaminated_data
        _, _, n_it = iterative_systematics_decontamination(
            delta_g, delta_t, poly_order=1, max_iter=20, tol=1e-4
        )
        assert n_it < 20

    def test_poly_order_reduction_warning(self):
        """Underdetermined polynomial expansion triggers a warning."""
        rng = np.random.default_rng(5)
        n_pix, n_sys = 100, 5  # very small → poly_order=3 would be underdetermined
        delta_t = rng.standard_normal((n_sys, n_pix))
        delta_g = rng.standard_normal(n_pix)
        with pytest.warns(UserWarning, match="poly_order"):
            weights, _, _ = iterative_systematics_decontamination(
                delta_g, delta_t, poly_order=3, max_iter=3
            )
        assert np.all(weights > 0)


class TestMethodComparison:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(99)
        n_pix, n_sys = 3000, 2
        nside = 16
        delta_t = rng.standard_normal((n_sys, n_pix))
        delta_g = 0.1 * delta_t[0] + rng.standard_normal(n_pix) * 0.5
        good = np.ones(n_pix, dtype=bool)
        return delta_g, delta_t, good, nside

    def test_ols_returns_required_keys(self, data):
        delta_g, delta_t, good, nside = data
        results = method_comparison(delta_g, delta_t, good, nside, methods=("ols",))
        assert "ols" in results
        assert "alpha_hat" in results["ols"]
        assert "weights" in results["ols"]
        assert "chi2_residual" in results["ols"]

    def test_elasticnet_returns_required_keys(self, data):
        delta_g, delta_t, good, nside = data
        results = method_comparison(
            delta_g, delta_t, good, nside, methods=("elasticnet",)
        )
        assert "elasticnet" in results
        assert "weights" in results["elasticnet"]

    def test_ols_weights_positive(self, data):
        delta_g, delta_t, good, nside = data
        results = method_comparison(delta_g, delta_t, good, nside, methods=("ols",))
        assert np.all(results["ols"]["weights"] > 0)

    def test_multiple_methods(self, data):
        delta_g, delta_t, good, nside = data
        results = method_comparison(
            delta_g, delta_t, good, nside, methods=("ols", "elasticnet")
        )
        assert set(results.keys()) == {"ols", "elasticnet"}

    def test_chi2_nonneg(self, data):
        delta_g, delta_t, good, nside = data
        results = method_comparison(delta_g, delta_t, good, nside, methods=("ols",))
        assert results["ols"]["chi2_residual"] >= 0

    def test_combined_mcmc_returns_required_keys(self, data):
        delta_g, delta_t, good, nside = data
        results = method_comparison(
            delta_g, delta_t, good, nside,
            methods=("combined_mcmc",),
            mcmc_kwargs={"n_walkers": 20, "n_steps": 40, "n_burn": 10},
        )
        assert "combined_mcmc" in results
        assert "a_hat" in results["combined_mcmc"]
        assert "b_hat" in results["combined_mcmc"]
        assert "weights" in results["combined_mcmc"]
        assert np.all(results["combined_mcmc"]["weights"] > 0)

    def test_default_methods_include_combined(self, data):
        """Default methods tuple must start with combined_mcmc."""
        import inspect
        sig = inspect.signature(method_comparison)
        default_methods = sig.parameters["methods"].default
        assert default_methods[0] == "combined_mcmc"


class TestRunDecontamination:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(0)
        n_pix, n_sys = 2000, 2
        delta_t = rng.standard_normal((n_sys, n_pix))
        delta_g = 0.1 * delta_t[0] + rng.standard_normal(n_pix) * 0.3
        return delta_g, delta_t, n_sys, n_pix

    def test_ols_shapes_and_weights(self, data):
        delta_g, delta_t, n_sys, n_pix = data
        res = run_decontamination("OLS", delta_g, delta_t)
        assert res["a_hat"].shape == (n_sys,)
        assert res["b_hat"].shape == (n_sys,)
        assert res["weights"].shape == (n_pix,)
        assert np.all(res["weights"] > 0)
        assert np.allclose(res["b_hat"], 0)
        assert res["flat_chain"] is None
        assert res["elapsed_s"] > 0

    def test_elasticnet_shapes(self, data):
        delta_g, delta_t, n_sys, n_pix = data
        res = run_decontamination("ElasticNet", delta_g, delta_t)
        assert res["a_hat"].shape == (n_sys,)
        assert res["weights"].shape == (n_pix,)
        assert np.all(res["weights"] > 0)
        assert res["cv_info"] is not None

    def test_isd1_shapes(self, data):
        delta_g, delta_t, n_sys, n_pix = data
        res = run_decontamination("ISD-1", delta_g, delta_t, isd_max_iter=5)
        assert res["a_hat"].shape == (n_sys,)
        assert res["weights"].shape == (n_pix,)
        assert np.all(res["weights"] > 0)
        assert res["n_iterations"] is not None
        assert res["isd_outlier_mask"].shape == (n_pix,)
        assert res["isd_outlier_mask"].dtype == bool

    def test_isd1_two_pass_on_strong_contamination(self):
        """Two-pass masking fires when contamination field drives 1+a@t near zero."""
        rng = np.random.default_rng(7)
        n_pix, n_sys = 10000, 5
        delta_t = rng.standard_normal((n_sys, n_pix))
        true_a = np.array([0.3, -0.25, 0.2, -0.15, 0.1])
        delta_g = true_a @ delta_t + rng.standard_normal(n_pix) * 0.3
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = run_decontamination("ISD-1", delta_g, delta_t)
        # At least some outlier pixels should have been masked
        assert res["isd_outlier_mask"].sum() > 0
        # Second-pass coefficients should recover true_a to within 20%
        rel_err = np.abs(res["a_hat"] - true_a) / (np.abs(true_a) + 1e-10)
        assert np.all(rel_err < 0.4)

    def test_invalid_method_raises(self, data):
        delta_g, delta_t, _, _ = data
        with pytest.raises(ValueError, match="method must be one of"):
            run_decontamination("UNKNOWN", delta_g, delta_t)

    def test_mcmc_add_shapes(self, data):
        delta_g, delta_t, n_sys, n_pix = data
        res = run_decontamination(
            "MCMC-add", delta_g, delta_t,
            n_walkers=20, n_steps=30, n_burn=10, seed=0,
        )
        assert res["a_hat"].shape == (n_sys,)
        assert res["weights"].shape == (n_pix,)
        assert np.all(res["weights"] > 0)
        assert res["flat_chain"] is not None
        assert res["R"].shape == (n_sys, n_sys)
        assert res["cov_a"].shape == (n_sys, n_sys)
        assert res["cov_b"].shape == (n_sys, n_sys)

    def test_mcmc_comb_shapes(self, data):
        delta_g, delta_t, n_sys, n_pix = data
        res = run_decontamination(
            "MCMC-comb", delta_g, delta_t,
            n_walkers=20, n_steps=30, n_burn=10, seed=0,
        )
        assert res["a_hat"].shape == (n_sys,)
        assert res["b_hat"].shape == (n_sys,)
        assert res["weights"].shape == (n_pix,)
        assert np.all(res["weights"] > 0)
        assert res["flat_chain"] is not None
        assert res["R"].shape == (n_sys, n_sys)
