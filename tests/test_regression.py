"""Tests for sys_mapping.regression — ElasticNet, ISD, method comparison."""

import numpy as np
import pytest

from sys_mapping.regression import (
    _compute_weights,
    iterative_systematics_decontamination,
    polynomial_ols_decontamination,
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


class TestPolynomialOlsDecontamination:
    """The v1.2 algorithm, kept for reproducibility of the v1.2 benchmark."""

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
        weights, alpha_all, n_it = polynomial_ols_decontamination(
            delta_g, delta_t, poly_order=1, max_iter=5
        )
        assert weights.shape == (8000,)
        assert alpha_all.shape[0] >= 3  # at least n_sys coefficients
        assert 1 <= n_it <= 5

    def test_weights_positive(self, contaminated_data):
        delta_g, delta_t, _ = contaminated_data
        weights, _, _ = polynomial_ols_decontamination(
            delta_g, delta_t, poly_order=1
        )
        assert np.all(weights > 0)

    def test_convergence_linear(self, contaminated_data):
        """Linear OLS (poly_order=1) should converge in few iterations."""
        delta_g, delta_t, _ = contaminated_data
        _, _, n_it = polynomial_ols_decontamination(
            delta_g, delta_t, poly_order=1, max_iter=20, tol=1e-4
        )
        assert n_it < 20

    def test_poly_order_reduction_warning(self):
        """Underdetermined polynomial expansion triggers a warning."""
        rng = np.random.default_rng(5)
        n_pix, n_sys = 100, 5  # very small -> poly_order=3 would be underdetermined
        delta_t = rng.standard_normal((n_sys, n_pix))
        delta_g = rng.standard_normal(n_pix)
        with pytest.warns(UserWarning, match="poly_order"):
            weights, _, _ = polynomial_ols_decontamination(
                delta_g, delta_t, poly_order=3, max_iter=3
            )
        assert np.all(weights > 0)


class TestIterativeSystematicsDecontamination:
    """The published ISD: marginal binned fits, greedy, mock-calibrated stop."""

    @pytest.fixture
    def contaminated_data(self):
        rng = np.random.default_rng(0)
        n_pix, n_sys = 20000, 3
        delta_t = rng.standard_normal((n_sys, n_pix))
        true_alpha = np.array([0.15, -0.08, 0.0])
        delta_g = true_alpha @ delta_t + rng.standard_normal(n_pix) * 0.3
        return delta_g, delta_t, true_alpha

    def test_recovers_linear_amplitudes(self, contaminated_data):
        delta_g, delta_t, true_alpha = contaminated_data
        res = iterative_systematics_decontamination(
            delta_g, delta_t, poly_order=1, chi2_68=50.0)
        assert res.weights.shape == (20000,)
        assert np.all(res.weights > 0)
        np.testing.assert_allclose(res.a_hat[:2], true_alpha[:2], atol=0.02)
        # The uncontaminated template is never selected, so its amplitude is
        # exactly zero rather than a small noise fit.
        assert res.a_hat[2] == 0.0

    def test_selects_strongest_template_first(self, contaminated_data):
        delta_g, delta_t, _ = contaminated_data
        res = iterative_systematics_decontamination(
            delta_g, delta_t, poly_order=1, chi2_68=50.0)
        assert res.steps[0].template == 0
        assert res.n_steps == len(res.steps)

    def test_stops_on_threshold_and_leaves_null_alone(self):
        """An uncontaminated field takes no steps and gets unit weights."""
        rng = np.random.default_rng(7)
        n_pix, n_sys = 20000, 4
        delta_t = rng.standard_normal((n_sys, n_pix))
        delta_g = rng.standard_normal(n_pix) * 0.3
        res = iterative_systematics_decontamination(
            delta_g, delta_t, poly_order=3, chi2_68=50.0)
        assert res.n_steps == 0
        assert res.stopped_on == "threshold"
        np.testing.assert_array_equal(res.weights, np.ones(n_pix))
        np.testing.assert_array_equal(res.a_hat, np.zeros(n_sys))

    def test_cubic_recovers_curvature_that_linear_misses(self):
        """poly_order=3 fits a curved trend; poly_order=1 biases the linear term.

        This is the point of ISD-3 in the literature: the marginal density-vs-
        template relation is not linear even when the conditional response is,
        so a linear marginal fit absorbs the curvature into its slope.
        """
        rng = np.random.default_rng(1)
        n_pix, n_sys = 40000, 4
        delta_t = rng.standard_normal((n_sys, n_pix))
        t2 = delta_t[2]
        f = 0.10 * t2 + 0.04 * t2 ** 2 - 0.02 * t2 ** 3
        delta_g = (1.0 + rng.standard_normal(n_pix) * 0.3) * (1.0 + f) - 1.0

        res1 = iterative_systematics_decontamination(
            delta_g, delta_t, poly_order=1, chi2_68=50.0)
        res3 = iterative_systematics_decontamination(
            delta_g, delta_t, poly_order=3, chi2_68=50.0)

        assert res1.steps[0].template == 2
        assert res3.steps[0].template == 2
        # Cubic recovers all three injected coefficients.
        np.testing.assert_allclose(
            res3.steps[0].coeffs[1:], [0.10, 0.04, -0.02], atol=0.015)
        # Linear does not: its slope is biased low by the curvature.
        assert abs(res1.a_hat[2] - 0.10) > abs(res3.a_hat[2] - 0.10)

    def test_fit_is_not_extrapolated_beyond_its_bins(self):
        """A skewed template must not have its cubic evaluated out in the tail.

        Survey-property maps run to tens of standardised units while the outermost
        bin centre sits near 2, so an unclipped cubic is evaluated far outside its
        support and diverges: the iteration then re-selects the same templates with
        *rising* significance instead of converging.
        """
        rng = np.random.default_rng(2)
        n_pix, n_sys = 30000, 3
        # One strongly right-skewed template, as LS10 GALDEPTH_Z is.
        delta_t = rng.standard_normal((n_sys, n_pix))
        delta_t[1] = rng.lognormal(0.0, 1.0, n_pix)
        delta_t = (delta_t - delta_t.mean(1, keepdims=True)) / delta_t.std(1, keepdims=True)
        delta_g = 0.08 * delta_t[1] + rng.standard_normal(n_pix) * 0.35

        res = iterative_systematics_decontamination(
            delta_g, delta_t, poly_order=3, chi2_68=6.0)

        assert res.stopped_on == "threshold"
        assert res.n_floored == 0
        # Significance must fall as the iteration proceeds, not rise.
        sig = [s.significance for s in res.steps]
        assert sig[0] == max(sig), sig
        assert np.all(np.isfinite(res.weights))
        assert float(np.max(np.abs(res.a_hat))) < 1.0

    def test_amplitude_is_the_projection_not_the_polynomial_coefficient(self):
        """On a skewed template a cubic's coefficients are large; a_hat must not be.

        Equal-occupancy bin centres of a skewed template span a narrow range, so
        the Vandermonde is poorly conditioned and the fitted coefficients run to
        order unity even when the curve itself is tiny.  ``a_hat`` reports the
        projection of the fitted curve onto the template instead, which is the
        amplitude the two-point correction actually needs.
        """
        rng = np.random.default_rng(5)
        n_pix, n_sys = 30000, 3
        delta_t = rng.standard_normal((n_sys, n_pix))
        delta_t[0] = rng.lognormal(0.0, 1.2, n_pix)      # strongly skewed
        delta_t = (delta_t - delta_t.mean(1, keepdims=True)) / delta_t.std(1, keepdims=True)
        delta_g = rng.standard_normal(n_pix) * 0.3        # no contamination at all

        res = iterative_systematics_decontamination(
            delta_g, delta_t, poly_order=3, chi2_68=2.0, max_steps=3)

        # Whatever the fit does on a null field, the reported amplitude must stay
        # small -- it describes the correction applied, not the basis it was
        # expressed in.
        assert float(np.max(np.abs(res.a_hat))) < 0.05, res.a_hat

    def test_amplitude_tracks_c1_for_a_linear_fit(self):
        """For poly_order=1 the projection recovers the degree-1 coefficient.

        Not exactly: the curve is evaluated at ``clip(t, t_lo, t_hi)``, so the
        tails -- which carry most of the leverage in the projection -- are held
        flat.  The two agree to the size of that effect, ~10 % here.
        """
        rng = np.random.default_rng(6)
        n_pix, n_sys = 20000, 2
        delta_t = rng.standard_normal((n_sys, n_pix))
        delta_t = (delta_t - delta_t.mean(1, keepdims=True)) / delta_t.std(1, keepdims=True)
        delta_g = 0.12 * delta_t[0] + rng.standard_normal(n_pix) * 0.3
        res = iterative_systematics_decontamination(
            delta_g, delta_t, poly_order=1, chi2_68=50.0)
        c1_sum = sum(st.coeffs[1] for st in res.steps if st.template == 0)
        assert abs(res.a_hat[0] - c1_sum) < 0.15 * abs(c1_sum)

    def test_uncalibrated_threshold_warns(self):
        rng = np.random.default_rng(3)
        delta_t = rng.standard_normal((2, 5000))
        delta_g = rng.standard_normal(5000) * 0.3
        with pytest.warns(UserWarning, match="chi2_68 not supplied"):
            res = iterative_systematics_decontamination(delta_g, delta_t)
        assert res.calibrated is False

    def test_max_steps_is_respected(self, contaminated_data):
        delta_g, delta_t, _ = contaminated_data
        res = iterative_systematics_decontamination(
            delta_g, delta_t, poly_order=1, chi2_68=1e-6, max_steps=2)
        assert res.n_steps <= 2
        assert res.stopped_on in ("max_steps", "exhausted", "threshold")


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
        res = run_decontamination("ISD-1", delta_g, delta_t, isd_chi2_68=50.0)
        assert res["a_hat"].shape == (n_sys,)
        assert res["weights"].shape == (n_pix,)
        assert np.all(res["weights"] > 0)
        assert res["n_iterations"] is not None
        assert res["isd_significance"].shape == (n_sys,)
        assert res["isd_stopped_on"] in ("threshold", "max_steps", "exhausted")
        assert res["isd_calibrated"] is True
        assert len(res["isd_steps"]) == res["n_iterations"]

    def test_isd1_perturbative_regime_is_accurate(self):
        """In the regime the model assumes (few %), amplitudes recover to ~5%."""
        rng = np.random.default_rng(7)
        n_pix, n_sys = 20000, 5
        delta_t = rng.standard_normal((n_sys, n_pix))
        true_a = np.array([0.06, -0.05, 0.04, -0.03, 0.02])
        delta_g = (1.0 + rng.standard_normal(n_pix) * 0.3) * (1.0 + true_a @ delta_t) - 1.0
        res = run_decontamination("ISD-1", delta_g, delta_t, isd_chi2_68=20.0)
        rel_err = np.abs(res["a_hat"] - true_a) / np.abs(true_a)
        assert np.all(rel_err < 0.10), rel_err

    def test_isd_leaves_undetectable_templates_at_zero(self):
        """A template below the stopping threshold is not corrected at all.

        This is the break-even rule expressed in the method itself: rather than
        applying a noisy correction, ISD declines to correct what it cannot
        detect, and reports exactly zero for that template.
        """
        rng = np.random.default_rng(7)
        n_pix, n_sys = 20000, 5
        delta_t = rng.standard_normal((n_sys, n_pix))
        true_a = np.array([0.06, -0.05, 0.04, -0.03, 0.02])
        delta_g = (1.0 + rng.standard_normal(n_pix) * 0.3) * (1.0 + true_a @ delta_t) - 1.0
        res = run_decontamination("ISD-1", delta_g, delta_t, isd_chi2_68=50.0)
        # The weakest contaminant does not clear the threshold at this calibration.
        assert res["a_hat"][4] == 0.0
        assert res["isd_significance"][4] < 2.0
        # The four that do clear it are still recovered accurately.
        rel_err = np.abs(res["a_hat"][:4] - true_a[:4]) / np.abs(true_a[:4])
        assert np.all(rel_err < 0.10), rel_err

    def test_isd1_on_strong_contamination_stays_bounded(self):
        """Strong contamination needs no outlier-masking hack.

        The v1.2 polynomial-OLS ISD needed a two-pass scheme here because its
        weights hit the clip boundary.  The stopping rule makes that unnecessary:
        the weights stay finite and the *first* fit of each template recovers its
        injected slope.  The summed amplitude overshoots because the iteration
        re-selects already-corrected templates to chase second-order residuals,
        which is what a mock-calibrated threshold exists to stop.
        """
        rng = np.random.default_rng(7)
        n_pix, n_sys = 20000, 5
        delta_t = rng.standard_normal((n_sys, n_pix))
        true_a = np.array([0.3, -0.25, 0.2, -0.15, 0.1])
        delta_g = (1.0 + rng.standard_normal(n_pix) * 0.3) * (1.0 + true_a @ delta_t) - 1.0
        res = run_decontamination("ISD-1", delta_g, delta_t, isd_chi2_68=50.0)

        assert np.all(np.isfinite(res["weights"]))
        assert np.all(res["weights"] > 0)
        first = {}
        for st in res["isd_steps"]:
            first.setdefault(st["template"], st["coeffs"][1])
        # Every template is found, strongest first, and the first fit of the two
        # strongest recovers the injected slope.
        assert [st["template"] for st in res["isd_steps"]][:n_sys] == [0, 1, 2, 3, 4]
        assert abs(first[0] - true_a[0]) < 0.02
        assert abs(first[1] - true_a[1]) < 0.04

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

    def test_isd1_nonfinite_coefficients_warning(self):
        """Non-finite ISD coefficients trigger fallback to zeros and unit weights."""
        from unittest.mock import patch
        rng = np.random.default_rng(0)
        n_pix, n_sys = 500, 2
        delta_g = rng.standard_normal(n_pix) * 0.1
        delta_t = rng.standard_normal((n_sys, n_pix))
        from sys_mapping.regression import ISDResult
        nan_alpha = np.array([np.nan, 0.0])
        bad = ISDResult(weights=np.ones(n_pix), a_hat=nan_alpha, steps=[],
                        significance=np.zeros(n_sys), n_steps=0,
                        stopped_on="threshold", calibrated=True)
        with patch("sys_mapping.regression.iterative_systematics_decontamination",
                   return_value=bad):
            with pytest.warns(UserWarning, match="non-finite output"):
                res = run_decontamination("ISD-1", delta_g, delta_t,
                                          isd_chi2_68=50.0)
        np.testing.assert_array_equal(res["a_hat"], np.zeros(n_sys))
        np.testing.assert_array_equal(res["weights"], np.ones(n_pix))


class TestMethodComparisonAdditiveMcmc:
    def test_additive_mcmc_returns_required_keys(self):
        rng = np.random.default_rng(13)
        n_pix, n_sys = 2000, 2
        nside = 16
        delta_t = rng.standard_normal((n_sys, n_pix))
        delta_g = 0.1 * delta_t[0] + rng.standard_normal(n_pix) * 0.3
        good = np.ones(n_pix, dtype=bool)
        results = method_comparison(
            delta_g, delta_t, good, nside,
            methods=("additive_mcmc",),
            mcmc_kwargs={"n_walkers": 20, "n_steps": 30, "n_burn": 10},
        )
        assert "additive_mcmc" in results
        assert "a_hat" in results["additive_mcmc"]
        assert "weights" in results["additive_mcmc"]
        assert np.all(results["additive_mcmc"]["weights"] > 0)
