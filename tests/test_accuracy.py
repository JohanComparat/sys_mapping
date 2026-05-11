"""Numerical accuracy and precision tests for sys_mapping functions.

Each test class verifies that computations meet the accuracy thresholds
documented in the corresponding function's docstring.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

N_PIX = 5_000
N_SYS = 3


def _make_data(n_pix: int = N_PIX, n_sys: int = N_SYS, seed: int = 42):
    rng = np.random.default_rng(seed)
    delta_t = rng.standard_normal((n_sys, n_pix))
    delta_t -= delta_t.mean(axis=1, keepdims=True)
    delta_t /= delta_t.std(axis=1, keepdims=True)
    delta_g = rng.standard_normal(n_pix) * 0.1
    a = rng.uniform(-0.05, 0.05, n_sys)
    b = rng.uniform(-0.03, 0.03, n_sys)
    return delta_g, delta_t, a, b


# ---------------------------------------------------------------------------
# contamination.py
# ---------------------------------------------------------------------------


class TestContaminationAccuracy:
    def test_roundtrip_max_error(self):
        """apply then invert recovers delta_g with max absolute error < 1e-5."""
        from sys_mapping.contamination import apply_contamination, invert_contamination

        delta_g, delta_t, a, b = _make_data()
        dg = jnp.asarray(delta_g)
        dt = jnp.asarray(delta_t)
        aj, bj = jnp.asarray(a), jnp.asarray(b)

        obs = apply_contamination(dg, dt, aj, bj)
        recovered = invert_contamination(obs, dt, aj, bj)
        err = float(jnp.max(jnp.abs(recovered - dg)))
        assert err < 1e-5, f"Roundtrip max error {err:.2e} ≥ 1e-5"

    def test_roundtrip_rms_error(self):
        """apply then invert roundtrip RMS error < 1e-6."""
        from sys_mapping.contamination import apply_contamination, invert_contamination

        delta_g, delta_t, a, b = _make_data(n_pix=50_000)
        obs = apply_contamination(jnp.asarray(delta_g), jnp.asarray(delta_t),
                                  jnp.asarray(a), jnp.asarray(b))
        recovered = invert_contamination(obs, jnp.asarray(delta_t),
                                         jnp.asarray(a), jnp.asarray(b))
        rms = float(jnp.sqrt(jnp.mean((recovered - jnp.asarray(delta_g)) ** 2)))
        assert rms < 1e-6, f"Roundtrip RMS error {rms:.2e} ≥ 1e-6"

    def test_zero_params_exact_identity(self):
        """With a=b=0, apply_contamination is the exact identity."""
        from sys_mapping.contamination import apply_contamination

        delta_g, delta_t, _, _ = _make_data()
        a = jnp.zeros(N_SYS)
        b = jnp.zeros(N_SYS)
        obs = apply_contamination(jnp.asarray(delta_g), jnp.asarray(delta_t), a, b)
        np.testing.assert_allclose(np.asarray(obs), delta_g, atol=1e-12)

    def test_two_point_correction_formula(self):
        """compute_two_point_correction matches analytic formula to rtol=1e-6."""
        from sys_mapping.contamination import compute_two_point_correction

        rng = np.random.default_rng(0)
        n_bins, n_sys = 12, 4
        w_obs = rng.standard_normal(n_bins) * 0.1
        a_sq = np.abs(rng.uniform(0, 0.05, n_sys))
        b_sq = np.abs(rng.uniform(0, 0.03, n_sys))
        tcorr = np.abs(rng.standard_normal((n_sys, n_bins))) * 0.01

        result = np.asarray(compute_two_point_correction(
            jnp.asarray(w_obs), jnp.asarray(a_sq),
            jnp.asarray(b_sq), jnp.asarray(tcorr),
        ))
        add_bias = (a_sq[:, None] * tcorr).sum(axis=0)
        mult_bias = (b_sq[:, None] * tcorr).sum(axis=0)
        expected = (w_obs - add_bias) / (1.0 + mult_bias)
        np.testing.assert_allclose(result, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# maps.py
# ---------------------------------------------------------------------------


class TestMapsAccuracy:
    def test_power_spectrum_unit_variance_all_families(self):
        """C_ℓ integrates to unit variance for all five template families."""
        from sys_mapping.maps import systematic_power_spectrum

        nside = 64
        ell = np.arange(3 * nside, dtype=float)
        for family in range(5):
            cl = systematic_power_spectrum(nside, family)
            variance = float(np.sum((2 * ell + 1) / (4 * np.pi) * cl))
            np.testing.assert_allclose(
                variance, 1.0, rtol=1e-10,
                err_msg=f"Family {family}: variance={variance:.8f} ≠ 1",
            )

    def test_power_spectrum_monopole_zero(self):
        """Monopole (ℓ=0) is forced to exactly zero for all families."""
        from sys_mapping.maps import systematic_power_spectrum

        for family in range(5):
            cl = systematic_power_spectrum(32, family)
            assert cl[0] == 0.0, f"Family {family}: monopole {cl[0]} ≠ 0"

    def test_generate_map_unit_std(self):
        """generate_systematic_map returns std exactly 1.0 (enforced by normalization)."""
        from sys_mapping.maps import generate_systematic_map

        for family in range(5):
            m = generate_systematic_map(64, family, seed=family * 100)
            np.testing.assert_allclose(
                np.std(m), 1.0, rtol=1e-10,
                err_msg=f"Family {family}: std={np.std(m):.8f} ≠ 1",
            )

    def test_generate_map_zero_mean(self):
        """generate_systematic_map returns mean exactly 0.0 (enforced by subtraction)."""
        from sys_mapping.maps import generate_systematic_map

        for family in range(5):
            m = generate_systematic_map(64, family, seed=family * 100)
            np.testing.assert_allclose(
                np.mean(m), 0.0, atol=1e-14,
                err_msg=f"Family {family}: mean={np.mean(m):.2e} ≠ 0",
            )

    def test_pixelize_catalog_count_conservation(self):
        """Total pixel count equals n_gal to machine precision."""
        from sys_mapping.maps import pixelize_catalog

        rng = np.random.default_rng(0)
        n_gal = 123_456
        ra = rng.uniform(0, 360, n_gal)
        dec = rng.uniform(-90, 90, n_gal)
        counts = pixelize_catalog(ra, dec, nside=32)
        assert int(counts.sum()) == n_gal

    def test_pixelize_catalog_weighted_conservation(self):
        """Sum of weighted pixel counts equals sum of weights."""
        from sys_mapping.maps import pixelize_catalog

        rng = np.random.default_rng(1)
        n_gal = 10_000
        ra = rng.uniform(0, 360, n_gal)
        dec = rng.uniform(-90, 90, n_gal)
        weights = rng.uniform(0.5, 1.5, n_gal)
        counts = pixelize_catalog(ra, dec, nside=32, weights=weights)
        np.testing.assert_allclose(counts.sum(), weights.sum(), rtol=1e-10)

    def test_overdensity_mean_near_zero(self):
        """Overdensity mean < 0.01 by construction of the normalization."""
        from sys_mapping.maps import pixelize_catalog, compute_overdensity

        rng = np.random.default_rng(2)
        nside = 32
        gal = pixelize_catalog(rng.uniform(0, 360, 100_000),
                                rng.uniform(-90, 90, 100_000), nside)
        rand = pixelize_catalog(rng.uniform(0, 360, 500_000),
                                 rng.uniform(-90, 90, 500_000), nside)
        delta_g, _ = compute_overdensity(gal, rand)
        assert abs(np.mean(delta_g)) < 0.01, f"Overdensity mean {np.mean(delta_g):.4f} too large"

    def test_overdensity_masking_removes_low_randoms(self):
        """Pixels with randoms below threshold are excluded from delta_g."""
        from sys_mapping.maps import compute_overdensity

        n_pix = 100
        gal = np.ones(n_pix)
        rand = np.ones(n_pix)
        rand[0] = 0.0  # this pixel should be masked (below 10% of max=1)
        _, good = compute_overdensity(gal, rand)
        assert not good[0], "Pixel with zero randoms should be masked"
        assert good[1:].all(), "Pixels with full randoms should be unmasked"


# ---------------------------------------------------------------------------
# likelihood.py
# ---------------------------------------------------------------------------


class TestLikelihoodAccuracy:
    def test_gaussian_matches_scipy_logpdf(self):
        """At a=b=0, Gaussian LL equals scipy.stats.norm.logpdf sum to rtol=1e-5."""
        from scipy.stats import norm
        from sys_mapping.likelihood import make_log_likelihood
        from sys_mapping.contamination import pack_params

        delta_g, delta_t, _, _ = _make_data()
        sigma = float(np.std(delta_g))
        theta = jnp.asarray(pack_params(np.zeros(N_SYS), np.zeros(N_SYS), sigma, model="combined"))

        log_lik = make_log_likelihood(N_SYS, "combined")
        ll = float(log_lik(theta, jnp.asarray(delta_g), jnp.asarray(delta_t)))
        expected = float(np.sum(norm.logpdf(delta_g, loc=0.0, scale=sigma)))
        np.testing.assert_allclose(ll, expected, rtol=1e-5,
                                   err_msg=f"LL={ll:.4f} ≠ scipy {expected:.4f}")

    def test_mle_gradient_below_1e4(self):
        """Gradient of additive Gaussian LL at the OLS solution is < 1e-4."""
        from sys_mapping.likelihood import make_log_likelihood
        from sys_mapping.contamination import pack_params

        rng = np.random.default_rng(99)
        n_pix = 10_000
        delta_g = rng.normal(0, 0.1, n_pix)
        delta_t = rng.standard_normal((N_SYS, n_pix))
        delta_t -= delta_t.mean(axis=1, keepdims=True)

        C = delta_t @ delta_t.T
        a_mle = np.linalg.solve(C, delta_t @ delta_g)
        residual = delta_g - a_mle @ delta_t
        sigma_mle = float(np.sqrt(np.mean(residual ** 2)))

        theta = jnp.asarray(pack_params(a_mle, None, sigma_mle, model="additive"))
        log_lik = make_log_likelihood(N_SYS, "additive")
        grad = np.asarray(jax.grad(
            lambda t: log_lik(t, jnp.asarray(delta_g), jnp.asarray(delta_t))
        )(theta))
        max_grad = float(np.max(np.abs(grad)))
        assert max_grad < 1e-4, f"Max gradient component {max_grad:.2e} ≥ 1e-4 at OLS MLE"

    def test_skew_gamma_zero_recovers_gaussian(self):
        """Skew-normal LL with gamma=0 matches Gaussian LL to atol=1e-6."""
        from sys_mapping.likelihood import make_log_likelihood
        from sys_mapping.contamination import pack_params

        delta_g, delta_t, _, _ = _make_data()
        sigma = 0.1
        a, b = np.zeros(N_SYS), np.zeros(N_SYS)

        theta_g = jnp.asarray(pack_params(a, b, sigma, model="combined"))
        theta_s = jnp.asarray(pack_params(a, b, sigma, gamma=0.0, model="combined"))

        dg, dt = jnp.asarray(delta_g), jnp.asarray(delta_t)
        ll_g = float(make_log_likelihood(N_SYS, "combined", use_skewed=False)(theta_g, dg, dt))
        ll_s = float(make_log_likelihood(N_SYS, "combined", use_skewed=True)(theta_s, dg, dt))
        np.testing.assert_allclose(ll_s, ll_g, atol=1e-6,
                                   err_msg=f"Skew(gamma=0)={ll_s:.4f} ≠ Gaussian={ll_g:.4f}")

    def test_jax_gradient_finite_everywhere(self):
        """JAX gradient of LL w.r.t. theta is finite for all three models."""
        from sys_mapping.likelihood import make_log_likelihood
        from sys_mapping.contamination import pack_params

        delta_g, delta_t, a, b = _make_data()
        sigma = 0.1
        dg, dt = jnp.asarray(delta_g), jnp.asarray(delta_t)

        for model in ("additive", "multiplicative", "combined"):
            theta = jnp.asarray(pack_params(a, b, sigma, model=model))
            log_lik = make_log_likelihood(N_SYS, model)
            grad = np.asarray(jax.grad(lambda t: log_lik(t, dg, dt))(theta))
            assert np.all(np.isfinite(grad)), \
                f"Gradient contains non-finite values for model={model}"


# ---------------------------------------------------------------------------
# correction.py
# ---------------------------------------------------------------------------


class TestCorrectionAccuracy:
    def test_rotate_templates_orthogonality(self):
        """Rotated template covariance matrix is diagonal to atol=1e-10."""
        from sys_mapping.correction import rotate_templates

        _, delta_t, _, _ = _make_data()
        delta_t_rot, _, _ = rotate_templates(delta_t)
        cov_rot = delta_t_rot @ delta_t_rot.T / N_PIX
        off_diag = cov_rot - np.diag(np.diag(cov_rot))
        max_off = float(np.max(np.abs(off_diag)))
        assert max_off < 1e-10, f"Off-diagonal covariance {max_off:.2e} ≥ 1e-10"

    def test_rotate_templates_rotation_matrix_orthogonal(self):
        """rotation_matrix @ rotation_matrix.T == I to atol=1e-12."""
        from sys_mapping.correction import rotate_templates

        _, delta_t, _, _ = _make_data()
        _, R, _ = rotate_templates(delta_t)
        np.testing.assert_allclose(R @ R.T, np.eye(N_SYS), atol=1e-12)

    def test_rotate_templates_variance_preserved(self):
        """Sum of PCA eigenvalues equals trace of original covariance."""
        from sys_mapping.correction import rotate_templates
        from sys_mapping.utils import compute_covariance_matrix

        _, delta_t, _, _ = _make_data()
        cov_orig = compute_covariance_matrix(delta_t)
        _, _, eigenvalues = rotate_templates(delta_t)
        np.testing.assert_allclose(
            np.sum(eigenvalues), np.trace(cov_orig), rtol=1e-10,
        )

    def test_transform_params_roundtrip(self):
        """Rotate then back-transform parameters recovers originals to atol=1e-12."""
        from sys_mapping.correction import rotate_templates, transform_params_from_rotated

        _, delta_t, a, b = _make_data()
        _, R, _ = rotate_templates(delta_t)
        a_rot, b_rot = R @ a, R @ b
        a_back, b_back = transform_params_from_rotated(a_rot, b_rot, R)
        np.testing.assert_allclose(a_back, a, atol=1e-12)
        np.testing.assert_allclose(b_back, b, atol=1e-12)

    def test_debias_exact_known_variance(self):
        """debias_params(a_hat, b_hat, var) recovers true a², b² to atol=1e-10."""
        from sys_mapping.correction import debias_params

        true_a_sq = np.array([0.0025, 0.0100, 0.0004])
        true_b_sq = np.array([0.0009, 0.0001, 0.0016])
        noise_a = np.array([0.0005, 0.0010, 0.0002])
        noise_b = np.array([0.0003, 0.0001, 0.0008])
        # â² = a² + noise  →  â = sqrt(a² + noise) * sign(a)
        a_hat = np.sqrt(true_a_sq + noise_a) * np.array([1, -1, 1])
        b_hat = np.sqrt(true_b_sq + noise_b)
        a_sq_deb, b_sq_deb = debias_params(a_hat, b_hat, noise_a, noise_b)
        np.testing.assert_allclose(a_sq_deb, true_a_sq, atol=1e-10)
        np.testing.assert_allclose(b_sq_deb, true_b_sq, atol=1e-10)

    def test_debias_clips_negative_to_zero(self):
        """debias_params clips â² - Var[â] < 0 to exactly 0."""
        from sys_mapping.correction import debias_params

        a_hat = np.array([0.001])   # very small
        var_a = np.array([0.01])    # variance larger than a²  →  should clip
        a_sq_deb, _ = debias_params(a_hat, np.array([0.0]), var_a, np.array([0.0]))
        assert a_sq_deb[0] == 0.0, f"Expected 0 but got {a_sq_deb[0]}"


# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------


class TestUtilsAccuracy:
    def test_covariance_matrix_symmetry(self):
        """Template covariance matrix is symmetric to atol=1e-14."""
        from sys_mapping.utils import compute_covariance_matrix

        _, delta_t, _, _ = _make_data()
        cov = compute_covariance_matrix(delta_t)
        np.testing.assert_allclose(cov, cov.T, atol=1e-14)

    def test_covariance_matrix_positive_semidefinite(self):
        """All eigenvalues of the covariance matrix are ≥ -1e-12."""
        from sys_mapping.utils import compute_covariance_matrix

        _, delta_t, _, _ = _make_data()
        cov = compute_covariance_matrix(delta_t)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-12), f"Min eigenvalue {eigvals.min():.2e} < 0"

    def test_covariance_identity_for_orthogonal_templates(self):
        """Unit-variance orthogonal templates give covariance close to identity."""
        from sys_mapping.utils import compute_covariance_matrix

        n_pix, n_sys = 200_000, 4
        rng = np.random.default_rng(0)
        delta_t = rng.standard_normal((n_sys, n_pix))
        cov = compute_covariance_matrix(delta_t)
        np.testing.assert_allclose(cov, np.eye(n_sys), atol=0.01)

    def test_amplitude_bias_exact_formula(self):
        """compute_amplitude_bias returns exact formula values."""
        from sys_mapping.utils import compute_amplitude_bias

        cov = np.diag([1.0, 2.0, 0.5])
        a_sq = np.array([0.10, 0.20, 0.30])
        b_sq = np.array([0.05, 0.10, 0.15])
        add_bias, mult_factor = compute_amplitude_bias(a_sq, b_sq, cov)
        # add = 0.1*1 + 0.2*2 + 0.3*0.5 = 0.65
        np.testing.assert_allclose(add_bias, 0.65, rtol=1e-10)
        # mult = 1 + 0.05*1 + 0.10*2 + 0.15*0.5 = 1.325
        np.testing.assert_allclose(mult_factor, 1.325, rtol=1e-10)

    def test_covariance_matrix_formula(self):
        """C = delta_t @ delta_t.T / n_pix exactly."""
        from sys_mapping.utils import compute_covariance_matrix

        _, delta_t, _, _ = _make_data()
        cov = compute_covariance_matrix(delta_t)
        expected = delta_t @ delta_t.T / N_PIX
        np.testing.assert_allclose(cov, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# model_selection.py
# ---------------------------------------------------------------------------


class TestModelSelectionAccuracy:
    def test_lambda_lr_zero_when_models_equivalent(self):
        """λ_LR = 0 when the combined model uses b=0 (equivalent to additive)."""
        from sys_mapping.model_selection import likelihood_ratio_test
        from sys_mapping.contamination import pack_params

        delta_g, delta_t, a, _ = _make_data(n_pix=2000, n_sys=N_SYS)
        sigma = float(np.std(delta_g))
        b_zero = np.zeros(N_SYS)
        theta_add = pack_params(a, None, sigma, model="additive")
        # Combined with b=0 is identical to additive → same LL → λ_LR = 0
        theta_comb = pack_params(a, b_zero, sigma, model="combined")
        result = likelihood_ratio_test(delta_g, delta_t, theta_add, theta_comb,
                                       "additive", "combined")
        np.testing.assert_allclose(result.lambda_lr, 0.0, atol=1e-5,
                                   err_msg=f"λ_LR={result.lambda_lr:.4f} ≠ 0 for equivalent models")

    def test_dof_additive_vs_combined(self):
        """Degrees of freedom = n_sys for additive vs combined comparison."""
        from sys_mapping.model_selection import likelihood_ratio_test
        from sys_mapping.contamination import pack_params

        delta_g, delta_t, a, b = _make_data(n_pix=2000)
        sigma = float(np.std(delta_g))
        theta_add = pack_params(a, None, sigma, model="additive")
        theta_comb = pack_params(a, b, sigma, model="combined")
        result = likelihood_ratio_test(delta_g, delta_t, theta_add, theta_comb,
                                       "additive", "combined")
        assert result.n_dof == N_SYS, f"n_dof={result.n_dof} ≠ {N_SYS}"

    def test_p_value_in_unit_interval(self):
        """p-value is in [0, 1]."""
        from sys_mapping.model_selection import likelihood_ratio_test
        from sys_mapping.contamination import pack_params

        delta_g, delta_t, a, b = _make_data(n_pix=2000)
        sigma = float(np.std(delta_g))
        theta_add = pack_params(a, None, sigma, model="additive")
        theta_comb = pack_params(a, b, sigma, model="combined")
        result = likelihood_ratio_test(delta_g, delta_t, theta_add, theta_comb,
                                       "additive", "combined")
        assert 0.0 <= result.p_value <= 1.0, f"p_value={result.p_value} outside [0,1]"

    def test_large_contamination_detected(self):
        """LRT detects strong multiplicative contamination (lambda_LR >> chi2 critical)."""
        from sys_mapping.model_selection import likelihood_ratio_test
        from sys_mapping.contamination import apply_contamination, pack_params

        rng = np.random.default_rng(7)
        n_pix, n_sys = 5000, 2
        delta_t = rng.standard_normal((n_sys, n_pix))
        delta_t -= delta_t.mean(axis=1, keepdims=True)
        delta_t /= delta_t.std(axis=1, keepdims=True)
        delta_g_clean = rng.standard_normal(n_pix) * 0.1

        b_true = np.array([0.5, -0.4])
        a_true = np.zeros(n_sys)
        delta_g_obs = np.asarray(apply_contamination(
            jnp.asarray(delta_g_clean), jnp.asarray(delta_t),
            jnp.asarray(a_true), jnp.asarray(b_true),
        ))
        sigma = float(np.std(delta_g_obs))
        theta_add = pack_params(a_true, None, sigma, model="additive")
        theta_comb = pack_params(a_true, b_true, sigma, model="combined")
        result = likelihood_ratio_test(delta_g_obs, delta_t, theta_add, theta_comb,
                                       "additive", "combined")
        assert result.lambda_lr > 10, f"λ_LR={result.lambda_lr:.2f} too small for strong contamination"


# ---------------------------------------------------------------------------
# bootstrap.py
# ---------------------------------------------------------------------------


class TestBootstrapAccuracy:
    def test_variance_nonnegative(self):
        """Block bootstrap variance estimate is ≥ 0."""
        from sys_mapping.bootstrap import block_bootstrap_variance

        nside = 16
        n_pix_full = 12 * nside ** 2
        rng = np.random.default_rng(0)
        good = np.ones(n_pix_full, dtype=bool)
        dg = rng.standard_normal(n_pix_full) * 0.1
        dt = rng.standard_normal((2, n_pix_full))

        var = block_bootstrap_variance(dg, dt, good, nside,
                                       lambda dg_b, dt_b: np.array([np.mean(dg_b)]),
                                       n_bootstrap=30, n_patches=6, seed=1)
        assert np.all(var >= 0), f"Bootstrap variance {var} < 0"

    def test_variance_decreases_with_more_data(self):
        """Bootstrap variance of mean decreases for larger n_pix (converges to 0)."""
        from sys_mapping.bootstrap import block_bootstrap_variance

        results = []
        for nside in (8, 16):
            n_pix_full = 12 * nside ** 2
            rng = np.random.default_rng(42)
            good = np.ones(n_pix_full, dtype=bool)
            dg = rng.standard_normal(n_pix_full) * 0.1
            dt = rng.standard_normal((2, n_pix_full))
            var = block_bootstrap_variance(dg, dt, good, nside,
                                           lambda dg_b, dt_b: np.array([np.mean(dg_b)]),
                                           n_bootstrap=50, n_patches=6, seed=0)
            results.append(float(var[0]))
        assert results[1] < results[0], \
            f"Bootstrap variance did not decrease: {results[0]:.4e} → {results[1]:.4e}"

    def test_patch_assignment_covers_all_pixels(self):
        """_assign_patches assigns a label to every unmasked pixel."""
        from sys_mapping.bootstrap import _assign_patches

        nside = 16
        n_pix = 12 * nside ** 2
        good = np.ones(n_pix, dtype=bool)
        good[:10] = False
        patch_ids = _assign_patches(good, nside, n_patches=8)
        n_good = int(good.sum())
        assert len(patch_ids) == n_good, \
            f"patch_ids length {len(patch_ids)} ≠ n_good={n_good}"
