"""Tests for correction.py: debiasing, template rotation, two-point correction."""

import warnings

import numpy as np
import pytest
import jax.numpy as jnp

import sys_mapping as sm

from sys_mapping.correction import (
    debias_params,
    rotate_templates,
    transform_params_from_rotated,
    correct_two_point_function,
)


class TestDebiasParams:
    def test_zero_variance_no_change(self):
        a = np.array([0.1, -0.2, 0.3])
        b = np.array([0.05, 0.1, -0.1])
        var_a = np.zeros(3)
        var_b = np.zeros(3)
        a_sq, b_sq = debias_params(a, b, var_a, var_b)
        np.testing.assert_allclose(a_sq, a**2, atol=1e-12)
        np.testing.assert_allclose(b_sq, b**2, atol=1e-12)

    def test_large_variance_clips_to_zero(self):
        a = np.array([0.1])
        b = np.array([0.1])
        var_a = np.array([1.0])  # larger than a^2
        var_b = np.array([1.0])
        a_sq, b_sq = debias_params(a, b, var_a, var_b)
        assert a_sq[0] == 0.0
        assert b_sq[0] == 0.0

    def test_partial_debiasing(self):
        a = np.array([0.4])
        b = np.array([0.5])
        var_a = np.array([0.1])  # a^2 = 0.16, debiased = 0.06
        var_b = np.array([0.2])  # b^2 = 0.25, debiased = 0.05
        a_sq, b_sq = debias_params(a, b, var_a, var_b)
        np.testing.assert_allclose(a_sq[0], 0.16 - 0.1, atol=1e-12)
        np.testing.assert_allclose(b_sq[0], 0.25 - 0.2, atol=1e-12)

    def test_non_negative_output(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 0.1, 10)
        var_a = rng.uniform(0, 0.05, 10)
        a_sq, _ = debias_params(a, a, var_a, var_a)
        assert np.all(a_sq >= 0)


class TestRotateTemplates:
    def test_orthogonality_of_rotated_templates(self):
        rng = np.random.default_rng(1)
        n_sys, n_pix = 4, 500
        delta_t = rng.standard_normal((n_sys, n_pix))
        delta_t_rot, R, eigvals = rotate_templates(delta_t)

        # Covariance of rotated templates should be diagonal
        cov_rot = delta_t_rot @ delta_t_rot.T / n_pix
        off_diag = cov_rot - np.diag(np.diag(cov_rot))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-10)

    def test_rotation_matrix_is_orthogonal(self):
        rng = np.random.default_rng(2)
        delta_t = rng.standard_normal((3, 300))
        _, R, _ = rotate_templates(delta_t)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_variance_preserved(self):
        """Total variance is preserved under orthogonal rotation."""
        rng = np.random.default_rng(3)
        delta_t = rng.standard_normal((3, 500))
        delta_t_rot, _, eigvals = rotate_templates(delta_t)
        var_orig = np.sum(np.var(delta_t, axis=1))
        var_rot = np.sum(np.var(delta_t_rot, axis=1))
        np.testing.assert_allclose(var_rot, var_orig, rtol=1e-10)

    def test_eigenvalues_descending(self):
        rng = np.random.default_rng(4)
        delta_t = rng.standard_normal((5, 300))
        _, _, eigvals = rotate_templates(delta_t)
        assert np.all(np.diff(eigvals) <= 0), "Eigenvalues not in descending order"

    def test_roundtrip_params(self):
        rng = np.random.default_rng(5)
        n_sys, n_pix = 3, 400
        delta_t = rng.standard_normal((n_sys, n_pix))
        a_orig = rng.standard_normal(n_sys) * 0.1
        b_orig = rng.standard_normal(n_sys) * 0.05

        delta_t_rot, R, _ = rotate_templates(delta_t)
        a_rot = R @ a_orig
        b_rot = R @ b_orig
        a_back, b_back = transform_params_from_rotated(a_rot, b_rot, R)

        np.testing.assert_allclose(a_back, a_orig, atol=1e-12)
        np.testing.assert_allclose(b_back, b_orig, atol=1e-12)


class TestCorrectTwoPoint:
    def test_zero_params_identity(self):
        n_bins, n_sys = 8, 3
        rng = np.random.default_rng(0)
        w_obs = rng.standard_normal(n_bins)
        a_hat = np.zeros(n_sys)
        b_hat = np.zeros(n_sys)
        var_a = np.zeros(n_sys)
        var_b = np.zeros(n_sys)
        tcorr = np.ones((n_sys, n_bins))

        w_corr = correct_two_point_function(w_obs, a_hat, b_hat, var_a, var_b, tcorr)
        np.testing.assert_allclose(w_corr, w_obs, atol=1e-12)

    def test_additive_subtracted(self):
        n_bins, n_sys = 5, 1
        w_obs = np.ones(n_bins)
        a_hat = np.array([0.5])   # a^2 = 0.25
        b_hat = np.zeros(1)
        var_a = np.zeros(1)
        var_b = np.zeros(1)
        tcorr = np.ones((1, n_bins))

        w_corr = correct_two_point_function(w_obs, a_hat, b_hat, var_a, var_b, tcorr)
        np.testing.assert_allclose(w_corr, np.full(n_bins, 1 - 0.25), atol=1e-12)

    def test_debiasing_affects_output(self):
        """With variance = a^2, debiased a_sq = 0, so no correction applied."""
        n_bins, n_sys = 5, 1
        a_hat = np.array([0.5])
        b_hat = np.zeros(1)
        var_a = np.array([0.25])  # equals a^2 → debias to 0
        var_b = np.zeros(1)
        w_obs = np.ones(n_bins)
        tcorr = np.ones((1, n_bins))

        w_corr = correct_two_point_function(w_obs, a_hat, b_hat, var_a, var_b, tcorr)
        np.testing.assert_allclose(w_corr, w_obs, atol=1e-12)


class TestCorrectTwoPointCovariance:
    def test_point_estimate_unchanged_and_shapes(self):
        n_bins, n_sys = 6, 3
        rng = np.random.default_rng(1)
        w_obs = rng.standard_normal(n_bins) * 0.01
        a_hat = np.array([0.05, -0.03, 0.02])
        b_hat = np.array([0.04, 0.01, -0.02])
        var_a = np.full(n_sys, 5e-4)
        var_b = np.full(n_sys, 3e-4)
        tcorr = np.abs(rng.standard_normal((n_sys, n_bins))) * 1e-3

        w_ref = correct_two_point_function(w_obs, a_hat, b_hat, var_a, var_b, tcorr)
        w_corr, cov = correct_two_point_function(
            w_obs, a_hat, b_hat, var_a, var_b, tcorr, return_cov=True, random_state=0)
        np.testing.assert_allclose(w_corr, w_ref, atol=1e-12)     # point estimate identical
        assert cov.shape == (n_bins, n_bins)
        np.testing.assert_allclose(cov, cov.T, atol=1e-14)
        assert np.all(np.linalg.eigvalsh(cov) >= -1e-12)

    def test_zero_uncertainty_gives_zero_cov(self):
        n_bins, n_sys = 5, 2
        w_obs = np.ones(n_bins)
        a_hat = np.array([0.1, -0.2]); b_hat = np.zeros(n_sys)
        var_a = np.zeros(n_sys); var_b = np.zeros(n_sys)
        tcorr = np.ones((n_sys, n_bins))
        _, cov = correct_two_point_function(
            w_obs, a_hat, b_hat, var_a, var_b, tcorr, return_cov=True, random_state=0)
        np.testing.assert_allclose(cov, 0.0, atol=1e-12)          # no inputs vary → no output var

    def test_w_obs_covariance_passes_through_at_null(self):
        """With a=b=0 (no correction) the corrected cov equals the input w_obs covariance."""
        n_bins, n_sys = 6, 2
        rng = np.random.default_rng(2)
        w_obs = rng.standard_normal(n_bins) * 0.01
        a_hat = np.zeros(n_sys); b_hat = np.zeros(n_sys)
        var_a = np.zeros(n_sys); var_b = np.zeros(n_sys)
        tcorr = np.abs(rng.standard_normal((n_sys, n_bins))) * 1e-3
        L = rng.standard_normal((n_bins, n_bins)) * 0.02
        cov_w_obs = L @ L.T
        _, cov = correct_two_point_function(
            w_obs, a_hat, b_hat, var_a, var_b, tcorr, return_cov=True,
            cov_w_obs=cov_w_obs, n_mc=40000, random_state=0)
        # w_corr_k = w_obs_k exactly here → sample cov ≈ cov_w_obs (MC-limited)
        np.testing.assert_allclose(cov, cov_w_obs, rtol=0.1, atol=1e-5)

    def test_reproducible(self):
        n_bins, n_sys = 5, 2
        rng = np.random.default_rng(3)
        args = (rng.standard_normal(n_bins) * 0.01, np.array([0.05, -0.02]),
                np.array([0.03, 0.01]), np.full(n_sys, 4e-4), np.full(n_sys, 2e-4),
                np.abs(rng.standard_normal((n_sys, n_bins))) * 1e-3)
        _, c1 = correct_two_point_function(*args, return_cov=True, random_state=7)
        _, c2 = correct_two_point_function(*args, return_cov=True, random_state=7)
        np.testing.assert_allclose(c1, c2, atol=1e-15)


class TestCorrectPowerSpectrumHarmonic:
    def test_output_shape(self):
        from sys_mapping.correction import correct_power_spectrum_harmonic

        n_ell, n_sys = 20, 2
        ell = np.arange(n_ell, dtype=float)
        cl_obs = np.ones(n_ell) * 1e-4
        alpha = np.array([0.1, -0.05])
        t_cls = np.ones((n_sys, n_ell)) * 5e-5
        cl_corr = correct_power_spectrum_harmonic(cl_obs, ell, n_sys, alpha, t_cls)
        assert cl_corr.shape == (n_ell,)

    def test_zero_alpha_applies_harmonic_bias_only(self):
        from sys_mapping.correction import correct_power_spectrum_harmonic
        from sys_mapping.power_spectrum import harmonic_bias

        n_ell, n_sys = 10, 2
        ell = np.arange(n_ell, dtype=float)
        cl_obs = np.ones(n_ell) * 1e-4
        alpha = np.zeros(n_sys)
        t_cls = np.ones((n_sys, n_ell)) * 5e-5
        cl_corr = correct_power_spectrum_harmonic(cl_obs, ell, n_sys, alpha, t_cls)
        np.testing.assert_allclose(cl_corr, cl_obs - harmonic_bias(n_sys, ell), rtol=1e-10)


class TestHarmonicMatchesConfiguration:
    """The harmonic and configuration corrections must remove the same quantity.

    A power spectrum is quadratic in the field, so a contaminant sum_i a_i t_i
    contributes sum_i a_i^2 C_l^{t_i} -- the same quadratic dependence the
    configuration-space correction applies as a_tilde_i^2 xi_i.  Subtracting a
    term *linear* in the amplitude, as the harmonic path used to, makes the two
    paths correct different quantities, and nothing tested that they agreed.
    """

    @staticmethod
    def _cl_to_wtheta(cl, theta_rad):
        """Legendre sum: w(theta) = sum_l (2l+1)/(4pi) C_l P_l(cos theta)."""
        from numpy.polynomial.legendre import legval
        ell = np.arange(len(cl))
        coef = (2 * ell + 1) / (4 * np.pi) * cl
        return legval(np.cos(theta_rad), coef)

    def test_amplitude_enters_squared(self):
        """Doubling the amplitude must quadruple what is subtracted."""
        import healpy as hp
        from sys_mapping.power_spectrum import subtract_template_cl

        nside, lmax = 16, 24
        rng = np.random.default_rng(0)
        npix = hp.nside2npix(nside)
        t = rng.standard_normal((1, npix))
        mask = np.ones(npix, dtype=bool)
        cl0 = np.ones(lmax + 1)

        d1 = cl0 - subtract_template_cl(cl0, t, mask, np.array([0.1]), lmax=lmax)
        d2 = cl0 - subtract_template_cl(cl0, t, mask, np.array([0.2]), lmax=lmax)
        np.testing.assert_allclose(d2, 4.0 * d1, rtol=1e-10)

    def test_harmonic_and_configuration_agree(self):
        """Round trip: both paths, on one field, must land on the same w(theta)."""
        import healpy as hp
        from sys_mapping.power_spectrum import subtract_template_cl

        nside, lmax = 32, 48
        rng = np.random.default_rng(1)
        npix = hp.nside2npix(nside)
        t = rng.standard_normal((1, npix))
        t = (t - t.mean(1, keepdims=True)) / t.std(1, keepdims=True)
        mask = np.ones(npix, dtype=bool)
        a = np.array([0.25])

        cl_obs = np.ones(lmax + 1) * 1e-3
        cl_corr = subtract_template_cl(cl_obs, t, mask, a, lmax=lmax)

        theta = np.radians(np.linspace(1.0, 20.0, 8))
        w_harm = self._cl_to_wtheta(cl_corr, theta)

        # Configuration space: subtract a^2 * xi_t, with xi_t the Legendre
        # transform of the same template spectrum.  var_a = 0, so a_sq = a^2.
        cl_t = hp.anafast(t[0] * mask.astype(float), lmax=lmax, use_pixel_weights=True)
        xi_t = self._cl_to_wtheta(cl_t, theta)
        w_obs = self._cl_to_wtheta(cl_obs, theta)
        w_conf = np.asarray(sm.compute_two_point_correction(
            jnp.asarray(w_obs), jnp.asarray(a**2), jnp.asarray(np.zeros(1)),
            jnp.asarray(xi_t[np.newaxis, :])))

        np.testing.assert_allclose(w_harm, w_conf, rtol=1e-8, atol=1e-12)


class TestEMPModeCount:
    def test_bias_scales_with_templates_not_multipoles(self):
        """harmonic_bias' first argument is a template count, not a mode count.

        The extended branch used to pass the number of multipoles that survived
        the cut, inflating the subtracted bias by their ratio -- 190 to 11 at
        NSIDE 64.
        """
        from sys_mapping.power_spectrum import mode_projection_bias

        ell = np.arange(2, 60)
        pseudo_cl = np.full(ell.size, 1e-4)
        coupling = np.eye(ell.size)

        _, bias_a = mode_projection_bias(pseudo_cl, coupling, ell, n_templates=4,
                                         mode="extended", threshold=1e-12)
        _, bias_b = mode_projection_bias(pseudo_cl, coupling, ell, n_templates=8,
                                         mode="extended", threshold=1e-12)
        nz = bias_a != 0
        assert nz.any(), "threshold should retain some multipoles"
        # Doubling the templates doubles the bias; the multipole count is unchanged.
        np.testing.assert_allclose(bias_b[nz], 2.0 * bias_a[nz], rtol=1e-10)


class TestCrossTemplateTerms:
    """The correction keeps xi_ij, not only xi_ii."""

    def test_matrix_debias_reduces_to_scalar_for_one_template(self):
        from sys_mapping.correction import debias_params_matrix

        a = np.array([0.1])
        var = np.array([0.002])
        A, _ = debias_params_matrix(a, np.zeros(1), np.diag(var), np.zeros((1, 1)))
        a_sq, _ = debias_params(a, np.zeros(1), var, np.zeros(1))
        assert A[0, 0] == pytest.approx(a_sq[0], rel=1e-12)

    def test_matrix_debias_is_unbiased_and_psd_projection_is_not(self):
        """The unbiased estimator is not PSD, and projecting it undoes the subtraction."""
        from sys_mapping.correction import debias_params_matrix

        rng = np.random.default_rng(0)
        n, n_draw = 5, 4000
        a_true = rng.normal(0, 0.03, n)
        cov = np.diag(rng.uniform(0.5, 1.5, n) * 4e-4)
        chol = np.linalg.cholesky(cov)
        plain = np.zeros((n, n))
        projected = np.zeros((n, n))
        for _ in range(n_draw):
            a_hat = a_true + chol @ rng.standard_normal(n)
            zero = np.zeros((n, n))
            plain += debias_params_matrix(a_hat, np.zeros(n), cov, zero)[0]
            projected += debias_params_matrix(a_hat, np.zeros(n), cov, zero,
                                              project_psd=True)[0]
        truth = np.outer(a_true, a_true)
        err_plain = np.abs(plain / n_draw - truth).mean()
        err_projected = np.abs(projected / n_draw - truth).mean()
        err_none = np.abs(cov).mean()
        assert err_plain < 0.2 * err_none          # the subtraction works
        assert err_projected > 5 * err_plain       # the projection undoes most of it
        A, _ = debias_params_matrix(a_true, np.zeros(n), cov, np.zeros((n, n)))
        np.testing.assert_allclose(A, A.T, atol=1e-14)
        assert np.linalg.eigvalsh(A).min() < 0     # rank-one minus positive definite

    def test_matrix_debias_projects_when_asked(self):
        from sys_mapping.correction import debias_params_matrix

        rng = np.random.default_rng(0)
        n = 5
        a = rng.normal(0, 0.05, n)
        C = rng.standard_normal((n, n))
        C = C @ C.T * 1e-3
        A, _ = debias_params_matrix(a, np.zeros(n), C, np.zeros((n, n)), project_psd=True)
        assert np.linalg.eigvalsh(A).min() >= -1e-12
        np.testing.assert_allclose(A, A.T, atol=1e-14)

    def test_full_form_reduces_to_auto_form_on_a_diagonal_matrix(self):
        rng = np.random.default_rng(1)
        n, nb = 4, 6
        xi = rng.standard_normal((n, n, nb)) * 1e-3
        xi = 0.5 * (xi + xi.transpose(1, 0, 2))
        autos = np.array([xi[i, i] for i in range(n)])
        A = np.diag(rng.uniform(0, 1e-3, n))
        w = np.full(nb, 1e-2)

        full = np.asarray(sm.compute_two_point_correction(
            jnp.asarray(w), jnp.asarray(A), jnp.asarray(np.zeros_like(A)),
            jnp.asarray(xi)))
        auto = np.asarray(sm.compute_two_point_correction(
            jnp.asarray(w), jnp.asarray(np.diag(A)), jnp.asarray(np.zeros(n)),
            jnp.asarray(autos)))
        np.testing.assert_allclose(full, auto, rtol=1e-12)

    def test_cross_terms_change_the_answer(self):
        """If they did not, dropping them would have been free."""
        rng = np.random.default_rng(2)
        n, nb = 4, 6
        xi = rng.standard_normal((n, n, nb)) * 1e-3
        xi = 0.5 * (xi + xi.transpose(1, 0, 2))
        autos = np.array([xi[i, i] for i in range(n)])
        a = rng.normal(0, 0.05, n)
        kw = dict(var_a=np.full(n, 1e-4), var_b=np.zeros(n))
        w = np.full(nb, 1e-2)
        full = correct_two_point_function(w, a, np.zeros(n),
                                          template_correlations=xi, **kw)
        auto = correct_two_point_function(w, a, np.zeros(n),
                                          template_correlations=autos, **kw)
        assert not np.allclose(full, auto)

    @pytest.mark.parametrize("bad", ["matrix_amp_auto_corr", "vector_amp_full_corr"])
    def test_rank_mismatch_is_named_not_broadcast(self, bad):
        n, nb = 3, 4
        xi3 = np.zeros((n, n, nb))
        xi2 = np.zeros((n, nb))
        A = np.zeros((n, n))
        v = np.zeros(n)
        amp, corr = ((A, xi2) if bad == "matrix_amp_auto_corr" else (v, xi3))
        with pytest.raises(ValueError, match="ranks disagree"):
            sm.compute_two_point_correction(jnp.asarray(np.zeros(nb)),
                                            jnp.asarray(amp), jnp.asarray(amp),
                                            jnp.asarray(corr))


class TestOvercorrectionGuard:
    """A corrected w(theta) that has gone negative must not be silent.

    ``sum_i a_i^2 xi_i(theta)`` is a sum of squares against auto-correlations,
    so nothing bounds it by ``w_obs``.  Where the galaxy signal has decayed but
    the survey-property maps are still coherent, the subtracted term can exceed
    the measurement and leave a negative correlation function.
    """

    @staticmethod
    def _inputs():
        w_obs = np.array([1.0, 0.5, 0.2, 0.05, 0.01])
        a_hat = np.array([0.2, 0.0])
        b_hat = np.zeros(2)
        zeros = np.zeros(2)
        # xi rises with theta, which is what a template basis restricted to a
        # footprint does at separations the galaxy signal has already left.
        xi = np.array([[0.0, 0.0, 0.0, 0.5, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        return w_obs, a_hat, b_hat, zeros, xi

    def test_negative_corrected_value_warns(self):
        w_obs, a_hat, b_hat, zeros, xi = self._inputs()
        with pytest.warns(RuntimeWarning, match="overshoots the signal"):
            w_corr = sm.correct_two_point_function(w_obs, a_hat, b_hat, zeros, zeros, xi)
        assert np.any(w_corr < 0.0)

    def test_a_benign_correction_is_silent(self):
        w_obs, a_hat, b_hat, zeros, xi = self._inputs()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            w_corr = sm.correct_two_point_function(
                w_obs, 0.01 * a_hat, b_hat, zeros, zeros, xi)
        assert np.all(w_corr > 0.0)

    def test_warning_reports_the_worst_bin(self):
        w_obs, a_hat, b_hat, zeros, xi = self._inputs()
        with pytest.warns(RuntimeWarning) as rec:
            sm.correct_two_point_function(w_obs, a_hat, b_hat, zeros, zeros, xi)
        message = str(rec[0].message)
        assert "1 of 5 bins" in message
        assert "w_corr/w_obs" in message
