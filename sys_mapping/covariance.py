"""Low-rank + diagonal pixel-covariance operator for the correlated-noise likelihood.

The Gaussian log-likelihood in :mod:`sys_mapping.likelihood` (Eq. 17) treats the clean
overdensity as spatially *independent* (covariance :math:`\\sigma^2 I`).  That is wrong when the
clean field is a clustered galaxy overdensity: it is spatially correlated, and because systematic
templates are smooth large-scale maps the correlated field projects onto them with far more
variance than the white-noise assumption predicts.  The pixel likelihood then reports posterior
error bars that are too tight (see the GLASS slow-methods validation).

This module supplies the missing piece: a fixed **unit-diagonal correlation** matrix

.. math::

    R = D + U U^\\top

(``D`` diagonal, ``U`` the leading :math:`M` spatial modes of the clustering covariance), so the
likelihood can use the generalized (GLS) quadratic form :math:`r^\\top R^{-1} r` in place of
:math:`\\sum_j r_j^2`.  ``R`` is normalized to unit diagonal, so :math:`\\sigma` keeps its meaning as
the per-pixel scale and :math:`C = \\sigma^2 R`.

Everything is applied through the **Woodbury identity** so we never form or invert the
:math:`N \\times N` matrix (:math:`N` ~ :math:`10^5` pixels):

.. math::

    R^{-1} v &= D^{-1} v - D^{-1} U \\, (I_M + U^\\top D^{-1} U)^{-1} \\, U^\\top D^{-1} v \\\\
    \\ln|R|  &= \\sum_j \\ln D_{jj} + \\ln\\big|I_M + U^\\top D^{-1} U\\big|

with an :math:`M \\times M` inner solve (:math:`M` = number of modes, ~200).  The operator is a
plain container of JAX arrays, so ``apply`` is fully differentiable and JIT-friendly and can be
captured in the log-likelihood closure.

This is the mock-mode-projection / template-covariance approach of Weaverdyck & Huterer 2021
(MNRAS 503, 5061): the modes ``U`` are estimated empirically from an ensemble of clean mock fields,
which is self-consistent with the mocks used to generate the sims.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class LowRankPrecision:
    """Precision operator ``R^{-1}`` for ``R = D + U U^T`` (unit-diagonal correlation).

    Built by :func:`build_lowrank_precision`; do not construct directly.  Holds the precomputed
    Woodbury pieces as JAX arrays and exposes:

    * :meth:`apply` — ``R^{-1} v`` for a vector (or column-stacked matrix) ``v``.
    * :meth:`quad` — the quadratic form ``v^T R^{-1} v``.

    The stored Woodbury pieces are ``d_inv`` (inverse diagonal ``1/D``, shape ``(n_pix,)``), ``u``
    (normalized mode matrix ``U``, ``(n_pix, n_modes)``), ``m_inv`` (``(I_M + U^T D^{-1} U)^{-1}``,
    ``(n_modes, n_modes)``), the scalar ``logdet`` (``ln|R|``), and the sizes ``n_pix`` / ``n_modes``.
    """

    d_inv: Array
    u: Array
    m_inv: Array
    logdet: float
    n_pix: int
    n_modes: int

    def apply(self, v: Array) -> Array:
        """Return ``R^{-1} v``.  ``v`` is ``(n_pix,)`` or ``(n_pix, k)``."""
        v = jnp.asarray(v)
        x = self.d_inv[:, None] * v if v.ndim == 2 else self.d_inv * v
        y = self.u.T @ x                       # (n_modes,) or (n_modes, k)
        z = self.m_inv @ y
        corr = self.u @ z                      # (n_pix,) or (n_pix, k)
        corr = self.d_inv[:, None] * corr if v.ndim == 2 else self.d_inv * corr
        return x - corr

    def quad(self, v: Array) -> Array:
        """Return the scalar quadratic form ``v^T R^{-1} v`` (``v`` is ``(n_pix,)``)."""
        return jnp.sum(v * self.apply(v))


def build_lowrank_precision(
    mock_fields: np.ndarray,
    shot_var: np.ndarray | float,
    *,
    n_modes: int | None = None,
    dtype=jnp.float64,
) -> LowRankPrecision:
    """Build a :class:`LowRankPrecision` for ``R = D + U U^T`` from mock clean fields.

    The clustering covariance is estimated empirically as the sample covariance of ``mock_fields``
    (an ensemble of *clean* overdensity realizations sampled on the same pixels the fit uses),
    added to a diagonal shot-noise term ``shot_var``, and normalized to unit diagonal so that
    :math:`\\sigma` in the likelihood remains the per-pixel scale.

    Parameters
    ----------
    mock_fields : (n_pix, n_mock)
        Column-stacked clean overdensity realizations on the fit pixels.  The mean over the
        ensemble is removed internally.  ``n_mock`` becomes the low-rank dimension ``M`` (optionally
        reduced by ``n_modes`` via an SVD truncation).
    shot_var : float or (n_pix,)
        Diagonal shot-noise variance to add to the clustering covariance.  A positive floor keeps
        ``D`` well-conditioned; pass e.g. the mean per-pixel Poisson variance.
    n_modes : int, optional
        If given and smaller than ``n_mock``, keep only the top ``n_modes`` SVD modes of the
        (mean-subtracted) ensemble.  Cheaper and denoises the tail.  Default keeps all columns.
    dtype :
        JAX dtype for the stored arrays (default float64).

    Returns
    -------
    LowRankPrecision

    Notes
    -----
    Let ``Delta`` be the mean-subtracted ensemble (``n_pix, m``).  The raw covariance is
    ``C_raw = Delta Delta^T / (m - 1) + diag(shot_var)``, i.e. ``U_raw = Delta / sqrt(m - 1)``.
    Writing ``d = diag(C_raw)`` and ``s = 1/sqrt(d)``, the unit-diagonal correlation is
    ``R = diag(shot_var) s^2 + (s ⊙ U_raw)(s ⊙ U_raw)^T`` which has ``diag(R) = 1`` by construction.
    """
    fields = np.asarray(mock_fields, dtype=np.float64)
    if fields.ndim != 2:
        raise ValueError(f"mock_fields must be 2-D (n_pix, n_mock); got shape {fields.shape}")
    n_pix, m = fields.shape
    if m < 2:
        raise ValueError(f"need >= 2 mock realizations to estimate a covariance; got {m}")

    delta = fields - fields.mean(axis=1, keepdims=True)
    u_raw = delta / np.sqrt(m - 1.0)                       # (n_pix, m): C_clust = u_raw @ u_raw.T

    if n_modes is not None and n_modes < m:
        # Truncated SVD of the ensemble: keep the leading spatial modes.
        left, sv, _ = np.linalg.svd(u_raw, full_matrices=False)
        u_raw = left[:, :n_modes] * sv[:n_modes]

    shot = np.broadcast_to(np.asarray(shot_var, dtype=np.float64), (n_pix,)).copy()
    if np.any(shot <= 0):
        raise ValueError("shot_var must be strictly positive (needed to keep D invertible)")

    # Normalize to unit diagonal: d = shot + rowsum(u_raw^2); s = 1/sqrt(d).
    d_raw = shot + np.einsum("ij,ij->i", u_raw, u_raw)
    s = 1.0 / np.sqrt(d_raw)
    d = shot * s**2                                        # normalized diagonal of R
    u = u_raw * s[:, None]                                 # normalized modes

    # Woodbury pieces.
    d_inv = 1.0 / d
    n_mode = u.shape[1]
    w = np.eye(n_mode) + (u.T * d_inv) @ u                 # I_M + U^T D^{-1} U
    sign, logdet_w = np.linalg.slogdet(w)
    if sign <= 0:
        raise np.linalg.LinAlgError("I + U^T D^{-1} U is not positive definite")
    logdet = float(np.sum(np.log(d)) + logdet_w)
    m_inv = np.linalg.inv(w)

    return LowRankPrecision(
        d_inv=jnp.asarray(d_inv, dtype=dtype),
        u=jnp.asarray(u, dtype=dtype),
        m_inv=jnp.asarray(m_inv, dtype=dtype),
        logdet=logdet,
        n_pix=int(n_pix),
        n_modes=int(n_mode),
    )


def mock_sandwich_covariance(
    delta_t: np.ndarray,
    mock_fields: np.ndarray,
) -> np.ndarray:
    """Additive-parameter covariance under correlated noise, from mock overdensity fields.

    For the linear template regression :math:`\\delta_g = T^\\top a + \\varepsilon` the OLS
    estimator :math:`\\hat a = (T T^\\top)^{-1} T \\delta_g` has covariance

    .. math::

        \\mathrm{Cov}(\\hat a) = (T T^\\top)^{-1}\\, (T C T^\\top)\\, (T T^\\top)^{-1},
        \\qquad C = \\mathrm{Cov}(\\varepsilon),

    which the white (:math:`\\sigma^2 I`) likelihood **underestimates** when the noise
    :math:`\\varepsilon` (here the reconstructed clean overdensity) is spatially correlated.
    The template-projected covariance :math:`T C T^\\top` is estimated directly as the sample
    covariance of the projected mock fields :math:`T \\varepsilon_k`, so no :math:`N \\times N`
    matrix is ever formed and the result is exact for the linear estimator (unlike the low-rank
    pixel precision, whose empirical modes need not span the template directions).

    This is the parameter-level realization of the correlated-noise (GLS) idea and is what
    calibrates the additive error bars for the GLASS slow-methods validation.  Use an ensemble of
    **uncontaminated** mock reconstructions (the reconstruction noise ``epsilon``) sampled on the
    same pixels as the fit.

    Parameters
    ----------
    delta_t : ``(n_sys, n_pix)``
        Template maps on the fit pixels (the same basis passed to the estimator).
    mock_fields : (n_mock, n_pix)
        Ensemble of uncontaminated overdensity realizations on those pixels.  The ensemble mean is
        removed internally.  ``n_mock`` should comfortably exceed ``n_sys``.

    Returns
    -------
    cov : ``(n_sys, n_sys)``  additive-parameter covariance ``(T T^T)^{-1} (T C T^T) (T T^T)^{-1}``.

    Notes
    -----
    Cost is one :math:`n_{\\rm sys} \\times n_{\\rm pix}` projection plus small dense inverses; the
    field ensemble is the only expensive input and is reused across all fits on the same basis.
    """
    T = np.asarray(delta_t, dtype=np.float64)
    fields = np.asarray(mock_fields, dtype=np.float64)
    if T.ndim != 2:
        raise ValueError(f"delta_t must be 2-D (n_sys, n_pix); got shape {T.shape}")
    if fields.ndim != 2:
        raise ValueError(f"mock_fields must be 2-D (n_mock, n_pix); got shape {fields.shape}")
    if fields.shape[1] != T.shape[1]:
        raise ValueError(
            f"pixel dimension mismatch: delta_t has {T.shape[1]}, mock_fields has {fields.shape[1]}"
        )
    n_mock = fields.shape[0]
    if n_mock < 2:
        raise ValueError(f"need >= 2 mock realizations; got {n_mock}")

    Minv = np.linalg.inv(T @ T.T)
    proj = T @ (fields - fields.mean(axis=0)).T      # (n_sys, n_mock) = T eps_k
    c_proj = np.cov(proj, ddof=1)                    # (n_sys, n_sys) = T C T^T
    c_proj = np.atleast_2d(c_proj)
    return Minv @ c_proj @ Minv


def sample_covariance(samples: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance of an ensemble of estimator realizations.

    For ``n_real`` independent realizations of a length-``n_bins`` estimator (e.g. :math:`w(\\theta)`
    measured on ``n_real`` mock skies), returns the ``(n_bins, n_bins)`` sample covariance with the
    ``ddof=1`` (Bessel) normalization, so the covariance estimate itself is unbiased.

    This is the **mock-ensemble** route to the :math:`w(\\theta)` covariance (the GLASS-mock primary
    in the Euclid validation): build ``samples`` by measuring the same estimator on an ensemble of
    *uncontaminated* mock realizations of the survey footprint.  When the covariance is to be
    *inverted* (for a :math:`\\chi^2` / GLS fit) multiply the inverse by :func:`hartlap_factor` to
    debias the precision.  For the parameter-level (template amplitude) analogue see
    :func:`mock_sandwich_covariance`.

    Parameters
    ----------
    samples : (n_real, n_bins)
        Ensemble of estimator vectors; ``n_real`` realizations along axis 0.  ``n_real`` should
        comfortably exceed ``n_bins`` for the covariance to be well conditioned.

    Returns
    -------
    cov : (n_bins, n_bins) unbiased sample covariance.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"samples must be 2-D (n_real, n_bins); got shape {x.shape}")
    if x.shape[0] < 2:
        raise ValueError(f"need >= 2 realizations; got {x.shape[0]}")
    return np.atleast_2d(np.cov(x, rowvar=False, ddof=1))


def hartlap_factor(n_real: int, n_bins: int) -> float:
    """Hartlap (2007) debias factor for an *inverted* sample covariance.

    A precision matrix formed by inverting a sample covariance from ``n_real`` realizations is
    biased high; the unbiased precision is ``hartlap_factor(n_real, n_bins) * inv(cov)`` with

    .. math::

        \\text{factor} = \\frac{n_\\text{real} - n_\\text{bins} - 2}{n_\\text{real} - 1}.

    This requires ``n_real > n_bins + 2`` for a positive factor (and ``n_real > n_bins`` merely
    for the sample covariance to be invertible) — the concrete reason the GLASS-mock ensemble
    must carry comfortably more realizations than there are :math:`w(\\theta)` bins.  Pairs with
    :func:`sample_covariance`.

    Parameters
    ----------
    n_real : int  number of ensemble realizations.
    n_bins : int  length of the estimator vector (covariance dimension).

    Returns
    -------
    factor : float  Hartlap correction in ``(0, 1)``.
    """
    if n_real <= n_bins + 2:
        raise ValueError(
            f"Hartlap factor needs n_real > n_bins + 2; got n_real={n_real}, n_bins={n_bins}"
        )
    return (n_real - n_bins - 2) / (n_real - 1)


def build_harmonic_precision(*args, **kwargs):
    """Full-rank correlated-noise precision from a fiducial angular power spectrum (NOT IMPLEMENTED).

    Follow-up path (see the GLASS slow-methods validation): the empirical low-rank
    :func:`build_lowrank_precision` does **not** calibrate the pixel likelihood when the fixed
    systematic templates lie outside the span of a tractable mock ensemble, because :math:`R^{-1}`
    then acts as the identity in exactly the template directions that carry the correlated variance.
    A GLS *likelihood* that calibrates needs a full-rank ``R`` from the theory clustering
    :math:`C_\\ell` (plus shot noise):

    * on the full sky, :math:`R^{-1}` is diagonal in spherical-harmonic space with eigenvalues
      :math:`1/(C_\\ell + N_\\ell)` — a pair of forward/adjoint SHTs per apply;
    * on the **cut sky** the mask couples multipoles, so ``R^{-1} v`` requires an iterative
      (conjugate-gradient) solve wrapping the harmonic operator, differentiable through JAX.

    Until then the calibrated error bars come from :func:`mock_sandwich_covariance` (parameter
    level).  Raising rather than returning a placeholder keeps callers honest.
    """
    raise NotImplementedError(
        "build_harmonic_precision (theory-C_ell full-rank GLS) is a documented follow-up; "
        "use mock_sandwich_covariance for calibrated additive errors in the meantime."
    )
