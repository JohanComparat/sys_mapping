"""Harmonic-space systematic mitigation via pseudo-Cℓ estimators.

Implements the template subtraction, harmonic bias correction, and
Basic/Extended Mode Projection (BMP/EMP) methods described in
Elsner, Leistedt & Peiris 2016 (MNRAS 456, 2095) and
Elsner, Leistedt & Peiris 2017 (MNRAS 465, 1847).
"""

import warnings

import healpy as hp
import numpy as np


def measure_pseudo_cl(
    delta_g: np.ndarray,
    mask: np.ndarray,
    lmax: int | None = None,
    *,
    use_pixel_weights: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the pseudo-Cℓ power spectrum of a galaxy overdensity map.

    Applies the survey mask, then calls ``healpy.anafast`` to compute the
    pseudo-Cℓ.  The mask is applied multiplicatively so that pixels outside
    the footprint do not contribute.

    Parameters
    ----------
    delta_g:
        Full-sky HEALPix overdensity map (shape ``(n_pix,)``).  Pixels
        outside the survey should be set to 0 (or masked by ``mask``).
    mask:
        Boolean or float weight map (shape ``(n_pix,)``).  ``True`` / 1.0
        marks good (included) pixels.
    lmax:
        Maximum multipole.  Defaults to ``3 * nside - 1``.
    use_pixel_weights:
        If ``True``, pass ``use_pixel_weights=True`` to ``healpy.anafast``
        for improved accuracy at high ℓ.

    Returns
    -------
    ell:
        Integer multipole array ``[0, 1, …, lmax]``.
    pseudo_cl:
        Pseudo-Cℓ estimates (shape ``(lmax+1,)``), not corrected for mask
        mode-coupling.

    Notes
    -----
    The pseudo-Cℓ is related to the true power spectrum via the mode-coupling
    matrix :math:`M_{\ell\ell'}` (MASTER equation):

    .. math::

        \\langle\\tilde C_\\ell\\rangle = \\sum_{\\ell'} M_{\\ell\\ell'}\\,C_{\\ell'}

    For a clean field without systematics, a simple correction by the mask
    power (mean-square of the mask) provides a first-order approximation.

    References
    ----------
    Elsner, Leistedt & Peiris 2016, MNRAS 456, 2095.
    Ho et al. 2012, ApJ 761, 14.
    """
    nside = hp.npix2nside(len(delta_g))
    if lmax is None:
        lmax = 3 * nside - 1

    mask_float = np.asarray(mask, dtype=float)
    masked_map = delta_g * mask_float

    cl = hp.anafast(masked_map, lmax=lmax, use_pixel_weights=use_pixel_weights)
    ell = np.arange(len(cl))
    return ell, cl


def subtract_template_cl(
    pseudo_cl: np.ndarray,
    delta_t: np.ndarray,
    mask: np.ndarray,
    alpha: np.ndarray,
    lmax: int | None = None,
) -> np.ndarray:
    """Subtract template pseudo-Cℓ contributions from the measured power spectrum.

    Implements the template subtraction (TS) method of Elsner+2016 (Eq. 1–2):

    .. math::

        \\tilde C_\\ell^{\\rm TS} = \\hat C_\\ell^{d\\times d}
        - \\sum_i \\hat\\alpha_i\\,\\hat C_\\ell^{t_i \\times t_i}

    Parameters
    ----------
    pseudo_cl:
        Measured pseudo-Cℓ of the galaxy field (shape ``(lmax+1,)``).
    delta_t:
        Template maps (shape ``(n_sys, n_pix)``).
    mask:
        Survey mask (shape ``(n_pix,)``).
    alpha:
        Contamination amplitude estimates (shape ``(n_sys,)``).
    lmax:
        Maximum multipole.  If ``None``, inferred from ``len(pseudo_cl) - 1``.

    Returns
    -------
    cl_cleaned:
        Template-subtracted pseudo-Cℓ (shape ``(lmax+1,)``).

    Notes
    -----
    The residual bias after template subtraction is

    .. math::

        b_\\ell = -\\frac{n_{\\rm sys}}{2\\ell+1}

    Use :func:`harmonic_bias` to compute and subtract this term.

    References
    ----------
    Elsner, Leistedt & Peiris 2016, MNRAS 456, 2095.
    """
    if lmax is None:
        lmax = len(pseudo_cl) - 1

    mask_float = np.asarray(mask, dtype=float)
    cl_cleaned = pseudo_cl.copy()

    for i, (a_i, t_i) in enumerate(zip(alpha, delta_t)):
        masked_template = t_i * mask_float
        cl_t = hp.anafast(masked_template, lmax=lmax, use_pixel_weights=True)
        cl_cleaned -= a_i * cl_t

    return cl_cleaned


def harmonic_bias(n_templates: int, ell: np.ndarray) -> np.ndarray:
    """Closed-form additive bias from template subtraction.

    Template subtraction introduces a systematic negative bias in each
    multipole band (Elsner+2016, Eq. 8):

    .. math::

        b_\\ell = -\\frac{n_{\\rm templates}}{2\\ell+1}

    Parameters
    ----------
    n_templates:
        Number of subtracted templates.
    ell:
        Multipole array (integer, shape ``(n_ell,)``).

    Returns
    -------
    bias:
        Additive bias on :math:`C_\\ell` (shape ``(n_ell,)``).  Subtract
        this from the measured :math:`C_\\ell` to obtain an unbiased
        estimate.

    References
    ----------
    Elsner, Leistedt & Peiris 2016, MNRAS 456, 2095, Eq. 8.
    """
    ell = np.asarray(ell, dtype=float)
    return -n_templates / (2.0 * ell + 1.0)


def mode_projection_bias(
    pseudo_cl: np.ndarray,
    coupling_matrix: np.ndarray | None,
    ell: np.ndarray,
    n_templates: int,
    *,
    mode: str = "basic",
    threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Basic or Extended Mode Projection (BMP/EMP) with bias correction.

    BMP marginalises over template modes by modifying the pixel covariance;
    the result is equivalent to projecting the templates from the data vector
    before computing the power spectrum.  EMP extends this by only projecting
    modes whose template-to-signal ratio exceeds a threshold.

    Parameters
    ----------
    pseudo_cl:
        Measured pseudo-Cℓ of the galaxy field (shape ``(n_ell,)``).
    coupling_matrix:
        Mode-coupling matrix :math:`M_{\\ell\\ell'}` (shape ``(n_ell, n_ell)``).
        If ``None``, the identity matrix is used (full-sky approximation).
    ell:
        Multipole array (shape ``(n_ell,)``).
    n_templates:
        Number of templates projected.
    mode:
        ``"basic"`` for BMP or ``"extended"`` for EMP.
    threshold:
        EMP-only.  Modes with template-to-signal ratio below this threshold
        are not projected.  Typical values: 0.05–0.20.

    Returns
    -------
    cl_deproj:
        Deprojected power spectrum estimate (shape ``(n_ell,)``).
    bias_correction:
        Additive bias that has been subtracted from ``pseudo_cl`` to obtain
        ``cl_deproj`` (shape ``(n_ell,)``).

    Notes
    -----
    **BMP** bias per multipole (Elsner+2016):

    .. math::

        b_\\ell^{\\rm BMP} = -\\frac{n_{\\rm templates}}{2\\ell+1}

    This is the same as the template subtraction bias — the advantage of
    BMP over TS is that the amplitude :math:`\\hat\\alpha_i` does not need
    to be estimated externally.

    **EMP** applies the same correction only to modes where
    :math:`\\hat C_\\ell^{t_i} / C_\\ell^{ss} > \\text{threshold}`.
    Modes below the threshold are left uncorrected (smaller variance penalty).

    When ``coupling_matrix`` is ``None`` (identity), the result is a
    full-sky approximation.  For cut-sky surveys use the mask coupling
    matrix from NaMaster (``pymaster.compute_coupling_matrix``).

    References
    ----------
    Elsner, Leistedt & Peiris 2016, MNRAS 456, 2095, Sec. 3.
    Leistedt & Peiris 2014, MNRAS 444, 2.
    """
    ell = np.asarray(ell, dtype=float)
    n_ell = len(pseudo_cl)

    if coupling_matrix is None:
        coupling_matrix = np.eye(n_ell)

    if mode == "basic":
        bias = harmonic_bias(n_templates, ell)
        # Bias is per true-C_ell band; apply coupling matrix to get pseudo-Cl bias
        bias_pseudo = coupling_matrix @ bias
        cl_deproj = pseudo_cl - bias_pseudo
        return cl_deproj, bias_pseudo

    elif mode == "extended":
        # EMP: only project modes where template power dominates
        # Estimate signal C_ell as the pseudo-Cl itself (iterative improvement
        # would use a prior or ILC estimate; this is the zeroth-order EMP)
        cl_signal = np.where(pseudo_cl > 0, pseudo_cl, 1e-30)
        # Template-to-signal ratio proxy: use the expected bias per mode
        template_ratio = np.abs(harmonic_bias(1, ell)) / (cl_signal / n_templates + 1e-30)
        project_mask = template_ratio > threshold
        n_projected = int(np.sum(project_mask))
        bias = np.where(project_mask, harmonic_bias(n_projected, ell), 0.0)
        bias_pseudo = coupling_matrix @ bias
        cl_deproj = pseudo_cl - bias_pseudo
        return cl_deproj, bias_pseudo

    else:
        raise ValueError(f"mode must be 'basic' or 'extended', got '{mode}'")
