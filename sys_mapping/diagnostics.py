"""Systematic mapping diagnostics: null tests, SNR ranking, footprint masking.

Implements:
  - Null test cross-correlations (Ross+2011)
  - SNR-based template ranking with three estimators (Tanidis+2026)
  - Footprint masking sensitivity diagnostics (Rodríguez-Monroy+2025)
"""

from __future__ import annotations

import numpy as np


def null_test_cross_correlations(
    weights: np.ndarray,
    delta_t: np.ndarray,
    n_bootstrap: int = 100,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Test for residual systematic contamination via weight × template correlations.

    After decontamination, the per-pixel systematic weights should be
    statistically independent of all template maps.  A significant
    Pearson correlation between ``weights`` and ``delta_t[i]`` indicates
    residual contamination from template ``i``.

    Parameters
    ----------
    weights:
        Per-pixel systematic weights (shape ``(n_pix,)``).
    delta_t:
        Template maps (shape ``(n_sys, n_pix)``).
    n_bootstrap:
        Number of permutation resamples for computing p-values.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    result:
        Dictionary with:

        - ``"correlations"``: Pearson :math:`r(w, t_i)` for each template
          (shape ``(n_sys,)``).
        - ``"p_values"``: two-sided permutation p-value for each template
          (shape ``(n_sys,)``).  Small values indicate significant residual
          contamination.

    Notes
    -----
    The Pearson correlation is:

    .. math::

        r_i = \\frac{\\sum_p (w_p - \\bar w)(t_{i,p} - \\bar t_i)}
                    {\\sqrt{\\sum_p (w_p-\\bar w)^2 \\sum_p (t_{i,p}-\\bar t_i)^2}}

    A well-corrected field satisfies :math:`|r_i| \\approx 0` for all
    templates.  Permutation p-values are computed by shuffling ``weights``
    and recomputing the correlation ``n_bootstrap`` times.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import null_test_cross_correlations
    >>> rng = np.random.default_rng(0)
    >>> n_pix, n_sys = 500, 3
    >>> weights = rng.standard_normal(n_pix) + 1.0
    >>> delta_t = rng.standard_normal((n_sys, n_pix))
    >>> result = null_test_cross_correlations(weights, delta_t, n_bootstrap=50)
    >>> result["correlations"].shape
    (3,)
    >>> result["p_values"].shape
    (3,)
    >>> np.all(result["p_values"] >= 0) and np.all(result["p_values"] <= 1)
    True

    References
    ----------
    Ross et al. 2011, MNRAS 417, 1350.
    """
    rng = np.random.default_rng(seed)
    n_sys = delta_t.shape[0]

    w_centered = weights - weights.mean()
    w_norm = np.sqrt(np.sum(w_centered**2))

    correlations = np.empty(n_sys)
    for i, t_i in enumerate(delta_t):
        t_centered = t_i - t_i.mean()
        t_norm = np.sqrt(np.sum(t_centered**2))
        if w_norm < 1e-30 or t_norm < 1e-30:
            correlations[i] = 0.0
        else:
            correlations[i] = float(np.sum(w_centered * t_centered) / (w_norm * t_norm))

    # Permutation p-values
    p_values = np.empty(n_sys)
    for i, t_i in enumerate(delta_t):
        t_centered = t_i - t_i.mean()
        t_norm = np.sqrt(np.sum(t_centered**2))
        obs_r = abs(correlations[i])
        count_extreme = 0
        for _ in range(n_bootstrap):
            w_shuffled = rng.permutation(w_centered)
            if w_norm > 1e-30 and t_norm > 1e-30:
                r_perm = abs(float(np.sum(w_shuffled * t_centered) / (w_norm * t_norm)))
            else:
                r_perm = 0.0
            if r_perm >= obs_r:
                count_extreme += 1
        p_values[i] = (count_extreme + 1) / (n_bootstrap + 1)

    return {"correlations": correlations, "p_values": p_values}


def snr_template_ranking(
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    *,
    method: str = "template",
) -> np.ndarray:
    """Rank systematic templates by signal-to-noise ratio of their contamination.

    Three SNR definitions adapted from Tanidis et al. 2026 (Sec. 3) for galaxy
    clustering:

    - ``"template"``: :math:`|\\hat\\alpha_i| / \\sigma_{\\hat\\alpha_i}` from OLS.
    - ``"data"``: :math:`|\\mathrm{Corr}(\\delta_g, t_i)|`, the absolute
      cross-correlation between the galaxy field and each template.
    - ``"peak"``: peak pseudo-:math:`C_\\ell` cross-spectrum between
      :math:`\\delta_g` and :math:`t_i`, divided by the noise level.

    Parameters
    ----------
    delta_g_obs:
        Observed galaxy overdensity (shape ``(n_pix,)``).
    delta_t:
        Template maps (shape ``(n_sys, n_pix)``).
    method:
        One of ``"template"``, ``"data"``, ``"peak"``.

    Returns
    -------
    snr:
        SNR value for each template (shape ``(n_sys,)``).  Higher values
        indicate a more contaminating template.

    Notes
    -----
    Templates can be sorted by ``np.argsort(snr)[::-1]`` to obtain a
    ranking from most to least contaminating.  The ``"peak"`` method
    requires ``healpy`` (already a package dependency) and assumes the
    overdensity map covers the full sky (or has zeros outside the mask).

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import snr_template_ranking
    >>> rng = np.random.default_rng(0)
    >>> n_pix, n_sys = 3000, 4
    >>> delta_t = rng.standard_normal((n_sys, n_pix))
    >>> # Inject strong contamination from template 2
    >>> delta_g = 0.5 * delta_t[2] + rng.standard_normal(n_pix) * 0.2
    >>> snr = snr_template_ranking(delta_g, delta_t, method="data")
    >>> snr.shape
    (4,)
    >>> np.argmax(snr) == 2
    True

    References
    ----------
    Tanidis et al. 2026, MNRAS 547.
    """
    n_sys, n_pix = delta_t.shape

    if method == "template":
        X = delta_t.T  # (n_pix, n_sys)
        alpha_hat, res, rank, sv = np.linalg.lstsq(X, delta_g_obs, rcond=None)
        # OLS variance estimate: sigma^2 = RSS / (n - p)
        residual = delta_g_obs - X @ alpha_hat
        dof = max(n_pix - n_sys, 1)
        sigma2 = float(np.sum(residual**2) / dof)
        XtX_inv = np.linalg.pinv(X.T @ X)
        param_var = sigma2 * np.diag(XtX_inv)
        param_var = np.maximum(param_var, 1e-30)
        snr = np.abs(alpha_hat) / np.sqrt(param_var)
        return snr

    elif method == "data":
        g_centered = delta_g_obs - delta_g_obs.mean()
        g_norm = np.sqrt(np.sum(g_centered**2))
        snr = np.empty(n_sys)
        for i, t_i in enumerate(delta_t):
            t_centered = t_i - t_i.mean()
            t_norm = np.sqrt(np.sum(t_centered**2))
            if g_norm < 1e-30 or t_norm < 1e-30:
                snr[i] = 0.0
            else:
                snr[i] = abs(float(np.sum(g_centered * t_centered) / (g_norm * t_norm)))
        return snr

    elif method == "peak":
        import healpy as hp

        nside = hp.npix2nside(n_pix)
        lmax = 3 * nside - 1
        cl_g = hp.anafast(delta_g_obs, lmax=lmax)
        noise_level = np.median(np.abs(cl_g)) + 1e-30
        snr = np.empty(n_sys)
        for i, t_i in enumerate(delta_t):
            cl_cross = hp.anafast(delta_g_obs, map2=t_i, lmax=lmax)
            snr[i] = float(np.max(np.abs(cl_cross))) / noise_level
        return snr

    else:
        raise ValueError(f"method must be 'template', 'data', or 'peak', got '{method}'")


def footprint_mask_diagnostics(
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    mask_fractions: np.ndarray,
    good_pixels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Quantify sensitivity of systematic parameters to the masking threshold.

    For each masking fraction in ``mask_fractions``, the most poorly
    connected pixels (those with the lowest overdensity variance from the
    template model) are excluded, and OLS contamination amplitudes are
    re-estimated.  Large variation of the amplitudes with masking threshold
    indicates sensitivity to the survey boundary or to systematic outliers.

    This is the footprint masking sensitivity analysis of
    Rodríguez-Monroy et al. 2025, Sec. 4.

    Parameters
    ----------
    delta_g_obs:
        Observed galaxy overdensity at unmasked pixels (shape ``(n_pix,)``).
    delta_t:
        Template maps at unmasked pixels (shape ``(n_sys, n_pix)``).
    mask_fractions:
        Array of additional pixel fractions to mask on top of the baseline,
        in increasing order (e.g. ``[0.05, 0.10, 0.15, 0.20]``).
    good_pixels:
        Boolean mask over the full footprint (shape ``(n_full_pix,)``).
        Used only for computing a pixel-quality ranking (template variance
        at each pixel).

    Returns
    -------
    result:
        Dictionary with:

        - ``"alpha_hat"``: OLS amplitudes per masking level
          (shape ``(n_thresholds, n_sys)``).
        - ``"scatter"``: standard deviation of ``alpha_hat`` across masking
          levels (shape ``(n_sys,)``).  Near-zero scatter means the
          amplitude estimates are robust to masking.

    Notes
    -----
    Pixels are ranked by the RMS template value
    :math:`\\sigma_p = \\sqrt{\\sum_i t_{i,p}^2 / n_{\\rm sys}}`.  Pixels
    with the **lowest** :math:`\\sigma_p` (fewest systematic features) are
    masked first, so that the remaining pixels have progressively higher
    systematic leverage.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import footprint_mask_diagnostics
    >>> rng = np.random.default_rng(0)
    >>> n_pix, n_sys = 4000, 3
    >>> delta_t = rng.standard_normal((n_sys, n_pix))
    >>> delta_g = 0.1 * delta_t[0] + rng.standard_normal(n_pix) * 0.5
    >>> good = np.ones(n_pix, dtype=bool)
    >>> fracs = np.array([0.0, 0.05, 0.10, 0.15])
    >>> result = footprint_mask_diagnostics(delta_g, delta_t, fracs, good)
    >>> result["alpha_hat"].shape
    (4, 3)
    >>> result["scatter"].shape
    (3,)

    References
    ----------
    Rodríguez-Monroy et al. 2025, arXiv:2509.07943.
    """
    n_sys, n_pix = delta_t.shape
    n_thresholds = len(mask_fractions)

    # Pixel-quality metric: RMS template value
    rms_per_pixel = np.sqrt(np.mean(delta_t**2, axis=0))  # (n_pix,)
    rank_order = np.argsort(rms_per_pixel)  # ascending — low rms pixels first

    alpha_all = np.empty((n_thresholds, n_sys))

    for k, frac in enumerate(mask_fractions):
        n_drop = int(np.floor(frac * n_pix))
        keep = np.ones(n_pix, dtype=bool)
        if n_drop > 0:
            drop_idx = rank_order[:n_drop]
            keep[drop_idx] = False

        dg_k = delta_g_obs[keep]
        dt_k = delta_t[:, keep]
        X = dt_k.T  # (n_keep, n_sys)
        alpha_k, *_ = np.linalg.lstsq(X, dg_k, rcond=None)
        alpha_all[k] = alpha_k

    scatter = np.std(alpha_all, axis=0)
    return {"alpha_hat": alpha_all, "scatter": scatter}
