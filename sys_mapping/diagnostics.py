"""Systematic mapping diagnostics: null tests, SNR ranking, footprint masking.

Implements:
  - Null test cross-correlations (Ross+2011)
  - SNR-based template ranking with three estimators (Tanidis+2026)
  - Footprint masking sensitivity diagnostics (Rodríguez-Monroy+2025)
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

if _JAX_AVAILABLE:
    # --- "data": batched Pearson |r| (n_sys, n_pix) × (n_pix,) → (n_sys,) ---
    @jax.jit
    def _jax_pearson(t_mat, g):
        g_c = g - g.mean()
        t_c = t_mat - t_mat.mean(axis=1, keepdims=True)
        g_norm = jnp.linalg.norm(g_c) + 1e-30
        t_norm = jnp.linalg.norm(t_c, axis=1) + 1e-30
        return jnp.abs(t_c @ g_c) / (g_norm * t_norm)

    # --- "template": per-template OLS |t|-stat via vmap ---
    def _one_tstat(t_i, g):
        denom = jnp.dot(t_i, t_i) + 1e-30
        alpha = jnp.dot(t_i, g) / denom
        sigma2 = jnp.mean((g - alpha * t_i) ** 2)
        return jnp.abs(alpha) / jnp.sqrt(sigma2 / denom + 1e-30)

    _jax_tstat = jax.jit(jax.vmap(_one_tstat, in_axes=(0, None)))

    # --- "isd": vmap over templates, poly_order=1 analytic ---
    # Bin-assignment: equal-WIDTH (default) or equal-OCCUPANCY (quantile) edges.
    # Quantile binning is robust when a template's pixel distribution is skewed —
    # equal-width bins then collapse most pixels into one bin and lose the trend.
    def _binidx_width(t_i, n_bins):
        t_min, t_max = t_i.min(), t_i.max()
        span = t_max - t_min + 1e-30
        return jnp.clip(jnp.floor((t_i - t_min) / span * n_bins).astype(jnp.int32),
                        0, n_bins - 1)

    def _binidx_quantile(t_i, n_bins):
        edges = jnp.quantile(t_i, jnp.linspace(0.0, 1.0, n_bins + 1))
        # place each pixel by the n_bins-1 interior quantile edges → indices 0..n_bins-1
        return jnp.clip(jnp.searchsorted(edges[1:-1], t_i, side="right").astype(jnp.int32),
                        0, n_bins - 1)

    # Per-bin inverse variance, shared by both ISD kernels.
    #
    # The centred (two-pass) form is REQUIRED.  The algebraically equivalent
    # E[g²] - E[g]² is what this code used to do, and under jit XLA contracts it
    # into an FMA whose rounding returns ~1.5e-20 instead of exactly 0 for a bin
    # holding a single pixel.  That squeaked past a `> 1e-20` guard and gave the
    # bin an inverse variance of ~7e19, which swamped the whole chi²: a pure-noise
    # template scored 6e17 against 389 for a genuinely contaminated one.  Eagerly
    # the same expression returns exactly 0.0, so the bug appeared only in the
    # compiled path and moves with the XLA version.
    #
    # Two further conditions make the test mean what it says: a bin holding one
    # pixel carries no variance information whatever the arithmetic reports, and
    # the floor is taken relative to the field's own scatter so it is scale-free.
    # This matches the NumPy fallback in `snr_template_ranking`, which uses
    # np.std (two-pass) and drops degenerate bins.
    def _bin_inv_var(g, bin_idx, n_bins, count, n_b, g_mean, g_bar):
        g_var = jnp.zeros(n_bins).at[bin_idx].add((g - g_mean[bin_idx]) ** 2) / n_b
        g_var_floor = 1e-12 * jnp.mean((g - g_bar) ** 2)
        valid = (count >= 2) & (g_var > g_var_floor)
        inv_s2 = jnp.where(valid, n_b / jnp.where(valid, g_var, 1.0), 0.0)
        return inv_s2, valid

    def _isd_core(t_i, g, bin_idx, n_bins):
        g_bar = g.mean()
        count  = jnp.zeros(n_bins).at[bin_idx].add(1)
        g_sum  = jnp.zeros(n_bins).at[bin_idx].add(g)
        t_sum  = jnp.zeros(n_bins).at[bin_idx].add(t_i)
        n_b    = count.clip(min=1)
        g_mean = g_sum / n_b
        t_mean = t_sum / n_b
        inv_s2, valid = _bin_inv_var(g, bin_idx, n_bins, count, n_b, g_mean, g_bar)
        chi2_null = jnp.sum(inv_s2 * (g_mean - g_bar) ** 2)
        W   = inv_s2
        S   = W.sum();   Ss  = (W * t_mean).sum();  Sn  = (W * g_mean).sum()
        Sss = (W * t_mean ** 2).sum();  Ssn = (W * t_mean * g_mean).sum()
        alpha = (S * Ssn - Ss * Sn) / (S * Sss - Ss ** 2 + 1e-30)
        beta  = (Sn - alpha * Ss) / (S + 1e-30)
        chi2_model = jnp.sum(inv_s2 * (g_mean - (alpha * t_mean + beta)) ** 2)
        return jnp.where(jnp.sum(valid) < 2, 0.0,
                         jnp.maximum(chi2_null - chi2_model, 0.0))

    _jax_isd_cache: dict[tuple[int, bool], object] = {}

    def _get_jax_isd(n_bins: int, quantile: bool = False):
        key = (n_bins, quantile)
        if key not in _jax_isd_cache:
            binf = _binidx_quantile if quantile else _binidx_width
            _jax_isd_cache[key] = jax.jit(
                jax.vmap(lambda t, g: _isd_core(t, g, binf(t, n_bins), n_bins),
                         in_axes=(0, None))
            )
        return _jax_isd_cache[key]

    # --- "isd", general: arbitrary polynomial order + optional fracdet weights ---
    def _one_isd_poly(t_i, g, w, n_bins, order, quantile):
        """Δχ² of a weighted degree-``order`` polynomial fit of binned n̄(s̄).

        Generalises :func:`_isd_core` (poly_order=1, unit weights) to any order and
        optional per-pixel weights ``w`` (fracdet).  Bin means use ``w``; the
        per-bin standard error uses the unweighted variance / pixel count, exactly
        as the NumPy fallback in :func:`snr_template_ranking`.  ``quantile`` selects
        equal-occupancy bins (robust to skewed templates) over equal-width.
        """
        w_total = w.sum()
        g_bar = jnp.dot(w, g) / (w_total + 1e-30)
        bin_idx = _binidx_quantile(t_i, n_bins) if quantile else _binidx_width(t_i, n_bins)
        W_b  = jnp.zeros(n_bins).at[bin_idx].add(w)          # Σ w
        wt_b = jnp.zeros(n_bins).at[bin_idx].add(w * t_i)    # Σ w t
        wg_b = jnp.zeros(n_bins).at[bin_idx].add(w * g)      # Σ w g
        cnt  = jnp.zeros(n_bins).at[bin_idx].add(1.0)        # pixel count
        g1   = jnp.zeros(n_bins).at[bin_idx].add(g)          # Σ g   (unweighted)

        s_arr = wt_b / jnp.maximum(W_b, 1e-30)               # weighted t mean
        n_arr = wg_b / jnp.maximum(W_b, 1e-30)               # weighted g mean
        n_b_uw = jnp.maximum(cnt, 1.0)
        g_mean_uw = g1 / n_b_uw
        inv_s2, ok = _bin_inv_var(g, bin_idx, n_bins, cnt, n_b_uw, g_mean_uw, g_bar)
        valid = (W_b > 1e-30) & ok
        inv_s2 = jnp.where(valid, inv_s2, 0.0)

        chi2_null = jnp.sum(inv_s2 * (n_arr - g_bar) ** 2)

        # Weighted polynomial least squares (weight = inv_s2): solve on the
        # sqrt-weighted Vandermonde to avoid squaring the condition number.
        powers = jnp.arange(order + 1)
        V = s_arr[:, None] ** powers[None, :]                # (n_bins, order+1)
        sw = jnp.sqrt(inv_s2)
        coeffs, *_ = jnp.linalg.lstsq(sw[:, None] * V, sw * n_arr, rcond=None)
        f_s = V @ coeffs
        chi2_model = jnp.sum(inv_s2 * (n_arr - f_s) ** 2)

        n_valid = jnp.sum(valid)
        ok = n_valid >= 2
        # The range the fit is actually supported on: the outermost *valid* bin
        # centres.  A polynomial fitted to binned means says nothing beyond them,
        # and survey-property maps are skewed enough that evaluating a cubic out
        # in the tail is not a small extrapolation -- it is the difference between
        # a correction and a divergence.
        s_lo = jnp.min(jnp.where(valid, s_arr, jnp.inf))
        s_hi = jnp.max(jnp.where(valid, s_arr, -jnp.inf))
        # Coefficients are returned alongside Delta chi^2 because the ISD *fit*
        # (regression.iterative_systematics_decontamination) needs the fitted
        # curve, not only its significance.  They are ascending in power:
        # F(t) = coeffs[0] + coeffs[1] t + ... + coeffs[order] t**order.
        return (jnp.where(ok, jnp.maximum(chi2_null - chi2_model, 0.0), 0.0),
                jnp.where(ok, coeffs, jnp.zeros_like(coeffs)),
                jnp.where(ok, s_lo, 0.0),
                jnp.where(ok, s_hi, 0.0))

    _jax_isd_poly_cache: dict[tuple[int, int, bool], object] = {}

    def _get_jax_isd_poly_full(n_bins: int, order: int, quantile: bool = False):
        """vmapped kernel returning ``(delta_chi2, coeffs, t_lo, t_hi)``."""
        key = (n_bins, order, quantile)
        if key not in _jax_isd_poly_cache:
            _jax_isd_poly_cache[key] = jax.jit(
                jax.vmap(lambda t, g, w: _one_isd_poly(t, g, w, n_bins, order, quantile),
                         in_axes=(0, None, None))
            )
        return _jax_isd_poly_cache[key]

    def _get_jax_isd_poly(n_bins: int, order: int, quantile: bool = False):
        """vmapped kernel returning Delta chi^2 only (ranking call sites)."""
        full = _get_jax_isd_poly_full(n_bins, order, quantile)
        return lambda t, g, w: full(t, g, w)[0]

    # --- null test: signed corr + permutation p-values, vmapped over resamples ---
    def _null_test_jax(weights, delta_t, n_bootstrap, seed):
        w_c = weights - weights.mean()
        w_norm = jnp.linalg.norm(w_c) + 1e-30
        t_c = delta_t - delta_t.mean(axis=1, keepdims=True)
        t_norm = jnp.linalg.norm(t_c, axis=1) + 1e-30
        signed = (t_c @ w_c) / (t_norm * w_norm)             # (n_sys,)
        obs = jnp.abs(signed)

        keys = jax.random.split(jax.random.PRNGKey(seed), n_bootstrap)
        one = lambda k: jnp.abs(t_c @ jax.random.permutation(k, w_c)) / (t_norm * w_norm)
        perm = jax.vmap(one)(keys)                           # (n_bootstrap, n_sys)
        count = jnp.sum(perm >= obs[None, :], axis=0)        # (n_sys,)
        p = (count + 1.0) / (n_bootstrap + 1.0)
        return signed, p


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

    .. warning::
       Read these per template.  ``max_i |r(w, t_i)|`` is **not** a scalar
       goodness-of-fit and no threshold on it is meaningful, because for an
       additive correction ``w = 1/(1 + sum_i a_i t_i)`` the statistic depends on
       the *support* of ``a``, not its size.  If ``a`` has one non-zero entry then
       ``w`` is a monotone function of that template and ``|r| -> 1`` identically,
       however small the amplitude: on three independent unit-variance templates,
       an amplitude of 1e-1 on one of them gives 0.9886 and an amplitude of 1e-6
       gives 1.0000.  Spreading the same amplitude over all three gives 0.57.  So a
       sparser and more accurate correction scores *worse*, and a method that fits
       nothing scores a perfect zero.  To ask whether a correction is over-fitted,
       compare the recovered amplitudes against a known truth instead.

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
    if _JAX_AVAILABLE:
        signed, p = _null_test_jax(
            jnp.asarray(weights, dtype=jnp.float64),
            jnp.asarray(delta_t, dtype=jnp.float64),
            int(n_bootstrap), int(seed),
        )
        return {"correlations": np.asarray(signed), "p_values": np.asarray(p)}

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
    n_bins: int = 10,
    poly_order: int = 1,
    fracdet: np.ndarray | None = None,
    binning: str = "width",
) -> np.ndarray:
    """Rank systematic templates by signal-to-noise ratio of their contamination.

    Four SNR definitions:

    - ``"template"``: :math:`|\\hat\\alpha_i| / \\sigma_{\\hat\\alpha_i}` from OLS.
    - ``"data"``: :math:`|\\mathrm{Corr}(\\delta_g, t_i)|`, the absolute
      cross-correlation between the galaxy field and each template.
    - ``"peak"``: peak pseudo-:math:`C_\\ell` cross-spectrum between
      :math:`\\delta_g` and :math:`t_i`, divided by the noise level.
    - ``"isd"``: :math:`\\Delta\\chi^2 = \\chi^2_{\\rm null} - \\chi^2_{\\rm model}`
      from the ISD 1D binned relation (Rodríguez-Monroy et al. 2025, Eqs. 4–5).
      For each template, pixels are binned by template value; the fracdet-weighted
      mean galaxy density per bin is compared to a polynomial fit.  Large
      :math:`\\Delta\\chi^2` indicates significant systematic contamination.

    Parameters
    ----------
    delta_g_obs:
        Observed galaxy overdensity (shape ``(n_pix,)``).
    delta_t:
        Template maps (shape ``(n_sys, n_pix)``).
    method:
        One of ``"template"``, ``"data"``, ``"peak"``, ``"isd"``.
    n_bins:
        Number of equal-width bins for ``method="isd"``.  Ignored otherwise.
    poly_order:
        Polynomial degree for the 1D fit in ``method="isd"`` (1 = linear,
        3 = cubic as used in DES Y6).  Ignored otherwise.
    fracdet:
        Per-pixel fractional coverage weights (shape ``(n_pix,)``).  Only
        used by ``method="isd"``; if ``None``, uniform weights are assumed.
    binning:
        Bin-edge scheme for ``method="isd"``.  ``"width"`` (default) uses
        ``n_bins`` equal-width bins across the template range; ``"quantile"``
        (aka ``"equal_occupancy"``) uses equal-occupancy bins at the template
        value quantiles.  Quantile binning is strongly preferred for real,
        skewed systematic maps, where equal-width bins pile most pixels into a
        single bin and lose the density–template trend.  Ignored otherwise.

    Returns
    -------
    snr:
        SNR value for each template (shape ``(n_sys,)``).  Higher values
        indicate a more contaminating template.

    .. warning::
       For ``method="template"`` and ``method="data"`` the denominator is the
       independent-pixel error, and on a clustered field that is too small by a
       factor of 5--13.  Measured on contamination-free simulations, a
       :math:`3\sigma` cut on these values fires on 76--96 % of realisations
       against the 0.27 % it should.  **Use them to rank templates, not to claim
       a detection.**  A calibrated significance needs
       :func:`~sys_mapping.covariance.mock_sandwich_covariance` in the
       denominator, or the mock-calibrated ``method="isd"`` route, which
       normalises by :math:`\Delta\chi^2_{68}` measured on nulls.

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
    Rodríguez-Monroy et al. 2025, arXiv:2509.07943, Sec. IV.A.1, Eqs. 4–5.
    """
    n_sys, n_pix = delta_t.shape

    if method == "template":
        if _JAX_AVAILABLE:
            return np.asarray(
                _jax_tstat(jnp.asarray(delta_t, dtype=jnp.float64),
                           jnp.asarray(delta_g_obs, dtype=jnp.float64))
            )
        # NumPy fallback: per-template OLS |t|-stat
        snr = np.empty(n_sys)
        for i, t_i in enumerate(delta_t):
            denom = float(np.dot(t_i, t_i)) + 1e-30
            alpha = float(np.dot(t_i, delta_g_obs)) / denom
            sigma2 = float(np.mean((delta_g_obs - alpha * t_i) ** 2))
            snr[i] = abs(alpha) / np.sqrt(sigma2 / denom + 1e-30)
        return snr

    elif method == "data":
        if _JAX_AVAILABLE:
            return np.asarray(
                _jax_pearson(jnp.asarray(delta_t, dtype=jnp.float64),
                             jnp.asarray(delta_g_obs, dtype=jnp.float64))
            )
        # NumPy fallback: Pearson |r| per template
        g_centered = delta_g_obs - delta_g_obs.mean()
        g_norm = np.sqrt(np.sum(g_centered ** 2)) + 1e-30
        snr = np.empty(n_sys)
        for i, t_i in enumerate(delta_t):
            t_centered = t_i - t_i.mean()
            t_norm = np.sqrt(np.sum(t_centered ** 2)) + 1e-30
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

    elif method == "isd":
        if binning not in ("width", "quantile", "equal_occupancy"):
            raise ValueError("binning must be 'width', 'quantile', or 'equal_occupancy'")
        quantile = binning in ("quantile", "equal_occupancy")
        if _JAX_AVAILABLE and poly_order == 1 and fracdet is None:
            return np.asarray(
                _get_jax_isd(n_bins, quantile)(
                    jnp.asarray(delta_t, dtype=jnp.float64),
                    jnp.asarray(delta_g_obs, dtype=jnp.float64),
                )
            )
        if _JAX_AVAILABLE:
            # General JAX path: arbitrary poly_order and/or fracdet weights.
            w = (jnp.ones(n_pix, dtype=jnp.float64) if fracdet is None
                 else jnp.asarray(fracdet, dtype=jnp.float64))
            return np.asarray(
                _get_jax_isd_poly(n_bins, int(poly_order), quantile)(
                    jnp.asarray(delta_t, dtype=jnp.float64),
                    jnp.asarray(delta_g_obs, dtype=jnp.float64),
                    w,
                )
            )

        # NumPy fallback (JAX unavailable)
        w = np.ones(n_pix, dtype=float) if fracdet is None else np.asarray(fracdet, dtype=float)
        w_total = float(np.sum(w))
        g_bar = float(np.dot(w, delta_g_obs) / w_total) if w_total > 1e-30 else 0.0

        snr = np.empty(n_sys)
        for i, t_i in enumerate(delta_t):
            t_min, t_max = float(t_i.min()), float(t_i.max())
            if t_min >= t_max:
                snr[i] = 0.0
                continue

            # Assign each pixel to one of n_bins bins (equal-width or equal-occupancy)
            if quantile:
                edges = np.quantile(t_i, np.linspace(0.0, 1.0, n_bins + 1))
                bin_idx = np.clip(np.searchsorted(edges[1:-1], t_i, side="right"),
                                  0, n_bins - 1)
            else:
                span = t_max - t_min
                bin_idx = np.floor((t_i - t_min) / span * n_bins).astype(int)
                bin_idx = np.clip(bin_idx, 0, n_bins - 1)

            s_b_list: list[float] = []
            n_b_list: list[float] = []
            sigma_b_list: list[float] = []
            for b in range(n_bins):
                mask_b = bin_idx == b
                if not np.any(mask_b):
                    continue
                w_b = w[mask_b]
                w_b_sum = float(np.sum(w_b))
                if w_b_sum < 1e-30:
                    continue
                g_b = delta_g_obs[mask_b]
                n_b_pix = int(np.sum(mask_b))
                std_g = float(np.std(g_b))
                if std_g < 1e-10:
                    continue  # degenerate bin — all pixels have same overdensity
                s_b_list.append(float(np.dot(w_b, t_i[mask_b]) / w_b_sum))
                n_b_list.append(float(np.dot(w_b, g_b) / w_b_sum))
                sigma_b_list.append(std_g / np.sqrt(n_b_pix))

            n_valid = len(s_b_list)
            if n_valid < 2:
                snr[i] = 0.0
                continue

            s_arr = np.array(s_b_list)
            n_arr = np.array(n_b_list)
            sigma_arr = np.array(sigma_b_list)
            inv_s2 = 1.0 / sigma_arr ** 2

            chi2_null = float(np.dot((n_arr - g_bar) ** 2, inv_s2))

            eff_order = min(poly_order, n_valid - 1)
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore")
                coeffs = np.polyfit(s_arr, n_arr, eff_order, w=1.0 / sigma_arr)
            f_s = np.polyval(coeffs, s_arr)
            chi2_model = float(np.dot((n_arr - f_s) ** 2, inv_s2))

            snr[i] = max(chi2_null - chi2_model, 0.0)
        return snr

    else:
        raise ValueError(f"method must be 'template', 'data', 'peak', or 'isd', got '{method}'")


def vet_templates_against_tracer(
    delta_t: np.ndarray,
    tracer: np.ndarray,
    *,
    patch_ids: np.ndarray | None = None,
    fracdet: np.ndarray | None = None,
    template_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Test each template for correlation with an external tracer of true LSS.

    A systematics template is only usable if it traces the *observing conditions*
    and not the structure being measured.  When it carries large-scale structure
    of its own -- because it was built from the same images the catalogue was
    detected in, or because it correlates with a genuine foreground -- the
    regression that removes it also removes signal, and no amount of care in the
    fit will reveal that: the contamination is detected at high significance and
    the clustering is biased low, consistently, on mocks as well as data.

    The test is to correlate each template against a map that traces the matter
    field independently of the survey -- CMB lensing convergence, a Compton-*y*
    map, a weak-lensing mass map -- and reject templates that correlate.  A
    detection is ambiguous in principle (the tracer may itself carry residual
    systematics that the template picks up), so use more than one tracer and do
    not over-read a single one.

    Spearman rank correlation is used rather than Pearson: it is insensitive to
    the extreme values that survey-property maps routinely carry, and it responds
    to monotonic non-linear dependence, which a linear coefficient would miss.

    Parameters
    ----------
    delta_t:
        Template maps at footprint pixels (shape ``(n_sys, n_pix)``).
    tracer:
        External tracer map at the same pixels (shape ``(n_pix,)``).
    patch_ids:
        Spatial patch label per pixel, from
        :func:`~sys_mapping.bootstrap.assign_spatial_patches`.  When given, the
        uncertainty is the delete-one-patch jackknife over patches, which is the
        only honest error bar here -- pixel-level errors would treat a correlated
        field as independent samples and overstate every significance.  When
        ``None`` the Gaussian approximation
        :math:`\\sigma = 1/\\sqrt{n_{\\rm pix} - 3}` is used and flagged.
    fracdet:
        Per-pixel coverage weights (shape ``(n_pix,)``), used to weight the
        jackknife patches by their observed area.
    template_names:
        Optional names, echoed back in the result for reporting.

    Returns
    -------
    dict with keys

        - ``"rho"`` -- ``(n_sys,)`` Spearman coefficient per template.
        - ``"sigma"`` -- ``(n_sys,)`` uncertainty on ``rho``.
        - ``"significance"`` -- ``(n_sys,)`` ``|rho| / sigma``.
        - ``"jackknife"`` -- bool, whether the errors are jackknife or Gaussian.
        - ``"names"`` -- the names passed in, or ``None``.

    Precision
    ---------
    The jackknife uncertainty is
    :math:`\\sigma^2 = \\frac{K-1}{K}\\sum_k (\\rho_k - \\bar\\rho)^2` over the
    ``K`` delete-one-patch estimates, the standard form for a statistic that is
    not a simple mean.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping.diagnostics import vet_templates_against_tracer
    >>> rng = np.random.default_rng(0)
    >>> n_pix = 5000
    >>> tracer = rng.standard_normal(n_pix)
    >>> clean = rng.standard_normal(n_pix)
    >>> dirty = 0.5 * tracer + 0.5 * rng.standard_normal(n_pix)
    >>> out = vet_templates_against_tracer(np.vstack([clean, dirty]), tracer)
    >>> bool(out["significance"][1] > out["significance"][0])
    True

    References
    ----------
    Weaverdyck et al. 2026, arXiv:2601.14484, Sec. III A and Fig. 7.
    Eggert & Leistedt 2023, ApJS 265, 30 (Legacy Survey image stacks).
    """
    from scipy.stats import rankdata

    delta_t = np.atleast_2d(np.asarray(delta_t, dtype=float))
    tracer = np.asarray(tracer, dtype=float)
    n_sys, n_pix = delta_t.shape
    if tracer.shape != (n_pix,):
        raise ValueError(
            f"tracer has shape {tracer.shape}, expected ({n_pix},) to match "
            f"delta_t {delta_t.shape}")

    w = np.ones(n_pix) if fracdet is None else np.asarray(fracdet, dtype=float)

    def _spearman(idx: np.ndarray) -> np.ndarray:
        """Weighted Spearman rho of every template against the tracer on ``idx``."""
        ww = w[idx]
        w_sum = float(np.sum(ww))
        if w_sum <= 0 or idx.size < 3:
            return np.zeros(n_sys)
        r_tr = rankdata(tracer[idx])
        r_tr = r_tr - np.dot(ww, r_tr) / w_sum
        den_tr = np.sqrt(np.dot(ww, r_tr ** 2)) + 1e-30
        out = np.empty(n_sys)
        for i in range(n_sys):
            r_t = rankdata(delta_t[i, idx])
            r_t = r_t - np.dot(ww, r_t) / w_sum
            den_t = np.sqrt(np.dot(ww, r_t ** 2)) + 1e-30
            out[i] = float(np.dot(ww, r_t * r_tr) / (den_t * den_tr))
        return out

    all_idx = np.arange(n_pix)
    rho = _spearman(all_idx)

    if patch_ids is not None:
        patch_ids = np.asarray(patch_ids)
        labels = np.unique(patch_ids)
        k = labels.size
        if k >= 2:
            jk = np.array([_spearman(all_idx[patch_ids != lab]) for lab in labels])
            sigma = np.sqrt((k - 1) / k * np.sum((jk - jk.mean(axis=0)) ** 2, axis=0))
            jackknife = True
        else:
            warnings.warn(
                "patch_ids defines fewer than two patches; falling back to the "
                "Gaussian error, which ignores the correlation of the field and "
                "will overstate every significance.",
                UserWarning, stacklevel=2,
            )
            sigma = np.full(n_sys, 1.0 / np.sqrt(max(n_pix - 3, 1)))
            jackknife = False
    else:
        sigma = np.full(n_sys, 1.0 / np.sqrt(max(n_pix - 3, 1)))
        jackknife = False

    return {
        "rho": rho,
        "sigma": sigma,
        "significance": np.abs(rho) / (sigma + 1e-30),
        "jackknife": jackknife,
        "names": template_names,
    }


def isd_marginal_fit(
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    *,
    n_bins: int = 10,
    poly_order: int = 1,
    fracdet: np.ndarray | None = None,
    binning: str = "quantile",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marginal (one-template-at-a-time) binned polynomial fit of the density.

    This is the fit that Iterative Systematics Decontamination actually performs:
    for each template independently, bin the footprint by that template's value,
    take the mean overdensity per bin, and fit a degree-``poly_order`` polynomial
    to the binned relation.  Nothing is fitted jointly and no cross-products
    between templates are formed, so the design matrix is
    ``(n_bins, poly_order+1)`` regardless of how many templates there are.

    :func:`snr_template_ranking` with ``method="isd"`` returns only the
    :math:`\Delta\chi^2` of this fit; this function additionally returns the
    fitted coefficients, which is what
    :func:`~sys_mapping.regression.iterative_systematics_decontamination` needs to
    build the intermediate weight.

    Parameters
    ----------
    delta_g_obs:
        Observed galaxy overdensity at footprint pixels (shape ``(n_pix,)``).
    delta_t:
        Template maps at footprint pixels (shape ``(n_sys, n_pix)``).
    n_bins:
        Number of template-value bins.
    poly_order:
        Degree of the 1D polynomial in the template value.  ``1`` is the DES Y1/Y3
        choice, ``3`` the DES Y6 choice.  Note that this is the order *in a single
        template's value*, not a multivariate polynomial order.
    fracdet:
        Per-pixel fractional coverage weights (shape ``(n_pix,)``); uniform if
        ``None``.
    binning:
        ``"quantile"`` (default, equal occupancy) or ``"width"`` (equal width).
        Equal occupancy is the robust choice for skewed templates, where
        equal-width bins collapse most pixels into a single bin.

    Returns
    -------
    delta_chi2 : ``(n_sys,)``
        :math:`\\chi^2_{\\rm null} - \\chi^2_{\\rm model}` per template, clipped at
        zero.  Zero for a template whose binned relation has fewer than two usable
        bins.
    coeffs : ``(n_sys, poly_order + 1)``
        Fitted polynomial coefficients in **ascending** power order, so that
        ``F_i(t) = sum_k coeffs[i, k] * t**k``.  All zero for a template that
        could not be fitted.
    t_range : ``(n_sys, 2)``
        The outermost valid bin centres, ``(t_lo, t_hi)`` per template.  The fit
        is supported only here; evaluating it outside is extrapolation, and for a
        cubic on a skewed template that is not a small effect.  Callers should
        clip the template value into this range before evaluating ``coeffs``.

    Precision
    ---------
    The JAX path solves the sqrt-weighted Vandermonde system rather than the
    normal equations, so the condition number is not squared.  It agrees with the
    NumPy fallback to ``~1e-12`` on well-conditioned inputs.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping.diagnostics import isd_marginal_fit
    >>> rng = np.random.default_rng(0)
    >>> n_pix = 20000
    >>> t = rng.standard_normal((2, n_pix))
    >>> g = 0.3 * t[0] + rng.standard_normal(n_pix) * 0.05
    >>> dchi2, coeffs, t_range = isd_marginal_fit(g, t, poly_order=1)
    >>> bool(dchi2[0] > dchi2[1])
    True
    >>> bool(abs(coeffs[0, 1] - 0.3) < 0.05)
    True
    >>> bool(t_range[0, 0] < 0 < t_range[0, 1])
    True

    References
    ----------
    Elvin-Poole et al. 2018, PRD 98, 042006 (DES Y1).
    Rodriguez-Monroy et al. 2022, MNRAS 511, 2665 (DES Y3).
    Weaverdyck et al. 2026, arXiv:2601.14484, Sec. III B (DES Y6, cubic fits).
    """
    if binning not in ("width", "quantile", "equal_occupancy"):
        raise ValueError("binning must be 'width', 'quantile', or 'equal_occupancy'")
    quantile = binning in ("quantile", "equal_occupancy")

    delta_g_obs = np.asarray(delta_g_obs, dtype=float)
    delta_t = np.atleast_2d(np.asarray(delta_t, dtype=float))
    n_sys, n_pix = delta_t.shape
    order = int(poly_order)

    if _JAX_AVAILABLE:
        w = (jnp.ones(n_pix, dtype=jnp.float64) if fracdet is None
             else jnp.asarray(fracdet, dtype=jnp.float64))
        dchi2, coeffs, t_lo, t_hi = _get_jax_isd_poly_full(n_bins, order, quantile)(
            jnp.asarray(delta_t, dtype=jnp.float64),
            jnp.asarray(delta_g_obs, dtype=jnp.float64),
            w,
        )
        return (np.asarray(dchi2), np.asarray(coeffs),
                np.stack([np.asarray(t_lo), np.asarray(t_hi)], axis=1))

    # NumPy fallback (JAX unavailable)
    w = np.ones(n_pix, dtype=float) if fracdet is None else np.asarray(fracdet, dtype=float)
    w_total = float(np.sum(w))
    g_bar = float(np.dot(w, delta_g_obs) / w_total) if w_total > 1e-30 else 0.0

    dchi2 = np.zeros(n_sys)
    coeffs = np.zeros((n_sys, order + 1))
    t_range = np.zeros((n_sys, 2))
    for i, t_i in enumerate(delta_t):
        t_min, t_max = float(t_i.min()), float(t_i.max())
        if t_min >= t_max:
            continue

        if quantile:
            edges = np.quantile(t_i, np.linspace(0.0, 1.0, n_bins + 1))
            bin_idx = np.clip(np.searchsorted(edges[1:-1], t_i, side="right"),
                              0, n_bins - 1)
        else:
            bin_idx = np.clip(
                np.floor((t_i - t_min) / (t_max - t_min) * n_bins).astype(int),
                0, n_bins - 1)

        s_b, n_b, sig_b = [], [], []
        for b in range(n_bins):
            m = bin_idx == b
            if not np.any(m):
                continue
            w_b = w[m]
            w_b_sum = float(np.sum(w_b))
            if w_b_sum < 1e-30:
                continue
            std_g = float(np.std(delta_g_obs[m]))
            if std_g < 1e-10:
                continue
            s_b.append(float(np.dot(w_b, t_i[m]) / w_b_sum))
            n_b.append(float(np.dot(w_b, delta_g_obs[m]) / w_b_sum))
            sig_b.append(std_g / np.sqrt(int(np.sum(m))))

        if len(s_b) < 2:
            continue

        s_arr, n_arr, sig_arr = np.array(s_b), np.array(n_b), np.array(sig_b)
        inv_s2 = 1.0 / sig_arr ** 2
        chi2_null = float(np.dot((n_arr - g_bar) ** 2, inv_s2))

        eff_order = min(order, len(s_arr) - 1)
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore")
            c_desc = np.polyfit(s_arr, n_arr, eff_order, w=1.0 / sig_arr)
        f_s = np.polyval(c_desc, s_arr)
        chi2_model = float(np.dot((n_arr - f_s) ** 2, inv_s2))

        dchi2[i] = max(chi2_null - chi2_model, 0.0)
        coeffs[i, : eff_order + 1] = c_desc[::-1]   # ascending powers
        t_range[i] = (s_arr.min(), s_arr.max())

    return dchi2, coeffs, t_range


def footprint_mask_diagnostics(
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    mask_fractions: np.ndarray,
    good_pixels: np.ndarray | None = None,
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
        Unused, accepted for backward compatibility only.  ``delta_g_obs`` and
        ``delta_t`` are already restricted to the fitted pixels, and the
        pixel-quality ranking is computed from ``delta_t`` itself, so a
        full-footprint mask has nothing to contribute here.  Passing one warns.

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
    if good_pixels is not None:
        warnings.warn(
            "footprint_mask_diagnostics ignores good_pixels: delta_g_obs and "
            "delta_t are already restricted to the fitted pixels and the "
            "pixel-quality ranking comes from delta_t. The argument is accepted "
            "for backward compatibility and will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )

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


def isd_template_significance(
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    good_pixels: np.ndarray,
    nside: int,
    n_total: int,
    z_edges: np.ndarray,
    nz: np.ndarray,
    *,
    n_total_footprint: int | None = None,
    n_mocks: int = 100,
    n_bins: int = 10,
    poly_order: int = 1,
    fracdet: np.ndarray | None = None,
    seed: int = 0,
    rand_factor: int = 10,
    n_jobs: int = 1,
    binning: str = "width",
    cl_amplitude: float = 5e-4,
    cl_input: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """ISD Δχ² significance: compare data against systematic-free GLASS mocks.

    .. warning::
       Prefer ``cl_input``.  A mock null is only a null for the sample, resolution
       and footprint it was matched to: what has to agree is the *large-scale*
       (two-halo) clustering, since that is the scale systematic templates vary
       on, and one scalar cannot set the shape of a spectrum.  Match it per setup
       with ``match_glass_to_data.py`` (sys_mapping_benchmark) and load the result
       with :func:`~sys_mapping.glass_mocks.load_matched_cl`.  There is no
       universal value.

       ``cl_amplitude`` is the parametric fallback and sets the clustering power
       of the null through a fixed-slope power law.  The default
       ``5e-4`` reproduces the data's *surface density* (hence its shot noise) but
       gives :math:`\\sigma_{\\rm clus} \\approx 0.08` against LS10's
       :math:`\\approx 0.39` --- 25x too little clustering variance.  A null that
       under-clusters is too narrow, so the p-values it yields remain
       anticonservative.  Fit the amplitude to the sample's measured
       :math:`\\hat\\sigma` (see ``calibrate_glass_clustering.py`` in the
       sys_mapping_benchmark repository) and pass it here.

    For each template map, computes the ISD contamination metric
    :math:`\\Delta\\chi^2 = \\chi^2_{\\rm null} - \\chi^2_{\\rm model}` on the
    data and on ``n_mocks`` systematic-free GLASS mocks generated on the same
    footprint and redshift range.  The fraction of mocks that exceed the data
    value defines the template's p-value.

    Parameters
    ----------
    delta_g_obs:
        Observed galaxy overdensity at footprint pixels (shape ``(n_good_pix,)``).
    delta_t:
        Template maps at footprint pixels (shape ``(n_sys, n_good_pix)``).
    good_pixels:
        Boolean footprint mask over the full HEALPix sky
        (shape ``(12 * nside**2,)``).
    nside:
        HEALPix resolution of the template and galaxy maps.
    n_total:
        Target galaxy count per GLASS mock (full-sky).  Ignored when
        ``n_total_footprint`` is provided.
    n_total_footprint:
        Number of galaxies in the survey footprint (i.e. ``len(ra_gal)`` from
        the real data).  When provided, ``n_total`` is computed automatically
        as ``n_total_footprint × n_full_pix / n_good_pix`` so that after
        trimming the mock to ``good_pixels``, the footprint surface density
        matches the data.  Preferred over passing a raw ``n_total``.
    z_edges:
        Redshift bin edges for the n(z) model (length ``n_bins_z + 1``).
    nz:
        Galaxy counts per redshift bin (length ``n_bins_z``).
    n_mocks:
        Number of systematic-free mocks to generate.
    n_bins:
        Number of template-value bins for the ISD 1D relation.
    poly_order:
        Polynomial degree for the 1D fit (1 = linear, 3 = cubic).
    fracdet:
        Per-pixel fractional coverage weights (shape ``(n_good_pix,)``).
        If ``None``, uniform weights are used.
    seed:
        Base random seed; mock *k* uses ``seed + k``.
    rand_factor:
        Ratio of random to galaxy counts in each GLASS mock.  Reducing this
        (e.g. to 2) cuts memory usage and generation time proportionally with
        negligible effect on the overdensity estimate.  Default is 10.
    n_jobs:
        Number of parallel worker processes for the (embarrassingly parallel,
        GLASS/healpy-bound) mock loop.  ``1`` (default) runs serially; ``-1``
        uses all cores.  Mock ``k`` always uses ``seed + k``, so results are
        reproducible regardless of ``n_jobs``.

    Returns
    -------
    result : dict with keys

        - ``"delta_chi2"`` — shape ``(n_sys,)``, Δχ² for the data.
        - ``"p_values"`` — shape ``(n_sys,)``, mock-based p-values in ``(0, 1]``.
          Small values indicate significant systematic contamination.
        - ``"delta_chi2_mocks"`` — shape ``(n_mocks, n_sys)``, Δχ² distribution
          from systematic-free mocks.

    Notes
    -----
    Mocks are generated with :func:`~glass_mocks.generate_glass_fullsky_mock`,
    pixelised with :func:`~maps.pixelize_catalog`, and restricted to the survey
    footprint via ``good_pixels``.  The same template maps ``delta_t`` are used
    for both the data and the mocks.

    p-value formula (smoothed to avoid zero):

    .. math::

        p_i = \\frac{\\#\\{k : \\Delta\\chi^2_{\\rm mock,k,i} \\ge
              \\Delta\\chi^2_{\\rm data,i}\\} + 1}{n_{\\rm mocks} + 1}

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import isd_template_significance
    >>> rng = np.random.default_rng(0)
    >>> n_pix, n_sys = 3072, 3   # nside=16 footprint size
    >>> delta_t = rng.standard_normal((n_sys, n_pix))
    >>> delta_g = 0.4 * delta_t[1] + rng.standard_normal(n_pix) * 0.1
    >>> good = np.ones(12 * 16 ** 2, dtype=bool)
    >>> z_edges = np.array([0.0, 0.5])
    >>> nz = np.array([1000.0])
    >>> result = isd_template_significance(
    ...     delta_g, delta_t, good, nside=16, n_total=3000,
    ...     z_edges=z_edges, nz=nz, n_mocks=5, seed=0)
    >>> set(result.keys()) == {"delta_chi2", "p_values", "delta_chi2_mocks"}
    True
    >>> result["delta_chi2"].shape
    (3,)
    >>> result["p_values"].shape
    (3,)
    >>> result["delta_chi2_mocks"].shape
    (5, 3)

    References
    ----------
    Rodríguez-Monroy et al. 2025, arXiv:2509.07943, Sec. IV.A.1.
    Tessore et al. 2023, OJAp, 6, 11.
    """
    import healpy as hp
    from .glass_mocks import generate_glass_fullsky_mock
    from .maps import pixelize_catalog

    n_sys = delta_t.shape[0]

    if n_total_footprint is not None:
        n_full_pix = hp.nside2npix(nside)
        n_good_pix = int(good_pixels.sum())
        n_total = int(n_total_footprint * n_full_pix / n_good_pix)

    delta_chi2_data = snr_template_ranking(
        delta_g_obs, delta_t, method="isd",
        n_bins=n_bins, poly_order=poly_order, fracdet=fracdet, binning=binning,
    )

    def _one_mock(k: int) -> np.ndarray:
        """Δχ² of a single systematic-free GLASS mock (mock ``k`` uses seed+k)."""
        cat = generate_glass_fullsky_mock(
            nside, n_total, z_edges, nz, seed=seed + k,
            rand_factor=rand_factor, cl_amplitude=cl_amplitude,
            cl_input=cl_input,
        )
        n_gal_full = pixelize_catalog(cat["ra"], cat["dec"], nside)
        n_rand_full = pixelize_catalog(cat["ra_rand"], cat["dec_rand"], nside)

        n_gal_foot = n_gal_full[good_pixels].astype(float)
        n_rand_foot = n_rand_full[good_pixels].astype(float)

        total_gal = float(np.sum(n_gal_foot))
        total_rand = float(np.sum(n_rand_foot))
        if total_gal < 1 or total_rand < 1:
            return np.zeros(n_sys)

        norm = total_gal / total_rand
        n_rand_safe = np.where(n_rand_foot > 0, n_rand_foot, 1.0)
        delta_g_mock = np.where(
            n_rand_foot > 0,
            n_gal_foot / (norm * n_rand_safe) - 1.0,
            0.0,
        )
        return snr_template_ranking(
            delta_g_mock, delta_t, method="isd",
            n_bins=n_bins, poly_order=poly_order, fracdet=fracdet, binning=binning,
        )

    if n_jobs == 1:
        delta_chi2_mocks = np.array([_one_mock(k) for k in range(n_mocks)])
    else:
        from joblib import Parallel, delayed
        delta_chi2_mocks = np.array(
            Parallel(n_jobs=n_jobs)(delayed(_one_mock)(k) for k in range(n_mocks))
        )

    n_extreme = np.sum(delta_chi2_mocks >= delta_chi2_data[np.newaxis, :], axis=0)
    p_values = (n_extreme + 1.0) / (n_mocks + 1.0)

    return {
        "delta_chi2": delta_chi2_data,
        "p_values": p_values,
        "delta_chi2_mocks": delta_chi2_mocks,
    }
