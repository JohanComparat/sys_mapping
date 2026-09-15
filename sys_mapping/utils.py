"""Utility functions for covariance estimation and two-point measurements.

Implements:
  - compute_covariance_matrix (Eq. 24): template auto-correlations
  - compute_amplitude_bias (Eq. 25-26): expected bias on clustering amplitude
  - measure_two_point_function: TreeCorr Landy-Szalay estimator wrapper
  - measure_two_point_function_corrfunc: Corrfunc Landy-Szalay estimator wrapper
  - measure_kk_correlation_treecorr: TreeCorr KK scalar-field correlator
  - measure_kk_correlation_corrfunc: Corrfunc KK scalar-field correlator
"""

from __future__ import annotations

import warnings

import numpy as np


def compute_covariance_matrix(delta_t: np.ndarray) -> np.ndarray:
    """Compute the template covariance matrix (Eq. 24).

    ``C_{ij} = (1/N_pix) Σ_p δ_{ti,p} δ_{tj,p}``

    .. note::
       This is the **uncentred second moment**, not a covariance in general.  It
       coincides with one because
       :func:`~sys_mapping.maps.load_real_template` standardises every template to
       zero mean over the footprint before any fit, so ``⟨δ_t⟩ = 0`` by
       construction.  It is a covariance *because of that invariant*, not because
       the mean is subtracted here.  Passing an un-standardised template gives a
       matrix dominated by the product of the means, and no error.

    Parameters
    ----------
    delta_t : ``(n_sys, n_pix)`` template values at unmasked pixels

    Returns
    -------
    (n_sys, n_sys) symmetric positive-semi-definite covariance matrix

    Performance
    -----------
    Measured on CPU (n_sys=5, n_pix=10_000): **~120 μs/call**.
    Uses ``numpy`` matrix multiply; scales as O(n_sys² × n_pix).

    Precision
    ---------
    Matrix is symmetric to ``atol < 1e-14`` (exact in float64).
    All eigenvalues ≥ −1e-12 (positive semi-definite).
    For unit-variance uncorrelated templates, result is close to the
    identity matrix (deviation < 0.01 for n_pix ≥ 200_000).

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import compute_covariance_matrix
    >>> rng = np.random.default_rng(0)
    >>> delta_t = rng.standard_normal((3, 10_000))
    >>> C = compute_covariance_matrix(delta_t)
    >>> C.shape
    (3, 3)
    >>> np.allclose(C, C.T)  # symmetric
    True
    """
    delta_t = np.asarray(delta_t)
    n_pix = delta_t.shape[1]
    means = delta_t.mean(axis=1)
    scale = np.sqrt(np.mean(delta_t ** 2, axis=1))
    scale = np.where(scale > 0, scale, 1.0)
    # The sample mean of n_pix unit-variance values has standard error
    # 1/sqrt(n_pix), so a fixed relative threshold fires on every finite draw of
    # a genuinely centred basis.  Flag a mean that is large against that.
    mean_tol = max(1e-3, 5.0 / np.sqrt(n_pix))
    if np.any(np.abs(means) / scale > mean_tol):
        warnings.warn(
            "compute_covariance_matrix is an uncentred second moment and assumes "
            "templates standardised to zero mean; the input has a relative mean of "
            f"{float(np.max(np.abs(means) / scale)):.3g} against a tolerance of "
            f"{mean_tol:.3g}. Standardise first (see "
            "maps.load_real_template) or the result is not a covariance.",
            RuntimeWarning,
            stacklevel=2,
        )
    # A basis standardised somewhere other than where it is used is not
    # standardised.  Survey-property maps are normalised over their own valid
    # region, which is larger than any one sample's footprint; restricted to the
    # footprint their variances drift, and every quantity that reads the basis in
    # "standardised units" -- the amplitudes, the condition number, the template
    # auto-correlations the two-point correction subtracts -- inherits the drift.
    if np.any((scale < 0.5) | (scale > 2.0)):
        worst = int(np.argmax(np.abs(np.log(scale))))
        warnings.warn(
            "templates are not standardised over the pixels supplied: rms ranges "
            f"{float(scale.min()):.3g} to {float(scale.max()):.3g} (template "
            f"{worst}). Amplitudes fitted against this basis are not in units of "
            "one standard deviation of the template on this footprint, and the "
            "sum of the covariance eigenvalues is "
            f"{float(np.sum(scale ** 2)):.4g} rather than {len(scale)}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return (delta_t @ delta_t.T) / n_pix


def compute_amplitude_bias(
    a_sq: np.ndarray,
    b_sq: np.ndarray,
    cov_matrix: np.ndarray,
) -> tuple[float, float]:
    """Estimate bias on the galaxy clustering amplitude (Eq. 25-26).

    Returns the additive offset and multiplicative factor that contamination
    imprints on the angle-averaged angular power spectrum.

    Additive bias:   ``Δw = Σ_i a²_i C_{ii}``
    Multiplicative:  ``f  = 1 + Σ_i b²_i C_{ii}``

    Parameters
    ----------
    a_sq : ``(n_sys,)`` debiased squared additive parameters
    b_sq : ``(n_sys,)`` debiased squared multiplicative parameters
    cov_matrix : ``(n_sys, n_sys)`` template covariance from :func:`compute_covariance_matrix`

    Returns
    -------
    additive_bias : float  Δw added to the measured correlation function
    mult_factor   : float  factor by which w is inflated (1 = no bias)

    Performance
    -----------
    Measured on CPU (n_sys=5): **~11 μs/call**. Two dot products.

    Precision
    ---------
    Matches analytic formula to ``rtol < 1e-10`` (float64 dot products).

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import compute_covariance_matrix, compute_amplitude_bias
    >>> rng = np.random.default_rng(0)
    >>> delta_t = rng.standard_normal((3, 10_000))
    >>> C = compute_covariance_matrix(delta_t)
    >>> a_sq = np.array([0.01, 0.02, 0.005])
    >>> b_sq = np.array([0.005, 0.01, 0.002])
    >>> add_bias, mult_factor = compute_amplitude_bias(a_sq, b_sq, C)
    >>> round(add_bias, 4)     # additive offset on w(θ)
    0.0347
    >>> round(mult_factor, 3)  # multiplicative factor, close to 1
    1.017
    """
    diag = np.diag(cov_matrix)
    additive_bias = float(np.dot(a_sq, diag))
    mult_factor = float(1.0 + np.dot(b_sq, diag))
    return additive_bias, mult_factor


def measure_two_point_function(
    ra_gal: np.ndarray,
    dec_gal: np.ndarray,
    ra_rand: np.ndarray,
    dec_rand: np.ndarray,
    min_sep: float = 0.06,
    max_sep: float = 30.0,
    nbins: int = 15,
    sep_units: str = "arcmin",
    metric: str = "Arc",
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the angular two-point correlation function via Landy-Szalay.

    Uses TreeCorr with log-spaced angular bins and the standard estimator
    ``w(θ) = (DD − 2DR + RR) / RR``.

    Parameters
    ----------
    ra_gal, dec_gal : (n_gal,) galaxy positions in degrees
    ra_rand, dec_rand : (n_rand,) random positions in degrees
    min_sep, max_sep : float  angular bin limits in ``sep_units``
    nbins : int  number of log-spaced bins
    sep_units : str  angular units for ``min_sep``/``max_sep`` (default ``'arcmin'``)
    metric : str  TreeCorr metric (default ``'Arc'``, great-circle separation).
        Matches :func:`measure_cross_two_point_function`; TreeCorr's own
        default is ``'Euclidean'``, a different distance measure.

    Returns
    -------
    theta : (nbins,) bin centers in ``sep_units``
    w : (nbins,) Landy-Szalay angular correlation function estimate

    Performance
    -----------
    Dominated by TreeCorr pair-counting; scales as O(n_gal log n_gal).
    Typical wall time: **seconds to minutes** for survey-scale catalogs.

    Precision
    ---------
    Landy-Szalay estimator is unbiased to O(1/N). Statistical error is
    approximately ``1/sqrt(DD_pairs)`` per bin.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import measure_two_point_function
    >>> rng = np.random.default_rng(0)
    >>> ra_g   = rng.uniform(30, 60, 10_000)  # small patch
    >>> dec_g  = rng.uniform(-10, 10, 10_000)
    >>> ra_r   = rng.uniform(30, 60, 50_000)
    >>> dec_r  = rng.uniform(-10, 10, 50_000)
    >>> theta, w = measure_two_point_function(
    ...     ra_g, dec_g, ra_r, dec_r,
    ...     min_sep=1.0, max_sep=100.0, nbins=10, sep_units="arcmin",
    ... )
    >>> theta.shape, w.shape
    ((10,), (10,))
    """
    try:
        import treecorr
    except ImportError as e:
        raise ImportError("treecorr is required for two-point measurements.") from e

    config = dict(
        min_sep=min_sep,
        max_sep=max_sep,
        nbins=nbins,
        sep_units=sep_units,
        # Great-circle separation, matching measure_cross_two_point_function.
        # TreeCorr's own default is "Euclidean" (chord), so leaving this unset put
        # the auto- and cross-correlation on different distance measures: the two
        # diverge as theta^3/24 and are not comparable bin for bin at degree
        # scales, which is exactly where w(theta) is measured here.
        metric=metric,
    )
    cat_gal = treecorr.Catalog(ra=ra_gal, dec=dec_gal, ra_units="degrees", dec_units="degrees")
    cat_rand = treecorr.Catalog(ra=ra_rand, dec=dec_rand, ra_units="degrees", dec_units="degrees")

    dd = treecorr.NNCorrelation(**config)
    rr = treecorr.NNCorrelation(**config)
    dr = treecorr.NNCorrelation(**config)

    dd.process(cat_gal)
    rr.process(cat_rand)
    dr.process(cat_gal, cat_rand)

    w, varxi = dd.calculateXi(rr=rr, dr=dr)
    theta = np.exp(dd.meanlogr)
    return theta, w


def measure_cross_two_point_function(
    ra1: np.ndarray,
    dec1: np.ndarray,
    ra2: np.ndarray,
    dec2: np.ndarray,
    ra_rand: np.ndarray,
    dec_rand: np.ndarray,
    w1: np.ndarray | None = None,
    w2: np.ndarray | None = None,
    min_sep: float = 0.06,
    max_sep: float = 30.0,
    nbins: int = 15,
    sep_units: str = "arcmin",
    metric: str = "Arc",
    bin_slop: float | None = None,
    dr=None,
    rr=None,
):
    """Measure the angular *cross*-correlation of two point samples (Landy-Szalay).

    Estimates ``w_x(θ)`` between sample 1 (e.g. galaxies) and sample 2 (e.g.
    stars) that share a common footprint, using a single random catalog drawn
    uniformly over that footprint for both samples::

        w_x(θ) = (D1D2 − D1R − R D2 + RR) / RR

    computed with TreeCorr's :meth:`NNCorrelation.calculateXi(rr, dr, rd)` where
    ``dr = D1R`` and ``rd = R D2``.  Unlike
    :func:`measure_two_point_function` (auto-correlation), this counts distinct
    ``D1×D2`` pairs, so it is the correct estimator for a galaxy×star signal.

    The ``D1R`` (``dr``) and ``RR`` (``rr``) pair counts depend only on sample 1
    and the randoms — not on sample 2 — so when scanning many sample-2 catalogs
    (e.g. magnitude-binned star samples) against a *fixed* sample 1, pass the
    returned ``dr``/``rr`` back in to skip recomputing them.

    Parameters
    ----------
    ra1, dec1 : (n1,) sample-1 positions in degrees
    ra2, dec2 : (n2,) sample-2 positions in degrees
    ra_rand, dec_rand : (n_rand,) random positions in degrees (common footprint)
    w1, w2 : optional per-object weights for samples 1 and 2
    min_sep, max_sep : float  angular bin limits in ``sep_units``
    nbins : int  number of log-spaced bins
    sep_units : str  angular units for ``min_sep``/``max_sep`` (default ``'arcmin'``)
    metric : str  TreeCorr metric (default ``'Arc'``, great-circle separation)
    bin_slop : float or None  TreeCorr ``bin_slop`` (None → TreeCorr default)
    dr, rr : optional pre-computed ``NNCorrelation`` objects (``D1R`` and ``RR``)
        to reuse across many sample-2 catalogs sharing the same sample 1 and
        randoms.  When ``None`` they are computed and returned.

    Returns
    -------
    theta : (nbins,) bin centers in ``sep_units``
    w : (nbins,) Landy-Szalay cross-correlation estimate
    varxi : (nbins,) TreeCorr shot-noise variance per bin
    dr : NNCorrelation  the ``D1R`` counts (reusable across sample-2 catalogs)
    rr : NNCorrelation  the ``RR`` counts (reusable across sample-2 catalogs)

    Notes
    -----
    For two statistically independent samples over the same footprint the
    estimator is consistent with ``w_x(θ) = 0`` within shot noise — the basis
    for the GLASS null test.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import measure_cross_two_point_function
    >>> rng = np.random.default_rng(0)
    >>> ra1  = rng.uniform(30, 60, 5_000);  dec1 = rng.uniform(-10, 10, 5_000)
    >>> ra2  = rng.uniform(30, 60, 5_000);  dec2 = rng.uniform(-10, 10, 5_000)
    >>> ra_r = rng.uniform(30, 60, 50_000); dec_r = rng.uniform(-10, 10, 50_000)
    >>> theta, w, var, dr, rr = measure_cross_two_point_function(
    ...     ra1, dec1, ra2, dec2, ra_r, dec_r,
    ...     min_sep=1.0, max_sep=100.0, nbins=8, sep_units="arcmin")
    >>> theta.shape, w.shape
    ((8,), (8,))
    """
    try:
        import treecorr
    except ImportError as e:
        raise ImportError("treecorr is required for two-point measurements.") from e

    config = dict(
        min_sep=min_sep,
        max_sep=max_sep,
        nbins=nbins,
        sep_units=sep_units,
        metric=metric,
    )
    if bin_slop is not None:
        config["bin_slop"] = bin_slop

    cat1 = treecorr.Catalog(ra=ra1, dec=dec1, w=w1, ra_units="degrees", dec_units="degrees")
    cat2 = treecorr.Catalog(ra=ra2, dec=dec2, w=w2, ra_units="degrees", dec_units="degrees")
    cat_rand = treecorr.Catalog(ra=ra_rand, dec=dec_rand, ra_units="degrees", dec_units="degrees")

    d1d2 = treecorr.NNCorrelation(**config)
    rd = treecorr.NNCorrelation(**config)
    d1d2.process(cat1, cat2)   # D1 D2  (depends on sample 2)
    rd.process(cat_rand, cat2)  # R  D2  (depends on sample 2)

    if dr is None:
        dr = treecorr.NNCorrelation(**config)
        dr.process(cat1, cat_rand)   # D1 R  (independent of sample 2)
    if rr is None:
        rr = treecorr.NNCorrelation(**config)
        rr.process(cat_rand)         # R  R  (independent of sample 2)

    w, varxi = d1d2.calculateXi(rr=rr, dr=dr, rd=rd)
    theta = np.exp(d1d2.meanlogr)
    return theta, w, varxi, dr, rr


# ---------------------------------------------------------------------------
# Corrfunc implementations
# ---------------------------------------------------------------------------

_SEP_TO_DEG: dict[str, float] = {
    "degrees": 1.0,
    "arcmin": 1.0 / 60.0,
    "arcsec": 1.0 / 3600.0,
    "radians": 180.0 / np.pi,
}


def measure_two_point_function_corrfunc(
    ra_gal: np.ndarray,
    dec_gal: np.ndarray,
    ra_rand: np.ndarray,
    dec_rand: np.ndarray,
    min_sep: float = 0.06,
    max_sep: float = 30.0,
    nbins: int = 15,
    sep_units: str = "arcmin",
    nthreads: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the angular two-point correlation function via Landy-Szalay.

    Drop-in replacement for :func:`measure_two_point_function` using
    Corrfunc instead of TreeCorr.  Pair-counting is exact (no bin_slop).

    Parameters
    ----------
    ra_gal, dec_gal : (n_gal,) galaxy positions in degrees
    ra_rand, dec_rand : (n_rand,) random positions in degrees
    min_sep, max_sep : float  angular bin limits in ``sep_units``
    nbins : int  number of log-spaced bins
    sep_units : str  angular units (default ``'arcmin'``)
    nthreads : int  OpenMP threads passed to Corrfunc (default 1)

    Returns
    -------
    theta : (nbins,) mean pair separation in ``sep_units``
    w : (nbins,) Landy-Szalay angular correlation function estimate
    """
    try:
        from Corrfunc.mocks import DDtheta_mocks
    except ImportError as e:
        raise ImportError("Corrfunc is required for measure_two_point_function_corrfunc.") from e

    factor = _SEP_TO_DEG[sep_units]
    bins = np.logspace(np.log10(min_sep * factor), np.log10(max_sep * factor), nbins + 1)

    n_gal = len(ra_gal)
    n_rand = len(ra_rand)

    ra_gal = np.asarray(ra_gal, dtype=np.float64)
    dec_gal = np.asarray(dec_gal, dtype=np.float64)
    ra_rand = np.asarray(ra_rand, dtype=np.float64)
    dec_rand = np.asarray(dec_rand, dtype=np.float64)

    DD = DDtheta_mocks(1, nthreads, bins, ra_gal, dec_gal, output_thetaavg=True)
    RR = DDtheta_mocks(1, nthreads, bins, ra_rand, dec_rand)
    DR = DDtheta_mocks(0, nthreads, bins, ra_gal, dec_gal, RA2=ra_rand, DEC2=dec_rand)

    # Corrfunc autocorr=1 counts ordered unique pairs n*(n-1), not n*(n-1)/2.
    # DR cross-correlation counts all n_gal*n_rand unordered pairs.
    dd = DD["npairs"] / (n_gal * (n_gal - 1))
    rr = RR["npairs"] / (n_rand * (n_rand - 1))
    dr = DR["npairs"] / (n_gal * float(n_rand))

    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(rr > 0, (dd - 2.0 * dr + rr) / rr, 0.0)

    # Use measured mean separation; fall back to geometric bin centre
    theta_deg = np.where(DD["thetaavg"] > 0, DD["thetaavg"], np.sqrt(bins[:-1] * bins[1:]))
    theta = theta_deg / factor
    return theta, w


def measure_kk_correlation_treecorr(
    ra: np.ndarray,
    dec: np.ndarray,
    k: np.ndarray,
    w: np.ndarray | None = None,
    *,
    ra2: np.ndarray | None = None,
    dec2: np.ndarray | None = None,
    k2: np.ndarray | None = None,
    w2: np.ndarray | None = None,
    min_sep: float = 10.0,
    max_sep: float = 250.0,
    nbins: int = 20,
    sep_units: str = "arcmin",
    bin_slop: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Angular scalar-field correlation (KK) using TreeCorr.

    Computes ``xi(theta) = <k1_i k2_j>_w`` for auto- or cross-correlation.

    Parameters
    ----------
    ra, dec : (n,) positions in degrees
    k : (n,) scalar field values (e.g. galaxy overdensity)
    w : (n,) optional pair weights; ones if None
    ra2, dec2, k2, w2 : second catalog for cross-correlation; if None, auto-correlation
    min_sep, max_sep : float  bin limits in ``sep_units``
    nbins : int  number of log-spaced bins
    sep_units : str  angular units (default ``'arcmin'``)
    bin_slop : float  TreeCorr bin_slop parameter (default 0.01)

    Returns
    -------
    theta : (nbins,) bin centres in ``sep_units``
    xi : (nbins,) correlation function
    """
    try:
        import treecorr
    except ImportError as e:
        raise ImportError("treecorr is required for measure_kk_correlation_treecorr.") from e

    cfg = dict(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
               sep_units=sep_units, bin_slop=bin_slop)
    cat1 = treecorr.Catalog(ra=ra, dec=dec, ra_units="degrees", dec_units="degrees", k=k, w=w)
    kk = treecorr.KKCorrelation(**cfg)
    if ra2 is None:
        kk.process(cat1)
    else:
        cat2 = treecorr.Catalog(ra=ra2, dec=dec2, ra_units="degrees", dec_units="degrees",
                                 k=k2, w=w2)
        kk.process(cat1, cat2)
    return np.exp(kk.meanlogr), kk.xi


def template_correlation_matrix(
    ra: np.ndarray,
    dec: np.ndarray,
    k: np.ndarray,
    *,
    min_sep: float = 0.5,
    max_sep: float = 300.0,
    nbins: int = 30,
    sep_units: str = "arcmin",
    bin_slop: float = 0.01,
    max_points: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""The full template correlation matrix :math:`\xi_{ij}(\theta)`, auto and cross.

    The two-point correction subtracts :math:`\sum_{ij} \tilde A_{ij}\,\xi_{ij}(\theta)`.
    Keeping only the diagonal assumes the templates are uncorrelated at every
    separation, but the PCA rotation diagonalises their covariance at zero lag only,
    so :math:`\xi_{ij}(\theta) \ne 0` for :math:`\theta > 0` and the neglected terms
    can be a large share of the correction.  This measures all of them.

    Pass the template values each *galaxy* carries, not the pixel map: no two pixel
    centres are closer than the pixel scale, so a pixel-grid measurement has no
    support below it, while a template is constant within a pixel and its correct
    correlation there is its covariance.

    Parameters
    ----------
    ra, dec:
        Galaxy positions in degrees, shape ``(n,)``.
    k:
        Template values at those galaxies, shape ``(n_sys, n)``.
    min_sep, max_sep, nbins, sep_units, bin_slop:
        TreeCorr binning; match the measured :math:`w(\theta)`.
    max_points:
        Draw at most this many galaxies without replacement.  The templates are smooth,
        so the correlation is determined long before every galaxy is used, and the
        :math:`n_{\rm sys}(n_{\rm sys}+1)/2` correlations otherwise scale with the full
        catalogue.  ``None`` uses them all.
    seed:
        Seed for the subsample.

    Returns
    -------
    theta:
        ``(nbins,)`` mean separation per bin in ``sep_units``.
    xi:
        ``(n_sys, n_sys, nbins)``, symmetric in its first two axes.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import template_correlation_matrix
    >>> rng = np.random.default_rng(0)
    >>> ra = rng.uniform(0, 40, 20000)
    >>> dec = np.degrees(np.arcsin(rng.uniform(-0.3, 0.3, 20000)))
    >>> k = np.vstack([np.sin(np.radians(ra) * 8), np.cos(np.radians(ra) * 8)])
    >>> theta, xi = template_correlation_matrix(ra, dec, k, nbins=8, max_sep=120.0)
    >>> xi.shape
    (2, 2, 8)
    >>> bool(np.allclose(xi, xi.transpose(1, 0, 2)))
    True
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    k = np.atleast_2d(np.asarray(k, dtype=float))
    n_sys, n = k.shape
    if ra.shape != (n,) or dec.shape != (n,):
        raise ValueError(f"ra, dec must have shape ({n},) to match k; got {ra.shape}, {dec.shape}")
    if max_points is not None and n > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(n, int(max_points), replace=False))
        ra, dec, k = ra[idx], dec[idx], k[:, idx]

    cfg = dict(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
               sep_units=sep_units, bin_slop=bin_slop)
    xi = np.zeros((n_sys, n_sys, nbins))
    theta = None
    for i in range(n_sys):
        for j in range(i, n_sys):
            if i == j:
                theta, x = measure_kk_correlation_treecorr(ra, dec, k[i], **cfg)
            else:
                # Same positions in both catalogues: each pair is counted in both
                # orders, which leaves the mean unchanged, and the zero-separation
                # self-pairs fall below min_sep.
                theta, x = measure_kk_correlation_treecorr(
                    ra, dec, k[i], ra2=ra, dec2=dec, k2=k[j], **cfg)
            xi[i, j] = xi[j, i] = x
    return theta, xi


def measure_kk_covariance_treecorr(
    ra: np.ndarray,
    dec: np.ndarray,
    k: np.ndarray,
    w: np.ndarray | None = None,
    *,
    npatch: int = 50,
    patch_centers=None,
    min_sep: float = 10.0,
    max_sep: float = 250.0,
    nbins: int = 20,
    sep_units: str = "arcmin",
    bin_slop: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Angular scalar-field :math:`w(\\theta)` (KK) with a spatial jack-knife covariance.

    Same estimator as :func:`measure_kk_correlation_treecorr` — the pixel-overdensity KK the
    Euclid real-data pipeline uses — but the catalog is split into ``npatch`` spatial patches so
    TreeCorr's leave-one-patch-out jack-knife returns the **full** ``(nbins, nbins)`` covariance
    from a *single* realization.  This is the data-internal route to a :math:`w(\\theta)`
    covariance (no mocks needed); cross-check it against the mock-ensemble
    :func:`~sys_mapping.covariance.sample_covariance` and, in the validation, against the sim-to-sim
    ensemble scatter.

    Parameters
    ----------
    ra, dec : (n,) positions in degrees (e.g. footprint pixel centres)
    k : (n,) scalar field values (e.g. galaxy overdensity :math:`\\delta_g`)
    w : (n,) optional weights (e.g. coverage); ones if None
    npatch : int  number of spatial jack-knife patches (TreeCorr k-means on positions)
    patch_centers : optional  fixed patch centres (array or file) to reuse the *same* patches
        across catalogs (so data and mocks share a jack-knife geometry); overrides ``npatch``
    min_sep, max_sep, nbins, sep_units, bin_slop : TreeCorr KK binning (match the pipeline)

    Returns
    -------
    theta : (nbins,) bin centres in ``sep_units``
    xi : (nbins,) KK correlation
    cov : (nbins, nbins) jack-knife covariance
    patch_centers : (npatch, 3) the patch centres used — pass back in to match another catalog.

    Notes
    -----
    The jack-knife needs ``npatch`` comfortably larger than ``nbins`` for a well-conditioned,
    invertible covariance; ``npatch`` also sets the largest reliable scale (patches must be larger
    than ``max_sep``).
    """
    try:
        import treecorr
    except ImportError as e:
        raise ImportError("treecorr is required for measure_kk_covariance_treecorr.") from e

    cfg = dict(min_sep=min_sep, max_sep=max_sep, nbins=nbins,
               sep_units=sep_units, bin_slop=bin_slop, var_method="jackknife")
    cat_kwargs = dict(ra=ra, dec=dec, ra_units="degrees", dec_units="degrees", k=k, w=w)
    if patch_centers is not None:
        cat_kwargs["patch_centers"] = patch_centers
    else:
        cat_kwargs["npatch"] = npatch
    cat = treecorr.Catalog(**cat_kwargs)

    kk = treecorr.KKCorrelation(**cfg)
    kk.process(cat)
    theta = np.exp(kk.meanlogr)
    cov = np.asarray(kk.estimate_cov("jackknife"), dtype=np.float64)
    return theta, kk.xi, cov, cat.patch_centers


def measure_kk_correlation_corrfunc(
    ra: np.ndarray,
    dec: np.ndarray,
    k: np.ndarray,
    w: np.ndarray | None = None,
    *,
    ra2: np.ndarray | None = None,
    dec2: np.ndarray | None = None,
    k2: np.ndarray | None = None,
    w2: np.ndarray | None = None,
    min_sep: float = 10.0,
    max_sep: float = 250.0,
    nbins: int = 20,
    sep_units: str = "arcmin",
    nthreads: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Angular scalar-field correlation (KK) using Corrfunc.

    Drop-in replacement for :func:`measure_kk_correlation_treecorr`.
    Computes ``xi(theta) = sum(w_i w_j k_i k_j) / sum(w_i w_j)`` per bin
    via two weighted pair-count passes with ``weight_type='pair_product'``.

    Parameters
    ----------
    ra, dec : (n,) positions in degrees
    k : (n,) scalar field values
    w : (n,) optional pair weights; ones if None
    ra2, dec2, k2, w2 : second catalog for cross-correlation; if None, auto-correlation
    min_sep, max_sep : float  bin limits in ``sep_units``
    nbins : int  number of log-spaced bins
    sep_units : str  angular units (default ``'arcmin'``)
    nthreads : int  OpenMP threads (default 1)

    Returns
    -------
    theta : (nbins,) mean pair separation in ``sep_units``
    xi : (nbins,) correlation function
    """
    try:
        from Corrfunc.mocks import DDtheta_mocks
    except ImportError as e:
        raise ImportError("Corrfunc is required for measure_kk_correlation_corrfunc.") from e

    factor = _SEP_TO_DEG[sep_units]
    bins = np.logspace(np.log10(min_sep * factor), np.log10(max_sep * factor), nbins + 1)

    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    w1 = np.ones(len(ra), dtype=np.float64) if w is None else np.asarray(w, dtype=np.float64)

    cross = ra2 is not None
    if cross:
        ra2 = np.asarray(ra2, dtype=np.float64)
        dec2 = np.asarray(dec2, dtype=np.float64)
        k2 = np.asarray(k2, dtype=np.float64)
        w2 = (np.ones(len(ra2), dtype=np.float64) if w2 is None
              else np.asarray(w2, dtype=np.float64))

    autocorr = 0 if cross else 1
    kw = dict(weight_type="pair_product", output_thetaavg=True)

    if cross:
        res_num = DDtheta_mocks(autocorr, nthreads, bins, ra, dec,
                                RA2=ra2, DEC2=dec2,
                                weights1=w1 * k, weights2=w2 * k2, **kw)
        res_den = DDtheta_mocks(autocorr, nthreads, bins, ra, dec,
                                RA2=ra2, DEC2=dec2,
                                weights1=w1, weights2=w2, **kw)
    else:
        res_num = DDtheta_mocks(autocorr, nthreads, bins, ra, dec,
                                weights1=w1 * k, **kw)
        res_den = DDtheta_mocks(autocorr, nthreads, bins, ra, dec,
                                weights1=w1, **kw)

    # xi = sum(w_i w_j k_i k_j) / sum(w_i w_j)
    #    = weightavg_num / weightavg_den   (npairs cancels)
    with np.errstate(invalid="ignore", divide="ignore"):
        xi = np.where(res_den["weightavg"] > 0,
                      res_num["weightavg"] / res_den["weightavg"],
                      0.0)

    theta_deg = np.where(res_num["thetaavg"] > 0,
                         res_num["thetaavg"],
                         np.sqrt(bins[:-1] * bins[1:]))
    theta = theta_deg / factor
    return theta, xi
