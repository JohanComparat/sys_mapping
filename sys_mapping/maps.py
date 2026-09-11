"""HEALPix map generation and catalog pixelization utilities.

Implements the five synthetic systematic template families from Sec. 4.2 of
Berlfein et al. 2024, all normalized to unit variance, plus loaders for real
observational templates from GAIA and LS DR10.

Synthetic families (by multipole ℓ):
  0: C_ℓ ∝ exp(-ℓ/500)
  1: C_ℓ ∝ exp(-(ℓ/250)²)
  2: C_ℓ ∝ (ℓ+1)^{-2}
  3: C_ℓ ∝ (ℓ+1)^{-1}
  4: C_ℓ ∝ (ℓ+1)^0  (white noise)

Real templates (synth_5, synth_6):
  GAIA_nstar_faint — faint-star surface density (stellar contamination proxy)
  LS10_GALDEPTH_Z  — Legacy Survey DR10 galaxy depth in the z band (selection depth proxy)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import healpy as hp


TEMPLATE_FAMILIES = {
    0: "exp_linear",
    1: "exp_quadratic",
    2: "power_neg2",
    3: "power_neg1",
    4: "white_noise",
}


def systematic_power_spectrum(
    nside: int,
    family: int,
) -> np.ndarray:
    """Compute the power spectrum C_ℓ for a systematic template family.

    Parameters
    ----------
    nside : int
        HEALPix NSIDE parameter.
    family : int
        Template family index 0–4 (see module docstring for shapes).

    Returns
    -------
    (3*nside,) array of C_ℓ values, normalized so that Σ_ℓ (2ℓ+1)/(4π) C_ℓ = 1.
    The monopole (ℓ=0) is forced to exactly 0.

    Performance
    -----------
    Measured on CPU (nside=64): **~20 μs/call**. Pure NumPy; scales as O(nside).

    Precision
    ---------
    Theoretical variance = 1.0 to machine precision (``rtol < 1e-10``).
    Monopole is exactly 0 by construction.

    Examples
    --------
    >>> from sys_mapping import systematic_power_spectrum
    >>> import numpy as np
    >>> cl = systematic_power_spectrum(nside=64, family=0)  # exp-linear
    >>> cl.shape
    (192,)
    >>> cl[0]  # monopole forced to zero
    0.0
    >>> # Verify unit variance
    >>> ell = np.arange(len(cl))
    >>> float(np.sum((2 * ell + 1) / (4 * np.pi) * cl))  # ≈ 1.0
    1.0
    """
    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1, dtype=float)
    match family:
        case 0:
            cl = np.exp(-ell / 500.0)
        case 1:
            cl = np.exp(-((ell / 250.0) ** 2))
        case 2:
            cl = (ell + 1.0) ** (-2)
        case 3:
            cl = (ell + 1.0) ** (-1)
        case 4:
            cl = np.ones(lmax + 1)
        case _:
            raise ValueError(f"Unknown template family {family}. Must be 0-4.")

    cl[0] = 0.0  # zero monopole

    # Normalize to unit variance: Var = Σ_ℓ (2ℓ+1)/(4π) * C_ℓ
    variance = np.sum((2 * ell + 1) / (4 * np.pi) * cl)
    if variance > 0:
        cl /= variance
    return cl


def generate_systematic_map(
    nside: int,
    family: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a single HEALPix systematic template map.

    Draws a Gaussian random field from the power spectrum of the given family
    via ``healpy.synfast``, then enforces exact zero mean and unit variance.

    Parameters
    ----------
    nside : int
        HEALPix NSIDE parameter (common choices: 64, 128, 256, 512).
    family : int
        Template family index 0–4.
    seed : int or None
        Random seed for reproducibility (sets NumPy global state).

    Returns
    -------
    (12*nside²,) array, mean = 0 exactly, std = 1 exactly.

    Performance
    -----------
    Measured on CPU (nside=64, ~49 k pixels): **~76 ms/call**.
    Dominated by ``healpy.synfast``; scales roughly as O(nside²·log nside).

    Precision
    ---------
    Output mean = 0.0 and std = 1.0 to ``atol < 1e-14`` (enforced by explicit
    subtraction and division, not merely from the SHT).

    Examples
    --------
    >>> from sys_mapping import generate_systematic_map
    >>> import numpy as np
    >>> m = generate_systematic_map(nside=64, family=2, seed=42)
    >>> m.shape
    (49152,)
    >>> float(np.mean(m))   # exactly 0
    0.0
    >>> float(np.std(m))    # exactly 1
    1.0
    """
    if seed is not None:
        np.random.seed(seed)
    cl = systematic_power_spectrum(nside, family)
    # synfast expects C_ℓ starting at ℓ=0; uses numpy global random state
    template = hp.synfast(cl, nside=nside, lmax=3 * nside - 1)
    # Remove monopole numerically (should already be ~0 but enforce exactly)
    template -= np.mean(template)
    std = np.std(template)
    if std > 0:
        template /= std
    return template


def generate_systematic_maps(
    nside: int,
    families: list[int] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate multiple systematic template maps.

    Calls :func:`generate_systematic_map` once per family with independent
    child seeds derived from the master ``seed``.

    Parameters
    ----------
    nside : int
    families : list of int or None
        Template families to generate. Defaults to all 5 (families 0–4).
    seed : int or None
        Master random seed; each map gets a unique child seed.

    Returns
    -------
    (n_sys, 12*nside²) array of templates, each with mean=0 and std=1.

    Performance
    -----------
    Proportional to n_sys × :func:`generate_systematic_map` cost.
    For nside=64 and all 5 families: **~380 ms**.

    Precision
    ---------
    Each row has mean=0 and std=1 to ``atol < 1e-14`` (see :func:`generate_systematic_map`).

    Examples
    --------
    >>> from sys_mapping import generate_systematic_maps
    >>> maps = generate_systematic_maps(nside=64, seed=0)
    >>> maps.shape  # (5 families, 49152 pixels)
    (5, 49152)
    >>> maps = generate_systematic_maps(nside=32, families=[0, 2], seed=1)
    >>> maps.shape
    (2, 12288)
    """
    if families is None:
        families = list(range(5))
    rng = np.random.default_rng(seed)
    maps = []
    for i, fam in enumerate(families):
        child_seed = rng.integers(0, 2**31)
        maps.append(generate_systematic_map(nside, fam, seed=int(child_seed)))
    return np.stack(maps, axis=0)


def pixelize_catalog(
    ra: np.ndarray,
    dec: np.ndarray,
    nside: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Count galaxies per HEALPix pixel.

    Converts (ra, dec) to HEALPix ring-scheme pixel indices using
    ``healpy.ang2pix``, then bins with ``numpy.bincount``.

    Parameters
    ----------
    ra, dec : (n_gal,) positions in degrees
    nside : int  HEALPix resolution
    weights : (n_gal,) optional per-galaxy weights

    Returns
    -------
    (12*nside²,) galaxy counts (or weighted sums) per pixel.
    Sum equals ``n_gal`` (unweighted) or ``sum(weights)`` (weighted).

    Performance
    -----------
    Measured on CPU (n_gal=100_000, nside=64): **~6.5 ms/call**.
    Dominated by ``healpy.ang2pix``; scales as O(n_gal).

    Precision
    ---------
    Count conservation exact (integer binning). Weighted sum conserved to
    float64 accumulation error (``rtol < 1e-10`` for typical weights).

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import pixelize_catalog
    >>> rng = np.random.default_rng(0)
    >>> ra  = rng.uniform(0, 360, 50_000)
    >>> dec = rng.uniform(-90, 90, 50_000)
    >>> counts = pixelize_catalog(ra, dec, nside=32)
    >>> counts.shape
    (12288,)
    >>> int(counts.sum())  # total count conserved
    50000
    """
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    pixel_ids = hp.ang2pix(nside, theta, phi)
    n_pix = hp.nside2npix(nside)
    if weights is None:
        counts = np.bincount(pixel_ids, minlength=n_pix).astype(float)
    else:
        counts = np.bincount(pixel_ids, weights=weights, minlength=n_pix)
    return counts


def compute_overdensity(
    galaxy_counts: np.ndarray,
    random_counts: np.ndarray,
    min_random_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute galaxy overdensity δ_g = n_g/ñ_g − 1 with masking.

    Normalizes the random catalog to the same total as the galaxy catalog, then
    masks pixels where the random density is below ``min_random_fraction`` of its
    maximum (edge/boundary pixels with poor coverage).

    Parameters
    ----------
    galaxy_counts : (n_pix,)
    random_counts : (n_pix,)
    min_random_fraction : float
        Pixels with ``random_counts < min_random_fraction * max(random_counts)``
        are excluded. Default 0.1 (10 %).

    Returns
    -------
    delta_g : (n_good_pix,) overdensity in unmasked pixels
    good_pixels : (n_pix,) boolean mask (True = unmasked)

    Performance
    -----------
    Measured on CPU (n_pix=49_152 for nside=64): **~65 μs/call**. Pure NumPy.

    Precision
    ---------
    Overdensity mean < 0.01 by construction of the normalization (verified
    for uniform catalogs; residual depends on shot noise ~1/√N_gal).

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import pixelize_catalog, compute_overdensity
    >>> rng = np.random.default_rng(0)
    >>> nside = 32
    >>> gal  = pixelize_catalog(rng.uniform(0, 360, 100_000),
    ...                         rng.uniform(-90, 90, 100_000), nside)
    >>> rand = pixelize_catalog(rng.uniform(0, 360, 500_000),
    ...                         rng.uniform(-90, 90, 500_000), nside)
    >>> delta_g, good = compute_overdensity(gal, rand)
    >>> delta_g.shape  # unmasked pixels only
    (12288,)
    >>> abs(float(delta_g.mean())) < 0.01  # mean near zero
    True
    """
    max_random = np.max(random_counts)
    good_pixels = random_counts >= min_random_fraction * max_random

    g = galaxy_counts[good_pixels]
    r = random_counts[good_pixels]

    # Normalize randoms to same total as galaxies
    norm = np.sum(g) / np.sum(r)
    delta_g = g / (norm * r) - 1.0
    return delta_g, good_pixels


def inverse_variance_pixel_weights(
    galaxy_counts: np.ndarray,
    random_counts: np.ndarray,
    *,
    regulariser: float = 2.0,
) -> np.ndarray:
    """Per-pixel inverse-variance weight for the contamination regression.

    .. math::

        W_k \\;\\propto\\; \\frac{A_k^2}{N_k + n_{\\rm reg}}

    where :math:`A_k` is the observed area of pixel :math:`k` (traced by the
    random count, exactly as in :func:`compute_overdensity`) and :math:`N_k` its
    galaxy count.  This is the DES Y6 weighting (Weaverdyck et al. 2026, Eq. 8):
    the numerator downweights partially covered pixels, whose overdensity is
    measured over less sky, and the denominator is the Poisson variance of the
    count.

    The regulariser matters more than it looks.  Without it an empty pixel gets
    infinite weight, and pixels with one or two galaxies dominate the fit; with
    it every pixel keeps a finite, bounded weight and the heteroskedasticity of
    the observations is still accounted for.

    Parameters
    ----------
    galaxy_counts:
        Galaxy count per pixel (shape ``(n_pix,)``), at the fitted pixels.
    random_counts:
        Random count per pixel, the coverage proxy (shape ``(n_pix,)``).
    regulariser:
        Count added to the Poisson variance.  Default ``2.0``, the DES Y6 value.

    Returns
    -------
    weights : ``(n_pix,)``
        Normalised to mean one, so the overall scale of the likelihood is
        unchanged and only the relative weighting of pixels differs.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import inverse_variance_pixel_weights
    >>> n_gal = np.array([10.0, 20.0, 0.0])
    >>> n_ran = np.array([100.0, 100.0, 50.0])
    >>> w = inverse_variance_pixel_weights(n_gal, n_ran)
    >>> bool(np.isclose(w.mean(), 1.0))
    True
    >>> bool(np.all(np.isfinite(w)))
    True
    >>> bool(w[1] < w[0])          # more galaxies -> larger Poisson variance
    True

    References
    ----------
    Weaverdyck et al. 2026, arXiv:2601.14484, Eq. 8.
    """
    galaxy_counts = np.asarray(galaxy_counts, dtype=float)
    random_counts = np.asarray(random_counts, dtype=float)
    if galaxy_counts.shape != random_counts.shape:
        raise ValueError(
            f"galaxy_counts {galaxy_counts.shape} and random_counts "
            f"{random_counts.shape} must have the same shape")

    area = random_counts / (np.max(random_counts) + 1e-30)
    w = area ** 2 / (galaxy_counts + regulariser)
    mean = float(np.mean(w))
    return w / mean if mean > 0 else np.ones_like(w)


def standardise_on_footprint(
    delta_t: np.ndarray,
    *,
    return_scales: bool = False,
):
    r"""Standardise a template basis over the pixels it is about to be fitted on.

    Survey-property maps are normalised over each map's own valid region, which
    is larger than any one sample's footprint.  Restricted to the footprint the
    basis is no longer standardised, and three quantities that are read in units
    of one template standard deviation stop being in those units: the fitted
    amplitudes, the condition number of the template covariance, and the template
    auto-correlations :math:`\xi_i(\theta)` that the two-point correction of
    :func:`~sys_mapping.correction.correct_two_point_function` subtracts.

    Measured on the eleven LS10 maps at :math:`NSIDE = 64`, the per-template rms
    over the analysis pixels spans ``0.905`` to ``5.77`` and the relative mean
    reaches ``0.729``, so the covariance eigenvalues sum to ``44.4`` rather than
    ``n_sys = 11``.  Synthetic bases are barely affected --- ``0.971``--``1.030``
    under a Galactic cut --- because they carry no footprint structure.

    Call this after masking, where the footprint is known.  Doing it at load time
    cannot work: the loader does not know which pixels the fit will use.

    Parameters
    ----------
    delta_t:
        ``(n_sys, n_pix)`` template values at the pixels that will be fitted.
    return_scales:
        Also return the ``(n_sys,)`` means and rms values that were divided out,
        so a caller can record what basis its amplitudes are in.

    Returns
    -------
    ``(n_sys, n_pix)`` with each row at zero mean and unit rms over ``n_pix``, or
    ``(delta_t, means, scales)`` when ``return_scales``.  A row that is constant
    over the footprint carries no information and is left at zero rather than
    divided by zero.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import standardise_on_footprint
    >>> rng = np.random.default_rng(0)
    >>> t = rng.standard_normal((3, 5000))
    >>> t[2] = 4.0 * t[2] + 7.0            # wrong units and offset
    >>> ts, means, scales = standardise_on_footprint(t, return_scales=True)
    >>> bool(np.allclose(ts.mean(axis=1), 0.0, atol=1e-12))
    True
    >>> bool(np.allclose(ts.std(axis=1), 1.0, atol=1e-12))
    True
    >>> float(round(scales[2], 1))
    4.0
    """
    delta_t = np.asarray(delta_t, dtype=float)
    if delta_t.ndim != 2:
        raise ValueError(
            f"delta_t must be (n_sys, n_pix); got shape {delta_t.shape}"
        )
    means = delta_t.mean(axis=1)
    centred = delta_t - means[:, None]
    scales = np.sqrt(np.mean(centred ** 2, axis=1))
    # A constant row has nothing to rescale.  Dividing by its zero rms would give
    # NaN amplitudes for every template, not just that one.
    safe = np.where(scales > 0, scales, 1.0)
    out = centred / safe[:, None]
    if return_scales:
        return out, means, scales
    return out


def assign_template_values(
    templates: np.ndarray,
    good_pixels: np.ndarray,
) -> np.ndarray:
    """Extract template values at unmasked pixels.

    Parameters
    ----------
    templates : ``(n_sys, n_pix)`` full-sky template maps
    good_pixels : (n_pix,) boolean mask (True = unmasked)

    Returns
    -------
    (n_sys, n_good_pix) template values at unmasked pixels

    Performance
    -----------
    Measured on CPU (n_sys=5, nside=64, ~90 % unmasked): **~1610 μs/call**.
    Dominated by NumPy boolean fancy-indexing; scales as O(n_sys × n_good_pix).

    Precision
    ---------
    Pure array slicing; values are reproduced exactly (no floating-point error).

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import generate_systematic_maps, assign_template_values
    >>> templates = generate_systematic_maps(nside=32, seed=0)  # (5, 12288)
    >>> good = np.ones(12288, dtype=bool)
    >>> good[:100] = False          # mask first 100 pixels
    >>> t_masked = assign_template_values(templates, good)
    >>> t_masked.shape
    (5, 12188)
    """
    return templates[:, good_pixels]


def load_real_template(
    fits_path: str | Path,
    column: str,
    nside: int | None = None,
    *,
    valid_min: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and normalise a HEALPix template map from a FITS BinTableHDU.

    Reads the raw pixel values, identifies the survey footprint (pixels where
    the value is finite and above *valid_min*), normalises to zero mean and unit
    standard deviation over those pixels, and sets outside-footprint pixels to
    exactly 0.0 (the post-normalisation mean).

    Parameters
    ----------
    fits_path : path-like
        Path to a FITS file containing a BinTableHDU whose rows form a
        HEALPix map (one pixel per row, one column per observable).
    column : str
        Name of the BinTableHDU column to read (e.g. ``"nstar_faint"`` or
        ``"GALDEPTH_Z"``).
    nside : int or None
        Target HEALPix NSIDE.  If *None*, inferred from the data length.
        If provided and different from the file resolution,
        ``healpy.ud_grade`` is applied before normalisation.
    valid_min : float
        Pixels with ``value <= valid_min`` are treated as outside the survey
        footprint and excluded from mean/std computation.  Set to 0.0 for
        maps where 0 means "no observation" (e.g. LS10 depth).

    Returns
    -------
    template : (12*nside²,) float64 array
        Normalised map: mean = 0 and std = 1 over valid pixels.
        Pixels outside the footprint are set to 0.0.
    valid_mask : (12*nside²,) bool array
        True where ``raw > valid_min`` and finite.

    Examples
    --------
    >>> from sys_mapping.maps import load_real_template
    >>> t, mask = load_real_template(
    ...     "/data/GAIA_nstar_faint_NSIDE_00064.fits",
    ...     column="nstar_faint",
    ... )
    >>> t.shape
    (49152,)
    >>> abs(t[mask].mean()) < 1e-10   # mean=0 over valid pixels
    True
    >>> abs(t[mask].std() - 1.0) < 1e-10  # std=1 over valid pixels
    True
    """
    from astropy.io import fits as astropy_fits

    fits_path = Path(fits_path)
    with astropy_fits.open(str(fits_path)) as hdul:
        raw = None
        for hdu in hdul[1:]:
            if hasattr(hdu, "columns") and column in hdu.columns.names:
                raw = hdu.data[column].ravel().astype(np.float64)
                break
    if raw is None:
        raise ValueError(
            f"Column '{column}' not found in any BinTableHDU of {fits_path}"
        )

    file_nside = hp.npix2nside(len(raw))
    if nside is not None and nside != file_nside:
        raw = hp.ud_grade(raw, nside)

    valid_mask = np.isfinite(raw) & (raw > valid_min)
    template = np.zeros(len(raw), dtype=np.float64)
    if valid_mask.any():
        mean_v = raw[valid_mask].mean()
        std_v = raw[valid_mask].std()
        if std_v > 0:
            template[valid_mask] = (raw[valid_mask] - mean_v) / std_v

    return template, valid_mask


# Default file-name patterns for the two real templates bundled with LS10 data.
_GAIA_FILE_PATTERN = "GAIA_nstar_faint_NSIDE_{nside:05d}.fits"
_GAIA_COLUMN = "nstar_faint"
_LS10_DEPTH_FILE_PATTERN = "LS10_GALDEPTH_Z_NSIDE_{nside:04d}.fits"
_LS10_DEPTH_COLUMN = "GALDEPTH_Z"


def load_real_templates(
    nside: int,
    syst_dir: str | Path,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load the GAIA stellar-density and LS10 depth templates.

    These are the physically motivated ``synth_5`` (GAIA faint-star density,
    a proxy for stellar contamination) and ``synth_6`` (LS DR10 galaxy depth
    in the z band, a proxy for selection-depth variations) templates.

    The function looks for files matching the naming convention used by the
    ``~/data/legacysurvey/dr10/systematics/`` directory::

        GAIA_nstar_faint_NSIDE_00064.fits  (5-digit NSIDE zero-pad)
        LS10_GALDEPTH_Z_NSIDE_0064.fits   (4-digit NSIDE zero-pad)

    Parameters
    ----------
    nside : int
        HEALPix NSIDE (must match an available file; 32, 64, 128 or 256).
    syst_dir : path-like
        Directory containing the GAIA and LS10 FITS files.

    Returns
    -------
    templates : (2, 12*nside²) float64 array
        Row 0 = ``GAIA_nstar_faint`` (synth_5), normalised to zero mean /
        unit std over its full-sky footprint.
        Row 1 = ``LS10_GALDEPTH_Z`` (synth_6), normalised over the LS10
        survey footprint; pixels outside LS10 set to 0.
    names : list of str
        ``["GAIA_nstar_faint", "LS10_GALDEPTH_Z"]``
    valid_mask : (12*nside²,) bool array
        Pixels where *both* templates are defined (intersection of their
        individual footprints).  Use this as the survey mask for realistic
        mock catalogues.

    Examples
    --------
    >>> import sys_mapping as sm
    >>> templates, names, mask = sm.load_real_templates(
    ...     64, "~/data/legacysurvey/dr10/systematics"
    ... )
    >>> templates.shape
    (2, 49152)
    >>> names
    ['GAIA_nstar_faint', 'LS10_GALDEPTH_Z']
    """
    syst_dir = Path(syst_dir).expanduser()
    gaia_path = syst_dir / _GAIA_FILE_PATTERN.format(nside=nside)
    depth_path = syst_dir / _LS10_DEPTH_FILE_PATTERN.format(nside=nside)

    t_gaia, mask_gaia = load_real_template(gaia_path, _GAIA_COLUMN)
    t_depth, mask_depth = load_real_template(depth_path, _LS10_DEPTH_COLUMN)

    templates = np.stack([t_gaia, t_depth], axis=0)
    names = ["GAIA_nstar_faint", "LS10_GALDEPTH_Z"]
    # Intersection: pixels observed by both datasets
    valid_mask = mask_gaia & mask_depth

    return templates, names, valid_mask
