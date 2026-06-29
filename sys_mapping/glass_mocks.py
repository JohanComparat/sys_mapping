"""Full-sky lognormal galaxy mock generation using GLASS.

Generates systematic-free galaxy catalogs matched to the BGS LS10 redshift
distribution and surface density.  The key advantage over the lognormal
generator in ``mocks.py`` is that GLASS produces proper full-sky catalogs
without galactic-cut hacks, using the algorithm of Tessore et al. 2023
(arXiv:2302.01942).

Workflow
--------
1. Measure n(z) from a reference catalog (e.g. Uchuu).
2. Call ``generate_glass_fullsky_mock()`` with the measured n(z) to produce
   galaxy and random positions over the full sky.
3. Downstream pipeline code applies the survey footprint mask and injects
   systematics via ``sys_mapping.simulation``.

References
----------
Tessore et al. 2023, OJAp, 6, 11.  https://arxiv.org/abs/2302.01942
GLASS code: https://github.com/glass-dev/glass
"""

from __future__ import annotations

import warnings
from typing import TypedDict

import healpy as hp
import numpy as np


# Full-sky area in arcmin^2: 4π sr × (180/π × 60)² arcmin²/sr
_FULLSKY_ARCMIN2: float = 4.0 * np.pi * (180.0 / np.pi * 60.0) ** 2


class MockCatalogDict(TypedDict):
    """Dict returned by mock generators — compatible with simulation pipeline."""

    ra: np.ndarray
    dec: np.ndarray
    z: np.ndarray
    ra_rand: np.ndarray
    dec_rand: np.ndarray
    n_total: int
    nside: int
    seed: int | None


def measure_nz(
    z_array: np.ndarray,
    z_min: float,
    z_max: float,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram a redshift array into equal-width bins.

    Parameters
    ----------
    z_array:
        Galaxy redshifts.
    z_min, z_max:
        Redshift range to include (endpoints of the bin grid).
    n_bins:
        Number of histogram bins.

    Returns
    -------
    z_edges:
        Bin edges (length ``n_bins + 1``).
    nz:
        Galaxy counts per bin (length ``n_bins``).  Galaxies outside
        [z_min, z_max] are ignored.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping.glass_mocks import measure_nz
    >>> rng = np.random.default_rng(0)
    >>> z = rng.uniform(0.05, 0.26, 10000)
    >>> edges, nz = measure_nz(z, 0.05, 0.26, n_bins=10)
    >>> edges.shape
    (11,)
    >>> nz.shape
    (10,)
    >>> int(nz.sum()) <= 10000
    True
    """
    z_edges = np.linspace(z_min, z_max, n_bins + 1)
    nz, _ = np.histogram(z_array, bins=z_edges)
    return z_edges, nz.astype(float)


def _make_glass_cls(nside: int, amplitude: float = 5e-4) -> np.ndarray:
    """Return a simple galaxy C_ℓ power spectrum for GLASS.

    Uses C_ℓ ∝ (ℓ+1)^{-1.5} (intermediate slope, realistic for BGS scales).
    The monopole is forced to zero.

    Parameters
    ----------
    nside:
        HEALPix NSIDE; sets lmax = 3 * nside.
    amplitude:
        Overall C_ℓ amplitude at ℓ=1.  Default tuned for mild clustering.

    Returns
    -------
    (3*nside + 1,) C_ℓ array.
    """
    lmax = 3 * nside
    ells = np.arange(lmax + 1, dtype=float)
    cl = np.where(ells > 0, amplitude * (ells + 1) ** (-1.5), 0.0)
    return cl


def generate_glass_fullsky_mock(
    nside: int,
    n_total: int,
    z_edges: np.ndarray,
    nz: np.ndarray,
    *,
    cl_amplitude: float = 5e-4,
    rand_factor: int = 10,
    seed: int | None = None,
) -> MockCatalogDict:
    """Generate a full-sky lognormal galaxy mock catalog using GLASS.

    The mock has the angular clustering statistics of a lognormal random field
    with the given power spectrum amplitude, and the redshift distribution set
    by (z_edges, nz).

    Parameters
    ----------
    nside:
        HEALPix resolution for the underlying density field.
    n_total:
        Target total number of galaxies.  The actual count will be close but
        not exactly equal (Poisson sampling).
    z_edges:
        Bin edges of the n(z) histogram (length n_bins + 1).  The first edge
        defines z_min and the last edge defines z_max.
    nz:
        Galaxy counts per redshift bin (length n_bins).  Used to sample
        redshifts; does not need to be normalised.
    cl_amplitude:
        Amplitude of the galaxy angular power spectrum C_ℓ at ℓ=1.
    rand_factor:
        Ratio of randoms to data.  Default 10 gives accurate Landy-Szalay.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    MockCatalogDict with keys ``ra``, ``dec``, ``z``, ``ra_rand``,
    ``dec_rand``, ``n_total``, ``nside``, ``seed``.

    Notes
    -----
    The returned catalog is full-sky (no footprint mask applied).  Downstream
    code in ``sys_mapping.simulation`` applies the LSDR10 systematic maps and
    the survey footprint.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping.glass_mocks import generate_glass_fullsky_mock, measure_nz
    >>> rng_z = np.random.default_rng(1)
    >>> z = rng_z.uniform(0.05, 0.26, 5000)
    >>> z_edges, nz = measure_nz(z, 0.05, 0.26, n_bins=5)
    >>> cat = generate_glass_fullsky_mock(nside=16, n_total=5000,
    ...                                   z_edges=z_edges, nz=nz, seed=0)
    >>> len(cat['ra']) > 0
    True
    >>> cat['ra'].min() >= 0.0 and cat['ra'].max() < 360.0
    True
    >>> cat['dec'].min() >= -90.0 and cat['dec'].max() <= 90.0
    True
    """
    import glass
    import glass.fields as gf

    rng = np.random.default_rng(seed)

    # One tophat shell covering the full redshift range
    z_min = float(z_edges[0])
    z_max = float(z_edges[-1])
    z_glass = np.array([0.0, z_max])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        shells = glass.tophat_windows(z_glass)

    fields = glass.lognormal_fields(shells)

    cl = _make_glass_cls(nside, amplitude=cl_amplitude)
    cls_list = [cl]  # single shell → single auto-spectrum
    gls = glass.regularized_spectra(glass.solve_gaussian_spectra(fields, cls_list))

    ngal_per_arcmin2 = n_total / _FULLSKY_ARCMIN2

    all_lon: list[np.ndarray] = []
    all_lat: list[np.ndarray] = []
    for delta in glass.generate(fields, gls, nside, rng=rng):
        for lon_batch, lat_batch, _ in glass.positions_from_delta(
            ngal_per_arcmin2, delta, rng=rng
        ):
            all_lon.append(lon_batch)
            all_lat.append(lat_batch)

    lon = np.concatenate(all_lon) % 360.0  # wrap to [0, 360)
    lat = np.concatenate(all_lat)

    # Assign redshifts: sample from the measured n(z)
    n_gal = len(lon)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    nz_norm = np.asarray(nz, dtype=float)
    nz_norm = nz_norm / nz_norm.sum()
    bin_idx = rng.choice(len(nz_norm), size=n_gal, p=nz_norm)
    # Uniform draw within each chosen bin
    dz = np.diff(z_edges)
    z_gal = z_edges[bin_idx] + rng.uniform(0, 1, n_gal) * dz[bin_idx]
    z_gal = np.clip(z_gal, z_min, z_max)

    # Random catalog: uniform on the sphere
    n_rand = int(n_total * rand_factor)
    phi_rand = rng.uniform(0, 2 * np.pi, n_rand)
    cos_theta_rand = rng.uniform(-1, 1, n_rand)
    ra_rand = np.degrees(phi_rand)
    dec_rand = np.degrees(np.arcsin(cos_theta_rand))

    return {
        "ra": lon,
        "dec": lat,
        "z": z_gal,
        "ra_rand": ra_rand,
        "dec_rand": dec_rand,
        "n_total": n_gal,
        "nside": nside,
        "seed": seed,
    }


def generate_glass_delta_map(
    nside: int,
    z_max: float,
    *,
    cl_amplitude: float = 5e-4,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a single full-sky lognormal overdensity map with GLASS.

    Unlike :func:`generate_glass_fullsky_mock`, this returns the *field* itself
    (one tophat shell over ``[0, z_max]``) rather than sampled galaxy positions.
    The map can then be sampled multiple times — e.g. once for a clean catalog
    and once for a contaminated catalog that shares the same underlying
    structure — via :func:`sample_positions_from_delta`.

    Parameters
    ----------
    nside:
        HEALPix resolution of the field.  ``lmax = 3 * nside``.
    z_max:
        Upper edge of the single tophat redshift shell.
    cl_amplitude:
        Amplitude of the galaxy angular power spectrum C_ℓ at ℓ=1.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (12*nside²,) float array — the galaxy overdensity δ in RING ordering, with
    near-zero mean and ``1 + δ ≥ 0`` (lognormal field).

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping.glass_mocks import generate_glass_delta_map
    >>> delta = generate_glass_delta_map(nside=16, z_max=0.3, seed=0)
    >>> delta.shape
    (3072,)
    >>> bool((1.0 + delta).min() >= 0.0)
    True
    """
    import glass

    rng = np.random.default_rng(seed)
    z_glass = np.array([0.0, float(z_max)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        shells = glass.tophat_windows(z_glass)

    fields = glass.lognormal_fields(shells)
    cl = _make_glass_cls(nside, amplitude=cl_amplitude)
    gls = glass.regularized_spectra(glass.solve_gaussian_spectra(fields, [cl]))

    # Single shell → single delta map
    for delta in glass.generate(fields, gls, nside, rng=rng):
        return np.asarray(delta)
    raise RuntimeError("glass.generate yielded no fields")


def sample_positions_from_delta(
    delta: np.ndarray,
    ngal_per_arcmin2: float,
    *,
    vis: np.ndarray | None = None,
    bias: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Poisson-sample galaxy positions tracing an overdensity field.

    Thin wrapper over :func:`glass.positions_from_delta`.  The expected number
    of galaxies per pixel is ``ngal_per_arcmin2 × vis × (1 + bias·delta)``.
    Passing a visibility/footprint map in ``vis`` restricts the sample to the
    survey area without materialising full-sky positions.

    This is the count-modulation primitive used to inject systematics into the
    galaxy *number density* (as opposed to per-galaxy weights): sample once from
    a clean ``delta`` and once from a contaminated ``delta`` (see
    :func:`sys_mapping.contamination.apply_contamination`) sharing the same
    realisation.

    Parameters
    ----------
    delta:
        (n_pix,) overdensity map (RING ordering).  Must satisfy ``1 + δ ≥ 0``
        wherever ``vis > 0``; clip beforehand if a contamination model can push
        it negative.
    ngal_per_arcmin2:
        Target mean galaxy surface density.
    vis:
        Optional (n_pix,) visibility map (e.g. survey coverage in ``[0, 1]``).
        Pixels with ``vis = 0`` are never sampled.
    bias:
        Linear bias passed to GLASS (``δ_g = bias · δ``).  Use ``1.0`` to sample
        directly from ``delta`` as an already-biased galaxy overdensity.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    ra, dec : (n_gal,) arrays in degrees.  ``ra`` is wrapped to ``[0, 360)``.

    Examples
    --------
    >>> import numpy as np, healpy as hp
    >>> from sys_mapping.glass_mocks import generate_glass_delta_map, sample_positions_from_delta
    >>> nside = 16
    >>> delta = generate_glass_delta_map(nside=nside, z_max=0.3, seed=0)
    >>> vis = np.zeros(hp.nside2npix(nside)); vis[: vis.size // 2] = 1.0
    >>> ra, dec = sample_positions_from_delta(delta, 5.0, vis=vis, seed=1)
    >>> bool(ra.shape == dec.shape and ra.size > 0)
    True
    >>> bool(ra.min() >= 0.0 and ra.max() < 360.0)
    True
    """
    import glass

    rng = np.random.default_rng(seed)
    all_lon: list[np.ndarray] = []
    all_lat: list[np.ndarray] = []
    for lon_batch, lat_batch, _ in glass.positions_from_delta(
        ngal_per_arcmin2, np.asarray(delta), bias, vis, rng=rng
    ):
        all_lon.append(lon_batch)
        all_lat.append(lat_batch)

    if not all_lon:
        return np.empty(0), np.empty(0)
    return np.concatenate(all_lon) % 360.0, np.concatenate(all_lat)
