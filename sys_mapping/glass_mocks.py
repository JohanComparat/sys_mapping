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


def _make_glass_cls(nside: int, amplitude: float = 5e-4,
                    slope: float = -1.5) -> np.ndarray:
    """Return a power-law galaxy C_ℓ power spectrum for GLASS.

    C_ℓ = amplitude * (ℓ+1)^slope, with the monopole forced to zero.

    A power law with a *fixed* slope can match a single summary statistic of the
    data --- the pixel variance, say --- but not the scale dependence, and the
    scale dependence is what drives the variance inflation of the iid likelihood.
    Where the data's own spectrum is available, prefer passing it to
    :func:`generate_glass_fullsky_mock` as ``cl_input`` (see
    :func:`match_cl_to_target`) rather than fitting this two-parameter form.

    Parameters
    ----------
    nside:
        HEALPix NSIDE; sets lmax = 3 * nside.
    amplitude:
        Overall C_ℓ amplitude at ℓ=1.
    slope:
        Power-law index.  Default ``-1.5`` (intermediate, realistic for BGS
        scales) reproduces the historical behaviour.

    Returns
    -------
    (3*nside + 1,) C_ℓ array.
    """
    lmax = 3 * nside
    ells = np.arange(lmax + 1, dtype=float)
    cl = np.where(ells > 0, amplitude * (ells + 1) ** slope, 0.0)
    return cl


def sanitise_cl(cl: np.ndarray, lmax: int, *, floor_frac: float = 1e-8) -> np.ndarray:
    """Make a measured C_ℓ usable as a GLASS input spectrum.

    A spectrum measured from data is noisy and can go negative at high ℓ, where
    shot-noise subtraction over-subtracts.  GLASS needs a non-negative spectrum it
    can turn into a Gaussian one, so clip at a small positive floor, force the
    monopole to zero, and pad or truncate to ``lmax``.

    The floor is relative to the spectrum's own peak rather than absolute, so the
    function behaves the same whatever units or resolution it is handed.
    """
    cl = np.asarray(cl, dtype=float).copy()
    out = np.zeros(lmax + 1, dtype=float)
    n = min(cl.size, lmax + 1)
    out[:n] = cl[:n]
    if n < lmax + 1 and n > 2:
        out[n:] = out[n - 1]          # hold the last measured value, do not zero
    out[0] = 0.0
    peak = float(np.max(out)) if out.size else 0.0
    if peak > 0:
        np.clip(out, floor_frac * peak, None, out=out)
    out[0] = 0.0
    return out


def load_matched_cl(source, sample: str | None = None,
                    nside: int | None = None, *,
                    require_validated: bool = True) -> np.ndarray | None:
    """Load a spectrum produced by the mock-matching iteration, if one exists.

    There is no universal input spectrum.  What a mock has to reproduce is the
    large-scale clustering of *the particular sample, at the particular
    resolution, on the particular footprint* the analysis runs on --- change any
    of the three and the required spectrum changes.  So a matched spectrum is an
    artefact of one setup, stored per setup, and every consumer of the mocks
    loads the one matching its own configuration rather than sharing a constant.

    Parameters
    ----------
    source:
        A ``*_match.json`` written by ``match_glass_to_data.py``, or a directory
        holding them (in which case ``sample`` and ``nside`` select one).
    sample, nside:
        Used to build the filename when ``source`` is a directory.

    require_validated:
        Refuse a spectrum that has not passed the large-scale check (default).
        Set False only to inspect one deliberately.

    Returns
    -------
    The matched C_l, or ``None`` when no file matches --- so a caller can fall
    back to the parametric spectrum and say that it did, rather than failing.

    Raises
    ------
    ValueError
        If a file exists but has not passed the large-scale check.  Refusing is
        the point: an unverified null is the state this replaced.
    """
    from pathlib import Path
    import json
    import warnings

    def _validated(q) -> bool:
        """Has this file passed the large-scale check?"""
        try:
            v = json.loads(q.read_text()).get("validation")
        except (OSError, ValueError):
            return False
        return bool(v and v.get("passed"))

    path = Path(source)
    if path.is_dir():
        if sample is None:
            raise ValueError("sample is needed to pick a file from a directory")
        # The spectrum is a property of the SAMPLE and its footprint, not of the
        # map it was verified on: resolution limits what can be checked, not what
        # can be used.  Reaching rp = 10 Mpc/h needs NSIDE 128 for the
        # higher-redshift samples, while the LRT runs at 32 and 64, so a run must
        # be able to pick up a spectrum validated at finer resolution.
        cands = sorted(path.glob(f"{sample}_NSIDE*_match.json"))
        if not cands:
            return None
        res = lambda q: int(q.stem.split("_NSIDE")[1][:4])

        def nearest(qs):
            # The closest resolution at or above the target: it covers every
            # multipole the target map resolves and was fitted at the occupancy
            # nearest the target's.  The finest file is the wrong default for a
            # coarse target, since the finest fit is made where the sample is
            # sparsest.  With nothing at or above, the finest below is the
            # closest, extended past its last multipole by sanitise_cl.
            if nside is None:
                return max(qs, key=res)
            above = [q for q in qs if res(q) >= int(nside)]
            return min(above, key=res) if above else max(qs, key=res)
        exact = (path / f"{sample}_NSIDE{int(nside):04d}_match.json"
                 if nside is not None else None)
        if exact is not None and exact.exists() and _validated(exact):
            path = exact
        else:
            # Either there is no file at this resolution, or the one there did
            # not pass.  A failed fit at one resolution says nothing about a
            # passing fit at another for the same sample and footprint, so prefer
            # the nearest *validated* candidate rather than refusing outright --
            # but fall back to the nearest of any kind so that, when nothing
            # passed, the gate below still raises with a real diagnosis instead
            # of a bare None.
            passing = [q for q in cands if _validated(q)]
            path = nearest(passing) if passing else nearest(cands)
            if exact is not None and exact.exists() and path != exact:
                warnings.warn(
                    f"{exact.name} did not pass the large-scale check; using "
                    f"{path.name} instead, which did. The spectrum belongs to the "
                    f"sample and its footprint, not to the resolution it was "
                    f"verified at.",
                    UserWarning, stacklevel=2,
                )
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    cl = d.get("cl_matched")
    if not cl:
        return None

    # The gate.  A matched spectrum is only usable once it has been shown, on
    # seeds it was not fitted to, to reproduce the data's large-scale clustering
    # to within tolerance.  A file that records a failed check, or that predates
    # the check entirely, is refused rather than used quietly -- a null nobody
    # verified is exactly the state this whole exercise started from.
    val = d.get("validation")
    if val is None:
        if not require_validated:
            return np.asarray(cl, dtype=float)
        raise ValueError(
            f"{path} carries no validation block: it predates the large-scale "
            f"check and cannot be used.  Re-run match_glass_to_data.py, or pass "
            f"require_validated=False to accept it deliberately.")
    if not val.get("passed"):
        r = val.get("large_scale_ratio", float("nan"))
        msg = (f"{path} failed the large-scale check: mock/data power ratio "
               f"{r:.3f} over l={val.get('l_range')}, tolerance "
               f"{val.get('tol')}.  This mock is not calibrated to the data's "
               f"clustering and must not be used for a null.")
        if require_validated:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
    return np.asarray(cl, dtype=float)


_DEFAULT_CL_AMPLITUDE = 5e-4


def _resolve_spectrum(nside, cl_input, cl_amplitude, cl_slope, *, caller):
    """The spectrum a mock is drawn from, announcing the one case that is silent.

    A matched spectrum wins.  An explicit ``cl_amplitude`` is the caller's choice of a
    parametric field, which a simulation whose truth is that power law legitimately
    wants.  Neither given is the case that goes wrong: the default under-clusters a
    real galaxy sample by a large factor, so a null built on it is too narrow and any
    p-value calibrated against it is anticonservative.  That case warns.
    """
    if cl_input is not None:
        return sanitise_cl(cl_input, 3 * nside)
    if cl_amplitude is None:
        warnings.warn(
            f"{caller}: no cl_input and no cl_amplitude, so the mock uses the default "
            f"power law {_DEFAULT_CL_AMPLITUDE:g} (l+1)^{cl_slope:g}.  It is not matched "
            f"to any sample's clustering; a null built on it is not calibrated.  Pass "
            f"cl_input from load_matched_cl, or cl_amplitude to choose a parametric "
            f"field deliberately.",
            UserWarning, stacklevel=3)
        cl_amplitude = _DEFAULT_CL_AMPLITUDE
    return _make_glass_cls(nside, amplitude=cl_amplitude, slope=cl_slope)


def generate_glass_fullsky_mock(
    nside: int,
    n_total: int,
    z_edges: np.ndarray,
    nz: np.ndarray,
    *,
    cl_amplitude: float | None = None,
    cl_slope: float = -1.5,
    cl_input: np.ndarray | None = None,
    lognormal_shift: float | None = None,
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
        Amplitude of the galaxy angular power spectrum C_ℓ at ℓ=1.  Ignored when
        ``cl_input`` is given.
    cl_slope:
        Power-law index of the built-in spectrum.  Ignored when ``cl_input`` is
        given.
    cl_input:
        Tabulated C_ℓ to use directly, indexed from ℓ=0.  Overrides
        ``cl_amplitude``/``cl_slope``.  This is the way to give the mock the
        *measured* clustering of a real sample rather than a parametric guess;
        see ``match_glass_to_data.py`` in the sys_mapping_benchmark repository,
        which iterates it to convergence.  Passed through :func:`sanitise_cl`.
    lognormal_shift:
        Lognormal shift :math:`\lambda` of the density field (GLASS's ``shift``).
        The field is :math:`\delta = \lambda(e^{G-\sigma_G^2/2}-1)`, so
        :math:`\lambda` bounds the underdensity at :math:`\delta \ge -\lambda`
        and sets how skewed the field is at fixed variance: small
        :math:`\lambda` gives deep voids and rarer high peaks, large
        :math:`\lambda` approaches a Gaussian field.  It is the closest thing
        GLASS offers to a galaxy-bias parameter, and it changes the *shape* the
        realised spectrum takes for a given input, not only its amplitude.
        ``None`` keeps the GLASS default.
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
    ...                                   z_edges=z_edges, nz=nz,
    ...                                   cl_amplitude=5e-4, seed=0)
    >>> len(cat['ra']) > 0
    True
    >>> bool(cat['ra'].min() >= 0.0 and cat['ra'].max() < 360.0)
    True
    >>> bool(cat['dec'].min() >= -90.0 and cat['dec'].max() <= 90.0)
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

    fields = (glass.lognormal_fields(shells) if lognormal_shift is None
              else glass.lognormal_fields(shells, shift=lambda _z: float(lognormal_shift)))

    cl = _resolve_spectrum(nside, cl_input, cl_amplitude, cl_slope,
                           caller="generate_glass_fullsky_mock")
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
    cl_amplitude: float | None = None,
    cl_slope: float = -1.5,
    cl_input: np.ndarray | None = None,
    lognormal_shift: float | None = None,
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
        Amplitude of a parametric spectrum ``cl_amplitude (l+1)^cl_slope``.  Pass it
        to choose a parametric field deliberately; leaving both this and
        ``cl_input`` unset warns.
    cl_slope:
        Slope of the parametric spectrum.
    cl_input:
        A tabulated spectrum, typically from :func:`load_matched_cl`; takes
        precedence over the parametric form.
    lognormal_shift:
        GLASS lognormal shift; ``None`` uses GLASS's default.
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
    >>> delta = generate_glass_delta_map(nside=16, z_max=0.3, cl_amplitude=5e-4, seed=0)
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

    fields = (glass.lognormal_fields(shells) if lognormal_shift is None
              else glass.lognormal_fields(shells, shift=lambda _z: float(lognormal_shift)))
    cl = _resolve_spectrum(nside, cl_input, cl_amplitude, cl_slope,
                           caller="generate_glass_delta_map")
    gls = glass.regularized_spectra(glass.solve_gaussian_spectra(fields, [cl]))

    # Single shell → single delta map
    for delta in glass.generate(fields, gls, nside, rng=rng):
        return np.asarray(delta)
    raise RuntimeError("glass.generate yielded no fields")


def draw_null_overdensity(
    nside: int,
    good_pixels: np.ndarray,
    n_total_footprint: float,
    z_max: float,
    *,
    seed: int,
    rand_factor: float = 2.0,
    cl_amplitude: float | None = None,
    cl_slope: float = -1.5,
    cl_input: np.ndarray | None = None,
    lognormal_shift: float | None = None,
) -> np.ndarray:
    """One uncontaminated overdensity realisation on the footprint, drawn per pixel.

    The GLASS field of :func:`generate_glass_delta_map` sets the expected galaxy count
    of each footprint pixel, :math:`\bar n (1 + \delta)` with
    :math:`\bar n = n_{\rm total}/n_{\rm good}`; galaxy and random counts are drawn
    as Poisson variates (randoms at ``rand_factor`` times the galaxy density) and
    reduced to an overdensity exactly as :func:`~sys_mapping.maps.compute_overdensity`
    reduces data.  This is the distribution a catalogue mock produces once pixelised
    (``glass.positions_from_delta`` draws a Poisson count per pixel), without
    generating and pixelising the positions: on 28 000 pixels at 100 galaxies per
    pixel it takes 0.01 s per realisation against about 1 s, with the same pixel
    variance and the same amplitude scatter across templates.

    The field uses ``seed``; the counts use an independent stream derived from it.

    Returns
    -------
    ``(n_good,)`` overdensity at ``good_pixels``.
    """
    good_pixels = np.asarray(good_pixels, dtype=bool)
    n_good = int(good_pixels.sum())
    delta = generate_glass_delta_map(
        nside, z_max, cl_amplitude=cl_amplitude, cl_slope=cl_slope, cl_input=cl_input,
        lognormal_shift=lognormal_shift, seed=seed)[good_pixels]
    nbar = float(n_total_footprint) / n_good
    rng = np.random.default_rng([int(seed), 1])
    n_gal = rng.poisson(nbar * np.clip(1.0 + delta, 0.0, None)).astype(float)
    n_rand = rng.poisson(rand_factor * nbar, n_good).astype(float)
    total_gal, total_rand = n_gal.sum(), n_rand.sum()
    if total_gal < 1 or total_rand < 1:
        return np.zeros(n_good)
    norm = total_gal / total_rand
    return np.where(n_rand > 0, n_gal / (norm * np.where(n_rand > 0, n_rand, 1.0)) - 1.0, 0.0)


def generate_glass_null_overdensity(
    n_mocks: int,
    nside: int,
    good_pixels: np.ndarray,
    n_total_footprint: float,
    z_max: float,
    *,
    seed: int = 0,
    k_start: int = 0,
    rand_factor: float = 2.0,
    cl_amplitude: float | None = None,
    cl_slope: float = -1.5,
    cl_input: np.ndarray | None = None,
    lognormal_shift: float | None = None,
) -> np.ndarray:
    """``(n_mocks, n_good)`` uncontaminated realisations from :func:`draw_null_overdensity`.

    Realisation ``k`` always uses ``seed + k``, so indices ``[k_start, k_start + n_mocks)``
    reproduce exactly the realisations a longer run would have drawn.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping.glass_mocks import generate_glass_null_overdensity
    >>> good = np.ones(12 * 16**2, dtype=bool)
    >>> fields = generate_glass_null_overdensity(3, 16, good, 3e5, 0.3, seed=1,
    ...                                          cl_amplitude=5e-4)
    >>> fields.shape
    (3, 3072)
    >>> later = generate_glass_null_overdensity(1, 16, good, 3e5, 0.3, seed=1, k_start=2,
    ...                                         cl_amplitude=5e-4)
    >>> bool(np.array_equal(fields[2], later[0]))
    True
    """
    good_pixels = np.asarray(good_pixels, dtype=bool)
    out = np.empty((int(n_mocks), int(good_pixels.sum())))
    for i, k in enumerate(range(k_start, k_start + int(n_mocks))):
        out[i] = draw_null_overdensity(
            nside, good_pixels, n_total_footprint, z_max, seed=seed + k,
            rand_factor=rand_factor, cl_amplitude=cl_amplitude, cl_slope=cl_slope,
            cl_input=cl_input, lognormal_shift=lognormal_shift)
    return out


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
