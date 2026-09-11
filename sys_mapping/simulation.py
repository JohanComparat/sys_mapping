"""Systematic contamination injection, FITS I/O, and w(θ) recovery.

This module provides the end-to-end infrastructure for simulation-based
validation of the decontamination pipeline:

1. Load reference catalogs (Uchuu lightcone or GLASS mock).
2. Load LSDR10 systematic maps.
3. Inject contamination into catalog positions using the pixel-level forward
   model from ``sys_mapping.contamination``.
4. Save contaminated catalogs to FITS.
5. Run all fast decontamination methods and compare recovered w(θ) to truth.

Contamination injection strategy
---------------------------------
For a catalog of N galaxies, the true overdensity at each pixel is estimated
from the catalog itself (galaxy counts / random counts − 1).  The forward
contamination model (Eq. 11-13 of Berlfein et al. 2024) is applied to yield a
contaminated overdensity field.  The per-galaxy contamination weight is then:

    WEIGHT_CONT(p) = (1 + δ_cont(p)) / (1 + δ_g(p))

When computing w(θ) with TreeCorr, passing WEIGHT_CONT as galaxy weights
exactly reproduces the effect that the contamination model would have on the
angular correlation function, without physically removing or adding galaxies.
The "truth" w(θ) is computed with uniform weights (no contamination).

References
----------
Berlfein et al. 2024, MNRAS 531, 4954.  arXiv:2401.12293
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import healpy as hp
import jax.numpy as jnp
import numpy as np

from .contamination import apply_contamination, apply_nonlinear_contamination
from .maps import (
    assign_template_values,
    compute_overdensity,
    load_real_template,
    pixelize_catalog,
)
from .regression import run_decontamination


# ── Systematic map catalogue ───────────────────────────────────────────────────
# Maps present in ~/data/legacysurvey/dr10/systematics/{nside_dir}/
# Each entry: (filename_template, column_name, nside_zero_padding)
_MAP_SPECS: dict[str, tuple[str, str, int]] = {
    "LS10_EBV":         ("LS10_EBV_NSIDE_{nside}.fits",         "EBV",         4),
    "LS10_GALDEPTH_Z":  ("LS10_GALDEPTH_Z_NSIDE_{nside}.fits",  "GALDEPTH_Z",  4),
    "LS10_PSFSIZE_R":   ("LS10_PSFSIZE_R_NSIDE_{nside}.fits",   "PSFSIZE_R",   4),
    "LS10_NOBS_R":      ("LS10_NOBS_R_NSIDE_{nside}.fits",      "NOBS_R",      4),
    "GAIA_nstar_faint": ("GAIA_nstar_faint_NSIDE_{nside}.fits", "nstar_faint", 5),
}

DEFAULT_MAP_NAMES: list[str] = [
    "LS10_EBV",
    "LS10_GALDEPTH_Z",
    "LS10_PSFSIZE_R",
    "LS10_NOBS_R",
    "GAIA_nstar_faint",
]

# Contamination amplitude magnitudes for each level
LEVELS: dict[str, float] = {"low": 0.02, "medium": 0.05, "high": 0.10}

DEFAULT_METHODS: list[str] = ["OLS", "ISD-1", "ElasticNet"]


# ── Data structures ────────────────────────────────────────────────────────────


@dataclass
class ContaminationConfig:
    """Parameters for a single contamination scenario.

    Attributes
    ----------
    level:
        Human-readable amplitude label: ``"low"``, ``"medium"``, or ``"high"``.
    scenario:
        Contamination model: ``"additive"``, ``"multiplicative"``, or
        ``"combined"``.
    a_true:
        Additive amplitudes (shape ``(n_sys,)``).  Zero for multiplicative.
    b_true:
        Multiplicative amplitudes (shape ``(n_sys,)``).  Zero for additive.
    """

    level: str
    scenario: str
    a_true: np.ndarray
    b_true: np.ndarray
    responses: tuple = ()
    shape: str = "linear"

    @property
    def n_sys(self) -> int:
        return len(self.a_true)

    @property
    def contaminated(self) -> tuple[int, ...]:
        """Indices of the templates this configuration actually contaminates.

        The true positives a selection rule is supposed to find.  For the linear
        scenarios it is read off the amplitudes; for a response scenario it is
        the entries of ``responses`` that are not ``None``.
        """
        if self.responses:
            return tuple(i for i, r in enumerate(self.responses) if r is not None)
        return tuple(int(i) for i in np.flatnonzero(
            (np.asarray(self.a_true) != 0) | (np.asarray(self.b_true) != 0)))

    def to_header_dict(self) -> dict:
        """Flat dict suitable for storage in a FITS header."""
        out: dict = {"LEVEL": self.level, "SCENARIO": self.scenario,
                     "SHAPE": self.shape}
        for i, (a, b) in enumerate(zip(self.a_true, self.b_true)):
            out[f"A_TRUE_{i}"] = float(a)
            out[f"B_TRUE_{i}"] = float(b)
        for i, r in enumerate(self.responses):
            if r is not None:
                out.update(r.to_header_dict(i))
        return out


def make_contamination_grid(
    n_sys: int,
    seed: int = 0,
) -> list[ContaminationConfig]:
    """Build a 3×3 grid of contamination configurations.

    Returns 9 ``ContaminationConfig`` objects: one for each combination of
    three amplitude levels (low/medium/high) × three scenarios
    (additive/multiplicative/combined).

    The *signs* of the amplitudes are drawn once from the given seed and
    shared across levels so that the impact of amplitude magnitude can be
    cleanly isolated.

    Parameters
    ----------
    n_sys:
        Number of systematic templates.
    seed:
        Random seed for the sign draws.

    Returns
    -------
    List of 9 ``ContaminationConfig`` objects.

    Examples
    --------
    >>> from sys_mapping.simulation import make_contamination_grid
    >>> cfgs = make_contamination_grid(n_sys=3)
    >>> len(cfgs)
    9
    >>> {c.level for c in cfgs} == {'low', 'medium', 'high'}
    True
    >>> {c.scenario for c in cfgs} == {'additive', 'multiplicative', 'combined'}
    True
    """
    rng = np.random.default_rng(seed)
    signs_a = rng.choice([-1, 1], size=n_sys).astype(float)
    signs_b = rng.choice([-1, 1], size=n_sys).astype(float)

    configs: list[ContaminationConfig] = []
    for level_name, mag in LEVELS.items():
        a_base = signs_a * mag
        b_base = signs_b * mag
        for scenario in ("additive", "multiplicative", "combined"):
            a = a_base.copy() if scenario in ("additive", "combined") else np.zeros(n_sys)
            b = b_base.copy() if scenario in ("multiplicative", "combined") else np.zeros(n_sys)
            configs.append(ContaminationConfig(level=level_name, scenario=scenario, a_true=a, b_true=b))
    return configs


#: Shape parameter per response kind, chosen so that at the medium level the
#: departure from linear is large enough to matter and small enough to stay a
#: perturbative correction.  See `evaluate_response` for the definitions.
RESPONSE_SHAPES: dict[str, float] = {
    "linear": 0.0,        # control: what the existing grid already injects
    "quadratic": 0.5,     # kappa
    "cubic": 0.3,         # kappa
    "tanh": 1.5,          # alpha, saturating
    "threshold": 0.5,     # t0, a depth cut
    "exp": 0.8,           # alpha, runaway tail
}


def make_response_grid(
    n_sys: int,
    seed: int = 0,
    *,
    kinds: Sequence[str] | None = None,
    n_contaminated: int | None = None,
) -> list[ContaminationConfig]:
    """Build the non-linear response grid: ``len(kinds)`` shapes x 3 levels.

    The existing :func:`make_contamination_grid` injects
    ``delta_g*(1 + sum_i b_i t_i) + sum_i a_i t_i``, which is linear in every
    template.  A linear marginal fit is already sufficient for that, so it cannot
    distinguish ``ISD-1`` from ``ISD-3`` and in practice does not: the two recover
    the same amplitude in every cell of campaign ``20260907c``.  This grid injects
    a response that is not linear, through
    :func:`~sys_mapping.contamination.apply_nonlinear_contamination`.

    Three of the default kinds (``linear``, ``quadratic``, ``cubic``) lie inside
    the basis a degree-3 marginal polynomial represents exactly; three
    (``tanh``, ``threshold``, ``exp``) do not, and are there to find where the
    method stops working rather than to confirm that it does.

    Parameters
    ----------
    n_sys:
        Number of systematic templates.
    seed:
        Seed for the choice of which templates are contaminated.
    kinds:
        Response kinds to include; defaults to all of
        :data:`~sys_mapping.contamination.RESPONSE_KINDS`.
    n_contaminated:
        How many of the ``n_sys`` templates to contaminate.  ``None`` (default)
        contaminates all of them.  Setting it below ``n_sys`` is what makes
        greedy template *selection* measurable: with every template contaminated
        there is no true negative for a selection rule to get wrong.

    Returns
    -------
    ``len(kinds) * 3`` configurations, labelled by ``shape`` and ``level``.

    Examples
    --------
    >>> from sys_mapping.simulation import make_response_grid
    >>> cfgs = make_response_grid(n_sys=5, n_contaminated=2)
    >>> len(cfgs)
    18
    >>> all(len(c.contaminated) == 2 for c in cfgs)
    True
    """
    from .contamination import RESPONSE_KINDS, TemplateResponse

    kinds = tuple(kinds) if kinds is not None else RESPONSE_KINDS
    rng = np.random.default_rng(seed)
    if n_contaminated is None or n_contaminated >= n_sys:
        picked = np.arange(n_sys)
    else:
        picked = np.sort(rng.choice(n_sys, size=int(n_contaminated), replace=False))
    signs = rng.choice([-1, 1], size=n_sys).astype(float)

    configs: list[ContaminationConfig] = []
    for level_name, mag in LEVELS.items():
        for kind in kinds:
            responses = tuple(
                TemplateResponse(kind, RESPONSE_SHAPES[kind], float(signs[i] * mag))
                if i in picked else None
                for i in range(n_sys)
            )
            # a_true records the *linear* amplitude the response was scaled to,
            # so the existing recovery metrics stay meaningful as a baseline.
            a = np.zeros(n_sys)
            a[picked] = signs[picked] * mag
            configs.append(ContaminationConfig(
                level=level_name, scenario="response", a_true=a,
                b_true=np.zeros(n_sys), responses=responses, shape=kind,
            ))
    return configs


# ── I/O functions ──────────────────────────────────────────────────────────────


def load_uchuu_mock(data_fits: str | Path, rand_fits: str | Path) -> dict:
    """Load Uchuu lightcone FITS catalog.

    Parameters
    ----------
    data_fits:
        Path to the ``*_DATA.fits`` file.  Expected columns: ``RA``, ``DEC``,
        ``redshift_S``.
    rand_fits:
        Path to the ``*_RAND.fits`` file.  Expected columns: ``RA``, ``DEC``.

    Returns
    -------
    dict with keys ``ra``, ``dec``, ``z`` (galaxies) and ``ra_rand``,
    ``dec_rand`` (randoms).

    Examples
    --------
    >>> import os
    >>> p = os.path.expanduser(
    ...     '~/data/Uchuu/FullSky/mock_catalogues/'
    ...     'MOCK_VLIM_ANY_10.65_Mstar_12.0_0.05_z_0.26_N_0923373/'
    ...     'MOCK_VLIM_ANY_10.65_Mstar_12.0_0.05_z_0.26_N_0923373_DATA.fits')
    >>> # doctest: +SKIP
    >>> cat = load_uchuu_mock(p, p.replace('_DATA', '_RAND'))
    """
    from astropy.io import fits as afits

    with afits.open(str(data_fits)) as hdul:
        data = hdul[1].data
        ra = np.asarray(data["RA"], dtype=float)
        dec = np.asarray(data["DEC"], dtype=float)
        z = np.asarray(data["redshift_S"], dtype=float)

    with afits.open(str(rand_fits)) as hdul:
        rand = hdul[1].data
        ra_rand = np.asarray(rand["RA"], dtype=float)
        dec_rand = np.asarray(rand["DEC"], dtype=float)

    return {"ra": ra, "dec": dec, "z": z, "ra_rand": ra_rand, "dec_rand": dec_rand}


def load_systematic_maps(
    syst_dir: str | Path,
    nside: int,
    map_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load LSDR10 systematic maps at a given HEALPix resolution.

    Parameters
    ----------
    syst_dir:
        Directory containing NSIDE subdirectories, e.g.
        ``~/data/legacysurvey/dr10/systematics/``.  The function looks for
        files in ``{syst_dir}/{nside:04d}/`` (NSIDE in 4-digit zero-padded
        subdirectory).
    nside:
        Target HEALPix resolution.
    map_names:
        Which maps to load.  Defaults to ``DEFAULT_MAP_NAMES`` (5 maps).

    Returns
    -------
    templates : ``(n_sys, 12*nside²)`` array, zero-mean and unit-std over valid pixels.
    names : list of map name strings.
    footprint : (12*nside²,) bool array — True where *all* loaded maps are valid.
        Use this to restrict mock catalogs to the survey footprint before
        injecting systematics or measuring w(θ).

    Examples
    --------
    >>> # doctest: +SKIP
    >>> t, names, footprint = load_systematic_maps('~/data/legacysurvey/dr10/systematics/', 64)
    >>> t.shape
    (5, 49152)
    >>> footprint.sum()  # number of pixels inside the LS10 footprint
    ...
    """
    syst_dir = Path(syst_dir).expanduser()
    if map_names is None:
        map_names = DEFAULT_MAP_NAMES

    nside_dir = syst_dir / f"{nside:04d}"
    templates = []
    valid_names = []
    footprint = np.ones(hp.nside2npix(nside), dtype=bool)
    for name in map_names:
        fname_tmpl, column, zfill = _MAP_SPECS[name]
        nside_str = str(nside).zfill(zfill)
        fname = fname_tmpl.replace("{nside}", nside_str)
        fpath = nside_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Systematic map not found: {fpath}")
        tmpl, valid_mask = load_real_template(fpath, column=column, nside=nside)
        templates.append(tmpl)
        valid_names.append(name)
        footprint &= valid_mask

    return np.stack(templates, axis=0), valid_names, footprint


def apply_footprint_mask(
    catalog: dict,
    footprint: np.ndarray,
    nside: int,
) -> dict:
    """Filter a catalog dict to positions inside the survey footprint.

    Keeps only galaxies and randoms whose HEALPix pixel (at *nside*) is
    flagged as valid in *footprint*.  All other keys in *catalog* are passed
    through unchanged.

    Parameters
    ----------
    catalog:
        Dict with at least ``ra``, ``dec``, ``ra_rand``, ``dec_rand`` keys
        (and optionally ``z``).  As returned by
        :func:`~sys_mapping.glass_mocks.generate_glass_fullsky_mock`.
    footprint:
        (12*nside²,) bool array — True = inside survey footprint.
        As returned by :func:`load_systematic_maps`.
    nside:
        HEALPix NSIDE matching *footprint*.

    Returns
    -------
    Filtered catalog dict.  ``n_total`` (if present) is updated to the number
    of galaxies remaining after filtering.

    Notes
    -----
    This must be called *before* :func:`inject_systematics` and
    :func:`run_wtheta_recovery` so that w(θ) is measured only within the
    footprint where systematic templates are defined.
    """
    def _pix_mask(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        pix = hp.ang2pix(nside, theta, phi)
        return footprint[pix]

    gal_keep = _pix_mask(catalog["ra"], catalog["dec"])
    rand_keep = _pix_mask(catalog["ra_rand"], catalog["dec_rand"])

    out = dict(catalog)
    out["ra"] = catalog["ra"][gal_keep]
    out["dec"] = catalog["dec"][gal_keep]
    out["ra_rand"] = catalog["ra_rand"][rand_keep]
    out["dec_rand"] = catalog["dec_rand"][rand_keep]
    if "z" in catalog:
        out["z"] = catalog["z"][gal_keep]
    if "n_total" in catalog:
        out["n_total"] = int(gal_keep.sum())
    return out


# ── Contamination injection ────────────────────────────────────────────────────


def inject_systematics(
    ra: np.ndarray,
    dec: np.ndarray,
    ra_rand: np.ndarray,
    dec_rand: np.ndarray,
    templates: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    nside: int,
    responses: Sequence | None = None,
) -> np.ndarray:
    """Compute per-galaxy contamination weights.

    The function estimates the pixel-level galaxy overdensity from the input
    catalog, applies the forward contamination model, and returns per-galaxy
    weights that reproduce the contaminated angular correlation function when
    passed to TreeCorr.

    Parameters
    ----------
    ra, dec:
        Galaxy positions in degrees.
    ra_rand, dec_rand:
        Random catalog positions in degrees.
    templates:
        Systematic template maps (shape ``(n_sys, n_pix)``).
    a:
        Additive contamination amplitudes (shape ``(n_sys,)``).
    b:
        Multiplicative contamination amplitudes (shape ``(n_sys,)``).
    nside:
        HEALPix NSIDE used for pixelisation.

    Returns
    -------
    WEIGHT_CONT : (n_gal,) per-galaxy weights.  Equal to 1 when a=b=0.
    Clipped to a minimum of 0.01 to prevent unphysical negative weights.

    Notes
    -----
    The weight is defined as::

        WEIGHT_CONT(p) = (1 + δ_cont(p)) / (1 + δ_g(p))

    where δ_cont is the contaminated overdensity and δ_g is the observed
    overdensity from the input catalog.  Pixels where 1 + δ_g(p) ≤ 0 are
    assigned weight 1.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping.simulation import inject_systematics
    >>> rng = np.random.default_rng(0)
    >>> n_gal, n_rand = 5000, 50000
    >>> ra = rng.uniform(0, 360, n_gal)
    >>> dec = rng.uniform(-90, 90, n_gal)
    >>> ra_r = rng.uniform(0, 360, n_rand)
    >>> dec_r = rng.uniform(-90, 90, n_rand)
    >>> templates = rng.standard_normal((2, 12 * 32**2))
    >>> a = np.array([0.0, 0.0])
    >>> b = np.array([0.0, 0.0])
    >>> w = inject_systematics(ra, dec, ra_r, dec_r, templates, a, b, nside=32)
    >>> w.shape == (n_gal,)
    True
    >>> np.allclose(w, 1.0, atol=0.15)   # near unity for zero contamination
    True
    """
    n_pix = hp.nside2npix(nside)

    gal_counts = pixelize_catalog(ra, dec, nside)
    rand_counts = pixelize_catalog(ra_rand, dec_rand, nside)

    # Full-map overdensity (don't apply good-pixel mask here)
    rand_norm = rand_counts * (gal_counts.sum() / rand_counts.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        delta_g_full = np.where(rand_norm > 0, gal_counts / rand_norm - 1.0, 0.0)

    # Apply contamination model on the full sky.  `responses` selects the
    # selection-efficiency model `(1+d)*prod(1+F_i(t_i))`, which is what ISD
    # inverts and which need not be linear in the template; without it the model
    # is Eq. 13, linear in every template by construction.
    if responses is not None:
        delta_cont_full = apply_nonlinear_contamination(
            delta_g_full, templates, responses)
    else:
        delta_cont_full = np.asarray(
            apply_contamination(
                jnp.asarray(delta_g_full),
                jnp.asarray(templates),
                jnp.asarray(a),
                jnp.asarray(b),
            )
        )

    # Per-galaxy weight from per-pixel ratio
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    pix_idx = hp.ang2pix(nside, theta, phi)

    denom = 1.0 + delta_g_full[pix_idx]
    numer = 1.0 + delta_cont_full[pix_idx]
    weight = np.where(np.abs(denom) > 0.01, numer / denom, 1.0)
    return np.clip(weight, 0.01, None)


# ── FITS I/O ───────────────────────────────────────────────────────────────────


def save_simulation_catalog(
    ra: np.ndarray,
    dec: np.ndarray,
    z: np.ndarray,
    weight_cont: np.ndarray,
    config: ContaminationConfig,
    output_path: str | Path,
) -> None:
    """Write a contaminated simulation catalog to a FITS BinTableHDU.

    Parameters
    ----------
    ra, dec:
        Galaxy positions in degrees.
    z:
        Galaxy redshifts.
    weight_cont:
        Per-galaxy contamination weights (from ``inject_systematics``).
    config:
        Contamination configuration stored in the FITS header.
    output_path:
        Destination FITS file path.  Parent directory is created if needed.
    """
    from astropy.io import fits as afits
    from astropy.table import Table

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tbl = Table(
        {
            "RA": ra.astype(np.float32),
            "DEC": dec.astype(np.float32),
            "Z": z.astype(np.float32),
            "WEIGHT_CONT": weight_cont.astype(np.float32),
        }
    )
    hdu = afits.BinTableHDU(tbl)
    for key, val in config.to_header_dict().items():
        hdu.header[key] = val

    hdul = afits.HDUList([afits.PrimaryHDU(), hdu])
    hdul.writeto(str(output_path), overwrite=True)


def load_simulation_catalog(fits_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ContaminationConfig]:
    """Load a contaminated simulation catalog from FITS.

    Returns
    -------
    ra, dec, z, weight_cont, config
    """
    from astropy.io import fits as afits

    with afits.open(str(fits_path)) as hdul:
        hdr = hdul[1].header
        data = hdul[1].data
        ra = np.asarray(data["RA"], dtype=float)
        dec = np.asarray(data["DEC"], dtype=float)
        z = np.asarray(data["Z"], dtype=float)
        weight_cont = np.asarray(data["WEIGHT_CONT"], dtype=float)

        level = hdr.get("LEVEL", "unknown")
        scenario = hdr.get("SCENARIO", "unknown")
        n_sys = sum(1 for k in hdr if k.startswith("A_TRUE_"))
        a_true = np.array([hdr[f"A_TRUE_{i}"] for i in range(n_sys)])
        b_true = np.array([hdr[f"B_TRUE_{i}"] for i in range(n_sys)])

    config = ContaminationConfig(level=level, scenario=scenario, a_true=a_true, b_true=b_true)
    return ra, dec, z, weight_cont, config


# ── Weighted two-point function ────────────────────────────────────────────────


def _measure_wtheta(
    ra: np.ndarray,
    dec: np.ndarray,
    ra_rand: np.ndarray,
    dec_rand: np.ndarray,
    weights: np.ndarray | None = None,
    min_sep: float = 0.1,
    max_sep: float = 10.0,
    nbins: int = 10,
    sep_units: str = "degrees",
) -> tuple[np.ndarray, np.ndarray]:
    """Landy-Szalay w(θ) optionally weighted per galaxy.

    Galaxy weights are passed to TreeCorr via ``Catalog(w=weights)``.
    Randoms always have uniform weight.
    """
    import treecorr

    cat_gal = treecorr.Catalog(ra=ra, dec=dec, w=weights, ra_units="degrees", dec_units="degrees")
    cat_rand = treecorr.Catalog(ra=ra_rand, dec=dec_rand, ra_units="degrees", dec_units="degrees")
    cfg = dict(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units=sep_units, metric="Arc")

    dd = treecorr.NNCorrelation(**cfg)
    dr = treecorr.NNCorrelation(**cfg)
    rr = treecorr.NNCorrelation(**cfg)

    dd.process(cat_gal)
    dr.process(cat_gal, cat_rand)
    rr.process(cat_rand)

    xi, _ = dd.calculateXi(rr=rr, dr=dr)
    theta = np.exp(dd.meanlogr)
    return theta, xi


# ── w(θ) recovery pipeline ────────────────────────────────────────────────────


def run_wtheta_recovery(
    catalog: dict,
    templates: np.ndarray,
    template_names: list[str],
    config: ContaminationConfig,
    nside: int = 64,
    methods: list[str] | None = None,
    min_sep: float = 0.1,
    max_sep: float = 10.0,
    nbins: int = 10,
    method_kwargs: dict | None = None,
) -> dict:
    """Run full decontamination pipeline and return w(θ) at each stage.

    Given a clean catalog and contamination parameters, this function:

    1. Computes the reference (truth) w(θ) from the clean catalog.
    2. Injects contamination to build per-galaxy ``WEIGHT_CONT``.
    3. Computes the contaminated w(θ) using those weights.
    4. For each decontamination method, fits the contamination model and
       applies per-galaxy correction weights; computes the recovered w(θ).

    Parameters
    ----------
    catalog:
        Dict with keys ``ra``, ``dec``, ``z``, ``ra_rand``, ``dec_rand``.
    templates:
        Full-sky systematic maps (shape ``(n_sys, n_pix)``).
    template_names:
        Names of the template maps (for metadata in output).
    config:
        Contamination configuration to inject.
    nside:
        HEALPix resolution for overdensity computation.
    methods:
        Decontamination methods to run.  Default: OLS, ISD-1, ElasticNet.
    min_sep, max_sep:
        Angular separation range in degrees.
    nbins:
        Number of angular bins.
    method_kwargs:
        Extra keyword arguments forwarded to
        :func:`~sys_mapping.regression.run_decontamination`, either as one dict
        applied to every method or as ``{method_name: {...}}``.  The ISD methods
        need ``isd_chi2_68`` here: without it their stopping threshold is in raw
        :math:`\\Delta\\chi^2` units and the iteration does not stop where it
        should.

    Returns
    -------
    dict with keys:

    - ``theta``: bin centres in degrees
    - ``w_true``: w(θ) of clean catalog (truth)
    - ``w_contaminated``: w(θ) with WEIGHT_CONT applied
    - ``w_recovered_{method}``: w(θ) after decontamination for each method
    - ``params_{method}``: dict of recovered a_hat, b_hat
    - ``config``: the ``ContaminationConfig`` used
    - ``n_gal``, ``n_rand``: catalog sizes
    """
    if methods is None:
        methods = DEFAULT_METHODS

    ra = catalog["ra"]
    dec = catalog["dec"]
    ra_rand = catalog["ra_rand"]
    dec_rand = catalog["dec_rand"]

    # 1. Truth w(θ) — uniform galaxy weights
    theta, w_true = _measure_wtheta(
        ra, dec, ra_rand, dec_rand,
        min_sep=min_sep, max_sep=max_sep, nbins=nbins,
    )

    # 2. Contamination weights
    weight_cont = inject_systematics(
        ra, dec, ra_rand, dec_rand, templates, config.a_true, config.b_true, nside,
        responses=config.responses or None,
    )

    # 3. Contaminated w(θ)
    _, w_cont = _measure_wtheta(
        ra, dec, ra_rand, dec_rand,
        weights=weight_cont,
        min_sep=min_sep, max_sep=max_sep, nbins=nbins,
    )

    # 4. Decontamination and recovery
    gal_counts = pixelize_catalog(ra, dec, nside, weights=weight_cont)
    rand_counts = pixelize_catalog(ra_rand, dec_rand, nside)
    delta_g_obs, good_pixels = compute_overdensity(gal_counts, rand_counts)
    delta_t_masked = assign_template_values(templates, good_pixels)

    result: dict = {
        "theta": theta,
        "w_true": w_true,
        "w_contaminated": w_cont,
        "config": config,
        "n_gal": len(ra),
        "n_rand": len(ra_rand),
        "template_names": template_names,
    }

    # Precompute per-galaxy pixel assignment once (independent of method)
    theta_pix = np.radians(90.0 - dec)
    phi_pix = np.radians(ra)
    pix_gal = hp.ang2pix(nside, theta_pix, phi_pix)

    method_kwargs = method_kwargs or {}
    for method in methods:
        kw = method_kwargs.get(method, method_kwargs)
        kw = {k: v for k, v in kw.items() if not isinstance(v, dict)}
        res = run_decontamination(method, delta_g_obs, delta_t_masked, **kw)
        weights_sys = res["weights"]  # (n_good_pix,)

        # Map good-pixel correction weights back to per-galaxy
        n_full = hp.nside2npix(nside)
        weight_full = np.ones(n_full)
        weight_full[good_pixels] = weights_sys
        weight_recovered = weight_full[pix_gal]

        # Apply the correction ON TOP of the contamination weight.
        # The contaminated catalog is represented as original galaxies with
        # weight_cont; decontamination must act on that contaminated sample,
        # not on the original clean galaxies.
        weight_decontaminated = weight_cont * weight_recovered

        _, w_rec = _measure_wtheta(
            ra, dec, ra_rand, dec_rand,
            weights=weight_decontaminated,
            min_sep=min_sep, max_sep=max_sep, nbins=nbins,
        )
        result[f"w_recovered_{method}"] = w_rec
        params = {
            "a_hat": res.get("a_hat"),
            "b_hat": res.get("b_hat"),
            "elapsed_s": res.get("elapsed_s"),
        }
        # ISD's amplitude is the linear projection of the fitted curve, so on a
        # non-linear response it is blind to exactly the thing the higher orders
        # buy.  The curvature lives in the per-step coefficients; keep them, and
        # the stopping record that says whether the run was calibrated at all.
        if res.get("isd_steps") is not None:
            params.update({
                "isd_poly_order": res.get("isd_poly_order"),
                "n_steps": res.get("n_iterations"),
                "stopped_on": res.get("isd_stopped_on"),
                "calibrated": res.get("isd_calibrated"),
                "n_floored": res.get("isd_n_floored"),
                "significance": res.get("isd_significance"),
                "steps": [
                    {"template": int(st["template"]),
                     "significance": float(st["significance"]),
                     "coeffs": np.asarray(st["coeffs"], dtype=float).tolist()}
                    for st in res["isd_steps"]
                ],
            })
        result[f"params_{method}"] = params

    return result
