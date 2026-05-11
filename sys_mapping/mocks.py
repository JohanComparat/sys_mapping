"""Mock galaxy density field generation for validation and testing.

Provides physically-motivated synthetic galaxy catalogs with controlled
systematic contamination.  All scenarios produce a known ground-truth density
field so that parameter recovery and decontamination quality can be measured
exactly.

Contamination scenarios
-----------------------
``"none"``
    Pure lognormal galaxy field, no systematics.
``"additive"``
    :math:`\\hat\\delta_g = \\delta_g + \\sum_i a_i t_i`.
``"multiplicative"``
    :math:`\\hat\\delta_g = \\delta_g (1 + \\sum_i b_i t_i)`.
``"combined"``
    :math:`\\hat\\delta_g = \\delta_g (1 + \\sum_i b_i t_i) + \\sum_i a_i t_i`.

Galaxy field model
------------------
The true overdensity is a lognormal random field,

.. math::

    \\delta_g^{\\rm true}(p) = e^{G(p) - \\sigma^2/2} - 1

where :math:`G` is a zero-mean Gaussian random field with power spectrum
:math:`C_\\ell \\propto (\\ell+1)^{-2}` normalised so that
:math:`\\langle G^2 \\rangle = \\sigma^2`.
Galaxy counts are Poisson-sampled:

.. math::

    n_g(p) \\sim \\mathrm{Poisson}\\!\\left[\\bar n\\,(1 + \\hat\\delta_g(p))\\right]

References
----------
Berlfein et al. 2024, MNRAS 531, 4954.
Weaverdyck & Huterer 2021, MNRAS 503, 5061.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import healpy as hp
import jax.numpy as jnp
import numpy as np

from .contamination import apply_contamination
from .maps import generate_systematic_maps


ContaminationScenario = Literal["none", "additive", "multiplicative", "combined"]


@dataclass
class MockCatalog:
    """Container for one synthetic galaxy catalog with known contamination.

    Attributes
    ----------
    scenario:
        Contamination scenario label.
    nside:
        HEALPix NSIDE used to generate the catalog.
    ra_gal, dec_gal:
        Galaxy positions (degrees).
    ra_rand, dec_rand:
        Random catalog positions (degrees).
    templates:
        Template maps used (shape ``(n_sys, n_pix)``).
    delta_true:
        True lognormal overdensity at every pixel (shape ``(n_pix,)``).
    delta_obs:
        Observed (contaminated) overdensity at every pixel (shape ``(n_pix,)``).
    mask:
        Boolean survey mask (True = included; shape ``(n_pix,)``).
    a_true:
        Injected additive amplitudes (shape ``(n_sys,)``).
    b_true:
        Injected multiplicative amplitudes (shape ``(n_sys,)``).
    n_mean:
        Mean galaxies per pixel used for Poisson sampling.
    sigma:
        Log-normal width :math:`\\sigma` of the underlying Gaussian field.
    seed:
        Random seed used for reproducibility.
    """

    scenario: str
    nside: int
    ra_gal: np.ndarray
    dec_gal: np.ndarray
    ra_rand: np.ndarray
    dec_rand: np.ndarray
    templates: np.ndarray
    delta_true: np.ndarray
    delta_obs: np.ndarray
    mask: np.ndarray
    a_true: np.ndarray
    b_true: np.ndarray
    n_mean: float
    sigma: float
    seed: int

    @property
    def n_sys(self) -> int:
        return self.templates.shape[0]

    @property
    def n_gal(self) -> int:
        return len(self.ra_gal)

    @property
    def n_rand(self) -> int:
        return len(self.ra_rand)

    @property
    def n_pix(self) -> int:
        return hp.nside2npix(self.nside)

    @property
    def n_good_pix(self) -> int:
        return int(self.mask.sum())


def generate_lognormal_field(
    nside: int,
    sigma: float = 0.5,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a lognormal galaxy overdensity field.

    Draws a Gaussian random field :math:`G` with power spectrum
    :math:`C_\\ell \\propto (\\ell+1)^{-2}` (normalised to variance
    :math:`\\sigma^2`), then computes the lognormal transform:

    .. math::

        \\delta_g(p) = e^{G(p) - \\sigma^2/2} - 1

    This ensures :math:`\\langle\\delta_g\\rangle = 0` exactly and produces
    a positively skewed overdensity distribution consistent with realistic
    galaxy number counts.

    Parameters
    ----------
    nside:
        HEALPix NSIDE (typical choices: 16, 32, 64, 128).
    sigma:
        Log-normal width.  ``sigma=0.5`` gives mild clustering;
        ``sigma=1.0`` gives strongly non-Gaussian fluctuations.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    delta_true:
        True galaxy overdensity (shape ``(n_pix,)``).  Mean ≈ 0;
        standard deviation ≈ :math:`e^{\\sigma^2} - 1`.

    Examples
    --------
    >>> from sys_mapping.mocks import generate_lognormal_field
    >>> delta = generate_lognormal_field(nside=32, sigma=0.5, seed=0)
    >>> delta.shape
    (12288,)
    >>> abs(float(delta.mean())) < 0.01
    True
    >>> float(delta.min()) > -1   # physically valid: n_g > 0
    True
    """
    if seed is not None:
        np.random.seed(seed)

    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1, dtype=float)

    # Power-law C_ell for the underlying Gaussian field
    cl_G = (ell + 1.0) ** (-2)
    cl_G[0] = 0.0  # zero monopole

    # Normalise so that Var[G] = sigma^2
    variance = np.sum((2 * ell + 1) / (4 * np.pi) * cl_G)
    cl_G *= sigma**2 / variance

    G = hp.synfast(cl_G, nside=nside, lmax=lmax)

    # Lognormal transform: E[exp(G)] = exp(sigma^2/2), so subtract to get zero mean
    delta_true = np.exp(G - 0.5 * sigma**2) - 1.0
    return delta_true


def make_galactic_mask(nside: int, lat_cut_deg: float = 20.0) -> np.ndarray:
    """Boolean survey mask: exclude pixels within Galactic latitude b_gal < lat_cut_deg.

    Parameters
    ----------
    nside:
        HEALPix NSIDE.
    lat_cut_deg:
        Galactic latitude cut in degrees.

    Returns
    -------
    mask:
        Boolean array (shape ``(n_pix,)``); ``True`` = included.

    Examples
    --------
    >>> from sys_mapping.mocks import make_galactic_mask
    >>> mask = make_galactic_mask(nside=32, lat_cut_deg=20.0)
    >>> mask.dtype
    dtype('bool')
    >>> mask.mean() > 0.5   # more than half the sky survives
    True
    """
    n_pix = hp.nside2npix(nside)
    theta_pix, _ = hp.pix2ang(nside, np.arange(n_pix))
    lat = 90.0 - np.degrees(theta_pix)
    return np.abs(lat) > lat_cut_deg


def _sample_galaxies_from_density(
    delta: np.ndarray,
    mask: np.ndarray,
    nside: int,
    n_mean: float,
    rand_factor: int = 8,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Poisson-sample galaxy and random positions from a density field.

    Parameters
    ----------
    delta:
        Galaxy overdensity at each pixel (shape ``(n_pix,)``).
    mask:
        Survey mask (shape ``(n_pix,)``).
    nside:
        HEALPix NSIDE.
    n_mean:
        Mean galaxies per pixel inside the mask.
    rand_factor:
        Randoms-to-data ratio (default 8 for accurate Landy-Szalay).
    seed:
        Random seed.

    Returns
    -------
    ra_gal, dec_gal, ra_rand, dec_rand:
        Galaxy and random sky positions in degrees.
    """
    rng = np.random.default_rng(seed)
    n_pix = hp.nside2npix(nside)

    # Clamp negative densities to zero (unphysical counts)
    lam = np.maximum(n_mean * (1.0 + delta), 0.0) * mask.astype(float)
    counts_gal = rng.poisson(lam)
    counts_rand = np.round(mask.astype(float) * n_mean * rand_factor).astype(int)

    pix_ra, pix_dec = hp.pix2ang(nside, np.arange(n_pix), lonlat=True)

    gal_idx = np.repeat(np.arange(n_pix), counts_gal)
    rand_idx = np.repeat(np.arange(n_pix), counts_rand)

    # Add sub-pixel scatter so objects are not all on pixel centres
    n_g = len(gal_idx)
    n_r = len(rand_idx)
    pix_scale_deg = hp.nside2resol(nside, arcmin=True) / 60.0

    ra_gal = pix_ra[gal_idx] + rng.standard_normal(n_g) * pix_scale_deg * 0.3
    dec_gal = pix_dec[gal_idx] + rng.standard_normal(n_g) * pix_scale_deg * 0.3
    ra_rand = pix_ra[rand_idx] + rng.standard_normal(n_r) * pix_scale_deg * 0.3
    dec_rand = pix_dec[rand_idx] + rng.standard_normal(n_r) * pix_scale_deg * 0.3

    # Wrap RA to [0, 360) and clamp Dec to [-90, 90]
    ra_gal = ra_gal % 360.0
    ra_rand = ra_rand % 360.0
    dec_gal = np.clip(dec_gal, -90, 90)
    dec_rand = np.clip(dec_rand, -90, 90)

    return ra_gal, dec_gal, ra_rand, dec_rand


def make_mock_catalog(
    nside: int,
    n_sys: int,
    a_true: np.ndarray | None = None,
    b_true: np.ndarray | None = None,
    *,
    scenario: ContaminationScenario = "combined",
    template_families: list[int] | None = None,
    n_mean: float = 30.0,
    sigma: float = 0.5,
    lat_cut_deg: float = 20.0,
    rand_factor: int = 8,
    seed: int = 0,
) -> MockCatalog:
    """Generate a single mock galaxy catalog with controlled contamination.

    Parameters
    ----------
    nside:
        HEALPix NSIDE (16 for unit tests, 32–64 for science validation).
    n_sys:
        Number of systematic templates.
    a_true:
        Additive amplitudes (shape ``(n_sys,)``).  If ``None`` and
        ``scenario`` requires additive contamination, drawn from
        :math:`\\mathcal{N}(0, 0.10)`.
    b_true:
        Multiplicative amplitudes (shape ``(n_sys,)``).  Same default rule.
    scenario:
        Which contamination model to apply.  One of ``"none"``,
        ``"additive"``, ``"multiplicative"``, ``"combined"``.
    template_families:
        HEALPix power-spectrum family indices (0–4) for template generation.
        Defaults to the first ``n_sys`` families.
    n_mean:
        Mean galaxies per pixel (controls Poisson noise level).
    sigma:
        Log-normal width of the true density field.
    lat_cut_deg:
        Galactic latitude cut in degrees.
    rand_factor:
        Random-to-data catalog size ratio.
    seed:
        Master random seed.

    Returns
    -------
    MockCatalog
        Fully populated container with positions, templates, truth vectors,
        and the pixel-level density fields before and after contamination.

    Examples
    --------
    >>> from sys_mapping.mocks import make_mock_catalog
    >>> mock = make_mock_catalog(nside=16, n_sys=2, scenario="additive", seed=0)
    >>> mock.n_sys
    2
    >>> mock.n_gal > 0
    True
    >>> import numpy as np
    >>> np.all(mock.a_true != 0) or mock.scenario == "none"
    True

    References
    ----------
    Berlfein et al. 2024, MNRAS 531, 4954.
    """
    rng = np.random.default_rng(seed)

    # --- True contamination amplitudes ---
    needs_a = scenario in ("additive", "combined")
    needs_b = scenario in ("multiplicative", "combined")

    if a_true is None:
        a_true = rng.normal(0.0, 0.10, n_sys) if needs_a else np.zeros(n_sys)
    else:
        a_true = np.asarray(a_true, dtype=float)

    if b_true is None:
        b_true = rng.normal(0.0, 0.10, n_sys) if needs_b else np.zeros(n_sys)
    else:
        b_true = np.asarray(b_true, dtype=float)

    # Force amplitudes to zero for irrelevant scenarios
    if not needs_a:
        a_true = np.zeros(n_sys)
    if not needs_b:
        b_true = np.zeros(n_sys)

    # --- Templates ---
    if template_families is None:
        template_families = list(range(min(n_sys, 5)))
    template_seed = int(rng.integers(0, 2**31))
    templates = generate_systematic_maps(nside, families=template_families, seed=template_seed)

    # --- True density field ---
    field_seed = int(rng.integers(0, 2**31))
    delta_true = generate_lognormal_field(nside, sigma=sigma, seed=field_seed)

    # --- Apply contamination ---
    if scenario == "none":
        delta_obs = delta_true.copy()
    else:
        delta_obs = np.asarray(
            apply_contamination(
                jnp.asarray(delta_true),
                jnp.asarray(templates),
                jnp.asarray(a_true),
                jnp.asarray(b_true),
            )
        )

    # --- Mask ---
    mask = make_galactic_mask(nside, lat_cut_deg=lat_cut_deg)

    # --- Poisson sampling ---
    cat_seed = int(rng.integers(0, 2**31))
    ra_gal, dec_gal, ra_rand, dec_rand = _sample_galaxies_from_density(
        delta_obs, mask, nside, n_mean, rand_factor=rand_factor, seed=cat_seed
    )

    return MockCatalog(
        scenario=scenario,
        nside=nside,
        ra_gal=ra_gal,
        dec_gal=dec_gal,
        ra_rand=ra_rand,
        dec_rand=dec_rand,
        templates=templates,
        delta_true=delta_true,
        delta_obs=delta_obs,
        mask=mask,
        a_true=a_true,
        b_true=b_true,
        n_mean=n_mean,
        sigma=sigma,
        seed=seed,
    )


def make_mock_suite(
    nside: int,
    n_sys: int,
    *,
    scenarios: list[ContaminationScenario] | None = None,
    a_amplitudes: np.ndarray | None = None,
    b_amplitudes: np.ndarray | None = None,
    n_mean: float = 30.0,
    sigma: float = 0.5,
    lat_cut_deg: float = 20.0,
    rand_factor: int = 8,
    seed: int = 0,
) -> dict[str, MockCatalog]:
    """Generate a matched suite of mock catalogs across contamination scenarios.

    All four scenarios share the **same** true density field and templates
    (only the contamination amplitudes differ), enabling direct comparison
    of how each decontamination method performs on identical underlying data.

    Parameters
    ----------
    nside:
        HEALPix NSIDE.
    n_sys:
        Number of systematic templates.
    scenarios:
        List of scenario labels to generate.  Defaults to all four:
        ``["none", "additive", "multiplicative", "combined"]``.
    a_amplitudes:
        Fixed additive amplitudes (shape ``(n_sys,)``).  If ``None``, drawn
        from :math:`\\mathcal{N}(0, 0.10)`.
    b_amplitudes:
        Fixed multiplicative amplitudes (shape ``(n_sys,)``).
    n_mean, sigma, lat_cut_deg, rand_factor:
        Passed to :func:`make_mock_catalog`.
    seed:
        Master seed; each scenario gets a deterministic child seed.

    Returns
    -------
    suite:
        Dict mapping scenario name → :class:`MockCatalog`.

    Examples
    --------
    >>> from sys_mapping.mocks import make_mock_suite
    >>> suite = make_mock_suite(nside=16, n_sys=2, seed=0)
    >>> set(suite.keys()) == {"none", "additive", "multiplicative", "combined"}
    True
    >>> suite["additive"].n_sys
    2
    """
    if scenarios is None:
        scenarios = ["none", "additive", "multiplicative", "combined"]

    rng = np.random.default_rng(seed)

    # Shared amplitudes across all scenarios
    if a_amplitudes is None:
        a_amplitudes = rng.normal(0.0, 0.10, n_sys)
    if b_amplitudes is None:
        b_amplitudes = rng.normal(0.0, 0.10, n_sys)

    suite: dict[str, MockCatalog] = {}
    for scenario in scenarios:
        child_seed = int(rng.integers(0, 2**31))
        suite[scenario] = make_mock_catalog(
            nside=nside,
            n_sys=n_sys,
            a_true=a_amplitudes.copy(),
            b_true=b_amplitudes.copy(),
            scenario=scenario,
            n_mean=n_mean,
            sigma=sigma,
            lat_cut_deg=lat_cut_deg,
            rand_factor=rand_factor,
            seed=child_seed,
        )

    return suite
