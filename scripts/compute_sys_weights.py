#!/usr/bin/env python3
"""
Systematic weight computation for LS10 BGS volume-limited samples.

For every DATA/RAND pair found in CATALOG_DIR, this script:
  1. Loads galaxy and random catalogs.
  2. Loads and normalises HEALPix systematic templates.
  3. Pixelises both catalogs and computes the galaxy overdensity δ_g.
  4. Runs MCMC for the **additive** model  (b_i = 0).
  5. Runs MCMC for the **combined** model  (a_i, b_i free).
  6. Back-transforms inferred parameters from the PCA-rotated basis to the
     original template basis.
  7. Computes per-pixel systematic weights for each model and assigns them to
     individual galaxies by pixel look-up.
  8. Writes a FITS weight file (same row order as DATA, one column per model).
  9. Writes a JSON metadata file with MAP parameters and chain diagnostics.
 10. Appends the sample to a YAML summary catalogue consumed by sum_stat.

Weighting scheme
----------------
Contamination model (Berlfein et al. 2024, Eq. 11-13):

    δ_g_obs(p) = δ_g_true(p) · (1 + Σ_i b_i t_i(p)) + Σ_i a_i t_i(p)

Per-pixel weights that approximately invert the contamination:

    WEIGHT_ADD(p)  = 1 / max(1 + Σ_i a_i_add  · t_i(p), ε)   [additive model]
    WEIGHT_COMB(p) = 1 / max(1 + Σ_i b_i_comb · t_i(p), ε)   [combined model]

where a_i_add are the MAP additive parameters (additive MCMC) and b_i_comb are
the MAP multiplicative parameters (combined MCMC), both in the original
(un-rotated) template basis.  Galaxies whose pixel has no valid template
coverage receive weight 1.0.

Output naming convention
------------------------
  {OUTPUT_DIR}/{sample_id}_NSIDE{nside:04d}_WEIGHTS.fits
  {OUTPUT_DIR}/{sample_id}_NSIDE{nside:04d}_params.json
  {OUTPUT_DIR}/summary_NSIDE{nside:04d}.yaml

GPU usage
---------
Use --device gpu  (or set JAX_DEVICE=gpu in the environment).  The JAX
log-likelihood is JIT-compiled and runs on whatever device JAX targets.
Combined with --vectorize (the default on GPU), all emcee walkers are
evaluated in a single jax.vmap kernel per step instead of n_walkers
sequential dispatches — this is the dominant source of GPU speedup.

Device selection notes
~~~~~~~~~~~~~~~~~~~~~~
* ``--device cpu`` sets ``JAX_PLATFORMS=cpu`` before JAX initialises so the
  GPU is skipped even when ``jax[cuda]`` is installed.
* ``--device gpu`` does **not** set ``JAX_PLATFORMS``.  JAX auto-detects the
  GPU when ``jax[cuda12_pip]`` (or equivalent) is installed; it silently falls
  back to CPU otherwise.  Setting ``JAX_PLATFORMS=gpu`` explicitly crashes when
  CUDA JAX is absent, so we avoid it.
* A clear warning is printed when GPU was requested but the active backend is
  not GPU, along with the ``pip install`` command to fix it.
"""

from __future__ import annotations

# ── Device selection (must precede any JAX import) ───────────────────────────
import os as _os
import sys as _sys

def _detect_device() -> str:
    """Extract --device from argv without argparse."""
    argv = _sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == '--device' and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith('--device='):
            return arg.split('=', 1)[1]
    return _os.environ.get('JAX_DEVICE', 'auto')

_JAX_DEVICE = _detect_device()
if _JAX_DEVICE == 'cpu':
    # Force CPU even when CUDA JAX is installed.
    _os.environ.setdefault('JAX_PLATFORMS', 'cpu')
# For 'gpu'/'cuda'/'auto': do NOT set JAX_PLATFORMS.
# Setting JAX_PLATFORMS=gpu crashes when CUDA JAX is not installed.
# If jax[cuda] is installed JAX auto-selects the GPU without any env var;
# if it is not installed JAX falls back to CPU gracefully.

# ── Standard imports (non-JAX) ───────────────────────────────────────────────
import argparse
import json
import warnings
from pathlib import Path

import healpy as hp
import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── JAX-dependent imports (device already configured above) ──────────────────
import sys_mapping as sm
from sys_mapping.contamination import unpack_params
from sys_mapping.correction import transform_params_from_rotated

# ── NSIDE zero-padding widths for template filename discovery ─────────────────
_NSIDE_ZFILL: dict[str, int] = {
    "LS10": 4,   # LS10_EBV_NSIDE_0064.fits
    "GAIA": 5,   # GAIA_nstar_bright_NSIDE_00064.fits
}

# ── Minimum weight denominator to avoid division by ≈0 ───────────────────────
_WEIGHT_EPSILON = 0.01


# ---------------------------------------------------------------------------
# Template loading  (mirrors the notebook cell 9 helper)
# ---------------------------------------------------------------------------

def load_templates(
    template_dir: Path,
    nside: int,
    sources: list[str],
    norm_method: str = "standardize",
) -> tuple[np.ndarray, list[str]]:
    """Return (n_sys, n_pix) float64 template array and a list of short names."""
    fits_files: list[Path] = []
    for src in sources:
        zfill = _NSIDE_ZFILL.get(src, 4)
        nside_str = str(nside).zfill(zfill)
        fits_files.extend(sorted(template_dir.glob(f"{src}_*_NSIDE_{nside_str}.fits")))

    if not fits_files:
        raise FileNotFoundError(
            f"No FITS templates found in {template_dir} for sources={sources}, NSIDE={nside}."
        )

    maps, names = [], []
    for f in fits_files:
        m = hp.read_map(str(f), field=0, verbose=False)
        if len(m) != hp.nside2npix(nside):
            m = hp.ud_grade(m, nside_out=nside)
        valid = m != hp.UNSEEN
        if not valid.any():
            continue
        if norm_method == "minmax":
            lo, hi = m[valid].min(), m[valid].max()
            m[valid] = (m[valid] - lo) / (hi - lo if hi > lo else 1.0)
        else:  # standardize
            mu, sig = m[valid].mean(), m[valid].std()
            m[valid] = (m[valid] - mu) / (sig if sig > 0 else 1.0)
        maps.append(m)
        stem = f.stem
        for src in sources:
            if stem.startswith(f"{src}_"):
                zfill = _NSIDE_ZFILL.get(src, 4)
                suffix = f"_NSIDE_{str(nside).zfill(zfill)}"
                qty = stem.removeprefix(f"{src}_").removesuffix(suffix)
                names.append(f"{src}:{qty}")
                break
        else:
            names.append(stem)

    return np.array(maps), names


# ---------------------------------------------------------------------------
# Per-galaxy weight computation
# ---------------------------------------------------------------------------

def compute_pixel_weights(
    templates: np.ndarray,
    params: np.ndarray,
    nside: int,
    epsilon: float = _WEIGHT_EPSILON,
) -> np.ndarray:
    """Return a full-sky weight map (n_pix,).

    For pixels where all templates are valid, the weight is
    1 / max(1 + Σ_i params[i] * templates[i, p], epsilon).
    Pixels with any UNSEEN template value receive weight 1.0.

    Parameters
    ----------
    templates : (n_sys, n_pix) normalised template maps
    params    : (n_sys,) contamination parameters (a or b, original basis)
    nside     : HEALPix resolution
    epsilon   : floor for the weight denominator
    """
    n_pix = hp.nside2npix(nside)
    weight_map = np.ones(n_pix, dtype=np.float64)

    # Pixels where every template has a valid value
    valid_mask = np.ones(n_pix, dtype=bool)
    for t in templates:
        valid_mask &= (t != hp.UNSEEN)

    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        return weight_map

    contamination = np.einsum("i,ij->j", params, templates[:, valid_idx])
    denom = np.maximum(1.0 + contamination, epsilon)
    weight_map[valid_idx] = 1.0 / denom
    return weight_map


def assign_galaxy_weights(
    ra: np.ndarray,
    dec: np.ndarray,
    weight_map: np.ndarray,
    nside: int,
) -> np.ndarray:
    """Look up per-pixel weight for each galaxy.

    Parameters
    ----------
    ra, dec    : galaxy sky positions (degrees)
    weight_map : (n_pix,) weight map from compute_pixel_weights
    nside      : HEALPix resolution (RING ordering assumed)
    """
    pix = hp.ang2pix(nside, ra, dec, lonlat=True, nest=False)
    return weight_map[pix]


# ---------------------------------------------------------------------------
# All-method weight FITS output
# ---------------------------------------------------------------------------

_METHOD_COL: dict[str, tuple[str, str]] = {
    "OLS":        ("a_hat", "WEIGHT_OLS"),
    "ElasticNet": ("a_hat", "WEIGHT_ENET"),
    "ISD-1":      ("a_hat", "WEIGHT_ISD1"),
    "ISD-3":      ("a_hat", "WEIGHT_ISD3"),
    "MCMC-add":   ("a_hat", "WEIGHT_ADD"),
    "MCMC-comb":  ("b_hat", "WEIGHT_COMB"),
}
_METHOD_ORDER_FITS: list[str] = ["OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"]


def write_all_method_weights(
    sample_id: str,
    ra_gal: np.ndarray,
    dec_gal: np.ndarray,
    templates: np.ndarray,
    nside: int,
    all_method_results: dict,
    output_dir: Path,
    header_extras: "dict | None" = None,
) -> Path:
    """Write per-galaxy weight columns for all six methods to a FITS file.

    Parameters
    ----------
    all_method_results : dict mapping method name → dict with ``'a_hat'`` and/or ``'b_hat'``
    header_extras      : additional key/value pairs written into the FITS primary header
    """
    n_sys = templates.shape[0]
    cols = []
    for meth in _METHOD_ORDER_FITS:
        param_key, col_name = _METHOD_COL[meth]
        res = all_method_results.get(meth, {})
        p = np.asarray(res.get(param_key, np.zeros(n_sys)))
        if p.shape != (n_sys,):
            p = np.zeros(n_sys)
        wmap = compute_pixel_weights(templates, p, nside)
        wgal = assign_galaxy_weights(ra_gal, dec_gal, wmap, nside).astype(np.float32)
        cols.append(fits.Column(name=col_name, format="E", array=wgal))

    # WEIGHT_SYS = WEIGHT_COMB (recommended default)
    comb_idx = next((i for i, c in enumerate(cols) if c.name == "WEIGHT_COMB"), None)
    if comb_idx is not None:
        cols.append(fits.Column(name="WEIGHT_SYS", format="E",
                                array=cols[comb_idx].array.copy()))

    hdr = fits.Header()
    hdr["SAMPLE"] = sample_id[:68]
    hdr["NSIDE"]  = nside
    hdr["N_SYS"]  = n_sys
    hdr["W_SYS"]  = "alias for WEIGHT_COMB - recommended default"
    if header_extras:
        for k, v in header_extras.items():
            hdr[k] = v

    out_path = Path(output_dir) / f"{sample_id}_NSIDE{nside:04d}_WEIGHTS.fits"
    fits.BinTableHDU.from_columns(cols, header=hdr).writeto(str(out_path), overwrite=True)
    print(f"All-method weights written → {out_path}")
    return out_path


def _load_all_template_fits(template_dir: Path, nside: int) -> "tuple[np.ndarray, list[str]]":
    """Load every *.fits file in template_dir and standardise each map.

    Mirrors ``run_ls10_analysis.load_templates_from_dir``.
    """
    import warnings as _w
    fits_files = sorted(template_dir.glob("*.fits"))
    maps, names = [], []
    for f in fits_files:
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            m = hp.read_map(str(f), verbose=False)
        if len(m) != hp.nside2npix(nside):
            m = hp.ud_grade(m, nside)
        good = m != hp.UNSEEN
        if good.any():
            m[~good] = 0.0
            m -= np.mean(m[good])
            std = np.std(m[good])
            if std > 0:
                m /= std
        maps.append(m)
        names.append(f.stem)
    return np.array(maps), names


def collect_all_weights_from_jsons(
    catalog_dir: Path,
    template_dir: Path,
    output_dir: Path,
    nside: int,
    template_sources: list,
    norm_method: str = "standardize",
    sample_filter: "list | None" = None,
) -> None:
    """Read stored partial JSON files and write all-method FITS weight files.

    This function does not require JAX.  It is the back-end for ``--collect-weights``.
    """
    data_files = sorted(catalog_dir.glob("*_DATA.fits"))
    if not data_files:
        raise FileNotFoundError(f"No *_DATA.fits files found in {catalog_dir}")

    # Detect n_sys expected by the stored JSON files
    _expected_n_sys = None
    for _pjf_probe in sorted(output_dir.glob(f"*_NSIDE{nside:04d}_partial_*.json")):
        try:
            _pjf_data = json.loads(_pjf_probe.read_text())
            _expected_n_sys = _pjf_data.get("n_sys")
            if _expected_n_sys:
                break
        except Exception:
            pass

    # Load templates — fall back to all-FITS loading if source filter gives wrong n_sys
    templates, template_names = load_templates(
        template_dir, nside, template_sources, norm_method
    )
    if _expected_n_sys and templates.shape[0] != _expected_n_sys:
        print(f"  Note: source-filtered loading gave {templates.shape[0]} templates but "
              f"JSON files expect {_expected_n_sys}; loading all *.fits instead.")
        templates, template_names = _load_all_template_fits(template_dir, nside)

    n_sys = templates.shape[0]
    print(f"Templates loaded: {n_sys}  ({', '.join(template_names[:3])}{'…' if n_sys > 3 else ''})")

    processed = 0
    for data_path in data_files:
        sample_id = data_path.name.replace("_DATA.fits", "")
        if sample_filter is not None and sample_id not in sample_filter:
            continue

        print(f"\n{sample_id}")
        cat = Table.read(data_path)
        ra_gal  = np.asarray(cat["RA"],  dtype=float)
        dec_gal = np.asarray(cat["DEC"], dtype=float)

        all_method_results: dict = {}

        # Partial JSON files (OLS, ElasticNet, ISD-1, ISD-3, and partial MCMC)
        for pjf in sorted(output_dir.glob(f"{sample_id}_NSIDE{nside:04d}_partial_*.json")):
            try:
                pdata = json.loads(pjf.read_text())
                for meth in pdata.get("methods_run", []):
                    if meth in all_method_results or meth not in pdata:
                        continue
                    sub = pdata[meth]
                    a_hat = sub.get("a_hat", [])
                    b_hat = sub.get("b_hat", [])
                    all_method_results[meth] = {
                        "a_hat": np.array(a_hat) if a_hat else np.zeros(n_sys),
                        "b_hat": np.array(b_hat) if b_hat else np.zeros(n_sys),
                    }
            except Exception as exc:
                print(f"  [WARN] Could not load {pjf.name}: {exc}")

        # params.json for MCMC-add / MCMC-comb (new and old schema)
        params_path = output_dir / f"{sample_id}_NSIDE{nside:04d}_params.json"
        if params_path.exists():
            try:
                pd_ = json.loads(params_path.read_text())
                a_add  = pd_.get("a_hat_add")  or pd_.get("additive_model",  {}).get("a_hat",  [])
                b_comb = pd_.get("b_hat_comb") or pd_.get("combined_model", {}).get("b_hat", [])
                if "MCMC-add" not in all_method_results:
                    all_method_results["MCMC-add"] = {
                        "a_hat": np.array(a_add)  if a_add  else np.zeros(n_sys),
                        "b_hat": np.zeros(n_sys),
                    }
                if "MCMC-comb" not in all_method_results:
                    all_method_results["MCMC-comb"] = {
                        "a_hat": np.zeros(n_sys),
                        "b_hat": np.array(b_comb) if b_comb else np.zeros(n_sys),
                    }
            except Exception as exc:
                print(f"  [WARN] Could not load {params_path.name}: {exc}")

        if not all_method_results:
            print(f"  [SKIP] No result files found — skipping.")
            continue

        write_all_method_weights(
            sample_id, ra_gal, dec_gal, templates, nside,
            all_method_results, output_dir,
        )
        processed += 1

    print(f"\nDone. Processed {processed} sample(s).")


# ---------------------------------------------------------------------------
# MCMC runner with parameter extraction
# ---------------------------------------------------------------------------

def run_model(
    model: str,
    n_sys: int,
    delta_g: np.ndarray,
    delta_t_rot: np.ndarray,
    n_walkers: int,
    n_steps: int,
    n_burn: int,
    use_skewed: bool,
    rotation_matrix: np.ndarray,
    vectorize: bool = False,
    sampler: str = "auto",
    n_chains: int | None = None,
    nuts_n_warmup: int = 1000,
    nuts_n_samples: int = 1000,
    seed: int = 42,
) -> dict:
    """Run inference for one model and return a dict with extracted MAP parameters.

    ``sampler`` selects the backend (see :func:`sys_mapping.run_decontamination`):
    ``"auto"`` uses the exact analytic posterior for the additive model and
    BlackJAX NUTS for the combined / skew models; ``"emcee"`` forces the legacy
    gradient-free sampler.

    Returns
    -------
    dict with keys:
        a_hat, b_hat, sigma_hat  — MAP params in original template basis
        var_a, var_b             — posterior variances
        acceptance_fraction      — mean acceptance rate
        rhat, ess, num_divergences, sampler_backend — convergence diagnostics
        flat_chain               — (n_samples, n_dim) posterior samples
    """
    _sampler = sampler
    if _sampler == "auto":
        _sampler = "analytic" if (model == "additive" and not use_skewed) else "nuts"
    elif _sampler == "analytic" and (model != "additive" or use_skewed):
        _sampler = "nuts"  # analytic posterior only exists for additive Gaussian

    if _sampler == "analytic":
        flat_chain, sampler_obj = sm.run_additive_analytic(
            n_sys, delta_g_obs=delta_g, delta_t=delta_t_rot, seed=seed,
        )
    elif _sampler == "nuts":
        flat_chain, sampler_obj = sm.run_nuts(
            n_sys, model=model, delta_g_obs=delta_g, delta_t=delta_t_rot,
            use_skewed=use_skewed, n_chains=n_chains,
            n_warmup=nuts_n_warmup, n_samples=nuts_n_samples, seed=seed,
        )
    else:  # "emcee" — legacy gradient-free baseline
        n_dim = (2 * n_sys + 1) if model == "combined" else (n_sys + 1)
        n_w = max(n_walkers, 2 * n_dim + 2)
        flat_chain, sampler_obj = sm.run_mcmc(
            n_sys=n_sys,
            model=model,
            delta_g_obs=delta_g,
            delta_t=delta_t_rot,
            n_walkers=n_w,
            n_steps=n_steps,
            n_burn=n_burn,
            use_skewed=use_skewed,
            progress=True,
            vectorize=vectorize,
        )

    theta_hat = sm.get_mle_params(flat_chain)
    a_rot, b_rot, sigma_hat, _ = unpack_params(theta_hat, n_sys, model)
    a_hat, b_hat = transform_params_from_rotated(
        np.asarray(a_rot), np.asarray(b_rot), rotation_matrix
    )

    cov_a_rot, cov_b_rot = sm.get_param_covariance_from_chain(flat_chain, n_sys, model)
    var_a = np.diag(rotation_matrix.T @ cov_a_rot @ rotation_matrix)
    var_b = np.diag(rotation_matrix.T @ cov_b_rot @ rotation_matrix)

    return dict(
        a_hat=np.asarray(a_hat),
        b_hat=np.asarray(b_hat),
        sigma_hat=float(sigma_hat),
        var_a=var_a,
        var_b=var_b,
        acceptance_fraction=float(np.mean(sampler_obj.acceptance_fraction)),
        flat_chain=flat_chain,
        sampler_backend=_sampler,
        rhat=float(getattr(sampler_obj, "rhat", np.nan)),
        ess=float(getattr(sampler_obj, "ess", np.nan)),
        num_divergences=int(getattr(sampler_obj, "num_divergences", 0)),
    )


# ---------------------------------------------------------------------------
# OLS runner (analytic, no MCMC)
# ---------------------------------------------------------------------------

def run_ols(
    n_sys: int,
    delta_g: np.ndarray,
    delta_t_rot: np.ndarray,
    rotation_matrix: np.ndarray,
) -> dict:
    """Ordinary least-squares fit in the PCA-rotated basis.

    Covariance is estimated analytically: Var[alpha] = sigma2 * (X^T X)^{-1}
    where sigma2 = RSS / (n_pix - n_sys).

    Returns
    -------
    dict with the same keys as run_model() for uniform downstream handling.
    b_hat is always zero; acceptance_fraction and flat_chain are not applicable.
    """
    X = delta_t_rot.T  # (n_pix, n_sys)
    a_rot, _, _, _ = np.linalg.lstsq(X, delta_g, rcond=None)
    resid_vec = delta_g - X @ a_rot
    n_pix = len(delta_g)
    dof = max(n_pix - n_sys, 1)
    sigma2 = float(np.dot(resid_vec, resid_vec) / dof)

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        cov_a_rot = sigma2 * XtX_inv
    except np.linalg.LinAlgError:
        cov_a_rot = np.eye(n_sys) * sigma2

    a_hat, _ = transform_params_from_rotated(
        np.asarray(a_rot), np.zeros(n_sys), rotation_matrix
    )
    var_a = np.diag(rotation_matrix.T @ cov_a_rot @ rotation_matrix)

    return dict(
        a_hat=np.asarray(a_hat),
        b_hat=np.zeros(n_sys),
        sigma_hat=float(np.sqrt(sigma2)),
        var_a=var_a,
        var_b=np.zeros(n_sys),
        acceptance_fraction=float("nan"),
        flat_chain=None,
    )


# ---------------------------------------------------------------------------
# Main pipeline for one sample
# ---------------------------------------------------------------------------

def process_sample(
    data_path: Path,
    rand_path: Path,
    template_dir: Path,
    output_dir: Path,
    nside: int,
    n_walkers: int,
    n_steps: int,
    n_burn: int,
    use_skewed: bool,
    template_sources: list[str],
    norm_method: str,
    vectorize: bool = False,
    sampler: str = "auto",
    n_chains: int | None = None,
    nuts_n_warmup: int = 1000,
    nuts_n_samples: int = 1000,
) -> dict:
    """Run full systematic analysis for one DATA/RAND pair.

    Returns a sample-level dict suitable for the summary YAML.
    """
    sample_id = data_path.name.replace("_DATA.fits", "")
    print(f"\n{'='*70}")
    print(f"Sample: {sample_id}")
    print(f"{'='*70}")

    # ── Load catalogs ────────────────────────────────────────────────────────
    cat = Table.read(data_path)
    ra_gal = np.asarray(cat["RA"], dtype=float)
    dec_gal = np.asarray(cat["DEC"], dtype=float)
    n_gal = len(ra_gal)
    print(f"  Galaxies : {n_gal:,}")

    rand = Table.read(rand_path)
    ra_rand = np.asarray(rand["RA"], dtype=float)
    dec_rand = np.asarray(rand["DEC"], dtype=float)
    n_rand = len(ra_rand)
    print(f"  Randoms  : {n_rand:,}  ({n_rand/n_gal:.1f}×)")

    # ── Load templates ───────────────────────────────────────────────────────
    templates, template_names = load_templates(
        template_dir, nside, template_sources, norm_method
    )
    n_sys = templates.shape[0]
    print(f"  Templates: {n_sys} ({', '.join(template_names[:3])}{'…' if n_sys > 3 else ''})")

    # ── Pixelise and compute overdensity ─────────────────────────────────────
    gal_counts = sm.pixelize_catalog(ra_gal, dec_gal, nside)
    rand_counts = sm.pixelize_catalog(ra_rand, dec_rand, nside)
    delta_g, good_pix = sm.compute_overdensity(gal_counts, rand_counts)
    print(f"  Unmasked pixels: {good_pix.sum():,} / {hp.nside2npix(nside):,}")

    # ── Template values at good pixels + PCA rotation ────────────────────────
    delta_t = sm.assign_template_values(templates, good_pix)
    delta_t_rot, R, eigenvalues = sm.rotate_templates(delta_t)

    # ── OLS: fast analytic baseline ──────────────────────────────────────────
    print("\n  [0/3] OLS — analytic least-squares …")
    res_ols = run_ols(n_sys, delta_g, delta_t_rot, R)
    print(f"  σ_noise = {res_ols['sigma_hat']:.4f}")

    # ── MCMC: additive model ─────────────────────────────────────────────────
    print("\n  [1/3] MCMC — additive model …")
    res_add = run_model(
        model="additive",
        n_sys=n_sys,
        delta_g=delta_g,
        delta_t_rot=delta_t_rot,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burn=n_burn,
        use_skewed=use_skewed,
        rotation_matrix=R,
        vectorize=vectorize,
        sampler=sampler,
        n_chains=n_chains,
        nuts_n_warmup=nuts_n_warmup,
        nuts_n_samples=nuts_n_samples,
    )
    print(f"  acceptance = {res_add['acceptance_fraction']:.3f}  σ_noise = {res_add['sigma_hat']:.4f}")

    # ── MCMC: combined model ─────────────────────────────────────────────────
    print("\n  [2/3] MCMC — combined model …")
    res_comb = run_model(
        model="combined",
        n_sys=n_sys,
        delta_g=delta_g,
        delta_t_rot=delta_t_rot,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burn=n_burn,
        use_skewed=use_skewed,
        rotation_matrix=R,
        vectorize=vectorize,
        sampler=sampler,
        n_chains=n_chains,
        nuts_n_warmup=nuts_n_warmup,
        nuts_n_samples=nuts_n_samples,
    )
    print(f"  acceptance = {res_comb['acceptance_fraction']:.3f}  σ_noise = {res_comb['sigma_hat']:.4f}")

    # ── Likelihood ratio test ────────────────────────────────────────────────
    lrt = sm.likelihood_ratio_test(
        delta_g,
        delta_t_rot,
        sm.get_mle_params(res_add["flat_chain"]),
        sm.get_mle_params(res_comb["flat_chain"]),
        null_model="additive",
        alt_model="combined",
        significance=0.05,
    )
    print(f"\n  LRT  λ={lrt.lambda_lr:.2f}  p={lrt.p_value:.4f}  reject_additive={lrt.reject_null}")

    # ── Per-pixel weight maps ────────────────────────────────────────────────
    wmap_ols  = compute_pixel_weights(templates, res_ols["a_hat"],  nside)
    wmap_add  = compute_pixel_weights(templates, res_add["a_hat"],  nside)
    wmap_comb = compute_pixel_weights(templates, res_comb["b_hat"], nside)

    # ── Per-galaxy weights ───────────────────────────────────────────────────
    w_ols  = assign_galaxy_weights(ra_gal, dec_gal, wmap_ols,  nside)
    w_add  = assign_galaxy_weights(ra_gal, dec_gal, wmap_add,  nside)
    w_comb = assign_galaxy_weights(ra_gal, dec_gal, wmap_comb, nside)

    print(f"\n  WEIGHT_OLS   — mean={w_ols.mean():.4f}  std={w_ols.std():.4f}  "
          f"min={w_ols.min():.4f}  max={w_ols.max():.4f}")
    print(f"  WEIGHT_ADD   — mean={w_add.mean():.4f}  std={w_add.std():.4f}  "
          f"min={w_add.min():.4f}  max={w_add.max():.4f}")
    print(f"  WEIGHT_COMB  — mean={w_comb.mean():.4f}  std={w_comb.std():.4f}  "
          f"min={w_comb.min():.4f}  max={w_comb.max():.4f}")

    # ── Write weights FITS file ──────────────────────────────────────────────
    weights_name = f"{sample_id}_NSIDE{nside:04d}_WEIGHTS.fits"
    weights_path = output_dir / weights_name

    hdr = fits.Header()
    hdr["SAMPLE"] = sample_id
    hdr["NSIDE"] = nside
    hdr["NSYS"] = n_sys
    hdr["NGAL"] = n_gal
    hdr["NRAND"] = n_rand
    hdr["NORM"] = norm_method
    hdr["TEMPSRC"] = ",".join(template_sources)
    for i, name in enumerate(template_names):
        hdr[f"TMPL{i:03d}"] = name[:68]
    # OLS model params
    for i, v in enumerate(res_ols["a_hat"]):
        hdr[f"A_OLS{i:03d}"] = float(v)
    hdr["SIG_OLS"] = res_ols["sigma_hat"]
    # additive model params
    for i, v in enumerate(res_add["a_hat"]):
        hdr[f"A_ADD{i:03d}"] = float(v)
    hdr["SIG_ADD"] = res_add["sigma_hat"]
    hdr["ACC_ADD"] = res_add["acceptance_fraction"]
    # combined model params
    for i, v in enumerate(res_comb["a_hat"]):
        hdr[f"A_CMB{i:03d}"] = float(v)
    for i, v in enumerate(res_comb["b_hat"]):
        hdr[f"B_CMB{i:03d}"] = float(v)
    hdr["SIG_CMB"] = res_comb["sigma_hat"]
    hdr["ACC_CMB"] = res_comb["acceptance_fraction"]
    hdr["LRT_LAM"] = float(lrt.lambda_lr)
    hdr["LRT_PVL"] = float(lrt.p_value)
    hdr["LRT_REJ"] = bool(lrt.reject_null)
    # Weight scheme documentation
    hdr["W_OLS"]  = "1/max(1+sum_i a_i_ols*t_i(p), 0.01)"
    hdr["W_ADD"]  = "1/max(1+sum_i a_i_add*t_i(p), 0.01)"
    hdr["W_COMB"] = "1/max(1+sum_i b_i_comb*t_i(p), 0.01)"
    hdr["W_SYS"]  = "alias for WEIGHT_COMB - recommended default weight"

    primary = fits.PrimaryHDU(header=hdr)
    col_wols = fits.Column(name="WEIGHT_OLS",  format="D", array=w_ols)
    col_wadd = fits.Column(name="WEIGHT_ADD",  format="D", array=w_add)
    col_wcmb = fits.Column(name="WEIGHT_COMB", format="D", array=w_comb)
    col_wsys = fits.Column(name="WEIGHT_SYS",  format="D", array=w_comb)
    table_hdu = fits.BinTableHDU.from_columns([col_wols, col_wadd, col_wcmb, col_wsys])
    fits.HDUList([primary, table_hdu]).writeto(weights_path, overwrite=True)
    print(f"\n  Weights written → {weights_path}")
    print(f"  Use WEIGHT_SYS (= WEIGHT_COMB) as the recommended systematic weight.")

    # ── Write params JSON ────────────────────────────────────────────────────
    params_name = f"{sample_id}_NSIDE{nside:04d}_params.json"
    params_path = output_dir / params_name

    def _arr(x):
        return x.tolist() if hasattr(x, "tolist") else list(x)

    params_dict = {
        "sample_id": sample_id,
        "nside": nside,
        "n_sys": n_sys,
        "n_galaxies": int(n_gal),
        "n_randoms": int(n_rand),
        "template_names": template_names,
        "ols_model": {
            "a_hat": _arr(res_ols["a_hat"]),
            "var_a": _arr(res_ols["var_a"]),
            "sigma_hat": res_ols["sigma_hat"],
        },
        "additive_model": {
            "a_hat": _arr(res_add["a_hat"]),
            "var_a": _arr(res_add["var_a"]),
            "sigma_hat": res_add["sigma_hat"],
            "acceptance_fraction": res_add["acceptance_fraction"],
        },
        "combined_model": {
            "a_hat": _arr(res_comb["a_hat"]),
            "b_hat": _arr(res_comb["b_hat"]),
            "var_a": _arr(res_comb["var_a"]),
            "var_b": _arr(res_comb["var_b"]),
            "sigma_hat": res_comb["sigma_hat"],
            "acceptance_fraction": res_comb["acceptance_fraction"],
        },
        "lrt": {
            "lambda_lr": float(lrt.lambda_lr),
            "n_dof": int(lrt.n_dof),
            "p_value": float(lrt.p_value),
            "reject_additive": bool(lrt.reject_null),
        },
        "weight_scheme": {
            "WEIGHT_OLS":  "1 / max(1 + sum_i a_i_ols  * t_i(p), 0.01)",
            "WEIGHT_ADD":  "1 / max(1 + sum_i a_i_add  * t_i(p), 0.01)",
            "WEIGHT_COMB": "1 / max(1 + sum_i b_i_comb * t_i(p), 0.01)",
        },
    }
    params_path.write_text(json.dumps(params_dict, indent=2))
    print(f"  Params  written → {params_path}")

    # ── Return sample summary entry ──────────────────────────────────────────
    return {
        "sample_id": sample_id,
        "data":    str(data_path.resolve()),
        "rand":    str(rand_path.resolve()),
        "weights": str(weights_path.resolve()),
        "params":  str(params_path.resolve()),
        "n_galaxies": int(n_gal),
        "n_randoms":  int(n_rand),
        "n_sys": n_sys,
        "models": {
            "ols": {
                "weight_col": "WEIGHT_OLS",
                "weight_scheme": "1 / max(1 + sum_i a_i_ols * t_i(p), 0.01)",
                "sigma_noise": res_ols["sigma_hat"],
            },
            "additive": {
                "weight_col": "WEIGHT_ADD",
                "weight_scheme": "1 / max(1 + sum_i a_i_add * t_i(p), 0.01)",
                "sigma_noise": res_add["sigma_hat"],
                "acceptance_fraction": res_add["acceptance_fraction"],
                "sampler_backend": res_add.get("sampler_backend"),
                "rhat": res_add.get("rhat"),
                "ess": res_add.get("ess"),
                "num_divergences": res_add.get("num_divergences"),
            },
            "combined": {
                "weight_col": "WEIGHT_COMB",
                "weight_scheme": "1 / max(1 + sum_i b_i_comb * t_i(p), 0.01)",
                "sigma_noise": res_comb["sigma_hat"],
                "acceptance_fraction": res_comb["acceptance_fraction"],
                "sampler_backend": res_comb.get("sampler_backend"),
                "rhat": res_comb.get("rhat"),
                "ess": res_comb.get("ess"),
                "num_divergences": res_comb.get("num_divergences"),
                "lrt_lambda": float(lrt.lambda_lr),
                "lrt_pvalue": float(lrt.p_value),
                "reject_additive": bool(lrt.reject_null),
            },
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute per-galaxy systematic weights for LS10 BGS catalogues.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--catalog-dir",
        default=_os.path.expanduser("~/data/legacysurvey/dr10/sweep/BGS_VLIM_Mstar"),
        help="Directory containing *_DATA.fits and *_RAND.fits catalogues.",
    )
    p.add_argument(
        "--template-dir",
        default=_os.path.expanduser("~/data/legacysurvey/dr10/systematics"),
        help="Directory with HEALPix systematic template FITS files.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <repo_root>/data/sys_weights).",
    )
    p.add_argument("--nside",      type=int, default=64,   help="HEALPix NSIDE.")
    p.add_argument("--n-walkers",  type=int, default=210,  help="emcee walkers (floor is 2·n_dim+2).")
    p.add_argument("--n-steps",    type=int, default=2100, help="MCMC steps per walker.")
    p.add_argument("--n-burn",     type=int, default=200,  help="Burn-in steps to discard (emcee only).")
    p.add_argument("--sampler", default="auto",
                   choices=["auto", "analytic", "nuts", "emcee"],
                   help="Inference backend: auto (analytic additive + NUTS combined), "
                        "analytic, nuts, or emcee (legacy baseline).")
    p.add_argument("--n-chains", type=int, default=None,
                   help="Parallel NUTS chains (default: 4 on CPU, 8 on GPU).")
    p.add_argument("--nuts-warmup", type=int, default=1000,
                   help="NUTS window-adaptation steps.")
    p.add_argument("--nuts-samples", type=int, default=1000,
                   help="NUTS post-warmup draws per chain.")
    p.add_argument(
        "--template-sources",
        nargs="+",
        default=["LS10", "GAIA"],
        help="Systematic map source prefixes to load.",
    )
    p.add_argument(
        "--norm-method",
        choices=["standardize", "minmax"],
        default="standardize",
        help="Template normalisation method.",
    )
    p.add_argument(
        "--no-skewed",
        action="store_true",
        help="Disable skew-normal likelihood (use Gaussian).",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda"],
        help="JAX compute device.  'gpu'/'cuda' require jax[cuda] installed.  "
             "Must also be set before this script is imported; env var JAX_DEVICE "
             "is equivalent.  Default: auto (JAX picks the best available).",
    )
    p.add_argument(
        "--vectorize",
        action="store_true",
        default=None,
        help="Evaluate all emcee walkers in one batched jax.vmap GPU kernel per "
             "step.  Strongly recommended on GPU; default True when --device is "
             "gpu/cuda, False otherwise.",
    )
    p.add_argument(
        "--no-vectorize",
        dest="vectorize",
        action="store_false",
        help="Disable vectorized walker evaluation (scalar emcee mode).",
    )
    p.add_argument(
        "--sample",
        nargs="*",
        default=None,
        help="Process only these sample IDs (stem without _DATA.fits). "
             "Default: process all pairs found.",
    )
    p.add_argument(
        "--collect-weights",
        action="store_true",
        default=False,
        help="Read existing partial JSON files from OUTPUT_DIR and write a single "
             "all-method WEIGHTS.fits per sample.  No MCMC is run.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    catalog_dir  = Path(args.catalog_dir)
    template_dir = Path(args.template_dir)

    # Resolve output dir relative to the script's repo root
    if args.output_dir is None:
        repo_root  = Path(__file__).resolve().parent.parent
        output_dir = repo_root / "data" / "sys_weights"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── --collect-weights: assemble FITS from existing JSON files, no MCMC ───
    if args.collect_weights:
        collect_all_weights_from_jsons(
            catalog_dir=catalog_dir,
            template_dir=template_dir,
            output_dir=output_dir,
            nside=args.nside,
            template_sources=args.template_sources,
            norm_method=args.norm_method,
            sample_filter=args.sample,
        )
        return

    # ── Report active JAX backend and warn if GPU was requested but absent ────
    import jax as _jax
    actual_backend = _jax.default_backend()   # 'cpu', 'gpu', or 'tpu'
    active_device  = str(_jax.devices()[0])

    if args.device in ('gpu', 'cuda') and actual_backend != 'gpu':
        print(
            f"WARNING: --device {args.device} requested but JAX is running on "
            f"'{actual_backend}'.  CUDA-capable JAX is not installed in this "
            f"environment.\n"
            f"  To enable GPU support run:\n"
            f"    pip install 'jax[cuda12_pip]' "
            f"-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html\n"
            f"Continuing on {actual_backend.upper()}."
        )

    # ── Resolve vectorize default ────────────────────────────────────────────
    # Default True on GPU (maximises throughput), False on CPU (avoids overhead).
    if args.vectorize is None:
        vectorize = actual_backend == 'gpu'
    else:
        vectorize = args.vectorize

    print(f"JAX device : {active_device}")
    print(f"Vectorize  : {vectorize}")

    # ── Discover DATA/RAND pairs ─────────────────────────────────────────────
    data_files = sorted(catalog_dir.glob("*_DATA.fits"))
    if not data_files:
        raise FileNotFoundError(f"No *_DATA.fits files found in {catalog_dir}")

    pairs: list[tuple[Path, Path]] = []
    for df in data_files:
        rf = Path(str(df).replace("_DATA.fits", "_RAND.fits"))
        if not rf.exists():
            print(f"[WARN] No matching RAND for {df.name} — skipping.")
            continue
        sample_id = df.name.replace("_DATA.fits", "")
        if args.sample is not None and sample_id not in args.sample:
            continue
        pairs.append((df, rf))

    if not pairs:
        raise RuntimeError("No valid DATA/RAND pairs to process.")
    print(f"Found {len(pairs)} DATA/RAND pair(s) to process.")

    # ── Process each pair ────────────────────────────────────────────────────
    summary_entries: list[dict] = []
    for data_path, rand_path in pairs:
        entry = process_sample(
            data_path=data_path,
            rand_path=rand_path,
            template_dir=template_dir,
            output_dir=output_dir,
            nside=args.nside,
            n_walkers=args.n_walkers,
            n_steps=args.n_steps,
            n_burn=args.n_burn,
            use_skewed=not args.no_skewed,
            template_sources=args.template_sources,
            norm_method=args.norm_method,
            vectorize=vectorize,
            sampler=args.sampler,
            n_chains=args.n_chains,
            nuts_n_warmup=args.nuts_warmup,
            nuts_n_samples=args.nuts_samples,
        )
        summary_entries.append(entry)

    # ── Write summary YAML ───────────────────────────────────────────────────
    summary = {
        "nside":            args.nside,
        "template_dir":     str(template_dir.resolve()),
        "template_sources": args.template_sources,
        "norm_method":      args.norm_method,
        "output_dir":       str(output_dir.resolve()),
        "jax_device":       active_device,   # e.g. "CpuDevice(id=0)" or "CudaDevice(id=0)"
        "vectorized_mcmc":  vectorize,
        "recommended_weight_column": "WEIGHT_SYS",
    "weight_columns": {
            "WEIGHT_OLS":  "1 / max(1 + sum_i a_i_ols  * t_i(p), 0.01)  [OLS]",
            "WEIGHT_ADD":  "1 / max(1 + sum_i a_i_add  * t_i(p), 0.01)  [additive model]",
            "WEIGHT_COMB": "1 / max(1 + sum_i b_i_comb * t_i(p), 0.01)  [combined model]",
        },
        "samples": summary_entries,
    }

    summary_path = output_dir / f"summary_NSIDE{args.nside:04d}.yaml"
    summary_path.write_text(yaml.dump(summary, default_flow_style=False, sort_keys=False))
    print(f"\nSummary written → {summary_path}")
    print(f"\nDone. Processed {len(summary_entries)} sample(s).")


if __name__ == "__main__":
    main()
