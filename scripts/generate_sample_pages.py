"""Generate per-sample detail pages for LS10 BGS VLIM systematic analysis.

Reads params.json and partial_*.json files from data/sys_weights/ for
NSIDE 32, 64, 128, and 256 and writes one RST file per sample to docs/.
"""

import json
import sys
from pathlib import Path

import numpy as np

try:
    from astropy.io import fits
    HAS_FITS = True
except ImportError:
    HAS_FITS = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCS_DIR   = Path(__file__).parent.parent / "docs"
DATA_DIR   = Path(__file__).parent.parent / "data" / "sys_weights"
STATIC_DIR = Path(__file__).parent.parent / "docs" / "_static" / "results_ls10"

NSIDES = [32, 64, 128, 256]

SAMPLE_CONFIGS = [
    {"mstar": "9.0",  "z": "0.08", "tag": "9p0",   "n_gal": 523486,
     "sample_id": "LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486",
     "dw_w_30": -7.2, "dw_w_max": 8.4, "dw_w_rms": 5.9, "dw_w_theta_max": 23},
    {"mstar": "9.5",  "z": "0.12", "tag": "9p5",   "n_gal": 1432502,
     "sample_id": "LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502",
     "dw_w_30": -4.8, "dw_w_max": 4.9, "dw_w_rms": 3.7, "dw_w_theta_max": 15},
    {"mstar": "10.0", "z": "0.18", "tag": "10p0",  "n_gal": 2759238,
     "sample_id": "LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238",
     "dw_w_30": -0.4, "dw_w_max": 2.0, "dw_w_rms": 0.6, "dw_w_theta_max": 120},
    {"mstar": "10.25","z": "0.22", "tag": "10p25", "n_gal": 3308841,
     "sample_id": "LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841",
     "dw_w_30": None, "dw_w_max": None, "dw_w_rms": None, "dw_w_theta_max": None},
    {"mstar": "10.5", "z": "0.26", "tag": "10p5",  "n_gal": 3263228,
     "sample_id": "LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228",
     "dw_w_30": None, "dw_w_max": None, "dw_w_rms": None, "dw_w_theta_max": None},
    {"mstar": "10.75","z": "0.31", "tag": "10p75", "n_gal": 2802710,
     "sample_id": "LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710",
     "dw_w_30": None, "dw_w_max": None, "dw_w_rms": None, "dw_w_theta_max": None},
    {"mstar": "11.0", "z": "0.35", "tag": "11p0",  "n_gal": 1619838,
     "sample_id": "LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838",
     "dw_w_30": -0.1, "dw_w_max": 11.5, "dw_w_rms": 2.4, "dw_w_theta_max": 181},
    {"mstar": "11.25","z": "0.35", "tag": "11p25", "n_gal": 541855,
     "sample_id": "LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855",
     "dw_w_30": +2.2, "dw_w_max": 17.4, "dw_w_rms": 6.5, "dw_w_theta_max": 181},
    {"mstar": "11.5", "z": "0.35", "tag": "11p5",  "n_gal": 120882,
     "sample_id": "LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882",
     "dw_w_30": +0.7, "dw_w_max": 9.7, "dw_w_rms": 3.3, "dw_w_theta_max": 97},
]

METHODS = ["OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"]
METHOD_LABELS = {
    "OLS": "OLS", "ElasticNet": "ElasticNet", "ISD-1": "ISD-1",
    "ISD-3": "ISD-3 †", "MCMC-add": "MCMC-add", "MCMC-comb": "MCMC-comb",
}
WEIGHT_COLS = {
    "OLS": "WEIGHT_OLS", "ElasticNet": "WEIGHT_ENET",
    "ISD-1": "WEIGHT_ISD1", "ISD-3": "WEIGHT_ISD3",
    "MCMC-add": "WEIGHT_ADD", "MCMC-comb": "WEIGHT_COMB",
}

TEMPLATE_MEANINGS = {
    "GAIA_nstar_faint":       "GAIA faint stellar density (photometric mis-classification of faint stars as galaxies)",
    "GAIA_nstar_medium":      "GAIA medium stellar density (crowding and deblending near bright stars)",
    "GAIA_phot_g_mean_flux":  "GAIA mean stellar flux in G band (scattered-light / sky-background variations)",
    "GAIA_phot_bp_mean_flux": "GAIA mean stellar flux in BP band (blue scattered light)",
    "GAIA_phot_rp_mean_flux": "GAIA mean stellar flux in RP band (red scattered light)",
    "LS10_EBV":               "Galactic dust extinction E(B−V) — reddening shifts magnitudes and colours",
    "LS10_GALDEPTH_G":        "5σ galaxy detection depth in g band — selection completeness",
    "LS10_GALDEPTH_R":        "5σ galaxy detection depth in r band — primary BGS selection band",
    "LS10_GALDEPTH_Z":        "5σ galaxy detection depth in z band — selection completeness",
    "LS10_PSFSIZE_R":         "PSF FWHM in r band — affects star–galaxy separation quality",
    "LS10_NOBS_R":            "Number of r-band exposures — coadd depth and artefact rate",
}

NSIDE_VALUES = ["00032", "00064", "00128", "00256"]
NSIDE_NPIX   = {32: "5 600", 64: "21 600", 128: "84 000", 256: "330 000"}


def parse_template_name(name):
    """Return (short_name, nside_int, meaning) for a template name."""
    for nside in NSIDE_VALUES:
        suffix = f"_NSIDE_{nside}"
        if name.endswith(suffix):
            short = name[: -len(suffix)]
            nside_int = int(nside)
            meaning = TEMPLATE_MEANINGS.get(short, short)
            return short, nside_int, meaning
    return name, "?", name


def load_sample_data(sid, nside):
    """Load params.json + partials for a given sample and NSIDE."""
    params_path = DATA_DIR / f"{sid}_NSIDE{nside:04d}_params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else {}

    partials = {}
    combined = DATA_DIR / f"{sid}_NSIDE{nside:04d}_partial_ElasticNet_ISD-1_OLS.json"
    if combined.exists():
        d = json.loads(combined.read_text())
        for m in ["OLS", "ElasticNet", "ISD-1"]:
            partials[m] = d.get(m, {})
    else:
        for m in ["OLS", "ElasticNet", "ISD-1"]:
            p = DATA_DIR / f"{sid}_NSIDE{nside:04d}_partial_{m}.json"
            if p.exists():
                d = json.loads(p.read_text())
                partials[m] = d.get(m, {})

    isd3_path = DATA_DIR / f"{sid}_NSIDE{nside:04d}_partial_ISD-3.json"
    if isd3_path.exists():
        d = json.loads(isd3_path.read_text())
        partials["ISD-3"] = d.get("ISD-3", {})

    return params, partials


def compute_weight_stats(sid, nside=64):
    if not HAS_FITS:
        return {}
    fp = DATA_DIR / f"{sid}_NSIDE{nside:04d}_WEIGHTS.fits"
    if not fp.exists():
        return {}
    stats = {}
    with fits.open(str(fp), memmap=True) as hdul:
        data = hdul[1].data
        for m, col in WEIGHT_COLS.items():
            if col not in data.names:
                continue
            w = np.asarray(data[col], dtype=np.float64)
            w = w[np.isfinite(w)]
            stats[m] = {
                "n": len(w), "mean": float(np.mean(w)), "std": float(np.std(w)),
                "p1": float(np.percentile(w, 1)), "p5": float(np.percentile(w, 5)),
                "p50": float(np.median(w)),
                "p95": float(np.percentile(w, 95)), "p99": float(np.percentile(w, 99)),
            }
    return stats


def _sh(params, partials, method):
    """Get sigma_hat for a method from params or partials."""
    if method == "MCMC-add":
        return params.get("sigma_hat_add")
    if method == "MCMC-comb":
        return params.get("sigma_hat_comb")
    key_map = {"OLS": "sigma_hat_ols", "ElasticNet": "sigma_hat_enet",
               "ISD-1": "sigma_hat_isd1", "ISD-3": "sigma_hat_isd3"}
    if method in key_map:
        v = params.get(key_map[method])
        if v is not None:
            return v
    return partials.get(method, {}).get("sigma_hat")


def _fmt_lrt(lrt_dict):
    lam = lrt_dict.get("lambda_lr", float("nan"))
    pval = lrt_dict.get("p_value", float("nan"))
    dof = lrt_dict.get("n_dof", 0)
    rej = lrt_dict.get("reject_null", False)
    lam_s = f"{lam:.1f}" if lam == lam else "n/a"
    if pval != pval or pval is None:
        p_s = "n/a"
    elif pval == 0.0:
        p_s = "< 10\ :sup:`-300`"
    elif pval < 1e-10:
        p_s = f"< 10\ :sup:`{int(np.floor(np.log10(pval)))}`"
    else:
        p_s = f"{pval:.2e}"
    rej_s = "**Yes**" if rej else "No"
    return lam_s, dof, p_s, rej_s


def _fmt_lrt_inline(lrt_dict):
    lam = lrt_dict.get("lambda_lr", float("nan"))
    pval = lrt_dict.get("p_value", float("nan"))
    dof = lrt_dict.get("n_dof", 0)
    rej = lrt_dict.get("reject_null", False)
    lam_s = f"{lam:.1f}" if lam == lam else "n/a"
    rej_s = "**Reject H₀**" if rej else "No rejection"
    if pval != pval or pval is None:
        p_s = "n/a"
    elif pval == 0.0:
        p_s = "≈ 0"
    elif pval < 0.001:
        p_s = f"{pval:.1e}"
    else:
        p_s = f"{pval:.3f}"
    return f":math:`\\lambda_{{\\rm LR}} = {lam_s}` (dof = {dof}), p = {p_s} → {rej_s}"


def grid_fig_block(srcs, captions, alt_prefix, title=None):
    """2×2 HTML grid of figures. srcs and captions are lists of 4 items."""
    html = [".. raw:: html", ""]
    inner = []
    if title:
        inner.append(
            f'   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">{title}</p>'
        )
    inner.append(
        '   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;'
        'max-width:1100px;margin:auto">'
    )
    for src, cap in zip(srcs, captions):
        inner.append('     <figure style="text-align:center;margin:0">')
        inner.append(f'       <a href="{src}" target="_blank">')
        inner.append(f'         <img src="{src}" style="width:100%" alt="{alt_prefix} {cap}">')
        inner.append('       </a>')
        inner.append(f'       <figcaption style="font-size:0.82em;color:#555">{cap}</figcaption>')
        inner.append('     </figure>')
    inner.append('   </div>')
    html.extend(inner)
    html.append("")
    return "\n".join(html)


def cosmological_verdict(cfg, params64):
    mstar = float(cfg["mstar"])
    dw30 = cfg["dw_w_30"]
    dw_max = cfg["dw_w_max"]
    lrt = params64.get("lrt", {})
    lrt_reject = lrt.get("reject_null", False)
    lrt_str = f"{lrt.get('lambda_lr', 0):.1f}"

    if dw30 is not None and abs(dw30) > 4:
        sub_deg_regime = "strongly systematics-dominated"
        bias_verdict = "**not suitable** without correction"
        bias_after = "**suitable** after applying ``WEIGHT_COMB``"
    elif dw30 is not None and abs(dw30) > 1:
        sub_deg_regime = "moderately contaminated"
        bias_verdict = "**borderline** without correction"
        bias_after = "**suitable** after applying ``WEIGHT_COMB``"
    elif dw30 is not None:
        sub_deg_regime = "sub-percent contamination (below noise floor)"
        bias_verdict = "**suitable** even without correction"
        bias_after = "**suitable** — correction provides marginal improvement"
    else:
        sub_deg_regime = "unknown (no empirical two-point measurement available)"
        bias_verdict = "uncertain"
        bias_after = "**suitable** after applying ``WEIGHT_COMB``"

    large_angle_concern = dw_max is not None and dw_max > 5

    lines = []
    lines.append("Cosmological analysis verdict")
    lines.append("-----------------------------")
    lines.append("")
    lines.append(
        f"Sub-degree scales (:math:`\\theta < 30'`): "
        f"regime is **{sub_deg_regime}** "
        f"(:math:`\\delta w/w \\approx {cfg['dw_w_30']:+.1f}\\%` at 30 arcmin)."
        if dw30 is not None else
        f"Sub-degree scales: {sub_deg_regime}."
    )
    lines.append("")
    lines.append(
        f"* Without correction: {bias_verdict}."
    )
    lines.append(
        f"* After correction: {bias_after}."
    )
    if large_angle_concern:
        lines.append(
            f"* **Large-angle warning** (:math:`\\theta > 2°`): max correction "
            f"{dw_max:.1f}% at {cfg['dw_w_theta_max']} arcmin — "
            f"any analysis using angular scales > 1° **must** apply ``WEIGHT_COMB``."
        )
    lines.append("")
    lines.append(
        f"LRT (NSIDE 64): {_fmt_lrt_inline(lrt)} — "
        + ("multiplicative contamination is statistically detected." if lrt_reject
           else "multiplicative contamination not detected at 5%.")
    )
    lines.append("")
    lines.append("**Recommendation**: use ``WEIGHT_COMB`` (``WEIGHT_SYS``) for all science-grade analyses.")
    lines.append("")
    return "\n".join(lines)


def generate_page(cfg, data):
    """
    Parameters
    ----------
    cfg   : sample config dict
    data  : {nside: (params, partials)} for nside in NSIDES
    """
    sid   = cfg["sample_id"]
    mstar = cfg["mstar"]
    z_max = cfg["z"]
    n_gal = cfg["n_gal"]

    params64, partials64 = data.get(64, ({}, {}))
    params32, partials32 = data.get(32, ({}, {}))

    n_pix64 = params64.get("n_good_pix", "?")
    n_pix32 = params32.get("n_good_pix", "?")
    template_names = params64.get("template_names", params32.get("template_names", []))
    n_sys = len(template_names)

    a_hat_add = np.array(params64.get("a_hat_add", np.zeros(n_sys)))
    b_hat_comb = np.array(params64.get("b_hat_comb", np.zeros(n_sys)))
    a_hat_ols = np.array(partials64.get("OLS", {}).get("a_hat", np.zeros(n_sys)))

    lrt64 = params64.get("lrt", {})
    lrt32 = params32.get("lrt", {})
    lrt_str64 = f"{lrt64.get('lambda_lr', float('nan')):.1f}"

    # ── Title + header ─────────────────────────────────────────────────────
    title = f"BGS VLIM log M\\ :sub:`*` ≥ {mstar},  z < {z_max} — detailed systematic analysis"
    underline = "=" * (len(title) + 5)

    lines = []
    lines.append(f".. _sample-{cfg['tag']}:")
    lines.append("")
    lines.append(title)
    lines.append(underline)
    lines.append("")

    lrt_reject64 = lrt64.get("reject_null", False)
    if float(mstar) < 10:
        overview = (
            f"One of the two most contaminated BGS VLIM samples.  "
            f"With {n_gal:,} galaxies at z < {z_max}, faint stellar contamination "
            f"dominates the systematic budget.  "
            f"The LRT strongly detects multiplicative contamination at both resolutions "
            f"(:math:`\\lambda_{{\\rm LR}} = {lrt_str64}` at NSIDE 64)."
        )
    elif float(mstar) < 11:
        overview = (
            f"Intermediate-mass BGS VLIM sample ({n_gal:,} galaxies, z < {z_max}).  "
            f"The LRT " + ("rejects" if lrt_reject64 else "does not reject") +
            f" the additive null at NSIDE 64 "
            f"(:math:`\\lambda_{{\\rm LR}} = {lrt_str64}`)."
        )
    else:
        overview = (
            f"High-mass BGS VLIM sample ({n_gal:,} galaxies, z < {z_max}).  "
            f"Shot noise dominates per-pixel statistics at both resolutions.  "
            f"The LRT " + ("rejects" if lrt_reject64 else "does not reject") +
            f" the additive null at NSIDE 64 "
            f"(:math:`\\lambda_{{\\rm LR}} = {lrt_str64}`)."
        )

    lines.append(overview)
    lines.append("")
    lines.append(".. contents:: On this page")
    lines.append("   :local:")
    lines.append("   :depth: 1")
    lines.append("")
    lines.append(".. seealso::")
    lines.append("")
    lines.append("   :doc:`results_ls10` — summary tables and figures for all nine samples.")
    lines.append("")
    lines.append("----")
    lines.append("")

    # ── Sample statistics ──────────────────────────────────────────────────
    lines.append("Sample statistics")
    lines.append("-----------------")
    lines.append("")
    lines.append(".. csv-table::")
    lines.append('   :header: "Parameter", "NSIDE 32", "NSIDE 64", "NSIDE 128", "NSIDE 256"')
    lines.append("   :widths: 36, 16, 16, 16, 16")
    lines.append("")
    lines.append(f'   "Stellar-mass threshold", "log M* ≥ {mstar}", "log M* ≥ {mstar}", "log M* ≥ {mstar}", "log M* ≥ {mstar}"')
    lines.append(f'   "Redshift limit", "z < {z_max}", "z < {z_max}", "z < {z_max}", "z < {z_max}"')
    lines.append(f'   "N\\ :sub:`gal`", "{n_gal:,}", "{n_gal:,}", "{n_gal:,}", "{n_gal:,}"')

    npix_vals = []
    for ns in NSIDES:
        p = data.get(ns, ({}, {}))[0]
        npix = p.get("n_good_pix", "?")
        npix_vals.append(f"{npix:,}" if isinstance(npix, int) else str(npix))
    lines.append(f'   "N\\ :sub:`pix` (good footprint)", "{npix_vals[0]}", "{npix_vals[1]}", "{npix_vals[2]}", "{npix_vals[3]}"')

    nsys_vals = []
    for ns in NSIDES:
        p = data.get(ns, ({}, {}))[0]
        nsys_vals.append(str(p.get("n_sys", n_sys)))
    lines.append(f'   "N\\ :sub:`templates`", "{nsys_vals[0]}", "{nsys_vals[1]}", "{nsys_vals[2]}", "{nsys_vals[3]}"')
    lines.append(f'   "MCMC walkers", "210", "210", "210", "210"')
    lines.append(f'   "MCMC steps after burn-in", "1500", "1500", "1500", "1500"')
    lines.append("")
    lines.append("----")
    lines.append("")

    # ── Goodness-of-fit — σ̂ comparison table ─────────────────────────────
    lines.append("Goodness-of-fit: :math:`\\hat{\\sigma}` by method and resolution")
    lines.append("------------------------------------------------------------------")
    lines.append("")
    lines.append(
        "The noise parameter :math:`\\hat{\\sigma}` measures residual scatter "
        "after systematic subtraction — lower is better.  "
        "Results are shown for NSIDE 32, 64, 128, and 256.  "
        "ISD-3 is unavailable at NSIDE 128 and 256 (no partial files generated at those resolutions)."
    )
    lines.append("")
    lines.append(".. csv-table::")
    lines.append(
        '   :header: "Method", '
        '":math:`\\hat{\\sigma}` (N32)", '
        '":math:`\\hat{\\sigma}` (N64)", '
        '":math:`\\hat{\\sigma}` (N128)", '
        '":math:`\\hat{\\sigma}` (N256)", '
        '"Notes"'
    )
    lines.append("   :widths: 14, 12, 12, 12, 12, 38")
    lines.append("")

    method_notes = {
        "OLS":       "closed-form least-squares",
        "ElasticNet":"L1+L2 regularised; 3-fold CV",
        "ISD-1":     "iterative self-calibration; poly order 1",
        "ISD-3":     "† degree-3 polynomial; unavailable at NSIDE 128/256",
        "MCMC-add":  "",
        "MCMC-comb": "",
    }
    acc_add  = {ns: data.get(ns, ({}, {}))[0].get("acceptance_fraction_add")  for ns in NSIDES}
    acc_comb = {ns: data.get(ns, ({}, {}))[0].get("acceptance_fraction_comb") for ns in NSIDES}
    acc32_add  = acc_add[32];  acc64_add  = acc_add[64]
    acc32_comb = acc_comb[32]; acc64_comb = acc_comb[64]
    if acc32_add is not None:
        method_notes["MCMC-add"]  = (f"MCMC additive; acc N32={acc32_add:.3f}"
                                      + (f" N64={acc64_add:.3f}" if acc64_add else ""))
        method_notes["MCMC-comb"] = (f"MCMC combined; acc N32={acc32_comb:.3f}"
                                      + (f" N64={acc64_comb:.3f}" if acc64_comb else ""))

    for m in METHODS:
        sh_vals = []
        for ns in NSIDES:
            p, pt = data.get(ns, ({}, {}))
            sh = _sh(p, pt, m)
            sh_vals.append(f"{sh:.4f}" if sh is not None else "—")
        note = method_notes.get(m, "")
        lines.append(
            f'   "{METHOD_LABELS[m]}", '
            f'"{sh_vals[0]}", "{sh_vals[1]}", "{sh_vals[2]}", "{sh_vals[3]}", '
            f'"{note}"'
        )

    lines.append("")
    lines.append("† ISD-3 uses a degree-3 polynomial expansion and is ill-conditioned with correlated templates.")
    lines.append("")

    # ── LRT ───────────────────────────────────────────────────────────────
    lines.append("Likelihood Ratio Test (additive vs combined model)")
    lines.append("--------------------------------------------------")
    lines.append("")
    lines.append(".. csv-table::")
    lines.append('   :header: "Resolution", ":math:`\\lambda_{\\rm LR}`", "dof", "p-value", "Reject H₀"')
    lines.append("   :widths: 14, 14, 8, 22, 12")
    lines.append("")
    for ns in NSIDES:
        p, _ = data.get(ns, ({}, {}))
        lrt_d = p.get("lrt", {})
        if not lrt_d:
            continue
        lam_s, dof, p_s, rej_s = _fmt_lrt(lrt_d)
        lines.append(f'   "NSIDE {ns}", "{lam_s}", "{dof}", "{p_s}", "{rej_s}"')
    lines.append("")
    lines.append("MCMC acceptance fractions:")
    if acc32_add:
        parts = []
        for ns in NSIDES:
            aa = acc_add[ns];  ac = acc_comb[ns]
            if aa is not None:
                parts.append(f"NSIDE {ns}: add {aa:.3f}, comb {ac:.3f}")
        lines.append("  ".join(parts) + ".  Healthy range: 0.15–0.50.")
    lines.append("")
    lines.append("----")
    lines.append("")

    # ── Template amplitude rankings (NSIDE 64 primary) ────────────────────
    lines.append("Template amplitude ranking — additive model (MCMC-add, NSIDE 64)")
    lines.append("------------------------------------------------------------------")
    lines.append("")
    lines.append(
        f"All {n_sys} templates sorted by absolute MCMC-add additive amplitude "
        ":math:`|\\hat{a}_i|`.  OLS shown for comparison."
    )
    lines.append("")
    lines.append(".. csv-table::")
    lines.append(
        '   :header: "Rank", "Template", "NSIDE", '
        '":math:`\\hat{a}_i` (MCMC-add)", ":math:`\\hat{a}_i` (OLS)", "Physical meaning"'
    )
    lines.append("   :widths: 5, 28, 6, 14, 14, 50")
    lines.append("   :stub-columns: 1")
    lines.append("")
    if n_sys > 0:
        idx = np.argsort(-np.abs(a_hat_add))
        for rank, i in enumerate(idx, 1):
            tname = template_names[i] if i < len(template_names) else f"T{i}"
            short, nside_t, meaning = parse_template_name(tname)
            a_add = float(a_hat_add[i]) if i < len(a_hat_add) else 0.0
            a_ols = float(a_hat_ols[i]) if i < len(a_hat_ols) else 0.0
            bold = rank <= 5
            short_fmt = f"**{short}**" if bold else short
            lines.append(
                f'   "{rank}", "{short_fmt}", "{nside_t}", '
                f'"{a_add:+.4f}", "{a_ols:+.4f}", "{meaning}"'
            )
    lines.append("")
    lines.append("----")
    lines.append("")

    # ── Multiplicative amplitude ranking ──────────────────────────────────
    lines.append("Template amplitude ranking — multiplicative model (MCMC-comb, NSIDE 64)")
    lines.append("--------------------------------------------------------------------------")
    lines.append("")
    lines.append(
        f"All {n_sys} templates sorted by absolute MCMC-comb multiplicative amplitude "
        ":math:`|\\hat{b}_i|`."
    )
    lines.append("")
    lines.append(".. csv-table::")
    lines.append(
        '   :header: "Rank", "Template", "NSIDE", '
        '":math:`\\hat{b}_i` (MCMC-comb)", "Physical meaning"'
    )
    lines.append("   :widths: 5, 30, 6, 16, 50")
    lines.append("   :stub-columns: 1")
    lines.append("")
    if n_sys > 0:
        idx_b = np.argsort(-np.abs(b_hat_comb))
        for rank, i in enumerate(idx_b, 1):
            tname = template_names[i] if i < len(template_names) else f"T{i}"
            short, nside_t, meaning = parse_template_name(tname)
            b = float(b_hat_comb[i]) if i < len(b_hat_comb) else 0.0
            bold = rank <= 5
            short_fmt = f"**{short}**" if bold else short
            lines.append(
                f'   "{rank}", "{short_fmt}", "{nside_t}", "{b:+.4f}", "{meaning}"'
            )
    lines.append("")
    lines.append("----")
    lines.append("")

    # ── Per-galaxy weight statistics (NSIDE 64) ───────────────────────────
    lines.append("Per-galaxy weight statistics (NSIDE 64)")
    lines.append("----------------------------------------")
    lines.append("")
    weight_stats = compute_weight_stats(sid, 64)
    if weight_stats:
        lines.append(
            "From the ``*_NSIDE0064_WEIGHTS.fits`` file.  "
            "Mean ≈ 1 and small std indicate a well-behaved weight distribution."
        )
        lines.append("")
        lines.append(".. csv-table::")
        lines.append(
            '   :header: "Method", "N", "mean", "std", "p1", "p5", "p50", "p95", "p99"'
        )
        lines.append("   :widths: 14, 12, 8, 8, 8, 8, 8, 8, 8")
        lines.append("")
        for m in METHODS:
            ws = weight_stats.get(m)
            if ws is None:
                continue
            lines.append(
                f'   "{METHOD_LABELS[m]}", "{ws["n"]:,}", '
                f'"{ws["mean"]:.4f}", "{ws["std"]:.4f}", '
                f'"{ws["p1"]:.4f}", "{ws["p5"]:.4f}", '
                f'"{ws["p50"]:.4f}", "{ws["p95"]:.4f}", '
                f'"{ws["p99"]:.4f}"'
            )
    else:
        lines.append("Weight statistics not available (``*_WEIGHTS.fits`` not found).")
    lines.append("")
    lines.append("----")
    lines.append("")

    # ── Figures — weight maps 2×2 grid ────────────────────────────────────
    lines.append("Systematic weight maps")
    lines.append("----------------------")
    lines.append("")
    srcs_map = [f"_static/results_ls10/{sid}_NSIDE{ns:04d}_weight_map.png" for ns in NSIDES]
    caps_map = [f"NSIDE {ns} weight maps" for ns in NSIDES]
    lines.append(grid_fig_block(srcs_map, caps_map, "weight maps",
                                title="Mollweide weight maps — all six methods"))
    lines.append("")

    # ── Figures — weight distributions 2×2 grid ───────────────────────────
    lines.append("Systematic weight distributions")
    lines.append("-------------------------------")
    lines.append("")
    lines.append(
        "Narrow peaks near 1 indicate stable weight estimates.  "
        "ElasticNet weights may be exactly 1 when cross-validation selects zero amplitudes."
    )
    lines.append("")
    srcs_hist = [f"_static/results_ls10/{sid}_NSIDE{ns:04d}_weight_hist.png" for ns in NSIDES]
    caps_hist = [f"NSIDE {ns} weight distributions" for ns in NSIDES]
    lines.append(grid_fig_block(srcs_hist, caps_hist, "weight distributions",
                                title="Per-galaxy weight distributions — all six methods"))
    lines.append("")

    # ── Figures — angular clustering w(θ) 2×2 grid ────────────────────────
    lines.append("Angular clustering w(θ) before and after correction")
    lines.append("-----------------------------------------------------")
    lines.append("")
    lines.append(
        "Each panel shows the observed angular two-point correlation function "
        "(solid black) and the corrected :math:`w(\\theta)` for all six methods.  "
        "A well-corrected sample shows suppressed excess clustering at all scales.  "
        "Each panel corresponds to one map resolution."
    )
    lines.append("")
    srcs_wth = [f"_static/results_ls10/{sid}_NSIDE{ns:04d}_wtheta.png" for ns in NSIDES]
    caps_wth = [f"NSIDE {ns}" for ns in NSIDES]
    lines.append(grid_fig_block(srcs_wth, caps_wth, "wtheta",
                                title="w(θ): observed vs corrected — all six methods"))
    lines.append("")
    lines.append("----")
    lines.append("")

    # ── Cosmological verdict ──────────────────────────────────────────────
    lines.append(cosmological_verdict(cfg, params64))
    lines.append("")

    return "\n".join(lines)


def update_index_rst(tags):
    idx_path = DOCS_DIR / "index.rst"
    text = idx_path.read_text()
    marker = "   results_ls10\n"
    if marker not in text:
        print(f"WARNING: '{marker.strip()}' not found in index.rst; skipping update.")
        return
    insert_block = ""
    for tag in tags:
        entry = f"   results_ls10_{tag}\n"
        if entry not in text:
            insert_block += entry
    if insert_block:
        text = text.replace(marker, marker + insert_block)
        idx_path.write_text(text)
        print(f"  Updated index.rst with {len(tags)} new toctree entries.")
    else:
        print("  index.rst already contains all per-sample entries.")


def add_detail_links_to_results(tags_and_mstars):
    rst_path = DOCS_DIR / "results_ls10.rst"
    text = rst_path.read_text()
    for tag, mstar, next_anchor in tags_and_mstars:
        ref_label = f".. _ls10-sample-{tag}:"
        if ref_label not in text:
            continue
        seealso = (
            f"\n.. seealso::\n\n"
            f"   :doc:`results_ls10_{tag}` — full template tables, weight statistics, "
            f"and cosmological analysis verdict for log M* ≥ {mstar}.\n"
        )
        if next_anchor:
            next_ref = f".. _ls10-sample-{next_anchor}:"
            if next_ref in text and seealso not in text:
                text = text.replace(next_ref, seealso + "\n" + next_ref)
        else:
            marker = "\n----\n\nMAP parameters"
            if marker in text and seealso not in text:
                text = text.replace(marker, seealso + marker)
    rst_path.write_text(text)
    print("  Updated results_ls10.rst with seealso links.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--docs-dir", default=str(DOCS_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    docs_dir = Path(args.docs_dir)

    tags = []
    tags_and_mstars = []

    for i, cfg in enumerate(SAMPLE_CONFIGS):
        sid = cfg["sample_id"]
        print(f"\nProcessing {sid} ...")

        data = {}
        for ns in NSIDES:
            params, partials = load_sample_data(sid, ns)
            data[ns] = (params, partials)

        params64, _ = data.get(64, ({}, {}))
        if not params64:
            print(f"  WARNING: NSIDE 64 params.json not found; skipping.")
            continue

        rst = generate_page(cfg, data)
        out_path = docs_dir / f"results_ls10_{cfg['tag']}.rst"
        out_path.write_text(rst)
        n_lines = len(rst.splitlines())
        print(f"  Written → {out_path.name}  ({n_lines} lines)")

        tags.append(cfg["tag"])
        next_tag = SAMPLE_CONFIGS[i + 1]["tag"] if i + 1 < len(SAMPLE_CONFIGS) else None
        tags_and_mstars.append((cfg["tag"], cfg["mstar"], next_tag))

    print("\nUpdating docs/index.rst ...")
    update_index_rst(tags)
    print("Updating docs/results_ls10.rst ...")
    add_detail_links_to_results(tags_and_mstars)

    print(f"\nDone. Generated {len(tags)} pages:")
    for tag in tags:
        print(f"  docs/results_ls10_{tag}.rst")


if __name__ == "__main__":
    main()
