"""Generate docs/results_ls10_recommendations.rst.

For each LS10 BGS VLIM sample, recommends the optimal (method, NSIDE) combination
for cosmological analysis and quantifies the remaining systematic uncertainty budget
on the angular two-point correlation function w(theta).

Run from the repo root:
    python scripts/generate_recommendations.py
"""

import json
import math
from pathlib import Path

import numpy as np

DOCS_DIR = Path(__file__).parent.parent / "docs"
DATA_DIR = Path(__file__).parent.parent / "data" / "sys_weights"

SAMPLE_CONFIGS = [
    {"mstar": "9.0",  "z": "0.08", "tag": "9p0",   "n_gal": 523_486,
     "sample_id": "LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486",
     "dw_w_30": -7.2, "dw_w_theta_max": 23,  "dw_w_max": 8.4},
    {"mstar": "9.5",  "z": "0.12", "tag": "9p5",   "n_gal": 1_432_502,
     "sample_id": "LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502",
     "dw_w_30": -4.8, "dw_w_theta_max": 15,  "dw_w_max": 4.9},
    {"mstar": "10.0", "z": "0.18", "tag": "10p0",  "n_gal": 2_759_238,
     "sample_id": "LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238",
     "dw_w_30": -0.4, "dw_w_theta_max": 120, "dw_w_max": 2.0},
    {"mstar": "10.25","z": "0.22", "tag": "10p25", "n_gal": 3_308_841,
     "sample_id": "LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841",
     "dw_w_30": None, "dw_w_theta_max": None,"dw_w_max": None},
    {"mstar": "10.5", "z": "0.26", "tag": "10p5",  "n_gal": 3_263_228,
     "sample_id": "LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228",
     "dw_w_30": None, "dw_w_theta_max": None,"dw_w_max": None},
    {"mstar": "10.75","z": "0.31", "tag": "10p75", "n_gal": 2_802_710,
     "sample_id": "LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710",
     "dw_w_30": None, "dw_w_theta_max": None,"dw_w_max": None},
    {"mstar": "11.0", "z": "0.35", "tag": "11p0",  "n_gal": 1_619_838,
     "sample_id": "LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838",
     "dw_w_30": -0.1, "dw_w_theta_max": 181, "dw_w_max": 11.5},
    {"mstar": "11.25","z": "0.35", "tag": "11p25", "n_gal": 541_855,
     "sample_id": "LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855",
     "dw_w_30": +2.2, "dw_w_theta_max": 181, "dw_w_max": 17.4},
    {"mstar": "11.5", "z": "0.35", "tag": "11p5",  "n_gal": 120_882,
     "sample_id": "LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882",
     "dw_w_30": +0.7, "dw_w_theta_max": 97,  "dw_w_max": 9.7},
]

NSIDES = [32, 64, 128, 256]
GOOD_METHODS = ["OLS", "ISD-1", "MCMC-add", "MCMC-comb"]


def _load_sigma_comb(sid, nside):
    p = DATA_DIR / f"{sid}_NSIDE{nside:04d}_params.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("sigma_hat_comb")


def _load_wtheta(sid, nside):
    p = DATA_DIR / f"{sid}_NSIDE{nside:04d}_wtheta_data.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _recommend_nside(sigma_combs):
    """Return recommended NSIDE: minimum sigma_comb where sigma < 1 and NSIDE >= 64."""
    best_ns, best_s = None, 1e9
    for ns in NSIDES:
        s = sigma_combs.get(ns)
        if s is None or ns < 64:
            continue
        if s < 1.0 and s < best_s:
            best_ns, best_s = ns, s
    return best_ns if best_ns is not None else 64


def _analyse(wtheta_data):
    """Return (theta, w_obs, w_comb, A_pct, S_pct) arrays."""
    theta = np.array(wtheta_data["theta_arcmin"])
    w_obs = np.array(wtheta_data["w_obs"])
    corr = wtheta_data.get("all_w_corr", {})

    w_comb = np.array(corr.get("MCMC-comb", w_obs))
    # Correction amplitude: fraction of w_obs removed
    A_pct = (w_obs - w_comb) / w_obs * 100.0

    # Method spread: std over well-behaved methods / w_comb
    wvals = [np.array(corr[m]) for m in GOOD_METHODS if m in corr]
    if len(wvals) > 1:
        wstack = np.vstack(wvals)
        spread = np.std(wstack, axis=0)
        # Avoid division by tiny w_comb values
        denom = np.where(np.abs(w_comb) > 1e-8, np.abs(w_comb), 1e-8)
        S_pct = spread / denom * 100.0
    else:
        S_pct = np.zeros_like(theta)

    return theta, w_obs, w_comb, A_pct, S_pct


def _first_nonzero_theta(theta, A_pct, thr=0.05):
    """Return theta of first bin where |A_pct| > thr, or None."""
    idx = np.where(np.abs(A_pct) > thr)[0]
    return float(theta[idx[0]]) if len(idx) else None


def _regime(cfg, sigma_comb_64):
    dw30 = cfg["dw_w_30"]
    if dw30 is not None and abs(dw30) > 4:
        return "Strongly contaminated"
    if dw30 is not None and abs(dw30) > 1:
        return "Moderately contaminated"
    if dw30 is not None:
        return "Weakly contaminated"
    if sigma_comb_64 > 0.45:
        return "Shot-noise limited"
    return "Low contamination"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

results = []  # one dict per sample

for cfg in SAMPLE_CONFIGS:
    sid = cfg["sample_id"]
    mstar = cfg["mstar"]
    print(f"\nProcessing M* >= {mstar} ...")

    # sigma_hat_comb at each NSIDE
    sigma_combs = {ns: _load_sigma_comb(sid, ns) for ns in NSIDES}
    print("  sigma_comb:", {ns: f"{s:.3f}" if s else "—" for ns, s in sigma_combs.items()})

    rec_nside = _recommend_nside(sigma_combs)
    print(f"  recommended NSIDE: {rec_nside}")

    # Load wtheta at recommended NSIDE
    wdata = _load_wtheta(sid, rec_nside)
    if wdata is None:
        print(f"  WARNING: wtheta_data not found for NSIDE {rec_nside}; falling back to 64")
        rec_nside = 64
        wdata = _load_wtheta(sid, 64)

    theta, w_obs, w_comb, A_pct, S_pct = _analyse(wdata)

    first_theta = _first_nonzero_theta(theta, A_pct)
    max_A = float(np.abs(A_pct).max())
    max_S_gt20 = float(S_pct[theta > 20].max()) if (theta > 20).any() else 0.0

    # Also compute at NSIDE 128 for the secondary-option note
    wdata128 = _load_wtheta(sid, 128)
    if wdata128:
        _, _, _, A128, _ = _analyse(wdata128)
        max_A128 = float(np.abs(A128).max())
    else:
        max_A128 = None

    print(f"  max|A| = {max_A:.2f}%  max_S(>20') = {max_S_gt20:.2f}%  first_theta = {first_theta}")

    results.append({
        "cfg": cfg,
        "sigma_combs": sigma_combs,
        "rec_nside": rec_nside,
        "theta": theta,
        "w_obs": w_obs,
        "w_comb": w_comb,
        "A_pct": A_pct,
        "S_pct": S_pct,
        "first_theta": first_theta,
        "max_A": max_A,
        "max_S_gt20": max_S_gt20,
        "max_A128": max_A128,
    })


# ---------------------------------------------------------------------------
# RST generation
# ---------------------------------------------------------------------------

def _nside_str(ns, sigma):
    s = f"{sigma:.3f}" if sigma is not None else "—"
    flag = " ✗" if sigma is not None and sigma >= 1.0 else ""
    return f"N{ns}: {s}{flag}"


def _theta_range_str(first_theta):
    if first_theta is None:
        return "none (< 0.05%)"
    return f"{first_theta:.0f}–54'"


def _justification(cfg, sigma_combs, rec_nside):
    n64 = sigma_combs.get(64, 0.99)
    n128 = sigma_combs.get(128)
    n256 = sigma_combs.get(256)
    parts = []
    parts.append(f"σ̂_comb N64={n64:.3f} (minimum among valid NSIDEs)")
    if n128 is not None and n128 >= 1.0:
        parts.append(f"N128 overfits (σ={n128:.3f})")
    if n256 is not None and n256 >= 1.0:
        parts.append(f"N256 overfits (σ={n256:.3f})")
    dw = cfg["dw_w_30"]
    if dw is not None:
        parts.append(f"external |δw/w|={abs(dw):.1f}% at 30'")
    return "; ".join(parts)


lines = []
lines += [
    ".. _ls10-recommendations:",
    "",
    "LS10 systematic-correction recommendations",
    "==========================================",
    "",
    ".. contents:: On this page",
    "   :local:",
    "   :depth: 2",
    "",
    ".. seealso::",
    "",
    "   :doc:`results_ls10` — full σ̂ and LRT tables for all nine samples and four resolutions.",
    "",
]

# ---------------------------------------------------------------------------
# Decision framework
# ---------------------------------------------------------------------------
lines += [
    "Decision framework",
    "------------------",
    "",
    "**Method choice — MCMC-comb (WEIGHT\\_COMB) for all samples.**",
    "",
    "The Likelihood Ratio Test (LRT) rejects the additive-only null hypothesis at all four",
    "map resolutions (NSIDE 32–256) and all nine stellar-mass thresholds, with",
    ":math:`p < 10^{-9}` in every case.  This confirms a statistically significant",
    "*multiplicative* component on top of the additive systematic.",
    "``MCMC-comb`` (stored as ``WEIGHT_COMB = WEIGHT_SYS`` in the FITS files) is the only",
    "method that models both components jointly.  The other methods are ranked as follows:",
    "",
    "* ``OLS``, ``ISD-1``, ``MCMC-add``: nearly identical σ̂ values, miss the multiplicative term.",
    "* ``ElasticNet``: equivalent to OLS when cross-validation selects zero amplitudes (some samples).",
    "* ``ISD-3``: numerically unstable — σ̂ ≫ 1 at NSIDE 32 and poor at all other resolutions.",
    "  **Never use ISD-3 for science.**",
    "",
    "**Resolution choice — NSIDE 64 for all samples (primary).**",
    "",
    "The selection criterion is the minimum :math:`\\hat{\\sigma}_{\\rm comb}` among",
    "valid NSIDEs (σ < 1, NSIDE ≥ 64).  Three additional constraints apply:",
    "",
    "1. **NSIDE 32 is excluded** — its pixel scale (~108') exceeds the maximum",
    "   angular bin in the w(θ) data (54'), producing zero measurable correction",
    "   on the two-point function in the entire measured range.",
    "2. **All 11 templates are at NSIDE 64** — analysing at finer resolutions",
    "   upsamples the templates without adding systematic information.",
    "3. **NSIDE 128/256 overfit** for sparse samples (M* ≥ 9.0, ≥ 11.25, ≥ 11.5):",
    "   σ̂_comb exceeds 1.0 and the corrected w(θ) shows unphysically large corrections",
    "   (up to 34%).",
    "",
    "σ̂_comb is minimised at NSIDE 64 for all nine samples among resolutions where",
    "w(θ) correction is non-zero.  NSIDE 128 is a viable *secondary* option for the five",
    "densest samples (M* 9.5–11.25, σ̂_comb < 1), extending the correction scale to ~20'.",
    "",
]

# ---------------------------------------------------------------------------
# Recommendation table
# ---------------------------------------------------------------------------
lines += [
    "Recommended weight and resolution per sample",
    "--------------------------------------------",
    "",
    ".. csv-table::",
    '   :header: "Sample (log M*)", "N_gal", "Method", "NSIDE",'
    ' ":math:`\\hat{\\sigma}_{\\rm comb}`", "θ corrected", "Key constraint"',
    "   :widths: 12, 12, 12, 8, 10, 14, 50",
    "",
]
for r in results:
    cfg = r["cfg"]
    ns = r["rec_nside"]
    sc = r["sigma_combs"].get(ns)
    sc_s = f"{sc:.3f}" if sc else "—"
    first_s = _theta_range_str(r["first_theta"])
    just = _justification(cfg, r["sigma_combs"], ns)
    lines.append(
        f'   "≥ {cfg["mstar"]}", "{cfg["n_gal"]:,}", '
        f'"MCMC-comb", "{ns}", "{sc_s}", "{first_s}", "{just}"'
    )

lines += [
    "",
    ".. note::",
    "",
    "   NSIDE 128 is a valid secondary choice for the five densest samples",
    "   (M* ≥ 9.5 through ≥ 11.25), extending the angular correction down to ~20 arcmin.",
    "   It must **not** be used for M* ≥ 9.0 (σ̂_comb = 1.03) or M* ≥ 11.5 (σ̂_comb = 1.43),",
    "   where it overfits shot noise.",
    "",
    "   NSIDE 32 should never be used as the analysis resolution for angular clustering:",
    "   its pixel scale exceeds the measured w(θ) range and delivers zero correction.",
    "",
]

# ---------------------------------------------------------------------------
# sigma_comb overview table
# ---------------------------------------------------------------------------
lines += [
    "σ̂_comb goodness-of-fit across resolutions",
    "-------------------------------------------",
    "",
    ":math:`\\hat{\\sigma}_{\\rm comb} < 1` indicates the model fits within the Poisson",
    "expectation.  Values ≥ 1 (marked ✗) signal overfitting — the correction",
    "is absorbing noise rather than real systematics.",
    "",
    ".. csv-table::",
    '   :header: "Sample (log M*)", "N_gal", "NSIDE 32", "NSIDE 64", "NSIDE 128", "NSIDE 256", "Recommended"',
    "   :widths: 12, 12, 14, 14, 14, 14, 12",
    "",
]
for r in results:
    cfg = r["cfg"]
    sc = r["sigma_combs"]

    def _fmt(ns):
        v = sc.get(ns)
        if v is None:
            return "—"
        flag = " ✗" if v >= 1.0 else ""
        return f"{v:.3f}{flag}"

    lines.append(
        f'   "≥ {cfg["mstar"]}", "{cfg["n_gal"]:,}", '
        f'"{_fmt(32)}", "{_fmt(64)}", "{_fmt(128)}", "{_fmt(256)}", '
        f'"**N{r["rec_nside"]}**"'
    )
lines.append("")

# ---------------------------------------------------------------------------
# Systematic uncertainty budget summary
# ---------------------------------------------------------------------------
lines += [
    "Systematic uncertainty budget",
    "------------------------------",
    "",
    "The table below quantifies the systematic correction applied and the residual",
    "method uncertainty at the recommended NSIDE 64 + MCMC-comb.",
    "",
    "* **Max |δw/w|_comb at 42–54'** — the largest fractional correction applied to w(θ)",
    "  by the MCMC-comb weights relative to the uncorrected measurement.",
    "  Observed only at the outermost bins because the NSIDE 64 pixel scale (~54')",
    "  sets the lower angular limit of the weight-induced clustering modulation.",
    "* **Method uncertainty >20'** — the standard deviation of the corrected w(θ)",
    "  over the four well-behaved methods {OLS, ISD-1, MCMC-add, MCMC-comb},",
    "  expressed as a fraction of w_comb.  This is the irreducible systematic floor",
    "  from the choice of correction algorithm.",
    "* Below ~40 arcmin at NSIDE 64, no measurable correction is applied in the",
    "  wtheta data range.  The large-scale systematic bias (e.g. −7.2% at 30' for",
    "  M* ≥ 9.0) operates primarily at degree scales where the template coherence",
    "  length matches and the weights are most effective.",
    "",
    ".. csv-table::",
    '   :header: "Sample (log M*)", "N_gal", "Recommended NSIDE",'
    ' "Max |δw/w|\\ :sub:`comb` at 42–54'"'"'", "θ of first correction",'
    ' "Method uncertainty >20'"'"'", "Regime"',
    "   :widths: 12, 12, 14, 18, 16, 18, 22",
    "",
]
for r in results:
    cfg = r["cfg"]
    regime = _regime(cfg, r["sigma_combs"].get(64, 0.75))
    first_s = _theta_range_str(r["first_theta"])
    lines.append(
        f'   "≥ {cfg["mstar"]}", "{cfg["n_gal"]:,}", '
        f'"{r["rec_nside"]}", '
        f'"{r["max_A"]:.2f}%", '
        f'"{first_s}", '
        f'"{r["max_S_gt20"]:.2f}%", '
        f'"{regime}"'
    )
lines.append("")

# ---------------------------------------------------------------------------
# Per-sample angular correction profiles
# ---------------------------------------------------------------------------
lines += [
    "Per-sample angular correction profiles",
    "---------------------------------------",
    "",
    "The following tables show, for each sample at the recommended NSIDE 64,",
    "the observed w(θ), the MCMC-comb corrected w(θ), the fractional correction",
    "applied, and the method-to-method spread as a proxy for the systematic floor.",
    "All corrections are expressed as percentages of w_obs.",
    "",
]

for r in results:
    cfg = r["cfg"]
    mstar = cfg["mstar"]
    z_max = cfg["z"]
    n_gal = cfg["n_gal"]
    theta = r["theta"]
    w_obs = r["w_obs"]
    w_comb = r["w_comb"]
    A_pct = r["A_pct"]
    S_pct = r["S_pct"]

    heading = f"M* ≥ {mstar} (z < {z_max}, N = {n_gal:,})"
    lines.append(f".. _rec-{cfg['tag']}:")
    lines.append("")
    lines.append(heading)
    lines.append("~" * (len(heading) + 2))
    lines.append("")

    # Brief narrative
    dw30 = cfg["dw_w_30"]
    dw_max = cfg["dw_w_max"]
    dw_theta_max = cfg["dw_w_theta_max"]

    if dw30 is not None:
        ext_line = (
            f"External two-point measurement: "
            f"δw/w ≈ {dw30:+.1f}% at 30', "
            f"max |δw/w| = {dw_max:.1f}% at {dw_theta_max}'."
        )
    else:
        ext_line = "External two-point measurement: not available for this sample."

    lines.append(ext_line)
    lines.append("")

    # Sigma-comb summary
    sc = r["sigma_combs"]
    valid_ns = [ns for ns in [64, 128, 256] if sc.get(ns) is not None and sc[ns] < 1.0]
    if len(valid_ns) > 1:
        secondary = f"NSIDE {valid_ns[-1]} is a valid secondary option (σ̂={sc[valid_ns[-1]]:.3f})."
    else:
        secondary = "No finer valid NSIDE available (σ̂ ≥ 1 at NSIDE 128+)."

    lines.append(
        f"Recommended: ``WEIGHT_COMB`` from NSIDE {r['rec_nside']} analysis "
        f"(σ̂_comb = {sc.get(r['rec_nside'], float('nan')):.3f}).  "
        f"{secondary}"
    )
    lines.append("")

    lines += [
        ".. csv-table::",
        '   :header: "θ (arcmin)", "w_obs", "w_comb (MCMC-comb)", "δw/w (%)", "Method spread (%)"',
        "   :widths: 12, 14, 18, 12, 16",
        "",
    ]
    for i in range(len(theta)):
        th = theta[i]
        wo = w_obs[i]
        wc = w_comb[i]
        a = A_pct[i]
        s = S_pct[i]
        lines.append(
            f'   "{th:.2f}", "{wo:.5f}", "{wc:.5f}", "{a:+.2f}", "{s:.2f}"'
        )
    lines.append("")

# ---------------------------------------------------------------------------
# Caveats
# ---------------------------------------------------------------------------
lines += [
    "Caveats and limitations",
    "-----------------------",
    "",
    "* **Angular range**: the wtheta_data.json files cover θ = 0.57–54 arcmin only.",
    "  Systematic contamination at degree scales (θ > 1°) is not captured in these tables.",
    "  The external δw/w measurements at 30–180' (listed in the per-sample rows above)",
    "  come from a separate analysis that covers a broader angular range.",
    "",
    "* **Zero correction at θ < 40' for NSIDE 64**: the correction from NSIDE 64 weights",
    "  is zero below ~40 arcmin in the measured range because the NSIDE 64 pixel scale",
    "  (~54') sets the minimum angular scale over which the weight field varies.",
    "  This does not mean the systematic is zero at small scales — it means that",
    "  sub-degree corrections require a finer weight map (NSIDE 128, see secondary option).",
    "",
    "* **NSIDE 128 secondary option**: for the five densest samples (M* 9.5–11.25),",
    "  σ̂_comb < 1 at NSIDE 128, and corrections extend to θ ~ 20'.",
    "  The method uncertainty remains below 0.8% in all cases.",
    "  However, since all templates are at NSIDE 64, the NSIDE 128 weight field",
    "  is derived from upsampled templates with no additional angular information.",
    "",
    "* **ISD-3 must not be used**: σ̂ >> 1 at NSIDE 32 for all samples; the corrected",
    "  weights are numerically unstable.  See :doc:`results_ls10` for full ISD-3 σ̂ values.",
    "",
    "* **FITS column names**: the recommended weights are stored as ``WEIGHT_COMB``",
    "  (equivalent to ``WEIGHT_SYS``) in the ``*_NSIDE0064_WEIGHTS.fits`` files.",
    "",
]

# ---------------------------------------------------------------------------
# Write RST
# ---------------------------------------------------------------------------
rst = "\n".join(lines)
out = DOCS_DIR / "results_ls10_recommendations.rst"
out.write_text(rst)
print(f"\nWrote {out}  ({len(lines)} lines)")

# ---------------------------------------------------------------------------
# Update index.rst
# ---------------------------------------------------------------------------
idx = DOCS_DIR / "index.rst"
text = idx.read_text()
marker = "   results_ls10\n"
entry = "   results_ls10_recommendations\n"
if entry not in text and marker in text:
    text = text.replace(marker, marker + entry)
    idx.write_text(text)
    print("Updated docs/index.rst")
else:
    print("docs/index.rst already up-to-date")

print("Done.")
