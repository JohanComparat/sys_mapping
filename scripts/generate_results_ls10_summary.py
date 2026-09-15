#!/usr/bin/env python3
"""Generate the LS10 results pages from the pipeline products on disk.

Writes ``docs/results_ls10.rst``, ``docs/results_ls10_recommendations.rst`` and one
page per sample, ``docs/results_ls10_<anchor>.rst``.  Every number is read from:

``data/sys_weights_auto/``
    the products as issued, each sample at the finest NSIDE whose mean occupancy
    reaches 25 galaxies per pixel (``run_ls10_analysis.py --min-per-pixel``);
    ``*_WEIGHTS.fits`` are read for the weight statistics when present.
``data/sys_weights/``
    the same fits at NSIDE 32, 64 and 128, for the resolution comparison.
``results/ls10_mocklrt/NSIDE00xx/``
    the mock-calibrated additive-versus-combined likelihood ratio.

The issued products' figures are copied to ``docs/_static/results_ls10/issued/``.

Run from the repository root::

    python scripts/generate_results_ls10_summary.py
"""
from __future__ import annotations

import glob
import json
import re
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ISSUED = REPO / "data" / "sys_weights_auto"
GRID = REPO / "data" / "sys_weights"
LRT = REPO / "results" / "ls10_mocklrt"
DOCS = REPO / "docs"
STATIC = DOCS / "_static" / "results_ls10"
FIG = STATIC / "issued"

GRID_NSIDES = (32, 64, 128)
LRT_NSIDES = (32, 64)
METHODS = ["OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"]
COLUMN = {"OLS": "WEIGHT_OLS", "ElasticNet": "WEIGHT_ENET", "ISD-1": "WEIGHT_ISD1",
          "ISD-3": "WEIGHT_ISD3", "MCMC-add": "WEIGHT_ADD", "MCMC-comb": "WEIGHT_COMB"}
ALPHA = 0.05          # detection and rejection level for the recommendation
CLIP = (1 / 20, 20)
THETA_REF = 30.0      # arcmin


# ── loading ────────────────────────────────────────────────────────────────────

def _mstar(sample_id: str) -> str:
    return sample_id.split("_")[3]


def _anchor(mstar: str) -> str:
    return mstar.replace(".", "p")


def _zmax(sample_id: str) -> str:
    return re.search(r"_z_([0-9.]+)_N_", sample_id).group(1)


def _short(name: str) -> str:
    return name.split("_NSIDE")[0]


def load_issued() -> list[dict]:
    rows = []
    for f in glob.glob(str(ISSUED / "LS10_VLIM_ANY_*_params.json")):
        d = json.load(open(f))
        w = json.load(open(f.replace("_params.json", "_wtheta_data.json")))
        d["_wtheta"] = w
        d["_stem"] = Path(f).name.replace("_params.json", "")
        d["_weights"] = weight_stats(Path(f.replace("_params.json", "_WEIGHTS.fits")))
        rows.append(d)
    return sorted(rows, key=lambda d: float(_mstar(d["sample_id"])))


def weight_stats(path: Path) -> dict | None:
    if not path.exists():
        return None
    from astropy.io import fits
    out = {}
    with fits.open(path, memmap=True) as h:
        hdr, data = h[1].header, h[1].data
        out["WEIGHTVER"] = hdr.get("WEIGHTVER")
        out["TPLBASIS"] = hdr.get("TPLBASIS")
        for m, col in list(COLUMN.items()) + [("SYS", "WEIGHT_SYS")]:
            if col not in data.columns.names:
                continue
            w = np.asarray(data[col], float)
            out[m] = dict(min=float(w.min()), max=float(w.max()),
                          p1=float(np.percentile(w, 1)), p99=float(np.percentile(w, 99)),
                          clip=float(np.mean((w <= CLIP[0] * 1.002) | (w >= CLIP[1] * 0.999))),
                          unity=bool(np.all(w == 1.0)))
    return out


def load_grid(sample_id: str) -> dict[int, dict]:
    out = {}
    for ns in GRID_NSIDES:
        p = GRID / f"{sample_id}_NSIDE{ns:04d}_params.json"
        if p.exists():
            out[ns] = json.load(open(p))
    return out


def load_lrt(sample_id: str) -> dict[int, dict]:
    out = {}
    for ns in LRT_NSIDES:
        p = LRT / f"NSIDE{ns:04d}" / f"{sample_id}_NSIDE{ns:04d}_params.json"
        if p.exists():
            lrt = json.load(open(p)).get("lrt") or {}
            if lrt.get("calibration") == "mock":
                out[ns] = lrt
    return out


# ── derived quantities ─────────────────────────────────────────────────────────

def leading(d: dict) -> tuple[int, str]:
    s = d["significance"]["significance"]
    i = int(np.argmax(s))
    return i, _short(d["template_names"][i])


def fwp_str(sig: dict) -> str:
    if sig["family_wise_p"] <= sig["p_value_floor"]:
        return f"≤ {sig['p_value_floor']:.4f}"
    return f"{sig['family_wise_p']:.4f}"


def wtheta_depth(d: dict) -> dict[str, tuple[float, float]]:
    """Per method: the smallest corrected/observed ratio and the separation it occurs at."""
    w = d["_wtheta"]
    th, wo = np.asarray(w["theta_arcmin"]), np.asarray(w["w_obs"])
    pos = wo > 0
    out = {}
    for m in METHODS:
        r = np.asarray(w["all_w_corr"][m])[pos] / wo[pos]
        k = int(np.argmin(r))
        out[m] = (float(r[k]), float(th[pos][k]))
    return out


def ratio_at(d: dict, method: str, theta: float) -> float:
    w = d["_wtheta"]
    th, wo = np.asarray(w["theta_arcmin"]), np.asarray(w["w_obs"])
    k = int(np.argmin(np.abs(np.log(th / theta))))
    return float(w["all_w_corr"][method][k] / wo[k])


def lrt_nside(d: dict) -> int:
    """The likelihood-ratio grid nearest the resolution the sample is issued at."""
    return 64 if d["nside"] >= 64 else 32


def recommendation(d: dict, lrt: dict[int, dict]) -> tuple[str, str]:
    """(column, reason): contamination found by either calibrated test, and whether the
    likelihood ratio nearest the issued resolution requires the multiplicative term."""
    sig = d["significance"]
    ns = lrt_nside(d)
    l = lrt.get(ns)
    detected = sig["family_wise_p"] <= ALPHA
    rejects = l is not None and l["p_value"] <= ALPHA
    if not detected and not rejects:
        return ("WEIGHT_ISD3",
                f"neither test finds contamination (family-wise p = {sig['family_wise_p']:.3f}"
                + (f", NSIDE {ns} likelihood-ratio p = {l['p_value']:.3f}" if l else "")
                + "), so a correction is expected to add more variance than it removes")
    evidence = []
    if detected:
        evidence.append(f"a template is detected (family-wise p {fwp_str(sig)})")
    if rejects:
        evidence.append(f"the NSIDE {ns} likelihood ratio requires the multiplicative term "
                        f"(p = {l['p_value']:.3f})")
    reason = " and ".join(evidence)
    if not rejects and l is not None:
        reason += (f"; the NSIDE {ns} likelihood ratio does not require the multiplicative "
                   f"term (p = {l['p_value']:.3f}), so WEIGHT_ADD is the simpler alternative")
    return "WEIGHT_SYS", reason


# ── RST helpers ────────────────────────────────────────────────────────────────

def title(text: str, ch: str) -> list[str]:
    return [text, ch * len(text), ""]


def csv_table(header: list[str], rows: list[list[str]], caption: str = "",
              widths: list[int] | None = None) -> list[str]:
    out = [f".. csv-table::{(' ' + caption) if caption else ''}",
           "   :header: " + ", ".join(f'"{h}"' for h in header)]
    if widths:
        out.append("   :widths: " + ", ".join(map(str, widths)))
    out.append("")
    for r in rows:
        out.append("   " + ", ".join(f'"{c}"' for c in r))
    out.append("")
    return out


def figure(src: str, width: str, caption: str) -> list[str]:
    return [f".. figure:: /{src}", f"   :width: {width}", "", f"   {caption}", ""]


# ── pages ──────────────────────────────────────────────────────────────────────

def summary_page(issued: list[dict], lrts: dict[str, dict], grids: dict[str, dict]) -> str:
    L: list[str] = []
    L += title("Results: systematic weights", "=")
    L += ["Per-galaxy systematic weights for the nine LS10 BGS volume-limited stellar-mass",
          "threshold samples, computed by ``scripts/run_ls10_analysis.py`` with 11 templates",
          "(Gaia DR3 star counts and fluxes; LS10 extinction, depth, exposure count and PSF",
          "size). Each sample is issued at the finest NSIDE whose mean occupancy reaches 25",
          "galaxies per pixel. :doc:`results_ls10_recommendations` gives the column to use per",
          "sample, and :doc:`pipeline_ls10` how to reproduce the products.", "",
          ".. contents:: On this page", "   :local:", "   :depth: 1", ""]

    L += title("Issued products", "-")
    have_w = all(d["_weights"] for d in issued)
    header = ["log M* ≥", "z <", "N gal", "NSIDE", "N pix", "galaxies / pixel"]
    if have_w:
        header += ["WEIGHT_SYS 1–99 %", "clipped"]
    rows = []
    for d in issued:
        r = [_mstar(d["sample_id"]), _zmax(d["sample_id"]), f"{d['n_galaxies']:,}",
             str(d["nside"]), f"{d['n_good_pix']:,}", f"{d['n_galaxies'] / d['n_good_pix']:.1f}"]
        if have_w:
            s = d["_weights"]["SYS"]
            r += [f"{s['p1']:.3f}–{s['p99']:.3f}", f"{100 * s['clip']:.3f} %"]
        rows.append(r)
    L += csv_table(header, rows)
    if have_w:
        vers = sorted({str(d["_weights"]["WEIGHTVER"]) for d in issued})
        basis = sorted({str(d["_weights"]["TPLBASIS"]) for d in issued})
        L += [f"Every file records ``WEIGHTVER`` = {', '.join(vers)} and ``TPLBASIS`` = "
              f"``{', '.join(basis)}``. *Clipped* is the fraction of galaxies at the weight clip "
              f"[1/20, 20].", ""]
    L += ["The two-point correction uses the full cross-template matrix",
          r":math:`\xi_{ij}(\theta)`, measured from the template values the galaxies carry.", ""]

    L += title("Detection of systematics", "-")
    L += ["Each template amplitude is scored against its scatter over uncontaminated GLASS",
          "realisations drawn with the sample's matched spectrum",
          "(``sys_mapping.calibrated_template_significance``). The family-wise p compares the",
          "largest significance over the templates with the largest of each realisation, and",
          "is bounded below by 1/(N+1). κ is the calibrated error over the independent-pixel",
          "error.", ""]
    rows = []
    for d in issued:
        sig = d["significance"]
        i, name = leading(d)
        infl = np.asarray(sig["inflation"])
        rows.append([_mstar(d["sample_id"]), str(d["nside"]), name,
                     f"{sig['significance'][i]:.2f}", f"{sig['significance'][i] * infl[i]:.2f}",
                     fwp_str(sig), f"{np.median(infl):.1f}", f"{infl.min():.1f}–{infl.max():.1f}",
                     str(sig["n_null"])])
    L += csv_table(["log M* ≥", "NSIDE", "leading template", "S cal", "S iid", "family-wise p",
                    "κ median", "κ range", "N"], rows)

    L += title("Additive or combined model: likelihood ratio", "-")
    L += ["Both models are refined to their likelihood maxima, and the statistic is ranked",
          "against 50 uncontaminated realisations of the sample's matched spectrum fitted the",
          "same way. The Wilks p assumes independent pixels and is shown for contrast.", ""]
    rows = []
    for ns in LRT_NSIDES:
        for d in issued:
            l = lrts[d["sample_id"]].get(ns)
            if l is None:
                continue
            nl = np.asarray(l.get("null_lambda") or [np.nan])
            rows.append([_mstar(d["sample_id"]), str(ns), f"{l['lambda_lr']:.1f}",
                         f"{l['p_chi2']:.1e}", f"{l['p_value']:.3f}",
                         f"{np.nanmin(nl):.1f} / {l['null_lambda_mean']:.1f} / {l['null_lambda_max']:.1f}",
                         "**yes**" if l["p_value"] <= ALPHA else "no"])
    L += csv_table(["log M* ≥", "NSIDE", "λ LR", "Wilks p", "mock p", "null min / mean / max",
                    "rejects additive"], rows)
    L += ["The likelihood-ratio grids use templates standardised over each map's own valid",
          "region rather than over the footprint; on the footprint basis the NSIDE 64 statistic",
          "changes by a median of 0.7 %.", ""]

    L += title("Corrected angular correlation function", "-")
    L += figure("_static/results_ls10/wtheta_ratio_occupancy.png", "95%",
                "Corrected over observed w(θ), each sample at its issued resolution.")
    rows = []
    for d in issued:
        depth = wtheta_depth(d)
        rows.append([_mstar(d["sample_id"]), str(d["nside"]),
                     f"{ratio_at(d, 'MCMC-comb', THETA_REF):.3f}"]
                    + [f"{depth[m][0]:.3f} ({depth[m][1]:.0f}′)" for m in METHODS])
    L += csv_table(["log M* ≥", "NSIDE", "MCMC-comb at 30′"]
                   + [f"{m} min" for m in METHODS], rows)
    L += ["*min* is the smallest corrected/observed ratio over the 30 bins and the separation",
          "it occurs at.", ""]

    L += title("Resolution comparison", "-")
    L += ["The same fits at NSIDE 32, 64 and 128. Coarser pixels hold more galaxies, so",
          r":math:`\hat\sigma` tracks shot noise and does not select a resolution.", ""]
    for key, label in (("MCMC-add", "additive"), ("MCMC-comb", "combined")):
        rows = []
        for d in issued:
            g = grids[d["sample_id"]]
            rows.append([_mstar(d["sample_id"])]
                        + [f"{g[ns]['methods'][key]['sigma_hat']:.4f}" if ns in g else "—"
                           for ns in GRID_NSIDES])
        L += csv_table(["log M* ≥"] + [f"NSIDE {ns}" for ns in GRID_NSIDES], rows,
                       caption=rf"Residual scatter :math:`\hat\sigma` of the {label} model.")

    L += title("Per-sample pages", "-")
    L += [f"* :doc:`results_ls10_{_anchor(_mstar(d['sample_id']))}`" for d in issued] + [""]
    return "\n".join(L)


def recommendations_page(issued: list[dict], lrts: dict[str, dict]) -> str:
    L: list[str] = [".. _ls10-recommendations:", ""]
    L += title("LS10 systematic-correction recommendations", "=")
    L += ["The weight column to use per sample, from two tests on the issued products: whether",
          "any template is detected by the calibrated significance, and whether the",
          "mock-calibrated likelihood ratio on the grid nearest the issued resolution (NSIDE 32",
          f"below NSIDE 64, NSIDE 64 otherwise) requires the multiplicative term. Both at",
          f"α = {ALPHA}.", ""]
    rows = []
    for d in issued:
        col, reason = recommendation(d, lrts[d["sample_id"]])
        ns = lrt_nside(d)
        l = lrts[d["sample_id"]].get(ns)
        rows.append([_mstar(d["sample_id"]), str(d["nside"]), fwp_str(d["significance"]),
                     f"{l['p_value']:.3f} (NSIDE {ns})" if l else "—", f"``{col}``", reason])
    L += csv_table(["log M* ≥", "NSIDE", "family-wise p", "likelihood-ratio p", "column",
                    "reason"], rows, widths=[8, 7, 11, 13, 13, 48])
    L += ["``WEIGHT_SYS`` is ``WEIGHT_COMB``, the combined additive and multiplicative model.",
          "``WEIGHT_ISD3`` is recommended only where neither test finds contamination, following",
          "the break-even condition measured on :doc:`results_algorithm_characterisation`.", ""]
    return "\n".join(L)


def sample_page(d: dict, lrt: dict[int, dict], grid: dict[int, dict]) -> str:
    ms, sid = _mstar(d["sample_id"]), d["sample_id"]
    anchor, ns = _anchor(ms), d["nside"]
    sig = d["significance"]
    names = [_short(n) for n in d["template_names"]]
    i, lead = leading(d)
    col, reason = recommendation(d, lrt)
    L: list[str] = [f".. _sample-{anchor}:", ""]
    L += title(f"BGS VLIM log M* ≥ {ms}, z < {_zmax(sid)}", "=")
    L += [f"{d['n_galaxies']:,} galaxies, issued at NSIDE {ns} "
          f"({d['n_galaxies'] / d['n_good_pix']:.1f} galaxies per pixel over "
          f"{d['n_good_pix']:,} pixels). Leading template: ``{lead}`` at "
          f"{sig['significance'][i]:.2f} calibrated, family-wise p {fwp_str(sig)}. "
          f"Recommended column: ``{col}``; {reason}.", "",
          ".. contents:: On this page", "   :local:", "   :depth: 1", "",
          ".. seealso::", "", "   :doc:`results_ls10` — all nine samples.", ""]

    L += title("Template significance", "-")
    infl = np.asarray(sig["inflation"])
    rows = [[names[k], f"{sig['significance'][k]:.2f}",
             f"{sig['significance'][k] * infl[k]:.2f}", f"{sig['p_values'][k]:.4f}",
             f"{infl[k]:.2f}"] for k in np.argsort(sig["significance"])[::-1]]
    L += csv_table(["template", "S cal", "S iid", "p", "κ"], rows)
    L += [f"Calibrated on {sig['n_null']} realisations; p is per template, with floor "
          f"{sig['p_value_floor']:.4f}.", ""]

    L += title(f"Fitted amplitudes (NSIDE {ns})", "-")
    rows = []
    for k, n in enumerate(names):
        rows.append([n] + [f"{d['methods'][m]['a_hat'][k]:+.4f}" for m in METHODS]
                    + [f"{d['methods']['MCMC-comb']['b_hat'][k]:+.4f}"])
    L += csv_table(["template"] + [f"a {m}" for m in METHODS] + ["b MCMC-comb"], rows)
    L += ["Amplitudes are per unit template standard deviation on the footprint.", ""]

    if d["_weights"]:
        L += title("Weights", "-")
        rows = []
        for m in METHODS:
            s = d["_weights"].get(m)
            if s:
                rows.append([COLUMN[m], f"{s['min']:.3f}", f"{s['max']:.3f}", f"{s['p1']:.3f}",
                             f"{s['p99']:.3f}", f"{100 * s['clip']:.3f} %",
                             "yes" if s["unity"] else "no"])
        L += csv_table(["column", "min", "max", "1 %", "99 %", "clipped", "identically 1"], rows)
    L += figure(f"_static/results_ls10/issued/{d['_stem']}_weight_map.png", "95%",
                f"Weight maps at NSIDE {ns}, one panel per method.")
    L += figure(f"_static/results_ls10/issued/{d['_stem']}_weight_hist.png", "70%",
                "Weight distributions.")

    L += title("Angular correlation function", "-")
    L += figure(f"_static/results_ls10/issued/{d['_stem']}_wtheta.png", "80%",
                "Observed and corrected w(θ), full cross-template correction.")
    depth = wtheta_depth(d)
    L += csv_table(["method", "at 30′", "smallest ratio", "at θ"],
                   [[m, f"{ratio_at(d, m, THETA_REF):.3f}", f"{depth[m][0]:.3f}",
                     f"{depth[m][1]:.0f}′"] for m in METHODS],
                   caption="Corrected over observed w(θ).")

    if lrt:
        L += title("Likelihood ratio", "-")
        rows = [[str(n), f"{l['lambda_lr']:.1f}", f"{l['p_value']:.3f}",
                 f"{l['null_lambda_mean']:.1f} / {l['null_lambda_max']:.1f}", str(l["n_null"])]
                for n, l in sorted(lrt.items())]
        L += csv_table(["NSIDE", "λ LR", "mock p", "null mean / max", "N"], rows)

    if grid:
        L += title("Resolution comparison", "-")
        rows = [[m] + [f"{grid[n]['methods'][m]['sigma_hat']:.4f}" if n in grid else "—"
                       for n in GRID_NSIDES] for m in METHODS]
        L += csv_table(["method"] + [f"NSIDE {n}" for n in GRID_NSIDES], rows,
                       caption=r"Residual scatter :math:`\hat\sigma`.")
    return "\n".join(L)


def main() -> None:
    issued = load_issued()
    if len(issued) != 9:
        raise SystemExit(f"expected 9 issued products under {ISSUED}, found {len(issued)}")
    lrts = {d["sample_id"]: load_lrt(d["sample_id"]) for d in issued}
    grids = {d["sample_id"]: load_grid(d["sample_id"]) for d in issued}

    FIG.mkdir(parents=True, exist_ok=True)
    for d in issued:
        for kind in ("weight_map", "weight_hist", "wtheta"):
            src = ISSUED / f"{d['_stem']}_{kind}.png"
            if src.exists():
                shutil.copy2(src, FIG / src.name)

    (DOCS / "results_ls10.rst").write_text(summary_page(issued, lrts, grids))
    (DOCS / "results_ls10_recommendations.rst").write_text(recommendations_page(issued, lrts))
    for d in issued:
        anchor = _anchor(_mstar(d["sample_id"]))
        (DOCS / f"results_ls10_{anchor}.rst").write_text(
            sample_page(d, lrts[d["sample_id"]], grids[d["sample_id"]]))
    print(f"wrote results_ls10.rst, results_ls10_recommendations.rst and {len(issued)} sample pages")


if __name__ == "__main__":
    main()
