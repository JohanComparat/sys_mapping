#!/usr/bin/env python
"""Systematic-detectability law and the LS10 worked example.

The smallest systematic in galaxy density that a survey of given sky coverage, pixel
size and galaxy density can detect. The law is analytic; its normalisation is read
from the LS10 fit outputs on disk, and no simulation is run:

    field detection SNR    D = A * sqrt(N_pix) / sigma_hat = A * sqrt(N_eff)
    smallest detectable    A_min(nu) = nu * sigma_hat / sqrt(N_pix)
    effective galaxy count N_eff = N_pix / sigma_hat**2   (<= N_gal)

where ``A`` is the systematic field rms, ``N_pix`` the number of fit pixels and
``sigma_hat`` the residual per-pixel scatter of the fit (shot noise plus clustering).
The shot-noise floor is ``A_min = nu / sqrt(N_gal)``.

Writes ``docs/detectability_law.rst`` and figures/CSVs in
``docs/_static/detectability_law/``.

Run::

    python scripts/analyze_detectability_law.py
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import ast

import matplotlib.pyplot as plt
import numpy as np

try:
    import healpy as hp

    _HAS_HP = True
except Exception:  # pragma: no cover
    _HAS_HP = False

REPO = Path(__file__).resolve().parent.parent
STATIC = REPO / "docs" / "_static" / "detectability_law"
RST = REPO / "docs" / "detectability_law.rst"
SYSW = REPO / "data" / "sys_weights"
# One product per sample at the resolution its occupancy supports, carrying the
# significance calibrated against matched GLASS realisations.
SYSW_CAL = REPO / "data" / "sys_weights_auto"
PROG = REPO / "docs" / "_static" / "results_progressive_contamination" / "progressive_results.csv"

FIDUCIAL = "10.0"  # log M* threshold of the fiducial LS10 sample
NU_LEVELS = (3.0, 5.0)


# ---------------------------------------------------------------------------
# Analytic core (survey-agnostic)
# ---------------------------------------------------------------------------

def n_eff(n_pix: float, sigma_hat: float) -> float:
    """Effective galaxy count: N_pix / sigma_hat**2 (<= N_gal)."""
    return n_pix / sigma_hat**2


def a_min(nu: float, n_pix: float, sigma_hat: float) -> float:
    """Smallest detectable field RMS at ``nu`` sigma: nu * sigma_hat / sqrt(N_pix)."""
    return nu * sigma_hat / math.sqrt(n_pix)


def a_min_shot(nu: float, n_gal: float) -> float:
    """Shot-noise-floor detectable field RMS: nu / sqrt(N_gal)."""
    return nu / math.sqrt(n_gal)


def field_snr(amp: float, n_pix: float, sigma_hat: float) -> float:
    """Field detection SNR of a systematic of field RMS ``amp``."""
    return amp * math.sqrt(n_pix) / sigma_hat


def regime(nbar: float, sigma_hat: float) -> str:
    """Which noise term dominates sigma_hat**2 = 1/nbar + sigma_clus**2."""
    shot = 1.0 / nbar
    clus = max(sigma_hat**2 - shot, 0.0)
    return "shot-limited" if shot > clus else "clustering-limited"


# ---------------------------------------------------------------------------
# Data loaders (reuse existing LS10 fit outputs)
# ---------------------------------------------------------------------------

def _load_params(pattern: str):
    fs = sorted(glob.glob(str(SYSW / pattern)))
    return [json.load(open(f)) for f in fs]


def load_ls10_samples(nsstr: str = "0064"):
    """All LS10 stellar-mass samples at one NSIDE."""
    out = []
    for d in _load_params(f"LS10_VLIM_ANY_*NSIDE{nsstr}_params.json"):
        mstar = d["sample_id"].split("_")[3]
        npix = d["n_good_pix"]
        ngal = d["n_galaxies"]
        shat = d.get("sigma_hat_ols") or d.get("sigma_hat_add")
        ah = np.asarray(d.get("a_hat_add", []), float)
        va = np.asarray(d.get("var_a_add", []), float)
        names = [n.split("_NSIDE")[0] for n in d.get("template_names", [])]
        out.append(dict(mstar=mstar, ngal=ngal, npix=npix, nbar=ngal / npix,
                        sigma_hat=shat, a_hat=ah, var_a=va, names=names,
                        nside=d["nside"]))
    out.sort(key=lambda r: float(r["mstar"]))
    return out


def load_ls10_across_nside(mstar: str = FIDUCIAL):
    rows = []
    for nsstr in ("0032", "0064", "0128"):
        for r in load_ls10_samples(nsstr):
            if r["mstar"] == mstar:
                rows.append(r)
    rows.sort(key=lambda r: r["nside"])
    return rows


def load_calibrated():
    """Calibrated significances, one entry per sample at its chosen NSIDE.

    ``significance`` is the OLS amplitude over its scatter across uncontaminated
    matched realisations; ``inflation`` is that scatter over the iid error, so
    ``significance * inflation`` is the iid significance of the same amplitude.
    """
    out = {}
    for f in sorted(glob.glob(str(SYSW_CAL / "LS10_VLIM_ANY_*_params.json"))):
        d = json.load(open(f))
        s = d.get("significance")
        if not s:
            continue
        mstar = d["sample_id"].split("_")[3]
        out[mstar] = dict(
            mstar=mstar, nside=d["nside"], ngal=d["n_galaxies"], npix=d["n_good_pix"],
            names=[n.split("_NSIDE")[0] for n in d["template_names"]],
            significance=np.asarray(s["significance"], float),
            inflation=np.asarray(s["inflation"], float),
            p_values=np.asarray(s["p_values"], float),
            family_wise_p=float(s["family_wise_p"]),
            p_floor=float(s["p_value_floor"]), n_null=int(s["n_null"]),
        )
    return out


def load_progressive():
    try:
        import pandas as pd
    except Exception:
        return None
    if not PROG.exists():
        return None
    df = pd.read_csv(PROG)
    df = df[df["mode"] == "additive"].copy()

    def _amp(s):
        a = np.asarray(ast.literal_eval(s), float)
        nz = a[a != 0]
        return float(np.abs(nz).mean()) if nz.size else 0.0

    df["amp"] = df["a_true"].apply(_amp)
    return df.groupby("k").agg(amp=("amp", "mean"), tp=("tp", "mean"),
                               fp=("fp", "mean"), reject=("lrt_reject", "mean")).reset_index()


def fsky(npix: int, nside: int) -> float:
    if _HAS_HP:
        return npix / hp.nside2npix(nside)
    return npix / (12 * nside**2)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save(fig, name):
    STATIC.mkdir(parents=True, exist_ok=True)
    p = STATIC / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_amin_vs_ngal(samples):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ng = np.array([s["ngal"] for s in samples], float)
    amin = np.array([a_min(3.0, s["npix"], s["sigma_hat"]) for s in samples])
    ax.scatter(ng, amin, c="C0", zorder=3, label="LS10 samples (real fits, NSIDE 64)")
    for s in samples:
        ax.annotate(s["mstar"], (s["ngal"], a_min(3.0, s["npix"], s["sigma_hat"])),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    grid = np.logspace(np.log10(ng.min() * 0.7), np.log10(ng.max() * 1.4), 100)
    ax.plot(grid, 3.0 / np.sqrt(grid), "k--", label=r"shot floor $3/\sqrt{N_{\rm gal}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$N_{\rm gal}$"); ax.set_ylabel(r"$A_{\min}(3\sigma)$  (field RMS)")
    ax.set_title("Smallest detectable systematic vs galaxy count")
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    return _save(fig, "fig1_Amin_vs_Ngal.png")


def fig_amin_vs_nside(rows):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ns = np.array([r["nside"] for r in rows])
    amin = np.array([a_min(3.0, r["npix"], r["sigma_hat"]) for r in rows])
    ngal = rows[0]["ngal"]
    ax.plot(ns, amin, "o-", c="C0", label=r"$A_{\min}(3\sigma)=3\hat\sigma/\sqrt{N_{\rm pix}}$ (real)")
    ax.axhline(3.0 / math.sqrt(ngal), ls="--", c="k",
               label=r"shot floor $3/\sqrt{N_{\rm gal}}$")
    for r in rows:
        ax.annotate(rf"$\bar n$={r['nbar']:.0f}", (r["nside"], a_min(3.0, r["npix"], r["sigma_hat"])),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(ns); ax.set_xticklabels(ns)
    ax.set_xlabel("NSIDE (pixel size)"); ax.set_ylabel(r"$A_{\min}(3\sigma)$")
    ax.set_title(f"Pixel size: LS10 log$M_*\\geq${rows[0]['mstar']} at fixed $N_{{\\rm gal}}$")
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    return _save(fig, "fig2_Amin_vs_nside.png")


def fig_crossover(rows):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    nbar = np.array([r["nbar"] for r in rows])
    shot = 1.0 / nbar
    clus = np.array([max(r["sigma_hat"] ** 2 - 1.0 / r["nbar"], 1e-6) for r in rows])
    ax.plot(nbar, shot, "s-", c="C1", label=r"shot $1/\bar n_{\rm pix}$")
    ax.plot(nbar, clus, "o-", c="C0", label=r"clustering $\hat\sigma^2-1/\bar n_{\rm pix}$ (measured)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\bar n_{\rm pix}$ (galaxies / pixel)")
    ax.set_ylabel(r"per-pixel variance contribution")
    ax.set_title("Shot vs clustering: which limits the detection")
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    ax.annotate("coarse pixels /\nhigh density", (nbar.max(), shot.min()), fontsize=7,
                ha="right", va="bottom", color="grey")
    return _save(fig, "fig3_crossover.png")


def fig_neff(rows):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ns = np.array([r["nside"] for r in rows])
    frac = np.array([n_eff(r["npix"], r["sigma_hat"]) / r["ngal"] for r in rows])
    ax.plot(ns, frac, "o-", c="C0")
    ax.axhline(1.0, ls=":", c="k", label=r"$N_{\rm eff}=N_{\rm gal}$ (pure shot)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(list(ns))
    ax.set_xticklabels(list(ns))
    ax.set_xlabel("NSIDE"); ax.set_ylabel(r"$N_{\rm eff}/N_{\rm gal}$")
    ax.set_title(r"Clustering penalty: usable fraction of galaxies")
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    return _save(fig, "fig4_Neff_fraction.png")


def fig_pertemplate(cal):
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    order = np.argsort(cal["significance"])[::-1]
    names = [cal["names"][i] for i in order]
    sig = cal["significance"][order]
    iid = sig * cal["inflation"][order]
    x = np.arange(len(sig))
    ax.bar(x, iid, color="C0", label="iid (white-noise error)")
    ax.bar(x, sig, color="C3", width=0.5,
           label=f"calibrated ({cal['n_null']} matched realisations)")
    ax.axhline(3.0, ls="--", c="k", lw=1, label=r"$3\sigma$")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(r"per-template $|\hat a_i|/\sigma_i$")
    rel = r"\leq " if cal["family_wise_p"] <= cal["p_floor"] else "="
    ax.set_title(f"LS10 log$M_*\\geq${cal['mstar']} (NSIDE {cal['nside']}): "
                 f"family-wise $p{rel}{cal['family_wise_p']:.4f}$")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    return _save(fig, "fig5_per_template_snr.png")


def fig_detection_vs_amp(prog):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    if prog is not None and len(prog):
        ax.plot(prog["amp"], prog["tp"], "o-", c="C0", label="true-positive rate")
        ax.plot(prog["amp"], prog["fp"], "s--", c="C3", label="false-positive rate")
        ax.plot(prog["amp"], prog["reject"], "^:", c="C2", label="LRT reject-null rate")
        ax.set_xlabel("injected additive amplitude (per template)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Fast-method detection vs amplitude (progressive mocks, NSIDE 64)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    return _save(fig, "fig6_detection_vs_amplitude.png")


# ---------------------------------------------------------------------------
# Scorecard CSV
# ---------------------------------------------------------------------------

def write_scorecard(rows):
    STATIC.mkdir(parents=True, exist_ok=True)
    p = STATIC / "ls10_detectability_scorecard.csv"
    cols = ["survey", "sample", "nside", "n_pix", "n_gal", "nbar", "fsky",
            "sigma_hat", "neff_over_ngal", "regime", "Amin_3sig", "Amin_5sig",
            "Amin_shot_3sig", "top_template", "top_snr_iid"]
    lines = [",".join(cols)]
    for r in rows:
        snr = np.abs(r["a_hat"]) / np.sqrt(r["var_a"]) if r["var_a"].size else np.array([np.nan])
        top = int(np.nanargmax(snr)) if snr.size else 0
        rec = [
            "LS10", f"logM>={r['mstar']}", r["nside"], r["npix"], r["ngal"],
            f"{r['nbar']:.2f}", f"{fsky(r['npix'], r['nside']):.4f}",
            f"{r['sigma_hat']:.4f}",
            f"{n_eff(r['npix'], r['sigma_hat']) / r['ngal']:.4f}",
            regime(r["nbar"], r["sigma_hat"]),
            f"{a_min(3.0, r['npix'], r['sigma_hat']):.3e}",
            f"{a_min(5.0, r['npix'], r['sigma_hat']):.3e}",
            f"{a_min_shot(3.0, r['ngal']):.3e}",
            r["names"][top] if r["names"] else "",
            f"{snr[top]:.2f}",
        ]
        lines.append(",".join(str(x) for x in rec))
    p.write_text("\n".join(lines) + "\n")
    return p


def write_calibrated_table(cal):
    """One row per sample at its chosen NSIDE; the leading template by calibrated significance."""
    STATIC.mkdir(parents=True, exist_ok=True)
    p = STATIC / "ls10_calibrated_significance.csv"
    lines = ["log M* >=,NSIDE,galaxies per pixel,leading template,calibrated significance,"
             "family-wise p,median inflation,inflation range"]
    for mstar in sorted(cal, key=float):
        c = cal[mstar]
        i = int(np.argmax(c["significance"]))
        fwp = (f"<= {c['p_floor']:.4f}" if c["family_wise_p"] <= c["p_floor"]
               else f"{c['family_wise_p']:.4f}")
        lines.append(",".join([
            mstar, str(c["nside"]), f"{c['ngal'] / c['npix']:.1f}", c["names"][i],
            f"{c['significance'][i]:.2f}", fwp, f"{np.median(c['inflation']):.1f}",
            f"{c['inflation'].min():.1f}-{c['inflation'].max():.1f}",
        ]))
    p.write_text("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# RST page
# ---------------------------------------------------------------------------

_RST = r"""Systematic-detectability law
============================

This page gives the smallest systematic in galaxy density that a survey of given
sky coverage, pixel size and galaxy density can detect, and at what significance.
The law is analytic; its normalisation comes from the LS10 fit outputs and the
calibrated significances. The GLASS-mock sweep that tests the exponents is on
:doc:`survey_design_synthesis`.

The law
-------

We model the overdensity per pixel as a systematic field plus noise,
:math:`\delta_g = f + \varepsilon`, with :math:`f=\sum_i a_i t_i` for templates of
zero mean and unit variance. The per-pixel noise variance :math:`\hat\sigma^2` is the
residual scatter of the fit, shot noise plus clustering:
:math:`\hat\sigma^2 = 1/\bar n_{\rm pix} + \sigma_{\rm clus}^2`. With
:math:`N_{\rm gal}=\bar n_{\rm pix} N_{\rm pix}` and field rms :math:`A={\rm rms}(f)`,

.. math::

   {\rm SNR}_{\rm field} = \frac{\lVert f\rVert}{\hat\sigma}
   = A\,\frac{\sqrt{N_{\rm pix}}}{\hat\sigma} = A\sqrt{N_{\rm eff}},
   \qquad
   A_{\min}(\nu\sigma) = \nu\,\frac{\hat\sigma}{\sqrt{N_{\rm pix}}},
   \qquad
   N_{\rm eff}\equiv\frac{N_{\rm pix}}{\hat\sigma^2}\le N_{\rm gal}.

In the shot-noise limit this reduces to :math:`A_{\min}=\nu/\sqrt{N_{\rm gal}}`.
The field statistic does not depend on how collinear the templates are, whereas the
amplitude of template :math:`i` carries the variance-inflation factor
:math:`{\rm VIF}_i=1/\sqrt{1-R_i^2}`; the standardised LS10 basis at NSIDE 64 has
condition number :math:`1.4\times10^{3}`.

Rules of thumb
--------------

* In the shot regime, :math:`\bar n_{\rm pix}<1/\sigma_{\rm clus}^2`,
  :math:`A_{\min}\propto 1/\sqrt{N_{\rm gal}}`. In the clustering regime it
  saturates at a floor set by the number of independent modes,
  :math:`\propto 1/\sqrt{f_{\rm sky}}`, which more area lowers and more depth does not.
* At fixed :math:`N_{\rm gal}`, :math:`A_{\min}=\nu\hat\sigma/\sqrt{N_{\rm pix}}` falls
  as pixels refine, down to the shot floor :math:`\nu/\sqrt{N_{\rm gal}}`. Pixels
  finer than the coherence scale of the systematic add nothing.
* The independent-pixel error :math:`\sigma_i` ignores the correlation of the
  clustered field between pixels. Against %%NNULL%% uncontaminated GLASS
  realisations carrying each sample's matched spectrum, the median over templates of
  the amplitude scatter divided by :math:`\sigma_i` is %%INFLMIN%% to %%INFLMAX%% per
  sample and rises with resolution (%%INFLBYNSIDE%%). Single templates span
  %%TPLMIN%% to %%TPLMAX%%. ``sys_mapping.calibrated_template_significance`` measures
  the factor per template.
* A search over :math:`n_{\rm sys}` templates reports the largest significance, so
  its p-value is read from the largest significance of each null realisation (the
  family-wise p).
* The contamination of :math:`w(\theta)` grows as :math:`A^2`, so the field
  regression detects fainter systematics than :math:`w(\theta)` does.

LS10 worked example (log :math:`M_*\ge` %%FID%%)
------------------------------------------------

The fiducial sample has :math:`N_{\rm gal}=`\ %%NGAL%% and
:math:`f_{\rm sky}\approx`\ %%FSKY%%, and is %%REGIME%%. At NSIDE 64 the per-pixel noise
is :math:`\hat\sigma=`\ %%SHAT%% for :math:`\bar n_{\rm pix}=`\ %%NBAR%% (shot term
%%SHOT%%), so :math:`N_{\rm eff}/N_{\rm gal}=`\ %%NEFF%%. The smallest detectable field
rms is :math:`A_{\min}(3\sigma)=`\ %%AMIN3%% (:math:`5\sigma`: %%AMIN5%%), against a
shot floor of %%AMINSHOT%%. The leading template at NSIDE 64 is %%TOP%%, at
independent-pixel SNR %%TOPSNR%%.

Occupancy puts this sample at NSIDE %%CALNSIDE%% (%%CALNBAR%% galaxies per pixel,
floor 25). There the leading template is %%CALTOP%%, at calibrated significance
%%CALSIG%% (independent-pixel %%CALIID%%, inflation %%CALINFL%%) and family-wise
:math:`p` %%CALFWP%% from %%NNULL%% realisations. It traces the Gaia stellar density.

.. csv-table:: Calibrated significance per sample, each at the resolution its occupancy supports.
   The family-wise p is bounded below by 1/(N+1) for N realisations.
   :file: _static/detectability_law/ls10_calibrated_significance.csv
   :header-rows: 1

.. figure:: /_static/detectability_law/fig1_Amin_vs_Ngal.png
   :width: 88%

   Smallest detectable systematic against galaxy count for the nine LS10
   stellar-mass samples at NSIDE 64. At fixed footprint
   :math:`A_{\min}\propto\hat\sigma`, so the intermediate-mass samples, with the
   lowest :math:`\hat\sigma`, are the most sensitive.

.. figure:: /_static/detectability_law/fig2_Amin_vs_nside.png
   :width: 88%

   :math:`A_{\min}` against NSIDE (32 to 128) for the fiducial sample, with the shot
   floor.

.. figure:: /_static/detectability_law/fig3_crossover.png
   :width: 88%

   Shot and clustering per-pixel variance. LS10 is clustering-limited at every NSIDE
   tested.

.. figure:: /_static/detectability_law/fig4_Neff_fraction.png
   :width: 88%

   :math:`N_{\rm eff}/N_{\rm gal}`, equal to 1 for pure shot noise.

.. figure:: /_static/detectability_law/fig5_per_template_snr.png
   :width: 92%

   Per-template significance of the fiducial sample at NSIDE %%CALNSIDE%%,
   independent-pixel against calibrated on matched realisations.

.. figure:: /_static/detectability_law/fig6_detection_vs_amplitude.png
   :width: 88%

   Detection fraction against injected amplitude for OLS, ISD-1, ElasticNet and the
   likelihood-ratio test on the progressive mocks.

The per-sample (NSIDE 64) and per-NSIDE (fiducial sample) numbers are in
``_static/detectability_law/ls10_detectability_scorecard.csv``, the calibrated
significances in ``_static/detectability_law/ls10_calibrated_significance.csv``.

Reproduce
---------

.. code-block:: bash

   python scripts/analyze_detectability_law.py
   bash bash/build_docs.sh

The GLASS-mock sweep over ``nside × density × f_sky × amplitude`` is run with
``scripts/run_detectability_sweep.py`` (``--check`` validates the inputs; ``--fskys``
sets the footprint fractions) and staged by ``bash/run_remote_full.sh``:

.. code-block:: bash

   bash bash/run_remote_full.sh check
   bash bash/run_remote_full.sh sweep_ls10 sweep_euclid mcmc_anchors
   RESUME=1 bash bash/run_remote_full.sh sweep_euclid
"""


def write_rst(fid, rows_ns, cal):
    nside64 = next(r for r in rows_ns if r["nside"] == 64)
    snr = np.abs(fid["a_hat"]) / np.sqrt(fid["var_a"])
    top = int(np.nanargmax(snr))
    c = cal[FIDUCIAL]
    ctop = int(np.argmax(c["significance"]))
    med = {m: float(np.median(v["inflation"])) for m, v in cal.items()}
    by_nside = {}
    for m, v in cal.items():
        by_nside.setdefault(v["nside"], []).append(med[m])
    infl_by_nside = "; ".join(
        f"NSIDE {ns}: {min(v):.1f}" + (f" to {max(v):.1f}" if max(v) > min(v) else "")
        for ns, v in sorted(by_nside.items()))
    n_null = sorted({v["n_null"] for v in cal.values()})
    fwp = (f"≤ {c['p_floor']:.4f}" if c["family_wise_p"] <= c["p_floor"]
           else f"= {c['family_wise_p']:.4f}")
    subs = {
        "%%NNULL%%": "/".join(str(n) for n in n_null),
        "%%INFLMIN%%": f"{min(med.values()):.1f}",
        "%%INFLMAX%%": f"{max(med.values()):.1f}",
        "%%INFLBYNSIDE%%": infl_by_nside,
        "%%TPLMIN%%": f"{min(v['inflation'].min() for v in cal.values()):.1f}",
        "%%TPLMAX%%": f"{max(v['inflation'].max() for v in cal.values()):.1f}",
        "%%CALNSIDE%%": str(c["nside"]),
        "%%CALNBAR%%": f"{c['ngal'] / c['npix']:.1f}",
        "%%CALTOP%%": c["names"][ctop],
        "%%CALSIG%%": f"{c['significance'][ctop]:.2f}",
        "%%CALIID%%": f"{c['significance'][ctop] * c['inflation'][ctop]:.1f}",
        "%%CALFWP%%": fwp,
        "%%CALINFL%%": f"{c['inflation'][ctop]:.2f}",
        "%%FID%%": FIDUCIAL,
        "%%NGAL%%": f"{fid['ngal']:,}",
        "%%FSKY%%": f"{fsky(nside64['npix'], 64):.3f}",
        "%%REGIME%%": regime(nside64["nbar"], nside64["sigma_hat"]),
        "%%SHAT%%": f"{nside64['sigma_hat']:.3f}",
        "%%NBAR%%": f"{nside64['nbar']:.0f}",
        "%%SHOT%%": f"{1.0 / nside64['nbar']:.4f}",
        "%%NEFF%%": f"{n_eff(nside64['npix'], nside64['sigma_hat']) / nside64['ngal']:.3f}",
        "%%AMIN3%%": f"{a_min(3.0, nside64['npix'], nside64['sigma_hat']):.2e}",
        "%%AMIN5%%": f"{a_min(5.0, nside64['npix'], nside64['sigma_hat']):.2e}",
        "%%AMINSHOT%%": f"{a_min_shot(3.0, nside64['ngal']):.2e}",
        "%%TOP%%": fid["names"][top],
        "%%TOPSNR%%": f"{snr[top]:.1f}",
    }
    body = _RST
    for k, v in subs.items():
        body = body.replace(k, v)
    RST.write_text(body)
    return RST


def main():
    samples64 = load_ls10_samples("0064")
    rows_ns = load_ls10_across_nside(FIDUCIAL)
    fid = next(r for r in rows_ns if r["nside"] == 64)
    prog = load_progressive()
    cal = load_calibrated()
    if FIDUCIAL not in cal:
        raise SystemExit(f"no calibrated product for log M* >= {FIDUCIAL} under {SYSW_CAL}")

    fig_amin_vs_ngal(samples64)
    fig_amin_vs_nside(rows_ns)
    fig_crossover(rows_ns)
    fig_neff(rows_ns)
    fig_pertemplate(cal[FIDUCIAL])
    fig_detection_vs_amp(prog)

    # Scorecard: all 9 samples at NSIDE 64 (A_min-vs-N_gal) + fiducial across NSIDE.
    rows_scorecard = samples64 + [r for r in rows_ns if r["nside"] != 64]
    sc = write_scorecard(rows_scorecard)
    ct = write_calibrated_table(cal)
    rst = write_rst(fid, rows_ns, cal)
    print(f"[detectability_law] scorecard -> {sc}")
    print(f"[detectability_law] calibrated-> {ct}")
    print(f"[detectability_law] page      -> {rst}")
    print(f"[detectability_law] figures   -> {STATIC}")
    for r in rows_ns:
        print(f"  NSIDE {r['nside']:4d}: nbar={r['nbar']:7.1f} sigma_hat={r['sigma_hat']:.3f} "
              f"Amin(3s)={a_min(3.0, r['npix'], r['sigma_hat']):.2e} "
              f"Neff/Ngal={n_eff(r['npix'], r['sigma_hat']) / r['ngal']:.3f} [{regime(r['nbar'], r['sigma_hat'])}]")


if __name__ == "__main__":
    main()
