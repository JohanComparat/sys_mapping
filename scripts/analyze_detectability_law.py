#!/usr/bin/env python
"""Systematic-detectability law + LS10 worked example (Stage 1).

Rapid, analytic + reuse-only answer to: *given a survey setup (sky coverage,
pixel size, galaxy density), what is the smallest systematic affecting galaxy
number density that can be detected, and at what significance?*  No simulations
are run here -- the law is analytic and its normalisation is read from existing
LS10 fit outputs on disk.

The operational, empirically-anchored law (derived in the generated page):

    field detection SNR   D  = A * sqrt(N_pix) / sigma_hat = A * sqrt(N_eff)
    smallest detectable   A_min(nu) = nu * sigma_hat / sqrt(N_pix)
    effective galaxy count N_eff = N_pix / sigma_hat**2   (<= N_gal)

where ``A`` is the systematic field RMS, ``N_pix`` the number of fit pixels,
``sigma_hat`` the fit's residual per-pixel scatter (the honest per-pixel noise,
shot (+) clustering), and the shot-noise floor is ``A_min = nu / sqrt(N_gal)``.

Self-generates ``docs/detectability_law.rst`` and figures/CSVs in
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
    for nsstr in ("0032", "0064", "0128", "0256"):
        for r in load_ls10_samples(nsstr):
            if r["mstar"] == mstar:
                rows.append(r)
    rows.sort(key=lambda r: r["nside"])
    return rows


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


def fig_pertemplate(fid):
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    snr = np.abs(fid["a_hat"]) / np.sqrt(fid["var_a"])
    order = np.argsort(snr)[::-1]
    names = [fid["names"][i] for i in order]
    snr = snr[order]
    x = np.arange(len(snr))
    ax.bar(x, snr, color="C0", label="iid SNR (optimistic)")
    ax.bar(x, snr / IID_INFLATION, color="C3", width=0.5,
           label=rf"calibrated SNR ($\div{IID_INFLATION}$, sandwich)")
    ax.axhline(3.0, ls="--", c="k", lw=1, label=r"$3\sigma$")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(r"per-template $|\hat a_i|/\sigma_i$")
    ax.set_title(f"LS10 log$M_*\\geq${fid['mstar']} detections (NSIDE {fid['nside']}) — stellar density dominates")
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
            "Amin_shot_3sig", "top_template", "top_snr_iid", "top_snr_cal"]
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
            f"{snr[top]:.2f}", f"{snr[top] / IID_INFLATION:.2f}",
        ]
        lines.append(",".join(str(x) for x in rec))
    p.write_text("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# RST page
# ---------------------------------------------------------------------------

# Empirical iid->calibrated (sandwich) inflation of per-template SNR. The iid
# posterior sigma ignores the correlated field and is ~2x too tight for every
# method (see results_glass_slow_methods / results_snr_preselection).
IID_INFLATION = 1.9


_RST = r"""Systematic-detectability law — survey design rules of thumb
===========================================================

.. note::
   Rapid, analytic + reuse-only answer (Stage 1) to: *given a survey setup
   (sky coverage, pixel size, galaxy density), what is the smallest systematic
   affecting galaxy number density that can be detected, and at what
   significance?* No simulations are run; the law is analytic and its
   normalisation is read from existing LS10 fit outputs. The corresponding
   analysis for other surveys lives in their own repositories
   (``results_glass_detectability_law``); the cross-survey combination is on
   :doc:`survey_design_synthesis`.

The law
-------

Model the observed overdensity per pixel as a systematic field plus noise,
:math:`\delta_g = f + \varepsilon`, with :math:`f=\sum_i a_i t_i` (templates
mean-subtracted, unit-variance) and per-pixel noise variance
:math:`\hat\sigma^2` — the fit's residual scatter, which is **shot ⊕ clustering**:
:math:`\hat\sigma^2 = 1/\bar n_{\rm pix} + \sigma_{\rm clus}^2`. With
:math:`N_{\rm gal}=\bar n_{\rm pix} N_{\rm pix}` and field RMS :math:`A={\rm rms}(f)`:

.. math::

   {\rm SNR}_{\rm field} = \frac{\lVert f\rVert}{\hat\sigma}
   = A\,\frac{\sqrt{N_{\rm pix}}}{\hat\sigma} = A\sqrt{N_{\rm eff}},
   \qquad
   \boxed{\,A_{\min}(\nu\sigma) = \nu\,\frac{\hat\sigma}{\sqrt{N_{\rm pix}}}\,},
   \qquad
   N_{\rm eff}\equiv\frac{N_{\rm pix}}{\hat\sigma^2}\le N_{\rm gal}.

The single substitution :math:`\sqrt{N_{\rm gal}}\to\sqrt{N_{\rm eff}}` carries
the shot-noise idealisation :math:`A_{\min}=\nu/\sqrt{N_{\rm gal}}` into the
correlated-field reality. The field statistic is **VIF-free** (the recovered
combination is well-constrained even when individual templates are collinear —
the LS10 basis has condition number :math:`\sim10^8`), whereas the
per-template amplitude carries a variance-inflation factor
:math:`{\rm VIF}_i=1/\sqrt{1-R_i^2}`. This is the quantitative form of *judge by
the field, not the name*.

Rules of thumb
--------------

* **More galaxies help only until clustering dominates.** In the shot regime
  (:math:`\bar n_{\rm pix}<1/\sigma_{\rm clus}^2`) :math:`A_{\min}\propto
  1/\sqrt{N_{\rm gal}}`; in the clustering regime it saturates at a
  cosmic-variance floor set by the number of independent modes,
  :math:`\propto 1/\sqrt{f_{\rm sky}}` — **more area, not more depth**, lowers it.
* **Finer pixels resolve more modes.** At fixed :math:`N_{\rm gal}`,
  :math:`A_{\min}=\nu\hat\sigma/\sqrt{N_{\rm pix}}` keeps falling as pixels
  refine (more independent measurements of the same smooth systematic) until the
  shot floor :math:`\nu/\sqrt{N_{\rm gal}}` — refine to just below the
  systematic's coherence scale, no finer.
* **The honest error bar is the sandwich.** The per-template iid
  :math:`\sigma_i` is :math:`\sim2\times` too tight for every method; multiply
  detection SNRs by :math:`\approx1/%%IID%%` (or use
  ``sys_mapping.mock_sandwich_covariance``).
* **w(θ) is a weaker detector** — its contamination signal grows as
  :math:`A^2`, so the direct field regression sees fainter systematics.

LS10 worked example (log :math:`M_*\ge` %%FID%%)
------------------------------------------------

The fiducial LS10 volume-limited sample (:math:`N_{\rm gal}=`\ %%NGAL%%,
:math:`f_{\rm sky}\approx`\ %%FSKY%%) is **%%REGIME%%**: at the recommended
NSIDE 64 the per-pixel noise :math:`\hat\sigma=`\ %%SHAT%% is dominated by
clustering (:math:`\bar n_{\rm pix}=`\ %%NBAR%%, shot
:math:`1/\bar n_{\rm pix}=`\ %%SHOT%%), so only :math:`N_{\rm eff}/N_{\rm
gal}=`\ %%NEFF%% of the galaxies count toward detection. The smallest detectable
systematic field RMS is :math:`A_{\min}(3\sigma)=`\ %%AMIN3%% (:math:`5\sigma`:
%%AMIN5%%), versus the shot-floor %%AMINSHOT%%. The dominant real detection is
**%%TOP%%** at iid SNR %%TOPSNR%% (:math:`\approx`\ %%TOPCAL%% calibrated) — the
known LS10 BGS stellar-density systematic.

.. figure:: /_static/detectability_law/fig1_Amin_vs_Ngal.png
   :width: 88%

   Smallest detectable systematic vs galaxy count across the nine LS10
   stellar-mass samples (real fits). :math:`A_{\min}\propto\hat\sigma` at fixed
   footprint, so sensitivity peaks for the intermediate-mass samples (lowest
   :math:`\hat\sigma`), not the most numerous.

.. figure:: /_static/detectability_law/fig2_Amin_vs_nside.png
   :width: 88%

   Pixel size at fixed :math:`N_{\rm gal}`: refining NSIDE 32→256 lowers
   :math:`A_{\min}` toward the shot floor (more resolved modes).

.. figure:: /_static/detectability_law/fig3_crossover.png
   :width: 88%

   Shot vs clustering per-pixel variance. LS10 sits deep in the
   clustering-limited regime at every NSIDE tested.

.. figure:: /_static/detectability_law/fig4_Neff_fraction.png
   :width: 88%

   The usable fraction :math:`N_{\rm eff}/N_{\rm gal}` (1 = pure shot noise).

.. figure:: /_static/detectability_law/fig5_per_template_snr.png
   :width: 92%

   Per-template detection SNR (iid vs sandwich-calibrated); stellar density
   dominates. Individual templates are collinear (VIF-inflated); the *field* is
   robust.

.. figure:: /_static/detectability_law/fig6_detection_vs_amplitude.png
   :width: 88%

   Fast-method (OLS/ISD-1/ElasticNet + LRT) detection vs amplitude on the
   progressive mocks.

The per-sample (NSIDE 64) and per-NSIDE (fiducial sample) numbers are tabulated in
``_static/detectability_law/ls10_detectability_scorecard.csv``.

Empirical sweep on the remote (Stage 2 — run)
---------------------------------------------

The analytic curves above are anchored at the *measured* operating points. They
have been traced empirically over the full ``nside × density × f_sky × amplitude``
grid by a controlled sweep, which **has been run** on the compute host (14 400
LS10-geometry fits, 8 100 Euclid-geometry fits, 240 MCMC-add anchors, 30
simulations per cell). The fitted exponents, and the two levers that do *not*
come out at :math:`-1/2`, are reported on the :doc:`survey_design_synthesis`
page; this page is unaffected by that run, since its numbers come from the
measured LS10 weight maps rather than from mocks.

.. code-block:: bash

   # dry-run / input check only (zero compute):
   python scripts/run_detectability_sweep.py --check
   bash bash/run_remote_full.sh check
   # the run itself (compute host; ~7 core-h at the committed defaults):
   bash bash/run_remote_full.sh sweep_ls10 sweep_euclid mcmc_anchors
   # continue after a wall-clock timeout:
   RESUME=1 bash bash/run_remote_full.sh sweep_euclid

See ``scripts/run_detectability_sweep.py`` for the knobs (it adds the missing
``--fsky`` footprint-fraction dial) and ``bash/run_remote_full.sh`` for the
staged recipe and resource budget.

Reproduce
---------

.. code-block:: bash

   python scripts/analyze_detectability_law.py
   bash bash/build_docs.sh
"""


def write_rst(fid, rows_ns):
    nside64 = next(r for r in rows_ns if r["nside"] == 64)
    snr = np.abs(fid["a_hat"]) / np.sqrt(fid["var_a"])
    top = int(np.nanargmax(snr))
    subs = {
        "%%IID%%": f"{IID_INFLATION}",
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
        "%%TOPCAL%%": f"{snr[top] / IID_INFLATION:.1f}",
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

    fig_amin_vs_ngal(samples64)
    fig_amin_vs_nside(rows_ns)
    fig_crossover(rows_ns)
    fig_neff(rows_ns)
    fig_pertemplate(fid)
    fig_detection_vs_amp(prog)

    # Scorecard: all 9 samples at NSIDE 64 (A_min-vs-N_gal) + fiducial across NSIDE.
    rows_scorecard = samples64 + [r for r in rows_ns if r["nside"] != 64]
    sc = write_scorecard(rows_scorecard)
    rst = write_rst(fid, rows_ns)
    print(f"[detectability_law] scorecard -> {sc}")
    print(f"[detectability_law] page      -> {rst}")
    print(f"[detectability_law] figures   -> {STATIC}")
    for r in rows_ns:
        print(f"  NSIDE {r['nside']:4d}: nbar={r['nbar']:7.1f} sigma_hat={r['sigma_hat']:.3f} "
              f"Amin(3s)={a_min(3.0, r['npix'], r['sigma_hat']):.2e} "
              f"Neff/Ngal={n_eff(r['npix'], r['sigma_hat']) / r['ngal']:.3f} [{regime(r['nbar'], r['sigma_hat'])}]")


if __name__ == "__main__":
    main()
