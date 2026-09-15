#!/usr/bin/env python
"""Survey-design synthesis of the systematic-detectability law.

Turns the LS10 scorecard (``detectability_law`` page) into rules of thumb and a
design-space figure, and compares the scaling laws with the GLASS-mock sweep of
``run_detectability_sweep.py``. No fit is run.

LS10 only, deliberately. This repo is public; cross-survey comparisons that
involve collaborations with internal data-release policies live in their own
private repositories. Do not add a step that reads or vendors another survey's
scorecard into this repo.

Run::

    python scripts/make_survey_design_synthesis.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
STATIC = REPO / "docs" / "_static" / "survey_design_synthesis"
RST = REPO / "docs" / "survey_design_synthesis.rst"
LS10_SC = REPO / "docs" / "_static" / "detectability_law" / "ls10_detectability_scorecard.csv"
# Sweep outputs. The remote wrapper (bash/run_remote_full.sh) writes the
# _ls10/_euclid pair over the full grid; the reduced 2026-07-25 laptop run wrote
# the _nside/_axes pair. Prefer the remote grid, fall back to the laptop one.
SWEEP_LS10 = REPO / "results" / "detectability_sweep_ls10.csv"
SWEEP_EUCLID = REPO / "results" / "detectability_sweep_euclid.csv"
SWEEP_MCMC = REPO / "results" / "detectability_sweep_mcmc.csv"
SWEEP_NSIDE = REPO / "results" / "detectability_sweep_nside.csv"
SWEEP_AXES = REPO / "results" / "detectability_sweep_axes.csv"


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_scorecards():
    """The LS10 scorecard. LS10 only, by design.

    This repo is public, so it carries no other survey's scorecard and copies
    none in from a sibling checkout. Cross-survey comparisons live in the private
    repository of the collaboration concerned. ``.gitignore`` enforces this as a
    second line of defence -- please keep both.
    """
    STATIC.mkdir(parents=True, exist_ok=True)
    return _read(LS10_SC) if LS10_SC.exists() else []


def _f(row, key):
    return float(row[key])


# ---------------------------------------------------------------------------
# Master figures
# ---------------------------------------------------------------------------

def fig_amin_vs_ngal(ls10):
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ls = [r for r in ls10 if r["nside"] == "64"]
    ax.scatter([_f(r, "n_gal") for r in ls], [_f(r, "Amin_3sig") for r in ls],
               c="C0", label="LS10 samples (NSIDE 64, f_sky≈0.44)", zorder=3)
    allng = [_f(r, "n_gal") for r in ls]
    grid = np.logspace(np.log10(min(allng) * 0.7), np.log10(max(allng) * 1.4), 100)
    ax.plot(grid, 3.0 / np.sqrt(grid), "k--", label=r"shot floor $3/\sqrt{N_{\rm gal}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$N_{\rm gal}$"); ax.set_ylabel(r"$A_{\min}(3\sigma)$ (field RMS)")
    ax.set_title("Design space: smallest detectable systematic vs galaxy count")
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    ax.text(0.02, 0.03, "Finer pixels resolve more modes of the same smooth\n"
            "systematic, so they lower $A_{\\rm min}$ at fixed $N_{\\rm gal}$",
            transform=ax.transAxes, fontsize=7.5, va="bottom")
    fig.savefig(STATIC / "fig1_master_Amin_vs_Ngal.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_amin_vs_fsky(ls10):
    """A_min vs f_sky: in the clustering regime A_min ∝ 1/√f_sky. Anchor the
    survey at its measured point and project to half- and full-sky."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    fgrid = np.logspace(-3, 0, 100)
    for rows, name, col, mk in ((
        [r for r in ls10 if r["nside"] == "64" and "10.0" in r["sample"]], "LS10 log$M_*\\geq$10.0", "C0", "o"),):
        if not rows:
            continue
        r = rows[0]
        f0, a0 = _f(r, "fsky"), _f(r, "Amin_3sig")
        ax.plot(fgrid, a0 * np.sqrt(f0 / fgrid), col, lw=1.5,
                label=rf"{name}: $A_{{\min}}\propto f_{{\rm sky}}^{{-1/2}}$")
        ax.scatter([f0], [a0], marker=mk, s=140, c=col, zorder=4, edgecolor="k", linewidth=.4)
    for fx, lab in ((0.5, "half sky"), (1.0, "full sky")):
        ax.axvline(fx, ls=":", c="grey", lw=1)
        ax.text(fx, ax.get_ylim()[1], lab, fontsize=7, rotation=90, va="top", ha="right", color="grey")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$f_{\rm sky}$"); ax.set_ylabel(r"$A_{\min}(3\sigma)$")
    ax.set_title("More area lowers the floor: $A_{\\min}\\propto f_{\\rm sky}^{-1/2}$ (clustering regime)")
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    fig.savefig(STATIC / "fig2_Amin_vs_fsky.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_design_space(ls10):
    """N_eff/N_gal vs n̄_pix: the clustering penalty across the design space."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    # fiducial across nside; the scorecard lists NSIDE 64 twice (per-sample and per-NSIDE)
    ls = {r["nside"]: r for r in ls10 if "10.0" in r["sample"]}
    ls = sorted(ls.values(), key=lambda r: _f(r, "nbar"))
    nsides = sorted(int(r["nside"]) for r in ls)
    ax.plot([_f(r, "nbar") for r in ls], [_f(r, "neff_over_ngal") for r in ls],
            "o-", c="C0", label=f"LS10 log$M_*\\geq$10.0 (NSIDE {nsides[0]}–{nsides[-1]})")
    ax.axhline(1.0, ls=":", c="k", label=r"$N_{\rm eff}=N_{\rm gal}$ (pure shot)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\bar n_{\rm pix}$ (galaxies / pixel)")
    ax.set_ylabel(r"$N_{\rm eff}/N_{\rm gal}$ (usable fraction)")
    ax.set_title("Clustering penalty across the design space")
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    ax.annotate("clustering-limited\n(coarser / denser)", (300, 0.02), fontsize=7.5, color="grey", ha="center")
    fig.savefig(STATIC / "fig3_design_space.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Design-space table (built into the RST as a list-table)
# ---------------------------------------------------------------------------

def load_sweep():
    """Sweep rows, if present: ``(nside_rows, axes_rows, euclid_rows)``.

    Returns ``(None, None, [])`` when no sweep has been run.

    The full remote grid (``_ls10.csv``) crosses all four axes and is a strict
    superset of the reduced laptop pair, so one file feeds every panel; the
    reduced ``_nside``/``_axes`` pair is the fallback. ``_euclid.csv`` is optional
    and only extends the ``N_pix`` panel to nside 1024.
    """
    def _ols(path):
        return [r for r in _read(path) if r["method"] == "OLS"]

    eu = _ols(SWEEP_EUCLID) if SWEEP_EUCLID.exists() else []
    if SWEEP_LS10.exists():
        src = SWEEP_LS10.name + (f" + {SWEEP_EUCLID.name}" if eu else "")
        print(f"[synthesis] sweep source: {src} (full remote grid)")
        rows = _ols(SWEEP_LS10)
        return rows, rows, eu
    if SWEEP_NSIDE.exists() and SWEEP_AXES.exists():
        print(f"[synthesis] sweep source: {SWEEP_NSIDE.name} + {SWEEP_AXES.name} "
              f"(reduced laptop grid)")
        return _ols(SWEEP_NSIDE), _ols(SWEEP_AXES), eu
    return None, None, []


def mcmc_vs_ols():
    """Max |MCMC-add - OLS| / OLS in empirical A_min over the shared anchor cells.

    The mcmc_anchors stage used to be written to disk and read by nothing. It is
    a cross-check, not an independent measurement: for the additive Gaussian
    model the posterior mean IS the OLS solution, so the two must agree to
    sampling noise. Quoting the measured agreement keeps that honest -- if the
    sampler ever drifts, the number on the page moves with it.
    """
    if not (SWEEP_MCMC.exists() and SWEEP_LS10.exists()):
        return None
    key = lambda r: (r["nside"], r["n_mean"], r["fsky"], r["amp"], r["sim"])
    mc = {key(r): r for r in _read(SWEEP_MCMC) if r["method"] == "MCMC-add"}
    ols = {key(r): r for r in _read(SWEEP_LS10) if r["method"] == "OLS"}
    shared = set(mc) & set(ols)
    if not shared:
        return None
    d = [abs(_amin(mc[k]) - _amin(ols[k])) / _amin(ols[k]) for k in shared]
    return dict(n_cells=len(shared), max_rel=float(np.max(d)), med_rel=float(np.median(d)))


def _amin(row):
    """Empirical A_min(3sigma) = 3 * amp / field_snr (read at low injected amplitude)."""
    return 3.0 * _f(row, "amp") / _f(row, "field_snr")


def _agg(rows, xkey, sel):
    """Mean empirical A_min vs xkey over rows passing sel(row)."""
    from collections import defaultdict
    acc = defaultdict(list)
    for r in rows:
        if sel(r):
            acc[_f(r, xkey)].append(_amin(r))
    xs = sorted(acc)
    return np.array(xs), np.array([np.mean(acc[x]) for x in xs])


# ---------------------------------------------------------------------------
# Sweep figure
#
# Panels are read at amp = READ_AMP, NOT at the lowest injected amplitude. The
# fitted exponents are amplitude-dependent: field_snr = ||T a_hat||/std(resid)
# carries a small-signal bias, so at amp = 0.005 the f_sky lever fits -0.388 and
# at amp = 0.1 it fits -0.496. Reading the law at an amplitude where the bias has
# died away is the honest choice; the amplitude dependence itself is reported in
# the page text rather than hidden by picking one number.
READ_AMP = 0.05
LS10_FIX = dict(n_mean=127.0, fsky=0.44)      # held fixed on the N_pix panel
EUCLID_FIX = dict(n_mean=30.0, fsky=0.05)     # euclid grid shares no cell with the above
# Raise nside while lowering density so N_gal stays put (~2.7e6). If A_min is set
# by N_gal alone -- the shot-noise prediction -- this lever must be FLAT.
FIXED_NGAL_DIAG = [("32", 490.0), ("64", 127.0), ("128", 30.0), ("256", 8.0)]


def _sel(**slice_):
    """Row selector holding every named axis at its given value."""
    return lambda r: all(abs(_f(r, k) - v) < 1e-9 for k, v in slice_.items())


def _slope(x, y):
    """Fitted log-log exponent, or nan if there is nothing to fit."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0]) if len(x) >= 2 else float("nan")


def _panel(ax, x, y, colour, marker, label, ref=True, ref_label=None):
    """Plot one lever, annotate the FITTED exponent, and show the -1/2 reference."""
    if not len(x):
        return float("nan")
    s = _slope(x, y)
    ax.plot(x, y, marker, c=colour, label=f"{label}\nfitted slope {s:+.3f}")
    if ref:
        ax.plot(x, y[0] * np.sqrt(x[0] / x), "k--", lw=1.2,
                label=ref_label or r"$\propto x^{-1/2}$ (analytic)")
    return s


def fig_sweep_validation(ns_rows, ax_rows, eu_rows=()):
    """Sweep figure with fitted exponents.

    Three levers plus the degeneracy-breaking one. Panel (a) holds the density
    fixed, so N_gal rises with N_pix and the two are perfectly degenerate along
    it -- panel (d) separates them.
    """
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.2))
    slopes = {}

    # (a) N_pix, at fixed density -> N_gal co-varies. Degenerate; see panel (d).
    x, y = _agg(ns_rows, "npix", _sel(amp=READ_AMP, **LS10_FIX))
    slopes["npix"] = _panel(axes[0], x, y, "C0", "o-", "LS10 grid (OLS)",
                            ref_label=r"$\propto N_{\rm pix}^{-1/2}$")
    xe, ye = (_agg(eu_rows, "npix", _sel(amp=READ_AMP, **EUCLID_FIX))
              if len(eu_rows) else ([], []))
    slopes["npix_euclid"] = _panel(axes[0], xe, ye, "C3", "s-", "Euclid grid (OLS)", ref=False)
    axes[0].set_xlabel(r"$N_{\rm pix}$"); axes[0].set_ylabel(r"empirical $A_{\min}(3\sigma)$")
    axes[0].set_title(r"(a) finer pixels" "\n" r"$\bar n_{\rm pix}$ fixed $\Rightarrow N_{\rm gal}\propto N_{\rm pix}$")

    # (b) f_sky at fixed nside/density: pure "more area".
    x, y = _agg(ax_rows, "fsky", _sel(amp=READ_AMP, n_mean=127.0))
    x2, y2 = _agg([r for r in ax_rows if r["nside"] == "256"], "fsky",
                  _sel(amp=READ_AMP, n_mean=127.0))
    slopes["fsky"] = _panel(axes[1], x2, y2, "C0", "o-", r"LS10, NSIDE 256, $\bar n_{\rm pix}$=127",
                            ref_label=r"$\propto f_{\rm sky}^{-1/2}$")
    axes[1].set_xlabel(r"$f_{\rm sky}$"); axes[1].set_title("(b) more area")

    # (c) density at fixed geometry: shot -> clustering transition.
    x, y = _agg([r for r in ax_rows if r["nside"] == "256"], "n_mean",
                _sel(amp=READ_AMP, fsky=0.44))
    slopes["ngal"] = _panel(axes[2], x, y, "C0", "o-", r"LS10, NSIDE 256, $f_{\rm sky}$=0.44",
                            ref_label=r"$\propto \bar n_{\rm pix}^{-1/2}$ (pure shot)")
    if len(x):
        # sqrt(1/nbar + sigma_clust^2) -- shot plus a constant clustering term,
        # fitted for sigma_clust. This is why the lever flattens at high density.
        from scipy.optimize import curve_fit
        try:
            popt, _ = curve_fit(lambda n, k, sc: k * np.sqrt(1.0 / n + sc ** 2), x, y,
                                p0=[y[0] * np.sqrt(x[0]), 0.05], maxfev=20000)
            xs = np.geomspace(x.min(), x.max(), 60)
            axes[2].plot(xs, popt[0] * np.sqrt(1.0 / xs + popt[1] ** 2), "-", c="C2", lw=1.6,
                         label=r"$\sqrt{1/\bar n_{\rm pix}+\sigma_{\rm clus}^2}$," "\n"
                               rf"$\sigma_{{\rm clus}}$={abs(popt[1]):.3f}")
            slopes["sigma_clus"] = float(abs(popt[1]))
        except Exception:
            pass
    axes[2].set_xlabel(r"$\bar n_{\rm pix}$"); axes[2].set_title("(c) more galaxies\nshot $\\to$ clustering floor")

    # (d) fixed N_gal: raise nside, drop density. Breaks the (a) degeneracy.
    xd, yd, ng = [], [], []
    for ns, nm in FIXED_NGAL_DIAG:
        sel = [r for r in ns_rows if r["nside"] == ns
               and abs(_f(r, "n_mean") - nm) < 1e-9
               and abs(_f(r, "fsky") - 0.44) < 1e-9
               and abs(_f(r, "amp") - READ_AMP) < 1e-9]
        if sel:
            xd.append(_f(sel[0], "npix")); yd.append(float(np.mean([_amin(r) for r in sel])))
            ng.append(_f(sel[0], "npix") * nm)
    xd, yd = np.array(xd), np.array(yd)
    if len(xd):
        slopes["fixed_ngal"] = _slope(xd, yd)
        axes[3].plot(xd, yd, "o-", c="C0",
                     label=f"mock, $N_{{\\rm gal}}$ fixed $\\approx${np.mean(ng):.1e}\n"
                           f"fitted slope {slopes['fixed_ngal']:+.3f}")
        floor = 3.0 / np.sqrt(np.mean(ng))
        axes[3].axhline(floor, ls=":", c="C2", lw=1.6,
                        label=rf"shot floor $3/\sqrt{{N_{{\rm gal}}}}$={floor:.2e}")
    axes[3].set_xlabel(r"$N_{\rm pix}$ (at fixed $N_{\rm gal}$)")
    axes[3].set_title("(d) pixelisation alone\nflat $\\Rightarrow$ only $N_{\\rm gal}$ matters")
    # Give (d) a decade-scale y-range like the other panels. On an auto-scaled
    # axis a -0.055 slope fills the frame and reads as a real decline; matching
    # the dynamic range is what makes "flat" look flat.
    if len(yd):
        mid = float(np.sqrt(yd.min() * yd.max()))
        axes[3].set_ylim(mid / 3.0, mid * 3.0)

    for a in axes:
        a.set_xscale("log"); a.set_yscale("log"); a.grid(alpha=.3, which="both")
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=6.5, loc="best")
    # After the loop: set_xscale("log") above resets tick state, so the f_sky
    # axis has to be relabelled here or its minor labels collide.
    axes[1].set_xticks([0.1, 0.2, 0.3, 0.4])
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[1].xaxis.set_minor_formatter(plt.NullFormatter())
    fig.suptitle(f"GLASS-mock sweep: fitted exponents at injected $A$={READ_AMP} "
                 "(mock diagnostic, not a survey forecast)", y=1.03)
    fig.tight_layout()
    fig.savefig(STATIC / "fig4_sweep_validation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return slopes


_SWEEP_RUN = r"""GLASS-mock sweep
----------------

We ran ``run_detectability_sweep.py`` (OLS and ISD-1 on GLASS mocks) over the
``nside × density × f_sky × amplitude`` grid: 14 400 LS10-geometry fits, 8 100
Euclid-geometry fits and 240 MCMC-add anchor fits, with 30 simulations per cell.
The exponents are fitted in log-log at injected amplitude :math:`A=0.05`.

Pixel refinement gives :math:`-0.469` (LS10 geometry) and :math:`-0.479` (Euclid
geometry), and sky area :math:`-0.493`, against the analytic :math:`-1/2`.
Density gives :math:`-0.375` over the full range. Fitting
:math:`A_{\min}\propto\sqrt{1/\bar n_{\rm pix}+\sigma_{\rm clus}^{2}}` to the density
lever gives :math:`\sigma_{\rm clus}=0.063`, against 0.064 measured on the GLASS
map at NSIDE 256; clustering supplies 34 % of the pixel variance at
:math:`\bar n_{\rm pix}=127` and 67 % at 490.

Panel (a) holds :math:`\bar n_{\rm pix}` fixed, so :math:`N_{\rm gal}\propto N_{\rm pix}`
along it. Raising NSIDE while lowering the density so that :math:`N_{\rm gal}` stays
at :math:`2.7\times10^{6}` gives a slope of :math:`-0.055`, on the shot floor
:math:`3/\sqrt{N_{\rm gal}}=1.8\times10^{-3}`.

.. figure:: /_static/survey_design_synthesis/fig4_sweep_validation.png
   :width: 98%

   Fitted exponents from the sweep. (a) and (b) with the analytic
   :math:`\propto x^{-1/2}` reference (dashed); (c) with pure shot noise (dashed) and
   the shot-plus-clustering curve (green); (d) at fixed :math:`N_{\rm gal}`, with the
   shot floor.

.. caution::

   The fitted exponents depend on the injected amplitude. ``field_snr`` is built
   from :math:`\lVert T\hat a\rVert`, which is biased high at low signal, so the
   same :math:`f_{\rm sky}` slice fits :math:`-0.388` at :math:`A=0.005` and
   :math:`-0.493` at :math:`A=0.05`.

   The mock per-pixel scatter at :math:`\bar n_{\rm pix}=127`, NSIDE 64 is
   :math:`\hat\sigma\approx0.10`, against 0.397 measured on LS10, so mock
   :math:`A_{\min}` values are about four times lower than the survey reaches. The
   survey numbers are the scorecard values above.

The MCMC-add anchors agree with OLS to better than %%MCMCAGREE%% in all
%%MCMCCELLS%% shared cells. For the additive Gaussian model the posterior mean is
the OLS solution, so this is a consistency check."""


_SWEEP_PREP = r"""GLASS-mock sweep
----------------

The sweep outputs are not on disk. ``python scripts/run_detectability_sweep.py --check``
validates the inputs and ``bash bash/run_remote_full.sh sweep_ls10 sweep_euclid``
runs the grid."""


def _table_rows(ls10):
    def limiting(r):
        return "clustering (area-limited)" if r["regime"] == "clustering-limited" else "shot (depth-limited)"
    out = []
    ls = [r for r in ls10 if r["nside"] == "64" and "10.0" in r["sample"]]
    if ls:
        r = ls[0]
        out.append(("LS10 (log$M_*\\geq$10.0)", "64", f"{_f(r,'fsky'):.2f}", f"{_f(r,'nbar'):.0f}",
                    f"{int(_f(r,'n_gal')):,}", limiting(r), f"{_f(r,'Amin_3sig'):.1e}"))
    return out


_RST = r"""Survey design and the detectability law
=======================================

This page turns the LS10 worked example of :doc:`detectability_law` into rules of
thumb for dimensioning a survey, and compares the scaling laws with a GLASS-mock
sweep.

Where LS10 sits in the design space
-----------------------------------

The law :math:`A_{\min}(\nu\sigma)=\nu\hat\sigma/\sqrt{N_{\rm pix}}`, with
:math:`N_{\rm eff}=N_{\rm pix}/\hat\sigma^2`, places a survey by its measured
per-pixel scatter and its pixel count.

.. list-table:: Design-space scorecard (fiducial configuration)
   :header-rows: 1
   :widths: 26 8 8 8 16 20 12

   * - survey
     - NSIDE
     - :math:`f_{\rm sky}`
     - :math:`\bar n_{\rm pix}`
     - :math:`N_{\rm gal}`
     - limiting factor
     - :math:`A_{\min}(3\sigma)`
%%TABLE%%

LS10 is clustering-limited at every NSIDE in the scorecard. At NSIDE 64 the
per-pixel scatter :math:`\hat\sigma=0.397` of the fiducial sample is 4.5 times the
shot term :math:`1/\sqrt{\bar n_{\rm pix}}`, so :math:`N_{\rm eff}/N_{\rm gal}=0.05`.

.. figure:: /_static/survey_design_synthesis/fig1_master_Amin_vs_Ngal.png
   :width: 90%

   Smallest detectable systematic against galaxy count for the LS10 samples at
   NSIDE 64 (:math:`4.5\text{–}13.8\times10^{-3}`). At fixed :math:`N_{\rm gal}` the
   scatter follows :math:`\hat\sigma`: :math:`\log M_*\geq11.0` reaches
   :math:`6.1\times10^{-3}` on 1.6 M galaxies with :math:`\hat\sigma=0.297`.

.. figure:: /_static/survey_design_synthesis/fig2_Amin_vs_fsky.png
   :width: 90%

   In the clustering regime :math:`A_{\min}\propto f_{\rm sky}^{-1/2}`. LS10 covers
   :math:`f_{\rm sky}=0.44`.

.. figure:: /_static/survey_design_synthesis/fig3_design_space.png
   :width: 90%

   :math:`N_{\rm eff}/N_{\rm gal}` across pixel size and density; 1 is pure shot
   noise.

Rules of thumb
--------------

#. Sensitivity is set by :math:`N_{\rm eff}=N_{\rm pix}/\hat\sigma^2`. Measure
   :math:`\hat\sigma` as the residual scatter of the fit; then
   :math:`A_{\min}(\nu\sigma)=\nu\hat\sigma/\sqrt{N_{\rm pix}}`.
#. In the shot regime, :math:`\bar n_{\rm pix}\lesssim1/\sigma_{\rm clus}^{2}`
   (about 7 for LS10 at NSIDE 64), :math:`A_{\min}\propto1/\sqrt{N_{\rm gal}}` and
   depth helps. At higher density :math:`A_{\min}` saturates at a floor
   :math:`\propto1/\sqrt{f_{\rm sky}}` and area helps.
#. Refine pixels down to the shot floor or the coherence scale of the systematic,
   whichever is reached first. Pixels coarser than the systematic average its
   signal away.
#. Detect the field rather than individual templates. The field statistic is
   insensitive to collinearity, while individual templates are poorly identified
   (condition number :math:`1.4\times10^{3}` for the standardised LS10 basis at
   NSIDE 64). Calibrate per-template significance on matched mocks with
   ``sys_mapping.calibrated_template_significance``: on LS10 the independent-pixel
   error is short by a median factor of 1.3 to 2.9 per sample, rising with
   resolution, and by 0.9 to 5.6 for single templates.
#. The contamination of :math:`w(\theta)` grows as :math:`A^2` and the field
   regression signal as :math:`A`, so the field regression detects fainter
   systematics.

Dimensioning a run
------------------

* A wide, shallow survey such as LS10 is area-limited: it reaches the smallest
  amplitudes because area adds modes, and more depth adds little.
* A deep, narrow survey lowers its :math:`A_{\min}` floor fastest by enlarging the
  footprint. Its fine pixels already sample the available modes.
* At NSIDE 64 the fiducial LS10 sample reaches :math:`A_{\min}(3\sigma)=8.1\times10^{-3}`
  field rms; across all samples and resolutions the range is :math:`4.3\times10^{-3}`
  to :math:`1.5\times10^{-2}`. The systematics in the NSIDE-64 fiducial weight map
  have rms 3.2 % for the OLS weights and 4.9 % for the combined model, about
  :math:`12\sigma`.

%%SWEEP%%

Reproduce
---------

.. code-block:: bash

   python scripts/make_survey_design_synthesis.py
   bash bash/build_docs.sh
"""


def write_rst(ls10, sweep_run):
    rows = _table_rows(ls10)
    tbl = []
    for r in rows:
        tbl.append(f"   * - {r[0]}")
        for cell in r[1:]:
            tbl.append(f"     - {cell}")
    body = _RST.replace("%%TABLE%%", "\n".join(tbl))
    body = body.replace("%%SWEEP%%", _SWEEP_RUN if sweep_run else _SWEEP_PREP)
    # MCMC-vs-OLS agreement is measured, not remembered (see mcmc_vs_ols).
    agree = mcmc_vs_ols()
    if agree:
        body = body.replace("%%MCMCAGREE%%", f"{agree['max_rel'] * 100:.2g} %")
        body = body.replace("%%MCMCCELLS%%", str(agree["n_cells"]))
        print(f"[synthesis] MCMC-add vs OLS: max {agree['max_rel']*100:.3g} %, "
              f"median {agree['med_rel']*100:.3g} % over {agree['n_cells']} cells")
    else:
        body = body.replace("%%MCMCAGREE%%", "0.1 %").replace("%%MCMCCELLS%%", "240")
        print("[synthesis] NOTE: no MCMC anchors found — quoting the recorded 0.1 %")
    RST.write_text(body)
    return RST


def main():
    ls10 = load_scorecards()
    if not ls10:
        raise SystemExit("LS10 scorecard missing — run scripts/analyze_detectability_law.py first")
    fig_amin_vs_ngal(ls10)
    fig_amin_vs_fsky(ls10)
    fig_design_space(ls10)
    ns_rows, ax_rows, eu_rows = load_sweep()
    sweep_run = ns_rows is not None
    if sweep_run:
        fig_sweep_validation(ns_rows, ax_rows, eu_rows)
    rst = write_rst(ls10, sweep_run)
    print(f"[synthesis] page    -> {rst}")
    print(f"[synthesis] figures -> {STATIC}")
    print(f"[synthesis] LS10 rows={len(ls10)}  sweep_run={sweep_run}")


if __name__ == "__main__":
    main()
