"""Pedagogical figure: correction budget measured on Uchuu sub-areas.

For the M_min = 1e11, 1e12, 1e13 samples, every quantity is measured on
sub-areas (HEALPix discs) of the full-sky Uchuu lightcone at |b| > 25 deg (template: SFD E(B-V) or Gaia stars),
fitted with a power law in area and extrapolated to large surveys:

    sigma_eps(A) = C_eps (A / 100 deg2)^-alpha   scatter of the fitted amplitude
    sigma_b(A)   = C_b   (A / 100 deg2)^-beta    scatter of the recovered bias
    eps_tol(A)   = sigma_m,pix sqrt((b + sigma_b)^2 - b^2)
    safe window  |k - 1| < eps_tol / eps,  reliable while sigma_eps < eps_tol

The physical form sigma_eps^2 = f^2 (1/Nbar + e^{sigma^2} - 1) A_pix / A is
also fitted, with one factor f for the pixel-to-pixel correlations.

Run:  python scripts/plot_pedagogy_correction_budget.py [--eps 0.15] [--ndisc 200]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_pedagogy_bias_broadening import CACHE, LOGM, ROOT, footprint, measure, theory  # noqa: E402
from plot_pedagogy_decontamination import (  # noqa: E402
    B_CUT, TEMPLATES, contaminate, full_sky_template, correct, fit_eps, sigma_pln, sigma_to_bias)


OUT = ROOT / "docs" / "_static" / "pedagogy_correction_budget"
ROWS = [0, 2, 4]
COLS = ["#2a78d6", "#1baf7a", "#e87ba4"]
INK, MUTED = "#1f1f1f", "#6b6b6b"
NSIDE = 64
AREAS = np.array([25, 50, 100, 200, 400, 800, 1600, 3200, 6400])
A_EXTRA = [5000, 14000, 20000]
K_TEST = [0.5, 1.0, 1.5]


def power_fit(A, y, w):
    """Weighted fit of log y = log C - alpha log(A/100); returns (C, alpha, cov)."""
    x = np.log(A / 100)
    p, cov = np.polyfit(x, np.log(y), 1, w=w, cov="unscaled")
    return np.exp(p[1]), -p[0], cov


def power_eval(A, C, alpha, cov):
    x = np.log(np.asarray(A, float) / 100)
    y = np.log(C) - alpha * x
    J = np.stack([x, np.ones_like(x)], axis=-1)
    err = np.sqrt(np.einsum("...i,ij,...j->...", J, cov, J))
    return np.exp(y), np.exp(y - err), np.exp(y + err)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--template", default="SFD", choices=list(TEMPLATES))
    ap.add_argument("--ndisc", type=int, default=200)
    a = ap.parse_args()

    m = dict(np.load(CACHE)) if CACHE.exists() else measure(NSIDE)
    pix_oct = footprint(NSIDE)
    tmap, keep_full = full_sky_template(a.template, NSIDE)
    keep = keep_full[pix_oct]
    pix = pix_oct[keep]
    t = tmap[pix]
    t = (t - t.mean()) / t.std()  # once, globally
    area_pix = hp.nside2pixarea(NSIDE, degrees=True)
    A0 = len(pix) * area_pix
    lookup = np.full(hp.nside2npix(NSIDE), -1)
    lookup[pix] = np.arange(len(pix))

    _, sig2_m, _ = theory(np.array([12.0]), area_pix)
    sig_m = np.sqrt(sig2_m)

    # Sub-areas: discs of area A centred on random footprint pixels, >= 90% inside.
    rng = np.random.default_rng(2)
    discs = {}
    for A in AREAS:
        rad = np.arccos(1 - np.radians(1) ** 2 * A / (2 * np.pi))
        sel, tries = [], 0
        while len(sel) < a.ndisc and tries < 50 * a.ndisc:
            tries += 1
            c = pix[rng.integers(len(pix))]
            dp = hp.query_disc(NSIDE, hp.pix2vec(NSIDE, c), rad, inclusive=False)
            idx = lookup[dp]
            if (idx >= 0).mean() >= 0.9:
                sel.append(idx[idx >= 0])
        discs[A] = sel
        print(f"A = {A:4d} deg2: {len(sel)} discs")
    discs[A0] = [np.arange(len(pix))]
    all_A = np.array(list(discs))

    res = []
    for i in ROWS:
        n = m["counts"][i][keep].astype(float)
        nb_all, s_all = n.mean(), sigma_pln(n)
        out = dict(nb=nb_all, nd=nb_all / area_pix, s=s_all, b=sigma_to_bias(s_all, sig2_m),
                   A=all_A, s_eps=[], s_b=[], ndisc=[], frac={k: [] for k in K_TEST})
        for A in all_A:
            e0, bt, brec = [], [], {k: [] for k in K_TEST}
            for idx in discs[A]:
                # Each disc is a survey of its own: its template is normalised to zero
                # mean and unit variance over the disc (Eq. template-norm of the pipeline),
                # so eps is the amplitude per template standard deviation in that survey.
                nn, tt = n[idx], t[idx]
                tc = (tt - tt.mean()) / tt.std()
                e0.append(fit_eps(nn, tc))
                bt.append(sigma_to_bias(sigma_pln(nn), sig2_m))
                nc = contaminate(nn, tc, a.eps, "multiplicative")
                eh = fit_eps(nc, tc)
                for k in K_TEST:
                    brec[k].append(sigma_to_bias(sigma_pln(correct(nc, tc, eh, k,
                                                                    "multiplicative")), sig2_m))
            out["s_eps"].append(np.std(e0, ddof=1) if len(e0) > 1 else np.nan)
            out["s_b"].append(np.std(bt, ddof=1) if len(bt) > 1 else np.nan)
            out["ndisc"].append(len(e0))
            out["_bt"] = np.array(bt)
            for k in K_TEST:
                out["frac"][k].append(np.array(brec[k]) - np.array(bt))
        for key in ("s_eps", "s_b", "ndisc"):
            out[key] = np.array(out[key], float)
        res.append(out)

    # Fits on A <= A0/10: larger discs miss the variance common to the whole footprint.
    fitm = all_A <= A0 / 10
    n_ind = np.minimum(np.array([len(discs[A]) for A in all_A]), A0 / all_A)
    for r in res:
        w = np.sqrt(2 * (n_ind[fitm] - 1))  # 1 / relative error of a std
        r["fe"] = power_fit(all_A[fitm], r["s_eps"][fitm], w)
        r["fb"] = power_fit(all_A[fitm], r["s_b"][fitm], w)
        var1 = 1 / r["nb"] + np.expm1(r["s"] ** 2)
        r["var1"] = var1
        r["f2"] = r["s_eps"][fitm] ** 2 * all_A[fitm] / (area_pix * var1)
    # f and g drift with area (large-scale structure in template and field): fit
    # f(A) = f100 (A/100)^gamma_f jointly over the samples, likewise g(A).
    A_fit = np.tile(all_A[fitm], len(res))
    gam_f, lf100 = np.polyfit(np.log(A_fit / 100),
                              0.5 * np.log(np.concatenate([r["f2"] for r in res])), 1)
    f100 = float(np.exp(lf100))

    def f_of(A):
        return f100 * (np.asarray(A, float) / 100) ** gam_f

    f = f100
    # Physical form of the bias error: error on the clustering variance v, propagated to b.
    #   sigma_b / b = g * (1/2) sqrt(2) (1/Nbar + v) / v * sqrt(A_pix / A)
    g2 = []
    for r in res:
        v = np.expm1(r["s"] ** 2)
        r["v"] = v
        shape = 0.5 * np.sqrt(2) * (1 / r["nb"] + v) / v * np.sqrt(area_pix / all_A[fitm])
        g2.append((r["s_b"][fitm] / r["b"] / shape) ** 2)
    gam_g, lg100 = np.polyfit(np.log(A_fit / 100), 0.5 * np.log(np.concatenate(g2)), 1)
    g100 = float(np.exp(lg100))

    def g_of(A):
        return g100 * (np.asarray(A, float) / 100) ** gam_g

    g = g100
    print(f"f(A) = {f100:.2f} (A/100)^{gam_f:.3f}, g(A) = {g100:.2f} (A/100)^{gam_g:.3f}")
    # Offset of the recovered bias in units of sigma_b(A), mean and scatter over sub-areas.
    for r in res:
        sb_A = power_eval(all_A, *r["fb"])[0]
        # Median and half the 16-84% range: a few small discs blow up where 1 + k eps t ~ 0.
        r["off"] = {k: (np.array([np.median(d) for d in r["frac"][k]]) / sb_A,
                        np.array([np.diff(np.percentile(d, [16, 84]))[0] / 2 for d in r["frac"][k]])
                        / sb_A) for k in K_TEST}
    rule = dict(f=f, g=g, f100=f100, gamma_f=float(gam_f), g100=g100, gamma_g=float(gam_g),
                fit_area_max_deg2=float(all_A[fitm].max()), footprint_deg2=float(A0), nside=NSIDE, area_pix_deg2=area_pix, sigma_m_pix_uchuu=float(sig_m),
                template=a.template, b_cut_deg=B_CUT,
                note="sigma_eps = f(A) sqrt((1/Nbar + v) A_pix/A); "
                     "sigma_b/b = g(A) (1/2) sqrt(2) (1/Nbar + v)/v sqrt(A_pix/A); "
                     "f(A) = f100 (A/100 deg2)^gamma_f, g(A) = g100 (A/100 deg2)^gamma_g; "
                     "v = exp(sigma^2) - 1 = b^2 sigma_m,pix^2; Nbar per A_pix pixel; "
                     "eps per template standard deviation over the survey footprint",
                uchuu_samples=[dict(log_mmin=float(LOGM[i]), n_deg2=float(r["nd"]), b=float(r["b"]))
                               for r, i in zip(res, ROWS)])
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "pedagogy_correction_budget_rule.json").write_text(json.dumps(rule, indent=2))

    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10})
    fig = plt.figure(figsize=(21, 7.4), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[0.22, 1])
    top.suptitle("Correction budget measured on sub-areas of the full-sky Uchuu lightcone "
                 f"({TEMPLATES[a.template]}, |b| > {B_CUT:.0f}$^\\circ$, {A0:.0f} deg$^2$) and extrapolated to large surveys",
                 fontsize=11, x=0.01, ha="left")
    alphas = ", ".join(f"{r['fe'][1]:.2f}" for r in res)
    betas = ", ".join(f"{r['fb'][1]:.2f}" for r in res)
    eqs = [
        "Fit precision of the systematic amplitude\n"
        r"$\sigma_\epsilon = C_\epsilon\,(A/100\,{\rm deg}^2)^{-\alpha}$, "
        rf"$\alpha$ = {alphas}" "\n"
        r"$\sigma_\epsilon^2 = f^2(1/\bar N + e^{\sigma^2}-1)\,A_{\rm pix}/A$, "
        rf"$f = {f100:.2f}\,(A/100)^{{{gam_f:.2f}}}$",
        "Bias error from the sub-area scatter\n"
        r"$\sigma_b = C_b\,(A/100\,{\rm deg}^2)^{-\beta}$, "
        rf"$\beta$ = {betas}" "\n"
        r"$b = \sqrt{e^{\sigma^2}-1}\,/\,\sigma_{m,\rm pix}$",
        "Safe correction window\n"
        r"$\epsilon_{\rm tol} = \sigma_{m,\rm pix}\sqrt{(b+\sigma_b)^2-b^2}$" "\n"
        r"$|k-1| < \epsilon_{\rm tol}/\epsilon$, reliable while $\sigma_\epsilon < \epsilon_{\rm tol}$" "\n"
        r"$t$ normalised over each (sub-)survey",
        "Mis-correction offset, rule of thumb\n"
        r"$b'^2 = b^2 + (1-k)^2\epsilon^2/\sigma^2_{m,\rm pix}$, offset $(b'-b)/\sigma_b$" "\n"
        r"$\sigma_b/b = g\,\frac{\sqrt{2}}{2}\,\frac{1/\bar N + v}{v}\sqrt{A_{\rm pix}/A}$, "
        rf"$v=b^2\sigma^2_{{m,\rm pix}}$, $g = {g100:.2f}\,(A/100)^{{{gam_g:.2f}}}$",
    ]
    for j, eq in enumerate(eqs):
        top.text(0.01 + j * 0.25, 0.70, eq, ha="left", va="top", fontsize=10.5,
                 linespacing=1.6, color=INK)
    ax = bot.subplots(1, 4)
    A_line = np.logspace(np.log10(20), np.log10(30000), 120)
    inside = A_line <= A0

    def label(r, i):
        return (rf"$10^{{{LOGM[i]:.0f}}}$: {r['nd']:.0f} deg$^{{-2}}$, b={r['b']:.2f}, "
                rf"$N_{{\rm tot}}$={r['nd'] * A0:.1e} / {r['nd'] * 14000:.1e}")

    def law(axp, C, alpha, cov, c, scale=1.0):
        y, lo, hi = power_eval(A_line, C, alpha, cov)
        axp.plot(A_line[inside], y[inside] / scale, color=c, lw=2)
        axp.plot(A_line[~inside], y[~inside] / scale, color=c, lw=2, ls="--")
        axp.fill_between(A_line, lo / scale, hi / scale, color=c, alpha=0.15, lw=0)

    def points(axp, y, c, err_rel):
        for A, yy, fm, e in zip(all_A, y, fitm, err_rel):
            axp.errorbar(A, yy, yerr=yy * e, fmt="o", ms=7, color=c,
                         mfc=c if fm else "white", mec=c if not fm else "white", mew=1.2)

    err_rel = 1 / np.sqrt(2 * np.maximum(n_ind - 1, 1))
    for r, i, c in zip(res, ROWS, COLS):
        # Panel 1
        points(ax[0], r["s_eps"], c, err_rel)
        law(ax[0], *r["fe"], c)
        ax[0].plot(A_line, f_of(A_line) * np.sqrt(r["var1"] * area_pix / A_line), color=c, lw=1,
                   ls=":")
        ax[0].plot([], [], color=c, lw=2, label=label(r, i))
        # Panel 2
        points(ax[1], r["s_b"] / r["b"], c, err_rel)
        law(ax[1], *r["fb"], c, scale=r["b"])
        # Panel 3
        se = power_eval(A_line, *r["fe"])[0]
        sb = power_eval(A_line, *r["fb"])[0]
        tol = sig_m * np.sqrt((r["b"] + sb) ** 2 - r["b"] ** 2)
        win = tol / a.eps
        ax[2].plot(A_line, win, color=c, lw=2, ls="--")
        ax[2].plot(A_line, se / a.eps, color=c, lw=2)
        ok = se < win * a.eps
        ax[2].fill_between(A_line, se / a.eps, win, where=ok, color=c, alpha=0.12, lw=0)
        r["A_min"] = A_line[np.argmax(ok)] if ok.any() else np.nan
        r["tol"], r["se"], r["sb"] = tol, se, sb
        # Panel 4: offset of b for k = 0.5 and 1.5, measured and predicted
        ax[1].plot(A_line, g_of(A_line) * 0.5 * np.sqrt(2) * (1 / r["nb"] + r["v"]) / r["v"]
                   * np.sqrt(area_pix / A_line), color=c, lw=1, ls=":")
        for k, mk, dx in [(0.5, "o", 0.94), (1.5, "s", 1.06)]:
            mu, sd = r["off"][k]
            ax[3].errorbar(all_A[:-1] * dx, mu[:-1], yerr=sd[:-1], fmt=mk, ms=6, color=c,
                           mfc=c if k == 0.5 else "white", mec=c, mew=1.2, elinewidth=1)
        bprime = np.sqrt(r["b"] ** 2 + (0.5 * a.eps) ** 2 / sig2_m)
        pred = (bprime - r["b"]) / sb
        ax[3].plot(A_line[inside], pred[inside], color=c, lw=2)
        ax[3].plot(A_line[~inside], pred[~inside], color=c, lw=2, ls="--")
    ax[0].plot([], [], color=MUTED, lw=2, label="power-law fit (dashed: extrapolated)")
    ax[0].plot([], [], color=MUTED, lw=1, ls=":", label=r"physical form with $f(A)$")
    ax[0].plot([], [], "o", color=MUTED, mfc="white", label="not in fit (overlapping discs)")
    ax[2].plot([], [], color=MUTED, lw=2, label=r"fit noise $\sigma_\epsilon/\epsilon$")
    ax[2].plot([], [], color=MUTED, lw=2, ls="--", label=r"tolerance $\epsilon_{\rm tol}/\epsilon$")
    ax[3].plot([], [], "o", color=MUTED, label="k = 0.5, sub-areas (median, 16-84%)")
    ax[3].plot([], [], "s", color=MUTED, mfc="white", label="k = 1.5, sub-areas")
    ax[3].plot([], [], color=MUTED, lw=2, label=r"prediction, $|1-k| = 0.5$")
    ax[2].axhline(0.5, color=INK, lw=1, ls="-.")
    ax[2].text(22, 0.52, "k = 0.5 or 1.5 (panel 4)", fontsize=8, color=MUTED, va="bottom")
    ax[3].axhline(1, color=INK, lw=1)
    ax[3].text(22, 1.05, r"offset = 1 $\sigma_b$: mis-correction detectable", fontsize=8,
               color=MUTED, va="bottom")

    titles = [r"1. Fit precision $\sigma_\epsilon$ of the amplitude",
              r"2. Relative bias error $\sigma_b/b$",
              r"3. Safe window $|k-1|$ (shaded: correctable)",
              r"4. Bias offset of a 50% mis-correction ($k$ = 1 gives 0)"]
    ylabels = [r"$\sigma_\epsilon$ (per 1 std of the template)", r"$\sigma_b / b$",
               r"$|k - 1|$", r"$(b_{\rm rec} - b)\,/\,\sigma_b(A)$"]
    for axp, ti, yl in zip(ax, titles, ylabels):
        axp.set_xscale("log")
        axp.set_xlabel(r"survey area $A$ [deg$^2$]")
        axp.set_ylabel(yl)
        axp.set_title(ti, loc="left", fontsize=10)
        axp.axvspan(A0, A_line[-1], color=MUTED, alpha=0.07, lw=0)
        axp.axvline(A0, color=INK, lw=0.8)
        axp.grid(alpha=0.25, lw=0.5, which="both")
        axp.set_xlim(A_line[0], A_line[-1])
    for axp in ax[:3]:
        axp.set_yscale("log")
    ax[0].text(A0 * 1.1, 0.95, "extrapolated", transform=ax[0].get_xaxis_transform(),
               fontsize=8, color=MUTED, va="top")
    ax[3].set_ylim(-0.4, 6)
    ax[3].axhline(0, color=MUTED, lw=0.8)
    ax[0].legend(fontsize=7, frameon=False, loc="lower left",
                 title=r"$M_{\min}$: density, b, $N_{\rm tot}$ at footprint / 14000 deg$^2$",
                 title_fontsize=7)
    ax[2].legend(fontsize=8, frameon=False, loc="lower left")
    ax[3].legend(fontsize=8, frameon=False, loc="upper left")
    ax[1].plot([], [], color=MUTED, lw=1, ls=":", label=r"physical form with $g(A)$")
    ax[1].legend(fontsize=8, frameon=False, loc="lower left")

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=130)

    print(f"\nf = {f:.2f}, g = {g:.2f}, sigma_m,pix = {sig_m:.3f}, eps = {a.eps}")
    for r, i in zip(res, ROWS):
        print(f"\nlog M_min = {LOGM[i]:.0f}: n = {r['nd']:.1f}/deg2, b = {r['b']:.2f}, "
              f"alpha = {r['fe'][1]:.2f}, beta = {r['fb'][1]:.2f}, "
              f"A_min(safe) = {r['A_min']:.0f} deg2")
        print(f"{'A':>7} {'sig_eps':>8} {'sig_b':>7} {'|k-1|max':>9} {'fit noise/eps':>13}")
        for A in [A0, *A_EXTRA]:
            se = power_eval(A, *r["fe"])[0]
            sb = power_eval(A, *r["fb"])[0]
            tol = sig_m * np.sqrt((r["b"] + sb) ** 2 - r["b"] ** 2)
            print(f"{A:7.0f} {float(se):8.4f} {float(sb):7.4f} {float(tol) / a.eps:9.3f} "
                  f"{float(se) / a.eps:13.3f}")
        print("offset/sigma_b k=0.5, 1.5 per A:",
              {int(A): (round(float(r['off'][0.5][0][j]), 2), round(float(r['off'][1.5][0][j]), 2))
               for j, A in enumerate(all_A)})
    print(f"saved {OUT}.png")


if __name__ == "__main__":
    main()
