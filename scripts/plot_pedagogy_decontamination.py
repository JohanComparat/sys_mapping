"""Pedagogical figure: inject a systematic, decontaminate, over- and under-correct.

Takes the Uchuu pixel counts of figure 1 (``plot_pedagogy_bias_broadening.py``)
for M_min = 1e11, 1e12, 1e13 M_sun/h and contaminates them with a real survey
template t (zero mean, unit std over the footprint):

    additive        N_c = N + eps * Nbar * t
    multiplicative  N_c = N * (1 + eps * t)

The amplitude eps_hat is fitted by least squares of N_c/Nbar - 1 against t, and
the correction is applied with strength k (k = 1 nominal, 0.5 under, 1.5 over).

Run:  python scripts/plot_pedagogy_decontamination.py [--eps 0.15] [--template SFD|GAIA]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))
import healpy as hp  # noqa: E402
from plot_pedagogy_bias_broadening import (  # noqa: E402
    CACHE, LOGM, ROOT, footprint, measure, theory)


OUT = ROOT / "docs" / "_static" / "pedagogy_decontamination"
SFD_DIR = Path("~/data/dust_maps/sfd").expanduser()
GAIA_MAP = Path("~/data/gaia_cat/GAIA_stellar_density_NSIDE_{nside:05d}.fits").expanduser()
TEMPLATES = {"SFD": "SFD E(B-V)", "GAIA": "Gaia stellar density"}
B_CUT = 25.0


def full_sky_template(name: str, nside: int, b_cut: float = B_CUT):
    """Full-sky template in RING ordering and the mask |b| > b_cut.

    ``SFD`` is the E(B-V) map of Schlegel, Finkbeiner & Davis (1998) read at the
    pixel centres with ``dustmaps``; ``GAIA`` is the Gaia stellar density map
    (NESTED, equatorial), reordered to RING.
    """
    import warnings

    from astropy.coordinates import SkyCoord
    import astropy.units as u

    npix = hp.nside2npix(nside)
    ra, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    if name == "SFD":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from dustmaps.sfd import SFDQuery
            tmap = np.asarray(SFDQuery(map_dir=str(SFD_DIR))(c), float)
    elif name == "GAIA":
        from astropy.io import fits
        d = fits.getdata(str(GAIA_MAP).format(nside=nside))
        nest = np.asarray(d["stellar_density"], float)[np.argsort(d["IDX"])]
        tmap = hp.reorder(nest, n2r=True)
    else:
        raise ValueError(name)
    keep = (np.abs(c.galactic.b.deg) > b_cut) & np.isfinite(tmap)
    return tmap, keep
ROWS = [0, 2, 4]  # log M_min = 11, 12, 13
K_STATES = {"under-corrected, k = 0.5": 0.5, "corrected, k = 1": 1.0,
            "over-corrected, k = 1.5": 1.5}
COL = {"true": "#9a9a9a", "contaminated": "#eb6834", "corrected, k = 1": "#2a78d6",
       "under-corrected, k = 0.5": "#1baf7a", "over-corrected, k = 1.5": "#e87ba4"}
INK, MUTED = "#1f1f1f", "#6b6b6b"


def sigma_pln(n: np.ndarray) -> float:
    """Log-normal width of the density field with Poisson shot noise removed."""
    nb, var = n.mean(), n.var()
    return float(np.sqrt(np.log(1 + max(var - nb, 1e-12) / nb ** 2)))


def contaminate(n, t, eps, mode):
    nb = n.mean()
    return n + eps * nb * t if mode == "additive" else n * (1 + eps * t)


def correct(nc, t, eps_hat, k, mode):
    nb = nc.mean()
    return nc - k * eps_hat * nb * t if mode == "additive" else nc / (1 + k * eps_hat * t)


def fit_eps(nc, t):
    y = nc / nc.mean() - 1
    return float(np.dot(y, t) / np.dot(t, t))


def sigma_to_bias(sig, sig2_m):
    """Invert sigma^2 = ln(1 + b^2 sigma_m^2)."""
    return np.sqrt(np.expm1(np.asarray(sig) ** 2) / sig2_m)


def jackknife(stat, n_reg, reg):
    """Delete-one-region jackknife mean and error of stat(mask)."""
    vals = np.array([stat(reg != j) for j in range(n_reg)])
    return np.sqrt((n_reg - 1) / n_reg * ((vals - vals.mean(axis=0)) ** 2).sum(axis=0))


EQS = [
    "Contamination ($t$: template, mean 0, std 1)\n"
    "additive:  $N_c = N + \\epsilon\\,\\bar N\\,t$\n"
    "multiplicative:  $N_c = N\\,(1+\\epsilon\\,t)$",
    "Fit and correction with strength $k$\n"
    "$\\hat\\epsilon = \\sum (N_c/\\bar N-1)\\,t \\,/\\, \\sum t^2$\n"
    "add.: $N_k = N_c - k\\hat\\epsilon\\bar N t$;   mult.: $N_k = N_c/(1+k\\hat\\epsilon t)$\n"
    "$k$ = fraction of the fitted systematic removed: 0 none, 0.5 half (under),\n"
    "1 all (nominal), 1.5 50% too much (over: an inverted copy is imprinted)",
    "Clustering width, shot noise removed\n"
    "(Poisson-log-normal moments)\n"
    "$\\sigma^2 = \\ln\\left[1 + (\\mathrm{Var}\\,N_k - \\bar N)/\\bar N^2\\right]$",
    "Width to bias to halo mass\n"
    "$\\sigma^2 = \\ln(1 + b^2\\sigma^2_{m,\\rm pix})$  $\\Rightarrow$  "
    "$b = \\sqrt{e^{\\sigma^2}-1}\\,/\\,\\sigma_{m,\\rm pix}$\n"
    "$b(M_{\\min})$: Comparat17 averaged over $n(>M_{\\min})$",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--template", default="SFD", choices=list(TEMPLATES))
    a = ap.parse_args()

    nside = 64
    m = dict(np.load(CACHE)) if CACHE.exists() else measure(nside)
    pix_oct = footprint(nside)
    tmap, keep_full = full_sky_template(a.template, nside)
    keep = keep_full[pix_oct]
    t = tmap[pix_oct][keep]
    t = (t - t.mean()) / t.std()
    print(f"{keep.sum()} of {len(pix_oct)} full-sky pixels at |b| > {B_CUT:.0f} deg")

    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10})
    fig = plt.figure(figsize=(20, 13.5), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[0.14, 1])
    for j, eq in enumerate(EQS):
        top.text(0.01 + j * 0.25, 0.80, eq, ha="left", va="top", fontsize=10.5, color=INK,
                 linespacing=1.6)
    ax = bot.subplots(3, 4)

    # Jackknife regions: NSIDE=4 parents of the footprint pixels, small ones merged away.
    th, ph = hp.pix2ang(nside, pix_oct[keep])
    parent = hp.ang2pix(4, th, ph)
    uniq, inv, cnt = np.unique(parent, return_inverse=True, return_counts=True)
    big = cnt >= 0.3 * cnt.max()
    reg_ids = np.cumsum(big) - 1
    reg = np.where(big[inv], reg_ids[inv], -1)
    n_reg = int(big.sum())
    print(f"{n_reg} jackknife regions")

    # sigma_m,pix and the b(M_min) relation from figure 1's theory.
    area_pix = hp.nside2pixarea(nside, degrees=True)
    lm_grid = np.linspace(10.5, 14.2, 75)
    b_grid, sig2_m, _ = theory(lm_grid, area_pix)
    b_th = np.interp(LOGM, lm_grid, b_grid)

    def sig_k(n_true, mode, k, mask):
        """Contaminate, refit eps on the mask, correct with strength k."""
        tt = t[mask]
        nc = contaminate(n_true[mask], tt, a.eps, mode)
        return sigma_pln(correct(nc, tt, fit_eps(nc, tt), k, mode))
    ks = np.linspace(-0.05, 2.0, 42)
    print(f"{'M_min':>6} {'mode':>14} {'s_true':>7} {'s_cont':>7} {'s_k.5':>7} "
          f"{'s_k1':>7} {'s_k1.5':>7} {'eps_hat':>8}")

    for r, i in enumerate(ROWS):
        n = m["counts"][i][keep].astype(float)
        nb = n.mean()
        s_true = sigma_pln(n)
        step = max(1, int(np.ceil(0.1 * nb)))
        kedges = np.arange(-0.5, 3 * nb + step, step)
        e = kedges / nb

        def hist(x):
            h, _ = np.histogram(x, bins=kedges)
            return h / (len(x) * np.diff(e))

        for c, mode in enumerate(["additive", "multiplicative"]):
            nc = contaminate(n, t, a.eps, mode)
            eh = fit_eps(nc, t)
            states = {"contaminated": nc}
            states.update({lab: correct(nc, t, eh, k, mode) for lab, k in K_STATES.items()})
            axp = ax[r, c]
            axp.stairs(hist(n), e, fill=True, color=COL["true"], alpha=0.45,
                       label=f"true ($\\sigma$={s_true:.3f})")
            for lab, x in states.items():
                axp.stairs(hist(x), e, color=COL[lab], lw=2.2 if lab == "corrected, k = 1" else 1.6,
                           ls="--" if lab == "corrected, k = 1" else "-",
                           label=f"{lab} ($\\sigma$={sigma_pln(x):.3f})")
            axp.set_xlim(0, 3)
            axp.set_title(f"{mode}, $M_{{\\min}}=10^{{{LOGM[i]:.0f}}}$: "
                          f"$\\epsilon$={a.eps:.2f}, fitted $\\hat\\epsilon$={eh:.3f}",
                          loc="left", fontsize=10)
            axp.set_ylabel("PDF (integral = 1)")
            axp.legend(fontsize=7.5, frameon=False, loc="upper right")
            axp.grid(alpha=0.25, lw=0.5)
            if r == 2:
                axp.set_xlabel(r"$N_{\rm pix} / \bar N$")

            sk = [sigma_pln(correct(nc, t, eh, k, mode)) for k in ks]
            ax[r, 2].plot(ks, sk, color=INK, lw=2, ls="-" if mode == "multiplicative" else "--",
                          label=mode)
            if mode == "multiplicative":
                for lab, k in [("contaminated", 0.0), *K_STATES.items()]:
                    ax[r, 2].plot(k, sigma_pln(correct(nc, t, eh, k, mode)), "o", ms=9,
                                  color=COL[lab], mec="white", mew=1.5, zorder=3)
                excess = sigma_pln(nc) / s_true - 1
            print(f"{LOGM[i]:6.1f} {mode:>14} {s_true:7.3f} {sigma_pln(nc):7.3f} "
                  + " ".join(f"{sigma_pln(correct(nc, t, eh, k, mode)):7.3f}" for k in (0.5, 1, 1.5))
                  + f" {eh:8.3f}")

        a3 = ax[r, 2]
        a3.axhline(s_true, color=COL["true"], lw=1.5, zorder=0)
        a3.text(ks[-1], s_true, "true $\\sigma$", ha="right", va="bottom", fontsize=8, color=MUTED)
        a3.set_title(f"$\\sigma$ vs correction strength: +"
                     f"{100 * excess:.0f}% if uncorrected", loc="left", fontsize=10)
        a3.set_ylabel(r"clustering width $\sigma$ (shot noise removed)")
        a3.legend(fontsize=8, frameon=False, loc="upper center")
        a3.grid(alpha=0.25, lw=0.5)
        a3.set_xlabel(r"correction strength $k$ (correction uses $k\,\hat\epsilon$)")
        a3.set_xticks([0, 0.5, 1, 1.5, 2])
        a3.set_xticklabels(["0\nnone", "0.5\nunder", "1\nnominal", "1.5\nover", "2"])
        a3.axvspan(-0.05, 1, color=COL["under-corrected, k = 0.5"], alpha=0.06, lw=0)
        a3.axvspan(1, 2.0, color=COL["over-corrected, k = 1.5"], alpha=0.06, lw=0)
        a3.set_xlim(-0.05, 2.0)

        # Column 4: recovered bias with jackknife errors, against the truth.
        a4 = ax[r, 3]
        ok_all = np.ones(len(n), bool)
        b_true = sigma_to_bias(s_true, sig2_m)
        eb_true = jackknife(lambda msk: sigma_to_bias(sigma_pln(n[msk & (reg >= 0)]), sig2_m),
                            n_reg, reg)
        a4.axhspan(b_true - eb_true, b_true + eb_true, color=COL["true"], alpha=0.3, lw=0)
        a4.axhline(b_true, color=COL["true"], lw=1.5)
        a4.axhline(b_th[i], color=INK, lw=1, ls=":")
        a4.text(1.95, b_true + eb_true, f"true counts: b = {b_true:.2f} $\\pm$ {eb_true:.2f}",
                ha="right", va="bottom", fontsize=8, color=MUTED)
        a4.text(1.95, min(b_th[i], b_true - eb_true), f"Comparat17, $M_{{\\min}}=10^{{{LOGM[i]:.0f}}}$: b = {b_th[i]:.2f}",
                ha="right", va="top", fontsize=8, color=MUTED)
        for mode, ls, off in [("additive", "--", -0.04), ("multiplicative", "-", 0.04)]:
            kk = np.linspace(-0.05, 2.0, 42)
            bk = sigma_to_bias([sig_k(n, mode, k, ok_all) for k in kk], sig2_m)
            ebk = jackknife(lambda msk: sigma_to_bias(
                [sig_k(n, mode, k, msk & (reg >= 0)) for k in kk], sig2_m), n_reg, reg)
            a4.plot(kk, bk, color=INK, lw=2, ls=ls, label=mode)
            if mode == "multiplicative":
                a4.fill_between(kk, bk - ebk, bk + ebk, color=INK, alpha=0.12, lw=0,
                                label="jackknife 1$\\sigma$")
            for lab, k in [("contaminated", 0.0), *K_STATES.items()]:
                j = np.argmin(abs(kk - k))
                a4.errorbar(k + off, bk[j], yerr=ebk[j], fmt="o" if ls == "-" else "s", ms=7,
                            color=COL[lab], mec="white", mew=1.2, elinewidth=1.5, capsize=0,
                            zorder=3)
            j0, j1 = np.argmin(abs(kk)), np.argmin(abs(kk - 1))
            print(f"   {mode:>14}: b_true={b_true:.3f}+-{eb_true:.3f}  b(k=0)={bk[j0]:.3f}+-{ebk[j0]:.3f}"
                  f"  b(k=1)={bk[j1]:.3f}+-{ebk[j1]:.3f}  Comparat17={b_th[i]:.3f}")
        a4.set_xlim(-0.05, 2.0)
        a4.margins(y=0.08)
        a4.set_xticks([0, 0.5, 1, 1.5, 2])
        a4.set_xticklabels(["0\nnone", "0.5\nunder", "1\nnominal", "1.5\nover", "2"])
        a4.set_xlabel("correction strength $k$")
        a4.set_ylabel(r"recovered bias $b$")
        a4.set_title(f"bias from $\\sigma$: {b_true:.2f} true, "
                     f"{sigma_to_bias(sigma_pln(contaminate(n, t, a.eps, 'multiplicative')), sig2_m):.2f}"
                     " uncorrected (mult.)", loc="left", fontsize=10)
        a4.legend(fontsize=8, frameon=False, loc="upper center")
        a4.grid(alpha=0.25, lw=0.5)

    top.suptitle(f"Template {TEMPLATES[a.template]}, |b| > {B_CUT:.0f}$^\\circ$ (Uchuu full sky, NSIDE={nside}): the same systematic "
                 "matters less for the broader, more biased samples", fontsize=11, x=0.01,
                 ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=130)
    print(f"saved {OUT}.png")


if __name__ == "__main__":
    main()
