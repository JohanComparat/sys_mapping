"""Pedagogical figure: halo bias sets the width of the sky count distribution.

Uses the Uchuu-UM lightcone shell at 0.22 < z < 0.27 over the full sky (the eight
octant replications, ``replication_*/glist.fits``) and selects
central haloes above a minimum mass M_min.

Panel 1: abundance matching, n(>M_min) per deg2 against M_min.
Panel 2: HEALPix pixel counts, normalised to their mean, with a Poisson-log-normal
         model (Poisson sampling of a log-normal density field).
Panel 3: clustering width sigma of the log-normal field against M_min, measured
         with shot noise removed and predicted from ln(1 + b^2 sigma^2_m,pix).

Run:  python scripts/plot_pedagogy_bias_broadening.py [--nside 64] [--norm integral|peak]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.integrate import simpson
from scipy.special import j1
from scipy.stats import poisson

GLISTS = sorted(Path("~/data/Uchuu/FullSky/z0p25").expanduser().glob("replication_*/glist.fits"))
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "_static" / "pedagogy_bias_broadening"
# Halo mass function and bias of Comparat et al. (2017), fitted on MultiDark, as in colossus.
MF_MODEL, BIAS_MODEL = "comparat17", "comparat17"
BIAS_LABEL = "Comparat17"
CACHE = ROOT / "results" / "pedagogy_bias_broadening.npz"
LOGM = np.array([11.0, 11.5, 12.0, 12.5, 13.0, 13.5])
Z_LO, Z_HI = 0.21975, 0.27482
SKY_DEG2 = 4 * np.pi * (180 / np.pi) ** 2


def footprint(nside: int) -> np.ndarray:
    """All pixels: the eight octants tile the full sky."""
    return np.arange(hp.nside2npix(nside))


def measure(nside: int) -> dict:
    assert len(GLISTS) == 8, f"expected 8 octants, found {len(GLISTS)}"
    logm, ra, dec = [], [], []
    for fn in GLISTS:
        d = fits.getdata(fn, memmap=True)
        cen = np.asarray(d["upid"]) == -1
        logm.append(np.log10(np.asarray(d["Mvir"])[cen]))
        ra.append(np.asarray(d["RA"])[cen])
        dec.append(np.asarray(d["DEC"])[cen])
        del d
    logm, ra, dec = np.concatenate(logm), np.concatenate(ra), np.concatenate(dec)
    pix_in = footprint(nside)
    area_pix = hp.nside2pixarea(nside, degrees=True)
    ipix = hp.ang2pix(nside, ra, dec, lonlat=True)

    grid = np.linspace(10.5, 15.0, 91)
    srt = np.sort(logm)
    ncum = (len(srt) - np.searchsorted(srt, grid)) / SKY_DEG2

    counts = []
    for lm in LOGM:
        c = np.bincount(ipix[logm >= lm], minlength=hp.nside2npix(nside))
        counts.append(c[pix_in])
    return dict(grid=grid, ncum=ncum, counts=np.array(counts), area_pix=area_pix,
                nside=nside, n_files=len(GLISTS), logm_all=srt[srt > 10.5][:: max(1, len(srt) // 200000)])


def theory(logm_min: np.ndarray, area_pix: float):
    """Effective halo bias above M_min (BIAS_MODEL) and matter variance in one pixel."""
    from colossus.cosmology import cosmology
    from colossus.lss import bias, mass_function

    cosmo = cosmology.setCosmology("planck15")
    zc = 0.5 * (Z_LO + Z_HI)
    lm = np.linspace(10.5, 15.8, 400)
    M = 10 ** lm
    dndlnM = mass_function.massFunction(M, zc, mdef="vir", model=MF_MODEL, q_out="dndlnM")
    bM = bias.haloBias(M, zc, mdef="vir", model=BIAS_MODEL)
    b_eff, n3d = [], []
    for l0 in logm_min:
        s = lm >= l0
        w = dndlnM[s]
        b_eff.append(simpson(w * bM[s], x=np.log(M[s])) / simpson(w, x=np.log(M[s])))
        n3d.append(simpson(w, x=np.log(M[s])))

    # Limber C_ell of matter for a volume-weighted top-hat shell, then a disc window.
    z = np.linspace(Z_LO, Z_HI, 40)
    chi = cosmo.comovingDistance(0.0, z)  # Mpc/h
    dchidz = 2997.92458 / cosmo.Ez(z)
    nz = chi ** 2 * dchidz
    nz /= simpson(nz, x=z)
    ell = np.unique(np.logspace(0, 4.3, 300).astype(int))
    cl = np.empty(len(ell))
    for i, L in enumerate(ell):
        k = (L + 0.5) / chi
        pk = np.array([cosmo.matterPowerSpectrum(kk, zz) for kk, zz in zip(k, z)])
        cl[i] = simpson(nz ** 2 / dchidz / chi ** 2 * pk, x=z)
    theta = np.sqrt(np.radians(1) ** 2 * area_pix / np.pi)
    x = ell * theta
    win = 2 * j1(x) / x
    sig2_m = simpson(ell * cl * win ** 2 / (2 * np.pi), x=ell)
    return np.array(b_eff), sig2_m, np.array(n3d)


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(80)
GH_W = GH_W / GH_W.sum()


def pln_pmf(k, mu, s):
    """Poisson log-normal pmf: N ~ Poisson(e^{mu + s g}), g ~ N(0, 1)."""
    lam = np.exp(mu + s * GH_X)
    return (poisson.pmf(np.asarray(k)[:, None], lam[None, :]) * GH_W).sum(axis=1)


def pln_ml(cnt):
    """Maximum-likelihood (mu, s) of the Poisson log-normal for integer counts."""
    from scipy.optimize import minimize

    vals, nval = np.unique(cnt, return_counts=True)
    nb, var = cnt.mean(), cnt.var()
    s0 = np.sqrt(np.log(1 + max(var - nb, 1e-6) / nb ** 2))

    def nll(p):
        return -np.sum(nval * np.log(np.maximum(pln_pmf(vals, p[0], abs(p[1])), 1e-300)))

    r = minimize(nll, [np.log(nb) - s0 ** 2 / 2, s0], method="Nelder-Mead",
                 options=dict(xatol=1e-5, fatol=1e-3))
    return r.x[0], abs(r.x[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--norm", choices=["integral", "peak"], default="integral")
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()

    if (CACHE.exists() and not a.recompute and int(np.load(CACHE)["nside"]) == a.nside
            and int(np.load(CACHE).get("n_files", 1)) == len(GLISTS)):
        m = dict(np.load(CACHE))
    else:
        m = measure(a.nside)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, **m)

    counts, area_pix = m["counts"], float(m["area_pix"])
    nbar = counts.mean(axis=1)
    var = counts.var(axis=1)
    # Poisson-log-normal: N ~ Poisson(Nbar (1+delta)), ln(1+delta) ~ Gauss(-s^2/2, s).
    # Moments: Var = Nbar + Nbar^2 (e^{s^2} - 1), so s is the clustering width with shot noise removed.
    sig_G = np.sqrt(np.log(1 + (var - nbar) / nbar ** 2))
    b_eff, sig2_m, _ = theory(LOGM, area_pix)
    lm_fine = np.linspace(LOGM[0] - 0.25, LOGM[-1] + 0.25, 60)
    b_fine, _, _ = theory(lm_fine, area_pix)
    sig_pred = np.sqrt(np.log(1 + b_fine ** 2 * sig2_m))
    sig_pred_pts = np.sqrt(np.log(1 + b_eff ** 2 * sig2_m))
    # Maximum-likelihood fit of the whole count distribution (mean and width free):
    # the moment width matches the variance, the ML fit matches the shape.
    ml = [pln_ml(c) for c in counts]

    # Categorical slots 1-6 of the dataviz reference palette: one distinct hue per M_min.
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    ink, muted = "#1f1f1f", "#6b6b6b"
    plt.rcParams.update({"axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    # Panel 1: abundance matching
    a1 = ax[0]
    a1.plot(m["grid"], m["ncum"], color=ink, lw=2)
    ndeg = nbar / area_pix
    for c, lm, nd in zip(cols, LOGM, ndeg):
        a1.plot(lm, nd, "o", ms=9, color=c, mec="white", mew=1.5, zorder=3)
        a1.annotate(f"{nd:.3g}", (lm, nd), xytext=(7, 4), textcoords="offset points",
                    fontsize=8, color=muted)
    a1.set_yscale("log")
    a1.set_xlabel(r"$\log_{10} M_{\min}\ [M_\odot/h]$")
    a1.set_ylabel(r"$N(>M_{\min})$ [deg$^{-2}$]")
    a1.set_title("1. Abundance matching: rarer haloes, fewer per deg$^2$",
                 loc="left", fontsize=10)
    a1.grid(alpha=0.25, lw=0.5)
    a1.text(0.03, 0.04, f"Uchuu-UM centrals, full sky, {Z_LO:.2f} < z < {Z_HI:.2f}",
            transform=a1.transAxes, fontsize=8, color=muted)

    # Panel 2: pixel density distributions and the Poisson-log-normal model
    a2 = ax[1]
    for c, lm, cnt, (mu_ml, s_ml) in zip(cols, LOGM, counts, ml):
        nb = cnt.mean()
        # Bin edges sit between integer counts so sparse samples show no aliasing spikes.
        step = max(1, int(np.ceil(0.05 * nb)))
        kedges = np.arange(-0.5, 4 * nb + step, step)
        e = kedges / nb
        h, _ = np.histogram(cnt, bins=kedges)
        h = h / (h.sum() * np.diff(e))
        k = np.arange(0, int(kedges[-1]) + 1)
        pmf = pln_pmf(k, mu_ml, s_ml)
        mod = np.add.reduceat(pmf, np.searchsorted(k, kedges[:-1] + 0.5)) / np.diff(e)
        mod = mod[: len(h)]
        if a.norm == "peak":
            mod, h = mod / h.max(), h / h.max()
        a2.stairs(h, e, color=c, lw=2, label=rf"$\geq 10^{{{lm:.1f}}}$")
        a2.stairs(mod, e, color=c, lw=1.2, ls="--")
    a2.plot([], [], color=muted, lw=2, label="Uchuu")
    a2.plot([], [], color=muted, lw=1.2, ls="--", label="Poisson-log-normal, ML fit")
    a2.set_xlabel(r"$N_{\rm pix} / \bar N = 1+\delta$")
    a2.set_xlim(0, 4)
    a2.set_ylabel("PDF (integral = 1)" if a.norm == "integral" else "PDF / max")
    a2.set_title(f"2. HEALPix counts (NSIDE={a.nside}, {area_pix:.2f} deg$^2$)",
                 loc="left", fontsize=10)
    a2.legend(title=r"$M_{\min}\ [M_\odot/h]$", fontsize=8, title_fontsize=8, frameon=False)
    a2.text(0.97, 0.40,
            "Poisson log-normal\n"
            r"$P(N) = \int \mathrm{d}g\, \frac{e^{-g^2/2}}{\sqrt{2\pi}}\,"
            r"\frac{\lambda(g)^N e^{-\lambda(g)}}{N!}$" "\n"
            r"$\lambda(g) = e^{\mu + \sigma g}$, "
            r"$\langle N\rangle = e^{\mu+\sigma^2/2}$" "\n"
            r"($\mu$, $\sigma$: maximum likelihood)",
            transform=a2.transAxes, ha="right", va="top", fontsize=8.5, color=ink,
            linespacing=1.7)
    a2.grid(alpha=0.25, lw=0.5)

    # Panel 3: clustering width against M_min
    a3 = ax[2]
    a3.plot(lm_fine, sig_pred, color=ink, lw=2,
            label=r"prediction: $\sigma^2 = \ln(1 + b^2\sigma^2_{m,\rm pix})$")
    for c, lm, s, b in zip(cols, LOGM, sig_G, b_eff):
        a3.plot(lm, s, "o", ms=9, color=c, mec="white", mew=1.5, zorder=3)
        a3.annotate(f"b={b:.2f}", (lm, s), xytext=(-14, 10), textcoords="offset points",
                    fontsize=8, color=muted)
    for c, lm, (mu_ml, s_ml) in zip(cols, LOGM, ml):
        a3.plot(lm, s_ml, "D", ms=6, color="white", mec=c, mew=1.5, zorder=3)
    a3.plot([], [], "o", color=muted, label="Uchuu, moments (shot noise removed)")
    a3.plot([], [], "D", color="white", mec=muted, label="Uchuu, ML fit of the distribution")
    a3.set_xlabel(r"$\log_{10} M_{\min}\ [M_\odot/h]$")
    a3.set_ylabel(r"log-normal width $\sigma$ of $1+\delta$")
    a3.set_title(rf"3. Width grows with bias; $\sigma_{{m,\rm pix}}$ = {np.sqrt(sig2_m):.3f}",
                 loc="left", fontsize=10)
    a3.legend(fontsize=8, frameon=False, loc="upper left")
    a3.grid(alpha=0.25, lw=0.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=150)
    for lm, nb, s, sp, b, (mu_ml, s_ml) in zip(LOGM, nbar, sig_G, sig_pred_pts, b_eff, ml):
        print(f"logM>={lm:.1f}  Nbar/pix={nb:9.2f}  sigma_mom={s:.3f}  sigma_ML={s_ml:.3f}  "
              f"mean_ML/Nbar={np.exp(mu_ml + s_ml ** 2 / 2) / nb:.3f}  sigma_pred={sp:.3f}  b={b:.2f}")
    print(f"sigma_m,pix={np.sqrt(sig2_m):.4f}; saved {OUT}.png")


if __name__ == "__main__":
    main()
