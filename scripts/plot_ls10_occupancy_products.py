#!/usr/bin/env python
"""Figure: corrected over observed w(theta) for the occupancy-resolved LS10 products.

Each sample is issued at the finest NSIDE its mean occupancy supports, so the nine
panels do not share a resolution; each is titled with its own.  Reads the
``*_wtheta_data.json`` and ``*_params.json`` files that
``run_ls10_analysis.py --min-per-pixel`` writes.

Run::

    python scripts/plot_ls10_occupancy_products.py [--products data/sys_weights_auto]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
METHODS = ["OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"]


def load(products: Path) -> list[dict]:
    rows = []
    for f in glob.glob(str(products / "LS10_VLIM_ANY_*_params.json")):
        d = json.load(open(f))
        w = json.load(open(f.replace("_params.json", "_wtheta_data.json")))
        rows.append(dict(mstar=float(re.search(r"ANY_([0-9.]+)_", f).group(1)),
                         nside=d["nside"], nbar=d["n_galaxies"] / d["n_good_pix"],
                         theta=np.asarray(w["theta_arcmin"]), w_obs=np.asarray(w["w_obs"]),
                         w_corr={m: np.asarray(w["all_w_corr"][m]) for m in METHODS}))
    return sorted(rows, key=lambda r: r["mstar"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--products", default=str(REPO / "data" / "sys_weights_auto"))
    ap.add_argument("--out", default=str(REPO / "docs" / "_static" / "results_ls10"
                                         / "wtheta_ratio_occupancy.png"))
    args = ap.parse_args()
    rows = load(Path(args.products))
    if not rows:
        raise SystemExit(f"no products under {args.products}")

    fig, axes = plt.subplots(3, 3, figsize=(10, 8.4), sharex=True, sharey=True)
    for ax, r in zip(axes.flat, rows):
        pos = r["w_obs"] > 0
        for m in METHODS:
            ax.plot(r["theta"][pos], r["w_corr"][m][pos] / r["w_obs"][pos], lw=1.2, label=m)
        ax.axhline(1.0, c="k", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(rf"$\log M_\star\geq{r['mstar']}$, NSIDE {r['nside']}", fontsize=9)
        ax.grid(alpha=0.3, which="both")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\theta$ [arcmin]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$w_{\rm corr}/w_{\rm obs}$")
    axes[0, 0].set_ylim(0.55, 1.05)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
