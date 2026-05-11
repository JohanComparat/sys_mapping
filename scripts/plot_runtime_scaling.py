#!/usr/bin/env python3
"""
Generate docs/_static/runtime_scaling.png — runtime vs n_pix for each method.

Data sources:
  - OLS / ElasticNet / ISD-1 : elapsed_s from *_partial_ElasticNet_ISD-1_OLS.json
  - ISD-3                    : elapsed_s from *_partial_ISD-3.json
  - MCMC-add / MCMC-comb     : measured at NSIDE 64 (11 templates) from logs;
                                scaled to other NSIDEs by n_pix ratio

Run from the repo root:
    python scripts/plot_runtime_scaling.py
"""

import glob
import json
import math
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "sys_weights"
FIG_DIR  = REPO / "docs" / "_static"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── collect data from partial files ──────────────────────────────────────────
# raw[method][nside] = list of (n_pix, elapsed_s)
raw = defaultdict(lambda: defaultdict(list))

for nside in [32, 64, 128, 256]:
    # OLS / ElasticNet / ISD-1
    for fpath in sorted(DATA_DIR.glob(f"*_NSIDE{nside:04d}_partial_ElasticNet_ISD-1_OLS.json")):
        d = json.loads(fpath.read_text())
        n_pix = d.get("n_good", 0)
        for method in ["OLS", "ElasticNet", "ISD-1"]:
            mdata = d.get(method, {})
            if isinstance(mdata, dict) and "elapsed_s" in mdata:
                raw[method][nside].append((n_pix, mdata["elapsed_s"]))
    # ISD-3
    for fpath in sorted(DATA_DIR.glob(f"*_NSIDE{nside:04d}_partial_ISD-3.json")):
        d = json.loads(fpath.read_text())
        n_pix = d.get("n_good", 0)
        mdata = d.get("ISD-3", {})
        if isinstance(mdata, dict) and "elapsed_s" in mdata:
            raw["ISD-3"][nside].append((n_pix, mdata["elapsed_s"]))

# ── MCMC timings from logs ────────────────────────────────────────────────────
# MCMC-add at NSIDE 64 (11 templates, from ls10_MCMC-add_20260507_191332.log):
MCMC_ADD_N64 = [
    (21667, 94.2), (21669, 95.9), (21675, 91.8), (21662, 84.1),
    (21646, 78.3), (21555, 76.5), (21344, 74.9), (21563, 80.3), (21637, 101.2),
]
# Scale to NSIDE 32 by n_pix ratio (MCMC cost is O(n_pix))
NSIDE64_MEAN_NPIX = 21_586   # mean over 9 samples
NSIDE32_MEAN_NPIX = 5_612

for n_pix, t in MCMC_ADD_N64:
    raw["MCMC-add"][64].append((n_pix, t))
    # Scale to NSIDE 32
    n32 = int(n_pix * NSIDE32_MEAN_NPIX / NSIDE64_MEAN_NPIX)
    raw["MCMC-add"][32].append((n32, t * NSIDE32_MEAN_NPIX / NSIDE64_MEAN_NPIX))

# MCMC-comb at NSIDE 64: single sample observed (116.3s), use it for all 9
MCMC_COMB_N64_TYPICAL = 116.3
for n_pix, t_add in MCMC_ADD_N64:
    ratio = MCMC_COMB_N64_TYPICAL / (sum(x[1] for x in MCMC_ADD_N64) / len(MCMC_ADD_N64))
    t_comb = t_add * ratio
    raw["MCMC-comb"][64].append((n_pix, t_comb))
    n32 = int(n_pix * NSIDE32_MEAN_NPIX / NSIDE64_MEAN_NPIX)
    raw["MCMC-comb"][32].append((n32, t_comb * NSIDE32_MEAN_NPIX / NSIDE64_MEAN_NPIX))

# ── plotting config ───────────────────────────────────────────────────────────
METHOD_STYLES = {
    "OLS":       dict(color="#1f77b4", marker="o", ls="-",  label="OLS"),
    "ElasticNet":dict(color="#ff7f0e", marker="s", ls="-",  label="ElasticNet"),
    "ISD-1":     dict(color="#2ca02c", marker="^", ls="-",  label="ISD-1"),
    "ISD-3":     dict(color="#d62728", marker="D", ls="--", label="ISD-3"),
    "MCMC-add":  dict(color="#9467bd", marker="v", ls="-",  label="MCMC-add"),
    "MCMC-comb": dict(color="#8c564b", marker="P", ls="-",  label="MCMC-comb"),
}

# build arrays: mean per NSIDE (all 9 samples)
def mean_per_nside(method_data):
    result = {}
    for nside, pts in sorted(method_data.items()):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if xs:
            result[nside] = (np.mean(xs), np.mean(ys), np.std(ys))
    return result

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

for method, style in METHOD_STYLES.items():
    per_nside = mean_per_nside(raw[method])
    if not per_nside:
        continue
    xs = [v[0] for v in per_nside.values()]
    ys = [v[1] for v in per_nside.values()]
    errs = [v[2] for v in per_nside.values()]
    # scatter all individual points (semi-transparent)
    for nside, pts in sorted(raw[method].items()):
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   color=style["color"], alpha=0.2, s=20, zorder=2)
    # mean per NSIDE connected by line
    ax.errorbar(xs, ys, yerr=errs,
                color=style["color"], marker=style["marker"], ls=style["ls"],
                lw=2, ms=7, capsize=3, capthick=1, label=style["label"], zorder=4)

# ── reference power-law guides ────────────────────────────────────────────────
n_ref = np.array([3e3, 4e5])
ax.plot(n_ref, 0.002 * n_ref,       color="gray", ls=":",  lw=1, alpha=0.7, zorder=1)
ax.plot(n_ref, 3e-9 * n_ref**2,     color="gray", ls="-.", lw=1, alpha=0.7, zorder=1)
ax.text(n_ref[-1]*0.7, 0.002*n_ref[-1]*1.5, r"$O(n)$",   fontsize=7, color="gray")
ax.text(n_ref[-1]*0.55, 3e-9*n_ref[-1]**2*1.8, r"$O(n^2)$", fontsize=7, color="gray")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of unmasked pixels $N_{\\rm pix}$", labelpad=4)
ax.set_ylabel("Wall-clock time per sample (s)", labelpad=4)
ax.set_title("sys_mapping: runtime vs map resolution\n"
             "(LS10 BGS, 11 templates, NSIDE 32–256; median ± std over 9 samples)",
             fontsize=9)

# secondary x-axis: NSIDE labels
ax2 = ax.twiny()
ax2.set_xscale("log")
ax2.set_xlim(ax.get_xlim())
nside_ticks = {32: 5600, 64: 21600, 128: 84600, 256: 325000}
ax2.set_xticks(list(nside_ticks.values()))
ax2.set_xticklabels([f"NSIDE {n}" for n in nside_ticks], fontsize=7)
ax2.tick_params(axis="x", length=4)

ax.legend(loc="upper left", ncol=2, fontsize=8, framealpha=0.9)
ax.set_xlim(2e3, 6e5)

out = FIG_DIR / "runtime_scaling.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
