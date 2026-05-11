#!/usr/bin/env python3
"""
All-methods w(theta) figure for LS10 BGS systematic analysis at NSIDE = 64.

One panel per sample (all 9 samples).  Each panel:
  - top    : w(theta) log-log — observed + 6 corrected curves
  - bottom : ratio w_corr / w_obs for each method

Data source: data/sys_weights/*_wtheta_data.json
  (analytical correction, Eq. 15-16, written by run_ls10_analysis.py)

Output: docs/_static/results_ls10/wtheta_corrected_nside64.png
"""

from pathlib import Path
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

matplotlib.use("Agg")

REPO     = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "sys_weights"
FIG_DIR  = REPO / "docs" / "_static" / "results_ls10"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── shared colour/style palette ───────────────────────────────────────────────
from sys_mapping.plotting import (
    METHOD_COLORS, METHOD_LINESTYLES, METHOD_MARKERS,
    METHOD_LABELS, METHOD_ORDER,
)

OBS_COLOR = "#222222"

# ── global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   7,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "0.6",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

SAMPLES = [
    dict(
        sample_id="LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486",
        label=r"$\log_{10}M_*>9.0$,  $z<0.08$",
    ),
    dict(
        sample_id="LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502",
        label=r"$\log_{10}M_*>9.5$,  $z<0.12$",
    ),
    dict(
        sample_id="LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238",
        label=r"$\log_{10}M_*>10.0$,  $z<0.18$",
    ),
    dict(
        sample_id="LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841",
        label=r"$\log_{10}M_*>10.25$,  $z<0.22$",
    ),
    dict(
        sample_id="LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228",
        label=r"$\log_{10}M_*>10.5$,  $z<0.26$",
    ),
    dict(
        sample_id="LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710",
        label=r"$\log_{10}M_*>10.75$,  $z<0.31$",
    ),
    dict(
        sample_id="LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838",
        label=r"$\log_{10}M_*>11.0$,  $z<0.35$",
    ),
    dict(
        sample_id="LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855",
        label=r"$\log_{10}M_*>11.25$,  $z<0.35$",
    ),
    dict(
        sample_id="LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882",
        label=r"$\log_{10}M_*>11.5$,  $z<0.35$",
    ),
]


def load_json(sample_id: str) -> dict | None:
    path = DATA_DIR / f"{sample_id}_NSIDE0064_wtheta_data.json"
    if not path.exists():
        print(f"  [skip] {sample_id}: {path.name} not found")
        return None
    with open(path) as f:
        return json.load(f)


def plot_sample(ax_top, ax_bot, sample):
    d = load_json(sample["sample_id"])
    if d is None:
        ax_top.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax_top.transAxes, color="gray")
        return

    theta = np.asarray(d["theta_arcmin"])
    w_obs = np.asarray(d["w_obs"])
    pos_obs = (w_obs > 0) & np.isfinite(w_obs)

    kw_obs = dict(color=OBS_COLOR, marker="o", ls="-", lw=2.0, ms=5,
                  capsize=2, capthick=0.8, zorder=5)
    ax_top.plot(theta[pos_obs], w_obs[pos_obs], label="Observed", **{k: v for k, v in kw_obs.items() if k != "capsize" and k != "capthick"})

    ax_bot.axhline(1.0, color="k", lw=0.8, ls="--")

    for method in METHOD_ORDER:
        w_corr = np.asarray(d["all_w_corr"].get(method, []))
        if len(w_corr) != len(theta):
            continue
        pos = pos_obs & (w_corr > 0) & np.isfinite(w_corr)
        if not pos.any():
            continue

        color   = METHOD_COLORS[method]
        ls      = METHOD_LINESTYLES[method]
        marker  = METHOD_MARKERS[method]
        label   = METHOD_LABELS[method]

        kw = dict(color=color, ls=ls, marker=marker, lw=1.5, ms=4, zorder=3)
        ax_top.plot(theta[pos], w_corr[pos], label=label, **kw)

        w_safe = np.where(pos_obs, w_obs, np.nan)
        ratio  = w_corr / w_safe
        ax_bot.plot(theta[pos], ratio[pos],
                    color=color, ls=ls, marker=marker, lw=1.3, ms=3)

    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_top.set_ylabel(r"$w(\theta)$", labelpad=4)
    ax_top.set_title(sample["label"], fontsize=9, pad=4)
    ax_top.legend(loc="upper right", fontsize=6.5, handlelength=1.6,
                  ncol=2, columnspacing=0.8)

    ax_bot.set_xscale("log")
    ax_bot.set_ylabel(r"$w_{\rm corr}\,/\,w_{\rm obs}$", labelpad=4, fontsize=8)
    ax_bot.set_xlim(0.4, 70)


# ── build figure ─────────────────────────────────────────────────────────────
N      = len(SAMPLES)
N_COLS = 3
N_ROWS = (N + N_COLS - 1) // N_COLS

fig = plt.figure(figsize=(14, N_ROWS * 5.2))
outer_gs = GridSpec(N_ROWS, N_COLS, figure=fig, hspace=0.52, wspace=0.32)

for i, sample in enumerate(SAMPLES):
    row, col = i // N_COLS, i % N_COLS
    inner_gs = GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=outer_gs[row, col],
        hspace=0.06,
        height_ratios=[3, 1.2],
    )
    ax_top = fig.add_subplot(inner_gs[0])
    ax_bot = fig.add_subplot(inner_gs[1], sharex=ax_top)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_bot.set_xlabel(r"$\theta$  [arcmin]", labelpad=3, fontsize=8)

    plot_sample(ax_top, ax_bot, sample)

fig.suptitle(
    "LS10 BGS: $w(\\theta)$ before and after systematic correction — all methods\n"
    "(NSIDE = 64, analytical correction Eq. 15–16, 0.5–60 arcmin, 20 bins)",
    fontsize=10, y=1.01,
)

out = FIG_DIR / "wtheta_corrected_nside64.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
