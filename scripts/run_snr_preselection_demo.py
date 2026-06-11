#!/usr/bin/env python
"""Generate figures for the SNR template pre-selection documentation page.

Produces 9 PNG files in docs/_static/results_snr_preselection/:

  01_maps.png                  — HEALPix maps (clean field, templates,
                                  contaminated field, residual)
  02_snr_ranking_data.png      — SNR bar chart for the 'data' method
  02_snr_ranking_template.png  — SNR bar chart for the 'template' method
  02_snr_ranking_isd.png       — SNR bar chart for the 'isd' method
  03_detection_vs_amplitude.png — SNR of injected template vs amplitude
  04_pvalues.png               — ISD mock-based p-values at three levels
  05_pipeline_result.png       — combined detection summary table
  06_timing.png                — wallclock time per ranking method
  07_mock_convergence.png      — ISD p-value convergence vs N_mocks

Usage::

    python scripts/run_snr_preselection_demo.py

Requires: glass, healpy, matplotlib (all in the sys_map conda environment).
Runtime: ~15–30 min (dominated by GLASS mock generation at 15 M galaxies).
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import healpy as hp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sys_mapping.diagnostics import isd_template_significance, snr_template_ranking
from sys_mapping.glass_mocks import generate_glass_fullsky_mock
from sys_mapping.maps import (
    assign_template_values,
    compute_overdensity,
    generate_systematic_maps,
    pixelize_catalog,
)
from sys_mapping.model_selection import greedy_forward_select, snr_preselect
from sys_mapping.simulation import LEVELS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NSIDE        = 32
N_TOTAL      = 15_000_000      # galaxies
RAND_FACTOR  = 2               # keep memory < 1 GB (GLASS mocks)
N_TEMPLATES  = 20
CONTAM_IDX   = [2, 7]          # injected templates (family-2, two realisations)
FAMILIES     = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
                0, 1, 2, 3, 4, 0, 1, 2, 3, 4]  # 4 realisations × 5 families
Z_EDGES      = np.array([0.0, 0.5])
NZ           = np.array([5_000_000.0])          # n(z) counts normalised to N_TOTAL
N_MOCKS      = 100             # GLASS mocks for ISD p-value figures
SEED         = 7

# Mock counts to sweep for the convergence analysis (reuses the N_MOCKS=100 run)
MOCK_SWEEP   = [5, 10, 20, 30, 50, 75, 100]

# Output directory
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "docs", "_static", "results_snr_preselection"
)

# Colour scheme
METHOD_COLORS = {
    "data":     "#4E79A7",
    "template": "#59A14F",
    "isd":      "#F28E2B",
}
CONTAM_COLOR = "#E15759"
NOISE_COLOR  = "#BBBBBB"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contaminate(delta_g_clean, delta_t, amplitude):
    delta_g = delta_g_clean.copy()
    for i in CONTAM_IDX:
        delta_g = delta_g + amplitude * delta_t[i]
    return delta_g


def _bar_colors(n):
    colors = [NOISE_COLOR] * n
    for i in CONTAM_IDX:
        colors[i] = CONTAM_COLOR
    return colors


def _savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build the GLASS universe
    # ------------------------------------------------------------------
    t_start = time.perf_counter()
    print(f"Generating GLASS full-sky mock ({N_TOTAL:,} galaxies, NSIDE={NSIDE}) …")
    cat = generate_glass_fullsky_mock(
        NSIDE, N_TOTAL, Z_EDGES, NZ, seed=SEED, rand_factor=RAND_FACTOR,
    )

    print(f"Generating {N_TEMPLATES} systematic maps …")
    delta_t_full = generate_systematic_maps(NSIDE, families=FAMILIES, seed=SEED + 1)

    n_gal  = pixelize_catalog(cat["ra"],      cat["dec"],      NSIDE)
    n_rand = pixelize_catalog(cat["ra_rand"], cat["dec_rand"], NSIDE)
    del cat  # free ~1 GB of position arrays

    delta_g_clean, good = compute_overdensity(n_gal, n_rand)
    delta_t = assign_template_values(delta_t_full, good)  # (N_TEMPLATES, n_good)
    print(f"  Universe ready in {time.perf_counter()-t_start:.1f} s  "
          f"| n_good={good.sum()}")

    n_full = hp.nside2npix(NSIDE)
    delta_g_med = _contaminate(delta_g_clean, delta_t, LEVELS["medium"])
    delta_g_high = _contaminate(delta_g_clean, delta_t, LEVELS["high"])

    template_labels = [
        f"T{i}" + (" ★" if i in CONTAM_IDX else "")
        for i in range(N_TEMPLATES)
    ]
    bar_colors = _bar_colors(N_TEMPLATES)

    # ------------------------------------------------------------------
    # 2. Timing measurement — each method timed at high contamination level
    # ------------------------------------------------------------------
    print("\nTiming ranking methods …")
    timing = {}
    N_REPEATS = 5
    for method in ["data", "template", "isd"]:
        times = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            snr_template_ranking(delta_g_high, delta_t, method=method)
            times.append(time.perf_counter() - t0)
        timing[method] = min(times)   # best-of-5 (avoids warm-up noise)

    print("\n  Method timing (best of 5 repeats):")
    for method, t in timing.items():
        print(f"    {method:12s}: {t*1000:.2f} ms")

    # ------------------------------------------------------------------
    # 3. ISD significance — run once per level, reuse mocks for convergence
    # ------------------------------------------------------------------
    print(f"\nRunning ISD significance ({N_MOCKS} GLASS mocks × 3 levels) …")
    isd_results = {}
    for level in ["low", "medium", "high"]:
        print(f"  level={level} …", end="", flush=True)
        t0 = time.perf_counter()
        dg = _contaminate(delta_g_clean, delta_t, LEVELS[level])
        isd_results[level] = isd_template_significance(
            dg, delta_t, good, NSIDE, N_TOTAL, Z_EDGES, NZ,
            n_mocks=N_MOCKS, seed=SEED + 10, rand_factor=RAND_FACTOR,
        )
        print(f" {time.perf_counter()-t0:.0f} s")

    # ------------------------------------------------------------------
    # Figure 01: HEALPix maps
    # ------------------------------------------------------------------
    print("\nFigure 01: maps …")

    def _embed(arr, good, n_full, fill=hp.UNSEEN):
        m = np.full(n_full, fill)
        m[good] = arr
        return m

    panels = [
        (_embed(delta_g_clean,               good, n_full), "Clean overdensity $\\delta_g$",         "RdBu_r"),
        (_embed(delta_t[CONTAM_IDX[0]],      good, n_full), f"Template {CONTAM_IDX[0]} (injected)",  "viridis"),
        (_embed(delta_t[CONTAM_IDX[1]],      good, n_full), f"Template {CONTAM_IDX[1]} (injected)",  "viridis"),
        (_embed(delta_t[0],                  good, n_full), "Template 0 (noise)",                    "viridis"),
        (_embed(delta_g_med,                 good, n_full), "Contaminated $\\delta_g$ (medium)",     "RdBu_r"),
        (_embed(delta_g_med - delta_g_clean, good, n_full), "Residual (contamination only)",         "RdBu_r"),
    ]

    fig = plt.figure(figsize=(14, 9))
    for k, (m, title, cmap) in enumerate(panels):
        plt.subplot(2, 3, k + 1)
        hp.mollview(m, title=title, cmap=cmap, hold=True, cbar=True, unit="")
    fig.suptitle(
        f"GLASS full-sky simulation — NSIDE={NSIDE}, {N_TOTAL:,} galaxies, "
        f"{N_TEMPLATES} templates\nInjected: T{CONTAM_IDX[0]} and T{CONTAM_IDX[1]}",
        y=1.01, fontsize=11,
    )
    _savefig(fig, "01_maps.png")

    # ------------------------------------------------------------------
    # Figures 02: SNR bar charts per method
    # ------------------------------------------------------------------
    for method in ["data", "template", "isd"]:
        print(f"Figure 02: SNR bar chart — method={method} …")
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
        for ax, level in zip(axes, ["low", "medium", "high"]):
            amp = LEVELS[level]
            dg = _contaminate(delta_g_clean, delta_t, amp)
            snr = snr_template_ranking(dg, delta_t, method=method)
            ax.bar(range(N_TEMPLATES), snr, color=bar_colors, edgecolor="white", linewidth=0.4)
            ax.set_xticks(range(N_TEMPLATES))
            ax.set_xticklabels(template_labels, rotation=60, ha="right", fontsize=7)
            ax.set_title(f"Level: {level}  (a = {amp:.2f})")
            ax.set_xlabel("Template index")
            if ax is axes[0]:
                ylabel = {"data": "Pearson |r|", "template": "OLS |t|-stat",
                          "isd": r"$\Delta\chi^2$"}[method]
                ax.set_ylabel(ylabel)

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(color=CONTAM_COLOR, label="Injected (contaminating)"),
            Patch(color=NOISE_COLOR,  label="Noise template"),
        ]
        fig.legend(handles=legend_handles, loc="upper right", fontsize=9,
                   bbox_to_anchor=(0.99, 0.99))
        method_title = {"data": "Data cross-correlation", "template": "OLS t-statistic",
                        "isd": r"ISD $\Delta\chi^2$"}[method]
        fig.suptitle(f"SNR ranking — {method_title}", fontsize=12)
        fig.tight_layout()
        _savefig(fig, f"02_snr_ranking_{method}.png")

    # ------------------------------------------------------------------
    # Figure 03: SNR of injected template 2 vs contamination amplitude
    # ------------------------------------------------------------------
    print("Figure 03: detection vs amplitude …")
    amplitudes = np.array([0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.15])
    snr_vs_amp = {m: [] for m in ["data", "template", "isd"]}
    for amp in amplitudes:
        dg = _contaminate(delta_g_clean, delta_t, amp)
        for method in ["data", "template", "isd"]:
            s = snr_template_ranking(dg, delta_t, method=method)
            snr_vs_amp[method].append(s[CONTAM_IDX[0]])

    fig, ax = plt.subplots(figsize=(7, 4))
    for method, snr_list in snr_vs_amp.items():
        label = {"data": "Data cross-corr.", "template": "OLS t-stat",
                 "isd": r"ISD $\Delta\chi^2$"}[method]
        ax.plot(amplitudes, snr_list, color=METHOD_COLORS[method], marker="o", label=label)
    for level, amp in LEVELS.items():
        ax.axvline(amp, linestyle=":", color="grey", alpha=0.6)
        ax.text(amp, ax.get_ylim()[1] * 0.97, level, ha="center", fontsize=8, color="grey")
    ax.set_xlabel("Contamination amplitude $a$")
    ax.set_ylabel(f"SNR (injected template {CONTAM_IDX[0]})")
    ax.set_title(f"Detection sensitivity vs amplitude  (NSIDE={NSIDE}, {N_TOTAL:,} gal)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, "03_detection_vs_amplitude.png")

    # ------------------------------------------------------------------
    # Figure 04: ISD mock-based p-values
    # ------------------------------------------------------------------
    print("Figure 04: p-values …")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for ax, level in zip(axes, ["low", "medium", "high"]):
        p = isd_results[level]["p_values"]
        ax.bar(range(N_TEMPLATES), p, color=bar_colors, edgecolor="white", linewidth=0.4)
        ax.axhline(0.05, linestyle="--", color="red", linewidth=1.2, label="p = 0.05")
        ax.set_xticks(range(N_TEMPLATES))
        ax.set_xticklabels(template_labels, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"Level: {level}  (a = {LEVELS[level]:.2f})")
        ax.set_xlabel("Template index")
        if ax is axes[0]:
            ax.set_ylabel("p-value")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=CONTAM_COLOR, label="Injected (contaminating)"),
        Patch(color=NOISE_COLOR,  label="Noise template"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=9,
               bbox_to_anchor=(0.99, 0.99))
    fig.suptitle(
        f"ISD mock-based p-values (N_mocks={N_MOCKS})\n"
        "Red dashed line: p = 0.05 significance threshold",
        fontsize=12,
    )
    fig.tight_layout()
    _savefig(fig, "04_pvalues.png")

    # ------------------------------------------------------------------
    # Figure 05: Pipeline detection summary table
    # ------------------------------------------------------------------
    print("Figure 05: pipeline summary …")
    methods = ["data", "template", "isd"]
    method_labels_fig = {
        "data":     "Data cross-corr.",
        "template": "OLS t-stat",
        "isd":      r"ISD $\Delta\chi^2$",
    }
    levels = ["low", "medium", "high"]
    detected = np.zeros((len(methods), len(levels)), dtype=bool)
    for mi, method in enumerate(methods):
        for li, level in enumerate(levels):
            dg = _contaminate(delta_g_clean, delta_t, LEVELS[level])
            result = snr_preselect(dg, delta_t, method=method, n_top=3)
            top3 = set(result.selected_indices[:3])
            detected[mi, li] = all(ci in top3 for ci in CONTAM_IDX)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    cmap_binary = mcolors.ListedColormap(["#EEEEEE", "#59A14F"])
    ax.imshow(detected.astype(float), cmap=cmap_binary, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{lv}\n(a={LEVELS[lv]:.2f})" for lv in levels], fontsize=10)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([method_labels_fig[m] for m in methods], fontsize=10)
    ax.set_xlabel("Contamination level")
    ax.set_title(
        f"Both injected templates (T{CONTAM_IDX[0]}, T{CONTAM_IDX[1]}) in top-3\n"
        "Green = detected, Grey = not detected",
        fontsize=10,
    )
    for mi in range(len(methods)):
        for li in range(len(levels)):
            text = "✓" if detected[mi, li] else "✗"
            ax.text(li, mi, text, ha="center", va="center", fontsize=14,
                    color="white" if detected[mi, li] else "#888888")
    fig.tight_layout()
    _savefig(fig, "05_pipeline_result.png")

    # ------------------------------------------------------------------
    # Figure 06: Timing bar chart
    # ------------------------------------------------------------------
    print("Figure 06: timing …")
    fig, ax = plt.subplots(figsize=(6, 3))
    methods_ord = ["data", "template", "isd"]
    times_ms = [timing[m] * 1000 for m in methods_ord]
    colors_ord = [METHOD_COLORS[m] for m in methods_ord]
    labels_ord = ["Data cross-corr.", "OLS t-stat", r"ISD $\Delta\chi^2$"]
    bars = ax.barh(labels_ord, times_ms, color=colors_ord, edgecolor="white")
    x_max = max(times_ms) * 1.25
    for bar, t in zip(bars, times_ms):
        ax.text(
            bar.get_width() + x_max * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{t:.2f} ms",
            va="center", fontsize=9,
        )
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Wallclock time (ms)")
    ax.set_title(
        f"Method timing — best of {N_REPEATS} repeats\n"
        f"NSIDE={NSIDE}, {N_TEMPLATES} templates, {N_TOTAL:,} galaxies",
        fontsize=10,
    )
    fig.tight_layout()
    _savefig(fig, "06_timing.png")

    # ------------------------------------------------------------------
    # Figure 07: Mock convergence (reuses high-level mock Δχ² matrix)
    # ------------------------------------------------------------------
    print("Figure 07: mock convergence …")
    delta_chi2_data = isd_results["high"]["delta_chi2"]
    mocks_all = isd_results["high"]["delta_chi2_mocks"]   # (N_MOCKS, N_TEMPLATES)

    # Compute p-values for each n in MOCK_SWEEP using the first n rows
    pval_vs_n = {}
    for n in MOCK_SWEEP:
        sub = mocks_all[:n]
        n_extreme = np.sum(sub >= delta_chi2_data[np.newaxis, :], axis=0)
        pval_vs_n[n] = (n_extreme + 1.0) / (n + 1.0)

    noise_idx = [i for i in range(N_TEMPLATES) if i not in CONTAM_IDX]

    # Convergence metric: max absolute change vs reference (N_MOCKS)
    p_ref = pval_vs_n[MOCK_SWEEP[-1]]
    conv_metric = {n: float(np.max(np.abs(pval_vs_n[n][noise_idx] - p_ref[noise_idx])))
                   for n in MOCK_SWEEP}

    # Rule-of-thumb: first N where the max change < 0.05
    rot_n = next((n for n in MOCK_SWEEP if conv_metric[n] < 0.05), MOCK_SWEEP[-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: p-value curves
    n_arr = MOCK_SWEEP
    for i in range(N_TEMPLATES):
        is_contam = i in CONTAM_IDX
        color = CONTAM_COLOR if is_contam else NOISE_COLOR
        lw    = 2.0 if is_contam else 0.7
        alpha = 1.0 if is_contam else 0.6
        p_arr = [pval_vs_n[n][i] for n in n_arr]
        label = f"T{i} (injected)" if is_contam else None
        ax1.plot(n_arr, p_arr, color=color, lw=lw, alpha=alpha,
                 marker="o", markersize=4, label=label)

    # Theoretical lower bound 1/(N+1)
    n_dense = np.linspace(3, MOCK_SWEEP[-1] + 5, 300)
    ax1.plot(n_dense, 1 / (n_dense + 1), color="black", linestyle=":",
             lw=1.5, label="1/(N+1) lower bound")
    ax1.axhline(0.05, color="red", linestyle="--", lw=1.2, label="p = 0.05")
    ax1.set_xlabel("N_mocks")
    ax1.set_ylabel("p-value")
    ax1.set_title("p-values vs N_mocks (high contamination level)")
    ax1.set_xlim(0, MOCK_SWEEP[-1] + 5)
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(fontsize=8, loc="upper right")

    # Right: convergence metric vs N
    ns_conv = MOCK_SWEEP
    dp_conv = [conv_metric[n] for n in ns_conv]
    ax2.plot(ns_conv, dp_conv, "o-", color="#4E79A7", lw=2)
    ax2.axhline(0.05, color="red", linestyle="--", lw=1.2, label=r"$|\Delta p|$ = 0.05")
    ax2.axvline(rot_n, color="green", linestyle="--", lw=1.5,
                label=f"Convergence at N = {rot_n}")
    ax2.set_xlabel("N_mocks")
    ax2.set_ylabel(r"max$\,|\Delta p|$ (noise templates vs N=100)")
    ax2.set_title("Convergence of noise-template p-values")
    ax2.legend(fontsize=9)

    # Rule-of-thumb annotation
    rule_text = (
        "Rule of thumb:\n"
        r"$N_\mathrm{mocks} \geq \max\!\left(20,\,\lceil 5/\alpha \rceil\right)$"
        "\n"
        r"For $\alpha=0.05$: $N_\mathrm{mocks} \geq 100$"
    )
    fig.text(0.99, 0.01, rule_text, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    fig.suptitle("ISD mock convergence analysis", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _savefig(fig, "07_mock_convergence.png")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    total = time.perf_counter() - t_start
    print(f"\nAll figures saved to: {os.path.abspath(OUT_DIR)}")
    print(f"Total runtime: {total/60:.1f} min")

    # Print timing summary for RST table
    print("\nTiming summary (for documentation):")
    for m, t in timing.items():
        print(f"  {m:12s}: {t*1000:.2f} ms")
    print(f"\nRule-of-thumb N_mocks: {rot_n}  (convergence threshold |Δp| < 0.05)")


if __name__ == "__main__":
    main()
