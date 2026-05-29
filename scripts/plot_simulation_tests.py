#!/usr/bin/env python3
"""Generate all figures and tables for the simulation-tests documentation page.

Reads results from ``data/simulations/nside{NSIDE:04d}/results_summary.json``
(produced by ``scripts/run_simulation_tests.py``) and writes the following
outputs to ``docs/_static/results_simulation_tests/``:

Figures
-------
nz_comparison.png
    Redshift distribution n(z): Uchuu measured histogram vs GLASS generated.
templates_overview.png
    Mollweide projections of the 5 LSDR10 systematic maps used.
wtheta_recovery_grid.png
    3 × 3 grid (contamination level × scenario): truth / contaminated /
    recovered w(θ) for the GLASS mock (or Uchuu if GLASS absent).
wtheta_recovery_by_source.png
    Side-by-side: GLASS vs Uchuu recovery for the medium-combined config.
recovery_bias_heatmap.png
    Heatmap of mean |w_rec − w_true| / (|w_true| + ε) per method × config.
contamination_amplitude_scan.png
    Mean fractional bias in w(θ) vs contamination amplitude level, per
    scenario and method.

Tables (CSV)
------------
summary_table.csv
    One row per (source, level, scenario, method): n_gal, n_rand,
    mean_bias_contaminated, mean_bias_recovered, improvement_factor.

Usage
-----
python scripts/plot_simulation_tests.py \\
    [--nside 64] \\
    [--results-json data/simulations/nside0064/results_summary.json] \\
    [--syst-dir ~/data/legacysurvey/dr10/systematics/] \\
    [--uchuu-data ~/data/Uchuu/.../_DATA.fits] \\
    [--output-dir docs/_static/results_simulation_tests/nside0064]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parent.parent

_DEFAULT_SIMULATIONS_DIR = REPO / "data" / "simulations"
_DEFAULT_SYST_DIR = os.path.expanduser("~/data/legacysurvey/dr10/systematics/")
_DEFAULT_UCHUU_DATA = os.path.expanduser(
    "~/data/Uchuu/FullSky/mock_catalogues/"
    "MOCK_VLIM_ANY_10.65_Mstar_12.0_0.05_z_0.26_N_0923373/"
    "MOCK_VLIM_ANY_10.65_Mstar_12.0_0.05_z_0.26_N_0923373_DATA.fits"
)

# ── Shared style ───────────────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.6",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

METHOD_STYLES = {
    "OLS":        {"color": "C0", "ls": "-",   "marker": "o"},
    "ISD-1":      {"color": "C1", "ls": "--",  "marker": "s"},
    "ElasticNet": {"color": "C2", "ls": "-.",  "marker": "^"},
    "MCMC-add":   {"color": "C3", "ls": ":",   "marker": "D"},
    "MCMC-comb":  {"color": "C4", "ls": "--",  "marker": "v"},
}

LEVEL_ORDER = ["low", "medium", "high"]
SCENARIO_ORDER = ["additive", "multiplicative", "combined"]

LEVEL_LABELS = {"low": "Low  (|a|, |b| = 0.02)", "medium": "Medium  (|a|, |b| = 0.05)", "high": "High  (|a|, |b| = 0.10)"}
SCENARIO_LABELS = {"additive": "Additive", "multiplicative": "Multiplicative", "combined": "Combined"}


def _savefig(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    path = output_dir / name
    fig.savefig(path)
    plt.close(fig)
    try:
        rel = path.relative_to(REPO)
    except ValueError:
        rel = path
    print(f"  Saved {rel}")
    return path


# ── 1. n(z) comparison ────────────────────────────────────────────────────────


def plot_nz_comparison(output_dir: Path, uchuu_z_path: str | None) -> None:
    """Plot n(z) from Uchuu and GLASS alongside each other."""
    from sys_mapping.glass_mocks import generate_glass_fullsky_mock, measure_nz

    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Uchuu n(z)
    if uchuu_z_path and os.path.exists(uchuu_z_path):
        from astropy.io import fits as afits

        with afits.open(uchuu_z_path) as hdul:
            z_uchuu = np.asarray(hdul[1].data["redshift_S"], dtype=float)
        z_edges, nz_uchuu = measure_nz(z_uchuu, 0.05, 0.26, n_bins=20)
        dz = np.diff(z_edges)
        z_cen = 0.5 * (z_edges[:-1] + z_edges[1:])
        ax.bar(z_cen, nz_uchuu / nz_uchuu.sum() / dz, width=dz * 0.9,
               alpha=0.6, color="C3", label="Uchuu lightcone (data)")
        n_total = len(z_uchuu)
    else:
        rng = np.random.default_rng(0)
        z_ref = rng.uniform(0.05, 0.26, 50_000)
        z_edges, nz_uchuu = measure_nz(z_ref, 0.05, 0.26, n_bins=20)
        dz = np.diff(z_edges)
        z_cen = 0.5 * (z_edges[:-1] + z_edges[1:])
        ax.bar(z_cen, nz_uchuu / nz_uchuu.sum() / dz, width=dz * 0.9,
               alpha=0.6, color="C3", label="Uchuu lightcone (simulated n(z))")
        n_total = 923_373

    # GLASS n(z)
    glass_cat = generate_glass_fullsky_mock(
        nside=16, n_total=n_total, z_edges=z_edges, nz=nz_uchuu, seed=0
    )
    _, nz_glass = measure_nz(glass_cat["z"], 0.05, 0.26, n_bins=20)
    ax.step(z_edges[:-1], nz_glass / nz_glass.sum() / dz,
            where="post", color="C0", lw=1.5, label="GLASS mock")

    ax.set_xlabel("Redshift  z")
    ax.set_ylabel("n(z)  [normalised]")
    ax.set_title("Redshift distribution: Uchuu input vs GLASS output")
    ax.legend()
    ax.set_xlim(0.03, 0.28)
    fig.tight_layout()
    _savefig(fig, output_dir, "nz_comparison.png")


# ── 2. Template overview ───────────────────────────────────────────────────────


def plot_templates_overview(output_dir: Path, syst_dir: str, nside: int = 64) -> None:
    """Mollweide thumbnails of the 5 LSDR10 systematic maps."""
    from sys_mapping.simulation import DEFAULT_MAP_NAMES, _MAP_SPECS

    syst_path = Path(syst_dir).expanduser() / f"{nside:04d}"
    if not syst_path.exists():
        print(f"  WARNING: {syst_path} not found; skipping template overview")
        return

    from sys_mapping.maps import load_real_template

    n_maps = len(DEFAULT_MAP_NAMES)
    fig, axes = plt.subplots(1, n_maps, figsize=(3.5 * n_maps, 3.2),
                             subplot_kw={"projection": "mollweide"})

    for ax, name in zip(axes, DEFAULT_MAP_NAMES):
        fname_tmpl, column, zfill = _MAP_SPECS[name]
        nside_str = str(nside).zfill(zfill)
        fname = fname_tmpl.replace("{nside}", nside_str)
        fpath = syst_path / fname
        if not fpath.exists():
            ax.set_title(name.replace("_", "\n"), fontsize=7)
            continue

        tmpl, _ = load_real_template(fpath, column=column, nside=nside)
        # Healpy mollview writes to its own figure; we use a manual approach
        ax.axis("off")
        ax.set_title(name.replace("LS10_", "").replace("GAIA_", "GAIA\n").replace("_", " "),
                     fontsize=8)
        # Quick pixel scatter for overview
        n_pix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(n_pix))
        lon = np.degrees(phi) - 180.0  # [-180, 180)
        lat = 90.0 - np.degrees(theta)
        sc = ax.scatter(
            np.radians(lon), np.radians(lat),
            c=tmpl, s=0.1, cmap="RdBu_r", vmin=-2, vmax=2, rasterized=True,
        )

    fig.suptitle("LSDR10 systematic template maps (NSIDE = {})".format(nside), fontsize=9)
    fig.tight_layout()
    _savefig(fig, output_dir, "templates_overview.png")


# ── 3. w(θ) recovery grid ─────────────────────────────────────────────────────


def _result_key(results: list[dict], source: str, level: str, scenario: str) -> dict | None:
    for r in results:
        cfg = r.get("config", {})
        if isinstance(cfg, dict):
            lvl, scen = cfg.get("level"), cfg.get("scenario")
        else:
            lvl, scen = getattr(cfg, "level", None), getattr(cfg, "scenario", None)
        if r.get("source") == source and lvl == level and scen == scenario:
            return r
    return None


def _get_methods(results: list[dict]) -> list[str]:
    for r in results:
        methods = [k.replace("w_recovered_", "") for k in r if k.startswith("w_recovered_")]
        if methods:
            return methods
    return ["OLS", "ISD-1", "ElasticNet"]


def plot_wtheta_recovery_grid(
    results: list[dict], output_dir: Path, source: str = "glass"
) -> None:
    """3 × 3 grid: level (rows) × scenario (cols), w(θ) curves per method."""
    methods = _get_methods(results)
    fig, axes = plt.subplots(
        len(LEVEL_ORDER), len(SCENARIO_ORDER),
        figsize=(4.5 * len(SCENARIO_ORDER), 3.5 * len(LEVEL_ORDER)),
        sharey=False,
    )

    for row, level in enumerate(LEVEL_ORDER):
        for col, scenario in enumerate(SCENARIO_ORDER):
            ax = axes[row][col]
            r = _result_key(results, source, level, scenario)
            if r is None:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
                continue

            theta = np.asarray(r["theta"])
            w_true = np.asarray(r["w_true"])
            w_cont = np.asarray(r["w_contaminated"])
            denom = max(1e-3, float(np.max(np.abs(w_true))))

            ax.axhline(0, color="k", lw=1.5, ls="-", label="truth", zorder=5)
            ax.plot(theta, (w_cont - w_true) / denom,
                    color="0.6", lw=1.5, ls="--", label="contaminated")
            for method in methods:
                key = f"w_recovered_{method}"
                if key in r:
                    st = METHOD_STYLES.get(method, {"color": "C3", "ls": "-", "marker": "o"})
                    ax.plot(theta, (np.asarray(r[key]) - w_true) / denom,
                            color=st["color"], ls=st["ls"], lw=1.3, label=f"recovered ({method})")

            ax.set_ylim(-0.5, 0.5)
            ax.set_xscale("log")
            ax.set_xlabel("θ [deg]" if row == len(LEVEL_ORDER) - 1 else "")
            ax.set_ylabel(r"$(w - w_{\rm truth}) / |w_{\rm truth}|$" if col == 0 else "")
            if row == 0:
                ax.set_title(SCENARIO_LABELS[scenario], fontsize=10, fontweight="bold")
            if col == len(SCENARIO_ORDER) - 1:
                ax2 = ax.twinx()
                ax2.set_ylabel(LEVEL_LABELS[level].split("  ")[0],
                               fontsize=8, rotation=270, labelpad=14)
                ax2.set_yticks([])
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"w(θ) recovery — {source.upper()} mock | 5 LSDR10 templates",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    _savefig(fig, output_dir, f"wtheta_recovery_grid_{source}.png")


# ── 4. GLASS vs Uchuu comparison (medium-combined) ───────────────────────────


def plot_wtheta_by_source(results: list[dict], output_dir: Path) -> None:
    """Side-by-side: GLASS vs Uchuu, medium-combined config."""
    methods = _get_methods(results)
    sources = sorted({r.get("source", "unknown") for r in results})
    if not sources:
        return

    fig, axes = plt.subplots(1, len(sources), figsize=(5 * len(sources), 4), sharey=True)
    if len(sources) == 1:
        axes = [axes]

    for ax, source in zip(axes, sources):
        r = _result_key(results, source, "medium", "combined")
        if r is None:
            ax.set_title(source.upper())
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            continue

        theta = np.asarray(r["theta"])
        w_true = np.asarray(r["w_true"])
        w_cont = np.asarray(r["w_contaminated"])
        denom = max(1e-3, float(np.max(np.abs(w_true))))

        ax.axhline(0, color="k", lw=1.5, ls="-", label="truth", zorder=5)
        ax.plot(theta, (w_cont - w_true) / denom,
                color="0.6", lw=1.5, ls="--", label="contaminated")
        for method in methods:
            key = f"w_recovered_{method}"
            if key in r:
                st = METHOD_STYLES.get(method, {"color": "C3", "ls": "-"})
                ax.plot(theta, (np.asarray(r[key]) - w_true) / denom,
                        color=st["color"], ls=st["ls"], lw=1.3, label=f"{method}")

        ax.set_ylim(-0.5, 0.5)
        ax.set_xscale("log")
        ax.set_xlabel("θ [deg]")
        ax.set_ylabel(r"$(w - w_{\rm truth}) / |w_{\rm truth}|$")
        ax.set_title(f"{source.upper()} — medium contamination, combined scenario")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _savefig(fig, output_dir, "wtheta_recovery_by_source.png")


# ── 5. Recovery-bias heatmap ──────────────────────────────────────────────────


def _mean_frac_bias(w_rec: np.ndarray, w_true: np.ndarray) -> float:
    """Mean |Δw| / max|w_true| — bounded metric, uniform normalisation across scales."""
    w_true = np.asarray(w_true)
    scale = max(1e-3, float(np.max(np.abs(w_true))))
    return float(np.mean(np.abs(np.asarray(w_rec) - w_true) / scale))


def plot_recovery_bias_heatmap(results: list[dict], output_dir: Path) -> None:
    """Heatmap: mean |Δw/w| per method × (level-scenario) configuration."""
    methods = _get_methods(results)
    configs = [(lv, sc) for lv in LEVEL_ORDER for sc in SCENARIO_ORDER]
    col_labels = [f"{lv[:3]}/{sc[:3]}" for lv, sc in configs]

    # Collect biases separately for contaminated and each method
    cont_bias = []
    method_bias = {m: [] for m in methods}

    for lv, sc in configs:
        # Average over sources
        rows_for_config = [
            r for r in results
            if (r.get("config", {}) if isinstance(r.get("config"), dict) else {}).get("level") == lv
            or (hasattr(r.get("config"), "level") and r["config"].level == lv)
        ]
        # Simplify: just find first matching result regardless of source
        matching = []
        for r in results:
            cfg = r.get("config", {})
            rlv = cfg.get("level") if isinstance(cfg, dict) else getattr(cfg, "level", None)
            rsc = cfg.get("scenario") if isinstance(cfg, dict) else getattr(cfg, "scenario", None)
            if rlv == lv and rsc == sc:
                matching.append(r)

        if not matching:
            cont_bias.append(np.nan)
            for m in methods:
                method_bias[m].append(np.nan)
            continue

        cb = np.mean([_mean_frac_bias(r["w_contaminated"], r["w_true"]) for r in matching])
        cont_bias.append(float(cb))
        for m in methods:
            key = f"w_recovered_{m}"
            vals = [_mean_frac_bias(r[key], r["w_true"]) for r in matching if key in r]
            method_bias[m].append(float(np.mean(vals)) if vals else np.nan)

    n_rows = len(methods) + 1  # +1 for contaminated row
    matrix = np.array([cont_bias] + [method_bias[m] for m in methods])

    fig, ax = plt.subplots(figsize=(max(10, len(configs) * 0.9), 1 + n_rows * 0.5 + 0.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.5)

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(["contaminated"] + list(methods), fontsize=8)
    ax.set_title("Mean fractional bias  |w_rec − w_true| / |w_true|", fontsize=9)

    for i in range(n_rows):
        for j in range(len(configs)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=6,
                        color="white" if val > 0.25 else "black")

    plt.colorbar(im, ax=ax, label="mean fractional bias", shrink=0.8)
    # Vertical lines between level groups
    for k in range(1, 3):
        ax.axvline(k * len(SCENARIO_ORDER) - 0.5, color="white", lw=1.5)

    # Level group labels on top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    level_ticks = [len(SCENARIO_ORDER) / 2 - 0.5 + k * len(SCENARIO_ORDER) for k in range(3)]
    ax2.set_xticks(level_ticks)
    ax2.set_xticklabels([lv.upper() for lv in LEVEL_ORDER], fontsize=8, fontweight="bold")
    ax2.tick_params(top=False)

    fig.tight_layout()
    _savefig(fig, output_dir, "recovery_bias_heatmap.png")


# ── 6. Amplitude scan ─────────────────────────────────────────────────────────


def plot_amplitude_scan(results: list[dict], output_dir: Path) -> None:
    """Mean fractional bias vs amplitude level, per method and scenario."""
    methods = _get_methods(results)
    level_amps = {"low": 0.02, "medium": 0.05, "high": 0.10}
    x = [level_amps[lv] for lv in LEVEL_ORDER]

    fig, axes = plt.subplots(1, len(SCENARIO_ORDER),
                             figsize=(4.5 * len(SCENARIO_ORDER), 3.5), sharey=True)

    for ax, scenario in zip(axes, SCENARIO_ORDER):
        # Contamination bias
        cont_vals = []
        for lv in LEVEL_ORDER:
            matching = [
                r for r in results
                if (r.get("config", {}).get("level") if isinstance(r.get("config"), dict)
                    else getattr(r.get("config"), "level", None)) == lv
                and (r.get("config", {}).get("scenario") if isinstance(r.get("config"), dict)
                     else getattr(r.get("config"), "scenario", None)) == scenario
            ]
            if matching:
                cont_vals.append(np.mean([_mean_frac_bias(r["w_contaminated"], r["w_true"]) for r in matching]))
            else:
                cont_vals.append(np.nan)

        ax.plot(x, cont_vals, "k--", lw=2, marker="D", ms=6, label="contaminated", zorder=5)

        for method in methods:
            key = f"w_recovered_{method}"
            rec_vals = []
            for lv in LEVEL_ORDER:
                matching = [
                    r for r in results
                    if (r.get("config", {}).get("level") if isinstance(r.get("config"), dict)
                        else getattr(r.get("config"), "level", None)) == lv
                    and (r.get("config", {}).get("scenario") if isinstance(r.get("config"), dict)
                         else getattr(r.get("config"), "scenario", None)) == scenario
                    and key in r
                ]
                if matching:
                    rec_vals.append(np.mean([_mean_frac_bias(r[key], r["w_true"]) for r in matching]))
                else:
                    rec_vals.append(np.nan)

            st = METHOD_STYLES.get(method, {"color": "C3", "ls": "-", "marker": "o"})
            ax.plot(x, rec_vals, color=st["color"], ls=st["ls"],
                    marker=st["marker"], ms=5, lw=1.5, label=method)

        ax.set_xlabel("Contamination amplitude |a|, |b|")
        ax.set_ylabel("Mean fractional bias" if scenario == SCENARIO_ORDER[0] else "")
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.set_xticks(x)
        ax.set_xticklabels(["low\n(0.02)", "medium\n(0.05)", "high\n(0.10)"])
        if scenario == SCENARIO_ORDER[-1]:
            ax.legend(fontsize=8)

    fig.suptitle("w(θ) recovery bias vs contamination amplitude", fontsize=10, y=1.02)
    fig.tight_layout()
    _savefig(fig, output_dir, "contamination_amplitude_scan.png")


# ── 6b. Parameter recovery scatter ───────────────────────────────────────────

_LEVEL_COLORS = {"low": "C0", "medium": "C1", "high": "C2"}


def plot_parameter_recovery(results: list[dict], output_dir: Path) -> None:
    """Scatter plots of injected vs recovered aᵢ (and bᵢ) per method."""
    methods = _get_methods(results)
    if not methods:
        print("  No method results found; skipping parameter recovery plots.")
        return

    # Collect (true, hat, level) triples per method
    a_data: dict[str, list[tuple[float, float, str]]] = {m: [] for m in methods}
    b_data: dict[str, list[tuple[float, float, str]]] = {m: [] for m in methods}
    b_nonzero: set[str] = set()

    for r in results:
        cfg = r.get("config", {})
        level = cfg.get("level", "?") if isinstance(cfg, dict) else getattr(cfg, "level", "?")
        a_true = cfg.get("a_true", []) if isinstance(cfg, dict) else getattr(cfg, "a_true", [])
        b_true = cfg.get("b_true", []) if isinstance(cfg, dict) else getattr(cfg, "b_true", [])
        for m in methods:
            key = f"params_{m}"
            if key not in r:
                continue
            a_hat = r[key].get("a_hat", [])
            b_hat = r[key].get("b_hat", [])
            for at, ah in zip(a_true, a_hat):
                a_data[m].append((float(at), float(ah), level))
            for bt, bh in zip(b_true, b_hat):
                b_data[m].append((float(bt), float(bh), level))
                if abs(bh) > 1e-12:
                    b_nonzero.add(m)

    def _scatter_panel(ax: plt.Axes, pts: list[tuple[float, float, str]],
                       xlabel: str, ylabel: str, title: str, in_b_nonzero: bool) -> None:
        # For b-panels of additive-only methods: show a "not applicable" note
        # instead of data (b_hat ≡ 0 by design, 100 % relative error is misleading).
        if not in_b_nonzero and not xlabel.startswith("$a"):
            ax.text(0.5, 0.5, "b not estimated\n(additive-only method)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9,
                    color="0.5")
            ax.set_title(f"{title}\n(b ≡ 0 by design — not applicable)", fontsize=9)
            ax.set_axis_off()
            return
        if not pts:
            ax.set_visible(False)
            return
        xs_all = np.array([p[0] for p in pts])
        ys_all = np.array([p[1] for p in pts])
        lvs_all = np.array([p[2] for p in pts])

        # Relative error is undefined when the injected value is zero; skip those.
        valid = np.abs(xs_all) > 1e-9
        if not valid.any():
            ax.text(0.5, 0.5, "injected = 0\n(no relative error)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=8)
            ax.set_title(title, fontsize=9)
            return

        xs = xs_all[valid]
        ys_hat = ys_all[valid]
        lvs = lvs_all[valid]
        rel_err = np.abs(ys_hat - xs) / np.abs(xs)

        for lv, col in _LEVEL_COLORS.items():
            mask = lvs == lv
            if mask.any():
                ax.scatter(xs[mask], rel_err[mask], s=18, alpha=0.7, color=col, label=lv)

        xlim = float(np.abs(xs).max()) * 1.15 + 1e-9
        ax.axhline(0, color="k", lw=0.8, ls="--", label="perfect recovery")
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(0, max(float(np.percentile(rel_err, 95)) * 1.5, 0.5))

        mean_rel = float(np.mean(rel_err))
        std_rel = float(np.std(rel_err))
        ax.text(0.05, 0.95, f"mean={mean_rel:.3f}\nstd={std_rel:.3f}",
                transform=ax.transAxes, va="top", fontsize=7, family="monospace")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

    n_methods = len(methods)

    _YLABEL_A = r"$|\hat{a}_i - a_i| \,/\, |a_i|$"
    _YLABEL_B = r"$|\hat{b}_i - b_i| \,/\, |b_i|$"

    # Figure 1: aᵢ only
    fig_a, axes_a = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4), squeeze=False)
    for ax, m in zip(axes_a[0], methods):
        _scatter_panel(ax, a_data[m], r"$a_i$ injected", _YLABEL_A, m, True)
    fig_a.suptitle("Additive parameter recovery — relative error per method", fontsize=10, y=1.02)
    fig_a.tight_layout()
    _savefig(fig_a, output_dir, "parameter_recovery_a.png")

    # Figure 2: bᵢ — only show methods that actually estimate b.
    # Additive-only methods (OLS, ElasticNet, ISD) always return b_hat=0;
    # showing them here at 100% relative error is misleading.
    b_methods = [m for m in methods if m in b_nonzero]
    if not b_methods:
        fig_b, ax_b = plt.subplots(1, 1, figsize=(5, 4))
        ax_b.text(0.5, 0.5,
                  "No method estimates b.\n"
                  "OLS / ElasticNet / ISD return b ≡ 0 by design.\n"
                  "Use MCMC-comb to recover multiplicative amplitudes.",
                  transform=ax_b.transAxes, ha="center", va="center",
                  fontsize=9, color="0.4")
        ax_b.set_axis_off()
        fig_b.suptitle(r"Multiplicative parameter recovery — b ≡ 0 for all methods (not applicable)",
                       fontsize=10, y=1.02)
    else:
        fig_b, axes_b = plt.subplots(1, len(b_methods), figsize=(4 * len(b_methods), 4), squeeze=False)
        for ax, m in zip(axes_b[0], b_methods):
            _scatter_panel(ax, b_data[m], r"$b_i$ injected", _YLABEL_B, m, True)
        fig_b.suptitle(r"Multiplicative parameter recovery — relative error per method",
                       fontsize=10, y=1.02)
    fig_b.tight_layout()
    _savefig(fig_b, output_dir, "parameter_recovery_b.png")

    # Figure 3: combined two-row (aᵢ top, bᵢ bottom)
    fig_ab, axes_ab = plt.subplots(2, n_methods, figsize=(4 * n_methods, 8), squeeze=False)
    for ax, m in zip(axes_ab[0], methods):
        _scatter_panel(ax, a_data[m], r"$a_i$ injected", _YLABEL_A, m, True)
    for ax, m in zip(axes_ab[1], methods):
        _scatter_panel(ax, b_data[m], r"$b_i$ injected", _YLABEL_B, m, m in b_nonzero)
    fig_ab.suptitle(r"Contamination parameter recovery — relative error ($a_i$ top, $b_i$ bottom)",
                    fontsize=10, y=1.01)
    fig_ab.tight_layout()
    _savefig(fig_ab, output_dir, "parameter_recovery_ab.png")


# ── 7. Summary CSV table ──────────────────────────────────────────────────────


def write_summary_csv(results: list[dict], output_dir: Path) -> None:
    """Write summary_table.csv with one row per (source, level, scenario, method)."""
    import csv

    path = output_dir / "summary_table.csv"
    methods = _get_methods(results)

    rows = [["source", "level", "scenario", "n_gal", "n_rand",
             "bias_contaminated"] + [f"bias_{m}" for m in methods]
            + [f"improvement_{m}" for m in methods]]

    for r in results:
        cfg = r.get("config", {})
        source = r.get("source", "?")
        level = cfg.get("level") if isinstance(cfg, dict) else getattr(cfg, "level", "?")
        scenario = cfg.get("scenario") if isinstance(cfg, dict) else getattr(cfg, "scenario", "?")
        n_gal = r.get("n_gal", "?")
        n_rand = r.get("n_rand", "?")

        w_true = np.asarray(r["w_true"])
        w_cont = np.asarray(r["w_contaminated"])
        bias_cont = _mean_frac_bias(w_cont, w_true)

        row = [source, level, scenario, n_gal, n_rand, f"{bias_cont:.4f}"]
        for m in methods:
            key = f"w_recovered_{m}"
            if key in r:
                bm = _mean_frac_bias(r[key], w_true)
                row.append(f"{bm:.4f}")
            else:
                row.append("N/A")
        for m in methods:
            key = f"w_recovered_{m}"
            if key in r:
                bm = _mean_frac_bias(r[key], w_true)
                imp = bias_cont / (bm + 1e-9)
                row.append(f"{imp:.2f}")
            else:
                row.append("N/A")

        rows.append(row)

    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    try:
        rel = path.relative_to(REPO)
    except ValueError:
        rel = path
    print(f"  Saved {rel}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-json", default=None,
                   help="Path to results_summary.json (default: data/simulations/nside{NSIDE:04d}/results_summary.json)")
    p.add_argument("--syst-dir", default=_DEFAULT_SYST_DIR)
    p.add_argument("--uchuu-data", default=_DEFAULT_UCHUU_DATA)
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: docs/_static/results_simulation_tests/nside{NSIDE:04d}/)")
    p.add_argument("--no-templates", action="store_true", help="Skip template overview (slow for large NSIDE)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output_dir is None:
        output_dir = REPO / "docs" / "_static" / "results_simulation_tests" / f"nside{args.nside:04d}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve results JSON: default to the nside-specific subdirectory
    if args.results_json is None:
        results_path = _DEFAULT_SIMULATIONS_DIR / f"nside{args.nside:04d}" / "results_summary.json"
    else:
        results_path = Path(args.results_json)
    if not results_path.exists():
        print(f"WARNING: {results_path} not found.")
        print("Run  scripts/run_simulation_tests.py  first to generate results.")
        print("Generating n(z) and template figures only (no w(θ) results).")
        results = []
    else:
        with open(results_path) as f:
            results = json.load(f)
        try:
            rel = results_path.relative_to(REPO)
        except ValueError:
            rel = results_path
        print(f"Loaded {len(results)} result entries from {rel}")

    # 1. n(z) comparison
    print("\n[1/6] n(z) comparison")
    plot_nz_comparison(output_dir, args.uchuu_data)

    # 2. Template overview
    if not args.no_templates:
        print("\n[2/6] Template overview (Mollweide)")
        plot_templates_overview(output_dir, args.syst_dir, nside=min(args.nside, 64))
    else:
        print("\n[2/6] Skipping template overview (--no-templates)")

    if not results:
        print("\nNo results to plot — skipping w(θ) figures. Done.")
        return 0

    # 3. w(θ) recovery grids per source
    sources = sorted({r.get("source", "unknown") for r in results})
    print(f"\n[3/6] w(θ) recovery grids ({sources})")
    for source in sources:
        plot_wtheta_recovery_grid(results, output_dir, source=source)

    # 4. GLASS vs Uchuu side-by-side
    print("\n[4/6] GLASS vs Uchuu comparison (medium-combined)")
    if len(sources) >= 2:
        plot_wtheta_by_source(results, output_dir)
    else:
        print(f"  Only one source ({sources}); skipping side-by-side comparison")

    # 5. Bias heatmap
    print("\n[5/6] Recovery bias heatmap")
    plot_recovery_bias_heatmap(results, output_dir)

    # 6. Amplitude scan
    print("\n[6/8] Contamination amplitude scan")
    plot_amplitude_scan(results, output_dir)

    # 6b. Parameter recovery scatter
    print("\n[7/8] Parameter recovery scatter (aᵢ and bᵢ)")
    plot_parameter_recovery(results, output_dir)

    # 7. Summary CSV
    print("\n[8/8] Summary CSV table")
    write_summary_csv(results, output_dir)

    try:
        out_rel = output_dir.relative_to(REPO)
    except ValueError:
        out_rel = output_dir
    print(f"\nAll outputs written to {out_rel}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
