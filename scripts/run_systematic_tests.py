"""Systematic test matrix: all methods × all template configurations.

Runs a two-tier battery of tests to characterise how well each implemented
method recovers contamination parameters as a function of the number and type
of systematic templates.

Template set (7 synthetic maps)
--------------------------------
synth_0 : family 0, seed 0  — large-scale (C_ℓ ∝ exp(-ℓ/500))
synth_1 : family 1, seed 0  — intermediate-scale (C_ℓ ∝ exp(-(ℓ/250)²))
synth_2 : family 2, seed 0  — power-law ℓ^{-2}
synth_3 : family 3, seed 0  — power-law ℓ^{-1}
synth_4 : family 4, seed 0  — white noise
synth_5 : family 0, seed 5  — second large-scale map (GAIA stand-in)
synth_6 : family 2, seed 5  — second power-law map (depth stand-in)

Tier 1 — single contamination type
------------------------------------
Additive-only (b_i = 0):
  * Single template: synth_0 alone, …, synth_6 alone  (7 configs)
  * Multi-template:  synth_{0..1}, synth_{0..2}, …, synth_{0..6}  (6 configs)

Multiplicative-only (a_i = 0):
  * Same 13 configurations

Tier 2 — mixed additive + multiplicative
-----------------------------------------
Fixed 7 templates (synth_0..6); vary how many are multiplicative vs additive:
  n_mult = 1 : synth_0 multiplicative, synth_1..6 additive
  n_mult = 2 : synth_0..1 multiplicative, synth_2..6 additive
  …
  n_mult = 6 : synth_0..5 multiplicative, synth_6 additive

Methods
-------
  OLS, ElasticNet, ISD-1, ISD-3, MCMC-additive, MCMC-combined

All six run through ``sm.run_decontamination``.  MCMC-add is the exact analytic
posterior and MCMC-comb BlackJAX NUTS (``sampler="auto"``).  The ISD stopping
rule is calibrated per template and polynomial order on uncontaminated mocks
drawn by ``make_mock`` with every amplitude zero.

Output per configuration
------------------------
  * PNG histogram of (1+δ_corr)/(1+δ_true) per method
  * Row in master CSV: config_id, n_templates, tier, contamination_type,
                       method, mean_ratio, std_ratio, med_ratio, iqr_ratio

Usage
-----
    conda activate sys_map
    python scripts/run_systematic_tests.py --output-dir results/systematic_tests/

    # Fast smoke-test (OLS + MCMC-comb only, NSIDE=16):
    python scripts/run_systematic_tests.py --nside 16 --fast --isd-n-mocks 0 \\
        --output-dir /tmp/sys_test/
"""

from __future__ import annotations

import argparse
import csv
import time
import warnings
from pathlib import Path

import healpy as hp
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys_mapping as sm
from sys_mapping.contamination import apply_contamination
from sys_mapping.model_selection import likelihood_ratio_test

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Constants ──────────────────────────────────────────────────────────────

AMPLITUDE = 0.10          # injected |a_i| and |b_i| per active template
SIGMA_G = 0.5             # lognormal field width
N_MEAN = 50               # mean galaxies per pixel

ALL_METHODS = ["OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"]
FAST_METHODS = ["OLS", "MCMC-comb"]

METHOD_COLORS = {
    "OLS":         "#1f77b4",
    "ElasticNet":  "#ff7f0e",
    "ISD-1":       "#2ca02c",
    "ISD-3":       "#d62728",
    "MCMC-add":    "#9467bd",
    "MCMC-comb":   "#8c564b",
}

TEMPLATE_NAMES = [f"synth_{i}" for i in range(7)]


# ── Template generation ────────────────────────────────────────────────────

def generate_7_templates(nside: int, seed: int = 0) -> np.ndarray:
    """Generate 7 synthetic systematic templates at the given NSIDE.

    Returns shape (7, 12*nside²).  The first 5 use the standard families
    (0–4); synth_5 = family 0 with a different seed; synth_6 = family 2
    with a different seed.
    """
    rng_seeds = [seed, seed, seed, seed, seed, seed + 5, seed + 5]
    families  = [0,    1,    2,    3,    4,    0,         2       ]
    maps = np.stack([
        sm.generate_systematic_map(nside, f, seed=s)
        for f, s in zip(families, rng_seeds)
    ])
    return maps


# ── Mock generation ────────────────────────────────────────────────────────

def _lognormal_delta(nside: int, seed: int) -> np.ndarray:
    """Draw a lognormal galaxy overdensity field (full sky)."""
    np.random.seed(seed)
    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1, dtype=float)
    cl_G = (ell + 1.0) ** (-2)
    cl_G[0] = 0.0
    cl_G *= SIGMA_G**2 / np.sum((2 * ell + 1) / (4 * np.pi) * cl_G)
    G = hp.synfast(cl_G, nside=nside, lmax=lmax)
    return np.exp(G - 0.5 * SIGMA_G**2) - 1.0


def _galactic_mask(nside: int) -> np.ndarray:
    """Boolean mask: True for |b_gal| > 20° (~66% of sky)."""
    _, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    return np.abs(dec) > 20.0


def make_mock(nside: int, templates: np.ndarray, footprint: np.ndarray,
              a_true: np.ndarray, b_true: np.ndarray, seed: int = 0):
    """Generate Poisson-sampled galaxy and random count maps.

    Returns
    -------
    gal_counts : (n_pix,) int array
    rand_counts : (n_pix,) int array
    delta_true : (n_pix,) float array  — underlying lognormal field (full sky)
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)          # healpy synfast uses numpy random

    delta_true = _lognormal_delta(nside, seed)

    delta_cont = np.asarray(
        apply_contamination(
            jnp.asarray(delta_true),
            jnp.asarray(templates),
            jnp.asarray(a_true),
            jnp.asarray(b_true),
        )
    )

    lam = np.maximum(N_MEAN * (1.0 + delta_cont), 0.0) * footprint.astype(float)
    gal_counts = rng.poisson(lam).astype(float)
    rand_counts = (footprint.astype(float) * N_MEAN * 8.0)

    return gal_counts, rand_counts, delta_true


# ── ISD calibration ──────────────────────────────────────────────────────

def calibrate_isd(nside: int, templates: np.ndarray, footprint: np.ndarray,
                  n_mocks: int, seed: int, orders=(1, 3)) -> dict:
    """68th percentile per template of the ISD Delta chi^2 on uncontaminated mocks.

    The mocks are ``make_mock`` with every amplitude zero, seeds distinct from the
    contaminated run.  The statistic is the first-step marginal fit of
    ``iterative_systematics_decontamination`` (10 quantile bins).  A marginal fit
    involves one template at a time, so one calibration over all seven templates
    serves every configuration.  Returns ``{"ISD-1": (7,), "ISD-3": (7,)}``.
    """
    from sys_mapping.diagnostics import isd_marginal_fit

    n_sys = templates.shape[0]
    dchi2 = {order: [] for order in orders}
    for k in range(n_mocks):
        gal, ran, _ = make_mock(nside, templates, footprint, np.zeros(n_sys),
                                np.zeros(n_sys), seed=seed + 10_000 + k)
        delta_g, good = sm.compute_overdensity(gal, ran)
        delta_t = sm.assign_template_values(templates, good)
        for order in orders:
            dchi2[order].append(isd_marginal_fit(delta_g, delta_t, poly_order=order)[0])
    return {f"ISD-{order}": np.percentile(np.array(v), 68, axis=0)
            for order, v in dchi2.items()}


# ── Corrected overdensity per method ─────────────────────────────────────

def _run_methods_for_config(
    delta_g: np.ndarray,
    delta_t: np.ndarray,
    seed: int, methods: list[str],
    isd_chi2_68: dict | None = None,
    mcmc_kw: dict | None = None,
) -> dict:
    """Run all requested methods via run_decontamination(); return result dict."""
    results = {}
    for method in methods:
        kw = {"seed": seed}
        if method.startswith("ISD") and isd_chi2_68 is not None:
            kw["isd_chi2_68"] = isd_chi2_68[method]
        if method.startswith("MCMC"):
            kw.update(mcmc_kw or {})
        try:
            results[method] = sm.run_decontamination(method, delta_g, delta_t, **kw)
        except Exception as exc:
            warnings.warn(f"Method {method} failed: {exc}")
    return results


# ── Ratio statistics ──────────────────────────────────────────────────────

def ratio_stats(ratio: np.ndarray) -> dict:
    """Summary statistics for the (1+δ_corr)/(1+δ_true) ratio array."""
    q25, q75 = np.percentile(ratio, [25, 75])
    return {
        "mean":   float(np.mean(ratio)),
        "std":    float(np.std(ratio)),
        "median": float(np.median(ratio)),
        "iqr":    float(q75 - q25),
    }


# ── Plot histogram per configuration ─────────────────────────────────────

def plot_ratio_histogram(ratios_per_method: dict, config_label: str,
                         outpath: Path):
    """Plot overlaid histograms of (1+δ_corr)/(1+δ_true) for each method."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, ratio in ratios_per_method.items():
        color = METHOD_COLORS.get(method, "gray")
        ax.hist(ratio, bins=60, range=(0.5, 1.5), density=True,
                histtype="step", lw=1.5, label=method, color=color, alpha=0.85)
    ax.axvline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"$(1+\delta_g^{\rm corr})\,/\,(1+\delta_g^{\rm true})$",
                  fontsize=10)
    ax.set_ylabel("Probability density", fontsize=9)
    ax.set_title(config_label, fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(0.5, 1.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close()


# ── One full configuration run ────────────────────────────────────────────

def run_config(config_id: str, active_templates: np.ndarray,
               active_indices: list[int], all_templates: np.ndarray,
               nside: int, footprint: np.ndarray,
               contamination_type: str,
               methods: list[str], seed: int,
               outdir: Path, isd_chi2_68: dict | None = None,
               mcmc_kw: dict | None = None) -> list[dict]:
    """Run a single test configuration; return a list of CSV rows."""
    n_sys = active_templates.shape[0]
    active_names = [TEMPLATE_NAMES[i] for i in active_indices]
    n_tmpl = len(active_indices)

    # Injected parameters
    if contamination_type == "additive":
        a_true = np.full(n_sys, AMPLITUDE)
        b_true = np.zeros(n_sys)
    elif contamination_type == "multiplicative":
        a_true = np.zeros(n_sys)
        b_true = np.full(n_sys, AMPLITUDE)
    else:  # "combined_Nmult"
        n_mult = int(contamination_type.split("_")[1].replace("mult", ""))
        a_true = np.full(n_sys, AMPLITUDE)
        b_true = np.zeros(n_sys)
        b_true[:n_mult] = AMPLITUDE

    # Generate mock
    gal_counts, rand_counts, delta_true = make_mock(
        nside, active_templates, footprint, a_true, b_true, seed=seed
    )

    # Compute overdensity
    delta_g, good_pix = sm.compute_overdensity(gal_counts, rand_counts)
    delta_t = sm.assign_template_values(active_templates, good_pix)

    # True overdensity at good pixels (for ratio denominator)
    delta_true_good = delta_true[good_pix]

    # Run all requested methods via the unified interface
    active_c68 = (None if isd_chi2_68 is None else
                  {k: v[active_indices] for k, v in isd_chi2_68.items()})
    method_results = _run_methods_for_config(
        delta_g, delta_t, seed, methods, isd_chi2_68=active_c68, mcmc_kw=mcmc_kw,
    )

    # Compute ratios and stats
    rows = []
    ratios_for_plot = {}

    for method, res in method_results.items():
        if method in ("ISD-1", "ISD-3") and res.get("isd_stopped_on") == "max_steps":
            warnings.warn(f"{method} stopped on its step cap ({res['n_iterations']} steps)")

        w = res["weights"]
        if method == "MCMC-comb" and res.get("a_hat") is not None:
            _denom = np.maximum(1.0 + np.asarray(res["b_hat"]) @ delta_t, 1e-6)
            delta_corr = (delta_g - np.asarray(res["a_hat"]) @ delta_t) / _denom
        else:
            delta_corr = w * (1.0 + delta_g) - 1.0
        ratio = (1.0 + delta_corr) / (1.0 + delta_true_good)

        # Remove extreme outliers for statistics (keep 1st-99th percentile)
        lo, hi = np.percentile(ratio, [0.5, 99.5])
        ratio_clip = ratio[(ratio >= lo) & (ratio <= hi)]

        stats = ratio_stats(ratio_clip)
        ratios_for_plot[method] = ratio_clip

        rows.append({
            "config_id":          config_id,
            "n_templates":        n_tmpl,
            "active_templates":   "+".join(active_names),
            "contamination_type": contamination_type,
            "method":             method,
            "mean_ratio":         stats["mean"],
            "std_ratio":          stats["std"],
            "median_ratio":       stats["median"],
            "iqr_ratio":          stats["iqr"],
            "time_s":             res["elapsed_s"],
            "n_iter":             "" if res.get("n_iterations") is None else res["n_iterations"],
            "isd_stopped_on":     res.get("isd_stopped_on") or "",
            "rms_a_bias":         float(np.sqrt(np.mean((np.asarray(res["a_hat"]) - a_true) ** 2))),
        })

    # Plot
    label = (f"{config_id} | {contamination_type} | "
             f"n_tmpl={n_tmpl} | {'+'.join(active_names)}")
    plot_path = outdir / "histograms" / f"{config_id}.png"
    plot_ratio_histogram(ratios_for_plot, label, plot_path)

    return rows


# ── Master summary table plot ─────────────────────────────────────────────

def plot_summary_table(rows: list[dict], methods: list[str], outdir: Path):
    """One figure per (tier, contamination_type): std_ratio vs n_templates."""
    import pandas as pd
    df = pd.DataFrame(rows)

    for ctype in df["contamination_type"].unique():
        sub = df[df["contamination_type"] == ctype]
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in methods:
            m = sub[sub["method"] == method].sort_values("n_templates")
            if m.empty:
                continue
            ax.plot(m["n_templates"], m["std_ratio"],
                    marker="o", label=method,
                    color=METHOD_COLORS.get(method, "gray"))
        ax.set_xlabel("Number of templates", fontsize=10)
        ax.set_ylabel(r"$\sigma[(1+\delta^{\rm corr})\,/\,(1+\delta^{\rm true})]$",
                      fontsize=10)
        ax.set_title(f"Correction quality — {ctype}", fontsize=11)
        ax.legend(fontsize=8)
        ax.set_xticks(range(1, 8))
        plt.tight_layout()
        suf = ctype.replace(" ", "_").replace("+", "and")
        plt.savefig(outdir / f"summary_{suf}.png", dpi=120, bbox_inches="tight")
        plt.close()


# ── Timing summary plot ───────────────────────────────────────────────────

def plot_timing_table(rows: list[dict], methods: list[str], outdir: Path, nside: int = 32):
    """Wall-clock time per method as a function of n_templates."""
    import pandas as pd
    df = pd.DataFrame(rows)
    if "time_s" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        m = df[df["method"] == method].groupby("n_templates")["time_s"].mean().reset_index()
        if m.empty:
            continue
        ax.plot(m["n_templates"], m["time_s"], marker="o", label=method,
                color=METHOD_COLORS.get(method, "gray"))
    ax.set_xlabel("Number of templates", fontsize=10)
    ax.set_ylabel("Wall-clock time (s)", fontsize=10)
    ax.set_title(f"Compute time per method vs. number of templates (NSIDE={nside})", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(1, 8))
    plt.tight_layout()
    plt.savefig(outdir / "timing_vs_ntemplates.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Bar chart of mean time per method across all configs
    mean_time = df.groupby("method")["time_s"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [METHOD_COLORS.get(m, "gray") for m in mean_time.index]
    ax.barh(mean_time.index, mean_time.values, color=colors)
    ax.set_xlabel("Mean wall-clock time (s)", fontsize=10)
    ax.set_title(f"Mean compute time per method (all configs, NSIDE={nside})", fontsize=11)
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(outdir / "timing_mean_per_method.png", dpi=120, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Systematic test matrix: all methods × all configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--sampler", default="auto",
                        choices=["auto", "analytic", "nuts", "emcee"],
                        help="MCMC backend of run_decontamination: auto is the analytic "
                             "posterior for MCMC-add and NUTS for MCMC-comb.")
    parser.add_argument("--nuts-warmup", type=int, default=1000,
                        help="NUTS window-adaptation steps (MCMC-comb).")
    parser.add_argument("--nuts-samples", type=int, default=1000,
                        help="NUTS draws per chain (MCMC-comb).")
    parser.add_argument("--n-walkers", type=int, default=64, help="emcee only (--sampler emcee).")
    parser.add_argument("--n-steps",   type=int, default=200, help="emcee only (--sampler emcee).")
    parser.add_argument("--n-burn",    type=int, default=50, help="emcee only (--sampler emcee).")
    parser.add_argument("--isd-n-mocks", type=int, default=50,
                        help="Uncontaminated mocks calibrating the ISD stopping rule "
                             "(0 leaves it uncalibrated).")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--fast", action="store_true",
                        help="Run only OLS and MCMC-comb (skips ElasticNet/ISD).")
    parser.add_argument("--tier2-only", action="store_true",
                        help="Skip Tier 1 and run only Tier 2 (mixed contamination, 7 templates).")
    parser.add_argument("--output-dir", default="results/systematic_tests/")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    (outdir / "histograms").mkdir(parents=True, exist_ok=True)

    nside = args.nside
    methods = FAST_METHODS if args.fast else ALL_METHODS

    print(f"NSIDE={nside}, methods={methods}")
    print(f"Output: {outdir}")

    # Generate templates
    templates_all = generate_7_templates(nside, seed=0)  # (7, n_pix)
    footprint = _galactic_mask(nside)
    print(f"Footprint: {footprint.sum():,} pixels "
          f"({footprint.mean()*100:.1f}% of sky)")

    mcmc_kw = dict(sampler=args.sampler, nuts_n_warmup=args.nuts_warmup,
                   nuts_n_samples=args.nuts_samples, n_walkers=args.n_walkers,
                   n_steps=args.n_steps, n_burn=args.n_burn)
    isd_chi2_68 = None
    if args.isd_n_mocks > 0 and any(m.startswith("ISD") for m in methods):
        print(f"Calibrating the ISD stopping rule on {args.isd_n_mocks} uncontaminated mocks ...")
        isd_chi2_68 = calibrate_isd(nside, templates_all, footprint,
                                    args.isd_n_mocks, args.seed)
        for name, c68 in isd_chi2_68.items():
            print(f"  {name} chi2_68 = {np.array2string(c68, precision=2)}")
        import json
        (outdir / "isd_chi2_68.json").write_text(json.dumps(
            {"template_names": TEMPLATE_NAMES, "n_mocks": args.isd_n_mocks,
             **{k: v.tolist() for k, v in isd_chi2_68.items()}}, indent=2))

    all_rows = []

    # ── Tier 1: single contamination type ─────────────────────────────────
    if args.tier2_only:
        print("Skipping Tier 1 (--tier2-only).")
    for ctype in ("additive", "multiplicative") if not args.tier2_only else ():
        print(f"\n=== Tier 1 — {ctype} ===")

        # Single-template tests
        for i in range(7):
            config_id = f"T1_{ctype[:3]}_s{i}"
            print(f"  {config_id} ...", end=" ", flush=True)
            rows = run_config(
                config_id,
                active_templates=templates_all[[i]],
                active_indices=[i],
                all_templates=templates_all,
                nside=nside, footprint=footprint,
                contamination_type=ctype,
                methods=methods, seed=args.seed, outdir=outdir,
                isd_chi2_68=isd_chi2_68, mcmc_kw=mcmc_kw,
            )
            all_rows.extend(rows)
            std_vals = {r["method"]: f"{r['std_ratio']:.3f}" for r in rows}
            print("std:", std_vals)

        # Multi-template tests (cumulative: synth_0..k-1, k=2..7)
        for k in range(2, 8):
            config_id = f"T1_{ctype[:3]}_m{k}"
            print(f"  {config_id} ...", end=" ", flush=True)
            rows = run_config(
                config_id,
                active_templates=templates_all[:k],
                active_indices=list(range(k)),
                all_templates=templates_all,
                nside=nside, footprint=footprint,
                contamination_type=ctype,
                methods=methods, seed=args.seed, outdir=outdir,
                isd_chi2_68=isd_chi2_68, mcmc_kw=mcmc_kw,
            )
            all_rows.extend(rows)
            std_vals = {r["method"]: f"{r['std_ratio']:.3f}" for r in rows}
            print("std:", std_vals)

    # ── Tier 2: mixed additive + multiplicative ────────────────────────────
    print("\n=== Tier 2 — combined (additive + multiplicative) ===")
    for n_mult in range(1, 7):
        ctype = f"combined_{n_mult}mult"
        config_id = f"T2_comb_{n_mult}m"
        print(f"  {config_id} ({n_mult} mult + {7-n_mult} add) ...",
              end=" ", flush=True)
        rows = run_config(
            config_id,
            active_templates=templates_all,
            active_indices=list(range(7)),
            all_templates=templates_all,
            nside=nside, footprint=footprint,
            contamination_type=ctype,
            methods=methods, seed=args.seed, outdir=outdir,
            isd_chi2_68=isd_chi2_68, mcmc_kw=mcmc_kw,
        )
        all_rows.extend(rows)
        std_vals = {r["method"]: f"{r['std_ratio']:.3f}" for r in rows}
        print("std:", std_vals)

    # ── Write CSV ──────────────────────────────────────────────────────────
    csv_path = outdir / "systematic_test_summary.csv"
    fieldnames = [
        "config_id", "n_templates", "active_templates", "contamination_type",
        "method", "mean_ratio", "std_ratio", "median_ratio", "iqr_ratio",
        "time_s", "n_iter", "isd_stopped_on", "rms_a_bias",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV written to {csv_path}")

    # ── Summary plots ──────────────────────────────────────────────────────
    plot_summary_table(all_rows, methods, outdir)

    # ── Console summary ────────────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(all_rows)
    print("\n=== Mean std_ratio per method (lower = better) ===")
    print(df.groupby("method")["std_ratio"].agg(["mean", "min", "max"]).round(4))

    print("\n=== Mean compute time per method (seconds) ===")
    timing = df.groupby("method")["time_s"].agg(["mean", "min", "max"]).round(3)
    print(timing.sort_values("mean"))

    # ── Timing vs n_templates plots ────────────────────────────────────────
    plot_timing_table(all_rows, methods, outdir, nside=nside)

    print("\nDone.")


if __name__ == "__main__":
    main()
