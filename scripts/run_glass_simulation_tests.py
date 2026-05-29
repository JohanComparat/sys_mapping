#!/usr/bin/env python3
"""End-to-end validation: GLASS mocks with LSDR10 systematic contamination.

Phases
------
1. Generate a GLASS full-sky mock with a synthetic BGS-like n(z).
2. Load LSDR10 systematic maps.
3. For each of 9 contamination configurations (3 levels × 3 scenarios):
   a. Inject systematics → per-galaxy WEIGHT_CONT.
   b. Save contaminated catalog to FITS.
   c. Run decontamination methods (OLS, ISD-1, ElasticNet) and compute w(θ).
4. Save results summary to JSON.
5. Plot w(θ) recovery curves.

Each run is isolated in its own subdirectory keyed by NSIDE and galaxy count
so that runs with different parameters never overwrite each other::

    data/simulations_glass/nside0064_N1000000/
    data/simulations_glass/nside0064_N2000000/

Usage
-----
python scripts/run_glass_simulation_tests.py \\
    [--nside 64] \\
    [--n-glass 1000000] \\
    [--methods OLS ISD-1 ElasticNet] \\
    [--output-dir data/simulations_glass] \\
    [--syst-dir ~/data/legacysurvey/dr10/systematics/] \\
    [--min-sep 0.1] [--max-sep 10.0] [--nbins 10] \\
    [--seed 0] \\
    [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_DEFAULT_SYST_DIR = os.path.expanduser("~/data/legacysurvey/dr10/systematics/")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nside", type=int, default=64, help="HEALPix NSIDE for maps (default 64)")
    p.add_argument("--n-glass", type=int, default=1_000_000, help="Target galaxy count for GLASS mock")
    p.add_argument("--methods", nargs="+", default=["OLS", "ISD-1", "ElasticNet"],
                   choices=["OLS", "ISD-1", "ElasticNet", "MCMC-add", "MCMC-comb"])
    p.add_argument("--output-dir", default="data/simulations_glass", help="Output directory")
    p.add_argument("--syst-dir", default=_DEFAULT_SYST_DIR)
    p.add_argument("--min-sep", type=float, default=0.1, help="Min angular separation (degrees)")
    p.add_argument("--max-sep", type=float, default=10.0, help="Max angular separation (degrees)")
    p.add_argument("--nbins", type=int, default=10, help="Number of angular bins")
    p.add_argument("--seed", type=int, default=0, help="Master random seed")
    p.add_argument("--dry-run", action="store_true", help="Validate setup only, no computation")
    return p.parse_args()


def _print_step(step: int, msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Step {step}: {msg}")
    print(f"{'='*60}")


def _save_results(results: list[dict], output_dir: Path) -> None:
    """Serialise w(θ) results to JSON (numpy arrays → lists)."""
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    serializable = []
    for r in results:
        entry = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                entry[k] = v.tolist()
            elif hasattr(v, "level"):  # ContaminationConfig
                entry[k] = {
                    "level": v.level,
                    "scenario": v.scenario,
                    "a_true": v.a_true.tolist(),
                    "b_true": v.b_true.tolist(),
                }
            elif isinstance(v, dict):
                entry[k] = {kk: _convert(vv) for kk, vv in v.items()}
            else:
                entry[k] = _convert(v)
        serializable.append(entry)

    out_path = output_dir / "results_summary.json"
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"  Saved results to {out_path}")


def _plot_recovery(results: list[dict], output_dir: Path, methods: list[str]) -> None:
    """Plot w(θ) recovery for each configuration."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    for r in results:
        cfg = r.get("config", {})
        if hasattr(cfg, "level"):
            level, scenario = cfg.level, cfg.scenario
        else:
            level = cfg.get("level", "?")
            scenario = cfg.get("scenario", "?")

        fig, ax = plt.subplots(figsize=(7, 4))
        theta = np.asarray(r["theta"])
        ax.plot(theta, r["w_true"], "k-", lw=2, label="truth")
        ax.plot(theta, r["w_contaminated"], "r--", lw=1.5, label="contaminated")

        colors = {"OLS": "C0", "ISD-1": "C1", "ElasticNet": "C2"}
        for method in methods:
            key = f"w_recovered_{method}"
            if key in r:
                ax.plot(theta, r[key], color=colors.get(method, "C3"),
                        lw=1.5, ls="-.", label=f"recovered ({method})")

        ax.set_xscale("log")
        ax.set_xlabel("θ [deg]")
        ax.set_ylabel("w(θ)")
        ax.set_title(f"GLASS | {level} | {scenario}")
        ax.legend(fontsize=8)
        ax.axhline(0, color="gray", lw=0.5)

        fname = f"wtheta_glass_{level}_{scenario}.png"
        fig.savefig(plots_dir / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"  Plots saved to {plots_dir}/")


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    # Separate subdirectory per (NSIDE, N_glass) so runs never overwrite each other.
    run_dir = output_dir / f"nside{args.nside:04d}_N{args.n_glass}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}")

    from sys_mapping.glass_mocks import generate_glass_fullsky_mock, measure_nz
    from sys_mapping.simulation import (
        load_systematic_maps,
        apply_footprint_mask,
        make_contamination_grid,
        run_wtheta_recovery,
        save_simulation_catalog,
        inject_systematics,
    )

    # ── Step 1: Generate GLASS mock ───────────────────────────────────────────
    _print_step(1, f"Generating GLASS full-sky mock (n={args.n_glass:,}, nside={args.nside})")
    rng_ref = np.random.default_rng(args.seed)
    z_ref = rng_ref.uniform(0.05, 0.26, 50_000)
    z_edges, nz = measure_nz(z_ref, 0.05, 0.26, n_bins=20)

    t0 = time.perf_counter()
    glass_cat = generate_glass_fullsky_mock(
        nside=args.nside,
        n_total=args.n_glass,
        z_edges=z_edges,
        nz=nz,
        seed=args.seed,
    )
    print(f"  Generated {glass_cat['n_total']:,} galaxies in {time.perf_counter()-t0:.1f}s")

    if args.dry_run:
        print("\nDry-run complete: GLASS catalog prepared.")
        return 0

    # ── Step 2: Load systematic maps and derive footprint ─────────────────────
    _print_step(2, f"Loading LSDR10 systematic maps (NSIDE={args.nside})")
    t0 = time.perf_counter()
    templates, template_names, footprint = load_systematic_maps(args.syst_dir, nside=args.nside)
    n_sys = templates.shape[0]
    n_fp = int(footprint.sum())
    print(f"  Loaded {n_sys} maps: {template_names}  ({time.perf_counter()-t0:.1f}s)")
    print(f"  Footprint: {n_fp:,} / {len(footprint):,} pixels  "
          f"({100*n_fp/len(footprint):.1f}% of sky)")

    # Apply footprint: keep only galaxies and randoms inside the survey mask
    n_before = glass_cat["n_total"]
    glass_cat = apply_footprint_mask(glass_cat, footprint, nside=args.nside)
    n_after = glass_cat["n_total"]
    n_rand_after = len(glass_cat["ra_rand"])
    print(f"  GLASS after footprint cut: {n_after:,} galaxies  "
          f"({n_before:,} → {n_after:,}), {n_rand_after:,} randoms")

    # ── Step 3: Contaminate, save, recover ────────────────────────────────────
    configs = make_contamination_grid(n_sys=n_sys, seed=args.seed)
    all_results: list[dict] = []

    _print_step(3, f"Running {len(configs)} contamination configs on GLASS mock")

    for cfg_idx, config in enumerate(configs):
        label = f"{config.level}_{config.scenario}"
        print(f"\n  [{cfg_idx+1:02d}/{len(configs)}] GLASS | {label}")

        weight_cont = inject_systematics(
            glass_cat["ra"], glass_cat["dec"],
            glass_cat["ra_rand"], glass_cat["dec_rand"],
            templates, config.a_true, config.b_true,
            nside=args.nside,
        )

        z = glass_cat.get("z", np.zeros(len(glass_cat["ra"])))
        out_fits = run_dir / f"{label}_seed{args.seed}.fits"
        save_simulation_catalog(glass_cat["ra"], glass_cat["dec"], z, weight_cont, config, out_fits)
        print(f"    Saved contaminated catalog → {out_fits}")

        t0 = time.perf_counter()
        result = run_wtheta_recovery(
            glass_cat,
            templates,
            template_names,
            config,
            nside=args.nside,
            methods=args.methods,
            min_sep=args.min_sep,
            max_sep=args.max_sep,
            nbins=args.nbins,
        )
        result["source"] = "glass"
        elapsed = time.perf_counter() - t0
        print(f"    Recovery complete in {elapsed:.1f}s")

        theta = result["theta"]
        w_true = result["w_true"]
        w_cont = result["w_contaminated"]
        _scale = max(1e-3, float(np.max(np.abs(w_true))))
        cont_bias = float(np.mean(np.abs(w_cont - w_true) / _scale))
        print(f"    Mean |w_cont - w_true|/max|w_true| = {cont_bias:.3f}")
        for method in args.methods:
            key = f"w_recovered_{method}"
            if key in result:
                w_rec = result[key]
                rec_bias = float(np.mean(np.abs(w_rec - w_true) / _scale))
                print(f"    {method:12s}: residual = {rec_bias:.3f}")

        all_results.append(result)

    # ── Step 4: Save JSON summary ──────────────────────────────────────────────
    _print_step(4, "Saving results summary")
    _save_results(all_results, run_dir)

    # ── Step 5: Plots ──────────────────────────────────────────────────────────
    _print_step(5, "Generating w(θ) recovery plots")
    _plot_recovery(all_results, run_dir, args.methods)

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
