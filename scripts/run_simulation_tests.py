#!/usr/bin/env python3
"""End-to-end validation: GLASS and Uchuu mocks with LSDR10 systematic contamination.

Phases
------
1. Load Uchuu lightcone mock and measure n(z).
2. Generate a GLASS full-sky mock matched to the Uchuu n(z) and surface density.
3. Load LSDR10 systematic maps.
4. For each mock source (GLASS, Uchuu) and each of 9 contamination configurations
   (3 levels × 3 scenarios):
   a. Inject systematics → per-galaxy WEIGHT_CONT.
   b. Save contaminated catalog to FITS.
   c. Run decontamination methods (OLS, ISD-1, ElasticNet) and compute w(θ).
5. Save results summary to JSON.
6. Plot w(θ) recovery curves.

Usage
-----
python scripts/run_simulation_tests.py \\
    [--nside 64] \\
    [--n-glass 500000] \\
    [--methods OLS ISD-1 ElasticNet] \\
    [--output-dir data/simulations] \\
    [--syst-dir ~/data/legacysurvey/dr10/systematics/] \\
    [--uchuu-data ~/data/Uchuu/FullSky/mock_catalogues/.../..._DATA.fits] \\
    [--uchuu-only | --glass-only] \\
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

# Default paths
_DEFAULT_UCHUU_BASE = os.path.expanduser(
    "~/data/Uchuu/FullSky/mock_catalogues/"
    "MOCK_VLIM_ANY_10.65_Mstar_12.0_0.05_z_0.26_N_0923373/"
    "MOCK_VLIM_ANY_10.65_Mstar_12.0_0.05_z_0.26_N_0923373"
)
_DEFAULT_SYST_DIR = os.path.expanduser("~/data/legacysurvey/dr10/systematics/")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nside", type=int, default=64, help="HEALPix NSIDE for maps (default 64)")
    p.add_argument("--n-glass", type=int, default=500_000, help="Target galaxy count for GLASS mock")
    p.add_argument("--methods", nargs="+", default=["OLS", "ISD-1", "ElasticNet"],
                   choices=["OLS", "ISD-1", "ElasticNet", "MCMC-add", "MCMC-comb"])
    p.add_argument("--output-dir", default="data/simulations", help="Output directory")
    p.add_argument("--syst-dir", default=_DEFAULT_SYST_DIR)
    p.add_argument("--uchuu-data", default=_DEFAULT_UCHUU_BASE + "_DATA.fits")
    p.add_argument("--uchuu-only", action="store_true", help="Skip GLASS mock generation")
    p.add_argument("--glass-only", action="store_true", help="Skip Uchuu loading")
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
        source = r.get("source", "unknown")
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
        ax.set_title(f"{source} | {level} | {scenario}")
        ax.legend(fontsize=8)
        ax.axhline(0, color="gray", lw=0.5)

        fname = f"wtheta_{source}_{level}_{scenario}.png"
        fig.savefig(plots_dir / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"  Plots saved to {plots_dir}/")


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    # All outputs for a given NSIDE go into a dedicated subdirectory so that
    # runs with different NSIDE values never overwrite each other.
    run_dir = output_dir / f"nside{args.nside:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "glass").mkdir(exist_ok=True)
    (run_dir / "uchuu").mkdir(exist_ok=True)
    print(f"Output directory: {run_dir}")

    from sys_mapping.glass_mocks import generate_glass_fullsky_mock, measure_nz
    from sys_mapping.simulation import (
        load_systematic_maps,
        apply_footprint_mask,
        load_uchuu_mock,
        make_contamination_grid,
        run_wtheta_recovery,
        save_simulation_catalog,
        inject_systematics,
    )

    uchuu_rand = args.uchuu_data.replace("_DATA.fits", "_RAND.fits")
    catalogs: dict[str, dict] = {}

    # ── Step 1: Load Uchuu ────────────────────────────────────────────────────
    if not args.glass_only:
        _print_step(1, "Loading Uchuu mock catalog")
        if not os.path.exists(args.uchuu_data):
            print(f"  ERROR: Uchuu data not found at {args.uchuu_data}")
            if not args.uchuu_only:
                print("  Continuing with GLASS only.")
                args.glass_only = False
                args.uchuu_only = False
            else:
                return 1
        else:
            t0 = time.perf_counter()
            uchuu = load_uchuu_mock(args.uchuu_data, uchuu_rand)
            print(f"  Loaded {len(uchuu['ra']):,} galaxies in {time.perf_counter()-t0:.1f}s")
            catalogs["uchuu"] = uchuu

    # ── Step 2: Generate GLASS mock ───────────────────────────────────────────
    if not args.uchuu_only:
        _print_step(2, f"Generating GLASS full-sky mock (n={args.n_glass:,}, nside={args.nside})")
        if "uchuu" in catalogs:
            z_ref = catalogs["uchuu"]["z"]
        else:
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
        catalogs["glass"] = glass_cat

    if args.dry_run:
        print("\nDry-run complete: catalogs prepared.")
        return 0

    # ── Step 3: Load systematic maps and derive footprint ─────────────────────
    _print_step(3, f"Loading LSDR10 systematic maps (NSIDE={args.nside})")
    t0 = time.perf_counter()
    templates, template_names, footprint = load_systematic_maps(args.syst_dir, nside=args.nside)
    n_sys = templates.shape[0]
    n_fp = int(footprint.sum())
    print(f"  Loaded {n_sys} maps: {template_names}  ({time.perf_counter()-t0:.1f}s)")
    print(f"  Footprint: {n_fp:,} / {len(footprint):,} pixels  "
          f"({100*n_fp/len(footprint):.1f}% of sky)")

    # Apply footprint to full-sky mocks (GLASS and Uchuu are both 4π sr catalogs)
    for _src in ("glass", "uchuu"):
        if _src not in catalogs:
            continue
        n_before = catalogs[_src]["n_total"] if "n_total" in catalogs[_src] else len(catalogs[_src]["ra"])
        catalogs[_src] = apply_footprint_mask(catalogs[_src], footprint, nside=args.nside)
        n_after = len(catalogs[_src]["ra"])
        n_rand_after = len(catalogs[_src]["ra_rand"])
        if "n_total" not in catalogs[_src]:
            catalogs[_src]["n_total"] = n_after
        print(f"  {_src.upper()} after footprint cut: {n_after:,} galaxies  "
              f"({n_before:,} → {n_after:,}), {n_rand_after:,} randoms")

    # ── Step 4: Contaminate, save, recover ────────────────────────────────────
    configs = make_contamination_grid(n_sys=n_sys, seed=args.seed)
    all_results: list[dict] = []

    for source_name, catalog in catalogs.items():
        _print_step(4, f"Running {len(configs)} contamination configs on {source_name}")
        source_dir = run_dir / source_name
        source_dir.mkdir(exist_ok=True)

        for cfg_idx, config in enumerate(configs):
            label = f"{config.level}_{config.scenario}"
            print(f"\n  [{cfg_idx+1:02d}/{len(configs)}] {source_name} | {label}")

            # Inject contamination
            weight_cont = inject_systematics(
                catalog["ra"], catalog["dec"],
                catalog["ra_rand"], catalog["dec_rand"],
                templates, config.a_true, config.b_true,
                nside=args.nside,
            )

            # Save contaminated catalog
            z = catalog.get("z", np.zeros(len(catalog["ra"])))
            out_fits = source_dir / f"{label}_seed{args.seed}.fits"
            save_simulation_catalog(catalog["ra"], catalog["dec"], z, weight_cont, config, out_fits)
            print(f"    Saved contaminated catalog → {out_fits}")

            # Run w(θ) recovery
            t0 = time.perf_counter()
            result = run_wtheta_recovery(
                catalog,
                templates,
                template_names,
                config,
                nside=args.nside,
                methods=args.methods,
                min_sep=args.min_sep,
                max_sep=args.max_sep,
                nbins=args.nbins,
            )
            result["source"] = source_name
            elapsed = time.perf_counter() - t0
            print(f"    Recovery complete in {elapsed:.1f}s")

            # Print a quick summary of bias vs recovery
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

    # ── Step 5: Save JSON summary ──────────────────────────────────────────────
    _print_step(5, "Saving results summary")
    _save_results(all_results, run_dir)

    # ── Step 6: Plots ──────────────────────────────────────────────────────────
    _print_step(6, "Generating w(θ) recovery plots")
    _plot_recovery(all_results, run_dir, args.methods)

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
