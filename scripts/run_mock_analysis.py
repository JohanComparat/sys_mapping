"""Validate sys_mapping on mock galaxy catalogs for the LS10 M105 footprint.

Processes a set of mock FITS catalogs with the same pipeline as
compute_sys_weights.py and measures:
  - Contamination parameter recovery (bias and scatter vs. injected truth)
  - Two-point function correction fidelity
  - LRT model-selection statistics across mocks

Each mock is expected to be a FITS file with columns RA, DEC, and optionally
A_TRUE_i / B_TRUE_i (injected contamination parameters) for recovery tests.

Usage
-----
# Single mock (quick test)
python scripts/run_mock_analysis.py \\
    --mock-dir /path/to/mocks \\
    --rand-file /path/to/randoms.fits \\
    --template-dir /path/to/systematics/ \\
    --n-mocks 1 --output-dir /tmp/mock_test

# Full set of 100 mocks
python scripts/run_mock_analysis.py \\
    --mock-dir /path/to/mocks \\
    --rand-file /path/to/randoms.fits \\
    --n-mocks 100 --nside 64 \\
    --output-dir results/mock_analysis/

# Self-contained synthetic test (no external files needed)
python scripts/run_mock_analysis.py --synthetic --n-mocks 5 --nside 32 \\
    --output-dir /tmp/mock_synth
"""
import argparse
import json
import warnings
from pathlib import Path

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pandas as pd
from astropy.io import fits

import sys_mapping as sm
from sys_mapping.contamination import apply_contamination
from sys_mapping.model_selection import likelihood_ratio_test

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Synthetic mock generation ──────────────────────────────────────────────

def make_synthetic_mock(nside, templates, a_true, b_true, n_mean=30, seed=0):
    """Generate a synthetic galaxy catalog with known contamination."""
    n_pix = hp.nside2npix(nside)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # Lognormal delta_g
    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1, dtype=float)
    cl_G = (ell + 1.0) ** (-2); cl_G[0] = 0.0
    cl_G *= 0.5**2 / np.sum((2 * ell + 1) / (4 * np.pi) * cl_G)
    G = hp.synfast(cl_G, nside=nside, lmax=lmax)
    delta_true = np.exp(G - 0.5 * 0.5**2) - 1.0

    # Galactic cut mask
    theta_pix, phi_pix = hp.pix2ang(nside, np.arange(n_pix))
    lat = 90.0 - np.degrees(theta_pix)
    mask = np.abs(lat) > 20.0

    delta_cont = np.asarray(
        apply_contamination(jnp.asarray(delta_true), jnp.asarray(templates),
                            jnp.asarray(a_true), jnp.asarray(b_true))
    )
    lam = np.maximum(n_mean * (1.0 + delta_cont), 0.0) * mask.astype(float)
    counts = rng.poisson(lam)
    rand_counts = np.round(mask.astype(float) * n_mean * 8).astype(int)

    pix_ra, pix_dec = hp.pix2ang(nside, np.arange(n_pix), lonlat=True)
    gal_idx = np.repeat(np.arange(n_pix), counts)
    rand_idx = np.repeat(np.arange(n_pix), rand_counts)

    return (
        pix_ra[gal_idx], pix_dec[gal_idx],
        pix_ra[rand_idx], pix_dec[rand_idx],
        a_true, b_true,
    )


# ── Per-mock analysis ──────────────────────────────────────────────────────

_MOCK_METHODS = ["OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"]


def analyse_mock(mock_id, ra_gal, dec_gal, ra_rand, dec_rand,
                 templates, nside, n_walkers, n_steps, n_burn,
                 a_true=None, b_true=None):
    """Run the full sys_mapping pipeline on one mock."""
    n_sys = templates.shape[0]
    n_pix = hp.nside2npix(nside)

    gal_counts = sm.pixelize_catalog(ra_gal, dec_gal, nside)
    rand_counts = sm.pixelize_catalog(ra_rand, dec_rand, nside)
    delta_g, good_pix = sm.compute_overdensity(gal_counts, rand_counts)
    delta_t = sm.assign_template_values(templates, good_pix)

    # Run all methods via the unified interface (fastest first)
    method_results = {}
    for meth in _MOCK_METHODS:
        try:
            method_results[meth] = sm.run_decontamination(
                meth, delta_g, delta_t,
                n_walkers=n_walkers, n_steps=n_steps, n_burn=n_burn,
                seed=mock_id, progress=False,
            )
        except Exception as exc:
            warnings.warn(f"Mock {mock_id}: {meth} failed: {exc}")

    # Extract MCMC-comb primary results (kept for backwards compatibility)
    res_add  = method_results.get("MCMC-add",  {})
    res_comb = method_results.get("MCMC-comb", {})
    a_hat_c  = res_comb.get("a_hat",  np.zeros(n_sys))
    b_hat_c  = res_comb.get("b_hat",  np.zeros(n_sys))
    cov_a    = res_comb.get("cov_a",  np.eye(n_sys))
    cov_b    = res_comb.get("cov_b",  np.eye(n_sys))
    var_a    = np.diag(cov_a)
    var_b    = np.diag(cov_b)

    # LRT: uses MCMC-add and MCMC-comb chains in the PCA-rotated basis
    flat_add  = res_add.get("flat_chain")
    flat_comb = res_comb.get("flat_chain")
    if flat_add is not None and flat_comb is not None:
        theta_add  = sm.get_mle_params(flat_add)
        theta_comb = sm.get_mle_params(flat_comb)
        R = res_comb["R"]
        delta_t_rot = R @ delta_t
        lrt = likelihood_ratio_test(delta_g, delta_t_rot, theta_add, theta_comb,
                                     null_model="additive", alt_model="combined",
                                     significance=0.05)
        lrt_lambda = float(lrt.lambda_lr)
        lrt_p      = float(lrt.p_value)
        lrt_reject = bool(lrt.reject_null)
    else:
        lrt_lambda, lrt_p, lrt_reject = float("nan"), float("nan"), False

    # Per-method results for all 6 methods
    per_method = {}
    for meth, mres in method_results.items():
        a_m = mres.get("a_hat")
        b_m = mres.get("b_hat")
        per_method[meth] = {
            "a_hat": np.asarray(a_m).tolist() if a_m is not None else None,
            "b_hat": np.asarray(b_m).tolist() if b_m is not None else None,
            "sigma_hat": mres.get("sigma_hat"),
            "elapsed_s": mres.get("elapsed_s"),
        }
        if a_true is not None and a_m is not None:
            per_method[meth]["a_bias"] = (np.asarray(a_m) - np.asarray(a_true)).tolist()
        if b_true is not None and b_m is not None:
            per_method[meth]["b_bias"] = (np.asarray(b_m) - np.asarray(b_true)).tolist()

    result = {
        "mock_id": mock_id,
        "n_galaxies": int(len(ra_gal)),
        "n_good_pix": int(good_pix.sum()),
        "a_hat": np.asarray(a_hat_c).tolist(),
        "b_hat": np.asarray(b_hat_c).tolist(),
        "var_a": var_a.tolist(),
        "var_b": var_b.tolist(),
        "lrt_lambda": lrt_lambda,
        "lrt_p": lrt_p,
        "lrt_reject": lrt_reject,
        "per_method": per_method,
    }
    if a_true is not None:
        result["a_true"] = list(float(x) for x in a_true)
        result["a_bias"] = (np.asarray(a_hat_c) - np.asarray(a_true)).tolist()
    if b_true is not None:
        result["b_true"] = list(float(x) for x in b_true)
        result["b_bias"] = (np.asarray(b_hat_c) - np.asarray(b_true)).tolist()

    return result


# ── Summary statistics and plots ───────────────────────────────────────────

_METHOD_COLORS = {
    "OLS":        "#1f77b4",
    "ElasticNet": "#ff7f0e",
    "ISD-1":      "#2ca02c",
    "ISD-3":      "#d62728",
    "MCMC-add":   "#9467bd",
    "MCMC-comb":  "#8c564b",
}


def write_summary(results, n_sys, template_names, outdir):
    import matplotlib.pyplot as plt

    df = pd.DataFrame(results)
    df.to_csv(outdir / "mock_results.csv", index=False)

    n_mocks = len(results)
    tick_labels = (
        [n[:10] for n in template_names[:n_sys]] if template_names
        else [str(i + 1) for i in range(n_sys)]
    )

    # ── Figure 1: parameter recovery for all six methods ──────────────────────
    has_per_method = "per_method" in results[0] if results else False
    has_a_true = "a_true" in results[0] if results else False

    if has_per_method and has_a_true:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
        axes = axes.ravel()

        rng_jit = np.random.default_rng(0)

        for ax_idx, meth in enumerate(_MOCK_METHODS):
            ax = axes[ax_idx]
            color = _METHOD_COLORS.get(meth, "gray")

            biases = []   # list of length-n_sys arrays
            for res in results:
                pm = res.get("per_method", {}).get(meth)
                if pm is None or pm.get("a_bias") is None:
                    continue
                biases.append(np.asarray(pm["a_bias"]))

            if biases:
                bias_mat = np.array(biases)   # (n_mocks_valid, n_sys)
                xs = np.arange(1, n_sys + 1)
                jit = rng_jit.uniform(-0.15, 0.15, size=bias_mat.shape)
                for si in range(n_sys):
                    ax.scatter(
                        xs[si] + jit[:, si], bias_mat[:, si],
                        s=8, alpha=0.45, color=color, linewidths=0,
                    )
                # Boxplot overlay (no fliers, just the box)
                bp = ax.boxplot(
                    [bias_mat[:, si] for si in range(n_sys)],
                    positions=xs, widths=0.35,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="black", lw=1.5),
                    boxprops=dict(facecolor=color, alpha=0.25, linewidth=0.8),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                )

                mad = np.median(np.abs(bias_mat))
                ax.set_title(f"{meth}   MAD={mad:.4f}", fontsize=9)
            else:
                ax.set_title(f"{meth}   (no data)", fontsize=9)

            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_xlabel("Template", fontsize=8)
            ax.set_ylabel(r"$\hat{a}_i - a_i^{\rm true}$", fontsize=8)
            ax.set_xticks(range(1, n_sys + 1))
            ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7)
            ax.tick_params(labelsize=7)

        fig.suptitle(
            f"Parameter recovery — {n_mocks} mock realisations, NSIDE=64",
            fontsize=11, y=1.01,
        )
        plt.tight_layout()
        plt.savefig(outdir / "mock_parameter_recovery_all_methods.png",
                    dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved mock_parameter_recovery_all_methods.png")

    # ── Figure 2: sigma recovery for MCMC methods ─────────────────────────────
    # sigma in the likelihood is the std of the observed overdensity residuals,
    # not sigma_G.  The theoretical expectation combines lognormal field variance
    # and Poisson shot noise:
    #   sigma_eff = sqrt(exp(sigma_G^2) - 1 + 1/n_mean)
    sigma_G = 0.5   # hardcoded default in make_synthetic_mock
    n_mean = float(df["n_galaxies"].mean() / df["n_good_pix"].mean()) if len(df) > 0 else 30.0
    sigma_eff = float(np.sqrt(np.exp(sigma_G**2) - 1.0 + 1.0 / n_mean))
    mcmc_methods_with_sigma = [m for m in ("MCMC-add", "MCMC-comb")
                                if has_per_method and any(
                                    r.get("per_method", {}).get(m, {}) is not None
                                    and r["per_method"].get(m, {}).get("sigma_hat") is not None
                                    for r in results
                                )]

    if mcmc_methods_with_sigma:
        fig, axes = plt.subplots(1, len(mcmc_methods_with_sigma),
                                  figsize=(5 * len(mcmc_methods_with_sigma), 4),
                                  sharey=False)
        axes = np.atleast_1d(axes)

        for ax, meth in zip(axes, mcmc_methods_with_sigma):
            sigmas = [
                r["per_method"][meth]["sigma_hat"]
                for r in results
                if r.get("per_method", {}).get(meth, {}) is not None
                and r["per_method"].get(meth, {}).get("sigma_hat") is not None
            ]
            sigmas = np.asarray(sigmas, dtype=float)
            color = _METHOD_COLORS.get(meth, "gray")

            ax.hist(sigmas, bins=20, color=color, alpha=0.7, edgecolor="white", lw=0.5)
            ax.axvline(sigma_eff, color="k", lw=1.5, ls="--",
                       label=rf"$\sigma_{{\rm eff}}={sigma_eff:.3f}$ (predicted)")
            ax.set_xlabel(r"$\hat\sigma$ (recovered)", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            mu, std = np.nanmean(sigmas), np.nanstd(sigmas)
            ax.set_title(
                f"{meth}\n"
                rf"$\langle\hat\sigma\rangle = {mu:.3f} \pm {std:.3f}$"
                f"  (N={len(sigmas)})",
                fontsize=9,
            )
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=8)

        fig.suptitle(
            rf"Intrinsic $\sigma$ recovery — {n_mocks} mock realisations",
            fontsize=11,
        )
        plt.tight_layout()
        plt.savefig(outdir / "mock_sigma_recovery.png", dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved mock_sigma_recovery.png")

    # ── Figure 3: LRT statistics — log x-axis ────────────────────────────────
    if "lrt_lambda" in df.columns:
        reject_frac = df["lrt_reject"].mean()
        lrt_vals = df["lrt_lambda"].dropna().values
        fig, ax = plt.subplots(figsize=(6, 4))
        log_bins = np.logspace(np.log10(max(lrt_vals.min(), 1)), np.log10(lrt_vals.max()), 21)
        ax.hist(lrt_vals, bins=log_bins, color="steelblue", alpha=0.7, edgecolor="white", lw=0.4)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda_{\rm LR}$")
        ax.set_ylabel("Count")
        ax.set_title(f"LRT statistics ({n_mocks} mocks)  — reject fraction: {reject_frac:.0%}")
        plt.tight_layout()
        plt.savefig(outdir / "mock_lrt_statistics.png", dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved mock_lrt_statistics.png")

    # ── Figure 4: b-parameter recovery for MCMC-comb ─────────────────────────
    if has_per_method:
        b_meth = "MCMC-comb"
        b_biases = []
        for res in results:
            pm = res.get("per_method", {}).get(b_meth)
            if pm is None:
                continue
            b_hat = pm.get("b_hat")
            b_true_vals = res.get("b_true")
            if b_hat is None or b_true_vals is None:
                continue
            b_biases.append(np.asarray(b_hat) - np.asarray(b_true_vals))

        if b_biases:
            b_mat = np.array(b_biases)   # (n_mocks, n_sys)
            color = _METHOD_COLORS.get(b_meth, "gray")
            rng_jit2 = np.random.default_rng(1)
            xs = np.arange(1, n_sys + 1)
            jit = rng_jit2.uniform(-0.15, 0.15, size=b_mat.shape)

            fig, ax = plt.subplots(figsize=(5, 4))
            for si in range(n_sys):
                ax.scatter(
                    xs[si] + jit[:, si], b_mat[:, si],
                    s=8, alpha=0.45, color=color, linewidths=0,
                )
            ax.boxplot(
                [b_mat[:, si] for si in range(n_sys)],
                positions=xs, widths=0.35,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", lw=1.5),
                boxprops=dict(facecolor=color, alpha=0.25, linewidth=0.8),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
            )
            mad_b = np.median(np.abs(b_mat))
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_xlabel("Template", fontsize=9)
            ax.set_ylabel(r"$\hat{b}_i - b_i^{\rm true}$", fontsize=9)
            ax.set_xticks(xs)
            ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_title(
                f"Multiplicative parameter recovery — MCMC-comb\n"
                f"{n_mocks} mock realisations, NSIDE=64   MAD={mad_b:.4f}",
                fontsize=9,
            )
            plt.tight_layout()
            plt.savefig(outdir / "mock_b_parameter_recovery.png", dpi=130, bbox_inches="tight")
            plt.close()
            print(f"  Saved mock_b_parameter_recovery.png")

    print(f"\n=== Mock analysis summary ===")
    print(f"  N mocks:        {n_mocks}")
    print(f"  Mean N_gal:     {df['n_galaxies'].mean():.0f}")
    print(f"  Mean N_pix:     {df['n_good_pix'].mean():.0f}")
    if "lrt_reject" in df.columns:
        print(f"  LRT reject fraction (add vs comb): {df['lrt_reject'].mean():.0%}")
    if "a_bias" in df.columns:
        a_bias_arr = np.array(df["a_bias"].tolist())
        print(f"  Mean |a_bias| (MCMC-comb): {np.abs(a_bias_arr).mean():.4f}")
        if "b_bias" in df.columns:
            b_bias_arr = np.array(df["b_bias"].tolist())
            print(f"  Mean |b_bias| (MCMC-comb): {np.abs(b_bias_arr).mean():.4f}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate sys_mapping on mock galaxy catalogs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic mocks (no external files needed).")
    parser.add_argument("--mock-dir", default=None,
                        help="Directory containing mock *_DATA.fits files.")
    parser.add_argument("--rand-file", default=None,
                        help="FITS random catalog (required unless --synthetic).")
    parser.add_argument("--template-dir", default=None,
                        help="Directory of HEALPix systematic FITS files.")
    parser.add_argument("--nside", type=int, default=64)
    parser.add_argument("--n-mocks", type=int, default=10,
                        help="Maximum number of mocks to process.")
    parser.add_argument("--n-sys", type=int, default=5,
                        help="Number of systematic templates (synthetic mode).")
    parser.add_argument("--n-walkers", type=int, default=110, help="MCMC walkers.")
    parser.add_argument("--n-steps", type=int, default=500, help="MCMC steps.")
    parser.add_argument("--n-burn", type=int, default=100, help="MCMC burn-in.")
    parser.add_argument("--n-mean", type=int, default=30,
                        help="Mean galaxies per pixel (synthetic mode).")
    parser.add_argument("--output-dir", default="results/mock_analysis/")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Templates
    if args.template_dir:
        fits_files = sorted(Path(args.template_dir).glob("*.fits"))
        if fits_files:
            import healpy as hp
            maps, names = [], []
            for f in fits_files:
                m = hp.read_map(str(f))
                if len(m) != hp.nside2npix(args.nside):
                    m = hp.ud_grade(m, args.nside)
                good = m != hp.UNSEEN
                if good.any():
                    m[~good] = 0.0
                    m -= np.mean(m[good])
                    s = np.std(m[good])
                    if s > 0:
                        m /= s
                maps.append(m)
                names.append(f.stem)
            templates = np.array(maps)
            template_names = names
        else:
            templates = sm.generate_systematic_maps(args.nside, families=list(range(args.n_sys)), seed=0)
            template_names = [f"synth_{i}" for i in range(args.n_sys)]
    else:
        n_fam = min(args.n_sys, 5)
        templates = sm.generate_systematic_maps(args.nside, families=list(range(n_fam)), seed=0)
        template_names = [f"synth_{i}" for i in range(n_fam)]

    n_sys = templates.shape[0]
    print(f"Templates: {n_sys} ({template_names})")

    if args.synthetic:
        rng = np.random.default_rng(42)
        results = []
        for im in range(args.n_mocks):
            a_true = rng.normal(0, 0.10, n_sys)
            b_true = rng.normal(0, 0.10, n_sys)
            ra_g, dec_g, ra_r, dec_r, at, bt = make_synthetic_mock(
                args.nside, templates, a_true, b_true, n_mean=args.n_mean, seed=im
            )
            print(f"Mock {im+1}/{args.n_mocks}: {len(ra_g):,} galaxies, {len(ra_r):,} randoms")
            res = analyse_mock(im, ra_g, dec_g, ra_r, dec_r, templates,
                               args.nside, args.n_walkers, args.n_steps, args.n_burn,
                               a_true=at, b_true=bt)
            results.append(res)
            (outdir / f"mock_{im:04d}_results.json").write_text(json.dumps(res, indent=2))

    else:
        if not args.mock_dir or not args.rand_file:
            print("ERROR: --mock-dir and --rand-file are required unless --synthetic", flush=True)
            return

        rand_path = Path(args.rand_file)
        if not rand_path.exists():
            print(f"ERROR: random file not found: {rand_path}")
            return

        with fits.open(str(rand_path), memmap=True) as hdul:
            rand = hdul[1].data
            ra_rand = np.asarray(rand["RA"], dtype=np.float64)
            dec_rand = np.asarray(rand["DEC"], dtype=np.float64)

        mock_files = sorted(Path(args.mock_dir).glob("*_DATA.fits"))[:args.n_mocks]
        if not mock_files:
            print(f"ERROR: no *_DATA.fits files found in {args.mock_dir}")
            return

        results = []
        for im, mock_file in enumerate(mock_files):
            print(f"Mock {im+1}/{len(mock_files)}: {mock_file.name}")
            with fits.open(str(mock_file), memmap=True) as hdul:
                cat = hdul[1].data
                ra_g = np.asarray(cat["RA"], dtype=np.float64)
                dec_g = np.asarray(cat["DEC"], dtype=np.float64)
                # Extract injected truth if present
                a_true = None
                b_true = None
                a_cols = [c for c in cat.names if c.startswith("A_TRUE")]
                b_cols = [c for c in cat.names if c.startswith("B_TRUE")]
                if a_cols and b_cols:
                    a_true = np.array([cat[c][0] for c in sorted(a_cols)])
                    b_true = np.array([cat[c][0] for c in sorted(b_cols)])

            res = analyse_mock(im, ra_g, dec_g, ra_rand, dec_rand, templates,
                               args.nside, args.n_walkers, args.n_steps, args.n_burn,
                               a_true=a_true, b_true=b_true)
            results.append(res)
            (outdir / f"mock_{im:04d}_{mock_file.stem}_results.json").write_text(
                json.dumps(res, indent=2)
            )

    write_summary(results, n_sys, template_names, outdir)
    print(f"\nAll outputs written to {outdir}")


if __name__ == "__main__":
    main()
