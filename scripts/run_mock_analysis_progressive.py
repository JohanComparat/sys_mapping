"""Progressive template contamination study.

For each combination of (k contaminated templates, contamination mode), runs
n_mocks_per_case synthetic mocks and checks whether the sys_mapping pipeline
correctly recovers the contaminated templates and chooses the right model.

Contamination grid
------------------
k ∈ {1, 2, 3}       — number of contaminated templates (always the first k)
mode ∈ {additive, multiplicative, combined}
  additive      : a_true[0:k] ~ N(0, σ_a), b_true = 0
  multiplicative: a_true = 0,  b_true[0:k] ~ N(0, σ_b)
  combined      : a_true[0:k] ~ N(0, σ_a), b_true[0:k] ~ N(0, σ_b)
  templates k..n_sys-1 always have a_true = b_true = 0

Usage
-----
python scripts/run_mock_analysis_progressive.py \
    --nside 64 --n-sys 5 \
    --n-mocks-per-case 20 \
    --snr-threshold 2.0 \
    --output-dir results/mock_analysis_progressive/
"""
import argparse
import json
import warnings
from itertools import product
from pathlib import Path

import healpy as hp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

import sys_mapping as sm
from sys_mapping.contamination import apply_contamination
from sys_mapping.model_selection import likelihood_ratio_test

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

MODES = ["additive", "multiplicative", "combined"]
K_VALUES = [1, 2, 3]


# ── Mock generation ───────────────────────────────────────────────────────────

def make_mock_progressive(nside, templates, k, mode, sigma=0.15, n_mean=30, seed=0):
    """Generate a synthetic mock with exactly k contaminated templates.

    Returns ra_g, dec_g, ra_r, dec_r, a_true, b_true.
    """
    n_sys = templates.shape[0]
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    a_true = np.zeros(n_sys)
    b_true = np.zeros(n_sys)
    if mode in ("additive", "combined"):
        a_true[:k] = rng.normal(0, sigma, k)
    if mode in ("multiplicative", "combined"):
        b_true[:k] = rng.normal(0, sigma, k)

    n_pix = hp.nside2npix(nside)
    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1, dtype=float)
    cl_G = (ell + 1.0) ** (-2); cl_G[0] = 0.0
    cl_G *= 0.5**2 / np.sum((2 * ell + 1) / (4 * np.pi) * cl_G)
    G = hp.synfast(cl_G, nside=nside, lmax=lmax)
    delta_true = np.exp(G - 0.5 * 0.5**2) - 1.0

    theta_pix, _ = hp.pix2ang(nside, np.arange(n_pix))
    lat = 90.0 - np.degrees(theta_pix)
    mask = np.abs(lat) > 20.0

    delta_cont = np.asarray(
        apply_contamination(jnp.asarray(delta_true), jnp.asarray(templates),
                            jnp.asarray(a_true), jnp.asarray(b_true))
    )
    lam = np.maximum(n_mean * (1.0 + delta_cont), 0.0) * mask.astype(float)
    counts = np.random.default_rng(seed + 1000).poisson(lam)
    rand_counts = np.round(mask.astype(float) * n_mean * 8).astype(int)

    pix_ra, pix_dec = hp.pix2ang(nside, np.arange(n_pix), lonlat=True)
    gal_idx = np.repeat(np.arange(n_pix), counts)
    rand_idx = np.repeat(np.arange(n_pix), rand_counts)

    return (
        pix_ra[gal_idx], pix_dec[gal_idx],
        pix_ra[rand_idx], pix_dec[rand_idx],
        a_true, b_true,
    )


# ── Per-mock analysis ─────────────────────────────────────────────────────────

def analyse_one(nside, templates, k, mode, mock_id,
                n_walkers, n_steps, n_burn, snr_threshold, sigma):
    ra_g, dec_g, ra_r, dec_r, a_true, b_true = make_mock_progressive(
        nside, templates, k, mode, sigma=sigma, seed=mock_id * 100 + k * 10
    )
    n_sys = templates.shape[0]

    gal_counts = sm.pixelize_catalog(ra_g, dec_g, nside)
    rand_counts = sm.pixelize_catalog(ra_r, dec_r, nside)
    delta_g, good_pix = sm.compute_overdensity(gal_counts, rand_counts)
    delta_t = sm.assign_template_values(templates, good_pix)

    res_add = sm.run_decontamination(
        "MCMC-add", delta_g, delta_t,
        n_walkers=n_walkers, n_steps=n_steps, n_burn=n_burn,
        seed=mock_id, progress=False,
    )
    res_comb = sm.run_decontamination(
        "MCMC-comb", delta_g, delta_t,
        n_walkers=n_walkers, n_steps=n_steps, n_burn=n_burn,
        seed=mock_id, progress=False,
    )

    a_hat_add  = res_add["a_hat"]
    b_hat_comb = res_comb["b_hat"]
    var_a_add  = np.diag(res_add["cov_a"])
    var_b_comb = np.diag(res_comb["cov_b"])

    snr_a = np.abs(np.asarray(a_hat_add)) / np.sqrt(np.maximum(var_a_add, 1e-12))
    snr_b = np.abs(np.asarray(b_hat_comb)) / np.sqrt(np.maximum(var_b_comb, 1e-12))

    delta_t_rot = res_comb["R"] @ delta_t
    theta_add  = sm.get_mle_params(res_add["flat_chain"])
    theta_comb = sm.get_mle_params(res_comb["flat_chain"])
    lrt = likelihood_ratio_test(delta_g, delta_t_rot, theta_add, theta_comb,
                                 null_model="additive", alt_model="combined")

    # Detection: contaminated = 0..k-1, uncontaminated = k..n_sys-1
    contam_mask = np.zeros(n_sys, dtype=bool)
    contam_mask[:k] = True

    # For additive mode, signal is in a; for multiplicative, in b; for combined, both
    if mode == "additive":
        detected = snr_a > snr_threshold
        tp = (detected & contam_mask).sum() / k
        fp = (detected & ~contam_mask).sum() / max(n_sys - k, 1)
    elif mode == "multiplicative":
        detected = snr_b > snr_threshold
        tp = (detected & contam_mask).sum() / k
        fp = (detected & ~contam_mask).sum() / max(n_sys - k, 1)
    else:  # combined
        detected_a = snr_a > snr_threshold
        detected_b = snr_b > snr_threshold
        detected = detected_a | detected_b
        tp = (detected & contam_mask).sum() / k
        fp = (detected & ~contam_mask).sum() / max(n_sys - k, 1)

    # Correct LRT decision
    # additive mode → expect NOT to reject (null correct)
    # multiplicative/combined → expect to reject (null wrong)
    lrt_correct = (mode == "additive") == (not lrt.reject_null)

    return {
        "k": k,
        "mode": mode,
        "mock_id": mock_id,
        "n_galaxies": int(len(ra_g)),
        "lrt_lambda": float(lrt.lambda_lr),
        "lrt_reject": bool(lrt.reject_null),
        "lrt_correct": bool(lrt_correct),
        "snr_a": snr_a.tolist(),
        "snr_b": snr_b.tolist(),
        "a_true": a_true.tolist(),
        "b_true": b_true.tolist(),
        "a_hat": np.asarray(a_hat_add).tolist(),
        "b_hat": np.asarray(res_comb["b_hat"]).tolist(),
        "tp": float(tp),
        "fp": float(fp),
    }


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_snr_grid(df, n_sys, template_names, k_values, modes, snr_threshold, outdir):
    """Figure A: grid of S/N bar charts (rows=k, cols=mode).

    S/N uses the per-fit **MCMC posterior** width (``snr_a = |â_i|/σ_{a_i}``).  Note this is a
    different error model from the OLS analytic σ of the SNR pre-selection page: for the short
    emcee chains here the posterior is *wide* (wider than the mock-covariance sandwich, verified),
    so — unlike the OLS-σ pages — this S/N is **not** overconfident; see the page caveat.
    """
    nrows, ncols = len(k_values), len(modes)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                              sharey=False, sharex=True)
    x = np.arange(n_sys)
    width = 0.35

    for ri, k in enumerate(k_values):
        for ci, mode in enumerate(modes):
            ax = axes[ri, ci]
            sub = df[(df["k"] == k) & (df["mode"] == mode)]

            snr_a_all = np.array(sub["snr_a"].tolist())
            snr_b_all = np.array(sub["snr_b"].tolist())
            mean_a = snr_a_all.mean(axis=0)
            std_a  = snr_a_all.std(axis=0)
            mean_b = snr_b_all.mean(axis=0)
            std_b  = snr_b_all.std(axis=0)

            ax.bar(x - width / 2, mean_a, width, yerr=std_a,
                   color="steelblue", alpha=0.8, capsize=3,
                   label=r"$|\hat{a}_i|/\sigma_{a_i}$")
            ax.bar(x + width / 2, mean_b, width, yerr=std_b,
                   color="darkorange", alpha=0.8, capsize=3,
                   label=r"$|\hat{b}_i|/\sigma_{b_i}$")
            ax.axhline(snr_threshold, color="k", ls="--", lw=1.2)

            # Vertical separator between contaminated and uncontaminated
            if k < n_sys:
                ax.axvline(k - 0.5, color="red", ls=":", lw=1.5)

            ax.set_title(f"k={k}  mode={mode}", fontsize=9)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7, loc="upper right")
            if ri == nrows - 1:
                ax.set_xticks(x)
                ax.set_xticklabels([n[:8] for n in template_names], rotation=30, ha="right")
            if ci == 0:
                ax.set_ylabel("Mean S/N (±std)")

    plt.suptitle(
        "Parameter S/N by contamination level and mode\n"
        "Red dotted line separates contaminated (left) from uncontaminated (right) templates",
        fontsize=11,
    )
    plt.tight_layout()
    out = outdir / "progressive_snr_grid.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_lrt_summary(df, k_values, modes, outdir):
    """Figure B: LRT lambda violin plots (rows=k, cols=mode)."""
    nrows, ncols = len(k_values), len(modes)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))

    for ri, k in enumerate(k_values):
        for ci, mode in enumerate(modes):
            ax = axes[ri, ci]
            sub = df[(df["k"] == k) & (df["mode"] == mode)]
            lambdas = sub["lrt_lambda"].values
            correct = sub["lrt_correct"].values

            col = "steelblue" if correct.mean() >= 0.5 else "crimson"
            vp = ax.violinplot(lambdas, positions=[0], showmedians=True, showextrema=True)
            for body in vp["bodies"]:
                body.set_facecolor(col)
                body.set_alpha(0.7)
            ax.scatter(np.zeros(len(lambdas)) + np.random.uniform(-0.08, 0.08, len(lambdas)),
                       lambdas, s=15, color="k", alpha=0.6, zorder=5)
            rate = correct.mean()
            ax.set_title(f"k={k}  mode={mode}\ncorrect rate={rate:.0%}", fontsize=9)
            ax.set_xticks([])
            if ci == 0:
                ax.set_ylabel(r"$\lambda_{\rm LR}$")

    plt.suptitle(
        "LRT statistic λ_LR distribution by contamination level and mode\n"
        "(blue = LRT decision mostly correct, red = mostly wrong)",
        fontsize=11,
    )
    plt.tight_layout()
    out = outdir / "progressive_lrt_summary.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_detection_rates(df, k_values, modes, outdir):
    """Figure C: heatmaps of TP rate, FP rate, LRT correct rate."""
    metrics = [("tp", "True positive rate\n(contaminated tmpl detected)"),
               ("fp", "False positive rate\n(clean tmpl falsely detected)"),
               ("lrt_correct", "LRT correct-decision rate")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (col, title) in zip(axes, metrics):
        mat = np.zeros((len(k_values), len(modes)))
        for ri, k in enumerate(k_values):
            for ci, mode in enumerate(modes):
                sub = df[(df["k"] == k) & (df["mode"] == mode)]
                mat[ri, ci] = sub[col].mean()

        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(modes, rotation=20)
        ax.set_yticks(range(len(k_values)))
        ax.set_yticklabels([f"k={k}" for k in k_values])
        ax.set_title(title, fontsize=10)

        for ri in range(len(k_values)):
            for ci in range(len(modes)):
                ax.text(ci, ri, f"{mat[ri, ci]:.2f}", ha="center", va="center",
                        fontsize=11, color="black")

    plt.suptitle("Detection performance across contamination scenarios", fontsize=12)
    plt.tight_layout()
    out = outdir / "progressive_detection_rates.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Progressive template contamination validation study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nside", type=int, default=64)
    parser.add_argument("--n-sys", type=int, default=5)
    parser.add_argument("--n-mean", type=int, default=30)
    parser.add_argument("--n-mocks-per-case", type=int, default=20)
    parser.add_argument("--n-walkers", type=int, default=110)
    parser.add_argument("--n-steps", type=int, default=400)
    parser.add_argument("--n-burn", type=int, default=80)
    parser.add_argument("--snr-threshold", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=0.15,
                        help="Injection amplitude std (N(0, sigma))")
    parser.add_argument("--output-dir", default="results/mock_analysis_progressive/")
    parser.add_argument("--from-cache", action="store_true",
                        help="Reprocess the per-cell JSONs already in --output-dir (no MCMC re-fit) "
                             "— used to regenerate figures with the calibrated sandwich S/N overlay.")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.from_cache:
        results = [json.loads(p.read_text()) for p in sorted(outdir.glob("progressive_k*_mock*.json"))]
        print(f"[--from-cache] loaded {len(results)} cached cells from {outdir}")
        n_sys = len(results[0]["snr_a"])            # match the cached template count
    else:
        n_sys = args.n_sys
    templates = sm.generate_systematic_maps(args.nside, families=list(range(min(n_sys, 5))), seed=0)
    n_sys = templates.shape[0]
    template_names = [f"synth_{i}" for i in range(n_sys)]

    if not args.from_cache:
        total = len(K_VALUES) * len(MODES) * args.n_mocks_per_case
        print(f"Progressive study: {len(K_VALUES)} k-values × {len(MODES)} modes × "
              f"{args.n_mocks_per_case} mocks = {total} MCMC runs")
        results = []
        run_no = 0
        for k, mode in product(K_VALUES, MODES):
            print(f"\n--- k={k}  mode={mode} ---")
            for mock_id in range(args.n_mocks_per_case):
                run_no += 1
                print(f"  [{run_no}/{total}] mock {mock_id} ...", flush=True)
                res = analyse_one(
                    args.nside, templates, k, mode, mock_id,
                    args.n_walkers, args.n_steps, args.n_burn,
                    args.snr_threshold, args.sigma,
                )
                results.append(res)
                (outdir / f"progressive_k{k}_{mode}_mock{mock_id:03d}.json").write_text(
                    json.dumps(res, indent=2)
                )

        # Save CSV (list columns as JSON strings)
        df_csv = pd.DataFrame(results).copy()
        for col in ["snr_a", "snr_b", "a_true", "b_true", "a_hat", "b_hat"]:
            df_csv[col] = df_csv[col].apply(json.dumps)
        df_csv.to_csv(outdir / "progressive_results.csv", index=False)

    # Expand list columns for plotting
    df = pd.DataFrame(results)
    df["snr_a"] = df["snr_a"].apply(np.asarray)
    df["snr_b"] = df["snr_b"].apply(np.asarray)

    print("\nGenerating figures ...")
    plot_snr_grid(df, n_sys, template_names, K_VALUES, MODES, args.snr_threshold, outdir)
    plot_lrt_summary(df, K_VALUES, MODES, outdir)
    plot_detection_rates(df, K_VALUES, MODES, outdir)

    # Print summary
    print("\n=== Summary ===")
    for k, mode in product(K_VALUES, MODES):
        sub = df[(df["k"] == k) & (df["mode"] == mode)]
        lrt_rate = sub["lrt_correct"].mean()
        tp_mean = sub["tp"].mean()
        fp_mean = sub["fp"].mean()
        lam_med = sub["lrt_lambda"].median()
        print(f"  k={k} {mode:14s}: LRT correct={lrt_rate:.0%}  "
              f"TP={tp_mean:.2f}  FP={fp_mean:.2f}  median λ={lam_med:.1f}")

    print(f"\nAll results written to {outdir}")


if __name__ == "__main__":
    main()
