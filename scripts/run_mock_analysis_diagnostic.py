"""Deep-dive diagnostic figures for one synthetic mock.

Generates four publication-quality figures:
  1. mock_sky_overview.png      — galaxy / random / overdensity / contamination sky maps
  2. mock_templates_sky.png     — each synthetic template on the full sky
  3. mock_weight_histograms.png — per-pixel weight distributions with posterior credible band
  4. mock_param_significance.png — per-template S/N bar chart for both models

Usage
-----
python scripts/run_mock_analysis_diagnostic.py \
    --nside 64 --n-sys 5 --seed 0 \
    --output-dir results/mock_analysis_diagnostic/
"""
import argparse
import warnings
from pathlib import Path

import healpy as hp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import sys_mapping as sm
from sys_mapping.contamination import apply_contamination, unpack_params
from sys_mapping.correction import rotate_templates, transform_params_from_rotated
from sys_mapping.model_selection import likelihood_ratio_test

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})


# ── Synthetic mock generation (same as run_mock_analysis.py) ─────────────────

def make_synthetic_mock(nside, templates, a_true, b_true, n_mean=30, seed=0):
    n_pix = hp.nside2npix(nside)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1, dtype=float)
    cl_G = (ell + 1.0) ** (-2); cl_G[0] = 0.0
    cl_G *= 0.5**2 / np.sum((2 * ell + 1) / (4 * np.pi) * cl_G)
    G = hp.synfast(cl_G, nside=nside, lmax=lmax)
    delta_true = np.exp(G - 0.5 * 0.5**2) - 1.0

    theta_pix, phi_pix = hp.pix2ang(nside, np.arange(n_pix))
    lat = 90.0 - np.degrees(theta_pix)
    mask = np.abs(lat) > 20.0

    delta_cont = np.asarray(
        apply_contamination(jnp.asarray(delta_true), jnp.asarray(templates),
                            jnp.asarray(a_true), jnp.asarray(b_true))
    )
    lam = np.maximum(n_mean * (1.0 + delta_cont), 0.0) * mask.astype(float)
    pixel_counts = rng.poisson(lam)
    rand_pixel_counts = np.round(mask.astype(float) * n_mean * 8).astype(int)

    pix_ra, pix_dec = hp.pix2ang(nside, np.arange(n_pix), lonlat=True)
    gal_idx = np.repeat(np.arange(n_pix), pixel_counts)
    rand_idx = np.repeat(np.arange(n_pix), rand_pixel_counts)

    return (
        pix_ra[gal_idx], pix_dec[gal_idx],
        pix_ra[rand_idx], pix_dec[rand_idx],
        a_true, b_true,
        pixel_counts, rand_pixel_counts, delta_true, delta_cont, mask,
    )


# ── Sky-map figure helpers ────────────────────────────────────────────────────

def _mollview(data_full, fig, sub, title, unit="", cmap="RdBu_r",
              min=None, max=None, log=False):
    """Thin wrapper around hp.mollview that respects sub-figure layout."""
    m = data_full.copy()
    if log:
        m[m <= 0] = hp.UNSEEN
        good = m != hp.UNSEEN
        if good.any():
            m[good] = np.log10(m[good])
    hp.mollview(
        m, fig=fig.number, sub=sub,
        title=title, unit=unit, cmap=cmap,
        min=min, max=max,
        cbar=True, notext=True,
    )
    hp.graticule(dpar=30, dmer=60, lw=0.3, alpha=0.5)


# ── Figure 1: sky overview ────────────────────────────────────────────────────

def plot_sky_overview(nside, templates, a_true, b_true,
                      gal_counts_full, rand_counts_full,
                      delta_g, good_pix, outdir):
    n_pix = hp.nside2npix(nside)

    m_delta = np.full(n_pix, hp.UNSEEN)
    m_delta[good_pix] = delta_g

    cont_amp = np.full(n_pix, hp.UNSEEN)
    raw_amp = sum(float(a_true[i]) * templates[i] + float(b_true[i]) * templates[i]
                  for i in range(len(a_true)))
    cont_amp[good_pix] = raw_amp[good_pix]

    vmax_delta = max(abs(np.percentile(delta_g, 2)), abs(np.percentile(delta_g, 98)))
    vmax_cont = max(abs(np.nanpercentile(cont_amp[good_pix], 2)),
                    abs(np.nanpercentile(cont_amp[good_pix], 98)))

    fig = plt.figure(figsize=(14, 7))
    _mollview(gal_counts_full.astype(float), fig, (2, 2, 1),
              "Galaxy counts (log₁₀)", log=True, cmap="viridis")
    _mollview(rand_counts_full.astype(float), fig, (2, 2, 2),
              "Random counts (log₁₀)", log=True, cmap="viridis")
    _mollview(m_delta, fig, (2, 2, 3),
              r"Overdensity $\delta_g$", cmap="RdBu_r",
              min=-vmax_delta, max=vmax_delta)
    _mollview(cont_amp, fig, (2, 2, 4),
              r"Contamination $\Sigma_i\,(a_i+b_i)\,t_i$", cmap="PuOr",
              min=-vmax_cont, max=vmax_cont)

    plt.suptitle("Synthetic mock — sky overview (NSIDE=64, seed=0)", y=1.01, fontsize=12)
    plt.tight_layout()
    out = outdir / "mock_sky_overview.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure 2: template maps ───────────────────────────────────────────────────

def plot_templates_sky(templates, template_names, outdir):
    n_sys = len(templates)
    ncols = 3
    nrows = (n_sys + ncols - 1) // ncols

    fig = plt.figure(figsize=(14, 4 * nrows))
    for i, (tmpl, name) in enumerate(zip(templates, template_names)):
        vmax = 2.0  # templates are std-normalised
        _mollview(tmpl, fig, (nrows, ncols, i + 1),
                  name, cmap="RdYlBu_r", min=-vmax, max=vmax)

    plt.suptitle("Synthetic systematic templates (each: mean=0, std=1 over full sky)",
                 y=1.01, fontsize=12)
    plt.tight_layout()
    out = outdir / "mock_templates_sky.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure 3: weight histograms with posterior credible band ─────────────────

def plot_weight_histograms(delta_t, R,
                           flat_add, flat_comb,
                           a_hat_add, b_hat_comb,
                           n_sys, outdir, n_post=300, seed=1):
    rng = np.random.default_rng(seed)
    bins = np.linspace(0.5, 1.6, 60)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])

    def pixel_weights_add(a_vec):
        linear = delta_t.T @ a_vec          # (n_good_pix,)
        return 1.0 / np.maximum(1.0 + linear, 0.01)

    def pixel_weights_comb(b_vec):
        linear = delta_t.T @ b_vec
        return 1.0 / np.maximum(1.0 + linear, 0.01)

    # MAP weights
    w_map_add = pixel_weights_add(a_hat_add)
    w_map_comb = pixel_weights_comb(b_hat_comb)

    # Posterior histograms
    idx_add = rng.choice(len(flat_add), size=min(n_post, len(flat_add)), replace=False)
    idx_comb = rng.choice(len(flat_comb), size=min(n_post, len(flat_comb)), replace=False)

    hist_post_add = np.zeros((len(idx_add), len(bins) - 1))
    hist_post_comb = np.zeros((len(idx_comb), len(bins) - 1))

    for j, si in enumerate(idx_add):
        theta_s = flat_add[si]
        a_rot_s, _, _, _ = unpack_params(theta_s, n_sys, "additive")
        a_s, _ = transform_params_from_rotated(np.asarray(a_rot_s), np.zeros(n_sys), R)
        h, _ = np.histogram(pixel_weights_add(np.asarray(a_s)), bins=bins, density=True)
        hist_post_add[j] = h

    for j, si in enumerate(idx_comb):
        theta_s = flat_comb[si]
        _, b_rot_s, _, _ = unpack_params(theta_s, n_sys, "combined")
        _, b_s = transform_params_from_rotated(np.zeros(n_sys), np.asarray(b_rot_s), R)
        h, _ = np.histogram(pixel_weights_comb(np.asarray(b_s)), bins=bins, density=True)
        hist_post_comb[j] = h

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, label, w_map, hist_post, col in [
        (axes[0], "Additive model  (WEIGHT_ADD)", w_map_add, hist_post_add, "steelblue"),
        (axes[1], "Combined model  (WEIGHT_COMB, using b̂)", w_map_comb, hist_post_comb, "darkorange"),
    ]:
        h_map, _ = np.histogram(w_map, bins=bins, density=True)
        p16 = np.percentile(hist_post, 16, axis=0)
        p84 = np.percentile(hist_post, 84, axis=0)
        pmid = np.percentile(hist_post, 50, axis=0)

        ax.fill_between(bin_centres, p16, p84, alpha=0.35, color="gray",
                        label="Posterior 16–84 % band")
        ax.step(bin_centres, pmid, color="gray", lw=1.2, where="mid",
                label="Posterior median")
        ax.step(bin_centres, h_map, color=col, lw=2.0, where="mid",
                label="MAP weights")
        ax.axvline(1.0, color="k", lw=0.8, ls="--", label="No correction (w=1)")
        ax.set_xlabel("Per-pixel weight w(p)")
        ax.set_ylabel("Probability density")
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.suptitle("Per-pixel weight distributions — MAP vs. posterior credible band", fontsize=12)
    plt.tight_layout()
    out = outdir / "mock_weight_histograms.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure 4: parameter S/N bar chart ────────────────────────────────────────

def plot_param_significance(a_hat_add, var_a_add,
                            a_hat_comb, var_a_comb,
                            b_hat_comb, var_b_comb,
                            a_true, b_true,
                            template_names, outdir, snr_threshold=2.0):
    n_sys = len(a_hat_add)
    x = np.arange(n_sys)
    width = 0.35

    snr_a_add = np.abs(a_hat_add) / np.sqrt(np.maximum(var_a_add, 1e-12))
    snr_a_comb = np.abs(a_hat_comb) / np.sqrt(np.maximum(var_a_comb, 1e-12))
    snr_b_comb = np.abs(b_hat_comb) / np.sqrt(np.maximum(var_b_comb, 1e-12))

    injected_a = np.abs(a_true) > 0.05
    injected_b = np.abs(b_true) > 0.05

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

    # Left: additive model
    ax = axes[0]
    bars = ax.bar(x, snr_a_add, color="steelblue", alpha=0.8, label=r"$|\hat a_i|/\sigma_{a_i}$ (additive run)")
    ax.axhline(snr_threshold, color="k", ls="--", lw=1, label=f"S/N = {snr_threshold}")
    for xi, inj in zip(x, injected_a):
        if inj:
            ax.text(xi, snr_a_add[xi] + 0.3, "★", ha="center", va="bottom",
                    fontsize=14, color="red")
    ax.set_xticks(x)
    ax.set_xticklabels([n[:10] for n in template_names], rotation=30, ha="right")
    ax.set_ylabel("S/N")
    ax.set_title("Additive model — parameter S/N")
    ax.legend(fontsize=8)

    # Right: combined model (both a and b)
    ax = axes[1]
    bars_a = ax.bar(x - width / 2, snr_a_comb, width, color="steelblue", alpha=0.8,
                    label=r"$|\hat a_i|/\sigma_{a_i}$ (combined run)")
    bars_b = ax.bar(x + width / 2, snr_b_comb, width, color="darkorange", alpha=0.8,
                    label=r"$|\hat b_i|/\sigma_{b_i}$ (combined run)")
    ax.axhline(snr_threshold, color="k", ls="--", lw=1, label=f"S/N = {snr_threshold}")
    for xi, (inj_a, inj_b) in enumerate(zip(injected_a, injected_b)):
        if inj_a:
            ax.text(xi - width / 2, snr_a_comb[xi] + 0.3, "★", ha="center",
                    va="bottom", fontsize=14, color="steelblue")
        if inj_b:
            ax.text(xi + width / 2, snr_b_comb[xi] + 0.3, "★", ha="center",
                    va="bottom", fontsize=14, color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels([n[:10] for n in template_names], rotation=30, ha="right")
    ax.set_ylabel("S/N")
    ax.set_title("Combined model — parameter S/N")
    ax.legend(fontsize=8)

    plt.suptitle("Per-template S/N  (★ = injected with |truth| > 0.05)", fontsize=12)
    plt.tight_layout()
    out = outdir / "mock_param_significance.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic figures for one synthetic mock.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nside", type=int, default=64)
    parser.add_argument("--n-sys", type=int, default=5)
    parser.add_argument("--n-mean", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-walkers", type=int, default=110)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--n-burn", type=int, default=100)
    parser.add_argument("--output-dir", default="results/mock_analysis_diagnostic/")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_sys = min(args.n_sys, 5)
    templates = sm.generate_systematic_maps(args.nside, families=list(range(n_sys)), seed=0)
    template_names = [f"synth_{i}" for i in range(n_sys)]
    print(f"Templates: {n_sys}  NSIDE={args.nside}  seed={args.seed}")

    # Injected truth (same draw as in run_mock_analysis --synthetic)
    rng_truth = np.random.default_rng(42)
    a_true = rng_truth.normal(0, 0.10, n_sys)
    b_true = rng_truth.normal(0, 0.10, n_sys)
    print(f"  a_true: {np.round(a_true, 4)}")
    print(f"  b_true: {np.round(b_true, 4)}")

    # Generate mock
    (ra_g, dec_g, ra_r, dec_r,
     a_true, b_true,
     pixel_counts, rand_pixel_counts,
     delta_true, delta_cont, mask) = make_synthetic_mock(
        args.nside, templates, a_true, b_true, n_mean=args.n_mean, seed=args.seed
    )
    print(f"  Galaxies: {len(ra_g):,}   Randoms: {len(ra_r):,}")

    # Pixelize
    n_pix = hp.nside2npix(args.nside)
    gal_counts_full = sm.pixelize_catalog(ra_g, dec_g, args.nside)
    rand_counts_full = sm.pixelize_catalog(ra_r, dec_r, args.nside)
    delta_g, good_pix = sm.compute_overdensity(gal_counts_full, rand_counts_full)
    delta_t = sm.assign_template_values(templates, good_pix)  # (n_sys, n_good)
    n_good = int(good_pix.sum())
    print(f"  Good pixels: {n_good:,}")

    # Figure 1 & 2 (no MCMC needed)
    print("Figure 1: sky overview ...")
    plot_sky_overview(args.nside, templates, a_true, b_true,
                      gal_counts_full, rand_counts_full,
                      delta_g, good_pix, outdir)

    print("Figure 2: template maps ...")
    plot_templates_sky(templates, template_names, outdir)

    # MCMC
    print("Running MCMC (additive + combined) ...")
    delta_t_rot, R, eigenvalues = rotate_templates(delta_t)

    n_dim_add = n_sys + 1
    n_dim_comb = 2 * n_sys + 1
    nw_add = max(args.n_walkers, 2 * n_dim_add + 2)
    nw_comb = max(args.n_walkers, 2 * n_dim_comb + 2)

    flat_add, _ = sm.run_mcmc(n_sys=n_sys, model="additive",
                               delta_g_obs=delta_g, delta_t=delta_t_rot,
                               n_walkers=nw_add, n_steps=args.n_steps,
                               n_burn=args.n_burn, seed=args.seed, progress=True)
    flat_comb, _ = sm.run_mcmc(n_sys=n_sys, model="combined",
                                delta_g_obs=delta_g, delta_t=delta_t_rot,
                                n_walkers=nw_comb, n_steps=args.n_steps,
                                n_burn=args.n_burn, seed=args.seed, progress=True)

    # MAP parameters
    theta_add = sm.get_mle_params(flat_add)
    theta_comb = sm.get_mle_params(flat_comb)

    a_rot_add, _, _, _ = unpack_params(theta_add, n_sys, "additive")
    a_rot_comb, b_rot_comb, _, _ = unpack_params(theta_comb, n_sys, "combined")

    a_hat_add, _ = transform_params_from_rotated(np.asarray(a_rot_add), np.zeros(n_sys), R)
    a_hat_comb, b_hat_comb = transform_params_from_rotated(
        np.asarray(a_rot_comb), np.asarray(b_rot_comb), R
    )

    cov_a_add, _ = sm.get_param_covariance_from_chain(flat_add, n_sys, "additive")
    cov_a_comb, cov_b_comb = sm.get_param_covariance_from_chain(flat_comb, n_sys, "combined")
    var_a_add = np.diag(R.T @ cov_a_add @ R)
    var_a_comb = np.diag(R.T @ cov_a_comb @ R)
    var_b_comb = np.diag(R.T @ cov_b_comb @ R)

    # Figures 3 & 4
    print("Figure 3: weight histograms ...")
    plot_weight_histograms(delta_t, R,
                           flat_add, flat_comb,
                           np.asarray(a_hat_add), np.asarray(b_hat_comb),
                           n_sys, outdir)

    print("Figure 4: parameter S/N ...")
    plot_param_significance(
        np.asarray(a_hat_add), var_a_add,
        np.asarray(a_hat_comb), var_a_comb,
        np.asarray(b_hat_comb), var_b_comb,
        a_true, b_true,
        template_names, outdir,
    )

    lrt = likelihood_ratio_test(delta_g, delta_t_rot, theta_add, theta_comb,
                                 null_model="additive", alt_model="combined")
    print(f"\nLRT: lambda={lrt.lambda_lr:.1f}  p={lrt.p_value:.2e}  reject={lrt.reject_null}")
    print(f"All figures written to {outdir}")


if __name__ == "__main__":
    main()
