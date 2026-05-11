"""Reproduce Berlfein et al. 2024 (arXiv:2401.12293) on synthetic data.

Produces Figures 2–8 and Tables 2–3 from the paper using lognormal mocks.
All outputs are written to --output-dir.

Usage
-----
# Quick run (NSIDE=32, 2 realisations)
python scripts/run_paper_validation.py --nside 32 --n-real 2 --output-dir /tmp/paper_val

# Full paper settings (slow — use HPC)
python scripts/run_paper_validation.py --nside 512 --n-real 119 \\
    --n-walkers 250 --n-steps 1500 --n-burn 300 \\
    --output-dir results/paper_validation/
"""
import argparse
import warnings
from pathlib import Path

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats

import sys_mapping as sm
from sys_mapping.contamination import apply_contamination, unpack_params
from sys_mapping.correction import (
    correct_two_point_function,
    debias_params,
    rotate_templates,
    transform_params_from_rotated,
)
from sys_mapping.model_selection import likelihood_ratio_test

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Helpers ────────────────────────────────────────────────────────────────

def _legendre_sum(cl, cos_vals):
    """Compute w(theta) = sum_l (2l+1)/(4pi) * C_l * P_l(cos theta)."""
    n = len(cos_vals)
    lmax_loc = len(cl) - 1
    p_prev = np.ones(n)
    p_curr = cos_vals.copy()
    w = cl[0] * p_prev / (4 * np.pi)
    if lmax_loc >= 1:
        w += cl[1] * p_curr * 3 / (4 * np.pi)
    for l_val in range(2, lmax_loc + 1):
        p_next = ((2 * l_val - 1) * cos_vals * p_curr - (l_val - 1) * p_prev) / l_val
        w += cl[l_val] * p_next * (2 * l_val + 1) / (4 * np.pi)
        p_prev, p_curr = p_curr, p_next
    return w


def _make_lognormal_delta(nside, sigma_G=0.5, seed=None):
    """Lognormal overdensity field with zero mean (family-2 power spectrum for G)."""
    if seed is not None:
        np.random.seed(seed)
    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1, dtype=float)
    cl_G = (ell + 1.0) ** (-2)
    cl_G[0] = 0.0
    var_G = np.sum((2 * ell + 1) / (4 * np.pi) * cl_G)
    cl_G *= sigma_G**2 / var_G
    G = hp.synfast(cl_G, nside=nside, lmax=lmax)
    return np.exp(G - 0.5 * sigma_G**2) - 1.0


def _anafast_wtheta(delta, nside, theta_deg_arr):
    lmax = 3 * nside - 1
    cl = hp.anafast(delta, lmax=lmax)
    return _legendre_sum(cl, np.cos(np.radians(theta_deg_arr)))


def _fit_amplitude(w_measured, w_fiducial):
    mask = np.isfinite(w_measured) & np.isfinite(w_fiducial) & (np.abs(w_fiducial) > 1e-10)
    if mask.sum() < 3:
        return np.nan
    return float(np.dot(w_fiducial[mask], w_measured[mask]) / np.dot(w_fiducial[mask], w_fiducial[mask]))


# ── Analysis routines ──────────────────────────────────────────────────────

def run_fig2(nside, outdir):
    import matplotlib.pyplot as plt

    lmax = 3 * nside - 1
    ell = np.arange(1, lmax + 1)
    theta_deg = np.logspace(np.log10(0.1), np.log10(10), 50)
    cos_theta = np.cos(np.radians(theta_deg))
    family_labels = [
        r"$C_\ell \propto e^{-\ell/500}$",
        r"$C_\ell \propto e^{-(\ell/250)^2}$",
        r"$C_\ell \propto (\ell+1)^{-2}$",
        r"$C_\ell \propto (\ell+1)^{-1}$",
        r"$C_\ell = {\rm const}$",
    ]
    colors = ["C0", "C1", "C2", "C3", "C4"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for fam in range(5):
        cl = sm.systematic_power_spectrum(nside, fam)
        axes[0].semilogy(ell, cl[1:], color=colors[fam], label=family_labels[fam], lw=1.8)
        w_th = _legendre_sum(cl, cos_theta)
        axes[1].loglog(theta_deg, np.abs(w_th), color=colors[fam], label=family_labels[fam], lw=1.8)

    axes[0].set_xlabel(r"Multipole $\ell$")
    axes[0].set_ylabel(r"$C_\ell$")
    axes[0].set_title("Template angular power spectra")
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(1, lmax)
    axes[1].set_xlabel(r"$\theta$ [degrees]")
    axes[1].set_ylabel(r"$|w(\theta)|$")
    axes[1].set_title("Template angular correlation functions")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fpath = outdir / "fig2_template_power_spectra.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fpath}")


def run_fig3(nside, n_mean, sigma_G, n_sys, contam_params, templates, outdir):
    import matplotlib.pyplot as plt

    a_true = contam_params["a_true"].values
    b_true = contam_params["b_true"].values

    delta_true = _make_lognormal_delta(nside, sigma_G=sigma_G, seed=42)
    delta_cont = np.asarray(
        apply_contamination(jnp.asarray(delta_true), jnp.asarray(templates),
                            jnp.asarray(a_true), jnp.asarray(b_true))
    )
    lam = np.maximum(n_mean * (1.0 + delta_cont), 0.0)
    galaxy_counts = np.random.default_rng(0).poisson(lam).astype(float)
    random_counts = np.full(hp.nside2npix(nside), float(n_mean * 10))
    delta_obs, good_pix = sm.compute_overdensity(galaxy_counts, random_counts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, title in [
        (axes[0], delta_true[good_pix], r"True $\delta_g$ (lognormal)"),
        (axes[1], delta_obs, r"Observed $\delta_g$ (contaminated)"),
    ]:
        x = np.linspace(data.min(), data.max(), 300)
        ax.hist(data, bins=60, density=True, color="steelblue", alpha=0.5)
        mu_g, sig_g = stats.norm.fit(data)
        ax.plot(x, stats.norm.pdf(x, mu_g, sig_g), "k--", lw=2, label=f"Gaussian (σ={sig_g:.3f})")
        a_sn, xi_sn, sig_sn = stats.skewnorm.fit(data)
        ax.plot(x, stats.skewnorm.pdf(x, a_sn, xi_sn, sig_sn), "C1-", lw=2,
                label=f"Skew-normal (γ={a_sn:.2f})")
        ax.set_xlabel(r"$\delta_g$")
        ax.set_ylabel("PDF")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.text(0.97, 0.97, f"skewness = {stats.skew(data):.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)

    plt.tight_layout()
    fpath = outdir / "fig3_delta_g_distribution.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fpath}")


def run_single_systematic(nside, n_real, n_mean, sigma_G, n_walkers, n_steps, n_burn,
                           templates, contam_params, outdir):
    import matplotlib.pyplot as plt

    n_pix = hp.nside2npix(nside)
    isys = 2
    a1 = np.array([contam_params.loc[isys, "a_true"]])
    b1 = np.array([contam_params.loc[isys, "b_true"]])
    templates_single = templates[isys:isys + 1]
    fam_1 = contam_params.loc[isys, "family"]

    theta_deg = np.logspace(np.log10(0.5 / 60), np.log10(3.0), 15)
    tmpl_full = np.zeros(n_pix)
    tmpl_full[:] = templates[isys]
    ct_theta = _anafast_wtheta(tmpl_full, nside, theta_deg)

    results = {"w_obs": [], "w_corr_add": [], "w_corr_mul": [], "w_corr_com": [], "w_true": []}

    for ir in range(n_real):
        rng = np.random.default_rng(1000 + ir)
        delta_true = _make_lognormal_delta(nside, sigma_G=sigma_G, seed=int(rng.integers(0, 2**31)))
        delta_cont = np.asarray(apply_contamination(jnp.asarray(delta_true), jnp.asarray(templates_single),
                                                     jnp.asarray(a1), jnp.asarray(b1)))
        lam = np.maximum(n_mean * (1.0 + delta_cont), 0.0)
        galaxy_counts = rng.poisson(lam).astype(float)
        random_counts = np.full(n_pix, float(n_mean * 10))
        delta_obs, good_pix = sm.compute_overdensity(galaxy_counts, random_counts)
        delta_t = sm.assign_template_values(templates_single, good_pix)

        map_obs = np.zeros(n_pix); map_obs[good_pix] = delta_obs
        map_true = np.zeros(n_pix); map_true[good_pix] = delta_true[good_pix]
        results["w_obs"].append(_anafast_wtheta(map_obs, nside, theta_deg))
        results["w_true"].append(_anafast_wtheta(map_true, nside, theta_deg))

        dt_rot, R, _ = rotate_templates(delta_t)
        for model in ("additive", "multiplicative", "combined"):
            flat_chain, _ = sm.run_mcmc(n_sys=1, model=model, delta_g_obs=delta_obs,
                                         delta_t=dt_rot, n_walkers=n_walkers, n_steps=n_steps,
                                         n_burn=n_burn, seed=ir, progress=False)
            theta_r = sm.get_mle_params(flat_chain)
            a_r, b_r, _, _ = unpack_params(theta_r, 1, model)
            a_orig, b_orig = transform_params_from_rotated(np.asarray(a_r), np.asarray(b_r), R)
            var_a, var_b = sm.get_param_variance_from_chain(flat_chain, 1, model)
            var_a_orig = np.diag(R.T @ np.diag(var_a) @ R)
            var_b_orig = np.diag(R.T @ np.diag(var_b) @ R)
            w_c = correct_two_point_function(results["w_obs"][-1], a_orig, b_orig,
                                              var_a_orig, var_b_orig, ct_theta[np.newaxis, :])
            results[f"w_corr_{model[:3]}"].append(w_c)
        print(f"  Single-sys realization {ir+1}/{n_real}", end="\r")

    print()

    fig, ax = plt.subplots(figsize=(7, 5))
    theta_am = theta_deg * 60
    w_true_m = np.mean(results["w_true"], axis=0)
    ax.loglog(theta_am, np.abs(w_true_m), "k:", lw=2, label="True")
    ax.loglog(theta_am, np.abs(np.mean(results["w_obs"], axis=0)), "k-.", lw=2, label="Observed")
    for model, label, color in [("add", "Additive", "grey"),
                                  ("mul", "Multiplicative", "darkorange"),
                                  ("com", "Combined", "C0")]:
        w_m = np.mean(results[f"w_corr_{model}"], axis=0)
        w_s = np.std(results[f"w_corr_{model}"], axis=0)
        ax.loglog(theta_am, np.abs(w_m), color=color, lw=2, label=label)
        ax.fill_between(theta_am, np.abs(w_m) - w_s, np.abs(w_m) + w_s, color=color, alpha=0.2)
    ax.set_xlabel(r"$\theta$ [arcmin]")
    ax.set_ylabel(r"$|w(\theta)|$")
    ax.set_title(f"Single systematic (family {fam_1}): a={a1[0]:.3f}, b={b1[0]:.3f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fpath = outdir / "fig4_single_systematic.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fpath}")
    return results, theta_deg


def run_25sys(nside, n_real, n_sys, n_mean, sigma_G, n_walkers, n_steps, n_burn,
              templates, contam_params, theta_deg, outdir):
    import matplotlib.pyplot as plt

    n_pix = hp.nside2npix(nside)
    a_true = contam_params["a_true"].values
    b_true = contam_params["b_true"].values

    ct_all = np.zeros((n_sys, len(theta_deg)))
    for i in range(n_sys):
        tmpl = np.zeros(n_pix); tmpl[:] = templates[i]
        ct_all[i] = _anafast_wtheta(tmpl, nside, theta_deg)

    results_25 = {m: {"w_obs": [], "w_corr": [], "w_true": []}
                  for m in ("additive", "multiplicative", "combined")}

    for ir in range(n_real):
        rng = np.random.default_rng(2000 + ir)
        delta_true = _make_lognormal_delta(nside, sigma_G=sigma_G, seed=int(rng.integers(0, 2**31)))
        delta_cont = np.asarray(apply_contamination(jnp.asarray(delta_true), jnp.asarray(templates),
                                                     jnp.asarray(a_true), jnp.asarray(b_true)))
        lam = np.maximum(n_mean * (1.0 + delta_cont), 0.0)
        galaxy_counts = rng.poisson(lam).astype(float)
        random_counts = np.full(n_pix, float(n_mean * 10))
        delta_obs, good_pix = sm.compute_overdensity(galaxy_counts, random_counts)
        delta_t = sm.assign_template_values(templates, good_pix)

        map_obs = np.zeros(n_pix); map_obs[good_pix] = delta_obs
        map_true = np.zeros(n_pix); map_true[good_pix] = delta_true[good_pix]
        w_obs_r = _anafast_wtheta(map_obs, nside, theta_deg)
        w_true_r = _anafast_wtheta(map_true, nside, theta_deg)

        dt_rot, R, _ = rotate_templates(delta_t)
        for model in ("additive", "multiplicative", "combined"):
            flat_chain, _ = sm.run_mcmc(n_sys=n_sys, model=model, delta_g_obs=delta_obs,
                                         delta_t=dt_rot, n_walkers=n_walkers, n_steps=n_steps,
                                         n_burn=n_burn, seed=ir + 100, progress=False)
            theta_r = sm.get_mle_params(flat_chain)
            a_rot, b_rot, _, _ = unpack_params(theta_r, n_sys, model)
            a_orig, b_orig = transform_params_from_rotated(np.asarray(a_rot), np.asarray(b_rot), R)
            cov_a, cov_b = sm.get_param_covariance_from_chain(flat_chain, n_sys, model)
            var_a = np.diag(R.T @ cov_a @ R)
            var_b = np.diag(R.T @ cov_b @ R)
            w_c = correct_two_point_function(w_obs_r, a_orig, b_orig, var_a, var_b, ct_all)
            results_25[model]["w_obs"].append(w_obs_r)
            results_25[model]["w_corr"].append(w_c)
            results_25[model]["w_true"].append(w_true_r)
        print(f"  25-sys realization {ir+1}/{n_real}", end="\r")

    print()

    theta_am = theta_deg * 60
    w_true_m = np.mean(results_25["combined"]["w_true"], axis=0)
    w_obs_m = np.mean(results_25["combined"]["w_obs"], axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5]})
    ax = axes[0]
    ax.loglog(theta_am, np.abs(w_true_m), "k:", lw=2.5, label="True", zorder=10)
    ax.loglog(theta_am, np.abs(w_obs_m), "k-.", lw=2, label="Observed")
    for model, color, ls in [("additive", "grey", "--"),
                              ("multiplicative", "darkorange", "--"),
                              ("combined", "C0", "-")]:
        w_m = np.mean(results_25[model]["w_corr"], axis=0)
        w_s = np.std(results_25[model]["w_corr"], axis=0)
        ax.loglog(theta_am, np.abs(w_m), color=color, ls=ls, lw=2, label=model.capitalize())
        ax.fill_between(theta_am, np.maximum(np.abs(w_m) - w_s, 1e-8),
                        np.abs(w_m) + w_s, color=color, alpha=0.15)
    ax.set_ylabel(r"$|w(\theta)|$")
    ax.set_title(f"25-systematic correction: {n_real} realisations, NSIDE={nside}")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.axhline(0, color="k", lw=1)
    for model, color, ls in [("additive", "grey", "--"),
                              ("multiplicative", "darkorange", "--"),
                              ("combined", "C0", "-")]:
        w_m = np.mean(results_25[model]["w_corr"], axis=0)
        frac = (w_m - w_true_m) / (np.abs(w_true_m) + 1e-8)
        ax2.semilogx(theta_am, frac, color=color, ls=ls, lw=2)
    ax2.set_xlabel(r"$\theta$ [arcmin]")
    ax2.set_ylabel(r"$(w_{\rm corr} - w_{\rm true})/|w_{\rm true}|$")
    ax2.set_ylim(-1.5, 1.5)

    plt.tight_layout()
    fpath = outdir / "fig5_25sys_correction.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fpath}")
    return results_25, ct_all


def run_table2(nside, n_real, n_sys, n_mean, sigma_G, n_walkers, n_steps, n_burn,
               templates, contam_params, table2_ref, outdir):
    n_pix = hp.nside2npix(nside)
    a_true = contam_params["a_true"].values
    b_true = contam_params["b_true"].values
    lrt_rows = []

    for ir in range(n_real):
        rng = np.random.default_rng(3000 + ir)
        delta_true = _make_lognormal_delta(nside, sigma_G=sigma_G, seed=int(rng.integers(0, 2**31)))
        delta_cont = np.asarray(apply_contamination(jnp.asarray(delta_true), jnp.asarray(templates),
                                                     jnp.asarray(a_true), jnp.asarray(b_true)))
        lam = np.maximum(n_mean * (1.0 + delta_cont), 0.0)
        galaxy_counts = rng.poisson(lam).astype(float)
        random_counts = np.full(n_pix, float(n_mean * 10))
        delta_obs, good_pix = sm.compute_overdensity(galaxy_counts, random_counts)
        delta_t = sm.assign_template_values(templates, good_pix)
        dt_rot, _, _ = rotate_templates(delta_t)

        row = {"realization": ir}
        for null_m, alt_m in [("additive", "combined"), ("multiplicative", "combined")]:
            fc_null, _ = sm.run_mcmc(n_sys=n_sys, model=null_m, delta_g_obs=delta_obs,
                                      delta_t=dt_rot, n_walkers=n_walkers, n_steps=n_steps,
                                      n_burn=n_burn, seed=ir + 200, progress=False)
            fc_alt, _ = sm.run_mcmc(n_sys=n_sys, model=alt_m, delta_g_obs=delta_obs,
                                     delta_t=dt_rot, n_walkers=n_walkers, n_steps=n_steps,
                                     n_burn=n_burn, seed=ir + 300, progress=False)
            lrt = likelihood_ratio_test(delta_obs, dt_rot, sm.get_mle_params(fc_null),
                                        sm.get_mle_params(fc_alt), null_model=null_m,
                                        alt_model=alt_m, significance=0.05)
            tag = null_m[:3]
            row[f"lambda_{tag}"] = float(lrt.lambda_lr)
            row[f"p_{tag}"] = float(lrt.p_value)
            row[f"reject_{tag}"] = bool(lrt.reject_null)

        lrt_rows.append(row)
        print(f"  LRT realization {ir+1}/{n_real}", end="\r")

    print()
    lrt_df = pd.DataFrame(lrt_rows)
    lrt_df.to_csv(outdir / "table2_lrt_results.csv", index=False)
    print(f"\n=== Table 2 ===")
    print(f"Additive vs. combined:        {lrt_df['reject_add'].mean()*100:.0f}% reject null (N={n_real})")
    print(f"Multiplicative vs. combined:  {lrt_df['reject_mul'].mean()*100:.0f}% reject null")
    if table2_ref is not None:
        print("\nPaper Table 2 (Gaussian):")
        g = table2_ref[table2_ref["likelihood_type"] == "gaussian"]
        print(g[["null_model", "alt_model", "fraction_preferred_pct"]].to_string(index=False))
    return lrt_df


def run_table3(nside, results_25, theta_deg, outdir):
    n_sys_25 = len(results_25["combined"]["w_corr"][0])
    cl_fid = sm.systematic_power_spectrum(nside, 2)
    w_fid = _legendre_sum(cl_fid, np.cos(np.radians(theta_deg)))
    w_fid = w_fid / max(np.abs(w_fid).max(), 1e-12)

    amp = {k: [] for k in ("observed", "additive", "multiplicative", "combined")}
    n_real = len(results_25["combined"]["w_true"])

    for ir in range(n_real):
        w_true_ir = results_25["combined"]["w_true"][ir]
        A_true = _fit_amplitude(w_true_ir, w_fid)
        if np.isnan(A_true) or abs(A_true) < 1e-10:
            continue
        amp["observed"].append(_fit_amplitude(results_25["combined"]["w_obs"][ir], w_fid) / A_true)
        for m in ("additive", "multiplicative", "combined"):
            amp[m].append(_fit_amplitude(results_25[m]["w_corr"][ir], w_fid) / A_true)

    rows = []
    print("\n=== Table 3 ===")
    print(f"{'Model':<22}  {'ΔĀ (%)':>8}  {'σ_A (%)':>8}")
    print("-" * 42)
    for key, label in [("observed", "Uncorrected"), ("additive", "Additive"),
                        ("multiplicative", "Multiplicative"), ("combined", "Combined")]:
        arr = np.asarray(amp[key])
        arr = arr[np.isfinite(arr)]
        dA = (np.mean(arr) - 1.0) * 100
        sA = np.std(arr) * 100
        print(f"{label:<22}  {dA:+8.1f}%  {sA:8.1f}%")
        rows.append({"model": label, "delta_A_mean_pct": dA, "sigma_A_pct": sA})
    pd.DataFrame(rows).to_csv(outdir / "table3_amplitude_bias.csv", index=False)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Berlfein 2024 on synthetic lognormal mocks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nside", type=int, default=64, help="HEALPix NSIDE (paper: 512).")
    parser.add_argument("--n-sys", type=int, default=25, help="Number of systematic templates.")
    parser.add_argument("--n-real", type=int, default=4, help="Number of realisations (paper: 119).")
    parser.add_argument("--n-walkers", type=int, default=110, help="MCMC walkers (paper: 250).")
    parser.add_argument("--n-steps", type=int, default=200, help="MCMC steps (paper: 1500).")
    parser.add_argument("--n-burn", type=int, default=50, help="MCMC burn-in steps (paper: 300).")
    parser.add_argument("--n-mean", type=int, default=40, help="Mean galaxies per pixel.")
    parser.add_argument("--sigma-G", type=float, default=0.5, help="Lognormal field width.")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed.")
    parser.add_argument("--output-dir", default="results/paper_validation/",
                        help="Directory where outputs are written.")
    parser.add_argument("--data-dir", default="data/",
                        help="Directory containing data/ (paper tables, simulation configs).")
    parser.add_argument("--figures", action="store_true", default=True,
                        help="Generate all figures.")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    np.random.seed(args.seed)

    print(f"sys_mapping paper validation")
    print(f"  NSIDE={args.nside}  n_sys={args.n_sys}  n_real={args.n_real}")
    print(f"  walkers={args.n_walkers}  steps={args.n_steps}  burn={args.n_burn}")
    print(f"  Output: {outdir}")

    contam_csv = data_dir / "simulation_config/contamination_params_25sys.csv"
    table2_csv = data_dir / "paper_tables/table2_lrt_results.csv"

    contam_params = pd.read_csv(contam_csv) if contam_csv.exists() else None
    table2_ref = pd.read_csv(table2_csv) if table2_csv.exists() else None

    if contam_params is None:
        rng = np.random.default_rng(args.seed)
        contam_params = pd.DataFrame({
            "family": [i % 5 for i in range(args.n_sys)],
            "a_true": rng.normal(0, 0.15, args.n_sys),
            "b_true": rng.normal(0, 0.15, args.n_sys),
        })
        print("WARNING: contamination_params_25sys.csv not found — using random parameters.")

    n_sys = min(args.n_sys, len(contam_params))
    contam_params = contam_params.iloc[:n_sys].reset_index(drop=True)
    families = list(contam_params["family"].values)
    templates = sm.generate_systematic_maps(args.nside, families=families, seed=args.seed)
    print(f"Templates: {templates.shape}")

    theta_deg = np.logspace(np.log10(0.5 / 60), np.log10(3.0), 15)

    if args.figures:
        print("\n--- Figure 2: template power spectra ---")
        run_fig2(args.nside, outdir)

        print("--- Figure 3: overdensity distribution ---")
        run_fig3(args.nside, args.n_mean, args.sigma_G, n_sys, contam_params, templates, outdir)

        print("--- Figure 4: single-systematic correction ---")
        run_single_systematic(args.nside, args.n_real, args.n_mean, args.sigma_G,
                               args.n_walkers, args.n_steps, args.n_burn,
                               templates, contam_params, outdir)

    print("--- Figure 5 / Table 3: 25-systematic correction ---")
    results_25, ct_all = run_25sys(args.nside, args.n_real, n_sys, args.n_mean, args.sigma_G,
                                    args.n_walkers, args.n_steps, args.n_burn,
                                    templates, contam_params, theta_deg, outdir)

    print("--- Table 2: likelihood ratio test ---")
    run_table2(args.nside, args.n_real, n_sys, args.n_mean, args.sigma_G,
                args.n_walkers, args.n_steps, args.n_burn,
                templates, contam_params, table2_ref, outdir)

    print("--- Table 3: amplitude bias ---")
    run_table3(args.nside, results_25, theta_deg, outdir)

    print(f"\nAll outputs written to {outdir}")


if __name__ == "__main__":
    main()
