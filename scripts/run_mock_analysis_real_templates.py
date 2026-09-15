"""Mock validation with real GAIA stellar and LS10 depth templates (synth_5, synth_6).

Generates synthetic galaxy catalogs contaminated by real observational
systematic maps and compares recovery accuracy across ALL implemented methods:

  - OLS (ordinary least squares, pixel-level regression)
  - ElasticNet (L1+L2 penalised regression, cross-validated)
  - ISD-1 (Iterative Systematics Decontamination, poly_order=1)
  - ISD-3 (Iterative Systematics Decontamination, poly_order=3)
  - MCMC-additive (Berlfein+2024, b_i=0 forced)
  - MCMC-combined (Berlfein+2024, free a_i and b_i)

Template set (7 templates total):
  synth_0 … synth_2  — synthetic lognormal fields (families 0, 1, 2)
  synth_5 (GAIA)     — faint-star surface density from GAIA DR3
  synth_6 (LS10)     — galaxy depth in z band from Legacy Survey DR10

The survey footprint is the LS10 depth footprint (pixels where depth > 0),
restricted to the South Galactic Cap (declination < 30°, as in LS10 south).
This gives a realistic ~22 000 pixel footprint at NSIDE=64.

All six methods run through ``sm.run_decontamination``: MCMC-add is the exact
analytic posterior and MCMC-comb BlackJAX NUTS.  The ISD stopping rule is
calibrated on uncontaminated mocks from the same generator.

Usage
-----
python scripts/run_mock_analysis_real_templates.py \\
    --syst-dir ~/data/legacysurvey/dr10/systematics/0064 \\
    --n-mocks 5 --nside 64 \\
    --output-dir docs/_static/results_real_template_validation/

# Faster test run (3 mocks, OLS and MCMC only):
python scripts/run_mock_analysis_real_templates.py \\
    --syst-dir ~/data/legacysurvey/dr10/systematics/0064 \\
    --n-mocks 3 --nside 64 --no-regression \\
    --output-dir /tmp/mock_real_test/
"""
import argparse
import json
import warnings
from pathlib import Path

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pandas as pd

import sys_mapping as sm
from sys_mapping.contamination import apply_contamination
from sys_mapping.correction import rotate_templates
from sys_mapping.model_selection import likelihood_ratio_test
from sys_mapping.maps import load_real_templates

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Mock generation ────────────────────────────────────────────────────────

def make_mock_real_templates(nside, templates, footprint_mask, a_true, b_true,
                              n_mean=30, seed=0):
    """Generate a synthetic galaxy catalog using real templates as systematics.

    The survey footprint is ``footprint_mask`` (e.g. the LS10 depth footprint).
    Contamination amplitudes a_true / b_true are injected with known values so
    that recovery can be assessed against ground truth.

    Returns
    -------
    ra_gal, dec_gal, ra_rand, dec_rand : arrays
        Galaxy and random sky positions (degrees).
    a_true, b_true : arrays
        The injected contamination parameters (passed through for bookkeeping).
    """
    n_pix = hp.nside2npix(nside)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1, dtype=float)
    cl_G = (ell + 1.0) ** (-2)
    cl_G[0] = 0.0
    cl_G *= 0.5**2 / np.sum((2 * ell + 1) / (4 * np.pi) * cl_G)
    G = hp.synfast(cl_G, nside=nside, lmax=lmax)
    delta_true = np.exp(G - 0.5 * 0.5**2) - 1.0

    delta_cont = np.asarray(
        apply_contamination(
            jnp.asarray(delta_true),
            jnp.asarray(templates),
            jnp.asarray(a_true),
            jnp.asarray(b_true),
        )
    )

    lam = np.maximum(n_mean * (1.0 + delta_cont), 0.0) * footprint_mask.astype(float)
    counts = rng.poisson(lam)
    rand_counts = np.round(footprint_mask.astype(float) * n_mean * 8).astype(int)

    pix_ra, pix_dec = hp.pix2ang(nside, np.arange(n_pix), lonlat=True)
    gal_idx = np.repeat(np.arange(n_pix), counts)
    rand_idx = np.repeat(np.arange(n_pix), rand_counts)

    return (
        pix_ra[gal_idx], pix_dec[gal_idx],
        pix_ra[rand_idx], pix_dec[rand_idx],
        a_true, b_true,
    )


# ── Per-mock analysis — ALL methods ────────────────────────────────────────

def analyse_mock_all_methods(mock_id, ra_gal, dec_gal, ra_rand, dec_rand,
                              templates, nside, a_true, b_true, *,
                              run_regression=True, isd_chi2_68=None, mcmc_kw=None):
    """Run all implemented methods on one mock and return a flat result dict.

    Every method goes through ``sm.run_decontamination``, the production entry
    point: MCMC-add is the exact analytic posterior and MCMC-comb BlackJAX NUTS
    (``sampler="auto"``) unless ``mcmc_kw`` names another sampler.
    """
    gal_counts = sm.pixelize_catalog(ra_gal, dec_gal, nside)
    rand_counts = sm.pixelize_catalog(ra_rand, dec_rand, nside)
    delta_g, good_pix = sm.compute_overdensity(gal_counts, rand_counts)
    delta_t = sm.assign_template_values(templates, good_pix)

    result = {
        "mock_id": mock_id,
        "n_galaxies": int(len(ra_gal)),
        "n_good_pix": int(good_pix.sum()),
        "a_true": list(float(x) for x in a_true),
        "b_true": list(float(x) for x in b_true),
    }
    a_true = np.asarray(a_true)
    b_true = np.asarray(b_true)

    methods = [("OLS", "ols"), ("MCMC-add", "mcmc_add"), ("MCMC-comb", "mcmc_comb")]
    if run_regression:
        methods[1:1] = [("ElasticNet", "elasticnet"), ("ISD-1", "isd1"), ("ISD-3", "isd3")]

    fits = {}
    for method, key in methods:
        kw = {"seed": mock_id, "progress": False}
        if method == "ElasticNet":
            kw["cv_folds"] = 3
        if method.startswith("ISD") and isd_chi2_68 is not None:
            kw["isd_chi2_68"] = isd_chi2_68[method]
        if method.startswith("MCMC"):
            kw.update(mcmc_kw or {})
        try:
            res = sm.run_decontamination(method, delta_g, delta_t, **kw)
        except Exception as exc:          # a failed method must not lose the mock
            result[f"a_{key}_error"] = str(exc)
            continue
        fits[method] = res
        a_hat = np.asarray(res["a_hat"])
        result[f"a_{key}"] = a_hat.tolist()
        result[f"a_{key}_bias"] = (a_hat - a_true).tolist()
        result[f"time_{key}_s"] = float(res["elapsed_s"])
        if method.startswith("ISD"):
            result[f"{key}_n_steps"] = int(res["n_iterations"])
            result[f"{key}_stopped_on"] = res["isd_stopped_on"]

    comb = fits.get("MCMC-comb")
    if comb is not None:
        b_hat_comb = np.asarray(comb["b_hat"])
        result["b_mcmc_comb"] = b_hat_comb.tolist()
        result["b_mcmc_comb_bias"] = (b_hat_comb - b_true).tolist()
        result["sd_a_mcmc_comb"] = np.sqrt(np.diag(comb["cov_a"])).tolist()
        result["sd_b_mcmc_comb"] = np.sqrt(np.diag(comb["cov_b"])).tolist()
        smp = comb.get("sampler")
        if getattr(smp, "num_divergences", None) is not None:
            result["nuts_num_divergences"] = int(smp.num_divergences)
        if getattr(smp, "rhat", None) is not None:
            result["nuts_max_rhat"] = float(smp.rhat)
            result["nuts_min_ess"] = float(smp.ess)
    add = fits.get("MCMC-add")
    if add is not None:
        result["sd_a_mcmc_add"] = np.sqrt(np.diag(add["cov_a"])).tolist()

    # ── Likelihood ratio test ──────────────────────────────────────────────
    # Between the additive and combined maxima, in the PCA-rotated basis the
    # MCMC fits use.
    delta_t_rot, _, _ = rotate_templates(delta_t)
    mx = sm.lrt_from_maxima(delta_g, delta_t_rot)
    lrt = likelihood_ratio_test(
        delta_g, delta_t_rot, mx["theta_null"][0], mx["theta_alt"][0],
        null_model="additive", alt_model="combined", significance=0.05,
    )
    result["lrt_lambda"] = float(lrt.lambda_lr)
    result["lrt_p"] = float(lrt.p_value)
    result["lrt_reject"] = bool(lrt.reject_null)
    result["lrt_converged"] = bool(mx["converged"][0])
    result["lrt_max_grad"] = float(mx["max_grad"][0])

    return result


def calibrate_isd(nside, templates, footprint_mask, n_mean, n_mocks, orders=(1, 3)):
    """68th percentile per template of the ISD Delta chi^2 on uncontaminated mocks.

    The mocks come from ``make_mock_real_templates`` with every amplitude zero, so
    the null has the clustering, shot noise and footprint of the contaminated mocks.
    The statistic is the first-step marginal fit of
    ``iterative_systematics_decontamination`` (10 quantile bins) at each order.
    """
    from sys_mapping.diagnostics import isd_marginal_fit

    n_sys = templates.shape[0]
    dchi2 = {order: [] for order in orders}
    for k in range(n_mocks):
        ra_g, dec_g, ra_r, dec_r, _, _ = make_mock_real_templates(
            nside, templates, footprint_mask, np.zeros(n_sys), np.zeros(n_sys),
            n_mean=n_mean, seed=100_000 + k,
        )
        delta_g, good = sm.compute_overdensity(sm.pixelize_catalog(ra_g, dec_g, nside),
                                               sm.pixelize_catalog(ra_r, dec_r, nside))
        delta_t = sm.assign_template_values(templates, good)
        for order in orders:
            dchi2[order].append(isd_marginal_fit(delta_g, delta_t, poly_order=order)[0])
    return {f"ISD-{order}": np.percentile(np.array(v), 68, axis=0)
            for order, v in dchi2.items()}


# ── Summary and plots ──────────────────────────────────────────────────────

def write_summary(results, template_names, outdir, *, run_regression=True):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    df = pd.DataFrame(results)
    df.to_csv(outdir / "mock_results_real_templates.csv", index=False)

    n_sys = len(results[0]["a_true"])
    template_labels = [n[:14] for n in template_names[:n_sys]]

    methods_a = {
        "OLS":           "a_ols_bias",
        "MCMC-add":      "a_mcmc_add_bias",
        "MCMC-comb (a)": "a_mcmc_comb_bias",
    }
    if run_regression:
        methods_a = {
            "OLS":           "a_ols_bias",
            "ElasticNet":    "a_elasticnet_bias",
            "ISD-1":         "a_isd1_bias",
            "ISD-3":         "a_isd3_bias",
            "MCMC-add":      "a_mcmc_add_bias",
            "MCMC-comb (a)": "a_mcmc_comb_bias",
        }

    # ── Parameter recovery: a_i ────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, len(methods_a), figsize=(3.5 * len(methods_a), 4), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, (label, col) in zip(axes, methods_a.items()):
        if col not in df.columns:
            ax.set_title(label + "\n(n/a)")
            continue
        bias = np.array(df[col].tolist())
        ax.boxplot(bias, vert=True, widths=0.6)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Template", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel(r"$\hat{a}_i - a_i^{\rm true}$", fontsize=9)
        ax.set_xticks(range(1, n_sys + 1))
        ax.set_xticklabels(template_labels, rotation=45, ha="right", fontsize=6)
    fig.suptitle(
        f"Additive parameter recovery bias — {len(results)} mocks — real templates",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(outdir / "real_template_a_recovery.png", dpi=130, bbox_inches="tight")
    plt.close()

    # ── Parameter recovery: b_i (MCMC-combined only) ───────────────────────
    if "b_mcmc_comb_bias" in df.columns:
        b_bias = np.array(df["b_mcmc_comb_bias"].tolist())
        fig, ax = plt.subplots(figsize=(max(4, n_sys * 0.8), 4))
        ax.boxplot(b_bias, vert=True)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Template", fontsize=9)
        ax.set_ylabel(r"$\hat{b}_i - b_i^{\rm true}$ (MCMC-combined)", fontsize=9)
        ax.set_title(f"Multiplicative recovery — {len(results)} mocks", fontsize=10)
        ax.set_xticks(range(1, n_sys + 1))
        ax.set_xticklabels(template_labels, rotation=45, ha="right", fontsize=7)
        plt.tight_layout()
        plt.savefig(
            outdir / "real_template_b_recovery.png", dpi=130, bbox_inches="tight"
        )
        plt.close()

    # ── RMS bias comparison across methods ─────────────────────────────────
    rms_rows = []
    for label, col in methods_a.items():
        if col in df.columns:
            bias = np.array(df[col].tolist())
            rms_rows.append((label, float(np.sqrt(np.mean(bias**2)))))
    if rms_rows:
        method_labels, rms_vals = zip(*rms_rows)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(method_labels, rms_vals, color="steelblue", alpha=0.8)
        ax.set_ylabel(r"RMS$(|\hat{a}_i - a_i^{\rm true}|)$", fontsize=10)
        ax.set_title("Method comparison — real templates", fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(
            outdir / "real_template_method_rms.png", dpi=130, bbox_inches="tight"
        )
        plt.close()

    # ── LRT statistics ─────────────────────────────────────────────────────
    if "lrt_lambda" in df.columns:
        reject_frac = float(df["lrt_reject"].mean())
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["lrt_lambda"], bins=20, color="steelblue", alpha=0.7)
        ax.set_xlabel(r"$\lambda_{\rm LR}$", fontsize=10)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(
            f"LRT statistics — real templates ({len(results)} mocks)  "
            f"reject fraction: {reject_frac:.0%}",
            fontsize=10,
        )
        plt.tight_layout()
        plt.savefig(
            outdir / "real_template_lrt_statistics.png", dpi=130, bbox_inches="tight"
        )
        plt.close()

    # ── Console summary ────────────────────────────────────────────────────
    print("\n=== Mock analysis with real templates — summary ===")
    print(f"  N mocks:           {len(results)}")
    print(f"  Mean N_gal:        {df['n_galaxies'].mean():.0f}")
    print(f"  Mean N_good_pix:   {df['n_good_pix'].mean():.0f}")
    print(f"  LRT reject frac (add vs comb): {reject_frac:.0%}")
    print("\n  RMS bias per method:")
    for label, rms in rms_rows:
        print(f"    {label:20s}: {rms:.5f}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mock validation with real GAIA/LS10 templates — all methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--syst-dir",
        default="~/data/legacysurvey/dr10/systematics",
        help="Directory containing GAIA and LS10 FITS files.",
    )
    parser.add_argument("--nside", type=int, default=64)
    parser.add_argument("--n-mocks", type=int, default=10)
    parser.add_argument("--sampler", default="auto",
                        choices=["auto", "analytic", "nuts", "emcee"],
                        help="MCMC backend of run_decontamination: auto is the analytic "
                             "posterior for MCMC-add and NUTS for MCMC-comb.")
    parser.add_argument("--nuts-warmup", type=int, default=1000,
                        help="NUTS window-adaptation steps (MCMC-comb).")
    parser.add_argument("--nuts-samples", type=int, default=1000,
                        help="NUTS draws per chain (MCMC-comb).")
    parser.add_argument("--n-walkers", type=int, default=100, help="emcee only (--sampler emcee).")
    parser.add_argument("--n-steps", type=int, default=500, help="emcee only (--sampler emcee).")
    parser.add_argument("--n-burn", type=int, default=100, help="emcee only (--sampler emcee).")
    parser.add_argument("--isd-n-mocks", type=int, default=50,
                        help="Uncontaminated mocks calibrating the ISD stopping rule "
                             "(0 leaves it uncalibrated).")
    parser.add_argument("--n-mean", type=int, default=30,
                        help="Mean galaxies per pixel.")
    parser.add_argument("--n-synth", type=int, default=3,
                        help="Number of synthetic template families (0..n-1).")
    parser.add_argument("--a-sigma", type=float, default=0.10,
                        help="Std dev of injected a_i ~ N(0, a-sigma).")
    parser.add_argument("--b-sigma", type=float, default=0.10,
                        help="Std dev of injected b_i ~ N(0, b-sigma).")
    parser.add_argument("--no-regression", action="store_true",
                        help="Skip ElasticNet and ISD (faster).")
    parser.add_argument("--output-dir", default="results/mock_real_templates/")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    syst_dir = Path(args.syst_dir).expanduser()
    print(f"Loading real templates from {syst_dir} ...")
    real_tmpl, real_names, footprint_mask = sm.load_real_templates(args.nside, syst_dir)
    # synth_5 = GAIA_nstar_faint, synth_6 = LS10_GALDEPTH_Z

    synth_tmpl = sm.generate_systematic_maps(
        args.nside, families=list(range(args.n_synth)), seed=0
    )
    synth_names = [f"synth_{i}" for i in range(args.n_synth)]

    # Stack: synth_0..n-1 first, then real templates (synth_5, synth_6)
    templates = np.vstack([synth_tmpl, real_tmpl])
    template_names = synth_names + real_names
    n_sys = templates.shape[0]

    print(f"Templates ({n_sys}): {template_names}")
    print(f"Survey footprint: {footprint_mask.sum():,} pixels "
          f"({footprint_mask.mean()*100:.1f}% of sky at NSIDE={args.nside})")

    mcmc_kw = dict(sampler=args.sampler, nuts_n_warmup=args.nuts_warmup,
                   nuts_n_samples=args.nuts_samples, n_walkers=args.n_walkers,
                   n_steps=args.n_steps, n_burn=args.n_burn)
    run_regression = not args.no_regression

    isd_chi2_68 = None
    if run_regression and args.isd_n_mocks > 0:
        print(f"Calibrating the ISD stopping rule on {args.isd_n_mocks} uncontaminated mocks ...")
        isd_chi2_68 = calibrate_isd(args.nside, templates, footprint_mask, args.n_mean,
                                    args.isd_n_mocks)
        for name, c68 in isd_chi2_68.items():
            print(f"  {name} chi2_68 = {np.array2string(c68, precision=2)}")
        (outdir / "isd_chi2_68.json").write_text(json.dumps(
            {"template_names": template_names, "n_mocks": args.isd_n_mocks,
             **{k: v.tolist() for k, v in isd_chi2_68.items()}}, indent=2))

    rng = np.random.default_rng(42)
    results = []

    for im in range(args.n_mocks):
        a_true = rng.normal(0, args.a_sigma, n_sys)
        b_true = rng.normal(0, args.b_sigma, n_sys)

        ra_g, dec_g, ra_r, dec_r, at, bt = make_mock_real_templates(
            args.nside, templates, footprint_mask,
            a_true, b_true, n_mean=args.n_mean, seed=im,
        )
        print(f"Mock {im+1}/{args.n_mocks}: {len(ra_g):,} galaxies, "
              f"{len(ra_r):,} randoms", flush=True)

        res = analyse_mock_all_methods(
            im, ra_g, dec_g, ra_r, dec_r,
            templates, args.nside,
            a_true=at, b_true=bt,
            run_regression=run_regression,
            isd_chi2_68=isd_chi2_68, mcmc_kw=mcmc_kw,
        )
        results.append(res)
        (outdir / f"mock_{im:04d}_results.json").write_text(
            json.dumps(res, indent=2)
        )
        print(f"  LRT λ={res['lrt_lambda']:.1f}  reject={res['lrt_reject']}  "
              f"MCMC-comb |a_bias|={np.abs(res['a_mcmc_comb_bias']).mean():.4f}")

    write_summary(results, template_names, outdir, run_regression=run_regression)
    print(f"\nResults written to {outdir}/")


if __name__ == "__main__":
    main()
