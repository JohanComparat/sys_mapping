"""Multi-method, multi-scenario validation of sys_mapping.

Generates synthetic galaxy mocks under four contamination scenarios, applies
six decontamination methods to each, and compares their performance against
the known ground truth.

Scenarios
---------
none          Pure lognormal field, no systematics.
additive      δ̂_g = δ_g + Σ a_i t_i
multiplicative δ̂_g = δ_g (1 + Σ b_i t_i)
combined      δ̂_g = δ_g (1 + Σ b_i t_i) + Σ a_i t_i

Methods
-------
For each scenario the following methods are applied:
  1. OLS (ordinary least squares, no regularisation)
  2. ElasticNet (L1+L2 regularised regression, with cross-validation)
  3. ISD-1 (Iterative Systematics Decontamination, poly order 1)
  4. ISD-3 (Iterative Systematics Decontamination, poly order 3)
  5. MCMC-additive (Bayesian MCMC, additive model only)
  6. MCMC-combined (Bayesian MCMC, joint additive + multiplicative)

For each method the following metrics are computed:
  - chi2_resid: mean squared residual in the pixel overdensity
  - corr_with_true: Pearson r between recovered and true δ_g
  - rms_delta_error: RMS of (recovered - true) δ_g
  - amplitude recovery bias and scatter (MCMC methods, compared to injected truth)
  - null_test r: maximum |r(weights, template)| after weighting

Usage
-----
# Quick run (NSIDE=16, small MCMC) — ~3 min
python scripts/run_validation.py --nside 16 --n-sys 3 --n-mean 50 \\
    --n-walkers 60 --n-steps 300 --n-burn 60 \\
    --output-dir docs/_static/results_validation

# Science run (NSIDE=32) — ~15 min
python scripts/run_validation.py --nside 32 --n-sys 3 --n-mean 50 \\
    --n-walkers 100 --n-steps 600 --n-burn 100 \\
    --output-dir docs/_static/results_validation
"""
import argparse
import json
import warnings
from pathlib import Path

import healpy as hp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys_mapping as sm
from sys_mapping.mocks import make_mock_suite, MockCatalog, make_mock_catalog
from sys_mapping.covariance import mock_sandwich_covariance

warnings.filterwarnings("ignore")

SCENARIOS = ["none", "additive", "multiplicative", "combined"]
METHOD_LABELS = {
    "ols": "OLS",
    "elasticnet": "ElasticNet",
    "isd1": "ISD (order 1)",
    "isd3": "ISD (order 3)",
    "mcmc_add": "MCMC-additive",
    "mcmc_comb": "MCMC-combined",
}
METHOD_COLORS = {
    "ols": "#e41a1c",
    "elasticnet": "#ff7f00",
    "isd1": "#4daf4a",
    "isd3": "#377eb8",
    "mcmc_add": "#984ea3",
    "mcmc_comb": "#a65628",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recovery_metrics(
    delta_g_obs: np.ndarray,
    delta_g_recovered: np.ndarray,
    delta_g_true: np.ndarray,
) -> dict:
    """Compute recovery quality metrics against the true density field."""
    resid = delta_g_recovered - delta_g_true
    obs_resid = delta_g_obs - delta_g_true

    g_centered = delta_g_true - delta_g_true.mean()
    r_centered = delta_g_recovered - delta_g_recovered.mean()
    denom = np.sqrt(np.sum(g_centered**2) * np.sum(r_centered**2))
    corr = float(np.sum(g_centered * r_centered) / denom) if denom > 0 else 0.0

    return {
        "chi2_resid": float(np.mean(resid**2)),
        "chi2_obs": float(np.mean(obs_resid**2)),
        "corr_with_true": corr,
        "rms_delta_error": float(np.sqrt(np.mean(resid**2))),
        "rms_obs_error": float(np.sqrt(np.mean(obs_resid**2))),
    }


# ---------------------------------------------------------------------------
# Per-method decontamination
# ---------------------------------------------------------------------------

def _weights_to_recovered_field(
    res: dict,
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    key: str,
) -> np.ndarray:
    """Apply per-pixel weights to recover the (approximately) clean field.

    For MCMC-comb (combined contamination model δ_obs = a@t + (1+b@t)δ_true),
    the exact correction is (δ_obs − a@t)/(1+b@t).  For all other methods
    (additive model) the standard weight formula w*(1+δ_obs)−1 is used.
    """
    w = res["weights"]
    if key == "mcmc_comb" and res.get("a_hat") is not None:
        _denom = np.maximum(1.0 + np.asarray(res["b_hat"]) @ delta_t, 1e-6)
        return (delta_g_obs - np.asarray(res["a_hat"]) @ delta_t) / _denom
    return w * (1.0 + delta_g_obs) - 1.0


_METHOD_ORDER = [
    ("ols",        "OLS"),
    ("elasticnet", "ElasticNet"),
    ("isd1",       "ISD-1"),
    ("isd3",       "ISD-3"),
    ("mcmc_add",   "MCMC-add"),
    ("mcmc_comb",  "MCMC-comb"),
]


# ---------------------------------------------------------------------------
# Per-scenario analysis
# ---------------------------------------------------------------------------

def analyse_scenario(
    mock: MockCatalog,
    n_walkers: int,
    n_steps: int,
    n_burn: int,
    seed: int,
) -> dict:
    """Run all methods on one mock and collect results."""
    nside = mock.nside
    n_sys = mock.n_sys

    # Pixelise and compute overdensity from the catalog
    gal_counts = sm.pixelize_catalog(mock.ra_gal, mock.dec_gal, nside)
    rand_counts = sm.pixelize_catalog(mock.ra_rand, mock.dec_rand, nside)
    delta_g_obs, good_pix = sm.compute_overdensity(gal_counts, rand_counts)
    delta_t = sm.assign_template_values(mock.templates, good_pix)
    delta_g_true = mock.delta_true[good_pix]

    results = {
        "scenario": mock.scenario,
        "n_gal": mock.n_gal,
        "n_good_pix": int(good_pix.sum()),
        "a_true": mock.a_true.tolist(),
        "b_true": mock.b_true.tolist(),
        "obs_metrics": recovery_metrics(delta_g_obs, delta_g_obs, delta_g_true),
        "methods": {},
        "delta_g_obs": delta_g_obs,
        "delta_g_true": delta_g_true,
        "delta_t": delta_t,
        "good_pix": good_pix,
    }

    for key, sm_method in _METHOD_ORDER:
        seed_kw = seed + 1 if key == "mcmc_comb" else seed
        try:
            res = sm.run_decontamination(
                sm_method, delta_g_obs, delta_t,
                n_walkers=n_walkers, n_steps=n_steps, n_burn=n_burn,
                seed=seed_kw, progress=False,
            )
        except ImportError:
            continue  # scikit-learn missing
        w = res["weights"]
        delta_rec = _weights_to_recovered_field(res, delta_g_obs, delta_t, key)
        entry = {
            "alpha": res["a_hat"].tolist(),
            "weights": w,
            "delta_recovered": delta_rec,
            "metrics": recovery_metrics(delta_g_obs, delta_rec, delta_g_true),
            "null_test": float(np.max(np.abs(
                sm.null_test_cross_correlations(w, delta_t, n_bootstrap=50, seed=0)["correlations"]
            ))),
        }
        if key in ("mcmc_add", "mcmc_comb"):
            entry.update({
                "a_hat": res["a_hat"].tolist(),
                "b_hat": res["b_hat"].tolist(),
                "a_bias": (res["a_hat"] - mock.a_true).tolist(),
                "b_bias": (res["b_hat"] - mock.b_true).tolist(),
            })
        if key in ("isd1", "isd3"):
            entry["n_iterations"] = res["n_iterations"]
        results["methods"][key] = entry

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_density_maps(results, nside, outdir, scenario):
    """Plot pixel-level density maps: true, observed, and recovered by each method."""
    n_pix = hp.nside2npix(nside)
    good_pix = results["good_pix"]
    delta_g_obs = results["delta_g_obs"]
    delta_g_true = results["delta_g_true"]
    methods = results["methods"]

    method_names = list(methods.keys())
    n_methods = len(method_names)
    n_cols = 4
    n_rows = 2 + (n_methods + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.5, wspace=0.3)
    vmin, vmax = -1.5, 1.5

    def make_full_map(delta):
        m = np.full(n_pix, hp.UNSEEN)
        m[good_pix] = delta
        return m

    def add_map(ax, delta, title):
        m = make_full_map(delta)
        hp.mollview(m, fig=fig.number, title=title,
                    min=vmin, max=vmax, cmap="RdYlBu_r",
                    notext=True, sub=(n_rows, n_cols, ax + 1))

    ax_idx = 1
    hp.mollview(make_full_map(delta_g_true), fig=fig.number,
                title=f"True $\\delta_g$ ({scenario})",
                min=vmin, max=vmax, cmap="RdYlBu_r", notext=True,
                sub=(n_rows, n_cols, ax_idx)); ax_idx += 1

    hp.mollview(make_full_map(delta_g_obs), fig=fig.number,
                title="Observed $\\hat{\\delta}_g$",
                min=vmin, max=vmax, cmap="RdYlBu_r", notext=True,
                sub=(n_rows, n_cols, ax_idx)); ax_idx += 1
    ax_idx += 2  # skip to next row

    for k in method_names:
        rec = methods[k]["delta_recovered"]
        r2 = methods[k]["metrics"]["corr_with_true"]
        hp.mollview(make_full_map(rec), fig=fig.number,
                    title=f"{METHOD_LABELS[k]}\n$r={r2:.3f}$",
                    min=vmin, max=vmax, cmap="RdYlBu_r", notext=True,
                    sub=(n_rows, n_cols, ax_idx))
        ax_idx += 1

    plt.suptitle(f"Scenario: {scenario}", fontsize=14, y=1.01)
    out = outdir / f"map_{scenario}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    return out


def plot_delta_histogram(results, outdir, scenario):
    """Histogram of δ_g: true, observed, and per-method recovered."""
    delta_g_obs = results["delta_g_obs"]
    delta_g_true = results["delta_g_true"]
    methods = results["methods"]

    bins = np.linspace(-2.5, 2.5, 60)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delta_g_true, bins=bins, histtype="step", lw=2, color="k", label="True")
    ax.hist(delta_g_obs, bins=bins, histtype="step", lw=2, color="gray",
            ls="--", label="Observed")
    for k, mres in methods.items():
        ax.hist(mres["delta_recovered"], bins=bins, histtype="step", lw=1.2,
                color=METHOD_COLORS[k], label=METHOD_LABELS[k])
    ax.set_xlabel(r"$\delta_g$", fontsize=12)
    ax.set_ylabel("Pixel count", fontsize=12)
    ax.set_title(f"δ_g distribution — scenario: {scenario}", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    out = outdir / f"hist_{scenario}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    return out


def plot_scatter_true_vs_recovered(results, outdir, scenario):
    """Scatter plots of recovered δ_g vs. true δ_g for each method."""
    delta_g_true = results["delta_g_true"]
    methods = results["methods"]
    n = len(methods)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    lim = 2.5
    x_line = np.linspace(-lim, lim, 50)

    for ax, (k, mres) in zip(axes_flat, methods.items()):
        rec = mres["delta_recovered"]
        r = mres["metrics"]["corr_with_true"]
        rms = mres["metrics"]["rms_delta_error"]
        ax.hexbin(delta_g_true, rec, gridsize=50, cmap="Blues", mincnt=1,
                  extent=[-lim, lim, -lim, lim])
        ax.plot(x_line, x_line, "r--", lw=1)
        ax.set_title(f"{METHOD_LABELS[k]}\n$r={r:.3f}$, RMS={rms:.3f}", fontsize=10)
        ax.set_xlabel(r"$\delta_g^{\rm true}$", fontsize=10)
        ax.set_ylabel(r"$\delta_g^{\rm rec}$", fontsize=10)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.suptitle(f"Recovered vs. true δ_g — scenario: {scenario}", fontsize=13)
    plt.tight_layout()
    out = outdir / f"scatter_{scenario}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    return out


def plot_weight_maps(results, nside, outdir, scenario):
    """Mollweide maps of per-pixel weights for each method."""
    n_pix = hp.nside2npix(nside)
    good_pix = results["good_pix"]
    methods = results["methods"]
    n = len(methods)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(5 * ncols, 3.5 * nrows))
    for idx, (k, mres) in enumerate(methods.items()):
        w = mres["weights"]
        m = np.full(n_pix, hp.UNSEEN)
        m[good_pix] = w
        hp.mollview(m, fig=fig.number, title=METHOD_LABELS[k],
                    min=0.5, max=1.5, cmap="coolwarm", notext=True,
                    sub=(nrows, ncols, idx + 1))

    plt.suptitle(f"Systematic weights — scenario: {scenario}", fontsize=13)
    out = outdir / f"weights_{scenario}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    return out


def plot_summary_metrics(all_results, outdir):
    """Summary bar chart of RMS δ_g error and correlation with truth per method × scenario."""
    metric_keys = ["rms_delta_error", "corr_with_true"]
    metric_labels = {"rms_delta_error": r"RMS$(\delta_g^{\rm rec} - \delta_g^{\rm true})$",
                     "corr_with_true": r"$r(\delta_g^{\rm rec},\, \delta_g^{\rm true})$"}

    scenarios = [r["scenario"] for r in all_results]
    # Collect methods from first result
    method_keys = list(all_results[0]["methods"].keys())

    for mk in metric_keys:
        fig, axes = plt.subplots(1, len(scenarios),
                                 figsize=(4.5 * len(scenarios), 5), sharey=True)
        if len(scenarios) == 1:
            axes = [axes]

        for ax, res in zip(axes, all_results):
            values = []
            labels = []
            colors = []
            # Add "no correction" baseline
            values.append(res["obs_metrics"][mk])
            labels.append("No correction")
            colors.append("lightgray")
            for k in method_keys:
                values.append(res["methods"][k]["metrics"][mk])
                labels.append(METHOD_LABELS[k])
                colors.append(METHOD_COLORS[k])

            bars = ax.bar(range(len(values)), values, color=colors, edgecolor="k", lw=0.5)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(["No corr."] + [METHOD_LABELS[k] for k in method_keys],
                                rotation=40, ha="right", fontsize=8)
            ax.set_title(res["scenario"], fontsize=11)

        axes[0].set_ylabel(metric_labels[mk], fontsize=11)
        plt.suptitle(metric_labels[mk] + " by scenario and method", fontsize=13)
        plt.tight_layout()
        out = outdir / f"summary_{mk}.png"
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close()


def sandwich_sigma_a(nside, n_sys, n_mean, sigma, seed, ref_good, ref_delta_t, n_mock=100):
    """Per-template additive 1σ from the mock-covariance sandwich (correlated-noise error).

    The single-fit MCMC posterior std underestimates the additive error because the clean field is
    spatially correlated (see :doc:`results_validation`).  Instead, estimate ``Cov(â)`` directly from
    an ensemble of *uncontaminated* mock reconstructions (the fit noise ε) with
    :func:`sys_mapping.mock_sandwich_covariance` — exact for the linear estimator and calibrated.

    Returns ``(sigma_a (n_sys,), n_pix_used)`` on the reference footprint ``ref_good``.
    """
    npix = hp.nside2npix(nside)
    fields = []
    for i in range(n_mock):
        m = make_mock_catalog(nside, n_sys, scenario="none",
                              n_mean=n_mean, sigma=sigma, seed=seed + 5000 + i)
        gc = sm.pixelize_catalog(m.ra_gal, m.dec_gal, nside)
        rc = sm.pixelize_catalog(m.ra_rand, m.dec_rand, nside)
        dobs, good = sm.compute_overdensity(gc, rc)
        full = np.full(npix, np.nan)
        full[good] = dobs
        fields.append(full[ref_good])
    fields = np.asarray(fields)                       # (n_mock, n_ref)
    ok = np.all(np.isfinite(fields), axis=0)          # pixels populated in every mock
    cov = mock_sandwich_covariance(ref_delta_t[:, ok], fields[:, ok])
    return np.sqrt(np.clip(np.diag(cov), 0.0, None)), int(ok.sum())


def plot_amplitude_recovery(all_results, outdir, sigma_a=None):
    """Bar charts of amplitude bias per method (MCMC methods only).

    ``sigma_a`` (n_sys,) is the mock-covariance sandwich 1σ for the additive parameter — the
    correlated-noise-calibrated error, drawn as error bars on the additive-bias bars so recovery can
    be read as "unbiased within the calibrated uncertainty".  ``b`` has no calibrated single-fit
    error (weakly identified); its bars are shown without error bars.
    """
    mcmc_methods = ["mcmc_add", "mcmc_comb"]
    scenarios_with_a = ["additive", "combined"]
    scenarios_with_b = ["multiplicative", "combined"]

    fig, axes = plt.subplots(2, len(scenarios_with_a),
                             figsize=(6 * len(scenarios_with_a), 8))
    axes = np.atleast_2d(axes)

    for j, scenario in enumerate(scenarios_with_a):
        res = next((r for r in all_results if r["scenario"] == scenario), None)
        if res is None:
            continue

        for row, amp in enumerate(["a", "b"]):
            ax = axes[row, j]
            true_vals = res[f"{amp}_true"]
            n_sys = len(true_vals)
            x = np.arange(n_sys)
            width = 0.35

            # Calibrated 1σ band: sandwich for additive a; b has no calibrated single-fit error.
            yerr = sigma_a if (amp == "a" and sigma_a is not None) else None
            for mi, mk in enumerate(mcmc_methods):
                if mk not in res["methods"]:
                    continue
                bias = res["methods"][mk][f"{amp}_bias"]
                ax.bar(x + mi * width, bias, width=width, yerr=yerr,
                       color=METHOD_COLORS[mk], alpha=0.8,
                       error_kw=dict(ecolor="0.25", capsize=3, lw=1.0),
                       label=METHOD_LABELS[mk])

            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels([f"t{i}" for i in range(n_sys)])
            ax.set_xlabel("Template index", fontsize=10)
            ax.set_ylabel(f"$\\hat{{{amp}}}_i - {amp}_i^{{\\rm true}}$", fontsize=10)
            ax.set_title(f"{'Additive' if amp=='a' else 'Multiplicative'} bias — {scenario}",
                         fontsize=11)
            ax.legend(fontsize=9)

    plt.tight_layout()
    out = outdir / "amplitude_recovery.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()


def plot_null_tests(all_results, outdir):
    """Bar chart of maximum |r(w, t)| for each method and scenario."""
    method_keys = list(all_results[0]["methods"].keys())
    scenarios = [r["scenario"] for r in all_results]

    fig, axes = plt.subplots(1, len(scenarios),
                             figsize=(4 * len(scenarios), 4.5), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, res in zip(axes, all_results):
        null_vals = [res["methods"][k]["null_test"] for k in method_keys]
        bars = ax.bar(range(len(method_keys)), null_vals,
                      color=[METHOD_COLORS[k] for k in method_keys], edgecolor="k", lw=0.5)
        ax.set_xticks(range(len(method_keys)))
        ax.set_xticklabels([METHOD_LABELS[k] for k in method_keys],
                            rotation=40, ha="right", fontsize=8)
        ax.axhline(0.1, color="gray", ls="--", lw=0.8, label="|r|=0.10")
        ax.set_title(res["scenario"], fontsize=11)
        ax.set_ylim(0, None)

    axes[0].set_ylabel(r"max $|r(w,\, t_i)|$ (null test)", fontsize=11)
    plt.suptitle("Null test: max correlation of weights with templates", fontsize=13)
    plt.tight_layout()
    out = outdir / "null_tests.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()


def plot_snr_ranking(results_by_scenario, suite, outdir):
    """SNR template ranking for the contaminated scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    methods_snr = ["template", "data", "peak"]
    labels = [r"Template SNR ($|\hat\alpha|/\sigma$)",
              r"Data SNR ($|r(\delta_g, t_i)|$)",
              r"Peak $C_\ell$ SNR"]

    scenario = "combined"
    res = results_by_scenario.get(scenario)
    mock = suite.get(scenario)
    if res is None or mock is None:
        return

    delta_g_obs_masked = res["delta_g_obs"]
    delta_t_masked = res["delta_t"]
    good_pix = res["good_pix"]
    n_sys = delta_t_masked.shape[0]
    nside = mock.nside
    n_pix = hp.nside2npix(nside)

    # Embed masked arrays into full-sky maps for the "peak" method
    delta_g_full = np.zeros(n_pix)
    delta_g_full[good_pix] = delta_g_obs_masked
    delta_t_full = np.zeros((n_sys, n_pix))
    delta_t_full[:, good_pix] = delta_t_masked

    for ax, meth, label in zip(axes, methods_snr, labels):
        if meth == "peak":
            snr = sm.snr_template_ranking(delta_g_full, delta_t_full, method="peak")
        else:
            snr = sm.snr_template_ranking(delta_g_obs_masked, delta_t_masked, method=meth)
        order = np.argsort(snr)[::-1]
        ax.bar(range(n_sys), snr[order], color="steelblue", edgecolor="k")
        ax.set_xticks(range(n_sys))
        ax.set_xticklabels([f"t{i}" for i in order])
        ax.set_xlabel("Template (ranked)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label}\n(combined scenario)", fontsize=10)

    plt.tight_layout()
    out = outdir / "snr_ranking.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-method validation of sys_mapping on synthetic mocks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nside", type=int, default=16)
    parser.add_argument("--n-sys", type=int, default=3)
    parser.add_argument("--n-mean", type=float, default=50.0,
                        help="Mean galaxies per pixel.")
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="Log-normal sigma of the true density field.")
    parser.add_argument("--n-walkers", type=int, default=60)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--n-burn", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS,
                        choices=SCENARIOS)
    parser.add_argument("--output-dir", default="docs/_static/results_validation")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"NSIDE={args.nside}  n_sys={args.n_sys}  n_mean={args.n_mean}  "
          f"sigma={args.sigma}")
    print(f"MCMC: {args.n_walkers} walkers × {args.n_steps} steps "
          f"({args.n_burn} burn-in)")
    print(f"Scenarios: {args.scenarios}")
    print(f"Output: {outdir}\n")

    # Generate matched suite
    print("Generating mock suite...")
    suite = make_mock_suite(
        nside=args.nside,
        n_sys=args.n_sys,
        scenarios=args.scenarios,
        n_mean=args.n_mean,
        sigma=args.sigma,
        seed=args.seed,
    )
    print(f"  a_true = {suite[args.scenarios[0]].a_true.round(4)}")
    print(f"  b_true = {suite[args.scenarios[0]].b_true.round(4)}")
    for sc, mock in suite.items():
        print(f"  {sc}: {mock.n_gal:,} galaxies, "
              f"{mock.n_good_pix:,} good pixels")

    # Analyse each scenario
    all_results = []
    results_by_scenario = {}
    for sc in args.scenarios:
        mock = suite[sc]
        print(f"\n=== Scenario: {sc} ===")
        res = analyse_scenario(mock, args.n_walkers, args.n_steps,
                               args.n_burn, seed=args.seed)
        all_results.append(res)
        results_by_scenario[sc] = res

        print(f"  Obs RMS error: {res['obs_metrics']['rms_delta_error']:.4f}")
        for k, mres in res["methods"].items():
            print(f"  {METHOD_LABELS[k]:20s}: "
                  f"RMS={mres['metrics']['rms_delta_error']:.4f}  "
                  f"r={mres['metrics']['corr_with_true']:.4f}  "
                  f"null={mres['null_test']:.4f}")

        # Per-scenario plots
        plot_density_maps(res, args.nside, outdir, sc)
        plot_delta_histogram(res, outdir, sc)
        plot_scatter_true_vs_recovered(res, outdir, sc)
        plot_weight_maps(res, args.nside, outdir, sc)

        # Save numeric results (strip arrays)
        save_res = {
            k: v for k, v in res.items()
            if not isinstance(v, np.ndarray)
        }
        save_res["methods"] = {
            mk: {kk: vv for kk, vv in mres.items()
                 if not isinstance(vv, np.ndarray)}
            for mk, mres in res["methods"].items()
        }
        (outdir / f"results_{sc}.json").write_text(
            json.dumps(save_res, indent=2)
        )

    # Mock-covariance sandwich 1σ for the additive parameter (correlated-noise-calibrated error
    # bars on the amplitude-recovery figure; the iid MCMC posterior std would be ~2× too tight).
    sigma_a = None
    ref = next((r for r in all_results if r["scenario"] in ("additive", "combined", "none")), None)
    if ref is not None:
        print("\nBuilding mock-covariance sandwich (uncontaminated ensemble)…")
        sigma_a, npx = sandwich_sigma_a(args.nside, args.n_sys, args.n_mean, args.sigma,
                                        args.seed, ref["good_pix"], ref["delta_t"])
        print(f"  sandwich σ_a = {np.round(sigma_a, 4)}  ({npx} px)")

    # Cross-scenario summary plots
    print("\nGenerating summary plots...")
    plot_summary_metrics(all_results, outdir)
    plot_amplitude_recovery(all_results, outdir, sigma_a=sigma_a)
    plot_null_tests(all_results, outdir)
    plot_snr_ranking(results_by_scenario, suite, outdir)

    # Save run config
    config = vars(args)
    config["a_true"] = suite[args.scenarios[0]].a_true.tolist()
    config["b_true"] = suite[args.scenarios[0]].b_true.tolist()
    (outdir / "run_config.json").write_text(json.dumps(config, indent=2))

    print(f"\nAll outputs written to {outdir}")


if __name__ == "__main__":
    main()
