# sys_mapping

[![PyPI](https://img.shields.io/pypi/v/sys-mapping)](https://pypi.org/project/sys-mapping/)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://sys-mapping.readthedocs.io/en/latest/)
[![Tests](https://github.com/JohanComparat/sys_mapping/actions/workflows/tests.yml/badge.svg)](https://github.com/JohanComparat/sys_mapping/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/JohanComparat/sys_mapping/branch/main/graph/badge.svg)](https://codecov.io/gh/JohanComparat/sys_mapping)

**Docs:** https://sys-mapping.readthedocs.io/en/latest/

Joint inference of multiplicative and additive systematics in galaxy clustering, after
[Berlfein et al. 2024](https://arxiv.org/abs/2401.12293). The package pre-selects
templates, fits the contamination with six methods, calibrates every detection statistic
against GLASS realisations of the galaxy field, writes per-galaxy weights and corrects the
angular correlation function with the full cross-template matrix.

---

## Template pre-selection

| Method | Statistic |
|---|---|
| `"data"` | Pearson \|r\| between δ_g and each template |
| `"template"` | Per-template least-squares \|t\| |
| `"isd"` | ISD Δχ², with p-values from GLASS realisations |
| `"peak"` | Peak of the cross pseudo-Cℓ over a noise proxy |

`run_decontamination(..., preselect=True)` runs the pre-selection before the fit.

## Decontamination methods

| Method | Model | Estimation |
|---|---|---|
| OLS | Additive | Least-squares pixel regression |
| ElasticNet | Additive | ℓ₁+ℓ₂-regularised regression, cross-validated |
| ISD-1 | Selection weight | Iterative Systematics Decontamination: marginal binned fits, one template at a time, linear |
| ISD-3 | Selection weight | The same with a cubic marginal fit in one template's value |
| MCMC-add | Additive | Exact Normal–Inverse-Gamma posterior |
| MCMC-comb | Combined | BlackJAX NUTS with a dense mass matrix |

`--sampler emcee` selects the emcee ensemble sampler for the MCMC methods.

## Calibrated statistics

Every null the package builds is a set of uncontaminated GLASS realisations drawn from a
spectrum matched to the sample (`load_matched_cl`). Realisations are Poisson counts per
footprint pixel (`draw_null_overdensity`).

| Statistic | Function |
|---|---|
| Template significance and family-wise p-value | `calibrated_template_significance` |
| Residual correlation with held-out templates | `residual_template_correlation_test` |
| ISD stopping threshold | `isd_template_significance` |
| Likelihood-ratio test, additive against combined | `likelihood_ratio_test(null_lambda=...)` with the null from `lrt_from_maxima` |

The Wilks χ² p-value assumes independent pixels, which the clustered field violates, so it
is overconfident. `lrt_from_maxima` takes λ_LR between the maxima of both models, so
λ_LR ≥ 0.

---

## Pipeline

```
Template FITS maps (LS10, Gaia)       scripts/archive/build_systematic_maps.py
        │
        ▼
Galaxy and random catalogues ─► pixelise, overdensity, resolution by occupancy
        │
        ▼
Pre-selection (optional) ─► six methods ─► calibrated significance, LRT, residual test
        │
        ▼
Per-galaxy weights (FITS) + w(θ) corrected with the full template correlation matrix
        │
        ▼
data/sys_weights_auto/  ─► read by the sum_stat package
```

```python
import sys_mapping as sm

result = sm.run_decontamination(
    "ISD-1", delta_g, delta_t,
    preselect=True, preselect_method="isd",
    preselect_n_mocks=100, preselect_p_threshold=0.05,
    good_pixels=good_pix, n_total_footprint=len(ra_gal),
    z_edges=z_edges, nz=nz, nside=nside,
    preselect_cl_input=sm.load_matched_cl("matched_spectra/", sample, nside=nside),
)
print(result["preselect_indices"], result["a_hat"])
```

With `preselect_method="isd"` the pre-selection null also sets the ISD stopping threshold.
The [quickstart](https://sys-mapping.readthedocs.io/en/latest/quickstart.html) builds
`delta_g`, `delta_t` and a matched null from catalogues.

---

## Scripts

### `scripts/run_ls10_analysis.py`

The LS10 BGS pipeline: overdensity, all methods, calibrated statistics, weights and the
corrected w(θ) for one sample or a directory of samples.

```bash
python scripts/run_ls10_analysis.py \
    --catalog-dir /path/to/BGS_VLIM_Mstar \
    --template-dir ~/data/legacysurvey/dr10/systematics/0128 \
    --nside 128 --min-per-pixel 25 \
    --null-cl-file matched_spectra/ \
    --significance-n-mocks 400 --lrt-null-mocks 50 \
    --output-dir data/sys_weights_auto/
```

`--min-per-pixel` puts each sample at the finest NSIDE, no finer than `--nside`, whose
footprint holds that many galaxies per pixel. `--null-cl-file` is required whenever a null
is built; `--allow-parametric-null` uses a power law instead and records the product as
parametric. `--null-draw` (`pixel` or `catalogue`) and `--lrt-null-method` (`maxima` or
`nuts`) choose how null realisations are drawn and fitted. The phases, output keys and
sentinel files are described in the
[LS10 pipeline page](https://sys-mapping.readthedocs.io/en/latest/pipeline_ls10.html).

**Weights.** `run_decontamination` returns the weight that inverts the model each method
fitted, and `run_ls10_analysis.py` writes it:

| Form | Methods | Formula | Clip |
|---|---|---|---|
| linear | OLS, ElasticNet | `1 / max(1 + Σ_i a_i·t_i(p), 1e-6)` | `[1/20, 20]` |
| exact | MCMC-add, MCMC-comb | `(1 + δ_g,clean(p)) / max(1 + δ_g,obs(p), 1e-6)` | `[1/20, 20]` |
| product | ISD-1, ISD-3 | `Π_j 1 / (1 + F̂_j(t_j(p)))` | `[1/20, 20]` |

| Column | Model |
|---|---|
| `WEIGHT_OLS`, `WEIGHT_ENET` | Additive, linear |
| `WEIGHT_ISD1`, `WEIGHT_ISD3` | Selection efficiency, cumulative product |
| `WEIGHT_ADD` | Additive, exact inverse |
| `WEIGHT_COMB` | Combined, exact inverse |
| `WEIGHT_SYS` | Alias of `WEIGHT_COMB`, the recommended weight |

The header records the convention in `WEIGHTVER`, `WEIGHTCON`, `WMAXCLIP` and `TPLBASIS`.
Version 3 marks templates standardised over the analysis footprint, so amplitudes are in
units of one template standard deviation; lower versions are read with a warning.
`--no-footprint-standardise` and `--ct-from-pixels` write a version-2 product. `--skewed`
fits the skew-normal likelihood for MCMC-comb; MCMC-add stays Gaussian, and the
likelihood-ratio test compares additive and combined maxima that both carry the skewness.

### `scripts/compute_sys_weights.py`

Batch weights for every `*_DATA.fits` / `*_RAND.fits` pair: OLS, MCMC-add and MCMC-comb,
written as `WEIGHT_OLS`, `WEIGHT_ADD`, `WEIGHT_COMB` and `WEIGHT_SYS` in the linear form
(`WEIGHTCON = linear-from-a_hat`).

```bash
python scripts/compute_sys_weights.py --catalog-dir /path/to/BGS_VLIM_Mstar --nside 64
python scripts/compute_sys_weights.py --sample LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228
```

### Validation and characterisation scripts

| Script | Purpose |
|---|---|
| `run_validation.py` | All methods on synthetic mocks under four contamination scenarios |
| `run_systematic_tests.py` | All methods across template configurations |
| `run_mock_analysis.py` | Parameter recovery and LRT statistics on mocks (`--synthetic` or `--mock-dir`) |
| `run_mock_analysis_diagnostic.py` | Diagnostic figures for one synthetic mock |
| `run_mock_analysis_progressive.py` | Detection as the number of contaminated templates grows |
| `run_mock_analysis_real_templates.py` | Recovery with Gaia and LS10 depth templates on the LS10 footprint |
| `run_simulation_tests.py` | w(θ) recovery on GLASS and Uchuu mocks with injected systematics |
| `run_paper_validation.py` | Berlfein et al. 2024 Figures 2–8 and Tables 2–3 on lognormal mocks |
| `run_detectability_sweep.py` | Detection limit over NSIDE, density, f_sky and amplitude |
| `benchmark_corrfunc_vs_treecorr.py` | Corrfunc against TreeCorr for w(θ) |

All but `benchmark_corrfunc_vs_treecorr.py` take `--help`.

### Documentation generators

| Script | Output |
|---|---|
| `generate_results_ls10_summary.py` | `docs/results_ls10.rst`, `docs/results_ls10_recommendations.rst` and the nine sample pages |
| `plot_ls10_occupancy_products.py` | `docs/_static/results_ls10/wtheta_ratio_occupancy.png` |
| `analyze_detectability_law.py` | `docs/detectability_law.rst`, figures and tables |
| `make_survey_design_synthesis.py` | `docs/survey_design_synthesis.rst` and figures |
| `coverage_report.py` | `docs/coverage.rst` from a coverage JSON, `jax_coverage.py` and an optional profile |
| `plot_runtime_scaling.py` | `docs/_static/runtime_scaling.png` |
| `plot_simulation_tests.py` | simulation-test figures under `docs/_static/` |

---

## Background runs (`bash/`)

The scripts in `bash/` start a pipeline with `nohup`, write a timestamped log to `logs/`
and a `.pid` file, and set the thread counts to `$(nproc)`.

```bash
bash/ls10_analysis.sh
CATALOG_DIR=/path/to/BGS bash/ls10_analysis.sh --sample LS10_VLIM_ANY_10.5_...
bash/mock_analysis.sh --synthetic --n-mocks 20
bash/paper_validation.sh --nside 512 --n-real 119
bash/build_maps.sh --source gaia --nside 64 128

tail -f logs/ls10_analysis_<timestamp>.log
kill -0 $(cat logs/ls10_analysis_<timestamp>.pid) && echo running
```

`compute_sys_weights.py --device gpu` runs JAX on a GPU when `jax[cuda12]` is installed.

---

## Installation

Version 1.4.0 installs from the tagged source:

```bash
pip install "git+https://github.com/JohanComparat/sys_mapping@v1.4.0"
```

For development:

```bash
git clone https://github.com/JohanComparat/sys_mapping && cd sys_mapping
mamba env create -f environment.yml
mamba activate sys_map
pip install -e ".[dev]"
```

PyPI carries version 1.3.0 (`pip install sys-mapping`).

---

## Tests

```bash
pytest -m "not slow"          # fast suite, run on pull requests
pytest --cov=sys_mapping      # full suite with branch coverage, run on main and nightly
```

The suite collects the docstring examples, checks JAX kernels against NumPy references,
and records which public functions survive `jax.jit`, `jax.vmap` and `jax.grad`. Branch
coverage is 98.1 % and CI fails below 97 %. Per-module coverage, the JAX share and the wall
time by library are on the [coverage page](https://sys-mapping.readthedocs.io/en/latest/coverage.html);
test counts per module are on the [testing page](https://sys-mapping.readthedocs.io/en/latest/testing.html).
Real-data tests skip when `~/data/legacysurvey/dr10/systematics/` is absent.

---

## Related repositories

| Repository | Contents |
|---|---|
| [`sys_mapping_benchmark`](https://github.com/JohanComparat/sys_mapping_benchmark) | Timing harness, algorithm-characterisation scripts, and the GRICAD job scripts that produce the LS10 products |
| `sys_mapping_paper` (private) | The pipeline paper, with every equation traced to the code that evaluates it |

Their results are documented here: [benchmark](docs/results_benchmark.rst) and
[algorithm characterisation](docs/results_algorithm_characterisation.rst).

```bash
git clone https://github.com/JohanComparat/sys_mapping_benchmark
cd sys_mapping_benchmark && pip install -e ".[dev]"
python benchmark/benchmark_pipeline.py --quick
SYS_MAPPING_ROOT=~/software/sys_mapping python characterisation/run_crossterm_bias.py
```

---

## Documentation

```bash
cd docs && make html    # docs/_build/html/index.html
```

The documentation covers the method, the API, validation on mocks, the detectability law,
and the LS10 BGS results: nine stellar-mass samples, each at the resolution its occupancy
supports, with the same fits at NSIDE 32, 64 and 128 for comparison.

---

## Package layout

| Module | Public symbols |
|---|---|
| `contamination` | `apply_contamination`, `invert_contamination`, `apply_nonlinear_contamination`, `TemplateResponse`, `evaluate_response`, `compute_two_point_correction`, `pack_params`, `unpack_params`, `n_free_params` |
| `likelihood` | `make_log_likelihood`, a compiled Gaussian or skew-normal log-likelihood |
| `covariance` | `LowRankPrecision`, `build_lowrank_precision`, `build_harmonic_precision`, `mock_sandwich_covariance`, `method_marginalised_covariance`, `sample_covariance`, `hartlap_factor` |
| `maps` | `pixelize_catalog`, `compute_overdensity`, `assign_template_values`, `standardise_on_footprint`, `choose_nside_by_occupancy`, `inverse_variance_pixel_weights`, `load_real_template`, `load_real_templates`, `generate_systematic_map`, `generate_systematic_maps`, `systematic_power_spectrum` |
| `inference` | `run_additive_analytic`, `run_mcmc`, `make_log_prob`, `posterior_median_params`, `refine_to_mle`, `get_param_variance_from_chain`, `get_param_covariance_from_chain` |
| `nuts` | `run_nuts`, `build_logdensity`, `default_n_chains` |
| `correction` | `debias_params`, `debias_params_matrix`, `rotate_templates`, `transform_params_from_rotated`, `correct_two_point_function`, `correct_power_spectrum_harmonic` |
| `model_selection` | `likelihood_ratio_test`, `lrt_from_maxima`, `lrt_null_distribution`, `snr_preselect`, `greedy_forward_select` |
| `bootstrap` | `block_bootstrap_variance`, `jackknife_covariance` |
| `regression` | `run_decontamination`, `method_comparison`, `elasticnet_contamination_fit`, `iterative_systematics_decontamination`, `polynomial_ols_decontamination` |
| `diagnostics` | `calibrated_template_significance`, `residual_template_correlation_test`, `isd_template_significance`, `isd_marginal_fit`, `snr_template_ranking`, `null_test_cross_correlations`, `vet_templates_against_tracer`, `footprint_mask_diagnostics` |
| `mocks` | `generate_lognormal_field`, `make_galactic_mask`, `make_mock_catalog`, `make_mock_suite`, `MockCatalog` |
| `glass_mocks` | `load_matched_cl`, `draw_null_overdensity`, `generate_glass_null_overdensity`, `generate_glass_delta_map`, `generate_glass_fullsky_mock`, `measure_nz`, `sanitise_cl`, `sample_positions_from_delta` |
| `simulation` | `ContaminationConfig`, `LEVELS`, `make_contamination_grid`, `load_uchuu_mock`, `load_systematic_maps`, `apply_footprint_mask`, `inject_systematics`, `run_wtheta_recovery` |
| `power_spectrum` | `measure_pseudo_cl`, `subtract_template_cl`, `harmonic_bias`, `mode_projection_bias` |
| `utils` | `template_correlation_matrix`, `measure_two_point_function`, `measure_two_point_function_corrfunc`, `measure_cross_two_point_function`, `measure_kk_correlation_treecorr`, `measure_kk_correlation_corrfunc`, `measure_kk_covariance_treecorr`, `compute_covariance_matrix`, `compute_amplitude_bias` |
| `plotting` | `METHOD_COLORS`, `METHOD_LINESTYLES`, `METHOD_MARKERS`, `METHOD_LABELS`, `METHOD_ORDER` |

The symbols above import from `sys_mapping` directly;
`LikelihoodRatioResult` imports from `sys_mapping.model_selection`.

---

## Berlfein et al. 2024 equations in the code

| Equation | Description | Implementation |
|---|---|---|
| Eq. 11–13 | Additive, multiplicative and combined models | `contamination.py:apply_contamination` (the model follows from which of `a`, `b` are zero) |
| Eq. 15–16 | Two-point correction | `contamination.py:compute_two_point_correction`; `correction.py:correct_two_point_function` |
| Eq. 17 | Gaussian log-likelihood with Jacobian | `likelihood.py:make_log_likelihood` (`use_skewed=False`) |
| Eq. 18 | Skew-normal log-likelihood | `likelihood.py:make_log_likelihood` (`use_skewed=True`) |
| Eq. 19 | Likelihood-ratio test | `model_selection.py:likelihood_ratio_test` |
| Eq. 21 | Noise debiasing | `correction.py:debias_params`, `debias_params_matrix` |
| Eq. 24 | Template covariance matrix | `utils.py:compute_covariance_matrix` |
| Eq. 25–26 | Amplitude bias ΔĀ | `utils.py:compute_amplitude_bias` |
| App. A | PCA template rotation C = VDVᵀ | `correction.py:rotate_templates`, `transform_params_from_rotated` |

Notes on the implementation:

- The combined model is the paper form `δ̂_g = δ_g(1+b·t) + a·t`. The product
  `(δ_g + a·t)(1+b·t)` differs by the second-order term `a·t·b·t`.
- The Gaussian likelihood subtracts `Σ ln|1+b·t|`, the Jacobian of the forward map.
- In the skew-normal likelihood the residual is shifted by `ξ = −σ·δ·√(2/π)` with
  `δ = γ/√(1+γ²)`; the shifted residual `r` enters both the quadratic form and the CDF term
  `Σ log Φ(γ·r/σ)`, and `N·ln 2` comes from the `2/σ` prefactor.
- With `C = VDVᵀ`, `δ'_t = Vᵀ δ_t` and `a = V a'`, the covariance propagates as
  `Cov[a] = V · Cov[a'] · Vᵀ`, the full matrix (`R.T @ cov_rot @ R` in
  `regression.py:run_decontamination`).
- The two-point correction uses the full `(n_sys, n_sys)` amplitude matrices and the
  template correlation matrix from `template_correlation_matrix`, whose cross terms are
  `ξ_ij = [ξ(t_i+t_j) − ξ_ii − ξ_jj]/2`.

---

## Data files

`data/` is not tracked. `scripts/run_ls10_analysis.py` writes:

| Directory | Contents |
|---|---|
| `data/sys_weights_auto/` | The issued LS10 products, each sample at the resolution its occupancy supports |
| `data/sys_weights/` | The same fits at NSIDE 32, 64 and 128 |

| File | Contents |
|---|---|
| `{sample_id}_NSIDE{NNNN}_WEIGHTS.fits` | Per-galaxy weights, seven columns |
| `{sample_id}_NSIDE{NNNN}_params.json` | Point estimates, calibrated statistics and chain diagnostics |
| `{sample_id}_NSIDE{NNNN}_partial_*.json` | Results of the fast-method phases |

---

## Reference

Berlfein et al. 2024, *Joint inference of multiplicative and additive systematics in galaxy
clustering*, MNRAS 531, 4954, arXiv:2401.12293.
