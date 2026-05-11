# sys_mapping

[![PyPI](https://img.shields.io/pypi/v/sys-mapping)](https://pypi.org/project/sys-mapping/0.9/)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://sys-mapping.readthedocs.io/en/latest/#)
[![Tests](https://github.com/JohanComparat/sys_mapping/actions/workflows/tests.yml/badge.svg)](https://github.com/JohanComparat/sys_mapping/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/JohanComparat/sys_mapping/branch/main/graph/badge.svg)](https://codecov.io/gh/JohanComparat/sys_mapping)

**PyPI:** https://pypi.org/project/sys-mapping/0.9/
**Docs:** https://sys-mapping.readthedocs.io/en/latest/#

Joint inference of multiplicative and additive systematics in galaxy
clustering. Implements six decontamination methods with a shared
contamination model, template normalisation, noise debiasing, and
two-point function correction infrastructure.

---

## Six decontamination methods

| Method | Model | Parameter estimation |
|---|---|---|
| **OLS** | Additive | Ordinary least-squares pixel regression |
| **ElasticNet** | Additive | ℓ₁+ℓ₂-regularised regression, cross-validated |
| **ISD-1** | Additive–multiplicative | Iterative reweighted OLS, poly order 1 |
| **ISD-3** | Additive–multiplicative | Iterative reweighted OLS, poly order 3 |
| **MCMC-add** | Additive | Bayesian posterior sampling (emcee), b=0 |
| **MCMC-comb** | Combined | Bayesian posterior sampling (emcee), free a, b |

All methods produce per-galaxy weights (`WEIGHT_SYS` / `WEIGHT_COMB`) for use
downstream in two-point function estimators.

---

## Pipeline overview

```
Raw catalogs / randoms
        │
        ▼
scripts/build_systematic_maps.py   ← build HEALPix template maps (GAIA DR2, LS10)
        │
        ▼
    Template FITS files  (LS10_EBV_NSIDE_0064.fits, GAIA_nstar_faint_NSIDE_00064.fits, …)
        │
        ├──► scripts/run_ls10_analysis.py              ← LS10 BGS per-method weights + w(θ)
        │         (or scripts/run_all_methods_sequential.sh for phased multi-method run)
        ├──► scripts/run_mock_analysis.py              ← parameter recovery on mocks
        ├──► scripts/run_validation.py                 ← all methods × all scenarios
        ├──► scripts/run_systematic_tests.py           ← all methods × template configs
        └──► scripts/run_paper_validation.py           ← reproduce Berlfein 2024 Figs 2–8
                │
                ▼
        scripts/compute_sys_weights.py    ← batch per-galaxy weights → WEIGHT_SYS, WEIGHT_ADD, WEIGHT_COMB
                │
                ▼
        data/sys_weights/  ← consumed by sum_stat package
```

---

## Method

Core method: [Berlfein et al. 2024](https://arxiv.org/abs/2401.12293)
(MNRAS 531, 4954).

### Contamination model (Eq. 11–13)

The observed galaxy overdensity in pixel p is modelled as the linear combination
of three nested models:

| Model | Equation | Free params |
|---|---|---|
| Additive | `δ̂_g,p = δ_g,p + Σ_i a_i δ_{ti,p}` | a |
| Multiplicative | `δ̂_g,p = δ_g,p (1 + Σ_i a_i δ_{ti,p})` | a (b=a) |
| Combined | `δ̂_g,p = δ_g,p (1 + Σ_i b_i δ_{ti,p}) + Σ_i a_i δ_{ti,p}` | a, b |

### Likelihoods (Eq. 17–18)

Integrating out the true δ_g (assumed Gaussian with dispersion σ) gives:

```
ln L = −(N/2) ln(2πσ²) − Σ_p ln|1 + Σ_i b_i δ_{ti,p}|
       − (1/2σ²) Σ_p [(δ̂_g,p − Σ_i a_i δ_{ti,p}) / (1 + Σ_i b_i δ_{ti,p})]²
```

An optional skew-normal extension (Eq. 18) adds `N ln 2 + Σ log Φ(γ δ_g,p / σ)`.

### Template rotation (Appendix A)

When templates are correlated, the PCA-rotated basis δ'_t = Vᵀ δ_t
(where V diagonalises the template covariance C = VDVᵀ) makes each
contamination parameter independently identifiable. Parameters are
transformed back to the original basis after inference.

### Noise debiasing (Eq. 21)

Because E[â²] = a² + Var[â], the squared MLE is noise-inflated.
The debiased estimate is:

```
ã²_i = max(â²_i − Var[â_i], 0)
```

### Two-point correction (Eq. 15–16)

After debiasing, the observed angular correlation function is corrected:

```
ŵ_corr(θ) = [ŵ(θ) − Σ_i ã²_i C_{ti}(θ)] / [1 + Σ_i b̃²_i C_{ti}(θ)]
```

### Model selection (Eq. 19)

Nested models are compared with the likelihood ratio test:

```
λ_LR = 2 [ln L(Θ̂) − ln L(Θ̂₀)] ~ χ²(r)
```

where r is the number of additional free parameters.

---

## Scripts

### `scripts/run_all_methods_sequential.sh`

Main orchestration script: runs all six methods on LS10 BGS in phases
(fast methods first), writing incremental Sphinx documentation after
each phase. Environment variables control behaviour:

```bash
# Full run (nohup so the terminal can be closed)
nohup bash scripts/run_all_methods_sequential.sh > logs/run_all.log 2>&1 &

# OLS-only quick preview (seconds)
METHODS="OLS" bash scripts/run_all_methods_sequential.sh

# Resume from MCMC-add (OLS and ElasticNet already done)
METHODS="MCMC-add MCMC-comb" bash scripts/run_all_methods_sequential.sh

# Force re-run all bins
FORCE=1 METHODS="OLS" bash scripts/run_all_methods_sequential.sh
```

Key environment variables: `BINS`, `DEVICE` (cpu/gpu), `METHODS`,
`FORCE`, `SKIP_LS10`, `CATALOG_DIR`, `LS10_NSIDE`.

### `scripts/build_systematic_maps.py`

Build HEALPix systematic template maps from GAIA DR2 or LS10 BGS randoms.

```bash
# LS10 maps (EBV, GALDEPTH, PSFSIZE, NOBS)
python scripts/build_systematic_maps.py --source ls10 --nside 32 64 128 256

# GAIA DR2 maps (G/BP/RP flux, star counts by magnitude)
python scripts/build_systematic_maps.py --source gaia --nside 64
```

### `scripts/compute_sys_weights.py`

Batch pipeline: processes every `*_DATA.fits` / `*_RAND.fits` pair, runs
all methods, writes per-galaxy systematic weights for the `sum_stat` package.

```bash
# All samples, default settings (NSIDE=64)
python scripts/compute_sys_weights.py

# GPU — vectorised walker evaluation
python scripts/compute_sys_weights.py --device gpu

# Single sample
python scripts/compute_sys_weights.py \
    --sample LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228
```

**Weighting scheme**

| Column | Model | Formula |
|---|---|---|
| `WEIGHT_ADD` | Additive | `1 / max(1 + Σ_i a_i · t_i(p), 0.01)` |
| `WEIGHT_COMB` | Combined | `1 / max(1 + Σ_i b_i · t_i(p), 0.01)` |
| `WEIGHT_SYS` | Combined (recommended) | identical to `WEIGHT_COMB` |

### `scripts/run_ls10_analysis.py`

Full pipeline on LS10 BGS catalogs: overdensity → all methods → correction
→ w(θ) plots.

```bash
python scripts/run_ls10_analysis.py \
    --catalog-dir /path/to/BGS_VLIM_Mstar \
    --template-dir /path/to/systematics/ \
    --nside 64 --output-dir results/ls10/
```

### `scripts/run_validation.py`

Multi-method, multi-scenario validation on synthetic mocks (none /
additive / multiplicative / combined contamination). Applies all six
methods and compares against known ground truth.

```bash
python scripts/run_validation.py --nside 64 --n-mocks 50 \
    --output-dir results/validation/
```

### `scripts/run_systematic_tests.py`

Systematic test matrix: all methods × all template configurations (7
synthetic templates, tiers of single and combined contamination types).

```bash
python scripts/run_systematic_tests.py --nside 64
```

### `scripts/run_mock_analysis.py`

Parameter recovery on mock galaxy catalogs (LRT statistics).

```bash
# Synthetic mocks (no data files required)
python scripts/run_mock_analysis.py --synthetic --n-mocks 20 --nside 32

# Real mocks
python scripts/run_mock_analysis.py \
    --mock-dir /path/to/mocks --rand-file /path/to/randoms.fits \
    --n-mocks 100 --output-dir results/mock_analysis/
```

### `scripts/run_mock_analysis_diagnostic.py`

Deep-dive diagnostic figures for one synthetic mock (sky maps, template
maps, weight histograms, per-template S/N).

```bash
python scripts/run_mock_analysis_diagnostic.py \
    --nside 64 --n-sys 5 --seed 0 \
    --output-dir results/mock_analysis_diagnostic/
```

### `scripts/run_mock_analysis_progressive.py`

Progressive template contamination study: varies the number of
contaminated templates (k=1,2,3) and contamination mode across mocks.

### `scripts/run_mock_analysis_real_templates.py`

Validation with real GAIA stellar and LS10 depth templates on the actual
LS10 south footprint (~22 000 pixels at NSIDE=64).

### `scripts/run_paper_validation.py`

Reproduce Berlfein 2024 Figures 2–8 and Tables 2–3 on lognormal synthetic
mocks.

```bash
# Quick test (NSIDE=32, 2 realisations)
python scripts/run_paper_validation.py --nside 32 --n-real 2 --output-dir /tmp/val

# Full paper settings (use HPC)
python scripts/run_paper_validation.py --nside 512 --n-real 119 \
    --n-walkers 250 --n-steps 1500 --n-burn 300
```

### Documentation-generation scripts

| Script | Output |
|---|---|
| `scripts/generate_results_ls10_summary.py` | `docs/results_ls10.rst` from `*_params.json` files |
| `scripts/generate_sample_pages.py` | Per-sample RST pages in `docs/` |
| `scripts/generate_recommendations.py` | `docs/results_ls10_recommendations.rst` |
| `scripts/plot_ls10_wtheta_corrected.py` | `docs/_static/results_ls10/wtheta_corrected_nside64.png` |
| `scripts/plot_runtime_scaling.py` | `docs/_static/runtime_scaling.png` |
| `scripts/patch_params_from_partials.py` | Patches `*_params.json` from per-method partial JSON files |

### `scripts/benchmark_corrfunc_vs_treecorr.py`

Performance comparison between Corrfunc and TreeCorr for the angular
two-point function estimator.

---

## Server execution (bash/ — background, all cores)

`bash/` provides scripts that launch each pipeline in the background with
`nohup`, writing timestamped logs to `logs/` and a `.pid` file for process
tracking. Each script sets `OMP_NUM_THREADS=$(nproc)` and JAX XLA flags.

```bash
# LS10 analysis
bash/ls10_analysis.sh
CATALOG_DIR=/path/to/BGS bash/ls10_analysis.sh --sample LS10_VLIM_ANY_10.5_...

# Paper validation
bash/paper_validation.sh
bash/paper_validation.sh --nside 512 --n-real 119   # full paper settings

# Mock analysis
bash/mock_analysis.sh --synthetic --n-mocks 20
bash/mock_analysis.sh --mock-dir /path/to/mocks --rand-file /path/to/randoms.fits

# Build template maps
bash/build_maps.sh --source ls10
bash/build_maps.sh --source gaia --nside 64 128
```

**Monitoring**:
```bash
tail -f logs/ls10_analysis_20260505_143200.log
kill -0 $(cat logs/ls10_analysis_20260505_143200.pid) && echo running || echo done
kill $(cat logs/ls10_analysis_20260505_143200.pid)
```

**GPU notes**

`--device gpu` lets JAX auto-detect the GPU backend. To enable GPU:

```bash
pip install 'jax[cuda12_pip]' \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

## Installation

Install from [PyPI](https://pypi.org/project/sys-mapping/0.9/):

```bash
pip install sys-mapping==0.9
```

Or install from source:

```bash
cd ~/sys_mapping
mamba env create -f environment.yml
mamba activate sys_map
pip install -e .
```

---

## Tests

```bash
pytest tests/ -v
```

Test modules: `test_contamination`, `test_correction`, `test_likelihood`,
`test_maps`, `test_inference`, `test_model_selection`, `test_bootstrap`,
`test_power_spectrum`, `test_regression`, `test_diagnostics`, `test_mocks`,
`test_real_templates`, `test_accuracy`, `test_benchmarks`, `test_timing`.

---

## Documentation

Full Sphinx documentation is in `docs/`, covering: overview, installation,
quickstart, methods with paper references, validation results, LS10 BGS
analysis results (9 samples × 4 NSIDEs), and full API reference.

```bash
cd docs && make html
# open docs/_build/html/index.html
```

---

## Package layout

| Module | Public symbols |
|---|---|
| `contamination` | `apply_contamination`, `invert_contamination`, `compute_two_point_correction`, `pack_params`, `unpack_params`, `n_free_params` |
| `likelihood` | `make_log_likelihood` — factory returning a `@jax.jit` log-likelihood (Gaussian or skew-normal) |
| `maps` | `systematic_power_spectrum`, `generate_systematic_map`, `generate_systematic_maps`, `load_real_template`, `load_real_templates`, `pixelize_catalog`, `compute_overdensity`, `assign_template_values` |
| `inference` | `make_log_prob`, `run_mcmc`, `get_mle_params`, `get_param_variance_from_chain`, `get_param_covariance_from_chain` |
| `correction` | `debias_params` (Eq. 21), `rotate_templates` (App. A), `transform_params_from_rotated`, `correct_two_point_function`, `correct_power_spectrum_harmonic` |
| `model_selection` | `likelihood_ratio_test` → `LikelihoodRatioResult` (Eq. 19) |
| `bootstrap` | `block_bootstrap_variance` — spatial block bootstrap via HEALPix coarsening (Sec. 6.2); `jackknife_covariance` |
| `regression` | `elasticnet_contamination_fit`, `iterative_systematics_decontamination`, `method_comparison`, `run_decontamination` |
| `diagnostics` | `null_test_cross_correlations`, `snr_template_ranking`, `footprint_mask_diagnostics` |
| `mocks` | `generate_lognormal_field`, `make_galactic_mask`, `make_mock_catalog`, `make_mock_suite`, `MockCatalog` |
| `power_spectrum` | `measure_pseudo_cl`, `subtract_template_cl`, `harmonic_bias`, `mode_projection_bias` |
| `utils` | `compute_covariance_matrix` (Eq. 24), `compute_amplitude_bias` (Eq. 25-26), `measure_two_point_function` (Landy-Szalay via TreeCorr), `measure_two_point_function_corrfunc`, `measure_kk_correlation_treecorr`, `measure_kk_correlation_corrfunc` |
| `plotting` | `METHOD_COLORS`, `METHOD_LINESTYLES`, `METHOD_MARKERS`, `METHOD_LABELS`, `METHOD_ORDER` — shared colorblind-safe palette constants |

All symbols above are importable directly from `sys_mapping`.
`LikelihoodRatioResult` must be imported as
`from sys_mapping.model_selection import LikelihoodRatioResult`.

---

## Berlfein 2024 equation cross-reference

| Equation | Description | Implementation |
|---|---|---|
| Eq. 11 | Additive model | `contamination.py:apply_contamination` (model='additive') |
| Eq. 12 | Multiplicative model | `contamination.py:apply_contamination` (model='multiplicative') |
| Eq. 13 | Combined model | `contamination.py:apply_contamination` (model='combined') |
| Eq. 15-16 | 2PCF correction | `contamination.py:compute_two_point_correction`; `correction.py:correct_two_point_function` |
| Eq. 17 | Gaussian log-likelihood + Jacobian | `likelihood.py:make_log_likelihood` (skew=False) |
| Eq. 18 | Skew-normal log-likelihood | `likelihood.py:make_log_likelihood` (skew=True) |
| Eq. 19 | Likelihood ratio test | `model_selection.py:likelihood_ratio_test` |
| Eq. 21 | Noise debiasing | `correction.py:debias_params` |
| Eq. 24 | Template covariance matrix | `utils.py:compute_covariance_matrix` |
| Eq. 25-26 | Amplitude bias ΔĀ | `utils.py:compute_amplitude_bias` |
| App. A | PCA template rotation C = VDVᵀ | `correction.py:rotate_templates` + `transform_params_from_rotated` |

---

## Data files

### Generated outputs (`data/sys_weights/` — git-ignored)

Produced by `scripts/compute_sys_weights.py` or `scripts/run_ls10_analysis.py`:

| File | Contents |
|---|---|
| `{sample_id}_NSIDE{NNNN}_WEIGHTS.fits` | Per-galaxy systematic weights (`WEIGHT_SYS`, `WEIGHT_ADD`, `WEIGHT_COMB`) |
| `{sample_id}_NSIDE{NNNN}_params.json` | MAP parameters and chain diagnostics for all methods |
| `{sample_id}_NSIDE{NNNN}_partial_*.json` | Partial results from fast-method phases |

### Paper tables and simulation configs (`data/` — tracked)

| File | Contents |
|---|---|
| `data/paper_tables/table1_mock_properties.csv` | KiDS-HOD mock survey properties |
| `data/paper_tables/table2_lrt_results.csv` | LRT model-selection fractions |
| `data/paper_tables/table3_cosmological_impact.csv` | Amplitude bias ΔĀ per model |
| `data/simulation_config/contamination_params_25sys.csv` | 25 contamination parameters used in the paper |
| `data/simulation_config/angular_bins.csv` | Log-spaced angular bins (0.5–60 arcmin) |

---

## Equations vs. implementation — notes

- **Combined model**: the code implements the exact paper form
  `δ̂_g = δ_g(1+b·t) + a·t` (Eq. 13). The full product `(δ_g + a·t)(1+b·t)`
  differs by the second-order term a·t·b·t, which is negligible for |a|, |b| ≪ 1.
- **Jacobian sign**: the Gaussian likelihood subtracts `Σ ln|1+b·t|` (positive
  Jacobian of the forward map), consistent with Eq. 17.
- **Skew-normal**: the CDF term is `Σ log Φ(γ·r/σ) = Σ log_ndtr(z)`, which
  includes the `N·ln 2` from the `2/σ` prefactor. A previous version used
  `Σ log_ndtr(√2·z)` (off by √2 in the erf argument); this is corrected in
  commit `10c01d0`.
- **Variance propagation through PCA**: `Var[a_orig] = Vᵀ · diag(Var[a_rot]) · V`
  is correct when the rotated parameters are approximately uncorrelated.

---

## Reference

Berlfein et al. 2024, *Joint inference of multiplicative and additive
systematics in galaxy clustering*, MNRAS 531, 4954, arXiv:2401.12293.
