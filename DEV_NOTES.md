# CLAUDE.md — sys_mapping

## Environment

Activate before any Python work:

```bash
conda activate dev        # laptop: shared env from ~/software/dev_env
mamba activate sys_map    # server
```

Python executable: `$HOME/miniforge3/envs/sys_map/bin/python` (server) or `$HOME/software/miniforge3/envs/dev/bin/python` (laptop; shared `dev` env, never create a new one, missing deps go into `~/software/dev_env`)

## Install

```bash
pip install -e .
```

## Run tests

```bash
pytest tests/ -v
```

## Build docs

```bash
cd docs && make html
```

## Package structure

```
sys_mapping/
├── contamination.py    # forward/inverse model (Eq. 11-13), param packing
│                       # exports: apply_contamination, invert_contamination,
│                       #          compute_two_point_correction,
│                       #          pack_params, unpack_params, n_free_params
├── likelihood.py       # JAX JIT log-likelihood factory (Eq. 17-18)
│                       # exports: make_log_likelihood
├── maps.py             # HEALPix template generation (5 families, Sec. 4.2)
│                       # exports: systematic_power_spectrum,
│                       #          generate_systematic_map, generate_systematic_maps,
│                       #          pixelize_catalog, compute_overdensity,
│                       #          assign_template_values; TEMPLATE_FAMILIES dict
├── inference.py        # emcee MCMC wrapper + chain post-processing
│                       # exports: make_log_prob, run_mcmc,
│                       #          posterior_median_params, refine_to_mle
├── correction.py       # noise debiasing (Eq. 21), PCA rotation (App. A),
│                       # two-point correction
│                       # exports (top-level): debias_params, rotate_templates,
│                       #          correct_two_point_function
│                       # exports (module only): transform_params_from_rotated
├── model_selection.py  # likelihood ratio test (Eq. 19)
│                       # exports: likelihood_ratio_test → LikelihoodRatioResult
│                       #          (dataclass: lambda_lr, n_dof, p_value,
│                       #           reject_null, null_model, alt_model)
├── bootstrap.py        # block bootstrap variance (Sec. 6.2)
│                       # exports: block_bootstrap_variance
│                       # private: _assign_patches (HEALPix coarsening)
└── utils.py            # covariance matrix, amplitude bias, TreeCorr w(θ)
                        # exports: compute_covariance_matrix (Eq. 24),
                        #          compute_amplitude_bias (Eq. 25-26),
                        #          measure_two_point_function (Landy-Szalay)
```

## Key design notes

- `jax.config.update("jax_enable_x64", True)` is called in `__init__.py` at import time.
- `make_log_likelihood(n_sys, model, use_skewed)` returns a `@jax.jit` function.
  Python-level conditionals (model, use_skewed) are resolved at trace time.
- emcee receives numpy arrays; conversion to JAX happens inside the closure
  returned by `make_log_prob`.
- Noise debiasing: `â²_i - Var[â_i]`, clipped to ≥ 0.
- When templates are correlated, call `rotate_templates` first, infer in the
  PCA basis, then `transform_params_from_rotated` back.
- `transform_params_from_rotated` is in `correction.py` but not re-exported
  from `__init__.py`; import as `from sys_mapping.correction import transform_params_from_rotated`.
- `LikelihoodRatioResult` is a dataclass in `model_selection.py`; import as
  `from sys_mapping.model_selection import LikelihoodRatioResult`.
- `block_bootstrap_variance` uses `_assign_patches` internally; coarsens NSIDE
  so the footprint is divided into ~`n_patches` regions, then resamples with replacement.
- `compute_amplitude_bias` returns `(additive_bias, mult_factor)` where
  `additive_bias = Σ a²_i C_{ii}` and `mult_factor = 1 + Σ b²_i C_{ii}`.

## Parameter vector layout

| Model | Layout |
|---|---|
| additive | `[a_0,...,a_{N-1}, σ, (γ)]` |
| multiplicative | `[a_0,...,a_{N-1}, σ, (γ)]`  (b = a internally) |
| combined | `[a_0,...,a_{N-1}, b_0,...,b_{N-1}, σ, (γ)]` |

`γ` is only present when `use_skewed=True`.

## Performance benchmarks

Superseded. Timings now live in `docs/results_benchmark.rst`, generated from a
committed snapshot measured by the separate `sys_mapping_benchmark` repository
(which also holds `test_timing.py`, `test_benchmarks.py` and the characterisation
scripts). The table below is the old, undated, unattributed one — kept only until
anything still referring to it is updated.

## Old benchmark table (CPU, after JIT warmup)

| Operation | Time |
|---|---|
| `systematic_power_spectrum` (nside=64) | ~20 μs |
| `generate_systematic_map` (nside=64) | ~76 ms |
| `generate_systematic_maps` (nside=64, 5 families) | ~380 ms |
| `pixelize_catalog` (100k galaxies, nside=64) | ~6.5 ms |
| `compute_overdensity` (nside=64) | ~65 μs |
| `apply_contamination` (n_pix=10k, n_sys=5) | ~830 μs |
| `invert_contamination` (n_pix=10k, n_sys=5) | ~1730 μs |
| `make_log_likelihood` first call (JIT compile) | ~227 ms |
| log-likelihood Gaussian (n_pix=10k, n_sys=5) | ~284 μs |
| log-likelihood skew-normal (n_pix=10k, n_sys=5) | ~592 μs |
| `rotate_templates` (n_sys=5, n_pix=10k) | ~177 μs |
| `correct_two_point_function` (n_sys=5, n_bins=15) | ~1084 μs |
| `likelihood_ratio_test` first call (2× JIT compile) | ~481 ms |
| `block_bootstrap_variance` (n_boot=50, nside=32) | ~32 ms |

## Scripts

| Script | Description |
|---|---|
| `scripts/compute_sys_weights.py` | Batch MCMC pipeline: runs additive + combined models on every DATA/RAND pair in a catalog dir, writes per-galaxy weight FITS files and a summary YAML for sum_stat |

### compute_sys_weights.py — key design points

- Template loading and normalisation mirrors notebook cell 9 (same `_NSIDE_ZFILL` dict, same standardise/minmax logic).
- PCA rotation applied before MCMC; parameters back-transformed with `transform_params_from_rotated`.
- Per-galaxy weight formula:
  - `WEIGHT_ADD  = 1 / max(1 + Σ_i a_i_add  * t_i(p), 0.01)`  — additive model MAP `a`
  - `WEIGHT_COMB = 1 / max(1 + Σ_i b_i_comb * t_i(p), 0.01)`  — combined model MAP `b`
- Galaxies in pixels with no template coverage get weight 1.0.
- Weights FITS file is same row order as the DATA catalog; MAP params stored in FITS header.
- Summary YAML at `data/sys_weights/summary_NSIDE{NNNN}.yaml` — consumed by sum_stat.
- `data/sys_weights/` is in `.gitignore`.
- GPU: `--device gpu` sets `JAX_PLATFORMS` before JAX init; `--vectorize` (default True on GPU) uses `jax.vmap` so all walkers are evaluated in one GPU kernel per step.  `inference.py:run_mcmc` now accepts `vectorize=True` which passes to both `make_log_prob` and `emcee.EnsembleSampler`.

## Notebooks

| Notebook | Description |
|---|---|
| `notebooks/demo.ipynb` | End-to-end pipeline on a 3-template synthetic dataset |
| `notebooks/paper_analysis.ipynb` | Interactive reproduction of Berlfein et al. 2024 (exploration only) |
| `notebooks/observed_data_analysis.ipynb` | Demo pipeline for observed survey data |
| `notebooks/gaia_systematic_maps.ipynb` | Build GAIA DR2 HEALPix maps interactively |
| `notebooks/ls10_systematic_maps.ipynb` | Build LS10 HEALPix maps interactively |

Deprecated notebooks (pre-refactoring) are archived in `notebooks/deprecated/`.

## Reference

Berlfein et al. 2024, arXiv:2401.12293
