# Changelog

All notable changes to `sys_mapping` are documented here.

## [1.1.0] — 2026-07-08

Deeper JAX acceleration of the inference core. The MCMC stage dominated
per-sample wall time (30–90 min/sample on CPU, vs seconds for OLS/ISD/2PCF);
this release replaces the gradient-free emcee sampler on the critical path.

### New features

- **Analytic conjugate posterior for the additive model**
  (`sys_mapping.inference.run_additive_analytic`). The additive model is
  linear-Gaussian, so its posterior is Normal-Inverse-Gamma in closed form.
  Draws the exact posterior (no Monte-Carlo autocorrelation) in milliseconds,
  replacing the ~80 s emcee `MCMC-add` run. Matches an emcee chain's posterior
  mean and covariance to Monte-Carlo error, but is exact.

- **Gradient-based NUTS via BlackJAX** (`sys_mapping.nuts.run_nuts`) for the
  non-linear `combined` / skew-normal models. The JAX log-likelihood is already
  differentiable; NUTS explores the posterior with gradients (far higher
  effective sample size per step than emcee's stretch move) and runs the whole
  chain — window adaptation + sampling — under `jax.lax.scan`, eliminating the
  Python per-step loop and per-walker host↔device syncs. `σ > 0` is handled by
  an `exp` reparameterization (no hard `σ_min` discontinuity). Multiple chains
  run via `jax.vmap`; chain count auto-selects from the backend (CPU: 4,
  GPU: 8). Reports `rhat`, `ess`, and `num_divergences`.

- **Unified sampler dispatch** in `run_decontamination` via a new `sampler`
  argument: `"auto"` (default — analytic for `MCMC-add`, NUTS for `MCMC-comb`
  and skew), `"analytic"`, `"nuts"`, or `"emcee"` (the legacy gradient-free
  sampler, retained as the validation baseline). The result dict now carries
  `sampler_backend`, `rhat`, `ess`, and `num_divergences`.

- **CLI**: `scripts/run_ls10_analysis.py` and `scripts/compute_sys_weights.py`
  gain `--sampler {auto,analytic,nuts,emcee}`, `--n-chains`, `--nuts-warmup`,
  and `--nuts-samples`. Convergence diagnostics are written to the summary
  YAML/JSON.

- **JAX ISD Δχ² significance for `poly_order > 1` and `fracdet` weighting**
  (`sys_mapping.diagnostics`): a vmapped `_one_isd_poly` kernel closes the
  previous NumPy fallback, so cubic (ISD-3-style) and coverage-weighted
  pre-selection now run in JAX (reproduces the NumPy fallback to `~1e-13`).

- **JAX null-test cross-correlations** (`null_test_cross_correlations`):
  the correlations and permutation p-values are computed with a single
  `jax.vmap` over resamples (correlations reproduce the NumPy path exactly).

- **Opt-in JAX ISD reweighting loop**: `iterative_systematics_decontamination`
  gains `backend="jax"`, a `lax.while_loop` port of the fixed-point iteration
  (numerically identical to `"numpy"`, verified to `~1e-14`). The default stays
  `"numpy"` — on CPU the BLAS path is faster; `"jax"` is for GPU / device-resident
  pipelines.

- **Parallel GLASS pre-selection mocks**: `isd_template_significance` (and
  `run_decontamination(..., preselect_n_jobs=...)`, `--preselect-n-jobs`) gain a
  `n_jobs` option that runs the embarrassingly-parallel, GLASS/healpy-bound mock
  loop over processes with `joblib`. Bit-identical to serial (same per-mock seeds).

### Dependencies

- Added `blackjax >= 1.2` and `joblib`; `jax` pinned to `>= 0.9`.

### Notes

- Sampler results are statistically equivalent to the emcee baseline but not
  bit-identical (the analytic additive path is *more* accurate). Validate with
  `--sampler emcee` before comparing against v1.0.0 outputs. Default flips to the
  new samplers via `sampler="auto"`.

---

## [1.0.0] — 2026-06-11

### New features

- **Two-stage pre-selection pipeline** (`sys_mapping.diagnostics`, `sys_mapping.model_selection`)
  - `snr_template_ranking`: three SNR estimators — Pearson `|r|` (`"data"`),
    per-template OLS `|t|`-stat (`"template"`), and ISD Δχ² (`"isd"`).
  - `isd_template_significance`: GLASS systematic-free mocks on the same
    footprint → mock-based p-values for each template.
  - `snr_preselect`: convenience wrapper that returns a ranked, filtered
    `SnrPreselectionResult`.
  - `run_decontamination` now accepts `preselect=True` (and matching keyword
    arguments) to run Stage 1 automatically before any of the six Stage 2
    methods.

- **JAX acceleration** for all three SNR ranking methods:
  - `"data"`: batched Pearson correlation via `jax.jit` + matrix multiply.
  - `"template"`: per-template OLS via `jax.vmap`.
  - `"isd"`: vmap over templates, fixed-size bins, analytic linear regression —
    replaces the Python loop and `np.polyfit` for `poly_order=1`.
  - NumPy fallbacks retained for `poly_order > 1` and `fracdet` weighting.

- **Footprint-aware GLASS mocks** in `isd_template_significance`:
  - New parameter `n_total_footprint`: number of galaxies in the survey
    footprint. The function auto-computes the full-sky count as
    `n_total = n_total_footprint × N_full / N_good` so that after trimming
    to `good_pixels`, the mock surface density matches the data.

- **CLI**: `scripts/run_ls10_analysis.py` now supports `--preselect`,
  `--preselect-method`, `--preselect-n-top`, `--preselect-p-threshold`,
  `--preselect-n-mocks`.

### Bug fixes

- **ISD Δχ² overflow** (`SNR ≈ 10⁶¹`): bins where all pixels have identical
  overdensity (e.g. empty HEALPix cells with δg = −1) are now skipped instead
  of flooring σ to 1e-30, preventing 1/σ² overflow.

### Documentation

- New results page `docs/results_snr_preselection.rst`: 15 M galaxy,
  20-template validation at NSIDE=32 with three contamination levels, timing
  figures, and mock-convergence rule-of-thumb N_mocks ≥ max(20, ⌈5/α⌉).
- `docs/methods.rst`: new *Template pre-selection (Stage 1)* section with
  two-stage workflow code snippet.
- `README.md`: updated pipeline diagram, pre-selection table, quick-start
  snippet, and CLI example.

### Breaking changes

- The `"template"` method in `snr_template_ranking` now performs **per-template
  OLS** (one regressor at a time) instead of the previous joint multi-regressor
  OLS. Rankings are unchanged for uncorrelated templates; for correlated
  templates the per-template t-statistic is more appropriate for pre-selection.

---

## [0.9.5] — 2026-03-xx

- Simulation pipeline, five bug fixes, updated results.

## [0.9.0] — 2026-01-xx

- Initial public release.
