# Changelog

All notable changes to `sys_mapping` are documented here.

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
