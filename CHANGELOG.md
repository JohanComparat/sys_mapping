# Changelog

All notable changes to `sys_mapping` are documented here.

## [Unreleased]

## [1.4.0] — 2026-09-15

Every calibrated statistic draws its null from a validated matched spectrum, detection
significances are scored against that null with a family-wise p-value, each LS10 sample is
analysed at the resolution its occupancy supports, and the two-point correction carries
every cross-template term. The mock nulls, the likelihood-ratio null and the template
correlation matrix are one to two orders of magnitude faster, and the test suite measures
line and branch coverage.

### Removed

- **`get_mle_params`.** It returned the per-parameter posterior median, which is neither a
  maximum-likelihood point nor the joint MAP. Use `posterior_median_params` to report a
  point estimate and `refine_to_mle` wherever a likelihood is compared.

### Added

- **`calibrated_template_significance`** — each template's least-squares amplitude over its
  scatter across uncontaminated realisations, per-template p-values, and a family-wise
  p-value for the largest significance, with leave-one-out. Over 2000 clean fields with 1000
  realisations, `family_wise_p <= 0.05` occurs in 4.85 ± 0.48% and `<= 0.0027` in
  0.35 ± 0.13%; the independent-pixel significance exceeds 3 on some template in 99.3%.
- **`residual_template_correlation_test`** — correlation of the corrected density with each
  template, χ² over the null variance, leave-one-out null χ² and p-value. For templates the
  correction fitted the correlation is zero by construction, so the test applies to templates
  or transforms the fit did not use.
- **`choose_nside_by_occupancy`** — the finest NSIDE whose mean footprint occupancy reaches a
  floor (default 25 galaxies per pixel), counting pixels by the rule of `compute_overdensity`.
- **`template_correlation_matrix`** — the full `(n_sys, n_sys, n_θ)` template correlation from
  the values each galaxy carries, with optional per-object weights.
- **`standardise_on_footprint`** — zero mean and unit rms over the fitted pixels.
- **`lrt_from_maxima`** — additive and combined maxima of many fields at once (least squares,
  and L-BFGS vmapped over fields), for the likelihood-ratio null. 8 fields of 7 040 pixels with
  11 templates take 0.15 s after compilation, against about 50 s for per-mock NUTS fits refined
  to their maxima; λ agrees to a relative 8e-7. Each L-BFGS run stops at convergence (largest
  gradient component below 1e-9) or after `n_iter` iterations.
- **`draw_null_overdensity`, `generate_glass_null_overdensity`** — null realisations drawn as
  Poisson counts per footprint pixel from the GLASS field: 0.01 s per realisation against
  0.97 s for a pixelised catalogue, with the same pixel variance and amplitude scatter.
- `run_nuts(dense_mass_matrix=True)` (default) — on the LS10 combined fit at NSIDE 32, half the
  leapfrog steps per iteration and 2.5 times the minimum effective samples per second.
- `isd_template_significance(draw=)`, `"pixel"` by default.
- `scripts/run_ls10_analysis.py`: `--min-per-pixel`, `--significance-n-mocks`,
  `--significance-seed`, `--null-cl-file` (alias of `--lrt-null-cl-file`),
  `--allow-parametric-null`, `--null-draw`, `--lrt-null-method`, `--ct-max-galaxies`,
  `--ct-auto-only`.
- `scripts/jax_coverage.py`, `scripts/profile_by_library.py`, `scripts/coverage_report.py` and
  `docs/coverage.rst`: static JAX share, wall time by library, per-module line and branch
  coverage. `tests/test_jax_transformability.py` records which public numeric functions survive
  `jit`, `vmap` and `grad`.
- `scripts/plot_ls10_occupancy_products.py`; `scripts/generate_results_ls10_summary.py` writes
  the LS10 summary, recommendations and nine sample pages from the issued products.

### Changed

- **GLASS nulls need a spectrum somebody chose.** `generate_glass_fullsky_mock`,
  `generate_glass_delta_map`, `isd_template_significance` and `run_decontamination`'s
  pre-selection take `cl_amplitude=None`; with neither a matched spectrum nor an amplitude they
  warn and use the power law. `run_ls10_analysis.py` refuses to build a null without a matched
  spectrum unless `--allow-parametric-null` is given.
- **`load_matched_cl`** serves the validated spectrum nearest at or above the requested
  resolution, and the finest below only when none is finer.
- **`template_correlation_matrix`** builds cross terms from auto-correlations of summed
  fields, `ξ_ij = [ξ(t_i + t_j) − ξ_ii − ξ_jj]/2`: exact, and 1.7 times faster at 11 templates.
- **`make_log_likelihood`, `refine_to_mle` and `run_nuts`** compile once per configuration and
  take the data as arguments. Repeat calls on 20 000-pixel fields: likelihood-ratio test
  0.102 → 0.003 s, `refine_to_mle` 0.266 → 0.019 s, `run_nuts` 3.39 → 1.71 s.
- **The ISD, ranking and null-test statistics run on their JAX kernels only**; the NumPy loops
  they are checked against live in the test suite.
- `debias_params`, `debias_params_matrix` and `standardise_on_footprint` accept JAX arrays and
  trace under `jit`, `vmap` and `grad`; NumPy inputs behave as before.
- `run_ls10_analysis.py` issues each sample at its occupancy-chosen resolution, corrects w(θ)
  with the full cross-template matrix, stores the calibrated significance, draws every null per
  pixel, and builds the likelihood-ratio null from batched maxima. The data statistic keeps, per
  model, the higher-likelihood maximum of the NUTS refinement and the batched optimiser.
- The test suite collects the 70 docstring examples and measures branch coverage; CI runs the
  fast suite on pull requests and everything on `main` and nightly, and fails below 97%.

### Fixed

- `refine_to_mle` raised for `model="multiplicative"`: its analytic start passed `b=None`.
- `run_decontamination` with ISD pre-selection and a p-value cut passed ISD a threshold for the
  templates before the cut.
- `generate_glass_delta_map` raised `NameError` on every call.
- Under `--skewed`, the LS10 likelihood-ratio test compared a Gaussian additive fit with a
  skew-normal combined fit.
- A `--template-dir` without FITS maps silently switched the run to synthetic templates; it now
  stops.
- `isd_template_significance`'s `cl_amplitude=5e-4` default silenced the unmatched-spectrum
  warning.
- `compute_sys_weights.py` recorded `WEIGHTCON = library` for weights it reconstructs in the
  linear form; it records `linear-from-a_hat`.
- `lrt_from_maxima` returned the start (b = 0, λ = 0) when an L-BFGS step crossed a pixel where
  `1 + b·t = 0` and the run ran off along the |b| → ∞ ridge; it happened on one of 100 NSIDE-64
  synthetic mocks. The combined maximum is sought where every pixel's efficiency is positive, each
  run keeps its best finite point and stops once the value or gradient is not finite, and it is
  restarted from that point up to three times.
- With pre-selection, `run_decontamination` reused a linear, equal-width ISD null as the stopping
  threshold of an ISD fit of another degree on equal-occupancy bins; the pre-selection null now uses
  the degree, bin count and binning of the ISD method it serves. `run_ls10_analysis.py` calibrates the
  ISD threshold with its own cubic, equal-occupancy null and no longer reuses the pre-selection null.
- The validation and characterisation scripts (`run_validation.py`, `run_systematic_tests.py`,
  `run_mock_analysis*.py`, `run_simulation_tests.py`, `run_snr_preselection_demo.py`) run MCMC-add and
  MCMC-comb through `run_decontamination` (analytic posterior and NUTS; `--sampler`, `--nuts-warmup`,
  `--nuts-samples`), calibrate every ISD threshold on uncontaminated realisations of their own
  generator, and pass an explicit GLASS spectrum. `run_simulation_tests.py` reads the Uchuu catalogue
  from `.fits.gz` and stops when it is missing unless `--glass-only` is given.
- `run_mock_analysis.py`, `run_mock_analysis_progressive.py` and
  `run_mock_analysis_real_templates.py` evaluated the likelihood-ratio statistic at posterior
  medians, where it can be negative; they take it between the maxima from `lrt_from_maxima`.

## [1.3.0] — 2026-09-14

Three algorithms were doing something other than what they were named for, and the
LS10 products are re-issued on a basis where the amplitudes mean what they say.

`iterative_systematics_decontamination` now implements published ISD rather than a
multivariate polynomial expansion.  The likelihood ratio is evaluated at two
likelihood maxima, so it is non-negative and its magnitude carries meaning.  The
template basis is standardised over the analysis footprint, and the template
correlations are measured from the galaxies, so the two-point correction acts over the
whole measured range instead of the nine widest bins of thirty.

The 27 re-issued products at NSIDE 32, 64 and 128 carry `WEIGHTVER = 3`.  Their
amplitudes are not comparable with a version 2 product, whose basis is normalised
somewhere other than where the fit used it.

### Added

- **`apply_nonlinear_contamination`, `TemplateResponse`, `evaluate_response`,
  `RESPONSE_KINDS`** — injection of a template response that is not linear.

  Every simulation this package ships injects `apply_contamination`, i.e.
  `δ_g(1 + Σ b_i t_i) + Σ a_i t_i`, which is linear in every template. A linear
  marginal fit is already sufficient for that, so no campaign run to date can
  separate `ISD-1` from `ISD-3` — and none does: across all 900 NSIDE-64 cells of
  campaign `20260907c` the two recover the same amplitude to within 1%, and
  `ISD-3`'s only measurable difference is a *worse* false-positive rate. That is a
  property of the injection, not of the method.

  The new path injects `1 + δ_obs = (1 + δ_true)·Π_i (1 + F_i(t_i))` — the
  per-template selection efficiency that ISD's weight `Π_j 1/(1 + F̂_j)` actually
  inverts, and which is *not* Eq. 13. Six response shapes: `linear`, `quadratic`,
  `cubic` (inside a cubic marginal basis) and `tanh`, `threshold`, `exp` (outside
  it). Every shape is rescaled to the same `rms(F)`, or a ranking of shapes is a
  ranking of amplitudes.

  Measured over a 50-seed campaign at NSIDE 32 on the real LS10 templates, two of
  five contaminated, scoring by the fractional L2 error of the recovered `w(θ)` and
  comparing **per seed** (the two methods see the same mock and the same injection):

  | injected `F(t)` | low | medium | high | in a cubic basis? |
  |---|---|---|---|---|
  | `t` (control) | 1.07 / 42% | 0.83 / 76% | 0.82 / 92% | exactly |
  | `t + κt²` | 0.89 / 78% | 0.68 / 98% | 0.53 / 100% | exactly |
  | `t + κt³` | 0.96 / 64% | 0.85 / 96% | 0.73 / 100% | exactly |
  | `tanh(αt)/α` | 0.93 / 56% | 0.53 / 92% | **0.22** / 100% | no |
  | `t·1[t>t₀]` | 0.69 / 84% | 0.39 / 100% | **0.36** / 100% | no |
  | `(e^{αt}−1)/α` | 1.00 / 46% | 0.89 / 84% | 0.86 / 84% | no |

  *median ratio of ISD-3's error to ISD-1's on the same seed / fraction of seeds
  ISD-3 wins.* Every medium- and high-amplitude cell is significant at p < 0.02
  (Wilcoxon signed-rank); none of the low-amplitude cells below 60% is significant.

  The advantage appears with amplitude, is largest on the shapes a cubic *cannot*
  represent (`tanh`, `threshold`), and is present even on the linear control,
  because binning a lognormal density against a skewed survey-property map curves
  the marginal relation before any contamination is applied.

  The exponential is the exception and it locates the real limit: rescaled to fixed
  `rms(F)` on a template as skewed as `GALDEPTH_Z` (skew +5.9, reaching +19σ), an
  exponential response puts `|F| > 0.1` in 0.03% of pixels — all inside the
  outermost quantile bin, where a binned fit of any order has one degree of freedom.
  Splitting the seeds: 1.03 where a contaminated template has skew > 3, 0.32 where
  neither does. **The binding constraint on ISD is `n_bins`, not `poly_order`.**

- **`characterisation/isd_hyperparameter_sweep.py`** (benchmark repo) — the first scan
  of the ISD parameters. 50 settings x 18 configs x 20 seeds. Two results:

  * **`n_bins = 20` strictly dominates the shipped `n_bins = 10`** at `poly_order = 3`:
    better residual (0.028 vs 0.032), better selection precision (0.925 vs 0.870) and
    better recall (0.835 vs 0.808). Not applied yet — 20 bins has not been tested on
    the sparse LS10 samples at NSIDE >= 128, where the `N_beta >= 2` bin-admission rule
    could start rejecting bins.
  * **The usable polynomial order is set by the bin count**, not chosen independently.
    At 5 bins the best order is 2 and a cubic is already worse than a *linear* fit at
    20 bins; at 40 bins a quartic wins. `d=5` at 10 bins (0.200) is worse than `d=2` at
    10 bins (0.052). This is the practical content of the DES Y6 choice of `d=3`: it is
    the highest order ten bins support.

  Also measured: equal-occupancy binning beats equal-width by 2-3x at `d=1`;
  `threshold` is a clean monotone precision/recall dial (T=1 → 0.717/0.899,
  T=5 → 0.991/0.700); `max_reuse=1` triples the residual and above 3 changes nothing;
  `w_max` and `bad_pixel_frac` changed nothing on any metric at any setting, so they
  are insurance rather than knobs. Dropping the calibration gives the *lowest* residual
  in the whole scan (0.009) and the worst precision (0.516) — it removes more because
  it removes things that are not there.

- **`simulation.make_response_grid`** and `run_simulation_tests.py --responses`,
  `--response-kinds`, `--n-contaminated`. The last is what makes greedy *selection*
  measurable: the existing grid contaminates all five templates at equal amplitude,
  so there is no true negative for a selection rule to get wrong.

- **`run_simulation_tests.py --isd-null-cl-file`** — the ISD calibration null can
  now be a matched spectrum. Without it the null is the parametric power law at
  `cl_amplitude=5e-4`, which under-clusters LS10 by ~25× and leaves the calibrated
  threshold anticonservative.

- **`run_simulation_tests.py --isd-kwargs`** and the `isd_poly_order`,
  `isd_max_reuse`, `isd_w_max`, `isd_bad_pixel_frac` arguments to
  `run_decontamination`. `poly_order` was previously pinned by the method name and
  the other three were unreachable from any driver, so no hyper-parameter sweep was
  possible. `ISD-<d>` is now accepted for any degree.

### Added

- **`standardise_on_footprint`** — standardise a template basis over the pixels it
  is about to be fitted on.

  `load_real_template` and `load_templates_from_dir` normalise each survey-property
  map over that map's own valid region, which is larger than any one sample's
  footprint. Restricted to the footprint the basis is no longer standardised: the
  eleven LS10 maps at NSIDE 64 have per-template rms spanning 0.905 to 5.77 and a
  relative mean of 0.729, so the covariance eigenvalues sum to 44.4 rather than
  `n_sys = 11` with a leading eigenvalue of 37.5. Calling this after masking gives
  exactly 11.00 with a leading eigenvalue of 5.39, and drops the basis condition
  number from 1443 to 208.

  It cannot be done at load time: the loader does not know which pixels the fit will
  use. Both production scripts now call it after `assign_template_values`, record
  the means and rms they divided out in `params.json` under `template_basis`, and
  stamp `TPLBASIS` into the FITS header. `WEIGHTVER` goes to **3** for a
  footprint-standardised product. `--no-footprint-standardise` reproduces the old
  basis and keeps `WEIGHTVER = 2`.

  Synthetic bases are barely affected — 0.971 to 1.030 under a Galactic cut at both
  resolutions, eigenvalues summing to 4.93–5.05 against 5 — so the simulation-based
  results are unchanged. Only the real-template LS10 results move.

### Fixed

- **The template two-point functions are measured from the galaxies.**
  `run_ls10_analysis.py` ran TreeCorr KK on the rotated templates at the HEALPix
  pixel centres. No pair of distinct pixels is separated by less than the pixel
  scale, so `xi_i` came back as exactly zero below it while `w(theta)` is measured
  from the catalogue down to 0.5'. The correction was therefore identically zero
  over 21 of 30 bins at NSIDE 64 (below 49') and 24 of 30 at NSIDE 32 (below 94'),
  the range carrying most of the signal. A template is constant within a pixel, so
  its correct `xi_i` at sub-pixel separations is its *variance*, not zero.

  Each galaxy now carries the rotated-template value of its pixel and the KK
  correlation runs on the galaxy positions, over the same bins as `w(theta)`.
  Validated against the pixel version on a synthetic footprint: the two agree to
  within 2% above the pixel scale (0.752 against 0.737 at 115', 0.195 against 0.198
  at 270') and the galaxy measurement returns 0.93 to 1.01 of the template variance
  below it, where the pixel one returns zero in 21 of 30 bins. Cost is 29 s per
  template at 3M galaxies. `--ct-from-pixels` restores the old behaviour.

- **`correct_two_point_function` warns where the correction overshoots the
  signal.** `Σ_i ã_i² ξ_i(θ)` is a sum of squares against auto-correlations, so
  nothing bounds it by `w_obs(θ)`. At separations where the galaxy signal has
  decayed but the survey-property maps are still coherent it exceeds it, and the
  "corrected" correlation function comes back negative with no error. On the
  shipped LS10 grid this happens for `ISD-3` in all 18 cells, reaching
  `w_corr/w_obs = -39` at NSIDE 64 and `-232` at NSIDE 32, while the other five
  methods stay between 0.60 and 0.88. The bins are not usable; the caller decides
  whether to drop them, restrict the fitted range, or refuse the cell.

- **`compute_covariance_matrix` warns on a basis that is not standardised over
  the pixels supplied.** `load_templates_from_dir` normalises each survey-property
  map over that map's own valid region, which is larger than any one sample's
  footprint. Restricted to the pixels the fit uses, the eleven LS10 maps at
  NSIDE 64 have rms spanning 0.905 to 5.77 and a relative mean of 0.729, so the
  eigenvalues sum to 44.4 rather than `n_sys = 11` with a leading eigenvalue of
  37.5. Amplitudes fitted against that basis are not in units of one standard
  deviation of the template, and it is the same scaling that makes the two-point
  correction overshoot above.

- **The zero-mean warning no longer fires on every centred sample.** The sample
  mean of `n_pix` unit-variance values has standard error `1/sqrt(n_pix)`, so the
  fixed relative threshold of `1e-3` fired on any finite draw of a genuinely
  centred basis. The tolerance is now `max(1e-3, 5/sqrt(n_pix))`.

- **`run_ls10_analysis.py` gained the `--skewed` flag**, defaulting off, matching
  `compute_sys_weights.py`. Previously one script defaulted the skew-normal
  likelihood on and the other had no flag at all, so a column named `WEIGHT_SYS`
  meant a different model depending on which script wrote it. `run_decontamination`
  now also returns `gamma_hat`, which a caller refining the fit to an MLE needs in
  order to pack a starting point.

- **`plot_ls10_wtheta_corrected.py` no longer crops the correction away.** The
  ratio panel was gated on `w_corr > 0` and the x-axis fixed at 70', which between
  them hid every bin where the correction overshoots — the only bins where it acts
  at all. The axis now spans the measured range and the binning in the title is
  read from the data rather than restated.

- **The paper Makefile rebuilds on a figure or archive edit.** `figures/*.tex` and
  `findings_archive.tex` were not prerequisites of the PDF, so editing a figure
  source left a stale build that still passed `make check`.

- `run_wtheta_recovery` kept only `a_hat`, `b_hat` and `elapsed_s`, discarding the
  per-step coefficients, `n_steps`, `stopped_on`, `significance` and `calibrated`
  that `run_decontamination` returns. On a non-linear response the curvature lives
  entirely in those coefficients, so the results files could not have answered the
  question even with the right injection. `n_floored` was computed by `ISDResult`
  and dropped by `run_decontamination`; it is now returned.

- `null_test_cross_correlations` gained a warning that `max_i |r(w, t_i)|` is not a
  scalar goodness-of-fit. For an additive correction the statistic depends on the
  *support* of `â`, not its size: a single-template correction gives `|r| → 1` at
  any amplitude (1.0000 at `â = 1e-6`, 0.9886 at `â = 1e-1`), so a sparser and more
  accurate correction scores *worse*, and a method that fits nothing scores zero.

### Changed — breaking

- **`iterative_systematics_decontamination` now implements ISD.** Up to and
  including v1.2 this function — cited to Rodríguez-Monroy et al. in its own
  docstring, and exposed as the methods `ISD-1` and `ISD-3` — implemented a
  different algorithm from the one that paper and its DES Y1/Y3/Y6 predecessors
  describe.

  *What it did:* expand the template set into every monomial
  `t_i·t_j·t_k` up to total degree `d` — cross-products between different
  templates included, `binom(n_s+d, d) − 1` columns, 55 for `n_s = 5` and
  `d = 3` — fit them all simultaneously by unregularised iteratively reweighted
  least squares, and then build the weight from the first `n_s` linear
  coefficients alone. The fifty discarded columns are strongly collinear with the
  five retained ones, so fitting and dropping them inflated the variance of
  precisely the coefficients that determined the weight. On LS10 this gave
  `rms|â| ≈ 7.6`, some 35× the OLS solution on the same data, and a weight map
  saturated across the whole footprint.

  *What it does now:* for each template independently, bin the footprint into 10
  equal-occupancy bins of that template's value, fit the binned mean density with
  a polynomial of degree `d` **in that one template's value**, score it against a
  mock-calibrated `Δχ²/Δχ²₆₈`, apply the inverse of the single most significant
  fit as a weight, re-measure, and repeat until every template falls below
  `threshold` (default 2, the DES Y6 value). The design matrix of every fit is
  `n_bins × (d+1)` however many templates there are.

  `ISD-1` and `ISD-3` keep their names; `poly_order` keeps its name and now means
  what it means in the literature. The v1.2 routine survives as
  **`polynomial_ols_decontamination`**, documented as not being ISD, so the v1.2
  benchmark and timings stay reproducible.

  *Signature changes.* `iterative_systematics_decontamination` returns an
  `ISDResult` dataclass (`weights`, `a_hat`, `steps`, `significance`, `n_steps`,
  `stopped_on`, `calibrated`, `n_floored`) instead of the tuple
  `(weights, alpha_hat_all, n_iterations)`; `max_iter`, `tol`, `lambda_poly` and
  `backend` are gone, replaced by `n_bins`, `binning`, `threshold`, `chi2_68`,
  `max_steps`, `max_reuse`, `fracdet`, `w_max` and `bad_pixel_frac`. In
  `run_decontamination`, `isd_max_iter` and `isd_lambda_poly` are replaced by
  `isd_n_bins`, `isd_binning`, `isd_threshold`, `isd_chi2_68`, `isd_max_steps`
  and `isd_fracdet`; the result dict gains `isd_steps`, `isd_significance`,
  `isd_stopped_on` and `isd_calibrated`, and loses `isd_outlier_mask` and
  `isd_masked_fraction` (the two-pass outlier scheme is unnecessary now that the
  iteration stops on insignificance).

  *Never extrapolate a binned fit.* `F̂` is evaluated at `clip(t, t_lo, t_hi)`,
  the outermost bin centres it was constrained by. On real LS10 templates this is
  the difference between convergence and divergence: `GALDEPTH_Z` reaches +26
  standardised units while its outermost bin centre sits near +2, so an unclipped
  cubic is evaluated three orders of magnitude beyond its support. Without
  clipping the significances *rise* along the iteration (`S = 9.0 → 14.8 → 36.2`
  on one template) as each over-corrected step manufactures a larger trend for the
  next, and 339 of 22 000 pixels hit the weight floor; with clipping the same run
  floors none and stops on the threshold. `isd_marginal_fit` therefore returns the
  supported range alongside the coefficients.

  *The reported amplitude is a projection, not a coefficient.* `a_hat[i]` is
  `<F̂_i(clip(t_i)) t_i> / <t_i²>`, summed over the steps that selected template
  *i*. Taking the degree-1 polynomial coefficient instead is correct for
  `poly_order=1` but not for `poly_order=3`: equal-occupancy bin centres of a
  skewed template span a narrow range, so the Vandermonde is poorly conditioned
  and the coefficients are large even when the curve is small. On a GLASS null
  with no injected contamination one accepted step returned
  `c = (0.026, 0.681, 3.343, 4.365)` — `rms|â| = 0.44` for a field whose true
  amplitude is zero. The projection gives `6e-4` on the same data.

  *Calibration.* Without `chi2_68` the threshold is in raw `Δχ²` units and the
  value 2 means nothing, so the function warns. When Stage-1 pre-selection has
  already run with `preselect_method="isd"`, `run_decontamination` takes
  `Δχ²₆₈` from the GLASS null it already generated rather than paying for it
  twice.

### Added

- **`data/glass_match/` — 25 validated per-setup GLASS spectra.** Retrieved from the
  cluster campaign: all nine LS10 samples at NSIDE 32 and 64, seven at NSIDE 128,
  each fitted band by band and validated on 8 or 16 seeds held out of the fit. 24 of
  25 pass, with large-scale power ratios in 0.912–1.020 against a 10 % tolerance.
  `sigma_hat_mock` now spans 0.27–1.22 per sample where the default power law gave
  0.078 against a data value of 0.387. These are what `--lrt-null-cl-file` should
  point at; without them every `load_matched_cl` call returned `None` and every run
  silently fell back to the power law.

- **`sys_mapping.diagnostics.isd_marginal_fit`** — the per-template binned
  polynomial fit ISD is built from, returning both `Δχ²` and the fitted
  coefficients (ascending powers). `snr_template_ranking(method="isd")` returns
  only the former and is unchanged.

- **`sys_mapping.diagnostics.vet_templates_against_tracer`** — weighted Spearman
  rank correlation of each template against an external tracer of true structure
  (CMB lensing κ, Compton-*y*, weak-lensing convergence), with delete-one-patch
  jackknife errors. A template that correlates with real structure lets the
  regression fit the signal, and no care in the fit will reveal it: the
  contamination is detected at high significance and the clustering comes out low,
  on mocks as well as data. Eggert & Leistedt (2023) demonstrate exactly this for
  templates built from Legacy Survey image stacks, which is the data this package
  is applied to. (Weaverdyck et al. 2026, Sec. III A.)

- **`sys_mapping.correction.estimate_overcorrection_bias`** and
  **`debias_two_point_function`** — run the weighting on contamination-free mocks;
  whatever it changes there is over-correction, not systematics, and is subtracted
  from the data vector. Estimator-agnostic: the caller supplies the `w(θ)`
  measurement so the debias term uses the same binning and mask as the data.
  (Weaverdyck et al. 2026, Eqs. 21–23.)

- **`sys_mapping.covariance.method_marginalised_covariance`** — adds
  `Δᵢ Δⱼ` to the covariance, with `Δ` the difference between two methods'
  `w(θ)`. This package offers six methods and they disagree; reporting one of them
  with its own error bar prices that disagreement at zero. Supports block-diagonal
  application so each tomographic bin can prefer a different method.
  (Weaverdyck et al. 2026, Eq. 24.)

- **`sys_mapping.bootstrap.assign_spatial_patches`** — the patch labelling used by
  the bootstrap and jackknife, made public so the same partition can serve as
  cross-validation groups and as jackknife patches.

- **`sys_mapping.maps.inverse_variance_pixel_weights`** — the per-pixel weight
  `A_k²/(N_k + 2)`: coverage area squared over Poisson variance, with the
  regulariser keeping empty pixels finite. (Weaverdyck et al. 2026, Eq. 8.)

- **Spatially compact cross-validation for ElasticNet.** `patch_ids=` and
  `pixel_weights=` on `elasticnet_contamination_fit`. `ElasticNetCV` splits into
  contiguous index chunks by default, which mix spatially; on a clustered field
  the held-out pixels are then correlated with the training pixels, the prediction
  error is under-estimated, and the cross-validation picks too weak a penalty.
  With `patch_ids` the folds are whole patches (`GroupKFold`) and are spatially
  disjoint. `cv_info["cv_spatial"]` records which was used. DES Y6 uses ~200
  patches of about 5° diameter.

- **`--isd-n-mocks` in `scripts/run_simulation_tests.py`**, and `ISD-3` added to
  its `--methods` choices (it was absent, which is why no ISD-3 column appears in
  `data/simulations/*/results_summary.json`). The ISD `Δχ²₆₈` is calibrated once
  per mock source — the null depends on the footprint, resolution and surface
  density, not on the injected contamination — and reused across all nine
  configurations. `run_wtheta_recovery` gains `method_kwargs=` to carry it.

### Fixed

- **`load_matched_cl` preferred a failed fit over a validated one.** The exact-NSIDE
  file won whenever it existed, even when its `validation.passed` was false, so a
  sample whose fit failed at the requested resolution raised instead of using a
  spectrum that had passed at another. That contradicts the function's own stated
  rationale — the spectrum belongs to the sample and its footprint, and resolution
  limits what can be *checked*, not what can be *used* — and it is not hypothetical:
  LS10 `logM ≥ 11.5` fails at NSIDE 64 and passes at 128, so the NSIDE-64 LRT for that
  sample would have crashed. It now prefers the finest *validated* candidate and warns
  when it falls through, while still refusing outright when nothing passed.

- **ISD template SNR could rank a pure-noise template above a contaminated
  one.** The two JAX ISD kernels estimated the per-bin variance as
  `E[g²] − E[g]²`. For a bin holding a single pixel that is exactly zero in
  eager arithmetic, but under `jit` XLA contracts the expression into an FMA
  whose rounding returns ≈1.5 × 10⁻²⁰ — just above the `> 1e-20` guard the code
  used to reject degenerate bins. The bin was then admitted with an inverse
  variance of ≈7 × 10¹⁹, which swamped the Δχ² sum: on the unit-test universe a
  noise template scored 6.0 × 10¹⁷ against 389 for a genuinely contaminated one,
  so `snr_preselect(method="isd")` ranked it first. Because the fault lay in
  compiler fusion, it appeared only in the compiled path — the NumPy fallback in
  the same function was always correct — and could move with the XLA version.

  Both kernels now use the centred (two-pass) variance, require at least two
  pixels in a bin (one pixel carries no variance information whatever the
  arithmetic reports), floor the variance relative to the field's own scatter
  rather than at an absolute 10⁻²⁰, and return 0 when fewer than two bins
  survive — matching the NumPy reference, which they now reproduce to 4 × 10⁻¹⁵.
  **Scope, measured rather than assumed.** The published LS10 weights are
  unaffected: all 36 `params.json` record no pre-selection. The Stage-1 demo
  figures in `docs/results_snr_preselection.rst` are also unaffected — they
  regenerate byte-identical, because that demo uses synthetic templates on a
  well-populated NSIDE-32 footprint where no bin ever holds fewer than two
  pixels, so the degenerate-bin path never fires. What *is* affected is ISD
  ranking on the real LS10 templates, whose skewed pixel distributions leave
  near-empty tails in equal-width bins (25 bins with fewer than two pixels
  across 11 templates at NSIDE 64). `method="data"`, `"template"` and
  `"peak"`, and every Stage-2 method, are unaffected throughout.

- **`cl_amplitude` could not reach either GLASS mock null.** Both
  `isd_template_significance` and `build_lrt_null` took the 5 × 10⁻⁴ default with
  no way to override it, so the null was drawn from mocks with
  σ_clus ≈ 0.089 against LS10's ≈ 0.397 — roughly 25× too little clustering
  variance, making every p-value anticonservative. The parameter is now threaded
  through both, and `run_ls10_analysis.py` exposes `--lrt-null-cl-amplitude`
  (recorded in `params.json` as `null_cl_amplitude`).

- **`run_ls10_analysis.py --preselect` silently wrote unit weights.** With
  pre-selection enabled, `a_hat` / `b_hat` come back with length
  `len(selected) < n_sys`, and the weight writer discarded any vector whose shape
  did not match the full template basis — so every method column, and
  `WEIGHT_SYS`, was exactly 1.0 (no correction) with no warning. The selected
  coefficients are now scattered back into the full basis via the new
  `expand_preselected_params` helper; the zeroing path is retained only for a
  genuine shape mismatch and now emits a warning. Covered by
  `tests/test_ls10_script.py`. Results already on disk are unaffected — the LS10
  runs did not use `--preselect`.

### Repository split

The paper and the benchmarks moved to their own repositories so this one is the
package and pipeline alone. Their **results stay documented here** and render from
committed snapshots, with no external checkout required.

| Moved to | What |
|---|---|
| [`sys_mapping_benchmark`](https://github.com/JohanComparat/sys_mapping_benchmark) (public) | `scripts/benchmark_pipeline.py`, `tests/test_timing.py`, `tests/test_benchmarks.py`, and the four characterisation scripts |
| `sys_mapping_paper` (private) | the LaTeX document, its sections, figures and `make_tables.py` |

- **CI is materially lighter**: 259 timing and benchmark cases no longer run on every
  push. `tests/test_benchmarks.py` had **zero assertions** — it was a benchmark
  wearing a test's clothes — and `tests/test_timing.py` contributed 240 parametrised
  cases whose 10 assertions are time budgets, better placed beside the harness.
- **New**: `docs/results_benchmark.rst`, generated by
  `docs/generate_benchmark_page.py` from `docs/_static/benchmark/`.
- Both new repositories accept `SYS_MAPPING_ROOT` to read this package's pipeline
  outputs, and fall back to their own committed snapshots when it is absent, so each
  builds standalone.

### Added

- **The pipeline document** — a standalone LaTeX document (now in the private
  `sys_mapping_paper` repository; see *Repository split* above)
  stating every equation the package implements with a `file:line` reference to the
  code that evaluates it, the implementation path, timing tables, the LS10 DR10
  results, and a verification section recording 15 findings where the code, the
  published equations and the documentation disagree. Tables are generated from
  files on disk by its `make_tables.py`, never transcribed.
- **`benchmark_pipeline.py`** (now in `sys_mapping_benchmark`) — reproducible
  timing harness covering the
  per-function API, the HEALPix utilities, all four Stage-1 ranking statistics and
  all six Stage-2 methods. Writes `benchmarks.csv` plus a `machine.json` provenance
  record (CPU, cores, RAM, library versions, commit, date, load average),
  superseding the undated benchmark table in the developer notes. Results are
  rendered in `docs/results_benchmark.rst`.
- **`docs/roadmap.rst`** — prioritised next steps.
- **`docs/api/nuts.rst`, `docs/api/plotting.rst`** — both modules are public and
  exported but had no API page.
- **`bash/ls10_mocklrt_ns32_gaps.sh`** — fills the NSIDE-32 mock-calibrated LRT
  cells still holding `calibration = "failed"`, concurrently.

### Documentation

- **Priors corrected.** `docs/methods.rst` claimed `a_i, b_i ~ N(0,1)` amplitude
  priors and a half-normal prior on σ. Neither is implemented: the prior is flat
  and improper on all amplitudes, with only a hard `σ > 1e-6` floor.
- **Weight conventions documented.** The README now states all three formulas in
  circulation and which one the FITS columns actually use, and notes that the two
  production scripts fit different likelihoods (skew-normal vs Gaussian).
- **PCA covariance propagation** in the README had `V` and `Vᵀ` transposed relative
  to its own convention, and described propagating only the diagonal; the code
  propagates the full covariance. Corrected.
- **Skew-normal formula** clarified: the location shift ξ enters the Gaussian
  quadratic form as well as the Φ argument. Removed a citation to commit
  `10c01d0`, which does not exist in this repository.
- **All four SNR ranking statistics** (`template`, `data`, `peak`, `isd`) are now
  listed in both places that describe them; each previously listed only three.
- **Dropped cross-terms documented.** The `w(θ)` correction and the amplitude-bias
  estimator keep only auto-correlations; PCA diagonalises the template covariance
  at zero lag, not at `θ > 0`. Also notes that the harmonic path subtracts a term
  linear in α while the configuration-space path subtracts the debiased square.
- **Simulation-test conclusions corrected.** The page claimed all methods reduce
  the `w(θ)` bias substantially and that MCMC-comb is the best correction; the
  committed summary table for the same runs shows 37 of 45 Uchuu method-cells with
  an improvement factor below 1, MCMC-comb worst almost everywhere.
- **Correlated-field caveat** on the likelihood ratio test added to the README,
  matching the warning already carried by `model_selection.py` and the methods page.

## [1.2.0] — 2026-07-16

### Added

- `measure_cross_two_point_function` — Landy–Szalay galaxy × star cross-correlation
  with reusable `DR` / `RR` counts, for stellar-contamination null tests.

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

## [0.9.5] — 2026-05-29

- Simulation pipeline, five bug fixes, updated results.

## [0.9.0] — 2026-05-11

- Initial public release.
