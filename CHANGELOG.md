# Changelog

## [Unreleased]

## [1.4.5] — 2026-10-15

The package code is that of 1.4.4; this release records a revision of the documentation.

### Documentation

- The README, this changelog and the documentation pages are rewritten in one style, and their numbers, captions and tables are checked against the code and the results they report.
- The scripts that write the generated pages (`analyze_detectability_law`, `coverage_report`, `generate_results_ls10_summary`, `make_survey_design_synthesis`, `run_ls10_analysis`) carry the revised text.
  Fig. 6 of the detectability-law page names the resolution of the progressive mocks.

## [1.4.4] — 2026-09-16

### Fixed

- `run_nuts` starts every chain inside the region where each pixel efficiency is at least `MIN_EFFICIENCY`.
  Its random start draws `b` at scale 0.05, and on templates with long tails (Euclid TR1, six rotated templates) that start lay below the floor, where the log-density is −∞: the chains did not move, every draw was divergent and the acceptance was 0.
  Each chain's `b` is now shrunk until every efficiency is at least 1/2.
  `run_nuts` warns when no proposal is accepted.

## [1.4.3] — 2026-09-16

### Fixed

- `generate_glass_delta_map` sets a negative Gaussian power to zero, with a warning, instead of stopping.
  A spectrum matched on a small footprint (Euclid TR1 at NSIDE 1024, f_sky = 0.008) can have no lognormal realisation at a few multipoles, and the default GLASS regularisation raises on a negative diagonal.
  The GLASS `"clip"` method returns NaN for a single shell, so it is not an alternative.
  A spectrum without negative Gaussian power draws the same field as before.

## [1.4.2] — 2026-09-16

The combined likelihood is unbounded above: at any pixel `a` can cancel the residual while the efficiency `1 + b·t` goes to zero, and `−ln|1 + b·t|` then diverges.
Its maximum is meaningful only away from that limit.
On the 400-realisation likelihood-ratio grid of 1.4.1, one null draw in six reached the pole, and 1–3 in 400 returned −∞.

### Fixed

- Every fit of a model carrying `b` (`lrt_from_maxima`, `refine_to_mle` and `run_nuts`) stays where each pixel's efficiency is at least `MIN_EFFICIENCY` = 1/20, the floor that matches the weight clip (`likelihood.MIN_EFFICIENCY`).
  On 64 LS10 null fields at NSIDE 32, ten combined maxima sit on the floor, and their λ falls from 76–153 to 22–85.
- `lrt_from_maxima` evaluates each point itself, and replaces a combined maximum that is not finite by its feasible start.
  It took its best point from the value the L-BFGS line search leaves in its state, which after a failed search can belong to another point.
- `refine_to_mle` searches a model carrying `b` with the same JAX L-BFGS, whose line search backtracks from the infeasible region; the SciPy search stopped at its starting point.

### Changed

- `lrt_from_maxima` fits in the whitened template basis (unit second moment) and maps the amplitudes back; λ is unchanged and the search is shorter on a collinear basis.
  It returns `min_efficiency` and `at_efficiency_floor`, and counts a maximum on the floor as converged.


## [1.4.1] — 2026-09-16

This release fixes four defects of 1.4.0, found in a referee reading of the pipeline paper: an estimator that undid its own bias correction, a skewed fit that did not leave its starting point, an unconstrained sampler on a likelihood with a pole, and a two-point correction that carried no uncertainty for four of the six methods.
The LS10 products and every corrected `w(θ)` are regenerated.

### Fixed

- `debias_params_matrix` projects onto the positive semi-definite cone only on request (`project_psd=True`).
  `â âᵀ` has rank one, so `â âᵀ − Cov[â]` has one positive eigenvalue and `n_sys − 1` negative ones, and the projection discarded the covariance subtraction in every direction but one, returning a matrix more biased than the one it debiased.
  Over 4000 draws at `n_sys = 5`, the projected estimator misses the truth by more than five times the unprojected one.
  The unbiased estimator of a squared amplitude matrix is not positive semi-definite; the corrected `w(θ)` must stay physical, and `correct_two_point_function` already checks it.
- `lrt_from_maxima(use_skewed=True)` starts both models at γ = ±1.5 and keeps the better maximum; `refine_to_mle` adds the same starts when it is given γ = 0.
  γ = 0 is a stationary point of the skew-normal log-likelihood (its information for the skewness vanishes there), so an optimiser started on it did not move, and λ matched the Gaussian value to twelve digits.
- `run_nuts` restricts a model carrying `b` to the region where every fitted pixel has `1 + b·t(p) > 0`, the region that holds `b = 0` (`positive_efficiency=True`); outside it the log-density is −∞.
  The flat prior admitted both the pole at a vanishing efficiency and the ridge along `|b| → ∞` with `σ → 0`, where the combined fits of a companion pipeline were ending up.
- `run_ls10_analysis.py` uses one amplitude covariance for every method, the sandwich covariance of the significance realisations, recorded as `amplitude_covariance`; it debiased only the MCMC methods and propagated no uncertainty.
  The corrected `w(θ)` carries the covariance it implies: `all_w_corr_err` in `*_wtheta_data.json`, and a band on the figure.


## [1.4.0] — 2026-09-15

Every calibrated statistic draws its null from a validated matched spectrum, detection significances are scored against that null with a family-wise p-value, each LS10 sample is analysed at the resolution its occupancy supports, and the two-point correction carries every cross-template term.
The mock nulls, the likelihood-ratio null and the template correlation matrix are one to two orders of magnitude faster, and the test suite measures line and branch coverage.

### Removed

- `get_mle_params`.
  It returned the per-parameter posterior median, which is neither a maximum-likelihood point nor the joint MAP.
  Use `posterior_median_params` to report a point estimate and `refine_to_mle` wherever a likelihood is compared.

### Added

- `calibrated_template_significance`: each template's least-squares amplitude over its scatter across uncontaminated realisations, per-template p-values, and a family-wise p-value for the largest significance, with leave-one-out.
  Over 2000 clean fields with 1000 realisations, `family_wise_p <= 0.05` occurs in 4.85 ± 0.48% and `<= 0.0027` in 0.35 ± 0.13%; the independent-pixel significance exceeds 3 on some template in 99.3%.
- `residual_template_correlation_test`: correlation of the corrected density with each template, χ² over the null variance, leave-one-out null χ² and p-value.
  For templates the correction fitted the correlation is zero by construction, so the test applies to templates or transforms the fit did not use.
- `choose_nside_by_occupancy`: the finest NSIDE whose mean footprint occupancy reaches a floor (default 25 galaxies per pixel), counting pixels by the rule of `compute_overdensity`.
- `template_correlation_matrix`: the full `(n_sys, n_sys, n_θ)` template correlation from the values each galaxy carries, with optional per-object weights.
- `standardise_on_footprint`: zero mean and unit rms over the fitted pixels.
- `lrt_from_maxima`: additive and combined maxima of many fields at once (least squares, and L-BFGS vmapped over fields), for the likelihood-ratio null.
  8 fields of 7 040 pixels with 11 templates take 0.15 s after compilation, against about 50 s for per-mock NUTS fits refined to their maxima; λ agrees to a relative 8e-7.
  Each L-BFGS run stops at convergence (largest gradient component below 1e-9) or after `n_iter` iterations.
- `draw_null_overdensity`, `generate_glass_null_overdensity`: null realisations drawn as Poisson counts per footprint pixel from the GLASS field, in 0.01 s per realisation against 0.97 s for a pixelised catalogue, with the same pixel variance and amplitude scatter.
- `run_nuts(dense_mass_matrix=True)`, the default: on the LS10 combined fit at NSIDE 32, it takes half the leapfrog steps per iteration and gives 2.5 times the minimum effective samples per second.
- `isd_template_significance(draw=)`, `"pixel"` by default.
- `scripts/run_ls10_analysis.py`: `--min-per-pixel`, `--significance-n-mocks`, `--significance-seed`, `--null-cl-file` (alias of `--lrt-null-cl-file`), `--allow-parametric-null`, `--null-draw`, `--lrt-null-method`, `--ct-max-galaxies`, `--ct-auto-only`.
- `scripts/jax_coverage.py`, `scripts/profile_by_library.py`, `scripts/coverage_report.py` and `docs/coverage.rst`: static JAX share, wall time by library, per-module line and branch coverage.
  `tests/test_jax_transformability.py` records which public numeric functions survive `jit`, `vmap` and `grad`.
- `scripts/plot_ls10_occupancy_products.py`; `scripts/generate_results_ls10_summary.py` writes the LS10 summary, recommendations and nine sample pages from the issued products.

### Changed

- GLASS nulls need an explicitly chosen spectrum.
  `generate_glass_fullsky_mock`, `generate_glass_delta_map`, `isd_template_significance` and the pre-selection of `run_decontamination` take `cl_amplitude=None`; with neither a matched spectrum nor an amplitude they warn and use the power law.
  `run_ls10_analysis.py` refuses to build a null without a matched spectrum unless `--allow-parametric-null` is given.
- `load_matched_cl` serves the validated spectrum nearest at or above the requested resolution, and the finest below only when none is finer.
- `template_correlation_matrix` builds cross terms from auto-correlations of summed fields, `ξ_ij = [ξ(t_i + t_j) − ξ_ii − ξ_jj]/2`: exact, and 1.7 times faster at 11 templates.
- `make_log_likelihood`, `refine_to_mle` and `run_nuts` compile once per configuration and take the data as arguments.
  Repeat calls on 20 000-pixel fields: likelihood-ratio test 0.102 → 0.003 s, `refine_to_mle` 0.266 → 0.019 s, `run_nuts` 3.39 → 1.71 s.
- The ISD, ranking and null-test statistics run on their JAX kernels only; the NumPy loops they are checked against live in the test suite.
- `debias_params`, `debias_params_matrix` and `standardise_on_footprint` accept JAX arrays and trace under `jit`, `vmap` and `grad`; NumPy inputs behave as before.
- `run_ls10_analysis.py` issues each sample at its occupancy-chosen resolution, corrects w(θ) with the full cross-template matrix, stores the calibrated significance, draws every null per pixel, and builds the likelihood-ratio null from batched maxima.
  The data statistic keeps, per model, the higher-likelihood maximum of the NUTS refinement and the batched optimiser.
- The test suite collects the 70 docstring examples and measures branch coverage; CI runs the fast suite on pull requests and everything on `main` and nightly, and fails below 97%.

### Fixed

- `refine_to_mle` raised for `model="multiplicative"`: its analytic start passed `b=None`.
- `run_decontamination` with ISD pre-selection and a p-value cut passed ISD a threshold for the templates before the cut.
- `generate_glass_delta_map` raised `NameError` on every call.
- Under `--skewed`, the LS10 likelihood-ratio test compared a Gaussian additive fit with a skew-normal combined fit.
- A `--template-dir` without FITS maps switched the run to synthetic templates without a warning; it now stops.
- The `cl_amplitude=5e-4` default of `isd_template_significance` silenced the unmatched-spectrum warning.
- `compute_sys_weights.py` recorded `WEIGHTCON = library` for weights it reconstructs in the linear form; it records `linear-from-a_hat`.
- `lrt_from_maxima` returned the start (b = 0, λ = 0) when an L-BFGS step crossed a pixel where `1 + b·t = 0` and the run ran off along the |b| → ∞ ridge, on one of 100 NSIDE-64 synthetic mocks.
  The combined maximum is sought where every pixel's efficiency is positive; each run keeps its best finite point, stops once the value or gradient is not finite, and is restarted from that point up to three times.
- With pre-selection, `run_decontamination` reused a linear, equal-width ISD null as the stopping threshold of an ISD fit of another degree on equal-occupancy bins; the pre-selection null now uses the degree, bin count and binning of the ISD method it serves.
  `run_ls10_analysis.py` calibrates the ISD threshold with its own cubic, equal-occupancy null and no longer reuses the pre-selection null.
- The validation and characterisation scripts (`run_validation.py`, `run_systematic_tests.py`, `run_mock_analysis*.py`, `run_simulation_tests.py`, `run_snr_preselection_demo.py`) run MCMC-add and MCMC-comb through `run_decontamination` (analytic posterior and NUTS; `--sampler`, `--nuts-warmup`, `--nuts-samples`), calibrate every ISD threshold on uncontaminated realisations of their own generator, and pass an explicit GLASS spectrum.
  `run_simulation_tests.py` reads the Uchuu catalogue from `.fits.gz` and stops when it is missing unless `--glass-only` is given.
- `run_mock_analysis.py`, `run_mock_analysis_progressive.py` and `run_mock_analysis_real_templates.py` evaluated the likelihood-ratio statistic at posterior medians, where it can be negative; they take it between the maxima from `lrt_from_maxima`.

## [1.3.0] — 2026-09-14

This release corrects three algorithms and re-issues the LS10 products on a footprint-standardised template basis.
`iterative_systematics_decontamination` implements published ISD instead of a multivariate polynomial expansion.
The likelihood ratio is evaluated at two likelihood maxima, so it is non-negative and its magnitude carries meaning.
The template basis is standardised over the analysis footprint, and the template correlations are measured from the galaxies, so the two-point correction acts over the whole measured range instead of the nine widest bins of thirty.

The 27 re-issued products at NSIDE 32, 64 and 128 carry `WEIGHTVER = 3`.
Their amplitudes are not comparable with those of a version-2 product, whose basis is normalised over each map's own valid region instead of the fitted pixels.

### Changed (breaking)

- `iterative_systematics_decontamination`, exposed as the methods `ISD-1` and `ISD-3`, implements ISD as described in the paper of Rodríguez-Monroy et al., which its docstring already cited, and in its DES Y1/Y3/Y6 predecessors.

  Up to and including v1.2 it implemented a different algorithm: it expanded the template set into every monomial `t_i·t_j·t_k` up to total degree `d`, cross-products between different templates included (`binom(n_s+d, d) − 1` columns, 55 for `n_s = 5` and `d = 3`), fitted them all simultaneously by unregularised iteratively reweighted least squares, and built the weight from the first `n_s` linear coefficients alone.
  The fifty discarded columns are strongly collinear with the five retained ones, so fitting and dropping them inflated the variance of the coefficients that set the weight.
  On LS10 this gave `rms|â| ≈ 7.6`, some 35× the OLS solution on the same data, and a weight map saturated across the whole footprint.

  It now bins the footprint, for each template independently, into 10 equal-occupancy bins of that template's value, fits the binned mean density with a polynomial of degree `d` in that one template's value, and scores it against a mock-calibrated `Δχ²/Δχ²₆₈`.
  It applies the inverse of the single most significant fit as a weight, re-measures, and repeats until every template falls below `threshold` (default 2, the DES Y6 value).
  The design matrix of every fit is `n_bins × (d+1)`, whatever the number of templates.

  `ISD-1`, `ISD-3` and `poly_order` keep their names, and `poly_order` has its meaning in the literature.
  The v1.2 routine remains as `polynomial_ols_decontamination`, documented as not being ISD, so the v1.2 benchmark and timings stay reproducible.

  `iterative_systematics_decontamination` returns an `ISDResult` dataclass (`weights`, `a_hat`, `steps`, `significance`, `n_steps`, `stopped_on`, `calibrated`, `n_floored`) instead of the tuple `(weights, alpha_hat_all, n_iterations)`; `max_iter`, `tol`, `lambda_poly` and `backend` are replaced by `n_bins`, `binning`, `threshold`, `chi2_68`, `max_steps`, `max_reuse`, `fracdet`, `w_max` and `bad_pixel_frac`.
  In `run_decontamination`, `isd_max_iter` and `isd_lambda_poly` are replaced by `isd_n_bins`, `isd_binning`, `isd_threshold`, `isd_chi2_68`, `isd_max_steps` and `isd_fracdet`; the result dict gains `isd_steps`, `isd_significance`, `isd_stopped_on` and `isd_calibrated`, and loses `isd_outlier_mask` and `isd_masked_fraction` (the two-pass outlier scheme is unnecessary now that the iteration stops on insignificance).

  A binned fit is never extrapolated: `F̂` is evaluated at `clip(t, t_lo, t_hi)`, the outermost bin centres that constrain it.
  `GALDEPTH_Z` reaches +26 standardised units while its outermost bin centre sits near +2, so an unclipped cubic is evaluated three orders of magnitude beyond its support.
  On real LS10 templates without clipping, the significances rise along the iteration (`S = 9.0 → 14.8 → 36.2` on one template), as each over-corrected step creates a larger trend for the next, and 339 of 22 000 pixels hit the weight floor; with clipping the same run floors none and stops on the threshold.
  `isd_marginal_fit` returns the supported range alongside the coefficients.

  The reported amplitude `a_hat[i]` is the projection `<F̂_i(clip(t_i)) t_i> / <t_i²>`, summed over the steps that selected template *i*, not a polynomial coefficient.
  The degree-1 coefficient would be correct for `poly_order=1` but not for `poly_order=3`: equal-occupancy bin centres of a skewed template span a narrow range, so the Vandermonde matrix is poorly conditioned and the coefficients are large even when the curve is small.
  On a GLASS null with no injected contamination, one accepted step returned `c = (0.026, 0.681, 3.343, 4.365)`, `rms|â| = 0.44` for a field whose true amplitude is zero; the projection gives `6e-4` on the same data.

  Without `chi2_68` the threshold is in raw `Δχ²` units, where the value 2 has no calibrated meaning, so the function warns.
  When Stage-1 pre-selection has already run with `preselect_method="isd"`, `run_decontamination` takes `Δχ²₆₈` from the GLASS null it generated instead of drawing a second one.

### Added

- `apply_nonlinear_contamination`, `TemplateResponse`, `evaluate_response`, `RESPONSE_KINDS`: injection of a template response that is not linear.

  Every simulation this package ships injects `apply_contamination`, `δ_g(1 + Σ b_i t_i) + Σ a_i t_i`, which is linear in every template, so a linear marginal fit suffices and no campaign run to date separates `ISD-1` from `ISD-3`.
  Across all 900 NSIDE-64 cells of campaign `20260907c`, the two recover the same amplitude to within 1%, and `ISD-3` differs only by a worse false-positive rate.

  The new path injects `1 + δ_obs = (1 + δ_true)·Π_i (1 + F_i(t_i))`, the per-template selection efficiency that the ISD weight `Π_j 1/(1 + F̂_j)` inverts; it is not Eq. 13.
  It has six response shapes: `linear`, `quadratic` and `cubic` (inside a cubic marginal basis), and `tanh`, `threshold` and `exp` (outside it).
  Every shape is rescaled to the same `rms(F)`, so that shapes are compared at equal amplitude.

  We compared the two methods per seed (the same mock and the same injection) over a 50-seed campaign at NSIDE 32 on the real LS10 templates, two of five contaminated, scoring by the fractional L2 error of the recovered `w(θ)`:

  | injected `F(t)` | low | medium | high | in a cubic basis? |
  |---|---|---|---|---|
  | `t` (control) | 1.07 / 42% | 0.83 / 76% | 0.82 / 92% | exactly |
  | `t + κt²` | 0.89 / 78% | 0.68 / 98% | 0.53 / 100% | exactly |
  | `t + κt³` | 0.96 / 64% | 0.85 / 96% | 0.73 / 100% | exactly |
  | `tanh(αt)/α` | 0.93 / 56% | 0.53 / 92% | **0.22** / 100% | no |
  | `t·1[t>t₀]` | 0.69 / 84% | 0.39 / 100% | **0.36** / 100% | no |
  | `(e^{αt}−1)/α` | 1.00 / 46% | 0.89 / 84% | 0.86 / 84% | no |

  Each cell gives the median ratio of the `ISD-3` error to the `ISD-1` error on the same seed, then the fraction of seeds `ISD-3` wins.
  Every medium- and high-amplitude cell is significant at p < 0.02 (Wilcoxon signed-rank); none of the low-amplitude cells below 60% is.

  The advantage of `ISD-3` grows with amplitude, is largest on the shapes a cubic cannot represent (`tanh`, `threshold`), and is present on the linear control too, because binning a lognormal density against a skewed survey-property map curves the marginal relation before any contamination is applied.

  The exponential is the exception.
  Rescaled to fixed `rms(F)` on a template as skewed as `GALDEPTH_Z` (skew +5.9, reaching +19σ), an exponential response puts `|F| > 0.1` in 0.03% of pixels, all inside the outermost quantile bin, where a binned fit of any order has one degree of freedom.
  Split by seed, the ratio is 1.03 where a contaminated template has skew > 3, and 0.32 where neither does.
  So the binding constraint on ISD is `n_bins`, not `poly_order`.

- `characterisation/isd_hyperparameter_sweep.py` (benchmark repository): the first scan of the ISD parameters, 50 settings × 18 configurations × 20 seeds.
  Two results:

  * At `poly_order = 3`, `n_bins = 20` beats the shipped `n_bins = 10` on residual (0.028 against 0.032), selection precision (0.925 against 0.870) and recall (0.835 against 0.808).
    The default is not changed yet: 20 bins has not been tested on the sparse LS10 samples at NSIDE >= 128, where the `N_beta >= 2` bin-admission rule could start rejecting bins.
  * The usable polynomial order is set by the bin count, not chosen independently.
    At 5 bins the best order is 2, and a cubic is already worse than a linear fit at 20 bins; at 40 bins a quartic wins.
    At 10 bins, `d=5` (0.200) is worse than `d=2` (0.052).
    The DES Y6 choice of `d=3` is the highest order ten bins support.

  The scan also gives the following.
  Equal-occupancy binning beats equal-width by a factor 2–3 at `d=1`.
  `threshold` trades precision against recall monotonically (T=1 → 0.717/0.899, T=5 → 0.991/0.700).
  `max_reuse=1` triples the residual, and values above 3 change nothing.
  `w_max` and `bad_pixel_frac` changed no metric at any setting.
  Dropping the calibration gives the lowest residual of the scan (0.009) and the worst precision (0.516): it removes templates that are not contaminated.

- `simulation.make_response_grid`, and `--responses`, `--response-kinds` and `--n-contaminated` in `run_simulation_tests.py`.
  `--n-contaminated` makes greedy selection measurable: the existing grid contaminates all five templates at equal amplitude, so a selection rule has no true negative to get wrong.

- `run_simulation_tests.py --isd-null-cl-file`: the ISD calibration null can be drawn from a matched spectrum.
  Without it the null is the parametric power law at `cl_amplitude=5e-4`, which under-clusters LS10 by ~25× and leaves the calibrated threshold anticonservative.

- `run_simulation_tests.py --isd-kwargs`, and the `isd_poly_order`, `isd_max_reuse`, `isd_w_max` and `isd_bad_pixel_frac` arguments of `run_decontamination`.
  `poly_order` was pinned by the method name and the other three were unreachable from any driver, so no hyper-parameter sweep was possible.
  `ISD-<d>` is accepted for any degree.

- `standardise_on_footprint`: standardise a template basis over the pixels it is about to be fitted on.

  `load_real_template` and `load_templates_from_dir` normalise each survey-property map over that map's own valid region, which is larger than any one sample's footprint.
  Restricted to the footprint, the basis is no longer standardised: the eleven LS10 maps at NSIDE 64 have per-template rms spanning 0.905 to 5.77 and a relative mean of 0.729, so the covariance eigenvalues sum to 44.4 instead of `n_sys = 11`, with a leading eigenvalue of 37.5.
  Calling this function after masking gives exactly 11.00 with a leading eigenvalue of 5.39, and drops the basis condition number from 1443 to 208.

  It cannot run at load time, since the loader does not know which pixels the fit will use.
  Both production scripts call it after `assign_template_values`, record the means and rms they divided out in `params.json` under `template_basis`, and stamp `TPLBASIS` into the FITS header.
  `WEIGHTVER` goes to 3 for a footprint-standardised product.
  `--no-footprint-standardise` reproduces the old basis and keeps `WEIGHTVER = 2`.

  Synthetic bases change little (rms 0.971 to 1.030 under a Galactic cut at both resolutions, eigenvalues summing to 4.93–5.05 against 5), so the simulation-based results are unchanged; only the real-template LS10 results move.

- `data/glass_match/`: 25 validated per-setup GLASS spectra from the cluster campaign, for all nine LS10 samples at NSIDE 32 and 64 and seven at NSIDE 128.
  Each is fitted band by band and validated on 8 or 16 seeds held out of the fit; 24 of 25 pass, with large-scale power ratios in 0.912–1.020 against a 10% tolerance.
  With them `sigma_hat_mock` spans 0.27–1.22 per sample, where the default power law gave 0.078 against a data value of 0.387.
  `--lrt-null-cl-file` should point at them; without them every `load_matched_cl` call returned `None` and every run fell back to the power law without a warning.

- `sys_mapping.diagnostics.isd_marginal_fit`: the per-template binned polynomial fit ISD is built from, returning both `Δχ²` and the fitted coefficients (ascending powers).
  `snr_template_ranking(method="isd")` returns only the former and is unchanged.

- `sys_mapping.diagnostics.vet_templates_against_tracer`: weighted Spearman rank correlation of each template against an external tracer of true structure (CMB lensing κ, Compton-*y*, weak-lensing convergence), with delete-one-patch jackknife errors.
  A template that correlates with real structure lets the regression fit the signal, and no care in the fit reveals it: the contamination is detected at high significance and the clustering comes out low, on mocks as well as on data.
  Eggert & Leistedt (2023) show this for templates built from Legacy Survey image stacks, the data this package is applied to (Weaverdyck et al. 2026, Sec. III A).

- `sys_mapping.correction.estimate_overcorrection_bias` and `debias_two_point_function`: run the weighting on contamination-free mocks, where whatever it changes is over-correction, not systematics, and subtract that change from the data vector.
  Both are estimator-agnostic: the caller supplies the `w(θ)` measurement, so the debias term uses the same binning and mask as the data (Weaverdyck et al. 2026, Eqs. 21–23).

- `sys_mapping.covariance.method_marginalised_covariance`: adds `Δᵢ Δⱼ` to the covariance, with `Δ` the difference between two methods' `w(θ)`.
  This package offers six methods and they disagree; reporting one of them with its own error bar prices that disagreement at zero.
  It can be applied block-diagonally, so each tomographic bin can prefer a different method (Weaverdyck et al. 2026, Eq. 24).

- `sys_mapping.bootstrap.assign_spatial_patches`: the patch labelling used by the bootstrap and the jackknife, made public so the same partition can serve as cross-validation groups and as jackknife patches.

- `sys_mapping.maps.inverse_variance_pixel_weights`: the per-pixel weight `A_k²/(N_k + 2)`, coverage area squared over Poisson variance, with the regulariser keeping empty pixels finite (Weaverdyck et al. 2026, Eq. 8).

- Spatially compact cross-validation for ElasticNet: `patch_ids=` and `pixel_weights=` in `elasticnet_contamination_fit`.
  By default `ElasticNetCV` splits into contiguous index chunks, which mix spatially; on a clustered field the held-out pixels then correlate with the training pixels, the prediction error is under-estimated, and the cross-validation picks too weak a penalty.
  With `patch_ids` the folds are whole patches (`GroupKFold`), spatially disjoint.
  `cv_info["cv_spatial"]` records which was used.
  DES Y6 uses ~200 patches of about 5° diameter.

- `--isd-n-mocks` in `scripts/run_simulation_tests.py`, and `ISD-3` among its `--methods` choices; it was absent, so `data/simulations/*/results_summary.json` has no ISD-3 column.
  The ISD `Δχ²₆₈` is calibrated once per mock source, since the null depends on the footprint, resolution and surface density but not on the injected contamination, and is reused across all nine configurations.
  `run_wtheta_recovery` takes `method_kwargs=` to carry it.

- The pipeline document, a standalone LaTeX document (now in the private `sys_mapping_paper` repository; see *Repository split* below) stating every equation the package implements with a `file:line` reference to the code that evaluates it, the implementation path, timing tables, the LS10 DR10 results, and a verification section recording 15 findings where the code, the published equations and the documentation disagree.
  Its `make_tables.py` generates the tables from files on disk.
- `benchmark_pipeline.py` (now in `sys_mapping_benchmark`): a reproducible timing harness covering the per-function API, the HEALPix utilities, all four Stage-1 ranking statistics and all six Stage-2 methods.
  It writes `benchmarks.csv` and a `machine.json` provenance record (CPU, cores, RAM, library versions, commit, date, load average), replacing the undated benchmark table in the developer notes.
  The results are rendered in `docs/results_benchmark.rst`.
- `docs/roadmap.rst`: prioritised next steps.
- `docs/api/nuts.rst`, `docs/api/plotting.rst`: API pages for two public, exported modules that had none.
- `bash/ls10_mocklrt_ns32_gaps.sh`: fills, concurrently, the NSIDE-32 mock-calibrated LRT cells still holding `calibration = "failed"`.

### Fixed

- `run_ls10_analysis.py` measures the template two-point functions from the galaxies: each galaxy carries the rotated-template value of its pixel, and the KK correlation runs on the galaxy positions, over the same bins as `w(theta)`.
  It ran TreeCorr KK on the rotated templates at the HEALPix pixel centres, where no pair of distinct pixels is closer than the pixel scale, so `xi_i` was exactly zero below it while `w(theta)` is measured from the catalogue down to 0.5'.
  The correction was then zero over 21 of 30 bins at NSIDE 64 (below 49') and 24 of 30 at NSIDE 32 (below 94'), the range carrying most of the signal.
  A template is constant within a pixel, so its `xi_i` at sub-pixel separations is its variance.
  On a synthetic footprint the galaxy and pixel measurements agree to within 2% above the pixel scale (0.752 against 0.737 at 115', 0.195 against 0.198 at 270'), and the galaxy measurement returns 0.93 to 1.01 of the template variance below it.
  It costs 29 s per template at 3M galaxies.
  `--ct-from-pixels` restores the pixel-centre measurement.

- `correct_two_point_function` warns where the correction overshoots the signal.
  `Σ_i ã_i² ξ_i(θ)` is a sum of squares against auto-correlations, so nothing bounds it by `w_obs(θ)`; where the galaxy signal has decayed but the survey-property maps are still coherent it exceeds `w_obs`, and the corrected correlation function comes back negative without an error.
  On the shipped LS10 grid this happens for `ISD-3` in all 18 cells, reaching `w_corr/w_obs = -39` at NSIDE 64 and `-232` at NSIDE 32, while the other five methods stay between 0.60 and 0.88.
  Those bins are not usable; the caller decides whether to drop them, restrict the fitted range, or refuse the cell.

- `compute_covariance_matrix` warns on a basis that is not standardised over the pixels supplied, such as the LS10 maps normalised by `load_templates_from_dir` (numbers under `standardise_on_footprint` above).
  Amplitudes fitted against that basis are not in units of one template standard deviation, and the same scaling makes the two-point correction overshoot.

- The zero-mean warning tolerance is `max(1e-3, 5/sqrt(n_pix))`.
  The sample mean of `n_pix` unit-variance values has standard error `1/sqrt(n_pix)`, so the fixed relative threshold of `1e-3` fired on any finite draw of a centred basis.

- `run_ls10_analysis.py` has a `--skewed` flag, off by default, as `compute_sys_weights.py` does.
  One script defaulted the skew-normal likelihood on and the other had no flag, so a column named `WEIGHT_SYS` meant a different model depending on which script wrote it.
  `run_decontamination` also returns `gamma_hat`, which a caller refining the fit to an MLE needs to pack a starting point.

- `plot_ls10_wtheta_corrected.py` spans the measured range and reads the binning in the title from the data.
  The ratio panel was gated on `w_corr > 0` and the x-axis fixed at 70', which together hid every bin where the correction overshoots, the only bins where it acts.

- The paper Makefile rebuilds on a figure or archive edit.
  `figures/*.tex` and `findings_archive.tex` were not prerequisites of the PDF, so editing a figure source left a stale build that still passed `make check`.

- `run_wtheta_recovery` keeps the per-step coefficients, `n_steps`, `stopped_on`, `significance` and `calibrated` that `run_decontamination` returns, beside `a_hat`, `b_hat` and `elapsed_s`.
  On a non-linear response the curvature lives entirely in those coefficients, so the results files could not have separated the ISD methods even with the right injection.
  `n_floored`, computed by `ISDResult`, is now returned by `run_decontamination`, which dropped it.

- `null_test_cross_correlations` warns that `max_i |r(w, t_i)|` is not a scalar goodness-of-fit.
  For an additive correction the statistic depends on the *support* of `â`, not its size: a single-template correction gives `|r| → 1` at any amplitude (1.0000 at `â = 1e-6`, 0.9886 at `â = 1e-1`), so a sparser and more accurate correction scores worse, and a method that fits nothing scores zero.

- `load_matched_cl` prefers the finest validated candidate and warns when it falls through, while still refusing outright when nothing passed.
  The exact-NSIDE file won whenever it existed, even when its `validation.passed` was false, so a sample whose fit failed at the requested resolution raised instead of using a spectrum that had passed at another.
  The spectrum belongs to the sample and its footprint, and the resolution limits what can be checked, not what can be used.
  LS10 `logM ≥ 11.5` fails at NSIDE 64 and passes at 128, so the NSIDE-64 LRT for that sample would have crashed.

- The two JAX ISD kernels use the centred (two-pass) per-bin variance, require at least two pixels in a bin (one pixel carries no variance information), floor the variance relative to the field's own scatter instead of at an absolute 10⁻²⁰, and return 0 when fewer than two bins survive.
  They reproduce the NumPy reference to 4 × 10⁻¹⁵.
  They estimated the variance as `E[g²] − E[g]²`, which for a single-pixel bin is exactly zero in eager arithmetic; under `jit`, XLA contracts the expression into an FMA whose rounding returns ≈1.5 × 10⁻²⁰, just above the `> 1e-20` guard that rejected degenerate bins.
  The bin was then admitted with an inverse variance of ≈7 × 10¹⁹, which swamped the Δχ² sum: on the unit-test universe a noise template scored 6.0 × 10¹⁷ against 389 for a contaminated one, so `snr_preselect(method="isd")` ranked it first.
  The fault lay in compiler fusion, so it appeared only in the compiled path (the NumPy fallback in the same function was correct) and could move with the XLA version.
  The published LS10 weights are unaffected: all 36 `params.json` record no pre-selection.
  The Stage-1 demo figures in `docs/results_snr_preselection.rst` regenerate byte-identical, because that demo uses synthetic templates on a well-populated NSIDE-32 footprint where no bin holds fewer than two pixels.
  ISD ranking on the real LS10 templates is affected: their skewed pixel distributions leave near-empty tails in equal-width bins (25 bins with fewer than two pixels across 11 templates at NSIDE 64).
  `method="data"`, `"template"` and `"peak"`, and every Stage-2 method, are unaffected.

- `isd_template_significance` and `build_lrt_null` pass `cl_amplitude` through to the GLASS mock null, and `run_ls10_analysis.py` exposes `--lrt-null-cl-amplitude` (recorded in `params.json` as `null_cl_amplitude`).
  Both took the 5 × 10⁻⁴ default with no way to override it, so the null was drawn from mocks with σ_clus ≈ 0.089 against ≈ 0.397 for LS10, roughly 25× too little clustering variance, and every p-value was anticonservative.

- `run_ls10_analysis.py --preselect` scatters the selected coefficients back into the full template basis with the new `expand_preselected_params` helper, tested in `tests/test_ls10_script.py`.
  With pre-selection, `a_hat` and `b_hat` have length `len(selected) < n_sys`, and the weight writer discarded any vector whose shape did not match the full basis, so every method column, `WEIGHT_SYS` included, was exactly 1.0 (no correction) without a warning.
  The zeroing path remains only for a genuine shape mismatch, and warns.
  Results already on disk are unaffected: the LS10 runs did not use `--preselect`.

### Repository split

The paper and the benchmarks moved to their own repositories, leaving the package and the pipeline here.
Their results are still documented here and render from committed snapshots, with no external checkout required.

| Moved to | What |
|---|---|
| [`sys_mapping_benchmark`](https://github.com/JohanComparat/sys_mapping_benchmark) (public) | `scripts/benchmark_pipeline.py`, `tests/test_timing.py`, `tests/test_benchmarks.py`, and the four characterisation scripts |
| `sys_mapping_paper` (private) | the LaTeX document, its sections, figures and `make_tables.py` |

- CI no longer runs the 259 timing and benchmark cases on every push.
  `tests/test_benchmarks.py` had zero assertions, and `tests/test_timing.py` contributed 240 parametrised cases whose 10 assertions are time budgets, better placed beside the harness.
- `docs/results_benchmark.rst` (new), generated by `docs/generate_benchmark_page.py` from `docs/_static/benchmark/`.
- Both new repositories accept `SYS_MAPPING_ROOT` to read the pipeline outputs of this package, and fall back to their own committed snapshots when it is absent, so each builds standalone.

### Documentation

- `docs/methods.rst` states the implemented prior: flat and improper on all amplitudes, with only a hard `σ > 1e-6` floor.
  It claimed `a_i, b_i ~ N(0,1)` amplitude priors and a half-normal prior on σ.
- The README states the three weight formulas in circulation and which one the FITS columns use, and notes that the two production scripts fit different likelihoods (skew-normal against Gaussian).
- The README states the PCA covariance propagation in its own convention for `V` and `Vᵀ`, which it had transposed, and propagates the full covariance, as the code does, instead of the diagonal only.
- The skew-normal formula states that the location shift ξ enters the Gaussian quadratic form as well as the Φ argument.
  A citation to commit `10c01d0`, which does not exist in this repository, is removed.
- Both places that describe the SNR ranking statistics list all four (`template`, `data`, `peak`, `isd`); each listed three.
- The dropped cross-terms are documented: the `w(θ)` correction and the amplitude-bias estimator keep only auto-correlations, and PCA diagonalises the template covariance at zero lag, not at `θ > 0`.
  The harmonic path subtracts a term linear in α, while the configuration-space path subtracts the debiased square.
- The simulation-test page follows the committed summary table for the same runs: 37 of 45 Uchuu method-cells have an improvement factor below 1, and MCMC-comb is worst almost everywhere.
  It claimed that all methods reduce the `w(θ)` bias substantially and that MCMC-comb is the best correction.
- The README carries the correlated-field caveat on the likelihood ratio test, matching the warning in `model_selection.py` and on the methods page.

## [1.2.0] — 2026-07-16

### Added

- `measure_cross_two_point_function`: Landy–Szalay galaxy × star cross-correlation with reusable `DR` / `RR` counts, for stellar-contamination null tests.

## [1.1.0] — 2026-07-08

This release accelerates the inference core with JAX and replaces the gradient-free emcee sampler on the critical path.
The MCMC stage dominated the wall time per sample (30–90 min/sample on CPU, against seconds for OLS, ISD and the 2PCF).

### Added

- `sys_mapping.inference.run_additive_analytic`, the analytic conjugate posterior of the additive model.
  The additive model is linear-Gaussian, so its posterior is Normal-Inverse-Gamma in closed form.
  It draws the exact posterior (no Monte-Carlo autocorrelation) in milliseconds, replacing the ~80 s emcee `MCMC-add` run, and matches an emcee chain's posterior mean and covariance to Monte-Carlo error.

- `sys_mapping.nuts.run_nuts`, gradient-based NUTS with BlackJAX for the non-linear `combined` and skew-normal models.
  NUTS explores the posterior with the gradients of the JAX log-likelihood, at a higher effective sample size per step than the emcee stretch move, and runs the whole chain (window adaptation and sampling) under `jax.lax.scan`, without the Python per-step loop and the per-walker host↔device synchronisations.
  `σ > 0` is enforced by an `exp` reparametrisation, without a hard `σ_min` discontinuity.
  Chains run in parallel through `jax.vmap`, and their number follows the backend (4 on CPU, 8 on GPU).
  It reports `rhat`, `ess` and `num_divergences`.

- A `sampler` argument of `run_decontamination`: `"auto"` (default; analytic for `MCMC-add`, NUTS for `MCMC-comb` and the skew-normal model), `"analytic"`, `"nuts"`, or `"emcee"` (the gradient-free sampler, kept as the validation baseline).
  The result dict carries `sampler_backend`, `rhat`, `ess` and `num_divergences`.

- `scripts/run_ls10_analysis.py` and `scripts/compute_sys_weights.py` take `--sampler {auto,analytic,nuts,emcee}`, `--n-chains`, `--nuts-warmup` and `--nuts-samples`.
  Convergence diagnostics are written to the summary YAML/JSON.

- The ISD Δχ² significance (`sys_mapping.diagnostics`) runs in JAX for `poly_order > 1` and `fracdet` weighting: a vmapped `_one_isd_poly` kernel replaces the NumPy fallback, so cubic (ISD-3-style) and coverage-weighted pre-selection run in JAX, reproducing the NumPy fallback to `~1e-13`.

- `null_test_cross_correlations` computes the correlations and permutation p-values with a single `jax.vmap` over resamples; the correlations reproduce the NumPy path exactly.

- `iterative_systematics_decontamination(backend="jax")`, an opt-in `lax.while_loop` port of the fixed-point iteration, identical to `"numpy"` to `~1e-14`.
  The default stays `"numpy"`, because the BLAS path is faster on CPU; `"jax"` is for GPU and device-resident pipelines.

- `n_jobs` in `isd_template_significance` (`preselect_n_jobs` in `run_decontamination`, `--preselect-n-jobs` in the scripts) runs the embarrassingly parallel GLASS and healpy mock loop over processes with `joblib`, bit-identical to the serial loop (same per-mock seeds).

### Dependencies

- Added `blackjax >= 1.2` and `joblib`; `jax` pinned to `>= 0.9`.

### Notes

- Sampler results are statistically equivalent to the emcee baseline but not bit-identical (the analytic additive path is more accurate).
  The default switches to the new samplers through `sampler="auto"`; validate with `--sampler emcee` before comparing against v1.0.0 outputs.

---

## [1.0.0] — 2026-06-11

### Added

- Two-stage pre-selection pipeline (`sys_mapping.diagnostics`, `sys_mapping.model_selection`):
  - `snr_template_ranking`: three SNR estimators, Pearson `|r|` (`"data"`), per-template OLS `|t|`-stat (`"template"`) and ISD Δχ² (`"isd"`).
  - `isd_template_significance`: GLASS systematic-free mocks on the same footprint, giving mock-based p-values for each template.
  - `snr_preselect`: a wrapper that returns a ranked, filtered `SnrPreselectionResult`.
  - `run_decontamination` accepts `preselect=True` (and matching keyword arguments) to run Stage 1 before any of the six Stage-2 methods.

- JAX acceleration of all three SNR ranking methods:
  - `"data"`: batched Pearson correlation with `jax.jit` and a matrix multiply.
  - `"template"`: per-template OLS with `jax.vmap`.
  - `"isd"`: vmap over templates, fixed-size bins and analytic linear regression, replacing the Python loop and `np.polyfit` for `poly_order=1`.
  - NumPy fallbacks remain for `poly_order > 1` and `fracdet` weighting.

- Footprint-aware GLASS mocks in `isd_template_significance`:
  - the parameter `n_total_footprint`, the number of galaxies in the survey footprint, from which the function computes the full-sky count `n_total = n_total_footprint × N_full / N_good`, so that after trimming to `good_pixels` the mock surface density matches the data.

- `scripts/run_ls10_analysis.py` takes `--preselect`, `--preselect-method`, `--preselect-n-top`, `--preselect-p-threshold` and `--preselect-n-mocks`.

### Fixed

- ISD Δχ² overflow (`SNR ≈ 10⁶¹`): bins where all pixels have identical overdensity (e.g. empty HEALPix cells with δg = −1) are skipped instead of flooring σ to 1e-30, which overflowed 1/σ².

### Documentation

- New results page `docs/results_snr_preselection.rst`: a 15 M galaxy, 20-template validation at NSIDE=32 with three contamination levels, timing figures, and the mock-convergence rule of thumb N_mocks ≥ max(20, ⌈5/α⌉).
- `docs/methods.rst`: new *Template pre-selection (Stage 1)* section with a two-stage workflow code snippet.
- `README.md`: updated pipeline diagram, pre-selection table, quick-start snippet and CLI example.

### Changed (breaking)

- The `"template"` method of `snr_template_ranking` performs per-template OLS (one regressor at a time) instead of the joint multi-regressor OLS.
  Rankings are unchanged for uncorrelated templates; for correlated templates the per-template t-statistic is more appropriate for pre-selection.

---

## [0.9.5] — 2026-05-29

- Simulation pipeline, five bug fixes, updated results.

## [0.9.0] — 2026-05-11

- Initial public release.
