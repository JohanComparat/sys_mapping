Changelog
=========

0.9.5 (2026-05-29)
------------------

**New modules — simulation pipeline and GLASS mocks**

* ``sys_mapping.simulation`` — end-to-end validation framework for systematic
  contamination recovery.  Provides
  :func:`~sys_mapping.simulation.inject_systematics` (per-galaxy contamination
  weights from the forward model, Eq. 11-13 of Berlfein et al. 2024),
  :class:`~sys_mapping.simulation.ContaminationConfig` (9-point grid of
  level × scenario), FITS I/O helpers, and
  :func:`~sys_mapping.simulation.run_wtheta_recovery` (runs all decontamination
  methods and compares recovered :math:`w(\theta)` to truth).

* ``sys_mapping.glass_mocks`` — full-sky lognormal mock generation via GLASS
  (Tessore et al. 2023):
  :func:`~sys_mapping.glass_mocks.generate_glass_fullsky_mock` and
  :func:`~sys_mapping.glass_mocks.measure_nz`.  Advantage over the existing
  ``mocks.make_mock_catalog``: proper full-sky catalogs with physically realistic
  clustering, no galactic-cut artefacts.

**Bug fixes (five pipeline bugs, all in regression / simulation / scripts)**

* **Bug A — w(θ) correction applied to wrong sample.**
  ``run_wtheta_recovery`` was multiplying the correction weight by the *clean*
  galaxy catalog instead of the contaminated one.  The effective weight is now
  correctly ``weight_cont × weight_recovered``.  Before this fix, the recovered
  :math:`w(\theta)` was *anti-contaminated* — biased in the opposite direction
  from truth — producing residuals of order 10 000× the contamination bias.

* **Bug B — bias metric blow-up at zero-crossings.**
  The fractional-bias metric used a per-bin denominator
  ``|w_true(θ)| + 1e-6``.  For shot-noise-dominated mocks (e.g. GLASS
  full-sky) where :math:`w_{\rm true} \sim 10^{-4}`, this produced
  artifacts of order :math:`10^6`.  Fixed to scalar
  ``max_θ|w_true(θ)|`` throughout
  ``run_simulation_tests.py``, ``plot_simulation_tests.py``, and
  ``run_paper_validation.py``.

* **Bug C — unclipped OLS / MCMC correction weights.**
  ``_compute_weights`` returned values up to :math:`10^6` for pixels where
  :math:`1 + \hat{a}\cdot t \to 0`.  OLS and MCMC methods now clip to
  :math:`[1/20,\, 20]`, matching the existing ISD and ElasticNet behaviour.
  Same fix applied in ``method_comparison()``,
  ``run_mock_analysis_real_templates.py``, and
  ``run_mock_analysis_diagnostic.py``.

* **Bug D — MCMC-comb correction weight was a flawed approximation.**
  The combined-model weight used :math:`1/(1+(a+b)\cdot t)`, which
  overcorrects by ~2× when the inferred :math:`\hat{b}\neq 0` in a purely
  additive scenario (where :math:`b_{\rm true}=0`).  Replaced with the exact
  pixel-level inverse of the contamination model:
  :math:`w(p) = (1+\hat\delta_g^{\rm clean}(p))/(1+\delta_g^{\rm obs}(p))`
  computed via :func:`~sys_mapping.contamination.invert_contamination`.  This
  cancels ``weight_cont`` exactly when parameters equal their true values.

* **Bug E — MCMC walker initialisation too narrow; chain too short.**
  Walkers were initialised at :math:`\mathcal{N}(0,\,10^{-3})` — 20–100×
  narrower than the typical contamination amplitude (0.02–0.10).  Changed
  to :math:`\mathcal{N}(0,\,0.05)`.  Default ``n_steps`` raised from 600 to
  1200; ``n_burn`` from 100 to 200.

**Test suite**

* 28 real-template integration tests (``tests/test_real_templates.py``) now
  run instead of skipping: FITS files reside in
  ``~/data/legacysurvey/dr10/systematics/0032/``, not the directory root.
* ElasticNet timing budget raised from 120 s to 600 s to accommodate
  CPU-only machines under load.
* New tests: ``tests/test_simulation.py`` (19 tests) and
  ``tests/test_glass_mocks.py`` (12 tests).

**Deprecations / cleanup**

* Five scripts moved to ``scripts/archive/``:
  ``build_systematic_maps.py``, ``generate_recommendations.py``,
  ``generate_results_ls10_summary.py``, ``generate_sample_pages.py``,
  ``patch_params_from_partials.py``.
  These were development utilities referenced only in the changelog.

**Documentation**

* New results page ``results_simulation_tests.rst`` covering the full
  3 × 3 contamination grid on GLASS and Uchuu mocks at NSIDE 32 and 64.
* New API pages ``api/simulation.rst`` and ``api/glass_mocks.rst``.
* ``results_real_template_validation.rst`` updated to reflect NSIDE = 64
  footprint (22 641 pixels) and correct FITS file paths.
* All five documentation result pages regenerated with fresh figures from
  the fixed pipeline: mock analysis, systematic tests, validation,
  progressive contamination, real-template validation.

**Package metadata**

* Version bumped to **0.9.5**.
* ``glass>=2026.1`` and ``corrfunc`` added as optional extras in
  ``pyproject.toml`` (``pip install sys_mapping[glass]``,
  ``pip install sys_mapping[corrfunc]``).

0.2.4 (2026-05-11)
------------------

**ISD-3 speed and accuracy improvements:**

* **Speed — normal equations replace per-iteration SVD.**
  ``iterative_systematics_decontamination`` now precomputes
  ``Xt = X.T`` once outside the loop.  Each iteration forms the
  weighted normal equations
  :math:`(\mathbf{X}^\top \mathbf{W} \mathbf{X} + \mathbf{\Lambda})\hat{\boldsymbol\alpha}
  = \mathbf{X}^\top \mathbf{W} \hat{\boldsymbol\delta}_g`
  via a single broadcast-scale + BLAS-3 matmul, then calls ``lstsq`` on
  the resulting :math:`(N_{\rm exp} \times N_{\rm exp})` system (e.g.
  55×55 for :math:`n_s=5, d=3`) rather than on the full
  :math:`(N_{\rm pix} \times N_{\rm exp})` matrix.  Measured speedup:
  **5–15× per iteration** for :math:`N_{\rm pix} \gtrsim 10^4`.

* **Accuracy — ridge penalty on polynomial-only columns.**
  New ``lambda_poly: float = 0.0`` parameter added to
  :func:`~sys_mapping.regression.iterative_systematics_decontamination`.
  When positive it adds a diagonal ridge term
  :math:`\lambda_{\rm poly}` to all polynomial expansion columns
  (indices :math:`n_s:`) while leaving the :math:`n_s` linear columns
  unpenalised.  This prevents the polynomial cross-terms from absorbing
  signal that belongs to the linear coefficients — which are the only
  ones used in the weight update — resolving the fixed-point
  inconsistency that caused ISD-3 to converge to implausible solutions.
  Recommended value: ``1e-3 * np.var(delta_g_obs)``.  Default ``0.0``
  (backward-compatible).

* **Interface.**
  :func:`~sys_mapping.regression.run_decontamination` gains
  ``isd_lambda_poly: float = 0.0``, forwarded to both the pass-1 and
  pass-2 ISD calls.

* **Documentation.**
  :doc:`methods` ISD section rewritten to document the normal-equations
  formulation, the fixed-point inconsistency root cause, and the
  ``lambda_poly`` parameter with its mathematical derivation.

0.2.3 (2026-05-06)
------------------

**Pipeline refactoring — method-first execution:**

* New orchestration script ``scripts/run_all_methods_sequential.sh`` runs the
  LS10 BGS dataset for each fitting method in speed order: OLS (seconds) →
  ElasticNet (minutes) → MCMC-add (hours) → MCMC-comb (hours).  Sphinx HTML
  docs are rebuilt after each method phase so intermediate results are
  immediately browsable.
* All four analysis scripts gain ``--only-methods`` (restrict which methods run
  in a given invocation) and ``--force`` (ignore skip-if-done cache).
  The shell script exposes ``FORCE=1`` as an environment variable.
* **Skip-if-done cache:** ``run_bin()`` and ``run_sample()`` check for an
  existing ``results_partial_{slug}.json`` (partial run) or ``results.json``
  (full run) at the start of each bin/sample and return immediately if found.
  Delete the file or pass ``--force`` to recompute.
* **Partial results JSON** written after fast-method phases (OLS, ElasticNet,
  MCMC-add) — lightweight dict with ``schema_version``, ``methods_run``,
  ``timestamp_utc``, ``n_sys_*``, ``syst_names``, ``delta_g_std/skew``, and
  per-method sub-dicts.  The MCMC-comb phase merges these into the final
  ``results.json``.
* **Incremental RST updates** via sentinel-comment protocol:
  ``.. _auto-method-OLS-start:`` / ``.. _auto-method-OLS-end:`` markers are
  written to ``results_ls10.rst`` after each
  method phase; re-running replaces (not duplicates) the section.

**Figures — consistent per-method color palette:**

* New module ``sys_mapping/plotting.py`` with shared ``METHOD_COLORS``,
  ``METHOD_LINESTYLES``, ``METHOD_MARKERS``, ``METHOD_LABELS``, and
  ``METHOD_ORDER`` constants exported from the package.  The palette is
  Tableau-10-inspired and colorblind-safe:
  OLS (#4E79A7 steel blue), ElasticNet (#59A14F forest green),
  ISD-1 (#F28E2B amber), ISD-3 (#E15759 coral red),
  MCMC-add (#B07AA1 purple), MCMC-comb (#76B7B2 teal).
* ``run_ls10_analysis.py``: 2PCF diagnostic plot updated to use
  ``METHOD_COLORS``.

0.2.2 (2026-05-06)
------------------

* **Bug fix (runtime):** ``scripts/run_mock_analysis_real_templates.py`` —
  both ISD blocks computed ``np.asarray(alpha_hat[-1])[:n_sys]`` instead of
  ``np.asarray(alpha_hat)[:n_sys]``.  ``alpha_hat`` is a 1-D array of shape
  ``(n_expanded,)``; ``alpha_hat[-1]`` is a scalar, and slicing it raises
  ``IndexError``.  The surrounding ``except Exception`` silently swallowed the
  error, so ISD results were always missing from this script.  Fixed by
  removing the erroneous ``[-1]`` index.

* **Bug fix (consistency):** ``scripts/run_systematic_tests.py`` — ElasticNet
  discarded the weight map returned by
  :func:`~sys_mapping.regression.elasticnet_contamination_fit` and applied
  additive correction ``δ_obs − α̂·t`` instead of weight-based
  ``w(p)·(1+δ_obs)−1``.  Fixed to store the weights in the results dict so
  that ``corrected_delta()`` applies the paper-correct correction consistent
  with ``run_validation.py`` and Weaverdyck & Huterer (2020) Eq. 46-47.

* **Consistency:** ``scripts/run_systematic_tests.py`` — OLS and MCMC-additive
  now also use weight-based correction ``w(p) = 1/(1 + α̂·t(p))``, making all
  six methods consistent with Weaverdyck & Huterer (2020) Eq. 46-47 and with
  ``run_validation.py``.  The
  :func:`~sys_mapping.regression._compute_weights` helper is used directly.

* **Documentation:** re-ran ``run_systematic_tests.py``, ``run_validation.py``,
  ``run_mock_analysis.py``, and ``run_mock_analysis_progressive.py``; updated
  all result tables throughout the docs.

0.2.1 (2026-05-06)
------------------

* **Bug fix (performance):** ``iterative_systematics_decontamination`` was
  allocating an explicit ``(n_pix × n_pix)`` diagonal weight matrix via
  ``np.diag(weights)`` at every iteration.  For realistic pixel counts
  (NSIDE ≥ 64, n_pix ≳ 50 000) this consumed tens of GB and made each ISD
  iteration orders of magnitude slower than necessary.  Fixed by replacing
  ``X.T @ np.diag(w)`` with the equivalent broadcast ``X.T * w[np.newaxis, :]``
  (``regression.py``).

* **Bug fix (correctness):** The ``lstsq`` fallback path inside the ISD loop
  was scaling both sides by ``weights`` instead of ``sqrt(weights)``, minimising
  :math:`\sum_p w(p)^2 (\delta - \alpha t)^2` rather than the correct weighted
  least-squares objective :math:`\sum_p w(p)(\delta - \alpha t)^2`.  Fixed by
  using ``sw = sqrt(weights)`` on both the design matrix and the response
  (``regression.py``).

* **Minor:** moved ``X = delta_t_expanded.T`` outside the ISD iteration loop
  (it is constant across iterations).

* **Validation:** re-ran ``scripts/run_validation.py`` at NSIDE=32; all ISD
  metrics are numerically identical to the pre-fix run (the bugs affected
  performance, not output for the converging cases in the validation suite).
  ElasticNet results added to all scenario tables in ``docs/results_validation.rst``.

0.2.0 (2026-05-05)
------------------

* Scripts: converted all notebook-based analyses to standalone Python scripts in
  ``scripts/`` (build_systematic_maps, run_paper_validation, run_ls10_analysis,
  run_mock_analysis).
* Execution: added SLURM job templates in ``slurm/`` and nohup bash wrappers in
  ``bash/`` for 32-core server execution.
* Bug fix: corrected template 2PCF estimator and covariance propagation
  (commit ``10c01d0``); all prior results should be regenerated.
* Environment: switched from conda to mamba (``$HOME/miniforge3/envs/sys_map``).
* Notebooks: deprecated executed/bloated notebooks moved to
  ``notebooks/deprecated/``; clean versions kept for interactive exploration.
* Corrfunc: added ``measure_two_point_function_corrfunc`` utility and benchmarking
  script ``scripts/benchmark_corrfunc_vs_treecorr.py``.

0.1.0 (2026-04-23)
------------------

* Initial release.
* Joint inference of additive and multiplicative systematics via JAX JIT-compiled likelihoods.
* emcee MCMC backend (250 walkers, 1500 steps).
* Noise debiasing (Eq. 21) and two-point function correction (Eq. 15-16).
* PCA template rotation (Appendix A).
* Likelihood ratio model selection (Eq. 19).
* Block bootstrap variance estimation (Sec. 6.2).
* Five HEALPix systematic template families (Sec. 4.2).
* TreeCorr Landy-Szalay two-point function measurement.
