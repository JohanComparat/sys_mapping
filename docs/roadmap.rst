Roadmap
=======

Open work, ordered by its effect on the scientific products.
:doc:`bibliography` gives the per-paper implementation status.

.. contents:: On this page
   :local:
   :depth: 1

----

Statistics
----------

Full-rank GLS likelihood
~~~~~~~~~~~~~~~~~~~~~~~~

The pixel likelihood assumes independent pixels, so on a clustered field its posterior widths
and :math:`\chi^2` p-values are too small.
Calibrated errors and p-values come from mock nulls
(:func:`~sys_mapping.diagnostics.calibrated_template_significance`,
:func:`~sys_mapping.covariance.mock_sandwich_covariance`, ``null_lambda`` in
:func:`~sys_mapping.model_selection.likelihood_ratio_test`).
A full-rank precision built from the theory :math:`C_\ell` plus shot noise would calibrate the
likelihood itself.
On the full sky :math:`R^{-1}` is diagonal in harmonic space; on the cut sky
:math:`R^{-1}v` is a conjugate-gradient solve around a spherical-harmonic operator,
differentiable in JAX.
:func:`~sys_mapping.covariance.build_harmonic_precision` is the entry point and raises
``NotImplementedError``.
The low-rank :class:`~sys_mapping.covariance.LowRankPrecision` does not calibrate the
likelihood when the templates lie outside the span of the mock ensemble.

Fisher-matrix bias propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given the residual :math:`\Delta C_\ell` or :math:`\Delta w(\theta)` left by a correction,
the induced parameter bias is
:math:`\delta\theta = F^{-1}\,\partial\mu^{\rm T} C^{-1} \Delta\mu`
(DeRose et al. 2026).
The likelihood is differentiable in JAX (``likelihood.make_log_likelihood``,
``nuts.build_logdensity``), so ``jax.jacfwd`` supplies the derivatives; the missing piece is
a theory data vector for :math:`\Omega_m` and :math:`b\sigma_8`.
This states the break-even rule of :doc:`detectability_law` in cosmological parameters.

----

Contamination model
-------------------

The composition term
~~~~~~~~~~~~~~~~~~~~

Galaxy types respond differently to the same observing conditions, so the sample composition,
:math:`n(z)` and :math:`b(z)` vary across the footprint even after a perfect first-order
correction (Kong et al. 2026).
The residual is second order in the templates,
:math:`\sum_{kk'} h_k h_{k'} \langle f_k, f_{k'}\rangle \langle \delta_k, \delta_{k'}\rangle`,
does not correlate with them, and is invisible to the residual and null tests.
The package has no sub-sample model: the work is to fit :math:`f_k` per sub-sample (split by
magnitude or colour) and bound the term.

The monopole
~~~~~~~~~~~~

:func:`~sys_mapping.maps.compute_overdensity` normalises the randoms to the galaxy total, so
:math:`\langle\hat\delta_g\rangle = 0` by construction and the additive contribution
:math:`\boldsymbol\alpha\cdot\bar{\mathbf M}` to :math:`\bar n_g` is absorbed.
That contribution rescales every :math:`C_\ell` by a constant (Hernández-Monteagudo et al. 2025).
Recovering it needs an external :math:`\bar n_g`, for example from a purity measurement; the
deliverable is to accept one and report the induced amplitude offset.

----

Two-point correction
--------------------

Weighted cross-term matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~sys_mapping.utils.template_correlation_matrix` accepts per-object weights.
The correction subtracts :math:`\sum_{ij}\tilde A_{ij}\,\xi_{ij}(\theta)` from
:math:`\hat w(\theta)`, which is consistent when both are measured with the same pair weights.
``scripts/run_ls10_analysis.py`` measures both unweighted.
A survey that weights its pairs by coverage (``fracdet``) or by systematic weights needs the
matrix measured with those weights and a test that the weighted correction recovers the
clean :math:`w(\theta)`.

----

Products and validation
-----------------------

Euclid matched spectra
~~~~~~~~~~~~~~~~~~~~~~

Matched spectra exist for the nine LS10 samples, in ``matched_spectra/`` of the
``sys_mapping_benchmark`` repository.
The Euclid samples have none, so no calibrated statistic (null, significance, mock LRT) can be
issued for them.
The work is to run ``characterisation/match_glass_to_data.py`` from that repository per Euclid
sample and resolution, and pass the large-scale validation that
:func:`~sys_mapping.glass_mocks.load_matched_cl` requires.

Re-run the older result grids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following pages come from runs of code older than 1.4.0 and are to be regenerated with it:
:doc:`results_validation`, :doc:`results_simulation_tests`,
:doc:`results_systematic_tests`, :doc:`results_mock_analysis`,
:doc:`results_real_template_validation`, :doc:`results_progressive_contamination`,
:doc:`results_snr_preselection` and :doc:`results_benchmark`.

NUTS divergences on the LS10 combined fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At NSIDE 32 the LS10 ``MCMC-comb`` fit has about 7% divergent transitions, 277 to 279 of 4000,
with either the dense or the diagonal mass matrix.
The divergent region is not located.
The work is to locate it from the divergent positions, then reparametrise or raise
``target_acceptance_rate`` (default 0.8) in :func:`~sys_mapping.nuts.run_nuts`.

ISD bin count at high resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``isd_n_bins`` defaults to 10.
A sweep at ``poly_order = 3`` favours 20 bins on residual (0.028 against 0.032), precision
(0.925 against 0.870) and recall (0.835 against 0.808).
It has not been run on the sparse LS10 samples at NSIDE 128 and above, where a bin needs at
least two pixels to enter the fit.
The work is to run it there, then set the default.

----

Infrastructure
--------------

JAX transformability backlog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Eight public functions fail ``jax.jit`` or ``jax.vmap`` and are listed in
``_NUMPY_ON_TRACER`` in ``tests/test_jax_transformability.py``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Blocked by
   * - ``correct_two_point_function``
     - ``np.asarray`` on ``w_obs`` (``correction.py``)
   * - ``rotate_templates``
     - ``np.linalg.eigh`` on the second moment (``correction.py``)
   * - ``sample_covariance``
     - ``np.cov`` (``covariance.py``)
   * - ``mock_sandwich_covariance``
     - ``np.linalg`` on the mock fields (``covariance.py``)
   * - ``calibrated_template_significance``
     - ``np.linalg.pinv`` and input validation (``diagnostics.py``)
   * - ``residual_template_correlation_test``
     - ``np.asarray`` and input validation (``diagnostics.py``)
   * - ``snr_template_ranking``
     - ``np.asarray`` before the JAX kernel (``diagnostics.py``)
   * - ``posterior_median_params``
     - ``np.median`` (``inference.py``)

Their cases are strict ``xfail``, so each port removes an entry; see :doc:`testing` and
:doc:`coverage`.

API page check
~~~~~~~~~~~~~~

No test asserts that every module reachable from ``sys_mapping.__all__`` has a page under
``docs/api/``.

----

From the literature
-------------------

Capabilities named in recent papers that the package does not have.

- Power-law template linearisation: fit
  :math:`n_g/\langle n_g\rangle \propto (M_j/\langle M_j\rangle)^{\alpha_j}` and use
  :math:`M_j^{1/|\alpha_j|}` in place of :math:`M_j` when :math:`|\alpha_j| > 1`, before
  standardisation (Hernández-Monteagudo et al. 2025).
- Variance-minimisation estimator of :math:`b`: subtract the OLS linear part, then minimise
  the variance of :math:`\delta_g / \prod_i (1 + \beta_i t_i)`; an estimator of the
  multiplicative amplitude independent of the likelihood-ratio test
  (Hernández-Monteagudo et al. 2025).
- Residual forecast per multipole: the residual on :math:`C_\ell` as a function of
  :math:`\ell` for a contamination configuration, where :doc:`detectability_law` gives a
  scalar (Hernández-Monteagudo et al. 2025).
- Window-decomposed estimator
  :math:`w_{\rm obs} = \sum_{AB} {\rm Win}_{AB}\, w_{AB}` with randoms binned by
  systematics-map value; ``utils.measure_cross_two_point_function`` already reuses ``dr`` and
  ``rr`` pair counts (Kong et al. 2026).
- Spatially varying :math:`n(z, {\rm sys})` and :math:`b(z, {\rm sys})`, with the
  four-parameter :math:`n(z)` model of DeRose et al. 2026 (Kong et al. 2026).
- Leverage mask: cut pixels with a large diagonal of the hat matrix of the template design
  matrix (Weaverdyck et al. 2026).
- Data-split consistency test: :math:`\chi^2 = \Delta^{\rm T} (C/4)^{-1} \Delta` between
  :math:`w(\theta)` in two halves of the footprint (Weaverdyck et al. 2026).
