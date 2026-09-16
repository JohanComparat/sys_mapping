Testing
=======

The suite collects 798 items: 725 test functions in the 25 files of ``tests/`` and 73
docstring examples from ``sys_mapping/``.
Eight tests carry the ``slow`` marker.
Timing benchmarks live in the separate
`sys_mapping_benchmark <https://github.com/JohanComparat/sys_mapping_benchmark>`_
repository, with the measurements in :doc:`results_benchmark`.

----

Running the tests
-----------------

Install the package with the extras CI uses, then run pytest from the repository root::

   pip install -e ".[dev,regression,glass]"

   pytest                               # everything, docstring examples included
   pytest -m "not slow"                 # the fast suite
   pytest tests/test_contamination.py   # one module
   pytest --cov=sys_mapping --cov-branch --cov-report=term-missing

``pyproject.toml`` sets ``testpaths = ["tests", "sys_mapping"]`` and
``addopts = "-v --doctest-modules"``, so a bare ``pytest`` collects the docstring examples
alongside ``tests/``.

We ran the full suite with
``python -m pytest -q -p no:cacheprovider --cov=sys_mapping --cov-branch`` in the ``sys_map``
environment, with the LS10 and Gaia maps present and the Uchuu mocks absent:
776 passed, 7 skipped, 19 xfailed in 4 min 40 s, at 98.0 % branch coverage.
Six of the skips are the ``test_glass_mocks.py`` tests that need the Uchuu mocks and one is a
docstring example marked ``+SKIP``; the 19 expected failures are the transformability
entries described below.

Tests that need data outside the repository skip when it is absent:
28 of the 39 tests in ``test_real_templates.py`` and parts of ``test_maps.py`` and
``test_simulation.py`` need the LS10 and Gaia maps under
``~/data/legacysurvey/dr10/systematics/0032/``, six tests in ``test_glass_mocks.py`` need
the Uchuu mocks, and the Corrfunc tests in ``test_utils.py`` need Corrfunc.

----

Test modules
------------

Counts are from ``python -m pytest --collect-only -qqq``.

.. list-table::
   :header-rows: 1
   :widths: 34 8 58

   * - File
     - Tests
     - Scope
   * - ``test_accuracy.py``
     - 34
     - Numerical thresholds stated in the docstrings: contamination round trip, maps,
       likelihood, correction, utilities, model selection, bootstrap
   * - ``test_bootstrap.py``
     - 7
     - :mod:`~sys_mapping.bootstrap`: spatial patches, block bootstrap, jackknife
   * - ``test_contamination.py``
     - 19
     - :mod:`~sys_mapping.contamination`: parameter layout, pack/unpack, forward and inverse
       model, two-point correction
   * - ``test_correction.py``
     - 30
     - :mod:`~sys_mapping.correction`: debiasing, PCA rotation, two-point correction and its
       covariance, cross-template terms, harmonic correction, over-correction warning
   * - ``test_covariance.py``
     - 19
     - :mod:`~sys_mapping.covariance`: low-rank precision against a dense reference, mock
       sandwich, sample covariance, Hartlap factor
   * - ``test_des_y6_features.py``
     - 19
     - Spatial cross-validation folds, inverse-variance pixel weights, template vetting,
       over-correction debias, method-marginalised covariance
   * - ``test_diagnostics.py``
     - 50
     - :mod:`~sys_mapping.diagnostics`: null test, SNR ranking, footprint masking, ISD
       significance, residual correlation test, calibrated significance
   * - ``test_edge_cases.py``
     - 53
     - Input checks, fallbacks, optional arguments and missing-dependency errors of the
       public API
   * - ``test_glass_mocks.py``
     - 36
     - :mod:`~sys_mapping.glass_mocks`: :math:`n(z)`, full-sky mocks, Uchuu loading, matched
       spectra, spectrum choice, per-pixel null draws
   * - ``test_inference.py``
     - 15
     - :mod:`~sys_mapping.inference`: log-probability, emcee sampler, chain summaries,
       ``refine_to_mle``
   * - ``test_jax_acceleration.py``
     - 15
     - JAX kernels against NumPy references: ISD :math:`\Delta\chi^2`,
       ``isd_marginal_fit``, null test, polynomial-OLS backend, parallel GLASS mocks
   * - ``test_jax_transformability.py``
     - 46
     - ``jax.jit``, ``jax.vmap`` and ``jax.grad`` on 17 public numeric cases
   * - ``test_likelihood.py``
     - 11
     - :mod:`~sys_mapping.likelihood`: Gaussian and skew-normal likelihoods, gradients, GLS
       precision
   * - ``test_ls10_script.py``
     - 23
     - Helpers of ``scripts/run_ls10_analysis.py``: parameter expansion, weight convention,
       skew flag, mandatory null spectrum, null fields
   * - ``test_maps.py``
     - 34
     - :mod:`~sys_mapping.maps`: power spectra, synthetic maps, pixelisation, overdensity,
       template assignment, real-template loading
   * - ``test_mocks.py``
     - 40
     - :mod:`~sys_mapping.mocks`: lognormal field, Galactic mask, mock catalogues and
       suites, pipeline integration
   * - ``test_model_selection.py``
     - 39
     - :mod:`~sys_mapping.model_selection`: likelihood ratio test, forward selection, SNR
       pre-selection, ``lrt_from_maxima``
   * - ``test_nonlinear_response.py``
     - 39
     - Non-linear template responses: normalisation, injection, ISD-1 against ISD-3
   * - ``test_power_spectrum.py``
     - 21
     - :mod:`~sys_mapping.power_spectrum`: harmonic bias, pseudo-:math:`C_\ell`, template
       subtraction, mode projection
   * - ``test_real_templates.py``
     - 39
     - Every method on a mock built with the LS10 and Gaia templates (below); footprint
       standardisation; resolution from occupancy
   * - ``test_regression.py``
     - 39
     - :mod:`~sys_mapping.regression`: weights, ElasticNet, polynomial OLS, ISD, method
       comparison, ``run_decontamination``
   * - ``test_samplers.py``
     - 20
     - Analytic additive posterior, NUTS, chain execution, sampler dispatch
   * - ``test_simulation.py``
     - 21
     - :mod:`~sys_mapping.simulation`: contamination grid, injection, FITS round trip,
       systematic maps, footprint mask, :math:`w(\theta)` recovery
   * - ``test_snr_preselection.py``
     - 35
     - Two-stage SNR pre-selection on a GLASS mock with two injected templates
   * - ``test_utils.py``
     - 23
     - :mod:`~sys_mapping.utils`: TreeCorr and Corrfunc two-point functions, KK correlations
       and covariance, template correlation matrix
   * - ``sys_mapping/*.py`` (docstring examples)
     - 73
     - Every ``Examples`` section, 1 to 11 per module

The slow tests are three calibration checks on clustered fields in ``test_diagnostics.py``
(residual test size and power, calibrated-significance false-positive rate), two comparisons
of the analytic posterior and NUTS with emcee in ``test_samplers.py``, and three checks of
template-correlation support at the pixel scale in ``test_utils.py``.

----

Coverage and CI
---------------

``[tool.coverage.run]`` in ``pyproject.toml`` measures ``sys_mapping`` with
``branch = true``; ``[tool.coverage.report]`` sets ``fail_under = 97`` on the combined line
and branch figure, which pytest-cov enforces whenever ``--cov`` is given.
The measured coverage, per module, is in :doc:`coverage`: 98.0% combined, 98.8% of lines and
94.8% of branches.

``.github/workflows/tests.yml`` runs on Python 3.11 and 3.12 with ``JAX_PLATFORMS=cpu`` and
``XLA_PYTHON_CLIENT_PREALLOCATE=false``, after installing ``.[dev,regression,glass]``.
Pull requests run the fast suite (``-m "not slow"``); pushes to ``main``, the nightly schedule
and manual runs run the full suite.
Both measure branch coverage and upload ``coverage.xml`` to Codecov with a flag per Python
version (``py3.11``, ``py3.12``).
The LS10, Gaia and Uchuu data are absent there, so the tests that need them skip.

----

Docstring examples
------------------

Docstring examples run under ``--doctest-modules`` with the ``ELLIPSIS`` and
``NORMALIZE_WHITESPACE`` flags.
NumPy 2 prints scalars as ``np.float64(0.5)`` and ``np.True_``; the root ``conftest.py``
selects ``np.set_printoptions(legacy="1.25")`` whenever doctests are collected, so the
examples print the plain form.

----

JAX tests
---------

``tests/test_jax_transformability.py`` wraps 17 public numeric cases as functions of one
array and applies ``jax.jit``, ``jax.vmap`` and, where a gradient is meaningful,
``jax.grad``, 46 cases in all.
A case passes when the transform runs on traced inputs and reproduces the eager result.
The functions written against NumPy are listed in ``_NUMPY_ON_TRACER`` and their cases
marked strict ``xfail``: 8 functions and 19 cases.
Porting one of them turns its cases into unexpected passes, which fail the suite until the
entry is removed.

``tests/test_jax_acceleration.py`` checks the JAX kernels that run the ISD, ranking and
null-test statistics against NumPy reference implementations kept in the test file.

----

.. _real-template-tests:

Real-template integration tests
-------------------------------

``tests/test_real_templates.py`` builds one mock at NSIDE 32 with four templates, synthetic
families 0 and 1, ``GAIA_nstar_faint`` and ``LS10_GALDEPTH_Z``, on the pixels where both
real maps are finite and positive (the mask of
:func:`~sys_mapping.maps.load_real_templates`).

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Parameter
     - Value
   * - :math:`a_i^{\rm true}`
     - :math:`(0.08, -0.05, 0.06, -0.04)`
   * - :math:`b_i^{\rm true}`
     - :math:`(0.04, 0, -0.03, 0.05)`
   * - Clean field
     - lognormal, :math:`C_\ell^G \propto (\ell+1)^{-2}`, :math:`\sigma_G = 0.5`
   * - Galaxies per pixel
     - 50 (Poisson)
   * - Randoms
     - 8 times the galaxy density, no noise
   * - Seed
     - 7

The tests check shapes, finiteness and a mean absolute amplitude error below a loose
tolerance: 0.20 for OLS, 0.25 for ElasticNet (3 folds), ISD-1 and ``run_mcmc`` additive
chains, and 0.30 for :math:`a_i` and :math:`b_i` of combined chains.
ISD runs with a fixed ``chi2_68 = 50`` in place of a mock calibration, and ISD-3 is checked for
finite, positive weights and termination.
The likelihood-ratio test must reject the additive null at 5%.
The residual test must detect contamination on a template held out of the correction,
against 30 null realisations put through the same correction.
The SNR ranking must be non-negative with at least one entry above 0.01.

----

Adding tests
------------

1. One test class per public function or function group.
2. ``@pytest.fixture(scope="class")`` or ``"module"`` for expensive shared inputs.
3. Test shapes, value ranges and known analytic limits.
4. Seed every random draw.
5. Mark long-running tests with ``@pytest.mark.slow``; pull requests run without them.
6. Add the file to the table above.

Standalone scripts such as ``scripts/run_systematic_tests.py`` produce the result pages and
are not collected by pytest.
