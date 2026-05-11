Testing
=======

``sys_mapping`` has a comprehensive test suite in the ``tests/`` directory.
Tests are organised by module and cover correctness, numerical accuracy,
and performance.

Running the tests
-----------------

Install the package with the development extras first::

   pip install -e ".[dev]"

Then run the full suite::

   pytest tests/ -v

Run a single module::

   pytest tests/test_contamination.py -v

Run tests that require scikit-learn (skipped otherwise)::

   pip install scikit-learn
   pytest tests/test_regression.py -v

Run performance benchmarks (slow; uses ``--benchmark`` marker)::

   pytest tests/test_benchmarks.py -v

----

Test modules
------------

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - File
     - Tests
     - Coverage
   * - ``test_contamination.py``
     - 15
     - :mod:`~sys_mapping.contamination` — parameter layout, pack/unpack
       round-trip, forward model, two-point correction formula
   * - ``test_likelihood.py``
     - 7
     - :mod:`~sys_mapping.likelihood` — Gaussian and skew-normal log-likelihoods,
       JAX gradient correctness via finite differences
   * - ``test_maps.py``
     - 32
     - :mod:`~sys_mapping.maps` — power spectrum generation, synthetic map
       generation, catalog pixelisation, overdensity computation; plus
       ``load_real_template`` / ``load_real_templates`` (5 synthetic FITS
       fixture tests + 7 real-data tests guarded by ``@real_data``)
   * - ``test_correction.py``
     - 12
     - :mod:`~sys_mapping.correction` — noise debiasing, PCA rotation,
       round-trip param transform, two-point correction
   * - ``test_model_selection.py``
     - 7
     - :mod:`~sys_mapping.model_selection` — LRT statistic, chi-squared
       distribution under the null, p-value monotonicity
   * - ``test_accuracy.py``
     - 34
     - End-to-end accuracy: contamination round-trip errors < 1e-10;
       overdensity mean near zero; likelihood gradient norm; correction
       bias below tolerance; amplitude bias estimator
   * - ``test_power_spectrum.py``
     - 19
     - :mod:`~sys_mapping.power_spectrum` — harmonic bias formula
       :math:`b_\ell = -n/(2\ell+1)`, pseudo-Cℓ white noise validation,
       template subtraction identity (alpha=0), BMP/EMP mode projection
   * - ``test_regression.py``
     - 26
     - :mod:`~sys_mapping.regression` — weight bounds, ElasticNet recovery
       (skipped without scikit-learn), ISD convergence and polynomial
       order reduction, method comparison output keys;
       ``combined_mcmc`` keys and default-method assertion.
   * - ``test_diagnostics.py``
     - 19
     - :mod:`~sys_mapping.diagnostics` — null test on uncorrelated fields,
       SNR ranking shape and ordering, footprint masking output consistency
   * - ``test_mocks.py``
     - 40
     - :mod:`~sys_mapping.mocks` — lognormal field properties (skewness,
       positivity, reproducibility), galactic mask geometry, mock catalog
       shapes and physics, suite scenarios, pipeline integration
   * - ``test_benchmarks.py``
     - 19
     - Execution time regression: contamination, maps, likelihood,
       correction, and utils must complete within set thresholds
   * - ``test_timing.py``
     - 240
     - Wall-clock scaling of all six methods as a function of NSIDE
       (8, 16, 32, 64) and n_templates (1–10); 40 configs × 4 NSIDE
       values = 160 tests per fast method; MCMC tests marked
       ``@pytest.mark.slow`` (skip with ``-m "not slow"``)
   * - ``test_real_templates.py``
     - 28
     - Integration tests using real GAIA and LS DR10 HEALPix maps as
       systematic templates (synth_5, synth_6); all six methods exercised
       on a synthetic mock built within the LS10 survey footprint (NSIDE = 32,
       ~5 954 valid pixels).  Tests skipped automatically if FITS files are
       absent.  See :ref:`real-template-tests` below.

----

Test design philosophy
-----------------------

**Unit tests** check individual functions in isolation:

* Input/output shapes are always verified.
* Round-trip identities are tested (e.g. ``apply_contamination`` ∘
  ``invert_contamination`` = identity).
* Known closed-form results are tested to machine precision where applicable
  (e.g. the harmonic bias formula, pack/unpack).

**Accuracy tests** (``test_accuracy.py``) verify that numerical errors
remain below physically meaningful thresholds across a range of NSIDE
values, template counts, and noise levels.

**Integration tests** (``TestMockPipelineIntegration`` in ``test_mocks.py``)
run the full pipeline from raw catalogs to overdensity estimates and confirm
that OLS recovers injected additive amplitudes to within 50 % at low
signal-to-noise (NSIDE = 16, 100 galaxies / pixel).

**Skipped tests** — ``test_real_templates.py`` tests are skipped when the
GAIA and LS10 FITS files are absent from
``~/data/legacysurvey/dr10/systematics/``.  ElasticNet tests in
``test_regression.py`` are skipped when ``scikit-learn`` is not installed.

----

Continuous integration
-----------------------

The test suite is designed to run in the ``sys_map`` conda environment::

   conda activate sys_map
   pytest tests/ -v --tb=short

Expected output with scikit-learn and real data files present::

   258 passed in ~34 s

Without real data files (no GAIA/LS10 FITS)::

   230 passed in ~21 s

----

Systematic test matrix
-----------------------

Beyond the unit and integration test suite, a separate **systematic test
matrix** is available in ``scripts/run_systematic_tests.py``.  This script
is not run as part of ``pytest``; it is a standalone benchmark that exhaustively
evaluates all six methods across 32 contamination configurations (Tier 1:
additive-only and multiplicative-only with 1–7 templates; Tier 2: mixed
additive+multiplicative with varying numbers of multiplicative templates).

The key metric is :math:`\sigma[(1+\delta_g^{\rm corr})/(1+\delta_g^{\rm true})]`,
the standard deviation of the pixel-level correction ratio.  Results and figures
are documented in :doc:`results_systematic_tests`.

To run::

   conda activate sys_map
   python scripts/run_systematic_tests.py \\
       --nside 32 --n-walkers 64 --n-steps 200 --n-burn 50 \\
       --output-dir results/systematic_tests/

Expected runtime: ~8 minutes (NSIDE = 32, all 6 methods, 32 configurations).

After re-running with the updated script, the CSV will contain a ``time_s`` column
per method-configuration row, and two timing figures are written to
``results/systematic_tests/``:

* ``timing_vs_ntemplates.png`` — wall-clock time vs. n_templates per method (log scale)
* ``timing_mean_per_method.png`` — mean compute time per method across all 32 configs

----

Timing scaling tests
---------------------

``tests/test_timing.py`` parametrises all six methods over NSIDE ∈ {8, 16, 32, 64}
and n_templates ∈ {1, …, 10}, measuring actual wall-clock time for each
(NSIDE, n_templates) pair.  MCMC tests are marked ``@pytest.mark.slow`` and
can be skipped::

   pytest tests/test_timing.py -v -s -m "not slow"   # OLS / ElasticNet / ISD only
   pytest tests/test_timing.py -v -s                  # all methods (slow)

The ``-s`` flag is required to see the per-test timing printed to stdout.
Each test asserts that the method completes within a generous wall-clock bound
(30 s for OLS/ISD, 120 s for ElasticNet/ISD-3, 600 s for MCMC), so the suite
fails only if a method becomes catastrophically slow.  Use the stdout output for
the actual scaling data.

Expected test count: 40 (NSIDE × n_templates) per method class × 6 classes = 240 tests
(excluding MCMC slow tests from the default run: 160 fast tests).

----

Adding new tests
-----------------

Follow the existing pattern:

1. Create a test class per public function group (e.g. ``class TestMyFunction``).
2. Use ``@pytest.fixture(scope="class")`` for expensive shared objects.
3. Test shapes, dtype, value ranges, and known analytic limits.
4. For stochastic tests, fix the seed via ``seed=`` or ``np.random.default_rng(N)``.
5. Add the new file to this page.

----

.. _real-template-tests:

Real-template integration tests
---------------------------------

``tests/test_real_templates.py`` tests every implemented method end-to-end on
a synthetic galaxy mock built using **real observational systematic maps**
from GAIA DR3 and the Legacy Survey DR10.  These tests exercise the entire
pipeline — loading, normalisation, injection, inference, and model selection
— with physically realistic templates rather than toy random fields.

Template set
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 12 20 18 50

   * - Label
     - Source
     - NSIDE (tests)
     - Physical interpretation
   * - ``synth_0``
     - Synthetic, family 0 (:math:`C_\ell \propto e^{-\ell/500}`)
     - 32
     - Large-scale coherent artefact (e.g. zodiacal light)
   * - ``synth_1``
     - Synthetic, family 1 (:math:`C_\ell \propto e^{-(\ell/250)^2}`)
     - 32
     - Intermediate-scale artefact (e.g. airglow)
   * - **synth_5**
     - **GAIA DR3** — ``GAIA_nstar_faint_NSIDE_00032.fits`` (``nstar_faint`` column)
     - **32**
     - **Faint-star surface density** — proxy for stellar contamination
       (misclassified stars inflate galaxy counts)
   * - **synth_6**
     - **LS DR10** — ``LS10_GALDEPTH_Z_NSIDE_0032.fits`` (``GALDEPTH_Z`` column)
     - **32**
     - **Galaxy depth in the z band** — proxy for selection-depth variations
       (depth modulates survey completeness multiplicatively)

Survey footprint: the LS10 depth valid mask (pixels where GALDEPTH_Z > 0)
restricts the mock to **5 954 pixels** (48.4 % of the NSIDE = 32 sphere).
GAIA is a full-sky map (12 288 / 12 288 valid pixels); its values inside the
LS10 footprint are used.

Mock configuration and injected parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Parameter
     - Value
   * - NSIDE
     - 32 (pixel area ≈ 3.4 deg², 12 288 pixels total)
   * - Survey footprint pixels
     - 5 954 (LS10 depth mask)
   * - Templates :math:`n_s`
     - 4 (synth_0, synth_1, GAIA_nstar_faint, LS10_GALDEPTH_Z)
   * - :math:`a_i^{\rm true}` (additive)
     - :math:`(0.08,\ {-0.05},\ 0.06,\ {-0.04})`
   * - :math:`b_i^{\rm true}` (multiplicative)
     - :math:`(0.04,\ 0.00,\ {-0.03},\ 0.05)`
   * - Mean galaxies per pixel :math:`\bar n`
     - 50
   * - Random / galaxy ratio
     - 8×
   * - Seed
     - 7

The galaxy overdensity follows a lognormal field
:math:`\delta_g^{\rm true} = e^{G - \sigma_G^2/2} - 1` with
:math:`C_\ell^G \propto (\ell+1)^{-2}`, :math:`\sigma_G = 0.5`.
Note that template 1 (``synth_1``) has :math:`b_1^{\rm true} = 0` (purely
additive) while templates 0, 2, 3 have non-zero multiplicative amplitudes —
a realistic mixed-contamination scenario.

Method results
^^^^^^^^^^^^^^

Each of the six implemented methods is run once on this fixed mock:

.. list-table::
   :header-rows: 1
   :widths: 22 20 16 42

   * - Method
     - Mean :math:`|\hat a_i - a_i^{\rm true}|`
     - Tolerance
     - Notes
   * - **OLS**
     - < 0.20
     - 0.20
     - Ordinary least-squares pixel regression; fastest method
   * - **ElasticNet**
     - < 0.25
     - 0.25
     - Cross-validated (3 folds); requires ``scikit-learn ≥ 1.3``
   * - **ISD-1** (poly_order = 1)
     - < 0.25
     - 0.25
     - Converges in < 50 iterations
   * - **ISD-3** (poly_order = 3)
     - n/a (numerically unstable)
     - finite values only
     - Polynomial expansion produces 34 features for :math:`n_s = 4`,
       :math:`n_{\rm pix}/n_{\rm feat} \approx 175`; ill-conditioned
       with real correlated templates.  Check finiteness only.
   * - **MCMC-additive**
     - < 0.25
     - 0.25
     - Chain shape :math:`(n_w \times 160,\; n_s + 1)` with
       :math:`n_w \geq 2(n_s+1)+2 = 12`; :math:`n_{\rm dim} = 5`
   * - **MCMC-combined** (Berlfein+2024)
     - < 0.30 for :math:`\hat a_i`; < 0.30 for :math:`\hat b_i`
     - 0.30
     - Chain shape :math:`(n_w \times 160,\; 2n_s + 1)`;
       :math:`n_{\rm dim} = 9`.  Positive posterior variances confirmed.

Model selection and diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **LRT** (:math:`H_0`: additive, :math:`H_1`: combined) — the additive null
  is rejected at the 5 % level because :math:`b_0, b_2, b_3 \neq 0`.
  The test statistic :math:`\lambda_{\rm LR} = 2[\ln\mathcal{L}_1 - \ln\mathcal{L}_0]`
  is positive and :math:`p < 0.05`.

* **Null test** — the maximum Pearson correlation :math:`\max_i |r_i|` between
  the OLS-corrected weights and the template maps satisfies
  :math:`\max|r_i| < 0.50`, confirming partial residual removal.

* **SNR ranking** — the SNR array has shape :math:`(4,)`, all entries are
  :math:`\geq 0`, and at least one entry is :math:`> 0.01`, demonstrating
  that the real GAIA and LS10 maps carry detectable systematic signal.

Running the real-template tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure the FITS files are present at their default paths (see
:func:`~sys_mapping.maps.load_real_templates`), then::

   conda activate sys_map
   pytest tests/test_real_templates.py -v

Expected output::

   28 passed in ~62 s

To skip when files are absent, they degrade gracefully::

   28 skipped (reason: GAIA/LS10 FITS files not found)

----

.. rubric:: Test outcome

The full test suite was executed in the ``sys_map`` conda environment
(Python 3.11, JAX 64-bit, scikit-learn ≥ 1.3, real GAIA and LS10 FITS files
present) against the current codebase.

**Results:**

* **258 passed, 0 failed, 0 errors** (excluding ``test_timing.py``)
* ``test_real_templates.py`` — **28 passed** using real GAIA DR3 and LS10 DR10
  systematic maps; all six methods completed without error on the 5 954-pixel
  LS10 footprint.
* ``test_regression.py`` — **26 passed**, including the previously known edge
  case (``TestElasticNet::test_weights_bounded_positive``), which now passes.
* The systematic test matrix (``scripts/run_systematic_tests.py``, 32
  configurations) ran to completion; results are documented in
  :doc:`results_systematic_tests`.

**Status: PASSED** — all 258 tests pass with no known failures.
