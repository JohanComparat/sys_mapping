Real-template validation
========================

sys_mapping 1.4.0 (commit ``6f43dcc`` plus the uncommitted working-tree changes of 2026-09-15), laptop Intel Core i9-11900H, 2026-09-15; the mock run took 31 min on a shared machine and the test suite 14 s.

We contaminate synthetic catalogues with two observational systematic maps, the Gaia DR3 faint-star
density and the Legacy Survey DR10 z-band galaxy depth, next to three synthetic templates, and
recover the amplitudes with the six methods. The run is ``scripts/run_mock_analysis_real_templates.py``;
its per-mock results are in
``docs/_static/results_real_template_validation/mock_results_real_templates.csv``.
The integration tests in ``tests/test_real_templates.py`` exercise the same templates on a fixed
mock at NSIDE 32.

Mocks
-----

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Parameter
     - Value
   * - NSIDE
     - 64
   * - Footprint
     - pixels where both real maps are defined, 22 641 (46.1 % of the sky)
   * - Templates
     - synth_0, synth_1, synth_2 (families 0, 1, 2), ``GAIA_nstar_faint``, ``LS10_GALDEPTH_Z``;
       the real maps have zero mean and unit variance on their own footprints
   * - True field
     - lognormal, :math:`\sigma = 0.5`, :math:`C_\ell \propto (\ell+1)^{-2}`
   * - Contamination
     - :math:`\hat\delta_g = \delta_g (1 + \sum_i b_i t_i) + \sum_i a_i t_i`, with
       :math:`a_i, b_i \sim \mathcal{N}(0, 0.1^2)` drawn per mock
   * - Galaxies
     - 30 per pixel on average, 589 000 to 713 000 per mock; randoms at 8 times that density
   * - Mocks
     - 5, seeds 0 to 4

All methods run through :func:`~sys_mapping.regression.run_decontamination`: MCMC-add is the exact
analytic posterior and MCMC-comb NUTS with 4 chains of 1 000 warmup steps and 1 000 draws.
ISD is calibrated on 50 uncontaminated mocks from the same generator. Its
:math:`\Delta\chi^2_{68}` for synth_0, synth_1, synth_2, Gaia and depth is 1.8, 1.6, 167, 291 and
211 at degree 1, and 4.4, 4.8, 215, 497 and 428 at degree 3: the three large-scale maps correlate
with the clustered field by chance much more than synth_0 and synth_1.
The likelihood ratio :math:`\lambda_{\rm LR}` is taken between the additive and combined maxima.

Additive amplitudes
-------------------

.. figure:: _static/results_real_template_validation/real_template_a_recovery.png
   :width: 100%
   :align: center

   :math:`\hat a_i - a_i^{\rm true}` per template and method over the 5 mocks.

.. figure:: _static/results_real_template_validation/real_template_method_rms.png
   :width: 70%
   :align: center

   Rms of :math:`\hat a_i - a_i^{\rm true}` over the 25 template–mock pairs, per method.

.. list-table:: Rms of :math:`\hat a_i - a_i^{\rm true}` over the 5 mocks.
   :header-rows: 1
   :widths: 20 13 13 13 13 13 15

   * - Method
     - synth_0
     - synth_1
     - synth_2
     - Gaia
     - depth
     - all
   * - OLS
     - 0.011
     - 0.009
     - 0.040
     - 0.086
     - 0.019
     - 0.044
   * - ElasticNet
     - 0.019
     - 0.017
     - 0.030
     - 0.064
     - 0.022
     - 0.035
   * - ISD-1
     - 0.011
     - 0.008
     - 0.046
     - 0.081
     - 0.056
     - 0.049
   * - ISD-3
     - 0.011
     - 0.009
     - 0.046
     - 0.076
     - 0.051
     - 0.046
   * - MCMC-add
     - 0.011
     - 0.009
     - 0.040
     - 0.086
     - 0.019
     - 0.044
   * - MCMC-comb
     - 0.013
     - 0.010
     - 0.040
     - 0.079
     - 0.011
     - 0.040

MCMC-add reproduces OLS to the third decimal. ISD-1 and ISD-3 take 2 to 4 steps and stop on the
threshold in every mock. Every template they leave at :math:`\hat a = 0` is synth_2, Gaia or
depth, the three maps with the large :math:`\Delta\chi^2_{68}`; Gaia is left at zero in 9 of the
10 ISD fits. The Gaia amplitude has the largest error for every method.

Multiplicative amplitudes and sampling
--------------------------------------

.. figure:: _static/results_real_template_validation/real_template_b_recovery.png
   :width: 70%
   :align: center

   :math:`\hat b_i - b_i^{\rm true}` of MCMC-comb per template over the 5 mocks.

The rms of :math:`\hat b_i - b_i^{\rm true}` is 0.038, 0.033, 0.042 and 0.064 for synth_0,
synth_1, synth_2 and depth, and 0.51 for Gaia, whose errors are +0.74, −0.77, −0.15, +0.11 and
+0.38 in the five mocks.
NUTS converged in mocks 0 and 1 (:math:`\hat R = 1.00`, effective sample size 3 600 and 5 700,
64 s and 48 s). In mocks 2, 3 and 4 the chains did not mix: :math:`\hat R` is 23.0, 24.4 and 23.2,
the effective sample size 2, with no divergent transitions, in 317 s, 854 s and 534 s.
MCMC-add takes 0.1 s per mock after 5 s of compilation on the first.

Model selection
---------------

.. figure:: _static/results_real_template_validation/real_template_lrt_statistics.png
   :width: 70%
   :align: center

   :math:`\lambda_{\rm LR}` of the additive against the combined model over the 5 mocks.

Every mock carries non-zero :math:`b_i`, and :math:`\lambda_{\rm LR}` is 1 439, 1 647, 787, 858 and
783, so the :math:`\chi^2` test rejects the additive model at 5 % in all five. On a clustered field
that :math:`\chi^2` p-value is too small (:ref:`lrt-methods`).

Integration tests
-----------------

``tests/test_real_templates.py`` builds one mock at NSIDE 32 from synth_0, synth_1, Gaia and depth,
with :math:`a = (0.08, -0.05, 0.06, -0.04)`, :math:`b = (0.04, 0, -0.03, 0.05)`, 50 galaxies per
pixel and seed 7. Its 39 tests check shapes, finiteness and termination for OLS, ElasticNet, ISD-1
and ISD-3, MCMC chains of the additive and combined models, the likelihood ratio test and the
calibrated residual test, and recovery of the amplitudes within 0.20 to 0.30. All 39 passed in
14 s. The tests skip when the NSIDE 32 maps are absent from
``~/data/legacysurvey/dr10/systematics/0032/``.

Reproduce
---------

.. code-block:: bash

   python scripts/run_mock_analysis_real_templates.py \
       --syst-dir ~/data/legacysurvey/dr10/systematics/0064 \
       --nside 64 --n-mocks 5 \
       --output-dir docs/_static/results_real_template_validation/
   pytest tests/test_real_templates.py
