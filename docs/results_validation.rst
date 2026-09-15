End-to-end validation
=====================

sys_mapping 1.4.0 (commit ``6f43dcc`` plus the uncommitted working-tree changes of 2026-09-15), laptop Intel Core i9-11900H, 2026-09-15; the run took 4 min.

We apply the six decontamination methods to synthetic catalogues under four contamination
scenarios and compare the corrected field with the true one.
The figures and numbers below come from ``scripts/run_validation.py``; the JSON results are in
``docs/_static/results_validation/``.

Setup
-----

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Parameter
     - Value
   * - HEALPix NSIDE
     - 32, of which 8 070 pixels at :math:`|b| > 20^\circ`
   * - Templates
     - 3 Gaussian random fields (families 0, 1, 2 of
       :func:`~sys_mapping.maps.generate_systematic_maps`), unit variance on the footprint
   * - Galaxies
     - 50 per pixel on average, 401 000 to 407 000 per catalogue
   * - True field
     - lognormal, :math:`\sigma = 0.5`, :math:`C_\ell \propto (\ell+1)^{-2}`
   * - MCMC-add
     - exact analytic posterior
   * - MCMC-comb
     - NUTS, 4 chains of 1 000 warmup steps and 1 000 draws
   * - ISD calibration
     - :math:`\Delta\chi^2_{68}` per template from 100 uncontaminated realisations
   * - Seed
     - 42

The true field is :math:`\delta_g = \exp(G - \sigma^2/2) - 1` for a Gaussian field :math:`G`.
The four scenarios share one realisation of the field and of the templates and differ only in
the contamination:

.. math::

   \text{none: } \hat\delta_g = \delta_g, \qquad
   \text{additive: } \hat\delta_g = \delta_g + \textstyle\sum_i a_i t_i, \qquad
   \text{multiplicative: } \hat\delta_g = \delta_g\,(1 + \textstyle\sum_i b_i t_i),

and combined applies both terms. The amplitudes are drawn once from
:math:`\mathcal{N}(0, 0.1^2)`:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * -
     - t0
     - t1
     - t2
   * - :math:`a_i`
     - +0.0305
     - −0.1040
     - +0.0750
   * - :math:`b_i`
     - +0.0941
     - −0.1951
     - −0.1302

Methods
-------

All six run through :func:`~sys_mapping.regression.run_decontamination`.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Method
     - What it fits
   * - OLS
     - least squares of :math:`\hat\delta_g` on the templates; :math:`w = 1/(1 + \hat a\cdot t)`
   * - ElasticNet
     - L1 + L2 penalised regression, penalty by 5-fold cross-validation; same weight
   * - ISD-1, ISD-3
     - iterative marginal fits of degree 1 or 3 in one template at a time, 10 equal-occupancy
       bins, until :math:`\Delta\chi^2/\Delta\chi^2_{68} < 2`; the weight is the product of the
       per-step corrections
   * - MCMC-add
     - additive model; :math:`w = (1 + \delta_{\rm clean})/(1 + \hat\delta_g)` with
       :math:`\hat b = 0`
   * - MCMC-comb
     - combined model, same weight with the fitted :math:`\hat b`

The uncontaminated realisations give :math:`\Delta\chi^2_{68}` = 1.87, 1.40, 29.93 for ISD-1 and
4.50, 4.09, 48.94 for ISD-3 (t0, t1, t2).
The large-scale template t2 correlates with the clustered field by chance far more than t0 and t1.

Maps and distributions
----------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - none
     - additive
   * - .. figure:: _static/results_validation/map_none.png
          :width: 100%
     - .. figure:: _static/results_validation/map_additive.png
          :width: 100%
   * - multiplicative
     - combined
   * - .. figure:: _static/results_validation/map_multiplicative.png
          :width: 100%
     - .. figure:: _static/results_validation/map_combined.png
          :width: 100%

Each panel set shows the true field, the observed field and the field recovered by each method,
with its correlation :math:`r` with the truth.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - none
     - additive
   * - .. figure:: _static/results_validation/hist_none.png
          :width: 100%
     - .. figure:: _static/results_validation/hist_additive.png
          :width: 100%
   * - multiplicative
     - combined
   * - .. figure:: _static/results_validation/hist_multiplicative.png
          :width: 100%
     - .. figure:: _static/results_validation/hist_combined.png
          :width: 100%

Histograms of the pixel overdensity: true (black), observed (grey, dashed) and recovered.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - none
     - additive
   * - .. figure:: _static/results_validation/scatter_none.png
          :width: 100%
     - .. figure:: _static/results_validation/scatter_additive.png
          :width: 100%
   * - multiplicative
     - combined
   * - .. figure:: _static/results_validation/scatter_multiplicative.png
          :width: 100%
     - .. figure:: _static/results_validation/scatter_combined.png
          :width: 100%

Recovered against true overdensity per pixel, with the identity line.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - none
     - additive
   * - .. figure:: _static/results_validation/weights_none.png
          :width: 100%
     - .. figure:: _static/results_validation/weights_additive.png
          :width: 100%
   * - multiplicative
     - combined
   * - .. figure:: _static/results_validation/weights_multiplicative.png
          :width: 100%
     - .. figure:: _static/results_validation/weights_combined.png
          :width: 100%

Pixel weights of each method, on a common scale from 0.5 to 1.5.

Recovery
--------

We grade each correction by the rms of :math:`\delta_{\rm rec} - \delta_{\rm true}` and by the
Pearson correlation :math:`r` with the true field, over the unmasked pixels.

.. list-table::
   :header-rows: 1
   :widths: 16 11 10 11 10 11 10 11 10

   * - Method
     - none rms
     - r
     - additive rms
     - r
     - mult. rms
     - r
     - combined rms
     - r
   * - observed
     - 0.1474
     - 0.9577
     - 0.1905
     - 0.9322
     - 0.1879
     - 0.9353
     - 0.2244
     - 0.9130
   * - OLS
     - 0.1499
     - 0.9561
     - 0.1597
     - 0.9529
     - 0.1834
     - 0.9370
     - 0.1816
     - 0.9415
   * - ElasticNet
     - 0.1474
     - 0.9577
     - 0.1592
     - 0.9530
     - 0.1879
     - 0.9353
     - 0.1818
     - 0.9412
   * - ISD-1
     - 0.1474
     - 0.9577
     - 0.1584
     - 0.9530
     - 0.1879
     - 0.9353
     - 0.1837
     - 0.9398
   * - ISD-3
     - 0.1474
     - 0.9577
     - 0.1701
     - 0.9454
     - 0.1879
     - 0.9353
     - 0.1846
     - 0.9375
   * - MCMC-add
     - 0.1496
     - 0.9564
     - 0.1492
     - 0.9571
     - 0.1895
     - 0.9341
     - 0.1906
     - 0.9354
   * - MCMC-comb
     - 0.1504
     - 0.9557
     - 0.1498
     - 0.9565
     - 0.1609
     - 0.9493
     - 0.1593
     - 0.9512

Without contamination ElasticNet, ISD-1 and ISD-3 leave the field untouched: ElasticNet shrinks
every amplitude to zero and neither ISD takes a step. In the additive scenario MCMC-add and
MCMC-comb reach rms 0.149 to 0.150, OLS, ElasticNet and ISD-1 0.158 to 0.160, and ISD-3 0.170.
ISD-1 corrects t1, t0 and t2 in three steps; ISD-3 stops after t1 and t0, because the
contamination of t2 does not reach twice its :math:`\Delta\chi^2_{68}` of 48.9. In the
multiplicative scenario MCMC-comb lowers the rms from 0.188 to 0.161 and OLS to 0.183; ISD takes
no step, ElasticNet fits zero and MCMC-add reaches 0.190. In the combined scenario MCMC-comb reaches 0.159 against 0.182 to
0.191 for the other five.

.. figure:: _static/results_validation/summary_rms_delta_error.png
   :width: 80%
   :align: center

   Rms field error per method and scenario.

.. figure:: _static/results_validation/summary_corr_with_true.png
   :width: 80%
   :align: center

   Correlation with the true field per method and scenario.

Amplitude recovery
------------------

.. figure:: _static/results_validation/amplitude_recovery.png
   :width: 85%
   :align: center

   Recovered minus true amplitude per template for MCMC-add and MCMC-comb in the additive and
   combined scenarios. The error bars on :math:`a` are the mock-covariance sandwich standard
   deviation (:func:`~sys_mapping.covariance.mock_sandwich_covariance`) from the 100
   uncontaminated realisations; :math:`b` has none.

The sandwich standard deviation of :math:`\hat a` is 0.0080, 0.0068 and 0.0367 for t0, t1 and t2.
The MCMC-add posterior gives 0.0057 for all three, smaller by factors of 1.4, 1.2 and 6.5.
In the additive scenario the MCMC-add errors are +0.0011, +0.0121 and −0.0301, that is 0.1, 1.8
and 0.8 sandwich standard deviations. In the combined scenario MCMC-comb gives +0.0007, +0.0207
and −0.0208 on :math:`a` (0.1, 3.0 and 0.6 standard deviations) and −0.0026, +0.0408 and +0.0016
on :math:`b`, whose largest amplitude, :math:`b_1 = -0.195`, it recovers as −0.154.

Grading against the truth
-------------------------

A correction is graded here against the known true field. Correlating the corrected field with
the templates it fitted grades nothing: a least-squares residual is orthogonal to its regressors,
so that correlation is zero to rounding error whatever the correction left. On data,
:func:`~sys_mapping.diagnostics.residual_template_correlation_test` tests the corrected field
against a template the correction did not fit, calibrated on uncontaminated realisations put
through the same correction.

Template ranking
----------------

.. figure:: _static/results_validation/snr_ranking.png
   :width: 85%
   :align: center

   Template ranking in the combined scenario by the OLS :math:`|\hat\alpha_i|/\sigma_i`, the
   correlation :math:`|r(\hat\delta_g, t_i)|` and the peak cross-spectrum over the noise level.

In the combined scenario the OLS statistic is 5.1, 15.7 and 6.5 and the correlation 0.057, 0.172
and 0.073 for t0, t1 and t2, so both rank t1, the largest amplitude, first. Without contamination
the same statistics give 0.6, 0.6 and 4.0, and 0.007, 0.007 and 0.045; for the large-scale t2
the independent-pixel :math:`\sigma_i` is 6.5 times smaller than the sandwich error above.
The peak cross-spectrum ranks t2 first in every scenario, at 356 without contamination and 305
in the combined scenario.

Reproduce
---------

.. code-block:: bash

   python scripts/run_validation.py --nside 32 --n-sys 3 --n-mean 50 --seed 42 \
       --output-dir docs/_static/results_validation
