Synthetic mock analysis
=======================

sys_mapping 1.4.0 (commit ``6f43dcc`` plus the uncommitted working-tree changes of 2026-09-15),
GRICAD dahu campaign 20260915r, cell ``mockanalysis`` (node dahu153, Intel Xeon Gold 5318Y,
8 cores), 2026-09-15; the run took 61 min.

We fit 100 synthetic mocks of known contamination with the six methods of
``sm.run_decontamination`` and report the recovery of :math:`a` (Fig. 1, Table 1),
:math:`\lambda_{\rm LR}` (Fig. 2), the recovery of :math:`b` (Fig. 3) and of the scatter (Fig. 4).

Setup
-----

``scripts/run_mock_analysis.py --synthetic`` draws a Gaussian field :math:`G` with
:math:`C_\ell \propto (\ell+1)^{-2}` and :math:`\sigma_G = 0.5`, forms
:math:`\delta_g^{\rm true} = e^{G-\sigma_G^2/2} - 1`, masks latitudes within 20° of the Galactic
plane and contaminates it as

.. math::

   \delta_g^{\rm obs} = \delta_g^{\rm true}\bigl(1 + \textstyle\sum_i b_i t_i\bigr) + \sum_i a_i t_i,
   \qquad a_i, b_i \sim \mathcal{N}(0, 0.10).

Galaxy counts are Poisson with mean :math:`\bar n (1 + \delta_g^{\rm obs})`, :math:`\bar n = 30`,
and randoms number :math:`8\bar n` per unmasked pixel.
The templates ``synth_0`` … ``synth_2`` are ``sm.generate_systematic_maps`` families 0–2,
:math:`C_\ell \propto e^{-\ell/500}`, :math:`e^{-(\ell/250)^2}` and :math:`(\ell+1)^{-2}`, at unit variance.
MCMC-add is the analytic posterior and MCMC-comb NUTS (4 chains of 1000 warm-up and 1000 draws);
ISD stops on a :math:`\Delta\chi^2_{68}` calibrated per template and order on 50 uncontaminated
mocks from the same generator.
:math:`\lambda_{\rm LR}` is taken between the additive and combined maxima (``sm.lrt_from_maxima``)
and the additive model is rejected when the :math:`\chi^2(3)` (Wilks) p-value is below 0.05, with
no mock null; on a correlated field that p-value is too small (:ref:`lrt-methods`).

Results
-------

At NSIDE 64 each mock has 32 512 unmasked pixels and 972 318 ± 33 386 galaxies (range
886 194–1 074 394).
The calibrated :math:`\Delta\chi^2_{68}` (``isd_chi2_68.json``) was 2.12, 1.17 and 230.4 for ISD-1
and 5.03, 3.51 and 270.8 for ISD-3 on ``synth_0``, ``synth_1`` and ``synth_2``; both ISD variants
stopped on this threshold in all 100 mocks, after a median of 3 steps (range 1–6).

.. figure:: _static/results_mock_analysis/mock_parameter_recovery_all_methods.png
   :width: 100%
   :align: center
   :alt: Additive parameter bias per template for the six methods

   Fig. 1. :math:`\hat a_i - a_i^{\rm true}` per template for the six methods over 100 mocks;
   boxes are the inter-quartile range, whiskers 1.5 IQR.

.. list-table:: Table 1. Recovery of :math:`a` over 100 mocks: MAD of :math:`\hat a_i - a_i^{\rm true}` over the three templates, rms per template, and median time per mock on 8 cores.
   :header-rows: 1
   :widths: 20 14 14 14 14 24

   * - Method
     - MAD
     - rms ``synth_0``
     - rms ``synth_1``
     - rms ``synth_2``
     - Median time (s)
   * - OLS
     - 0.0059
     - 0.0066
     - 0.0053
     - 0.0352
     - 0.0009
   * - ElasticNet
     - 0.0073
     - 0.0095
     - 0.0077
     - 0.0353
     - 0.19
   * - ISD-1
     - 0.0070
     - 0.0077
     - 0.0065
     - 0.0384
     - 0.17
   * - ISD-3
     - 0.0075
     - 0.0082
     - 0.0070
     - 0.0396
     - 0.15
   * - MCMC-add
     - 0.0059
     - 0.0066
     - 0.0053
     - 0.0352
     - 0.079
   * - MCMC-comb
     - 0.0056
     - 0.0067
     - 0.0054
     - 0.0323
     - 31.6

The median posterior standard deviation of :math:`\hat a_i` was 0.0031 for MCMC-add and 0.0029
for MCMC-comb.
For MCMC-comb, NUTS gave :math:`\hat R \le 1.001`, an effective sample size of 2 983–6 082 and no
divergences in 99 mocks; in mock 45, :math:`\hat R = 6.1` and the effective sample size was 2.

All 100 mocks rejected the additive model; :math:`\lambda_{\rm LR}` had median 1 158,
16th–84th percentiles 389–2 734, minimum 21.0 and maximum 6 152 (Fig. 2).
Mock 98 was re-fitted on the laptop with the maximiser of sys_mapping 1.4.0 as released, whose
L-BFGS runs keep every pixel's efficiency :math:`1 + b\cdot t` positive; the campaign snapshot
had stopped at :math:`b = 0` and returned :math:`\lambda_{\rm LR} = 0` for this mock.
Its other results agree with the campaign run to the NUTS sampling noise.

.. figure:: _static/results_mock_analysis/mock_lrt_statistics.png
   :width: 70%
   :align: center
   :alt: Histogram of the likelihood-ratio statistic over 100 mocks

   Fig. 2. :math:`\lambda_{\rm LR}` over 100 mocks.

For MCMC-comb the MAD of :math:`\hat b_i - b_i^{\rm true}` was 0.0154, with rms 0.020, 0.022 and
0.038 per template and a median posterior standard deviation of 0.0037 (Fig. 3); with
:math:`b_i = 0` the error is :math:`|b_i^{\rm true}|`, of median 0.064.

.. figure:: _static/results_mock_analysis/mock_b_parameter_recovery.png
   :width: 65%
   :align: center
   :alt: Multiplicative parameter bias per template for MCMC-comb

   Fig. 3. :math:`\hat b_i - b_i^{\rm true}` per template for MCMC-comb over 100 mocks.

The scatter :math:`\sigma` of the Gaussian likelihood is compared with
:math:`\sigma_{\rm eff} = \sqrt{e^{\sigma_G^2} - 1 + 1/\bar n} = 0.563`.
We measured :math:`\langle\hat\sigma\rangle = 0.560 \pm 0.028` for MCMC-add and
:math:`0.554 \pm 0.027` for MCMC-comb (Fig. 4).

.. figure:: _static/results_mock_analysis/mock_sigma_recovery.png
   :width: 90%
   :align: center
   :alt: Histograms of the recovered scatter for MCMC-add and MCMC-comb

   Fig. 4. :math:`\hat\sigma` over 100 mocks for MCMC-add (left) and MCMC-comb (right); the dashed
   line is :math:`\sigma_{\rm eff}`.

Reproduce
---------

.. code-block:: bash

   python scripts/run_mock_analysis.py --synthetic --n-mocks 100 --n-sys 3 --nside 64 \
       --output-dir results/mock_analysis_100
   bash scripts/_post_mock100.sh results/mock_analysis_100

The first command writes one JSON per mock (reused with ``--resume``), ``mock_results.csv``,
``isd_chi2_68.json`` and the four figures; the second copies all but the JSON to
``docs/_static/results_mock_analysis/``.
