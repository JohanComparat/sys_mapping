Null test — bin 1
============================================================

**50 uncontaminated simulation realizations.**
No systematics are injected; the pipeline should detect nothing.

MCMC-comb verdict: **FAILS (FPR > 5%)**
(FPR at 3σ = 92.0%)

OLS verdict: **FAILS (FPR > 5%)**
(FPR at 3σ = 96.0%)

Summary statistics
------------------

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Metric
     - MCMC-comb
     - OLS
   * - SNR mean (all templates)
     - 1.4388
     - 1.8030
   * - SNR std (all templates)
     - 1.9017
     - 2.1930
   * - False positive rate (3σ)
     - 92.0%
     - 96.0%
   * - 2PCF ratio @ 10'
     - 0.9442 ± 0.0284
     - —
   * - 2PCF ratio @ 200'
     - -1.3462 ± 14.4443
     - —
   * - Mean \|w − 1\|
     - 0.1659
     - —

Figures
-------

.. figure:: /_static/results_sim_null_test/BIN_1/01_snr_distribution.png
   :width: 90%

   SNR distribution across all templates and sims.
   Reference N(0,1) overplotted (dashed). A well-calibrated method
   has mean ≈ 0 and std ≈ 1.

.. figure:: /_static/results_sim_null_test/BIN_1/02_max_snr_per_sim.png
   :width: 90%

   Max \|SNR\| per simulation (sorted). Simulations above the dashed
   3σ line are false detections.

.. figure:: /_static/results_sim_null_test/BIN_1/03_twopcf_ratios.png
   :width: 90%

   Two-point function correction ratios (w_corr / w_obs) at 10' and
   200'. Ideal value: 1.0 — no correction needed.

.. figure:: /_static/results_sim_null_test/BIN_1/04_weight_deviation.png
   :width: 90%

   Distribution of mean \|w − 1\| across sims. Values near 0 confirm
   no spurious weight corrections are applied.
