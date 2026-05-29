Comparison — bin 6
============================================================

**50 matched simulation pairs (contaminated vs uncontaminated).**

Contamination **detected** (detection rate significantly exceeds FPR).

- Detection rate (contaminated, 3σ): **100.0%**
- False positive rate (uncontaminated, 3σ): **80.0%**
- Mean signal excess: **1.09 ± 1.00** null-σ
- Most consistently detected template: **ZodiacalLight_VIS**

Summary statistics
------------------

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Metric
     - Contaminated
     - Uncontaminated
   * - Detection rate (3σ)
     - 100.0%
     - 80.0% (FPR)
   * - 2PCF ratio @ 10'
     - 0.7776
     - 0.9590
   * - 2PCF ratio @ 200'
     - 0.6774
     - 0.3125

Figures
-------

.. figure:: /_static/results_sim_comparison/BIN_6/01_snr_comparison.png
   :width: 90%

   Violin plot of max \|SNR\| per sim. Contaminated sims (right) should
   show systematically higher values than uncontaminated (left).

.. figure:: /_static/results_sim_comparison/BIN_6/02_signal_excess.png
   :width: 90%

   Signal excess in units of the null distribution's std. Values > 3
   indicate clear contamination detection.

.. figure:: /_static/results_sim_comparison/BIN_6/03_detection_rate.png
   :width: 60%

   Detection rate vs false positive rate at 3σ.

.. figure:: /_static/results_sim_comparison/BIN_6/04_twopcf_comparison.png
   :width: 90%

   Two-point function correction ratios. Deviation from 1.0 is larger
   for contaminated sims, confirming the pipeline finds and corrects
   the injected systematic.

.. figure:: /_static/results_sim_comparison/BIN_6/05_top_templates.png
   :width: 90%

   Templates most frequently identified as the dominant systematic
   across the 50 contaminated realizations.
