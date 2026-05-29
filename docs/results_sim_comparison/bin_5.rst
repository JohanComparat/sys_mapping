Comparison — bin 5
============================================================

**50 matched simulation pairs (contaminated vs uncontaminated).**

Contamination **detected** (detection rate significantly exceeds FPR).

- Detection rate (contaminated, 3σ): **100.0%**
- False positive rate (uncontaminated, 3σ): **76.0%**
- Mean signal excess: **2.82 ± 1.49** null-σ
- Most consistently detected template: **GalacticExtinction_VIS**

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
     - 76.0% (FPR)
   * - 2PCF ratio @ 10'
     - 0.7000
     - 0.9493
   * - 2PCF ratio @ 200'
     - 0.4271
     - 0.7083

Figures
-------

.. figure:: /_static/results_sim_comparison/BIN_5/01_snr_comparison.png
   :width: 90%

   Violin plot of max \|SNR\| per sim. Contaminated sims (right) should
   show systematically higher values than uncontaminated (left).

.. figure:: /_static/results_sim_comparison/BIN_5/02_signal_excess.png
   :width: 90%

   Signal excess in units of the null distribution's std. Values > 3
   indicate clear contamination detection.

.. figure:: /_static/results_sim_comparison/BIN_5/03_detection_rate.png
   :width: 60%

   Detection rate vs false positive rate at 3σ.

.. figure:: /_static/results_sim_comparison/BIN_5/04_twopcf_comparison.png
   :width: 90%

   Two-point function correction ratios. Deviation from 1.0 is larger
   for contaminated sims, confirming the pipeline finds and corrects
   the injected systematic.

.. figure:: /_static/results_sim_comparison/BIN_5/05_top_templates.png
   :width: 90%

   Templates most frequently identified as the dominant systematic
   across the 50 contaminated realizations.
