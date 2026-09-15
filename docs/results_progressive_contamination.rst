Progressive Template Contamination Study
=========================================

sys_mapping 1.4.0 (commit ``6f43dcc`` plus the uncommitted working-tree changes of 2026-09-15), GRICAD dahu campaign 20260915r, cell ``progressive`` (node dahu128, Intel Xeon Gold 5218, 8 cores), 2026-09-15; the run took 7 min.

We contaminate the first :math:`k` of four templates and measure how often a per-template S/N
cut detects the contaminated templates and how often the likelihood ratio test selects the right
model (Figs. 1–3, Table 1).

Setup
-----

``scripts/run_mock_analysis_progressive.py`` uses the mock generator of
:doc:`results_mock_analysis` at NSIDE 32 with templates ``synth_0`` … ``synth_3``.
The first :math:`k \in \{1, 2, 3\}` templates carry :math:`a_i \sim \mathcal{N}(0, 0.15)` (additive
mode), :math:`b_i \sim \mathcal{N}(0, 0.15)` (multiplicative mode) or both (combined mode); each of
the nine cells has 5 mocks.
The S/N is :math:`|\hat a_i|/\sigma_{a_i}` from the MCMC-add (analytic) posterior and
:math:`|\hat b_i|/\sigma_{b_i}` from the MCMC-comb (NUTS) posterior, and a template is detected
when the S/N of the injected amplitude exceeds 2 (either S/N in combined mode).
TP is the detected fraction of the :math:`k` contaminated templates, FP that of the :math:`4-k`
clean ones.
:math:`\lambda_{\rm LR}` is taken between the additive and combined maxima, and the LRT decision,
at a Wilks :math:`\chi^2(4)` p-value of 0.05 with no mock null, is correct when it keeps the
additive model in additive mode and rejects it otherwise; on a correlated field that p-value is too
small (:ref:`lrt-methods`).

Results
-------

TP was 0.87–1.00 in every cell and FP 0.20–0.90 (Table 1, Fig. 2).
The LRT decision was correct in all multiplicative and combined mocks and in 1, 0 and 1 of 5
additive mocks for :math:`k = 1, 2, 3`, with median :math:`\lambda_{\rm LR}` of 28–58 in
additive cells and 31–987 in the others (Fig. 3); the smallest :math:`\lambda_{\rm LR}` was 3.3.
NUTS converged in every mock (:math:`\hat R \le 1.001`, no divergent transitions).
MCMC-comb took 8.3 s per mock at the median and 13.6 s at most, MCMC-add 0.10 s.

.. csv-table:: Table 1. Mean TP rate, FP rate, LRT correct-decision rate and median :math:`\lambda_{\rm LR}` per cell (``progressive_summary.csv``).
   :file: _static/results_progressive_contamination/progressive_summary.csv
   :header-rows: 1

.. figure:: _static/results_progressive_contamination/progressive_snr_grid.png
   :width: 95%
   :align: center
   :alt: Per-template S/N for each k and contamination mode

   Fig. 1. Mean ± std over 5 mocks of :math:`|\hat a_i|/\sigma_{a_i}` (blue) and
   :math:`|\hat b_i|/\sigma_{b_i}` (orange) per template; the red dotted line separates
   contaminated from clean templates and the dashed line is S/N = 2.

.. figure:: _static/results_progressive_contamination/progressive_detection_rates.png
   :width: 90%
   :align: center
   :alt: Heatmaps of TP rate, FP rate and LRT correct-decision rate

   Fig. 2. TP rate (left), FP rate (centre) and LRT correct-decision rate (right) per cell.

.. figure:: _static/results_progressive_contamination/progressive_lrt_summary.png
   :width: 70%
   :align: center
   :alt: Distribution of the likelihood-ratio statistic per cell

   Fig. 3. :math:`\lambda_{\rm LR}` per cell over 5 mocks, between the additive and combined maxima; blue cells
   have a correct-decision rate of at least 0.5.

Reproduce
---------

.. code-block:: bash

   python scripts/run_mock_analysis_progressive.py --nside 32 --n-sys 4 --n-mocks-per-case 5 \
       --snr-threshold 2.0 --sigma 0.15 \
       --output-dir docs/_static/results_progressive_contamination/

This writes one JSON per mock, ``progressive_results.csv`` (read by
``scripts/analyze_detectability_law.py``), ``progressive_summary.csv`` and the three figures.
