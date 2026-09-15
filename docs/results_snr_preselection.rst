Results: SNR-based Template Pre-selection
=========================================

sys_mapping 1.4.0 (commit ``6f43dcc`` plus the uncommitted working-tree changes of 2026-09-15), laptop Intel Core i9-11900H, 2026-09-15; the run took 0.5 min.

Stage 1 ranks every candidate template by a fast statistic and keeps a short list; Stage 2 runs
the decontamination on that list. We test Stage 1 on a simulation in which two of 20 templates
are injected, with ``scripts/run_snr_preselection_demo.py``. Every number below is in
``docs/_static/results_snr_preselection/summary.json``.

The ranking statistics of :func:`~sys_mapping.diagnostics.snr_template_ranking` are:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Method
     - Statistic
   * - ``"data"``
     - Pearson :math:`|r|` between :math:`\delta_g` and the template
   * - ``"template"``
     - :math:`|\hat\alpha|/\sigma_{\hat\alpha}` of the one-template least-squares fit, with the
       independent-pixel :math:`\sigma_{\hat\alpha}`
   * - ``"isd"``
     - :math:`\Delta\chi^2 = \chi^2_{\rm null} - \chi^2_{\rm model}` of a polynomial fit to the
       binned relation between density and template value (Rodríguez-Monroy et al. 2025,
       Sec. IV.A.1); here degree 1 and 10 equal-width bins
   * - ``"peak"``
     - peak of the cross-spectrum :math:`C_\ell^{gt}` over the noise level (not used here)

:func:`~sys_mapping.diagnostics.isd_template_significance` turns :math:`\Delta\chi^2` into a
p-value by comparing it with uncontaminated GLASS realisations on the same footprint,
:math:`p = (N_{\ge} + 1)/(N_{\rm mocks} + 1)`.

The ISD statistic accumulates the per-bin variance in centred two-pass form. Under ``jax.jit``
the algebraically equal :math:`\langle\delta^2\rangle - \langle\delta\rangle^2` is contracted
into a fused multiply-add that returns :math:`\sim10^{-20}` instead of zero for a one-pixel bin.
The kernels therefore require :math:`N_\beta \ge 2` pixels per bin, a bin variance above
:math:`10^{-12}` times the variance of :math:`\delta_g`, and at least two surviving bins; they
agree with the NumPy fallback to :math:`4\times10^{-15}`.

Simulation
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Value
   * - Resolution
     - NSIDE 32, full sky (12 288 pixels)
   * - Galaxies
     - 15 000 000 (1 221 per pixel), one shell :math:`0 \le z \le 0.5`
   * - Clustering
     - GLASS lognormal field, parametric spectrum with ``cl_amplitude = 5e-4``
   * - Templates
     - 20 maps, four realisations of each of the families 0 to 4 of
       :func:`~sys_mapping.maps.generate_systematic_maps`
   * - Injected
     - T2 and T7, both family 2: :math:`\delta_g^{\rm obs} = \delta_g + a\,(t_2 + t_7)`
   * - Levels
     - low :math:`a = 0.02`, medium 0.05, high 0.10
   * - Null
     - 100 uncontaminated GLASS realisations with the same spectrum, seeds 17 to 116

The universe and the null share the parametric spectrum because the simulation has no data to
match; an analysis of data uses the sample's matched spectrum
(:func:`~sys_mapping.glass_mocks.load_matched_cl`).

.. figure:: /_static/results_snr_preselection/01_maps.png
   :width: 100 %
   :alt: HEALPix maps of the simulated universe and systematic templates

   Top: clean overdensity, the injected templates T2 and T7, and the noise template T0.
   Bottom: the overdensity at the medium level and the injected contamination alone.

Ranking
-------

.. figure:: /_static/results_snr_preselection/02_snr_ranking_data.png
   :width: 100 %
   :alt: SNR bar chart for the data cross-correlation method

   Pearson :math:`|r|` per template at the three levels; injected templates in red.

.. figure:: /_static/results_snr_preselection/02_snr_ranking_template.png
   :width: 100 %
   :alt: SNR bar chart for the OLS t-statistic method

   :math:`|\hat\alpha|/\sigma_{\hat\alpha}` with the independent-pixel error (bars) and with the
   error from 60 uncontaminated realisations (black ticks); dotted line at 3.

.. figure:: /_static/results_snr_preselection/02_snr_ranking_isd.png
   :width: 100 %
   :alt: SNR bar chart for the ISD Delta-chi2 method

   ISD :math:`\Delta\chi^2` per template at the three levels.

All three statistics rank T7 first, T2 second and the noise template T8 third at every level.

.. list-table:: Injected templates (T2, T7) and the largest noise template, per level.
   :header-rows: 1
   :widths: 22 26 26 26

   * - Statistic
     - low (0.02)
     - medium (0.05)
     - high (0.10)
   * - :math:`|r|`
     - 0.34, 0.38; 0.021
     - 0.58, 0.61; 0.035
     - 0.68, 0.69; 0.040
   * - :math:`|\hat\alpha|/\sigma`, independent pixels
     - 40.3, 46.0; 2.3
     - 79.5, 85.3; 3.9
     - 102.0, 106.4; 4.4
   * - :math:`|\hat\alpha|/\sigma`, realisations
     - 14.0, 12.8; 2.1
     - 36.5, 31.0; 4.2
     - 73.9, 61.4; 8.4
   * - ISD :math:`\Delta\chi^2`
     - 1 623, 2 130; 6.9
     - 6 335, 7 276; 17.5
     - 10 369, 11 210; 20.3

.. figure:: /_static/results_snr_preselection/03_detection_vs_amplitude.png
   :width: 80 %
   :alt: SNR of injected template 2 as a function of contamination amplitude

   Statistics of T2 against the injected amplitude :math:`a`; dotted lines mark the three levels.

For T2 at :math:`a = 0` the independent-pixel :math:`|\hat\alpha|/\sigma` is 2.99 and the
realisation-calibrated one 0.96. At :math:`a` = 0.01, 0.05 and 0.15 the calibrated value is 6.5,
36.5 and 111.4, and :math:`|r|` is 0.18, 0.58 and 0.70. :math:`\Delta\chi^2` rises from 6 at
:math:`a = 0` to 402 at 0.01 and 11 747 at 0.15.

ISD p-values
------------

.. figure:: /_static/results_snr_preselection/04_pvalues.png
   :width: 100 %
   :alt: Mock-based p-values from ISD significance test

   Mock-based p-values from 100 GLASS realisations; dashed line at :math:`p = 0.05`.

T2 and T7 reach the floor :math:`p = 1/101` at every level. The 18 noise templates have median
p-values of 0.66, 0.59 and 0.50 at the three levels. T8 has :math:`p = 0.099` at the low level
and 0.020 at the medium and high levels; no other noise template falls below 0.10.
The 68th percentile of the null :math:`\Delta\chi^2` is 10 to 15 for the four family-2 templates
and 0.8 to 2.7 for the others.

.. figure:: /_static/results_snr_preselection/05_pipeline_result.png
   :width: 60 %
   :alt: Detection summary table

   Whether both T2 and T7 are among the three highest-ranked templates, per statistic and level.

Both injected templates are in the top three for every statistic at every level.

Cost
----

.. figure:: /_static/results_snr_preselection/06_timing.png
   :width: 65 %
   :alt: Wallclock time per ranking method

   Best of five calls of :func:`~sys_mapping.diagnostics.snr_template_ranking`, 20 templates,
   12 288 pixels.

One ranking call takes 0.74 ms for ``"data"``, 0.96 ms for ``"template"`` and 3.3 ms for
``"isd"``. The 100-realisation null of :func:`~sys_mapping.diagnostics.isd_template_significance`,
drawn per pixel, takes 1 to 2 s per level.

Number of realisations
----------------------

.. figure:: /_static/results_snr_preselection/07_mock_convergence.png
   :width: 100 %
   :alt: Mock convergence of ISD p-values

   Left: p-values at the high level from the first :math:`N` of the 100 realisations, injected
   templates in red, with the floor :math:`1/(N+1)`. Right: the largest change of a noise-template
   p-value relative to :math:`N = 100`.

The p-values of T2 and T7 follow the floor :math:`1/(N+1)` from :math:`N = 5`. The largest change
of a noise-template p-value relative to :math:`N = 100` is 0.50 at :math:`N = 5`, 0.15 at 20, 0.062
at 50 and 0.035 at 75. The floor also sets the smallest reportable p-value: a threshold
:math:`\alpha` needs :math:`N \ge 1/\alpha - 1` realisations, 19 for 0.05 and 99 for 0.01.

Reproduce
---------

.. code-block:: bash

   python scripts/run_snr_preselection_demo.py

API reference
-------------

.. autofunction:: sys_mapping.diagnostics.snr_template_ranking
   :no-index:
.. autofunction:: sys_mapping.diagnostics.isd_template_significance
   :no-index:
.. autofunction:: sys_mapping.model_selection.snr_preselect
   :no-index:
.. autoclass:: sys_mapping.model_selection.SnrPreselectionResult
   :members:
   :no-index:
