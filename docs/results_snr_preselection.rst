Results: SNR-based Template Pre-selection
=========================================

Overview
--------

When tens or hundreds of imaging-systematic maps are available, running the
full Bayesian decontamination pipeline on all of them simultaneously is
computationally expensive and numerically ill-conditioned.  The SNR pre-selection
pipeline addresses this by inserting a fast Stage 1 before the full Stage 2
decontamination:

**Stage 1 — SNR ranking.**
All candidate templates are scored by their correlation with the observed galaxy
overdensity.  The top-:math:`K` templates are retained.

**Stage 2 — Full decontamination.**
The reduced template set is passed to the standard greedy forward selection or
Bayesian MCMC pipeline.

This document validates Stage 1 against a controlled simulation where the
injected templates are known a priori.

Three ranking statistics are implemented (see :mod:`sys_mapping.diagnostics`):

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Method key
     - Description
   * - ``"data"``
     - Pearson cross-correlation :math:`|r|` between the observed overdensity
       :math:`\delta_g` and each template :math:`\delta_t`.
   * - ``"template"``
     - OLS regression of :math:`\delta_g` on each template individually;
       SNR = :math:`|\hat\alpha| / \sigma_{\hat\alpha}` (absolute *t*-statistic).
   * - ``"isd"``
     - ISD :math:`\Delta\chi^2` contamination metric
       (Rodríguez-Monroy et al. 2025, arXiv:2509.07943, Sec. IV.A.1).
       Pixels are binned by template value; a polynomial is fit to the
       binned galaxy density; :math:`\Delta\chi^2 = \chi^2_\mathrm{null} -
       \chi^2_\mathrm{model}` measures how much the polynomial reduces the
       scatter.

A fourth optional statistic, ``"peak"``, ranks templates by the peak amplitude
of the cross-power spectrum :math:`C_\ell^{gT}`.

For a rigorous significance test, :func:`sys_mapping.diagnostics.isd_template_significance`
compares :math:`\Delta\chi^2_\mathrm{data}` against a distribution of
:math:`\Delta\chi^2_\mathrm{mock}` values computed from GLASS systematic-free
mocks on the same footprint, producing mock-based *p*-values.


Simulation setup
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Value
   * - HEALPix resolution
     - NSIDE = 32 (:math:`n_\mathrm{pix}` = 12 288)
   * - Galaxy count
     - 15 000 000 (full sky)
   * - Redshift shell
     - :math:`0 \le z \le 0.5` (single tophat bin)
   * - Systematic templates
     - 20 synthetic maps; families 0–4 generated 4× with independent seeds
       (see :func:`sys_mapping.maps.generate_systematic_maps`)
   * - Injected templates
     - Templates **2** and **7** (★ in figures) — two realisations of family 2
   * - Contamination model
     - Additive: :math:`\delta_g^\mathrm{obs} = \delta_g^\mathrm{clean}
       + a \,\delta_{t_2} + a \,\delta_{t_7}`
   * - Contamination levels
     - low :math:`(a=0.02)`, medium :math:`(a=0.05)`, high :math:`(a=0.10)`
   * - GLASS mocks for ISD
     - 100 systematic-free full-sky mocks on the same footprint

The figure-generating script is ``scripts/run_snr_preselection_demo.py``.


HEALPix maps
------------

The panel below shows the simulated galaxy overdensity, representative template
maps, and the contaminated field at medium level.

.. figure:: /_static/results_snr_preselection/01_maps.png
   :width: 100 %
   :alt: HEALPix maps of the simulated universe and systematic templates

   **Left to right, top row:** Clean galaxy overdensity :math:`\delta_g`,
   two injected templates (T2, T7), a representative noise template.
   **Bottom row:** Contaminated overdensity at medium level and the
   residual contamination signal.


SNR ranking
-----------

The bar charts below show the SNR score assigned to each template by each
ranking method at the three contamination levels.  The two injected
templates (★) are coloured in coral red; noise templates in grey.

Data cross-correlation method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: /_static/results_snr_preselection/02_snr_ranking_data.png
   :width: 100 %
   :alt: SNR bar chart for the data cross-correlation method

   **Data cross-correlation** :math:`|r|`.  Both injected templates rank
   first and second at all three contamination levels.  Noise templates
   have scores consistent with zero within statistical fluctuations.

OLS *t*-statistic method
^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: /_static/results_snr_preselection/02_snr_ranking_template.png
   :width: 100 %
   :alt: SNR bar chart for the OLS t-statistic method

   **OLS** :math:`|\hat\alpha|/\sigma_{\hat\alpha}`.  The signal-to-noise
   ratio is generally sharper than the Pearson correlation because the
   template standard deviation is factored into the denominator.

ISD :math:`\Delta\chi^2` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: /_static/results_snr_preselection/02_snr_ranking_isd.png
   :width: 100 %
   :alt: SNR bar chart for the ISD Delta-chi2 method

   **ISD** :math:`\Delta\chi^2` (Rodríguez-Monroy et al. 2025).
   The statistic grows quadratically with amplitude, making it highly
   sensitive at medium and high contamination but potentially
   less sensitive than the correlation at low levels.


Detection sensitivity vs amplitude
------------------------------------

.. figure:: /_static/results_snr_preselection/03_detection_vs_amplitude.png
   :width: 80 %
   :alt: SNR of injected template 2 as a function of contamination amplitude

   SNR assigned to injected template 2 as a function of contamination
   amplitude :math:`a` for the three methods.  Vertical dotted lines mark
   the three canonical levels.  All methods increase monotonically with
   amplitude.  The data and template methods are approximately linear in
   :math:`a`; the ISD :math:`\Delta\chi^2` grows roughly as :math:`a^2`.


ISD mock-based *p*-values
--------------------------

.. figure:: /_static/results_snr_preselection/04_pvalues.png
   :width: 100 %
   :alt: Mock-based p-values from ISD significance test

   Mock-based *p*-values from :func:`sys_mapping.diagnostics.isd_template_significance`
   with :math:`N_\mathrm{mocks} = 100` GLASS systematic-free mocks.
   The red dashed line marks the conventional :math:`p = 0.05` threshold.
   At all three levels, both injected templates (T2, T7) are significantly
   detected (:math:`p \le 1/N_\mathrm{mocks}`), while all 18 noise templates
   scatter around :math:`p \sim 0.5`.


Full pipeline: pre-selection then decontamination
--------------------------------------------------

.. figure:: /_static/results_snr_preselection/05_pipeline_result.png
   :width: 60 %
   :alt: Detection summary table

   Detection summary: green cell = both injected templates T2 and T7 appear
   in the top-3 ranking (out of 20); grey cell = at least one is missed.

All three methods successfully recover both injected templates in the top-3
at every contamination level.  In practice, selecting the top :math:`K`
templates (with :math:`K` chosen conservatively, e.g. :math:`K = N/2`
of the initial pool) and then running greedy forward selection on the
reduced set is a robust and computationally efficient strategy.


Computational cost
------------------

The ranking step is fast regardless of the galaxy count; it operates on
pixelised maps (here: 12 288 values) and 20 templates.

.. figure:: /_static/results_snr_preselection/06_timing.png
   :width: 65 %
   :alt: Wallclock time per ranking method

   Best-of-5 wallclock time for a single call to
   :func:`~sys_mapping.diagnostics.snr_template_ranking` with 20 templates on
   NSIDE = 32.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Method
     - Typical time
     - Notes
   * - ``"data"``
     - < 1 ms
     - Pearson correlation; JAX batched matrix multiply — :math:`\mathcal{O}(n_\mathrm{pix} \times n_\mathrm{templates})`
   * - ``"template"``
     - < 1 ms
     - Per-template OLS *t*-statistic; JAX vmap over templates
   * - ``"isd"``
     - 1–10 ms
     - JAX vmap over templates, fixed-size bins, analytic linear regression;
       falls back to NumPy for ``poly_order > 1`` or ``fracdet`` weighting

The dominant runtime cost in practice is the GLASS mock generation for
:func:`~sys_mapping.diagnostics.isd_template_significance`: each of the
:math:`N_\mathrm{mocks}` systematic-free mocks requires drawing galaxy
positions from the pixelised density field (:math:`\mathcal{O}(N_\mathrm{total})`).
At 15 M galaxies and ``rand_factor=2``, each mock takes roughly 1–5 s,
making the total ISD significance cost
:math:`\sim N_\mathrm{mocks} \times 3\,\mathrm{s}` on a modern workstation.


Choosing the number of mocks for ISD significance
--------------------------------------------------

The mock-based *p*-value estimate has a finite resolution of
:math:`1/(N_\mathrm{mocks}+1)` and a standard error that scales as
:math:`\sim 1/\sqrt{N_\mathrm{mocks}}` for noise-template p-values near 0.5.
The figure below quantifies how many mocks are needed for the noise-template
p-values to converge.

.. figure:: /_static/results_snr_preselection/07_mock_convergence.png
   :width: 100 %
   :alt: Mock convergence of ISD p-values

   **Left:** p-values for all 20 templates as a function of :math:`N_\mathrm{mocks}`
   (reusing the 100-mock run at the high contamination level by subsampling the
   mock :math:`\Delta\chi^2` matrix).
   Coral lines = injected templates (T2, T7); grey lines = 18 noise templates;
   red dashed = :math:`p=0.05` threshold; dotted black = theoretical lower
   bound :math:`1/(N+1)`.
   **Right:** convergence metric — maximum absolute change of any noise-template
   p-value relative to the :math:`N=100` reference, as a function of
   :math:`N_\mathrm{mocks}`.  The vertical green dashed line marks the first
   :math:`N` where this change falls below 0.05.

The injected templates saturate at the minimum attainable p-value
:math:`1/(N+1)` from the very first few mocks, confirming that even
:math:`N_\mathrm{mocks}=5` is sufficient to *detect* a strongly contaminating
template.  The noise-template p-values, however, require more mocks to stabilise:
the convergence metric falls below 0.05 only around :math:`N \approx 50\text{–}100`.

**Rule of thumb:**

.. math::

   N_\mathrm{mocks} \;\ge\; \max\!\left(20,\; \left\lceil \frac{5}{\alpha} \right\rceil\right)

where :math:`\alpha` is the target significance level.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Target :math:`\alpha`
     - Minimum :math:`N_\mathrm{mocks}`
     - Notes
   * - 0.10
     - 50
     - Fast pre-screening
   * - 0.05
     - 100
     - Standard significance level
   * - 0.01
     - 500
     - High-confidence selection


Conclusion
----------

* **All three SNR ranking methods** (data cross-correlation, OLS *t*-statistic,
  ISD :math:`\Delta\chi^2`) correctly identify both injected templates as the
  highest-ranking candidates at all tested contamination amplitudes
  (:math:`a \ge 0.02`) with 15 M galaxies and 20 templates.

* **The data and template methods** are approximately linear in amplitude,
  reliable from the lowest tested level, and execute in < 1 ms.  They are the
  fastest route to a short-list.

* **The ISD** :math:`\Delta\chi^2` **method** follows Rodríguez-Monroy et al.
  (2025).  Its mock-based p-value provides a rigorous significance test without
  assumptions about the null distribution.  The mock generation is the dominant
  cost; :math:`N_\mathrm{mocks} = 100` is sufficient for a 5 % significance
  threshold.

* **Recommended workflow:**

  1. Run ``method="data"`` on all candidate templates to build a short-list
     (< 1 ms; free).
  2. Apply :func:`~sys_mapping.diagnostics.isd_template_significance` on the
     short-list with :math:`N_\mathrm{mocks} \ge 100` to obtain rigorous
     p-values (minutes).
  3. Pass templates with :math:`p < 0.05` to
     :func:`~sys_mapping.model_selection.greedy_forward_select` or the Bayesian
     MCMC pipeline.

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
