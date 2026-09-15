sys_mapping.diagnostics
=======================

Template significance, null tests, template ranking and footprint masking
sensitivity.

**Inputs:** Per-pixel weight array ``weights``, template maps ``delta_t``,
observed overdensity ``delta_g_obs``.

**Outputs:** Correlation arrays, p-values, SNR arrays, or dictionaries of
masking-level results.

* :func:`~sys_mapping.diagnostics.calibrated_template_significance` — each
  template's least-squares amplitude divided by its scatter across uncontaminated
  realisations, per-template p-values, and the family-wise p-value of the largest
  significance.
* :func:`~sys_mapping.diagnostics.residual_template_correlation_test` — correlation
  of the corrected density with each template, a :math:`\chi^2` over the null
  variance and its p-value from the realisations. For a template the correction
  fitted, the correlation is zero by construction.
* :func:`~sys_mapping.diagnostics.null_test_cross_correlations` — Pearson
  :math:`r(w, t_i)` with permutation p-values.
* :func:`~sys_mapping.diagnostics.snr_template_ranking` — three SNR
  estimators (``"template"``, ``"data"``, ``"peak"``) to rank which
  templates carry the most contaminating power.
* :func:`~sys_mapping.diagnostics.footprint_mask_diagnostics` — stability
  of fitted amplitudes under varying mask thresholds.
* :func:`~sys_mapping.diagnostics.isd_marginal_fit` — the per-template binned
  polynomial fit ISD is built from, returning both :math:`\Delta\chi^2` and the
  fitted coefficients.
* :func:`~sys_mapping.diagnostics.isd_template_significance` — the same statistic
  calibrated against contamination-free GLASS realisations, drawn per pixel by
  default (``draw="pixel"``).
* :func:`~sys_mapping.diagnostics.vet_templates_against_tracer` — rank-correlate
  each template against an external tracer of true structure (CMB lensing,
  Compton-*y*, weak-lensing convergence) and reject those that correlate.

**Key papers:**
`Ross et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.417.1350R/abstract>`_;
`Tanidis et al. 2026 <https://ui.adsabs.harvard.edu/abs/2026MNRAS.547ag537T/abstract>`_;
`Weaverdyck & Huterer 2021 <https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.5061W/abstract>`_;
`Rodríguez-Monroy et al. 2025 <https://ui.adsabs.harvard.edu/abs/2025arXiv250907943R/abstract>`_;
`Weaverdyck et al. 2026 <https://arxiv.org/abs/2601.14484>`_;
`Eggert & Leistedt 2023 <https://ui.adsabs.harvard.edu/abs/2023ApJS..265...30E/abstract>`_ — see also :doc:`../methods`.

.. automodule:: sys_mapping.diagnostics
   :members:
   :show-inheritance:
