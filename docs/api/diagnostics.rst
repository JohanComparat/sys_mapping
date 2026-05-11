sys_mapping.diagnostics
=======================

Post-correction diagnostics: null tests, SNR-based template ranking,
and footprint masking sensitivity analysis.

**Inputs:** Per-pixel weight array ``weights``, template maps ``delta_t``,
observed overdensity ``delta_g_obs``.

**Outputs:** Correlation arrays, p-values, SNR arrays, or dictionaries of
masking-level results.

* :func:`~sys_mapping.diagnostics.null_test_cross_correlations` — Pearson
  :math:`r(w, t_i)` with permutation p-values; should be near zero after
  correction.
* :func:`~sys_mapping.diagnostics.snr_template_ranking` — three SNR
  estimators (``"template"``, ``"data"``, ``"peak"``) to rank which
  templates carry the most contaminating power.
* :func:`~sys_mapping.diagnostics.footprint_mask_diagnostics` — stability
  of fitted amplitudes under varying mask thresholds.

**Key papers:**
`Ross et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.417.1350R/abstract>`_;
`Tanidis et al. 2026 <https://ui.adsabs.harvard.edu/abs/2026MNRAS.547ag537T/abstract>`_;
`Weaverdyck & Huterer 2021 <https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.5061W/abstract>`_;
`Rodríguez-Monroy et al. 2025 <https://ui.adsabs.harvard.edu/abs/2025arXiv250907943R/abstract>`_ — see also :doc:`../methods`.

.. automodule:: sys_mapping.diagnostics
   :members:
   :show-inheritance:
