sys_mapping.regression
======================

Regression-based systematic decontamination: ElasticNet and Iterative
Systematics Decontamination (ISD).

**Inputs:** Observed overdensity ``delta_g_obs`` (shape ``(n_pix,)``),
template maps ``delta_t`` (shape ``(n_sys, n_pix)``).

**Outputs:** Amplitude vector ``alpha_hat``, per-pixel weight array
``weights``, and diagnostic metadata.

* :func:`~sys_mapping.regression.elasticnet_contamination_fit` — L1+L2
  penalised regression; requires ``scikit-learn >= 1.3``
  (``pip install sys_mapping[regression]``).
* :func:`~sys_mapping.regression.iterative_systematics_decontamination` —
  polynomial OLS expansion, no extra dependencies.
* :func:`~sys_mapping.regression.method_comparison` — run multiple methods
  and return a unified result dict.

**Key papers:**
`Weaverdyck & Huterer 2021 <https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.5061W/abstract>`_;
`Rodríguez-Monroy et al. 2025 <https://ui.adsabs.harvard.edu/abs/2025arXiv250907943R/abstract>`_ — see also :doc:`../methods`.

.. automodule:: sys_mapping.regression
   :members:
   :show-inheritance:
