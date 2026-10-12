sys_mapping.regression
======================

``sys_mapping.regression`` runs the regression-based decontamination methods: ElasticNet and Iterative Systematics Decontamination (ISD).
It takes the observed overdensity ``delta_g_obs`` (shape ``(n_pix,)``) and the template maps ``delta_t`` (shape ``(n_sys, n_pix)``),
and returns the amplitude vector ``alpha_hat``, the per-pixel weights ``weights``, and diagnostic metadata.

* :func:`~sys_mapping.regression.elasticnet_contamination_fit`: L1+L2 penalised regression;
  it requires ``scikit-learn >= 1.3`` (``pip install sys_mapping[regression]``).
* :func:`~sys_mapping.regression.iterative_systematics_decontamination`: marginal binned polynomial fits,
  greedy single-template weighting and a mock-calibrated stopping rule.
  It returns a :class:`~sys_mapping.regression.ISDResult`.
* :func:`~sys_mapping.regression.polynomial_ols_decontamination`: a joint least-squares fit on a multivariate polynomial basis of the templates.
* :func:`~sys_mapping.regression.run_decontamination`: the full pipeline on one field,
  with optional template pre-selection, the least-squares, ElasticNet, ISD and MCMC methods, and the weights and posterior summaries in one dict.
* :func:`~sys_mapping.regression.method_comparison`: several methods on one field, returned in a common dict.

Key papers:
`Weaverdyck & Huterer 2021 <https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.5061W/abstract>`_;
`Rodríguez-Monroy et al. 2025 <https://ui.adsabs.harvard.edu/abs/2025arXiv250907943R/abstract>`_;
`Weaverdyck et al. 2026 <https://arxiv.org/abs/2601.14484>`_.
See also :doc:`../methods`.

.. automodule:: sys_mapping.regression
   :members:
   :show-inheritance:
