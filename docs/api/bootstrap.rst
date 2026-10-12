sys_mapping.bootstrap
=====================

``sys_mapping.bootstrap`` estimates spatial variances and covariances by block bootstrap and jackknife resampling.
Both estimators divide the survey footprint into :math:`K` equal-area patches by HEALPix coarsening, which captures the spatial correlations a pixel-level bootstrap misses.

* :func:`~sys_mapping.bootstrap.block_bootstrap_variance`: resamples patches with replacement (:math:`B` times) to estimate the variance of any user-supplied estimator.
* :func:`~sys_mapping.bootstrap.jackknife_covariance`: the deterministic leave-one-patch-out estimator, with prefactor :math:`(K-1)/K`.
* :func:`~sys_mapping.bootstrap.assign_spatial_patches`: the patch labelling itself, exposed so the same partition can serve as cross-validation groups in
  :func:`~sys_mapping.regression.elasticnet_contamination_fit` and as jackknife patches in :func:`~sys_mapping.diagnostics.vet_templates_against_tracer`.

Key papers:
`Ross et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.417.1350R/abstract>`_;
`Ho et al. 2012 <https://ui.adsabs.harvard.edu/abs/2012ApJ...761...14H/abstract>`_;
`Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_, Sec. 6.2.
See also :doc:`../methods`.

.. automodule:: sys_mapping.bootstrap
   :members:
   :show-inheritance:
