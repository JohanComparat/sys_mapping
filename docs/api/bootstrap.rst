sys_mapping.bootstrap
=====================

Spatial covariance estimation via block bootstrap and jack-knife resampling.

Both estimators divide the survey footprint into :math:`K` equal-area patches
using HEALPix coarsening, capturing spatial correlations that pixel-level
bootstrap misses.

* :func:`~sys_mapping.bootstrap.block_bootstrap_variance` — resamples patches
  with replacement (:math:`B` times) to estimate the variance of any
  user-supplied estimator.
* :func:`~sys_mapping.bootstrap.jackknife_covariance` — leave-one-patch-out
  deterministic estimator with prefactor :math:`(K-1)/K`.

**Key papers:**
`Ross et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.417.1350R/abstract>`_;
`Ho et al. 2012 <https://ui.adsabs.harvard.edu/abs/2012ApJ...761...14H/abstract>`_;
`Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_, Sec. 6.2 — see also :doc:`../methods`.

.. automodule:: sys_mapping.bootstrap
   :members:
   :show-inheritance:
