sys_mapping.inference
=====================

Posterior sampling, posterior summaries and refinement to the likelihood maximum.

* :func:`~sys_mapping.inference.run_additive_analytic` — exact draws from the
  Normal–Inverse-Gamma posterior of the additive Gaussian model (MCMC-add).
* :func:`~sys_mapping.nuts.run_nuts` (in :mod:`sys_mapping.nuts`) — NUTS for the
  combined, multiplicative and skew-normal models (MCMC-comb).
* :func:`~sys_mapping.inference.run_mcmc` — the emcee ensemble sampler (250
  walkers, 1500 steps, 300 burn-in by default), selected with ``sampler="emcee"``.
* :func:`~sys_mapping.inference.posterior_median_params` — the per-parameter
  posterior median.
* :func:`~sys_mapping.inference.refine_to_mle` — the likelihood maximum, started
  from a point estimate, as a likelihood-ratio test needs.

**Key paper:** `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_,
Sec. 4 — see also :doc:`../methods`.

.. automodule:: sys_mapping.inference
   :members:
   :show-inheritance:
