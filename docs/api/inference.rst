sys_mapping.inference
=====================

MCMC posterior sampling via the `emcee <https://emcee.readthedocs.io>`_ ensemble
sampler, posterior summaries, and refinement to the likelihood maximum.

:func:`~sys_mapping.inference.run_mcmc` wraps ``emcee.EnsembleSampler``
with sensible defaults (250 walkers, 1500 steps, 300 burn-in) and handles
the rotated-template basis internally.
:func:`~sys_mapping.inference.posterior_median_params` returns the posterior median;
:func:`~sys_mapping.inference.refine_to_mle` maximises the likelihood from there, which
is what a likelihood ratio between nested models needs.

**Key paper:** `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_,
Sec. 4 — see also :doc:`../methods`.

.. automodule:: sys_mapping.inference
   :members:
   :show-inheritance:
