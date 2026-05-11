sys_mapping.inference
=====================

MCMC posterior sampling and MAP estimation via the `emcee
<https://emcee.readthedocs.io>`_ ensemble sampler.

:func:`~sys_mapping.inference.run_mcmc` wraps ``emcee.EnsembleSampler``
with sensible defaults (250 walkers, 1500 steps, 300 burn-in) and handles
the rotated-template basis internally.  :func:`~sys_mapping.inference.get_mle_params`
returns the posterior median as the MAP estimate.

**Key paper:** `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_,
Sec. 4 — see also :doc:`../methods`.

.. automodule:: sys_mapping.inference
   :members:
   :show-inheritance:
