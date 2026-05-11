sys_mapping.likelihood
======================

JAX-compiled Gaussian and skew-normal log-likelihoods for the contamination model.

The factory function :func:`~sys_mapping.likelihood.make_log_likelihood` returns
a JIT-compiled callable that evaluates :math:`\ln\mathcal{L}(\boldsymbol\theta)`
for a given combination of ``(n_sys, model, use_skewed)``.  The Jacobian
correction :math:`-\sum_p \ln|1 + \sum_i b_i t_{ip}|` is included for the
combined model (Berlfein et al. 2024, Eq. 17).

**Key paper:** `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_ — see also :doc:`../methods`.

.. automodule:: sys_mapping.likelihood
   :members:
   :show-inheritance:
