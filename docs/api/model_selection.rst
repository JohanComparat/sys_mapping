sys_mapping.model_selection
===========================

Likelihood ratio test (LRT) for comparing nested contamination models.

:func:`~sys_mapping.model_selection.likelihood_ratio_test` computes
:math:`\lambda_{\rm LR} = 2(\ln\mathcal{L}_{\rm alt} - \ln\mathcal{L}_{\rm null})`
and returns the p-value against a :math:`\chi^2(r)` distribution where
:math:`r` is the difference in free parameters.

Typical use: test additive (null) vs. combined (alternative) to decide
whether multiplicative contamination is significant.

**Key paper:** `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_, Eq. 19 — see also :doc:`../methods`.

.. automodule:: sys_mapping.model_selection
   :members:
   :show-inheritance:
