sys_mapping.model_selection
===========================

Likelihood ratio test (LRT) for comparing nested contamination models.

:func:`~sys_mapping.model_selection.likelihood_ratio_test` computes
:math:`\lambda_{\rm LR} = 2(\ln\mathcal{L}_{\rm alt} - \ln\mathcal{L}_{\rm null})`
and its p-value, from a mock null distribution when ``null_lambda`` is given and from
:math:`\chi^2(r)` otherwise, :math:`r` being the difference in free parameters. The
Wilks p-value assumes independent pixels, which the clustered field violates.

* :func:`~sys_mapping.model_selection.lrt_from_maxima` — the additive and combined
  maxima of many fields at once, and their :math:`\lambda_{\rm LR}`. The additive
  maximum is the least-squares solution; the combined maximum is found by L-BFGS
  from :math:`(\hat{\mathbf a}_{\rm OLS}, \mathbf b=0)`, vmapped over fields. On
  8 fields of 7 040 pixels with 11 templates it takes 0.15 s after compilation, against
  about 50 s for NUTS fits refined to their maxima, and matches them to a relative
  :math:`8\times10^{-7}` in :math:`\lambda_{\rm LR}`.
* :func:`~sys_mapping.model_selection.lrt_null_distribution` — the same null from a
  user-supplied fit function.
* :func:`~sys_mapping.model_selection.greedy_forward_select` and
  :func:`~sys_mapping.model_selection.snr_preselect` — template selection by
  successive likelihood-ratio tests and by SNR ranking.

**Key paper:** `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_, Eq. 19 — see also :doc:`../methods`.

.. automodule:: sys_mapping.model_selection
   :members:
   :show-inheritance:
   :exclude-members: selected_indices,rounds,p_threshold,n_initial,snr_values,method,snr_min,n_top
