sys_mapping.utils
=================

Angular two-point function measurement and covariance utilities.

Wrappers around `TreeCorr <https://rmjarvis.github.io/TreeCorr>`_ and
`Corrfunc <https://corrfunc.readthedocs.io>`_:

* :func:`~sys_mapping.utils.measure_two_point_function` and
  :func:`~sys_mapping.utils.measure_two_point_function_corrfunc` — measure
  the angular correlation function :math:`w(\theta)` from RA/Dec catalogs.
* :func:`~sys_mapping.utils.measure_kk_correlation_treecorr` and
  :func:`~sys_mapping.utils.measure_kk_correlation_corrfunc` — measure
  the kappa auto-correlation from HEALPix overdensity maps.
* :func:`~sys_mapping.utils.template_correlation_matrix` — the
  :math:`(n_{\rm sys}, n_{\rm sys}, n_\theta)` template correlation for the
  two-point correction, from the template values each galaxy carries, with optional
  per-object weights. Cross terms come from auto-correlations of summed fields,
  :math:`\xi_{ij}=[\xi(t_i+t_j)-\xi_{ii}-\xi_{jj}]/2`.
* :func:`~sys_mapping.utils.measure_cross_two_point_function` — cross-correlation
  of two catalogues.
* :func:`~sys_mapping.utils.compute_covariance_matrix` — jackknife or
  bootstrap covariance of :math:`w(\theta)`.
* :func:`~sys_mapping.utils.compute_amplitude_bias` — estimate the
  amplitude bias introduced by the correction procedure.

.. automodule:: sys_mapping.utils
   :members:
   :show-inheritance:
