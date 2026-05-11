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
* :func:`~sys_mapping.utils.compute_covariance_matrix` — jackknife or
  bootstrap covariance of :math:`w(\theta)`.
* :func:`~sys_mapping.utils.compute_amplitude_bias` — estimate the
  amplitude bias introduced by the correction procedure.

.. automodule:: sys_mapping.utils
   :members:
   :show-inheritance:
