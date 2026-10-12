sys_mapping.power_spectrum
==========================

``sys_mapping.power_spectrum`` mitigates systematics in harmonic space with pseudo-:math:`C_\ell` estimators.
It takes a full-sky HEALPix overdensity map, the survey mask, the template maps and the estimated contamination amplitudes,
and returns pseudo-:math:`C_\ell` arrays (shape ``(lmax+1,)``), bias corrections, and mode-projected power spectra.

The pipeline is:

1. :func:`~sys_mapping.power_spectrum.measure_pseudo_cl`: apply the mask and call ``healpy.anafast``.
2. :func:`~sys_mapping.power_spectrum.subtract_template_cl`: subtract :math:`\sum_i \hat\alpha_i \hat C_\ell^{t_i}`.
3. :func:`~sys_mapping.power_spectrum.harmonic_bias`: compute the residual bias :math:`b_\ell = -n/(2\ell+1)`.
4. :func:`~sys_mapping.power_spectrum.mode_projection_bias`: apply BMP or EMP deprojection.

Key papers:
`Elsner et al. 2016 <https://ui.adsabs.harvard.edu/abs/2016MNRAS.456.2095E/abstract>`_;
`Elsner et al. 2017 <https://ui.adsabs.harvard.edu/abs/2017MNRAS.465.1847E/abstract>`_;
`Leistedt & Peiris 2014 <https://ui.adsabs.harvard.edu/abs/2014MNRAS.444....2L/abstract>`_.
See also :doc:`../methods`.

.. automodule:: sys_mapping.power_spectrum
   :members:
   :show-inheritance:
