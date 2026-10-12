sys_mapping.correction
======================

``sys_mapping.correction`` applies the post-inference corrections: noise debiasing, template PCA rotation, parameter back-transformation, and the two-point function correction.

After the MCMC, the usual sequence is:

1. :func:`~sys_mapping.correction.debias_params`: subtract the noise bias from the squared MAP estimates.
2. :func:`~sys_mapping.correction.correct_two_point_function`: apply the debiased parameters to the observed :math:`w(\theta)`.
3. :func:`~sys_mapping.correction.correct_power_spectrum_harmonic`: apply the harmonic-space correction (Elsner et al. 2016).
4. :func:`~sys_mapping.correction.estimate_overcorrection_bias`: measure, on contamination-free mocks, how much true structure the weighting itself removes;
   :func:`~sys_mapping.correction.debias_two_point_function` subtracts it from the data vector.

Key papers: `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_ (Eq. 15–16, 21, Appendix A);
`Elsner et al. 2016 <https://ui.adsabs.harvard.edu/abs/2016MNRAS.456.2095E/abstract>`_;
`Weaverdyck et al. 2026 <https://arxiv.org/abs/2601.14484>`_ (Eq. 21–23).
See also :doc:`../methods`.

.. automodule:: sys_mapping.correction
   :members:
   :show-inheritance:
