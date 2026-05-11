sys_mapping.correction
======================

Post-inference corrections: noise debiasing, template PCA rotation,
parameter back-transformation, and two-point function correction.

The typical post-MCMC workflow is:

1. :func:`~sys_mapping.correction.debias_params` — subtract noise bias
   from squared MAP estimates.
2. :func:`~sys_mapping.correction.correct_two_point_function` — apply the
   debiased parameters to the observed :math:`w(\theta)`.
3. :func:`~sys_mapping.correction.correct_power_spectrum_harmonic` — apply
   the harmonic-space correction (Elsner et al. 2016).

**Key papers:** `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_ (Eq. 15–16, 21, Appendix A);
`Elsner et al. 2016 <https://ui.adsabs.harvard.edu/abs/2016MNRAS.456.2095E/abstract>`_ — see also :doc:`../methods`.

.. automodule:: sys_mapping.correction
   :members:
   :show-inheritance:
