sys_mapping.covariance
======================

Correlated-noise (GLS) error models for the contamination parameters.

The pixel Gaussian likelihood (:mod:`sys_mapping.likelihood`) assumes **independent pixels**
(:math:`\sigma^2 I`). On a spatially-correlated galaxy field that underestimates the parameter
errors, because smooth systematic templates project onto the correlated field with far more
variance than white noise predicts. On the nine LS10 samples, against realisations carrying each
sample's matched spectrum, the OLS σ is short by a median factor of 1.8 to 3.1 per sample, rising
with resolution, and by up to 7.7 for a single template. This module supplies the calibrated alternatives.

* :func:`~sys_mapping.covariance.mock_sandwich_covariance` — the **additive** parameter covariance
  :math:`(TT^\top)^{-1}(TCT^\top)(TT^\top)^{-1}`, with :math:`C` estimated from an ensemble of
  *uncontaminated* mock reconstructions. Exact for the linear estimator, no :math:`N\times N`
  inverse; this is the error that calibrates the additive bars (see :doc:`../results_validation`).
* :class:`~sys_mapping.covariance.LowRankPrecision` /
  :func:`~sys_mapping.covariance.build_lowrank_precision` — a low-rank + diagonal pixel precision
  :math:`R^{-1}` (Woodbury) for the opt-in **GLS likelihood** (``precision=`` in
  :func:`~sys_mapping.likelihood.make_log_likelihood`, ``pixel_precision=`` in
  :func:`~sys_mapping.regression.run_decontamination`).
* :func:`~sys_mapping.covariance.build_harmonic_precision` — **stub** for the full-rank theory-\
  :math:`C_\ell` harmonic precision (the version that would calibrate the single-fit pixel
  likelihood on a cut sky); documented follow-up, not yet implemented.

* :func:`~sys_mapping.covariance.method_marginalised_covariance` — fold the
  *choice of decontamination method* into the data-vector covariance as a rank-one
  term :math:`\Delta_i\Delta_j`, with :math:`\Delta` the difference between two
  methods' :math:`w(\theta)`.  Six methods are on offer and they do not agree;
  this turns that disagreement from an argument into an error bar.

**Key papers:**
`Weaverdyck & Huterer 2021 <https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.5061W/abstract>`_
(mock-mode-projection / template covariance);
`Weaverdyck et al. 2026 <https://arxiv.org/abs/2601.14484>`_ (Eq. 24, method
marginalisation) — see also :doc:`../methods`.

.. automodule:: sys_mapping.covariance
   :members:
   :show-inheritance:
   :exclude-members: d_inv, u, m_inv, n_pix, n_modes
