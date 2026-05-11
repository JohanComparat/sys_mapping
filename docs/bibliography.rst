Bibliography
============

This page documents the key literature on systematic effects mitigation in
photometric galaxy surveys. For each reference, the main method(s), their
mathematical formulation, pros/cons, and implementation status in
``sys_mapping`` are described.

.. contents:: Papers
   :local:
   :depth: 1

----

Ross et al. 2011
----------------

| **Ross, N. P., et al.** (2011). *Ameliorating Systematic Uncertainties in the Angular Clustering of Galaxies: A Study using the SDSS-III BOSS.*
| MNRAS 424(1): 564–586.
| `ADS <https://ui.adsabs.harvard.edu/abs/2011MNRAS.417.1350R/abstract>`__ · `arXiv:1105.2320 <https://arxiv.org/abs/1105.2320>`__

**Methods:**
Masking of stellar contamination, dust extinction corrections, sky
background removal, and jack-knife covariance estimation for systematic
null tests. Systematic corrections are applied by assigning per-pixel
weights based on observational condition maps (seeing, airmass, star
density, Galactic extinction).

**Key equation** — Landy-Szalay two-point estimator after systematic
weighting:

.. math::

   w(\theta) = \frac{DD - 2\,DR + RR}{RR}

**Null test**: cross-correlation of the weight map with each template
should be consistent with zero for a well-corrected field.

**Jack-knife covariance:**

.. math::

   \sigma_{\mathrm{jack}}^2(\theta) = \frac{N-1}{N} \sum_{i=1}^{N}
   \bigl[w(\theta) - w_i(\theta)\bigr]^2

where :math:`w_i` is the estimator with patch :math:`i` removed.

.. rubric:: Pros and cons

**Pros:**
Intuitive per-pixel weighting. Jack-knife uncertainties are conservative
and capture spatial correlations. Multiple independent null tests can be
run simultaneously.

**Cons:**
Masking reduces effective area. Jack-knife requires many patches for the
prefactor to be accurate. Correlations between systematics are not
jointly modelled.

.. rubric:: Implementation status in sys_mapping

- Landy-Szalay estimator: **implemented** (``utils.measure_two_point_function``).
- Jack-knife covariance: **implemented** (``bootstrap.jackknife_covariance``).
- Null test cross-correlations: **implemented** (``diagnostics.null_test_cross_correlations``).

----

Ho et al. 2012
--------------

| **Ho, S., et al.** (2012). *Clustering of Sloan Digital Sky Survey III photometric luminous galaxies: the measurement, systematics and cosmological implications.*
| ApJ 761(1): 14.
| `ADS <https://ui.adsabs.harvard.edu/abs/2012ApJ...761...14H/abstract>`__ · `arXiv:1201.2137 <https://arxiv.org/abs/1201.2137>`__

**Methods:**
Optimal quadratic estimator (QMV) for the galaxy angular power spectrum
with systematic control. The covariance-weighted estimator minimises
variance while accounting for mode-coupling from the survey mask.
Systematic templates are handled via mode projection in harmonic space.

**Key equations** — optimal quadratic estimator and Fisher matrix:

.. math::

   \hat{C}_\ell^{ss} = \sum_i N_i^{-1}\, d^T\, E_i\, d, \qquad
   E_i = C^{-1}\,\frac{\partial C}{\partial C_i}\,C^{-1}

.. math::

   N_{\ell\ell'} = \frac{1}{2}\,\mathrm{tr}\!\left[
   C^{-1}\frac{\partial C}{\partial C_\ell}
   C^{-1}\frac{\partial C}{\partial C_{\ell'}}\right]

.. rubric:: Pros and cons

**Pros:**
Statistically optimal (minimum variance). Rigorous treatment of
data covariance including noise and mask coupling.

**Cons:**
Requires explicit covariance matrix inversion — scales poorly to large
pixel numbers. Systematic templates must be characterised a priori.

.. rubric:: Implementation status in sys_mapping

- Harmonic-space pseudo-:math:`C_\ell` estimator (simpler, mask-based):
  **implemented** (``power_spectrum.measure_pseudo_cl``).
- Full QMV optimal estimator: **not planned** (high computational cost;
  NaMaster/pymaster covers this use case).

----

Elsner, Leistedt & Peiris 2016
-------------------------------

| **Elsner, F., Leistedt, B., Peiris, H. V.** (2016). *Unbiased methods for removing systematics from galaxy clustering measurements.*
| MNRAS 456(2): 2095–2104.
| `ADS <https://ui.adsabs.harvard.edu/abs/2016MNRAS.456.2095E/abstract>`__ · `arXiv:1509.08933 <https://arxiv.org/abs/1509.08933>`__

**Methods:**
Three complementary harmonic-space approaches with closed-form bias expressions:

1. **Template subtraction (TS):** subtract :math:`\hat\alpha_i\,\tilde C_\ell^{t_i}` from the measured pseudo-:math:`C_\ell`.
2. **Basic mode projection (BMP):** marginalise over template modes via a modified pixel covariance.
3. **Extended mode projection (EMP):** threshold-based exclusion of modes dominated by templates, with an analytical bias correction.

**Key equations:**

Cleaned power spectrum (TS):

.. math::

   \tilde C_\ell^{\rm TS} = \hat C_\ell^{d\times d}
   - \sum_i \hat\alpha_i\,\hat C_\ell^{t_i\times t_i}

Additive bias from template subtraction (:math:`n` templates):

.. math::

   b_\ell = -\frac{n}{2\ell+1}

EMP bias (Eq. 27 of Elsner+16):

.. math::

   b_\ell = -\sqrt{\frac{2}{\pi}}\,
   \frac{k^2}{2(2\ell+1)}\,
   e^{k^2/(2C_\ell^{ss})}\,
   \mathrm{erfc}\!\left(\frac{k}{\sqrt{2C_\ell^{ss}}}\right)

.. rubric:: Pros and cons

**Pros:**
Closed-form bias corrections; analytically tractable. Template subtraction
is fast. Mode projection does not require fitting contamination amplitudes.

**Cons:**
Template subtraction biases scale as :math:`n/(2\ell+1)` — significant at
low :math:`\ell` with many templates. EMP bias depends on signal-to-noise
and threshold choice. Mixing from the survey mask breaks spherical harmonic
orthogonality.

.. rubric:: Implementation status in sys_mapping

- Template subtraction in harmonic space: **implemented** (``power_spectrum.subtract_template_cl``).
- Harmonic bias formula: **implemented** (``power_spectrum.harmonic_bias``).
- Basic and extended mode projection: **implemented** (``power_spectrum.mode_projection_bias``).
- Full correction pipeline: **implemented** (``correction.correct_power_spectrum_harmonic``).

----

Leistedt & Peiris 2014
-----------------------

| **Leistedt, B., Peiris, H. V.** (2014). *Exploiting the full potential of photometric quasar surveys: optimal power spectra through blind mitigation of systematics.*
| MNRAS 444(1): 2–19.
| `ADS <https://ui.adsabs.harvard.edu/abs/2014MNRAS.444....2L/abstract>`__ · `arXiv:1404.6530 <https://arxiv.org/abs/1404.6530>`__

**Methods:**
Introduces *extended mode projection* (EMP) as a blind systematic
mitigation strategy. Templates are identified from the data themselves
by iteratively projecting out the modes most correlated with external
maps, without prior knowledge of contamination amplitudes.

**Key idea:** project a set of templates :math:`\{f_i\}` out of the
data vector before power spectrum estimation:

.. math::

   d_{\rm proj} = d - \sum_i (d \cdot f_i)\, f_i / (f_i \cdot f_i)

This is iterated until the cross-power spectrum of the cleaned field with
each template is consistent with zero.

.. rubric:: Pros and cons

**Pros:**
Blind — no prior knowledge of contamination amplitudes needed.
Naturally handles correlated templates.

**Cons:**
Each projected template costs one mode per :math:`\ell` band — significant
variance penalty for large template sets. Power loss must be corrected via
a transfer function.

.. rubric:: Implementation status in sys_mapping

- EMP basis: **implemented** as part of ``power_spectrum.mode_projection_bias``.
- Blind iterative variant: **planned** (future extension of ``power_spectrum``).

----

Elsner, Leistedt & Peiris 2017
-------------------------------

| **Elsner, F., Leistedt, B., Peiris, H. V.** (2017). *Unbiased pseudo-Cℓ power spectrum estimation with mode projection.*
| MNRAS 465(2): 1847–1855.
| `ADS <https://ui.adsabs.harvard.edu/abs/2017MNRAS.465.1847E/abstract>`__ · `arXiv:1609.03577 <https://arxiv.org/abs/1609.03577>`__

**Methods:**
Integrates mode projection directly into the pseudo-:math:`C_\ell`
estimator, deriving exact closed-form expressions for the deprojection
bias within the MASTER/PCL framework. Extends the 2016 work to handle
arbitrary mask geometry and pixel weights.

**Key result:** the pseudo-:math:`C_\ell` after EMP is related to the
true power spectrum by a modified coupling matrix
:math:`M_{\ell\ell'}^{\rm proj}` that is efficiently computable from the
mask.

.. rubric:: Pros and cons

**Pros:**
Provides exact bias correction in the pseudo-:math:`C_\ell` framework.
Suitable for non-Gaussian fields and realistic survey footprints.

**Cons:**
Coupling matrix computation scales as :math:`O(\ell_{\max}^3)`;
large surveys require optimised implementations (e.g. NaMaster).

.. rubric:: Implementation status in sys_mapping

- Coupling matrix computation: **partially implemented** via
  ``power_spectrum.mode_projection_bias`` (analytical approximation).
- Full coupling matrix: **deferred** (requires NaMaster as optional backend).

----

Weaverdyck & Huterer 2021
--------------------------

| **Weaverdyck, N., Huterer, D.** (2021). *Mitigating contamination in LSS surveys: a comparison of methods.*
| MNRAS 503(4): 5061–5084.
| `ADS <https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.5061W/abstract>`__ · `arXiv:2007.14499 <https://arxiv.org/abs/2007.14499>`__

**Methods:**
Systematic comparison of four decontamination methods applied to the
same simulated and real data:

1. **DES-Y1 iterative method:** multiplicative weights :math:`w = (1 + \alpha T)^{-1}` iterated to convergence.
2. **Template subtraction:** subtract the cross-correlation component.
3. **Mode projection:** harmonic-space marginalisation.
4. **ElasticNet regression:** L1+L2 penalised linear regression.

**ElasticNet loss:**

.. math::

   \mathcal{L} = \frac{1}{2N_{\rm pix}}
   \left\|\delta_g - \sum_i\alpha_i\,t_i\right\|_2^2
   + \lambda_1\|\boldsymbol\alpha\|_1
   + \lambda_2\|\boldsymbol\alpha\|_2^2

**Per-pixel weight from regression:**

.. math::

   w(p) = \frac{1}{1 + \hat{\boldsymbol\alpha}\cdot\mathbf{t}(p)}

.. rubric:: Pros and cons

**Pros:**
ElasticNet naturally prevents overfitting and finds sparse solutions.
The comparison framework reveals method-dependent biases.
No iterative convergence issue for ElasticNet.

**Cons:**
Regularisation hyperparameters (:math:`\lambda_1, \lambda_2`) require
tuning. No single method dominates in all scenarios.

.. rubric:: Implementation status in sys_mapping

- ElasticNet regression: **implemented** (``regression.elasticnet_contamination_fit``).
- Iterative OLS: **implemented** (``regression.iterative_systematics_decontamination``).
- Method comparison framework: **implemented** (``regression.method_comparison``).

----

Rezaie et al. 2020
-------------------

| **Rezaie, M., et al.** (2020). *Improving Galaxy Clustering Measurements with Deep Learning: analysis of the DECaLS DR7 data.*
| ApJS 253(2): 32.
| `arXiv:1907.11355 <https://arxiv.org/abs/1907.11355>`__

**Methods:**
Artificial neural networks (ANN) trained to predict the galaxy density
field from observational systematic maps (Galactic extinction, seeing,
stellar density). The ratio of the predicted to mean density defines
per-pixel weights. The method is model-free and captures non-linear
systematic dependencies.

**Weight definition:**

.. math::

   w(p) = \frac{\bar n_g}{n_g^{\rm ANN}(p)}

where :math:`n_g^{\rm ANN}` is the ANN-predicted galaxy count from
systematic templates.

.. rubric:: Pros and cons

**Pros:**
Captures non-linear and cross-template systematic dependencies.
No functional form assumed for contamination.

**Cons:**
Prone to overfitting when templates are correlated with large-scale
structure. Requires careful validation (e.g. cross-validation on
disjoint sky regions). Less interpretable than linear methods.

.. rubric:: Implementation status in sys_mapping

- ANN-based systematic mapping: **not planned** (out of scope for the
  current Bayesian likelihood framework; recommended as external
  pre-processing step).

----

Alonso et al. 2019 — NaMaster
------------------------------

| **Alonso, D., Sanchez, J., Slosar, A.** (2019). *A unified pseudo-Cℓ framework.*
| MNRAS 484(3): 4127–4151.
| `ADS <https://ui.adsabs.harvard.edu/abs/2019MNRAS.484.4127A/abstract>`__ · `arXiv:1809.09603 <https://arxiv.org/abs/1809.09603>`__

**Methods:**
NaMaster (pymaster) provides a unified pseudo-:math:`C_\ell` estimator
with:
- Exact mask mode-coupling matrix computation.
- E/B-mode purification for spin-2 fields.
- Template deprojection in harmonic space.
- Support for arbitrary pixelisation schemes (HEALPix and flat-sky).

**MASTER equation:**

.. math::

   \langle\tilde C_\ell\rangle = \sum_{\ell'} M_{\ell\ell'}\,C_{\ell'}

where :math:`M_{\ell\ell'}` is the mode-coupling matrix computed from
the mask power spectrum.

.. rubric:: Pros and cons

**Pros:**
Highly optimised coupling matrix computation.
Standard reference implementation used by DES, DESI, Euclid.

**Cons:**
External dependency; adds build complexity. Coupling matrix computation
can be expensive for large :math:`\ell_{\max}`.

.. rubric:: Implementation status in sys_mapping

- ``measure_pseudo_cl`` uses ``healpy.anafast`` (no NaMaster required).
- NaMaster: **optional backend** — planned as an optional dependency for
  exact coupling matrix computation in ``power_spectrum.mode_projection_bias``.

----

Berlfein, Mandelbaum & Schafer 2024
-------------------------------------

| **Berlfein, F., Mandelbaum, R., Dodelson, S., Schafer, C.** (2024). *Joint inference of multiplicative and additive systematics in galaxy density fluctuations and clustering measurements.*
| MNRAS 531: 4954–4974.
| `ADS <https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.4954B/abstract>`__ · `arXiv:2401.12293 <https://arxiv.org/abs/2401.12293>`__

**Methods:**
The primary reference for ``sys_mapping``. Defines three nested
contamination models and a joint Bayesian MCMC framework for inferring
their parameters.

**Contamination model (combined):**

.. math::

   \hat\delta_g(p) = \delta_g(p)\!\left(1 + \sum_i b_i\,\delta_{t,i}(p)\right)
   + \sum_i a_i\,\delta_{t,i}(p)

**Gaussian log-likelihood (Eq. 17):**

.. math::

   \ln\mathcal{L} = -\frac{N_{\rm pix}}{2}\ln(2\pi\sigma^2)
   - \frac{1}{2\sigma^2}\sum_p \delta_g(p)^2
   + \sum_p \ln|J(p)|

**Two-point function correction (Eq. 15–16):**

.. math::

   \hat w_{\rm corr}(\theta) =
   \frac{\hat w(\theta) - \sum_i\tilde a_i^2\,\xi_i(\theta)}
        {1 + \sum_i\tilde b_i^2\,\xi_i(\theta)}

**Noise debiasing (Eq. 21):**

.. math::

   \tilde a_i^2 = \max\!\left(\hat a_i^2 - \mathrm{Var}[\hat a_i],\, 0\right)

.. rubric:: Pros and cons

**Pros:**
Joint treatment of additive and multiplicative systematics avoids
double-counting. PCA template rotation decorrelates parameters.
Noise debiasing removes variance inflation. Skew-normal likelihood
handles lognormal galaxy fields.

**Cons:**
MCMC is slower than regression approaches. Assumes Gaussian/skew-normal
pixel overdensities. Linear contamination model may miss non-linear
systematics.

.. rubric:: Implementation status in sys_mapping

All core methods from Berlfein+2024 are **fully implemented**:

- Forward/inverse contamination model: ``contamination``
- JAX JIT-compiled likelihoods: ``likelihood``
- emcee MCMC inference: ``inference``
- PCA template rotation + noise debiasing: ``correction``
- Two-point function correction: ``contamination.compute_two_point_correction``
- Likelihood ratio model selection: ``model_selection``

----

Rodríguez-Monroy et al. 2025
------------------------------

| **Rodríguez-Monroy, M., et al.** (2025). *Dark Energy Survey Year 6 Results: improved mitigation of spatially varying observational systematics with masking.*
| arXiv:2509.07943.
| `ADS <https://ui.adsabs.harvard.edu/abs/2025arXiv250907943R/abstract>`__ · `arXiv:2509.07943 <https://arxiv.org/abs/2509.07943>`__

**Methods:**
Two complementary strategies applied to DES Y6 galaxy clustering:

1. **Iterative Systematics Decontamination (ISD):** OLS regression on
   polynomial expansions of templates up to 3rd order, iterated to
   convergence.
2. **Footprint masking:** conservative masking of survey regions with
   high systematic sensitivity, enabling simpler and more robust
   mitigation.

**ISD cleansed overdensity:**

.. math::

   \delta_g^{\rm clean}(p) =
   \frac{\delta_g(p) - f_{\rm add}(p)}{1 + f_{\rm mult}(p)}

where :math:`f_{\rm add}` and :math:`f_{\rm mult}` are the additive and
multiplicative systematic components estimated from polynomial template regression.

.. rubric:: Pros and cons

**Pros:**
ISD captures non-linear systematic dependencies via polynomial template
expansion. Footprint masking provides conservative, robust improvement.

**Cons:**
Polynomial expansion grows combinatorially with template number and
polynomial order. Masking reduces effective survey area.

.. rubric:: Implementation status in sys_mapping

- ISD (polynomial OLS iteration): **implemented** (``regression.iterative_systematics_decontamination``).
- Footprint masking diagnostics: **implemented** (``diagnostics.footprint_mask_diagnostics``).

----

Cornish et al. 2026
--------------------

| **Cornish, T., Alonso, D., Leistedt, B., Wolz, K.** (2026). *Systematics mitigation for catalogue-based angular power spectra.*
| MNRAS 547.
| `ADS <https://ui.adsabs.harvard.edu/abs/2026MNRAS.547ag360C/abstract>`__ · `arXiv:2510.19912 <https://arxiv.org/abs/2510.19912>`__

**Methods:**
Extends template deprojection to work directly on discrete source
catalogues (not pixelised maps), avoiding pixelisation-induced power
leakage. Introduces a transfer function calibration loop to correct for
power loss from deprojection.

**Catalogue-based deprojection:**

.. math::

   a_i^{\rm deproj} = a_i - \sum_p A_p\, f_i^p

**Transfer function calibration:**

.. math::

   T_f(\ell) = \frac{\langle C_\ell^{\rm cont}\rangle_{\rm sim}}
                    {\langle C_\ell^{\rm before}\rangle_{\rm sim}}

.. rubric:: Pros and cons

**Pros:**
Avoids pixelisation bias. Transfer function corrects for deprojection
power loss. Exact treatment for discrete catalogues.

**Cons:**
Transfer function calibration requires simulations (expensive).
Mode-coupling increases in complexity for catalogue-based fields.

.. rubric:: Implementation status in sys_mapping

- Catalogue-based deprojection: **deferred** (planned as ``deprojection``
  module once ``power_spectrum`` is validated).
- Transfer function calibration: **deferred**.

----

Tanidis et al. 2026
--------------------

| **Tanidis, K., Alonso, D., Miller, L., Harnois-Déraps, J.** (2026). *Reconstructing spatially-varying multiplicative bias for Stage IV weak lensing galaxy surveys with a quadratic estimator.*
| MNRAS 547.
| `ADS <https://ui.adsabs.harvard.edu/abs/2026MNRAS.547ag537T/abstract>`__ · `arXiv:2512.13679 <https://arxiv.org/abs/2512.13679>`__

**Methods:**
Quadratic estimator exploiting E/B mode coupling to reconstruct
spatially-varying multiplicative bias :math:`m(\hat n)` in weak lensing
surveys. Three signal-to-noise ratio definitions are introduced for
template detection and ranking.

**Quadratic estimator:**

.. math::

   \hat m(\mathbf{L}) = N(\mathbf{L})^{-1}
   \int\frac{d^2\ell}{(2\pi)^2}\,
   E^{\rm obs}(\boldsymbol\ell)\,
   B^{\rm obs}(\boldsymbol\ell - \mathbf{L})\,
   W(\boldsymbol\ell, \boldsymbol\ell-\mathbf{L})

**Three SNR definitions:**

.. math::

   \mathrm{SNR}_{\rm template} = \frac{|\hat\alpha_i|}{\sigma_{\hat\alpha_i}}, \quad
   \mathrm{SNR}_{\rm data} = |\mathrm{Corr}(\delta_g, t_i)|, \quad
   \mathrm{SNR}_{\rm peak} = \frac{\max_\ell \hat C_\ell^{\delta_g t_i}}{\sigma_\ell}

.. rubric:: Pros and cons

**Pros:**
Detects percent-level multiplicative bias at high significance for
Stage IV surveys. Mode-coupling signal is insensitive to additive
(c-bias) contamination.

**Cons:**
Primarily developed for weak lensing shear fields; adaptation to galaxy
density requires modification. Requires knowledge of the cosmic shear
power spectrum.

.. rubric:: Implementation status in sys_mapping

- SNR-based template ranking (all three definitions adapted for galaxy
  clustering): **implemented** (``diagnostics.snr_template_ranking``).
- Full quadratic estimator for shear E/B coupling: **not planned**
  (specific to weak lensing; outside the scope of galaxy density maps).

