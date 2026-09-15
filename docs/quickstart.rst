Quickstart
==========

This page runs the pipeline on a synthetic lognormal mock with three templates: the six
decontamination methods, a null from a matched spectrum, the likelihood-ratio test, and the
two-point correction with its cross-template terms.
The contamination model is that of
`Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_; notation is in :doc:`overview`
and the mathematics in :doc:`methods`.
The code blocks run in sequence, and the printed values are from one run of them.

----

Methods overview
----------------

:func:`~sys_mapping.regression.run_decontamination` runs every method through one interface.

.. list-table::
   :header-rows: 1
   :widths: 13 27 18 30 12

   * - Method
     - Reference
     - Model
     - Fit
     - Time
   * - OLS
     - `Ross et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.417.1350R/abstract>`_
     - additive
     - least squares for :math:`a_i`
     - 1 ms
   * - ElasticNet
     - `Weaverdyck & Huterer 2021 <https://doi.org/10.1093/mnras/stab818>`_
     - additive
     - L1+L2-penalised :math:`a_i`, 5-fold cross-validation (scikit-learn)
     - 0.1–1 s
   * - ISD-1
     - `Rodríguez-Monroy et al. 2025 <https://arxiv.org/abs/2509.07943>`_
     - per-template response
     - degree-1 polynomial in one template's value, iterated over templates
     - 0.1–1 s
   * - ISD-3
     - `Rodríguez-Monroy et al. 2025 <https://arxiv.org/abs/2509.07943>`_
     - per-template response
     - degree-3 polynomial in one template's value, iterated over templates
     - 0.1–1 s
   * - MCMC-add
     - `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_
     - additive
     - exact Normal–Inverse-Gamma posterior of :math:`a_i, \sigma`
     - 1 s
   * - MCMC-comb
     - `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_
     - combined
     - BlackJAX NUTS on :math:`a_i, b_i, \sigma`
     - 10 s

Times are orders of magnitude for the blocks below (NSIDE 32, three templates, first call
including JAX compilation), measured on a machine shared with other jobs.
``sampler="emcee"`` selects the emcee sampler for either MCMC method instead.

Each method returns per-pixel weights, clipped to :math:`[1/20, 20]`.
OLS and ElasticNet return :math:`w(p) = 1/(1 + \sum_i \hat a_i t_i(p))`; ISD returns the
product of its step weights :math:`\prod_j 1/(1 + \hat F_j(t_j(p)))`; both MCMC methods return
:math:`w(p) = (1 + \delta_g^{\rm clean}(p))/(1 + \hat\delta_g(p))`, with
:math:`\delta_g^{\rm clean}` from :func:`~sys_mapping.contamination.invert_contamination` at
the fitted :math:`\hat a, \hat b`.

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   import healpy as hp
   import sys_mapping as sm

----

Step 1 — Template maps
----------------------

Templates :math:`t_i(p)` are HEALPix maps of observing conditions, standardised to zero mean
and unit variance so that the amplitudes :math:`a_i, b_i` are dimensionless.
:func:`~sys_mapping.maps.generate_systematic_maps` returns an array of shape
:math:`(n_{\rm sys}, 12\,\texttt{NSIDE}^2)`.

.. code-block:: python

   NSIDE = 32
   N_PIX = hp.nside2npix(NSIDE)                  # 12 288 pixels
   templates = sm.generate_systematic_maps(NSIDE, families=[0, 1, 2], seed=0)
   # templates.shape == (3, 12288)

----

Step 2 — Galaxy field and contamination
---------------------------------------

:func:`~sys_mapping.mocks.generate_lognormal_field` draws a Gaussian field :math:`G` with
:math:`C_\ell \propto (\ell+1)^{-2}` and variance :math:`\sigma_G^2 = 0.25`, and returns
:math:`\delta_g^{\rm true} = \exp(G - \sigma_G^2/2) - 1`
(`Coles & Jones 1991 <https://ui.adsabs.harvard.edu/abs/1991MNRAS.248....1C/abstract>`_).
The contaminated field is

.. math::

   \hat\delta_g(p) = \delta_g^{\rm true}(p)\left(1 + \sum_i b_i\,t_i(p)\right)
                   + \sum_i a_i\,t_i(p),

with :math:`\mathbf a^{\rm true} = (0.05, -0.03, 0.08)` and
:math:`\mathbf b^{\rm true} = (0.04, 0, -0.06)`.
Galaxy counts are Poisson draws with 40 galaxies per pixel on average, and random counts are
Poisson draws at ten times that density.

.. code-block:: python

   a_true = np.array([0.05, -0.03, 0.08])
   b_true = np.array([0.04, 0.00, -0.06])

   delta_g_true = sm.generate_lognormal_field(NSIDE, sigma=0.5, seed=1)
   delta_g_cont = np.asarray(sm.apply_contamination(
       jnp.asarray(delta_g_true), jnp.asarray(templates),
       jnp.asarray(a_true), jnp.asarray(b_true)))

   rng = np.random.default_rng(0)
   N_MEAN = 40                                   # galaxies per pixel
   galaxy_counts = rng.poisson(np.clip(N_MEAN * (1.0 + delta_g_cont), 0.0, None)).astype(float)
   random_counts = rng.poisson(10 * N_MEAN, N_PIX).astype(float)

----

Step 3 — Observed overdensity
-----------------------------

:func:`~sys_mapping.maps.compute_overdensity` keeps the pixels whose random count is at least
10% of the maximum and returns, on those pixels,

.. math::

   \hat\delta_g(p) = \frac{n_g(p)}{f\,n_r(p)} - 1, \qquad
   f = \frac{\sum_p n_g(p)}{\sum_p n_r(p)}.

:func:`~sys_mapping.maps.assign_template_values` restricts the templates to the same pixels,
with shape :math:`(n_{\rm sys}, N_{\rm good})`, the layout every fitting function takes.

.. code-block:: python

   delta_g, good = sm.compute_overdensity(galaxy_counts, random_counts)
   delta_t = sm.assign_template_values(templates, good)
   # delta_g.shape == (n_good,), good.shape == (N_PIX,), delta_t.shape == (3, n_good)

----

Step 4 — PCA rotation
---------------------

:func:`~sys_mapping.correction.rotate_templates` diagonalises the template covariance,
:math:`\tilde t = R\,t` with the eigenvectors as the rows of :math:`R`.
The MCMC methods fit in this basis: ``run_decontamination`` applies the same rotation
internally and returns amplitudes in the original basis, and Step 8 uses ``delta_t_rot`` and
``R`` again.
A small eigenvalue marks a near-degenerate combination of templates.

.. code-block:: python

   delta_t_rot, R, eigenvalues = sm.rotate_templates(delta_t)
   print("template eigenvalues:", eigenvalues.round(3))

.. code-block:: text

   template eigenvalues: [1.014 0.999 0.987]

----

Step 5 — Run the six methods
----------------------------

``run_decontamination`` takes the overdensity and the templates in the original basis.
``MCMC-add`` draws from the analytic posterior; ``MCMC-comb`` runs NUTS
(:func:`~sys_mapping.nuts.run_nuts`), here with 500 warm-up and 500 sampling steps per chain.
Without ``isd_chi2_68`` the ISD stopping threshold is in raw :math:`\Delta\chi^2` units and a
warning says so; :func:`~sys_mapping.diagnostics.isd_template_significance` supplies the
mock calibration.

.. code-block:: python

   METHODS = ["OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"]
   results = {}
   for method in METHODS:
       res = sm.run_decontamination(method, delta_g, delta_t,
                                    nuts_n_warmup=500, nuts_n_samples=500, seed=42)
       results[method] = res
       sigma = "" if res["sigma_hat"] is None else f"  sigma = {res['sigma_hat']:.3f}"
       print(f"{method:10s}  a = {res['a_hat'].round(3)}  b = {res['b_hat'].round(3)}{sigma}")

.. code-block:: text

   OLS         a = [ 0.056 -0.024  0.074]  b = [0. 0. 0.]
   ElasticNet  a = [ 0.056 -0.024  0.073]  b = [0. 0. 0.]
   ISD-1       a = [ 0.051 -0.024  0.088]  b = [0. 0. 0.]
   ISD-3       a = [ 0.05  -0.023  0.078]  b = [0. 0. 0.]
   MCMC-add    a = [ 0.056 -0.024  0.074]  b = [0. 0. 0.]  sigma = 0.550
   MCMC-comb   a = [ 0.055 -0.024  0.08 ]  b = [ 0.044 -0.013 -0.041]  sigma = 0.549

Only ``MCMC-comb`` fits :math:`b_i`; the other methods return zeros.
:math:`\sigma` is the pixel scatter of the clean field plus shot noise,
:math:`(e^{\sigma_G^2} - 1)^{1/2} = 0.53` from the field alone.

----

Step 6 — Point estimates and convergence
----------------------------------------

``a_hat`` and ``b_hat`` are per-parameter posterior medians
(:func:`~sys_mapping.inference.posterior_median_params`) in the original basis, and
``cov_a``, ``cov_b`` the posterior covariances there.
A median is a point estimate to report; a likelihood comparison needs the maximum from
:func:`~sys_mapping.inference.refine_to_mle` (Step 8).
The posterior assumes independent pixels, so on a clustered field its widths are too small;
Step 7 calibrates the significances on a null.

.. code-block:: python

   res = results["MCMC-comb"]
   a_hat, b_hat = res["a_hat"], res["b_hat"]     # original template basis
   cov_a, cov_b = res["cov_a"], res["cov_b"]     # (3, 3) posterior covariances
   print(f"{res['sampler_backend']}: R-hat {res['rhat']:.3f}, ESS {res['ess']:.0f}, "
         f"divergences {res['num_divergences']}")

.. code-block:: text

   nuts: R-hat 1.002, ESS 2426, divergences 0

----

Step 7 — A null from a matched spectrum
---------------------------------------

A calibrated statistic compares the data with contamination-free realisations that carry
the data's clustering.
For a survey sample that spectrum comes from :func:`~sys_mapping.glass_mocks.load_matched_cl`,
which reads the validated ``*_match.json`` for the sample and resolution; for this mock the
spectrum of the clean field stands in for it.
:func:`~sys_mapping.glass_mocks.generate_glass_null_overdensity` draws the realisations on the
footprint as Poisson counts per pixel from a GLASS field
(:func:`~sys_mapping.glass_mocks.draw_null_overdensity` draws one).
:func:`~sys_mapping.diagnostics.calibrated_template_significance` scores each template's
least-squares amplitude against its scatter across the null and returns a family-wise
p-value for the largest significance, whose floor is :math:`1/(N_{\rm null} + 1)`.

.. code-block:: python

   # For a survey sample: cl_null = sm.load_matched_cl("matched_spectra/", sample_id, NSIDE)
   cl_null = hp.anafast(delta_g_true)

   null = sm.generate_glass_null_overdensity(
       100, NSIDE, good, n_total_footprint=galaxy_counts[good].sum(), z_max=0.3,
       seed=100, rand_factor=10, cl_input=cl_null)
   # null.shape == (100, n_good)

   sig = sm.calibrated_template_significance(delta_g, delta_t, null)
   print("significance:", sig["significance"].round(1),
         " family-wise p:", round(sig["family_wise_p"], 4))

.. code-block:: text

   significance: [9.4 4.4 3.6]  family-wise p: 0.0099

----

Step 8 — Likelihood ratio test
------------------------------

The test compares the combined model with the additive one (:math:`b_i = 0`):

.. math::

   \lambda_{\rm LR} = 2\left[\ln\mathcal L(\hat\Theta_{\rm comb})
                      - \ln\mathcal L(\hat\Theta_{\rm add})\right],

with :math:`r = (2n_{\rm sys} + 1) - (n_{\rm sys} + 1) = 3` extra parameters.
Both :math:`\hat\Theta` must be likelihood maxima, so the posterior medians are refined with
``refine_to_mle``.
The chains are in the rotated basis, so the refinement and the test use ``delta_t_rot``;
:func:`~sys_mapping.correction.transform_params_from_rotated` maps the maximum back.
:func:`~sys_mapping.model_selection.likelihood_ratio_test` reads the p-value from
:math:`\chi^2(r)` by default, which assumes independent pixels and is too small on a clustered
field.

.. code-block:: python

   n_sys = delta_t.shape[0]
   theta_add = sm.refine_to_mle(
       sm.posterior_median_params(results["MCMC-add"]["flat_chain"]),
       delta_g, delta_t_rot, model="additive")
   theta_comb = sm.refine_to_mle(
       sm.posterior_median_params(results["MCMC-comb"]["flat_chain"]),
       delta_g, delta_t_rot, model="combined")

   lrt = sm.likelihood_ratio_test(delta_g, delta_t_rot, theta_add, theta_comb,
                                  null_model="additive", alt_model="combined")
   print(f"lambda_LR = {lrt.lambda_lr:.1f}, dof = {lrt.n_dof}, chi2 p = {lrt.p_value:.1e}")

   a_mle, b_mle = sm.transform_params_from_rotated(
       theta_comb[:n_sys], theta_comb[n_sys:2 * n_sys], R)
   print("maximum, template basis: a =", a_mle.round(3), " b =", b_mle.round(3))

.. code-block:: text

   lambda_LR = 102.4, dof = 3, chi2 p = 4.6e-22
   maximum, template basis: a = [ 0.055 -0.024  0.08 ]  b = [ 0.044 -0.013 -0.041]

:func:`~sys_mapping.model_selection.lrt_from_maxima` computes :math:`\lambda_{\rm LR}` on
every null realisation at once, from least-squares and L-BFGS maxima vectorised over the
fields.
Passed as ``null_lambda``, it gives a p-value from the empirical tail instead of
:math:`\chi^2(r)`.
The 100 fields take about 4 s including compilation.

.. code-block:: python

   null_lambda = sm.lrt_from_maxima(null, delta_t_rot)["lambda"]
   lrt_mock = sm.likelihood_ratio_test(delta_g, delta_t_rot, theta_add, theta_comb,
                                       null_model="additive", alt_model="combined",
                                       null_lambda=null_lambda)
   print(f"null median lambda = {np.median(null_lambda):.1f}, mock p = {lrt_mock.p_value:.3f}")

.. code-block:: text

   null median lambda = 26.3, mock p = 0.079

In this run the null :math:`\lambda_{\rm LR}` had a median of 26, against 2.37 for
:math:`\chi^2(3)`, and the mock-calibrated test did not reject the additive model at 5%.

----

Step 9 — Two-point correction with cross-template terms
-------------------------------------------------------

With debiased amplitude matrices
:math:`\tilde A = \hat a\hat a^\top - {\rm Cov}[\hat a]` and
:math:`\tilde B = \hat b\hat b^\top - {\rm Cov}[\hat b]`, each projected onto positive
semi-definite matrices (:func:`~sys_mapping.correction.debias_params_matrix`), the corrected
correlation function is

.. math::

   w_{\rm corr}(\theta) =
   \frac{\hat w(\theta) - \sum_{ij} \tilde A_{ij}\,\xi_{ij}(\theta)}
        {1 + \sum_{ij} \tilde B_{ij}\,\xi_{ij}(\theta)},

where :math:`\xi_{ij}` is the template correlation matrix.
The PCA rotation removes :math:`\xi_{ij}` at zero separation only, so the off-diagonal terms
remain at :math:`\theta > 0`.
:func:`~sys_mapping.utils.template_correlation_matrix` measures all of
:math:`\xi_{ij}(\theta)` with TreeCorr, and
:func:`~sys_mapping.correction.correct_two_point_function` applies the full form when given
the :math:`(n_{\rm sys}, n_{\rm sys}, n_\theta)` matrix with ``cov_a`` and ``cov_b``; it equals
:func:`~sys_mapping.contamination.compute_two_point_correction` on the debiased matrices.
Given :math:`(n_{\rm sys}, n_\theta)` autos it keeps the diagonal only.

Here :math:`\hat w` and :math:`\xi_{ij}` are measured on pixel centres, above the 1.8 deg pixel
scale.
With a catalogue, pass the template values each galaxy carries.

.. code-block:: python

   ra, dec = hp.pix2ang(NSIDE, np.flatnonzero(good), lonlat=True)
   bins = dict(min_sep=4.0, max_sep=40.0, nbins=8, sep_units="deg")
   theta, w_obs = sm.measure_kk_correlation_treecorr(ra, dec, delta_g, **bins)
   _, xi = sm.template_correlation_matrix(ra, dec, delta_t, **bins)   # (3, 3, 8)

   var_a, var_b = np.diag(cov_a), np.diag(cov_b)
   w_auto = sm.correct_two_point_function(w_obs, a_hat, b_hat, var_a, var_b,
                                          np.einsum("iik->ik", xi))
   w_full = sm.correct_two_point_function(w_obs, a_hat, b_hat, var_a, var_b, xi,
                                          cov_a=cov_a, cov_b=cov_b)

   A, B = sm.debias_params_matrix(a_hat, b_hat, cov_a, cov_b)
   assert np.allclose(w_full, sm.compute_two_point_correction(w_obs, A, B, xi))

   _, w_true = sm.measure_kk_correlation_treecorr(ra, dec, delta_g_true[good], **bins)
   for row in zip(theta, w_obs, w_auto, w_full, w_true):
       print("{:5.1f} deg  obs {:.4f}  auto {:.4f}  full {:.4f}  true {:.4f}".format(*row))

.. code-block:: text

     4.6 deg  obs 0.1241  auto 0.1210  full 0.1210  true 0.1235
     6.1 deg  obs 0.0985  auto 0.0959  full 0.0958  true 0.0969
     8.3 deg  obs 0.0738  auto 0.0717  full 0.0716  true 0.0720
    11.1 deg  obs 0.0560  auto 0.0544  full 0.0543  true 0.0550
    14.8 deg  obs 0.0385  auto 0.0372  full 0.0372  true 0.0367
    19.9 deg  obs 0.0200  auto 0.0191  full 0.0191  true 0.0185
    26.6 deg  obs 0.0037  auto 0.0032  full 0.0031  true 0.0021
    35.7 deg  obs -0.0065  auto -0.0067  full -0.0067  true -0.0066

The three synthetic templates are independent, and the cross terms changed the correction by
at most :math:`10^{-4}` in this run; on correlated survey maps they are a larger share of it.
``return_cov=True`` also propagates the amplitude and measurement covariances into the
covariance of :math:`w_{\rm corr}`.

Recovery over many realisations is in :doc:`results_mock_analysis`.
