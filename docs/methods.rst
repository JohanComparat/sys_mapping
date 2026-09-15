Methods Reference
=================

This page gives the equations each function implements, its defaults, and the
publication it follows.  Notation is defined on :doc:`overview`; references are on
:doc:`bibliography`; timings are on :doc:`results_benchmark`.

.. contents::
   :local:
   :depth: 2

----

.. _template-preselection:

Template pre-selection
----------------------

:func:`~sys_mapping.diagnostics.snr_template_ranking` scores every template with one
of four statistics (``method=``, default ``"template"``):

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Key
     - Statistic
   * - ``"template"``
     - Single-template least squares, :math:`\hat\alpha_i = t_i\cdot\hat\delta_g / t_i\cdot t_i`,
       scored as :math:`|\hat\alpha_i| / \sqrt{\hat\sigma_i^2 / t_i\cdot t_i}` with
       :math:`\hat\sigma_i^2` the mean squared residual.
   * - ``"data"``
     - Absolute Pearson correlation :math:`|r(\hat\delta_g, t_i)|`.
   * - ``"peak"``
     - :math:`\max_\ell |\hat C_\ell^{\delta t_i}|` over
       :math:`{\rm median}_\ell |\hat C_\ell^{\delta\delta}|`, from ``healpy.anafast`` with
       :math:`\ell_{\max} = 3\,{\rm NSIDE} - 1` on a full-sky map.
   * - ``"isd"``
     - The binned :math:`\Delta\chi^2` of Rodríguez-Monroy et al. 2025, Eqs. 4–5
       (see :ref:`below <isd-statistic>`).

:func:`~sys_mapping.model_selection.snr_preselect` sorts templates by the statistic
and applies ``snr_min`` and ``n_top``.
:func:`~sys_mapping.diagnostics.isd_template_significance` turns the ``"isd"``
statistic into a p-value against ``n_mocks`` uncontaminated GLASS realisations on the
footprint,

.. math::

   p_i = \frac{1 + \#\{k : \Delta\chi^2_{k,i} \ge \Delta\chi^2_i\}}{1 + n_{\rm mocks}}.

``run_decontamination(..., preselect=True)`` ranks with ``preselect_method``
(default ``"isd"``), computes these p-values on ``preselect_n_mocks`` (100)
realisations with ``preselect_rand_factor`` (2) randoms per galaxy, keeps the templates
with :math:`p \le` ``preselect_p_threshold`` (0.05), or the top-ranked one if none
passes, and reuses the realisations to calibrate the ISD threshold:

.. code-block:: python

   import sys_mapping as sm

   cl = sm.load_matched_cl(match_dir, sample_id, nside)
   result = sm.run_decontamination(
       "ISD-1", delta_g, delta_t,
       preselect=True, preselect_method="isd",
       preselect_n_mocks=100, preselect_p_threshold=0.05,
       preselect_cl_input=cl,
       good_pixels=good_pix, n_total_footprint=len(ra_gal),
       z_edges=z_edges, nz=nz, nside=nside,
   )
   result["preselect_indices"], result["a_hat"]

A p-value floor of :math:`1/(1 + N)` sets the number of realisations; as a rule,
:math:`N_{\rm mocks} \ge \max(20, \lceil 5/\alpha\rceil)` for a threshold
:math:`\alpha`.  :func:`~sys_mapping.model_selection.greedy_forward_select` adds
templates one at a time by a :math:`\chi^2(1)` likelihood-ratio gate on the OLS maxima.
:doc:`results_snr_preselection` validates the procedure.

----

Contamination model
-------------------

Reference: `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_, Eqs. 11–13.

The combined forward model and its inverse are

.. math::

   \hat{\delta}_{g}(p) = \delta_{g}(p)\left(1 + \sum_i b_i\,t_i(p)\right) + \sum_i a_i\,t_i(p),
   \qquad
   \delta_g(p) = \frac{\hat{\delta}_g(p) - \sum_i a_i\,t_i(p)}{1 + \sum_i b_i\,t_i(p)},

implemented by :func:`~sys_mapping.contamination.apply_contamination` and
:func:`~sys_mapping.contamination.invert_contamination`.  The ``additive`` model fixes
:math:`b = 0`, the ``multiplicative`` model fixes :math:`a = 0`, and ``combined`` frees
both.  :func:`~sys_mapping.contamination.pack_params` lays the parameters out as

.. code-block:: text

   additive:        [a_0 .. a_{n-1}, sigma, (gamma)]
   multiplicative:  [b_0 .. b_{n-1}, sigma, (gamma)]
   combined:        [a_0 .. a_{n-1}, b_0 .. b_{n-1}, sigma, (gamma)]

with :math:`\gamma` present only for the skew-normal likelihood.

The weight each method writes is set in
:func:`~sys_mapping.regression.run_decontamination`.  OLS and ElasticNet use the
additive weight of Weaverdyck & Huterer 2021,

.. math::

   \delta_g^{\rm corr}(p) = w(p)\,(1 + \hat{\delta}_g(p)) - 1, \qquad
   w(p) = \frac{1}{1 + \hat{\mathbf a}\cdot\mathbf{t}(p)},

with the denominator floored at :math:`10^{-6}`.  ISD multiplies the per-step weights
:math:`1/(1 + \hat F_j(t_j))`.  MCMC-add and MCMC-comb use the exact inverse
:math:`w = (1 + \delta_g^{\rm clean})/(1 + \hat\delta_g)`, with :math:`1 + \hat\delta_g`
floored at :math:`10^{-6}`, which cancels the contamination when the amplitudes are
the true ones.  Every weight is clipped to :math:`[1/20, 20]`.

:func:`~sys_mapping.contamination.apply_nonlinear_contamination` injects
:math:`1 + \delta_{\rm obs} = (1 + \delta_g)\prod_i (1 + F_i(t_i))` with a per-template
response :class:`~sys_mapping.contamination.TemplateResponse` (``linear``,
``quadratic``, ``cubic``, ``tanh``, ``threshold``, ``exp``), rescaled to a target rms
and floored at an efficiency of 0.05.  It is for mocks; the inference models are the
linear ones above.

----

Likelihood
----------

Reference: Berlfein et al. 2024, Eqs. 17–18.

With independent Gaussian pixels,

.. math::

   \ln\mathcal{L}(\boldsymbol\Theta) =
   -\frac{N}{2}\ln(2\pi\sigma^2)
   - \sum_p \ln\left|1 + \sum_i b_i\,t_i(p)\right|
   - \frac{1}{2\sigma^2}\sum_p \left[\frac{\hat\delta_g(p) - \sum_i a_i t_i(p)}{1+\sum_i b_i t_i(p)}\right]^2,

where the second term is the Jacobian of the change of variables and vanishes for the
additive model.  The skew-normal likelihood is the density
:math:`\prod_p (2/\sigma)\,\phi(r_p/\sigma)\,\Phi(\gamma r_p/\sigma)` of the shifted
residual :math:`r_p = \delta_g(p) - \xi`:

.. math::

   \ln\mathcal{L}_{\rm skew} = \ln\mathcal{L}_{\rm Gauss}(r) + N\ln 2
       + \sum_p \ln\Phi\left(\gamma\,\frac{r_p}{\sigma}\right),
   \qquad
   \xi = -\sigma\,\frac{\gamma}{\sqrt{1+\gamma^2}}\sqrt{2/\pi},

so that :math:`\mathbb{E}[\delta_g] = 0`.  :math:`\sigma` is the scale, with
:math:`{\rm Var}[\delta_g] = \sigma^2(1 - 2\gamma^2/(\pi(1+\gamma^2)))`, and
:math:`\ln\Phi` is ``jax.scipy.special.log_ndtr``.

:func:`~sys_mapping.likelihood.make_log_likelihood` returns a JIT-compiled
``log_likelihood(theta, delta_g_obs, delta_t)``, cached per ``(n_sys, model,
use_skewed)``.  A ``precision`` operator :math:`R` from
:func:`~sys_mapping.covariance.build_lowrank_precision`
(:class:`~sys_mapping.covariance.LowRankPrecision`, :math:`R = D + UU^\top` with unit
diagonal, applied by the Woodbury identity) turns the quadratic form into
:math:`r^\top R^{-1} r` and adds :math:`-\tfrac12\ln|R|`.  With ``use_skewed=True`` a
precision operator raises ``ValueError``, because the product of :math:`N` univariate
factors is a density only for independent pixels.

The :math:`\sigma^2 I` covariance leaves the point estimates unbiased, but the galaxy
field is spatially correlated and smooth templates project onto it with more variance
than white noise, so the likelihood's parameter errors, any
:math:`|\hat a_i|/\sigma_{\hat a_i}` built from them, and the Wilks calibration of the
:ref:`likelihood ratio <lrt-methods>` are too narrow.  The calibrated alternatives are:

* :func:`~sys_mapping.covariance.mock_sandwich_covariance`,
  :math:`{\rm Cov}(\hat a) = (TT^\top)^{-1}(TCT^\top)(TT^\top)^{-1}` with :math:`TCT^\top`
  the covariance of the projected uncontaminated realisations;
* the correlated-noise likelihood above (``pixel_precision=`` in
  ``run_decontamination``, which forces NUTS);
  :func:`~sys_mapping.covariance.build_harmonic_precision`, the full-rank harmonic
  version, raises ``NotImplementedError``;
* :ref:`calibrated-significance` and the mock-calibrated likelihood ratio.

----

Posterior sampling
------------------

Reference: Berlfein et al. 2024, Sec. 4.

Prior and dispatch
~~~~~~~~~~~~~~~~~~

The posterior is :math:`p(\boldsymbol\Theta \mid \hat{\boldsymbol\delta}_g) \propto
\mathcal{L}(\boldsymbol\Theta)\,\pi(\boldsymbol\Theta)` with a flat prior on
:math:`a`, :math:`b` and :math:`\gamma` and on :math:`\sigma > 0`.
:func:`~sys_mapping.nuts.build_logdensity` accepts optional zero-mean Gaussian priors
``prior_scale_a`` and ``prior_scale_b``, both ``None`` by default.

``run_decontamination(method, ..., sampler=...)`` selects the backend for MCMC-add and
MCMC-comb:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - ``sampler``
     - Backend
   * - ``"auto"`` (default)
     - :func:`~sys_mapping.inference.run_additive_analytic` for the Gaussian additive
       model, :func:`~sys_mapping.nuts.run_nuts` otherwise.
   * - ``"analytic"``
     - As ``"auto"``: the analytic posterior exists only for the Gaussian additive model.
   * - ``"nuts"``
     - :func:`~sys_mapping.nuts.run_nuts` for both.
   * - ``"emcee"``
     - :func:`~sys_mapping.inference.run_mcmc`.

A ``pixel_precision`` operator forces ``"nuts"``, and ``use_skewed`` applies to
MCMC-comb only.  Both methods rotate the templates first (:ref:`Template PCA rotation`),
take the per-parameter posterior median
(:func:`~sys_mapping.inference.posterior_median_params`), and return the chain, the
rotated and back-transformed amplitudes, their covariances
(:func:`~sys_mapping.inference.get_param_covariance_from_chain`, ``ddof=1``,
back-transformed as :math:`V\,{\rm Cov}'\,V^\top`), ``sigma_hat``, ``gamma_hat``,
``acceptance_fraction``, ``rhat``, ``ess`` and ``num_divergences``.  ``rhat`` and ``ess``
are ``None`` for the analytic and emcee draws.

Analytic additive posterior
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The additive model :math:`\hat\delta_g = X a + \varepsilon`,
:math:`\varepsilon \sim \mathcal N(0,\sigma^2 I)`, :math:`X = T^\top`, is linear
regression, and under flat priors on :math:`a` and :math:`\sigma` its posterior is
Normal-Inverse-Gamma:

.. math::

   \sigma^2 \mid {\rm data} \sim \mathrm{Inv\text{-}Gamma}\left(
     \tfrac{N-n_s-1}{2},\ \tfrac{\rm RSS}{2}\right),\qquad
   a \mid \sigma^2,{\rm data} \sim \mathcal N\left(\hat a_{\rm OLS},\
     \sigma^2 (X^\top X)^{-1}\right).

:func:`~sys_mapping.inference.run_additive_analytic` draws ``n_samples`` (100 000)
independent samples of this hierarchy through a Cholesky factor of :math:`X^\top X`.

NUTS
~~~~

:func:`~sys_mapping.nuts.run_nuts` samples the unconstrained vector with
:math:`\sigma = e^{u_\sigma}` and adds the log-Jacobian :math:`u_\sigma`, which
reproduces the flat prior on :math:`\sigma`.  BlackJAX window adaptation runs for
``n_warmup`` (1000) steps at ``target_acceptance_rate`` 0.8, then ``n_samples`` (1000)
NUTS steps per chain run under ``jax.lax.scan``.  ``n_chains`` defaults to 4 on CPU and
8 on GPU (:func:`~sys_mapping.nuts.default_n_chains`); ``chain_method="vmap"`` runs the
chains together and ``"sequential"`` one at a time for less memory.  Chains start at
amplitudes drawn from :math:`\mathcal N(0, 0.05)` and
:math:`u_\sigma = \ln{\rm std}(\hat\delta_g) + \mathcal N(0, 0.1)`.  ``rhat`` is the
maximum over parameters (undefined for one chain) and ``ess`` the minimum.

``dense_mass_matrix=True`` (default) adapts a dense mass matrix rather than a diagonal
one, because the amplitudes are strongly correlated.  On the LS10 combined fit at
NSIDE 32 we measured half the leapfrog steps per iteration and 2.5 times the minimum
effective sample size per second.

emcee
~~~~~

:func:`~sys_mapping.inference.run_mcmc` runs the affine-invariant ensemble sampler
(Goodman & Weare 2010) on :func:`~sys_mapping.inference.make_log_prob`, which returns
:math:`-\infty` for :math:`\sigma \le 10^{-6}`; ``vectorize=True`` evaluates all walkers
in one ``jax.vmap`` call.  Its defaults are ``n_walkers=250``, ``n_steps=1500``,
``n_burn=300``; ``run_decontamination`` passes ``n_walkers=100``, ``n_steps=1200``,
``n_burn=200``, raising the walker count to :math:`2n_{\rm dim} + 2` when needed.
Walkers start at amplitudes drawn from :math:`\mathcal N(0, 0.05)`,
:math:`\sigma = {\rm std}(\hat\delta_g)(1 + \mathcal N(0, 0.1))` and, for the
skew-normal likelihood, :math:`\gamma \sim \mathcal N(0, 0.1)`.

Likelihood maxima
~~~~~~~~~~~~~~~~~

A posterior median is not a likelihood maximum.
:func:`~sys_mapping.inference.refine_to_mle` maximises :math:`\ln\mathcal L` with
L-BFGS-B on :math:`(a, b, \ln\sigma)` for up to ``max_iter`` (500) iterations from two
starts, the supplied point and an analytic one (the OLS amplitudes, with :math:`b = 0`
for the combined model), and keeps the higher.  If neither improves on the supplied point it returns that point
with a warning.

----

.. _Template PCA rotation:

Template PCA rotation
---------------------

Reference: Berlfein et al. 2024, Appendix A.

With the template covariance

.. math::

   C_{ij} = \frac{1}{N}\sum_p t_i(p)\,t_j(p) = (V D V^\top)_{ij},

:func:`~sys_mapping.correction.rotate_templates` returns the rotated templates
:math:`\mathbf{T}' = V^\top\mathbf{T}`, i.e. :math:`t'_j(p) = \sum_i V_{ij}\,t_i(p)`,
the rotation matrix :math:`V^\top` (rows are eigenvectors) and the eigenvalues in
descending order.  The rotated covariance is :math:`D`, so the rotated templates are
orthogonal over the footprint.  :func:`~sys_mapping.correction.transform_params_from_rotated`
returns :math:`\mathbf a = V\mathbf a'` and :math:`\mathbf b = V\mathbf b'`.  The
eigenvalues count the independent modes; the rotation diagonalises the covariance at
zero separation only, and :math:`\xi_{ij}(\theta) \ne 0` for :math:`i \ne j` at
:math:`\theta > 0`.

----

Noise debiasing
---------------

Reference: Berlfein et al. 2024, Eq. 21.

Since :math:`\mathbb{E}[\hat a_i \hat a_j] = a_i a_j + {\rm Cov}_{ij}[\hat a]`, squared
amplitudes are biased high by the estimator's covariance.
:func:`~sys_mapping.correction.debias_params_matrix` returns

.. math::

   \tilde A = \sum_k \max(\lambda_k, 0)\, v_k v_k^\top,
   \qquad
   \hat{\mathbf a}\hat{\mathbf a}^\top - {\rm Cov}[\hat{\mathbf a}] = \sum_k \lambda_k\, v_k v_k^\top,

and :math:`\tilde B` from :math:`\hat{\mathbf b}` and :math:`{\rm Cov}[\hat{\mathbf b}]`.
Clipping the eigenvalues projects onto positive semi-definite matrices, the constraint
on an outer product.  :func:`~sys_mapping.correction.debias_params` is the diagonal
form,

.. math::

   \tilde a_i^2 = \max\left(\hat a_i^2 - {\rm Var}[\hat a_i], 0\right),
   \qquad
   \tilde b_i^2 = \max\left(\hat b_i^2 - {\rm Var}[\hat b_i], 0\right),

and the two agree for one template.

----

Two-point function correction
-----------------------------

Reference: Berlfein et al. 2024, Eqs. 15–16.

Inserting the contamination model into
:math:`\hat w(\theta) = \langle\hat\delta_g(\hat n)\,\hat\delta_g(\hat n')\rangle_\theta`,
with templates uncorrelated with :math:`\delta_g`, gives to second order in the
amplitudes

.. math::

   \hat{w}(\theta) \approx
     w_g(\theta)\left[1 + \sum_{ij} b_i b_j\,\xi_{ij}(\theta)\right]
     + \sum_{ij} a_i a_j\,\xi_{ij}(\theta),

and with the debiased matrices

.. math::

   \hat{w}_{\rm corr}(\theta) =
   \frac{\hat{w}(\theta) - \sum_{ij} \tilde A_{ij}\,\xi_{ij}(\theta)}
        {1 + \sum_{ij} \tilde B_{ij}\,\xi_{ij}(\theta)}.

:func:`~sys_mapping.contamination.compute_two_point_correction` takes 1-D squared
amplitudes with ``(n_sys, n_bins)`` auto-correlations, or 2-D matrices with the
``(n_sys, n_sys, n_bins)`` matrix, and refuses mismatched ranks.
:func:`~sys_mapping.correction.correct_two_point_function` debiases and corrects: with
the 3-D matrix it uses ``cov_a`` and ``cov_b`` (or ``diag(var_a)`` and
``diag(var_b)``) in :func:`~sys_mapping.correction.debias_params_matrix`, and with the 2-D
form it uses :func:`~sys_mapping.correction.debias_params`.  It warns in every bin
where :math:`\hat w > 0` and :math:`\hat w_{\rm corr} < 0`.  With ``return_cov=True``
it draws ``n_mc`` (4000) amplitude vectors from their covariances, and :math:`\hat w`
from ``cov_w_obs`` when given, and returns the sample covariance of the corrected
curves.  With the OLS amplitudes of the nine LS10 samples, in the rotated basis, the
cross-template terms carry a median 23.9 % of the correction at NSIDE 32 and 22.5 % at
NSIDE 64 (:doc:`results_algorithm_characterisation`).

:func:`~sys_mapping.correction.estimate_overcorrection_bias` measures the bias a
weighting method imparts on uncontaminated realisations (Weaverdyck et al. 2026,
Eqs. 21–23), and :func:`~sys_mapping.correction.debias_two_point_function` subtracts
it.  :func:`~sys_mapping.correction.correct_power_spectrum_harmonic` is the harmonic
counterpart.

Template correlation matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~sys_mapping.utils.template_correlation_matrix` measures
:math:`\xi_{ij}(\theta)` with TreeCorr ``KKCorrelation`` (defaults 0.5 to 300 arcmin,
30 bins, ``bin_slop=0.01``).  The inputs are the template values the galaxies carry,
because pixel centres are never closer than the pixel scale while a template is
constant within a pixel.  Correlations are bilinear, so after the :math:`n_s` auto
passes the cross terms follow from auto-correlations of summed fields:

.. math::

   \xi_{ij}(\theta) = \tfrac12\left[\xi_{t_i + t_j}(\theta) - \xi_{ii}(\theta) - \xi_{jj}(\theta)\right].

An auto pass counts each pair once where a cross pass counts it twice; at 11 templates
the matrix is 1.7 times faster than with cross passes, and the result is exact.
Optional per-object weights ``w=`` apply to every correlation, and ``max_points``
subsamples the galaxies without replacement.

----

.. _lrt-methods:

Likelihood ratio test
---------------------

Reference: Berlfein et al. 2024, Eq. 19; Wilks 1938.

For nested models :math:`\mathcal{M}_0 \subset \mathcal{M}_1` with maxima
:math:`\hat\Theta_0` and :math:`\hat\Theta_1`,

.. math::

   \lambda_{\rm LR} = 2\left[\ln\mathcal{L}(\hat{\Theta}_1) -
                             \ln\mathcal{L}(\hat{\Theta}_0)\right]
   \;\xrightarrow{N\to\infty}\; \chi^2(r), \quad r = k_1 - k_0,

with :math:`r = n_s` for additive against combined.  The statistic is non-negative only
at the maxima; :func:`~sys_mapping.model_selection.likelihood_ratio_test` warns when it
is negative, and the parameter vectors should come from
:func:`~sys_mapping.inference.refine_to_mle`.

Wilks' theorem assumes independent observations.  On a correlated field the statistic
is inflated under the null and ``chi2.sf(lambda, r)`` is too small, so
``likelihood_ratio_test`` accepts ``null_lambda``, the statistic on uncontaminated
realisations fitted the same way, and returns

.. math::

   p = \frac{1 + \#\{\lambda_{\rm null} \ge \lambda_{\rm LR}\}}{1 + N_{\rm null}}

with ``calibration="mock"``.  :func:`~sys_mapping.model_selection.lrt_null_distribution`
builds that null from ``(n_pix, n_mock)`` fields and a ``fit_theta(model, delta_g,
delta_t)`` callback.

Batched maxima
~~~~~~~~~~~~~~

:func:`~sys_mapping.model_selection.lrt_from_maxima` computes
:math:`\lambda_{\rm LR}` for ``(n_field, n_pix)`` fields together.  The additive maximum
is the least-squares solution in closed form,
:math:`\hat a = (TT^\top)^{-1}T\hat\delta_g` with :math:`\hat\sigma^2` the mean squared
residual (refined by L-BFGS when ``use_skewed``).  The combined maximum is found by
L-BFGS (optax), stopped when the largest gradient component falls below
:math:`10^{-9}` or after ``n_iter`` (300) iterations, on :math:`(a, b, \ln\sigma)` started from
:math:`(\hat a_{\rm OLS}, b = 0)`, a point on the combined model's ridge.  The combined
maximum is sought where every pixel's efficiency :math:`1 + b\cdot t(p)` is positive, the
region holding :math:`b = 0`, since the likelihood has a pole wherever it vanishes; each
L-BFGS run keeps its best finite point, stops once the value or gradient is not finite,
and is restarted from that point up to three times.  The
optimisation is ``jax.vmap``-ped over fields in batches of ``batch_size`` (64), compiled
once per ``(n_sys, use_skewed, n_iter)``, and a field whose largest gradient component
exceeds ``grad_tol`` (:math:`10^{-3}`) is flagged as not converged.  For 8 realisations
of 7 040 pixels with 11 templates we measured 0.15 s after compilation, against about
50 s for per-realisation NUTS fits refined to their maxima, with :math:`\lambda_{\rm LR}` equal to a relative
:math:`8\times10^{-7}`.

----

.. _calibrated-significance:

Calibrated template significance
--------------------------------

:func:`~sys_mapping.diagnostics.calibrated_template_significance` scores the
least-squares amplitudes :math:`\hat a = (T^\top)^{+}\hat\delta_g` against ``null_delta``,
:math:`N_{\rm null} \ge n_s + 3` uncontaminated realisations with amplitudes
:math:`A_{ki}`:

.. math::

   S_i = \frac{|\hat a_i|}{\sigma_i},
   \qquad
   \sigma_i = {\rm std}_k(A_{ki}),
   \qquad
   p_i = \frac{1 + \#\{k : |A_{ki}| \ge |\hat a_i|\}}{1 + N_{\rm null}}.

The largest significance is compared with the largest of each realisation,

.. math::

   p_{\rm FW} = \frac{1 + \#\{k : \max_i |A_{ki}|/\sigma_i^{(-k)} \ge \max_i S_i\}}{1 + N_{\rm null}},

where :math:`\sigma_i^{(-k)}` omits realisation :math:`k`, which accounts for the
number of templates and their correlation.  The null must carry the sample's
clustering (:ref:`matched-spectrum`).

Over 2000 uncontaminated fields, each scored against 1000 realisations, we measured
``family_wise_p`` :math:`\le 0.05` in :math:`4.85 \pm 0.48` % of fields and
:math:`\le 0.0027` in :math:`0.35 \pm 0.13` %.  On the same fields the
independent-pixel significance exceeded 3 on some template in 99.3 %.  The LS10 script
records :math:`\sigma_i/\sigma_i^{\rm iid}` per template as ``inflation``.

----

Residual template correlation test
----------------------------------

:func:`~sys_mapping.diagnostics.residual_template_correlation_test` measures the
Pearson correlation :math:`r_i` of a corrected field with each template and calibrates
it on ``null_delta``:

.. math::

   \chi^2 = \sum_i \frac{r_i^2}{{\rm Var}_{\rm null}[r_i]},
   \qquad
   p = \frac{1 + \#\{k : \chi^2_k \ge \chi^2\}}{1 + N_{\rm null}},

with each :math:`\chi^2_k` using a variance that omits realisation :math:`k`.  The
corrected density is tested rather than the weights, which depend on every fitted
template by construction.  The null realisations must pass through the identical
correction, and the tested templates must be ones the correction did not fit, since a
least-squares residual is orthogonal to its regressors.

----

Null realisations
-----------------

.. _matched-spectrum:

Matched spectrum
~~~~~~~~~~~~~~~~

A null reproduces the large-scale clustering of one sample on one footprint.
:func:`~sys_mapping.glass_mocks.load_matched_cl` reads the ``cl_matched`` spectrum of a
``*_match.json`` written by ``match_glass_to_data.py`` (``sys_mapping_benchmark``), or
picks one from a directory by ``sample`` and ``nside``: the file at ``nside`` if it
passed validation, otherwise the nearest validated file at or above ``nside``,
otherwise the finest validated file below it.  It returns ``None`` when no file
matches and, with ``require_validated=True`` (default), raises ``ValueError`` for a file
that failed the large-scale check or has no validation block.
:func:`~sys_mapping.glass_mocks.sanitise_cl` truncates the spectrum at
:math:`\ell = 3\,{\rm NSIDE}` or extends it with its last value, zeroes the monopole
and floors it at :math:`10^{-8}` of its peak.

Without ``cl_input`` or ``cl_amplitude`` the GLASS generators warn and use
:math:`5\times10^{-4}(\ell+1)^{-1.5}`, which gives
:math:`\sigma_{\rm clus} \approx 0.08` against LS10's :math:`\approx 0.39`, so its
p-values are anticonservative.  ``scripts/run_ls10_analysis.py`` therefore stops when
a null is needed and ``--null-cl-file`` provides no spectrum, unless
``--allow-parametric-null`` is given.

Pixel draws
~~~~~~~~~~~

:func:`~sys_mapping.glass_mocks.draw_null_overdensity` draws a GLASS lognormal field
:math:`\delta` (:func:`~sys_mapping.glass_mocks.generate_glass_delta_map`) and, at the
:math:`N_{\rm good}` footprint pixels,

.. math::

   N_g(p) \sim {\rm Poisson}\left(\bar n\,\max(1 + \delta(p), 0)\right),
   \qquad
   N_r(p) \sim {\rm Poisson}(r\,\bar n),
   \qquad
   \bar n = N_{\rm footprint}/N_{\rm good},

with :math:`r` = ``rand_factor`` (2), then reduces the counts as
:func:`~sys_mapping.maps.compute_overdensity` does,
:math:`\hat\delta = N_g/(f N_r) - 1` with :math:`f = \sum N_g / \sum N_r` (zero where
:math:`N_r = 0`).  The field uses ``seed`` and the counts an independent stream.
:func:`~sys_mapping.glass_mocks.generate_glass_null_overdensity` stacks realisations
with seeds ``seed + k``, so ``k_start`` extends a set reproducibly.  This is the
distribution of a pixelised catalogue mock, since ``glass.positions_from_delta`` draws
a Poisson count per pixel.  At NSIDE 64 on 28 416 pixels we measured 0.01 s per
realisation against 0.97 s for a catalogue mock, with pixel variances of 0.03521 and
0.03523.  ``isd_template_significance(draw=...)`` and the script's ``--null-draw``
select ``"pixel"`` (default) or ``"catalogue"``.

----

Resolution, footprint and template basis
----------------------------------------

Footprint
~~~~~~~~~

:func:`~sys_mapping.maps.compute_overdensity` keeps pixels with
:math:`n_r(p) \ge 0.1\,\max_p n_r` (``min_random_fraction``) and returns
:math:`\hat\delta_g = n_g/(f_r n_r) - 1`, :math:`f_r = \sum n_g/\sum n_r`, on them.

Occupancy
~~~~~~~~~

:func:`~sys_mapping.maps.choose_nside_by_occupancy` evaluates the mean occupancy

.. math::

   \bar n({\rm NSIDE}) = \frac{N_{\rm gal}}{N_{\rm good}({\rm NSIDE})}

from ``nside_max`` downwards by factors of two, with the footprint of
``compute_overdensity``, and returns the first NSIDE with
:math:`\bar n \ge` ``min_per_pixel`` (25), or ``nside_min`` (8) if none qualifies,
together with every :math:`\bar n` examined.  At 25 galaxies the Poisson scatter of a
pixel count is 20 %.  The rule uses the footprint mean, because cutting individual
sparse pixels would carve the mask along the depth and stellar-density variations the
templates describe.  The LS10 samples run at NSIDE 16 to 128.

Standardisation
~~~~~~~~~~~~~~~

:func:`~sys_mapping.maps.standardise_on_footprint` sets each template to zero mean and
unit rms over the fitted pixels,
:math:`t_i \leftarrow (t_i - \mu_i)/s_i` with :math:`s_i^2 = \langle (t_i-\mu_i)^2\rangle`,
leaves a constant row at zero, and with ``return_scales=True`` also returns
:math:`\mu_i` and :math:`s_i`.  Maps normalised over their own valid region are not
standardised on a sample's footprint: on the eleven LS10 maps at NSIDE 64 the rms over
the analysis pixels spans 0.905 to 5.77 and the covariance eigenvalues sum to 44.4
rather than 11.

----

Pseudo-:math:`C_\ell` power spectrum
------------------------------------

References:
`Elsner, Leistedt & Peiris 2016 <https://ui.adsabs.harvard.edu/abs/2016MNRAS.456.2095E/abstract>`_;
`Elsner, Leistedt & Peiris 2017 <https://ui.adsabs.harvard.edu/abs/2017MNRAS.465.1847E/abstract>`_;
`Leistedt & Peiris 2014 <https://ui.adsabs.harvard.edu/abs/2014MNRAS.444....2L/abstract>`_;
`Ho et al. 2012 <https://ui.adsabs.harvard.edu/abs/2012ApJ...761...14H/abstract>`_.

:func:`~sys_mapping.power_spectrum.measure_pseudo_cl` masks the map and calls
``healpy.anafast`` (``use_pixel_weights=True``, :math:`\ell_{\max} = 3\,{\rm NSIDE}-1`):

.. math::

   \tilde{C}_\ell = \frac{1}{2\ell+1}\sum_{m=-\ell}^{\ell}
   \left|\sum_p W(p)\,\hat\delta_g(p)\,Y_{\ell m}^*(p)\,\Omega_p\right|^2,
   \qquad
   \langle\tilde{C}_\ell\rangle = \sum_{\ell'} M_{\ell\ell'}\,C_{\ell'}.

The result is not corrected for the mode-coupling matrix :math:`M_{\ell\ell'}`.
Template subtraction (:func:`~sys_mapping.power_spectrum.subtract_template_cl`) removes
the template power, quadratic in the amplitude,

.. math::

   \tilde{C}_\ell^{\rm TS} = \hat{C}_\ell^{\delta\delta} - \sum_i \hat\alpha_i^2\,\hat{C}_\ell^{t_i t_i},

the harmonic counterpart of :math:`\tilde a_i^2\,\xi_{ii}(\theta)`, and
:func:`~sys_mapping.power_spectrum.harmonic_bias` gives the bias of subtracting
:math:`n_s` templates, :math:`b_\ell = -n_s/(2\ell+1)`.
:func:`~sys_mapping.correction.correct_power_spectrum_harmonic` applies both.

:func:`~sys_mapping.power_spectrum.mode_projection_bias` subtracts :math:`b_\ell`
propagated through ``coupling_matrix`` (identity when ``None``): at every multipole for
``mode="basic"``, and for ``mode="extended"`` only where
:math:`|b_\ell(1)| / (\tilde C_\ell/n_s)` exceeds ``threshold`` (0.1).

----

Spatial patches, bootstrap and jack-knife
-----------------------------------------

References: Berlfein et al. 2024, Sec. 6.2;
`Ross et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.417.1350R/abstract>`_, Sec. 4.3;
Ho et al. 2012.

:func:`~sys_mapping.bootstrap.assign_spatial_patches` labels each footprint pixel with
the coarse HEALPix pixel containing it, at

.. math::

   {\rm NSIDE}_{\rm patch} = 2^{\,{\rm round}\left(\log_2\sqrt{n_{\rm patches}/(12 f_{\rm sky})}\right)},

clipped to :math:`[1, {\rm NSIDE}]`, where :math:`f_{\rm sky}` is the footprint sky
fraction (or an explicit ``nside_patch``).  Labels run from 0 to :math:`K-1` in the
order of ``np.where(good_pixels)[0]``.  The same patches serve as bootstrap blocks,
jack-knife regions, ElasticNet cross-validation groups and the jack-knife of
:func:`~sys_mapping.diagnostics.vet_templates_against_tracer`.

:func:`~sys_mapping.bootstrap.block_bootstrap_variance` draws :math:`K` patches with
replacement ``n_bootstrap`` (100) times, with ``n_patches`` (10), and returns

.. math::

   \widehat{\rm Var}[\hat\theta] = \frac{1}{B-1}\sum_{b=1}^B
   \left(\hat\theta_b - \bar{\hat\theta}\right)^2.

:func:`~sys_mapping.bootstrap.jackknife_covariance` drops one patch at a time and
returns

.. math::

   \widehat{\rm Cov}[\hat\theta] = \frac{K-1}{K}
   \sum_{k=1}^K \left(\hat\theta_{(-k)} - \bar{\hat\theta}\right)
                \left(\hat\theta_{(-k)} - \bar{\hat\theta}\right)^\top,

warning when ``n_patches`` is below 5.

----

ElasticNet regression
---------------------

Reference: `Weaverdyck & Huterer 2021 <https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.5061W/abstract>`_,
Eqs. 9–10.

:func:`~sys_mapping.regression.elasticnet_contamination_fit` minimises, with
scikit-learn,

.. math::

   \frac{1}{2N}\left\|\hat{\boldsymbol\delta}_g - \mathbf{T}^\top\boldsymbol\alpha\right\|_2^2
   + \lambda\rho\,\|\boldsymbol\alpha\|_1
   + \frac{\lambda(1-\rho)}{2}\,\|\boldsymbol\alpha\|_2^2,

with :math:`\rho` = ``l1_ratio`` (0.5) and no intercept.  The :math:`\ell_1` term sets
small amplitudes to zero and the :math:`\ell_2` term shares weight among correlated
templates.  With ``alpha_reg=None`` (default), ``ElasticNetCV`` chooses :math:`\lambda`
by ``cv_folds`` (5) folds of contiguous pixel indices, or, with ``patch_ids`` from
:func:`~sys_mapping.bootstrap.assign_spatial_patches`, by ``GroupKFold`` over whole
patches, so held-out pixels are spatially disjoint from the training set.
``pixel_weights`` enter as ``sample_weight``;
:func:`~sys_mapping.maps.inverse_variance_pixel_weights` builds the DES Y6 form
:math:`W_k \propto A_k^2/(N_k + 2)`, normalised to mean one.  The weights are
:math:`1/\max(1 + \hat{\boldsymbol\alpha}\cdot\mathbf t, 10^{-6})` clipped to
:math:`[1/20, 20]`.  ``run_decontamination("ElasticNet", ...)`` uses index folds and no
pixel weights.  It requires ``scikit-learn >= 1.3`` (``pip install
"sys_mapping[regression]"``).

----

Iterative Systematics Decontamination (ISD)
-------------------------------------------

References:
`Elvin-Poole et al. 2018 <https://arxiv.org/abs/1708.01536>`_ (DES Y1);
`Rodríguez-Monroy et al. 2022 <https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.2665R/abstract>`_ (DES Y3);
`Weaverdyck et al. 2026 <https://arxiv.org/abs/2601.14484>`_ (DES Y6, Sec. III B).

.. _isd-statistic:

Algorithm
~~~~~~~~~

:func:`~sys_mapping.regression.iterative_systematics_decontamination` repeats the
following on the field weighted so far.  For every template :math:`i` independently,
:func:`~sys_mapping.diagnostics.isd_marginal_fit`:

1. bins the footprint into ``n_bins`` (10) bins of :math:`t_i`, equal-occupancy by
   default (``binning="quantile"``; ``"width"`` gives the DES equal-width bins, which
   put most pixels of a skewed template in one bin), and takes the coverage-weighted
   mean overdensity per bin with inverse-variance weight :math:`n_b/{\rm Var}_b`,
   ignoring bins with fewer than two pixels;
2. fits a polynomial of degree ``poly_order`` in that template's value by weighted
   least squares on the square-root-weighted Vandermonde,

   .. math::

      \hat F_i(t) = \sum_{k=0}^{d} c_k^{(i)} t^k,

   an :math:`n_b \times (d+1)` system however many templates there are;
3. scores it as

   .. math::

      S_i = \frac{\Delta\chi^2_i}{\Delta\chi^2_{68}},
      \qquad
      \Delta\chi^2_i = \max(\chi^2_{\rm null} - \chi^2_{\rm model}, 0),

   with :math:`\Delta\chi^2_{68}` (``chi2_68``) the 68th percentile of the same
   statistic on uncontaminated realisations
   (:func:`~sys_mapping.diagnostics.isd_template_significance`).

If :math:`\max_i S_i \geq` ``threshold`` (2.0), the most significant template is
corrected and the loop repeats:

.. math::

   w(p) \longleftarrow \frac{w(p)}{1 + \hat F_j\left({\rm clip}(t_j(p), t_{\rm lo}, t_{\rm hi})\right)},
   \qquad j = \arg\max_i S_i,

where :math:`t_{\rm lo}` and :math:`t_{\rm hi}` are the outermost valid bin centres.
The cleaned field is :math:`w(1 + \hat\delta_g) - 1`.  ``poly_order=1`` is ``ISD-1``
(DES Y1/Y3) and ``poly_order=3`` is ``ISD-3`` (DES Y6).

The loop stops when every template is below the threshold, after ``max_steps``
(default :math:`4n_s`) accepted steps, or when every template has been selected
``max_reuse`` (3) times; ``stopped_on`` records which.  A template the data cannot
resolve is left uncorrected with :math:`\hat a_i = 0`.  Without ``chi2_68`` the
threshold is in raw :math:`\Delta\chi^2` units and the function warns.
``run_decontamination`` takes ``isd_chi2_68``, or the 68th percentile of the
pre-selection null when pre-selection ran with ``"isd"``.

The fit is held constant outside :math:`[t_{\rm lo}, t_{\rm hi}]`, because
survey-property maps are skewed: LS10's ``GALDEPTH_Z`` reaches :math:`+26` standardised
units while its outermost bin centre sits near :math:`+2`.  Where :math:`1 + \hat F`
falls below :math:`1/w_{\max}` in at most ``bad_pixel_frac`` (1 %) of the pixels the
denominator is floored there; above that fraction the step is refused, the template
retired, and a warning points to
:func:`~sys_mapping.diagnostics.footprint_mask_diagnostics`.  The running weight is
clipped to :math:`[1/w_{\max}, w_{\max}]` with :math:`w_{\max} = 20`.

Reported amplitude
~~~~~~~~~~~~~~~~~~

``a_hat[i]`` is the least-squares projection of the removed curve on the template,

.. math::

   \hat a_i = \sum_{\rm steps}
     \frac{\langle \hat F_i({\rm clip}(t_i))\, t_i \rangle}{\langle t_i^2 \rangle},

summed over the steps that selected template :math:`i`, so that
:math:`\sum_i \hat a_i t_i` reproduces the removed contamination; ``b_hat`` is zero.
The polynomial coefficients are kept in ``steps``.  The projection is used because
equal-occupancy bin centres of a skewed template span a narrow range and make the
cubic coefficients large: on an uncontaminated GLASS field one accepted step returned
:math:`c = (0.026, 0.681, 3.343, 4.365)`, while the projection is
:math:`6\times10^{-4}`.

Linear against cubic
~~~~~~~~~~~~~~~~~~~~

We injected :math:`F = 0.10\,t + 0.04\,t^2 - 0.02\,t^3` on one of four templates
(40 000 pixels, ``chi2_68=50``) and measured the :math:`\Delta\chi^2` left against that
template with a linear and a cubic probe:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Field
     - fitted coefficients
     - residual, linear probe
     - residual, cubic probe
   * - uncorrected
     - —
     - 946
     - 2049
   * - after ``ISD-1``
     - (0.052)
     - 37.7
     - 1087
   * - after ``ISD-3``
     - (0.105, 0.043, −0.025)
     - 5.6
     - 34.7

``ISD-3`` recovers the three injected coefficients and reduces the non-linear residual
by a factor of 31, while ``a_hat`` is 0.048 for ``ISD-1`` and 0.049 for ``ISD-3``.

Choosing a method
~~~~~~~~~~~~~~~~~

Marginal fits miss contamination that appears only as a linear combination of
templates, and a simultaneous linear fit misses curvature.  DES Y6 runs both and
marginalises over the difference with
:func:`~sys_mapping.covariance.method_marginalised_covariance`, which adds
:math:`\Delta\Delta^\top` with :math:`\Delta = \hat w^A - \hat w^B` to the data
covariance.  :func:`~sys_mapping.regression.polynomial_ols_decontamination` is a
different algorithm, iteratively reweighted least squares on all monomials of the
templates jointly, and is not called by ``run_decontamination``.

----

Null tests and template ranking
-------------------------------

References: Ross et al. 2011, Sec. 3.2;
`Tanidis et al. 2026 <https://ui.adsabs.harvard.edu/abs/2026MNRAS.547ag537T/abstract>`_;
`Rodríguez-Monroy et al. 2025 <https://ui.adsabs.harvard.edu/abs/2025arXiv250907943R/abstract>`_.

:func:`~sys_mapping.diagnostics.null_test_cross_correlations` returns the signed
Pearson correlation between the weights and each template and a permutation p-value
:math:`(1 + \#\{|r_{\rm perm}| \ge |r|\})/(1 + n_{\rm bootstrap})` over
``n_bootstrap`` (100) shuffles of the weights, vectorised with ``jax.vmap``.  The
statistic depends on which amplitudes are non-zero, not on their size: with one
non-zero amplitude :math:`w` is a monotone function of that template and
:math:`|r| \to 1`.  To test for residual contamination use
:func:`~sys_mapping.diagnostics.residual_template_correlation_test`.

The ``"template"`` and ``"data"`` rankings of
:func:`~sys_mapping.diagnostics.snr_template_ranking` divide by the independent-pixel
error, so they rank templates but do not detect contamination
(:ref:`calibrated-significance`).  Of the four rankings only ``"isd"`` has a calibrated
p-value, through :func:`~sys_mapping.diagnostics.isd_template_significance`.

:func:`~sys_mapping.diagnostics.footprint_mask_diagnostics` ranks pixels by
:math:`\sqrt{\sum_i t_i(p)^2/n_s}`, drops the lowest fraction for each entry of
``mask_fractions``, refits OLS, and returns the amplitudes per level and their standard
deviation across levels.  :func:`~sys_mapping.diagnostics.vet_templates_against_tracer`
returns the weighted Spearman correlation of each template with an external tracer of
the matter field, with a patch jack-knife error when ``patch_ids`` is given and
:math:`1/\sqrt{N-3}` otherwise (Weaverdyck et al. 2026, Sec. III A).

----

Mock catalogues
---------------

References: Berlfein et al. 2024;
`Coles & Jones 1991 <https://academic.oup.com/mnras/article/248/1/1/1033817>`_.

:func:`~sys_mapping.mocks.generate_lognormal_field` draws a Gaussian field :math:`G`
with ``healpy.synfast`` from :math:`C_\ell^G \propto (\ell+1)^{-2}` (zero monopole),
normalised so that :math:`\sum_\ell (2\ell+1)C_\ell^G/4\pi = \sigma^2`, and returns

.. math::

   \delta_g^{\rm true}(p) = e^{G(p) - \sigma^2/2} - 1,

which exceeds :math:`-1` and has zero mean.  :func:`~sys_mapping.mocks.make_mock_catalog`
contaminates it with :func:`~sys_mapping.contamination.apply_contamination` (amplitudes
drawn from :math:`\mathcal N(0, 0.10)` when not given), applies a Galactic latitude cut
(``lat_cut_deg`` 20, :func:`~sys_mapping.mocks.make_galactic_mask`), and in each kept
pixel draws :math:`n_g \sim {\rm Poisson}(\bar n\max(1 + \hat\delta_g, 0))` galaxies
and places :math:`{\rm round}(8\bar n)` randoms (``rand_factor`` 8, ``n_mean``
:math:`\bar n` = 30), scattering positions by a Gaussian of 0.3 pixel sizes.
:func:`~sys_mapping.mocks.make_mock_suite` builds the ``none``, ``additive``,
``multiplicative`` and ``combined`` scenarios on one density field and one template
set.  :func:`~sys_mapping.glass_mocks.generate_glass_fullsky_mock` generates GLASS
lognormal catalogues with a redshift distribution.

----

HEALPix map utilities
---------------------

`HEALPix <https://healpix.sourceforge.io>`_ divides the sphere into
:math:`12\,{\rm NSIDE}^2` equal-area pixels.

.. list-table::
   :header-rows: 1
   :widths: 15 25 30 30

   * - NSIDE
     - :math:`N_{\rm pix}`
     - Pixel area (deg²)
     - Pixel size (arcmin)
   * - 32
     - 12 288
     - 3.36
     - ≈ 110
   * - 64
     - 49 152
     - 0.84
     - ≈ 55
   * - 128
     - 196 608
     - 0.21
     - ≈ 27
   * - 512
     - 3 145 728
     - 0.013
     - ≈ 7

From catalogues to fit inputs:

1. :func:`~sys_mapping.maps.pixelize_catalog` counts objects, optionally weighted, per
   RING pixel.
2. :func:`~sys_mapping.maps.compute_overdensity` returns the overdensity on the
   footprint and the footprint mask.
3. :func:`~sys_mapping.maps.assign_template_values` selects the template values at the
   footprint pixels, ``templates[:, good_pixels]``.
4. :func:`~sys_mapping.maps.standardise_on_footprint` standardises them there.

:func:`~sys_mapping.maps.load_real_template` reads a FITS table column, optionally
changes its resolution with ``healpy.ud_grade``, normalises it over the finite pixels
above ``valid_min`` and sets the rest to 0.
:func:`~sys_mapping.maps.generate_systematic_map` and
:func:`~sys_mapping.maps.generate_systematic_maps` produce synthetic templates from five
spectral families.

----

Two-point function measurement
------------------------------

The angular correlation function uses the Landy–Szalay estimator (Landy & Szalay 1993):

.. math::

   w(\theta) = \frac{DD(\theta) - 2\,DR(\theta) + RR(\theta)}{RR(\theta)},

with normalised pair counts.
:func:`~sys_mapping.utils.measure_two_point_function` wraps TreeCorr
``NNCorrelation`` with great-circle separations (``metric="Arc"``) and returns
:math:`\exp\langle\ln\theta\rangle` per bin (defaults 0.06 to 30 arcmin, 15 bins);
:func:`~sys_mapping.utils.measure_cross_two_point_function` is the weighted cross
version.  :func:`~sys_mapping.utils.measure_two_point_function_corrfunc` computes the
same estimator with ``Corrfunc.mocks.DDtheta_mocks``.
:func:`~sys_mapping.utils.measure_kk_correlation_treecorr` and
:func:`~sys_mapping.utils.measure_kk_correlation_corrfunc` correlate scalar fields, and
:func:`~sys_mapping.utils.measure_kk_covariance_treecorr` adds a jack-knife covariance
over ``npatch`` (50) patches.

----

Method comparison table
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 34 26 14 26

   * - Method
     - Reference(s)
     - Status
     - Module
   * - Additive, multiplicative and combined models
     - Berlfein+2024
     - Implemented
     - ``contamination``, ``likelihood``
   * - Analytic additive posterior, NUTS, emcee
     - Berlfein+2024
     - Implemented
     - ``inference``, ``nuts``
   * - PCA template rotation
     - Berlfein+2024
     - Implemented
     - ``correction``
   * - Noise debiasing (diagonal and matrix)
     - Berlfein+2024
     - Implemented
     - ``correction``
   * - Two-point correction with cross-template terms
     - Berlfein+2024
     - Implemented
     - ``contamination``, ``correction``, ``utils``
   * - Likelihood ratio test, mock-calibrated
     - Berlfein+2024
     - Implemented
     - ``model_selection``
   * - Calibrated template significance
     - —
     - Implemented
     - ``diagnostics``
   * - Residual template correlation test
     - —
     - Implemented
     - ``diagnostics``
   * - Correlated-noise likelihood, mock sandwich covariance
     - Weaverdyck & Huterer 2021
     - Implemented
     - ``covariance``
   * - Block bootstrap covariance
     - Berlfein+2024
     - Implemented
     - ``bootstrap``
   * - Jack-knife covariance
     - Ross+2011, Ho+2012
     - Implemented
     - ``bootstrap``
   * - Null test cross-correlations
     - Ross+2011
     - Implemented
     - ``diagnostics``
   * - Pseudo-:math:`C_\ell` estimator
     - Elsner+2016, Ho+2012
     - Implemented
     - ``power_spectrum``
   * - Harmonic template subtraction and bias
     - Elsner+2016
     - Implemented
     - ``power_spectrum``
   * - Basic / Extended mode projection
     - Elsner+2016, Leistedt+2014
     - Implemented
     - ``power_spectrum``
   * - ElasticNet regression
     - Weaverdyck+2021
     - Implemented
     - ``regression``
   * - Iterative Systematics Decontamination
     - Elvin-Poole+2018, Rodríguez-Monroy+2022, Weaverdyck+2026
     - Implemented
     - ``regression``, ``diagnostics``
   * - SNR-based template ranking
     - Tanidis+2026, Rodríguez-Monroy+2025
     - Implemented
     - ``diagnostics``, ``model_selection``
   * - Footprint masking diagnostics
     - Rodríguez-Monroy+2025
     - Implemented
     - ``diagnostics``
   * - Full-rank harmonic precision
     - —
     - Raises ``NotImplementedError``
     - ``covariance``
   * - NaMaster coupling matrix (exact)
     - Alonso+2019, Elsner+2017
     - Not implemented
     - ``power_spectrum``
   * - Catalogue-based PCL deprojection
     - Cornish+2026
     - Not implemented
     - —
   * - Transfer function calibration
     - Cornish+2026
     - Not implemented
     - —
   * - ANN / deep-learning weights
     - Rezaie+2020
     - Not implemented
     - External pre-processing
   * - Optimal quadratic estimator (QMV)
     - Ho+2012
     - Not implemented
     - External (NaMaster)
   * - Shear E/B quadratic estimator
     - Tanidis+2026
     - Not implemented
     - Weak-lensing specific
