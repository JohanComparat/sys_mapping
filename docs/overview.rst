Overview
========

Observing conditions (seeing, Galactic extinction, sky background, stellar density,
depth) modulate the probability of detecting a galaxy, and so imprint coherent
large-scale fluctuations on the observed galaxy counts.  These fluctuations are
correlated with *template maps*, HEALPix maps of the observing conditions built
independently of the galaxy catalogue.  ``sys_mapping`` fits the amplitude of each
template in a galaxy overdensity map and corrects the map, the per-galaxy weights and
the angular correlation function :math:`w(\theta)`.

:func:`~sys_mapping.regression.run_decontamination` runs six methods behind one
interface:

* ``OLS``: least-squares regression of the overdensity on the templates
  (`Ross et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.417.1350R/abstract>`_;
  `Ho et al. 2012 <https://ui.adsabs.harvard.edu/abs/2012ApJ...761...14H/abstract>`_).
* ``ElasticNet``: :math:`\ell_1+\ell_2`-regularised regression with the penalty chosen
  by cross-validation
  (`Weaverdyck & Huterer 2021 <https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.5061W/abstract>`_).
* ``ISD-1`` and ``ISD-3``: Iterative Systematics Decontamination, a greedy sequence of
  marginal fits of the binned density against one template at a time, linear or cubic
  in the template value, stopped by a mock-calibrated threshold
  (`Elvin-Poole et al. 2018 <https://arxiv.org/abs/1708.01536>`_;
  `Rodríguez-Monroy et al. 2022 <https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.2665R/abstract>`_;
  `Weaverdyck et al. 2026 <https://arxiv.org/abs/2601.14484>`_).  ``ISD-<d>`` selects
  any other degree.
* ``MCMC-add``: the exact Normal-Inverse-Gamma posterior of the additive model
  (`Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_).
* ``MCMC-comb``: the posterior of the combined additive and multiplicative model,
  sampled with BlackJAX NUTS (Berlfein et al. 2024).

The mathematics of each step is on :doc:`methods`, the references on
:doc:`bibliography`, and the application to Legacy Survey DR10 on
:doc:`pipeline_ls10`.

----

Notation
--------

All maps use the `HEALPix <https://healpix.sourceforge.io>`_ pixelisation.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Symbol
     - Definition
   * - :math:`p`
     - HEALPix pixel index.
   * - :math:`N_{\rm pix}`
     - Number of pixels on the sphere, :math:`12\,{\rm NSIDE}^2`.
   * - :math:`N`
     - Number of footprint pixels used in the fit.
   * - :math:`n_g(p)`, :math:`n_r(p)`
     - Galaxy and random counts in pixel :math:`p`.
   * - :math:`\delta_g(p)`
     - Clean galaxy overdensity.
   * - :math:`\hat{\delta}_g(p)`
     - Observed (contaminated) galaxy overdensity.
   * - :math:`t_i(p)`
     - Template :math:`i`, standardised to zero mean and unit rms over the footprint,
       :math:`i = 1, \ldots, n_s`.
   * - :math:`a_i`, :math:`b_i`
     - Additive and multiplicative contamination amplitudes.
   * - :math:`\sigma`
     - Pixel scale of the clean overdensity (a nuisance parameter).
   * - :math:`\gamma`
     - Shape parameter of the skew-normal likelihood; :math:`\gamma=0` is Gaussian.
   * - :math:`V`
     - Eigenvectors of the template covariance (PCA rotation).
   * - :math:`\xi_{ij}(\theta)`
     - Angular correlation of templates :math:`i` and :math:`j`.
   * - :math:`w(p)`
     - Per-pixel systematic weight.

Arrays over the whole sphere have shape ``(n_pix,)`` with ``n_pix = 12 * nside**2``;
arrays over the footprint have shape ``(n_good,)`` and follow the order of
``np.where(good_pixels)[0]``.  Templates have shape ``(n_sys, n_good)``.

----

Observed overdensity
--------------------

:func:`~sys_mapping.maps.compute_overdensity` keeps the pixels whose random count
reaches a tenth of the maximum, :math:`n_r(p) \ge 0.1\,\max_p n_r` (the
``min_random_fraction`` argument), and on them computes

.. math::

   \hat{\delta}_g(p) = \frac{n_g(p)}{f_r\,n_r(p)} - 1,
   \qquad f_r = \frac{\sum_p n_g(p)}{\sum_p n_r(p)},

which has zero mean when weighted by the random counts.  It returns the overdensity at
the kept pixels and the boolean footprint mask.

----

Template maps
-------------

A template is a HEALPix map of an observing condition: dust extinction
:math:`E(B-V)`, stellar density or flux from *Gaia*, depth, exposure count or PSF size
from the imaging.  Maps loaded from disk are normalised over their own valid region,
which is larger than any one sample's footprint, so
:func:`~sys_mapping.maps.standardise_on_footprint` standardises the basis again over
the fitted pixels:

.. math::

   t_i(p) \;\leftarrow\; \frac{t_i(p) - \mu_i}{s_i},
   \qquad
   \mu_i = \frac{1}{N}\sum_p t_i(p),
   \quad
   s_i^2 = \frac{1}{N}\sum_p \bigl(t_i(p) - \mu_i\bigr)^2.

The amplitudes, the condition number of the template covariance and the template
correlations of the two-point correction are then in units of one template standard
deviation on the footprint.

----

Contamination model
-------------------

The forward model is Eq. 13 of Berlfein et al. 2024:

.. math::

   \hat{\delta}_{g}(p) = \delta_{g}(p)\left(1 + \sum_{i=1}^{n_s} b_i\,t_i(p)\right)
                        + \sum_{i=1}^{n_s} a_i\,t_i(p).

An additive term adds spurious objects or removes real ones along a template; a
multiplicative term scales the density contrast by a position-dependent efficiency.
Three nested models follow (:func:`~sys_mapping.contamination.n_free_params`):

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Model
     - Constraint
     - Free parameters
   * - ``additive``
     - :math:`b_i = 0`
     - :math:`a_i`, :math:`\sigma`: :math:`n_s + 1`
   * - ``multiplicative``
     - :math:`a_i = 0`
     - :math:`b_i`, :math:`\sigma`: :math:`n_s + 1`
   * - ``combined``
     - none
     - :math:`a_i`, :math:`b_i`, :math:`\sigma`: :math:`2n_s + 1`

The skew-normal likelihood adds :math:`\gamma` to each.  The model inverts to

.. math::

   \delta_g(p) = \frac{\hat{\delta}_g(p) - \sum_i a_i\,t_i(p)}{1 + \sum_i b_i\,t_i(p)},

which requires :math:`1 + \sum_i b_i\,t_i(p) \neq 0`
(:func:`~sys_mapping.contamination.invert_contamination`).

Each method estimates its amplitudes differently and returns a per-pixel weight
``weights``, clipped to :math:`[1/20, 20]`:

.. list-table::
   :header-rows: 1
   :widths: 13 22 65

   * - Method
     - Model
     - Estimate and weight
   * - OLS
     - additive
     - Least squares, :math:`\hat{\mathbf a} = \arg\min \|\hat{\boldsymbol\delta}_g - \mathbf{T}^\top\mathbf a\|^2`;
       :math:`w = 1/(1 + \hat{\mathbf a}\cdot\mathbf t)`.
   * - ElasticNet
     - additive
     - :func:`~sys_mapping.regression.elasticnet_contamination_fit`: penalty strength
       by 5-fold cross-validation at ``l1_ratio=0.5``;
       :math:`w = 1/(1 + \hat{\mathbf a}\cdot\mathbf t)`.
   * - ISD-1, ISD-3
     - per-template efficiency :math:`1 + \hat F_j(t_j)`
     - :func:`~sys_mapping.regression.iterative_systematics_decontamination`:
       :math:`w = \prod_j 1/(1 + \hat F_j(t_j))` over the accepted steps.  ``a_hat``
       is the projection of the removed curve on each template and ``b_hat`` is zero.
   * - MCMC-add
     - additive
     - :func:`~sys_mapping.inference.run_additive_analytic` on the PCA-rotated
       templates; posterior median;
       :math:`w = (1 + \delta_g^{\rm clean})/(1 + \hat\delta_g)` with :math:`b = 0`.
   * - MCMC-comb
     - combined
     - :func:`~sys_mapping.nuts.run_nuts` on the PCA-rotated templates; posterior
       median; :math:`w = (1 + \delta_g^{\rm clean})/(1 + \hat\delta_g)`.

The regression methods return point estimates.  The MCMC methods also return the
posterior draws and the amplitude covariances ``cov_a`` and ``cov_b`` in the original
template basis.  These pixel-likelihood widths assume independent pixels; calibrated
errors come from :func:`~sys_mapping.covariance.mock_sandwich_covariance` and
calibrated significances from
:func:`~sys_mapping.diagnostics.calibrated_template_significance`.

----

Likelihood
----------

With :math:`\delta_g(p) \sim \mathcal{N}(0, \sigma^2)` independently in each pixel, the
log-likelihood is Eq. 17 of Berlfein et al. 2024:

.. math::

   \ln\mathcal{L} = -\frac{N}{2}\ln(2\pi\sigma^2)
                    - \sum_p \ln\left|1 + \sum_i b_i\,t_i(p)\right|
                    - \frac{1}{2\sigma^2}\sum_p \delta_g(p)^2,

where :math:`\delta_g(p)` is the inverted clean field and the middle term is the
Jacobian of the change of variables from :math:`\hat\delta_g` to :math:`\delta_g`.  The
skew-normal form (Eq. 18) evaluates both terms at the shifted residual
:math:`r_p = \delta_g(p) - \xi`:

.. math::

   \ln\mathcal{L}_{\rm skew} = \ln\mathcal{L}_{\rm Gauss}(r)
       + N\ln 2
       + \sum_p \ln\Phi\left(\gamma\,\frac{r_p}{\sigma}\right),
   \qquad
   \xi = -\sigma\,\frac{\gamma}{\sqrt{1+\gamma^2}}\sqrt{2/\pi}.

:func:`~sys_mapping.likelihood.make_log_likelihood` returns a JIT-compiled function,
cached per ``(n_sys, model, use_skewed)``, with the data as arguments.  Its optional
``precision`` operator uses the covariance :math:`\sigma^2 R` in place of :math:`\sigma^2 I`; the
skew-normal form with a precision operator is refused, because the product form is a
density only for independent pixels.

----

Posterior sampling
------------------

``run_decontamination(..., sampler="auto")`` draws the additive Gaussian posterior
exactly from its Normal-Inverse-Gamma form and samples the combined and skew-normal
posteriors with BlackJAX NUTS, adapting a dense mass matrix by default.
``sampler="emcee"`` selects the ensemble sampler of
:func:`~sys_mapping.inference.run_mcmc`, and a ``pixel_precision`` operator forces NUTS.
Both MCMC methods fit the PCA-rotated templates and transform the amplitudes and their
covariance back to the original basis.  Details are on :doc:`methods`.

----

Template PCA rotation
---------------------

Correlated templates make the amplitudes degenerate.  With
:math:`C = \mathbf{T}\mathbf{T}^\top/N = V D V^\top`,
:func:`~sys_mapping.correction.rotate_templates` returns
:math:`\mathbf{T}' = V^\top \mathbf{T}`, whose covariance :math:`D` is diagonal, and
:func:`~sys_mapping.correction.transform_params_from_rotated` maps the fitted
amplitudes back as :math:`\mathbf a = V\mathbf a'`, :math:`\mathbf b = V\mathbf b'`.
The rotation diagonalises the covariance at zero separation only.

----

Noise debiasing
---------------

Because :math:`\mathbb{E}[\hat a_i \hat a_j] = a_i a_j + {\rm Cov}_{ij}[\hat a]`, the
squared amplitudes that enter the two-point correction are debiased (Eq. 21 of
Berlfein et al. 2024).  :func:`~sys_mapping.correction.debias_params_matrix` returns

.. math::

   \tilde A = \hat{\mathbf a}\hat{\mathbf a}^\top - {\rm Cov}[\hat{\mathbf a}],

and :math:`\tilde B` analogously.  The result is symmetric but not positive semi-definite:
the subtraction turns the :math:`n_{\rm sys}-1` null directions of the rank-one outer product
negative, and projecting them back to zero would undo the subtraction itself.
:func:`~sys_mapping.correction.debias_params` is the diagonal form,
:math:`\tilde a_i^2 = \max(\hat a_i^2 - {\rm Var}[\hat a_i], 0)`.

The covariance is the sandwich estimator of
:func:`~sys_mapping.covariance.mock_sandwich_covariance`, measured on the same uncontaminated
realisations that calibrate the significance: the independent-pixel covariance of a fit is too
small on a clustered field, and the regression methods carry none of their own.

----

Two-point function correction
-----------------------------

Inserting the contamination model into :math:`w(\theta)` and keeping second order in
the amplitudes gives (Eqs. 15–16 of Berlfein et al. 2024, with the cross-template
terms kept):

.. math::

   \hat{w}_{\rm corr}(\theta) =
   \frac{\hat{w}(\theta) - \sum_{ij} \tilde A_{ij}\,\xi_{ij}(\theta)}
        {1 + \sum_{ij} \tilde B_{ij}\,\xi_{ij}(\theta)}.

:func:`~sys_mapping.correction.correct_two_point_function` debiases with the full
covariances when given the ``(n_sys, n_sys, n_bins)`` correlation matrix, and with the
diagonal when given ``(n_sys, n_bins)`` auto-correlations.  It warns where the
correction drives :math:`w_{\rm corr}` negative while :math:`\hat w > 0`, and with
``return_cov=True`` propagates the amplitude and measurement covariances by
parametric bootstrap.

:func:`~sys_mapping.utils.template_correlation_matrix` measures :math:`\xi_{ij}` with
TreeCorr from the template values each galaxy carries.  The cross terms come from
auto-correlations of summed fields,

.. math::

   \xi_{ij} = \tfrac12\left[\xi(t_i + t_j) - \xi_{ii} - \xi_{jj}\right],

with optional per-object weights ``w=``.  With the OLS amplitudes of the nine LS10 samples
the cross terms carry a median 22.5 % of the correction at NSIDE 64 in the rotated basis.

----

Model selection
---------------

The likelihood ratio between the additive and combined models (Eq. 19 of Berlfein
et al. 2024) is

.. math::

   \lambda_{\rm LR} = 2\left[\ln\mathcal{L}_{\rm comb}(\hat{\Theta}_1) -
                             \ln\mathcal{L}_{\rm add}(\hat{\Theta}_0)\right],

evaluated at the likelihood maxima of both models, where it is non-negative.  Wilks'
theorem gives :math:`\chi^2(n_s)`, but the pixel likelihood treats a spatially
correlated field as independent pixels and inflates the statistic, so
:func:`~sys_mapping.model_selection.likelihood_ratio_test` accepts an empirical null
``null_lambda`` from uncontaminated realisations and returns
:math:`p = (1 + \#\{\lambda_{\rm null} \ge \lambda_{\rm LR}\})/(1 + N_{\rm null})`.
:func:`~sys_mapping.model_selection.lrt_from_maxima` computes that null for many
fields at once.

----

Calibrated detection
--------------------

:func:`~sys_mapping.diagnostics.calibrated_template_significance` scores each
least-squares amplitude against its scatter over uncontaminated realisations with the
sample's clustering, and returns a family-wise p-value for the most significant
template.  :func:`~sys_mapping.diagnostics.residual_template_correlation_test` tests a
corrected field for residual correlation with templates the correction did not fit.

By default each null realisation is drawn by
:func:`~sys_mapping.glass_mocks.draw_null_overdensity` from a GLASS lognormal field
with the sample's matched spectrum (:func:`~sys_mapping.glass_mocks.load_matched_cl`),
and galaxy and random counts drawn per footprint pixel.  The LS10 script refuses to
build a null without a matched spectrum unless ``--allow-parametric-null`` is given.

----

Per-galaxy weights
------------------

Each galaxy receives the weight of its pixel; galaxies outside the fitted footprint
receive 1.  ``scripts/run_ls10_analysis.py`` writes seven columns to
``<sample>_NSIDE<nside>_WEIGHTS.fits``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Weight
   * - ``WEIGHT_OLS``
     - OLS
   * - ``WEIGHT_ENET``
     - ElasticNet
   * - ``WEIGHT_ISD1``
     - ISD, degree 1
   * - ``WEIGHT_ISD3``
     - ISD, degree 3
   * - ``WEIGHT_ADD``
     - MCMC-add
   * - ``WEIGHT_COMB``
     - MCMC-comb
   * - ``WEIGHT_SYS``
     - copy of ``WEIGHT_COMB``

Every column is the ``weights`` array of ``run_decontamination``, so every weight lies
in :math:`[1/20, 20]`; the header records ``WMAXCLIP = 20``.
:doc:`results_ls10_recommendations` gives the column to use per sample.

----

LS10 analysis script
--------------------

``scripts/run_ls10_analysis.py`` processes each ``*_DATA.fits`` / ``*_RAND.fits`` pair
in ``--catalog-dir`` (or the one named by ``--sample``) in the following phases.

1. Resolution.  With ``--min-per-pixel``,
   :func:`~sys_mapping.maps.choose_nside_by_occupancy` picks the finest NSIDE, starting
   from ``--nside`` and halving, at which the footprint holds that many galaxies per
   pixel on average.  The templates in ``--template-dir`` are downgraded to it.
2. Matched spectrum.  When the run builds any GLASS null (ISD pre-selection, the ISD
   threshold, the calibrated significance or the LRT null), the sample's spectrum is
   loaded from ``--null-cl-file``.
3. Maps.  Galaxies (weighted by ``WEIGHT_COMP`` when present) and randoms are
   pixelised, the overdensity is computed, and the templates are standardised on the
   footprint (``--no-footprint-standardise`` keeps the load-time normalisation).
4. Optional pre-selection (``--preselect``).
5. ISD threshold.  :math:`\Delta\chi^2_{68}` is measured on ``--isd-n-mocks``
   uncontaminated realisations with a cubic, equal-occupancy fit, unless pre-selection
   already produced it.
6. Fits.  ``run_decontamination`` runs OLS, ElasticNet, ISD-1, ISD-3, MCMC-add and
   MCMC-comb in that order, or the subset given by ``--only-methods``.  Without
   MCMC-comb the script writes the partial JSON and stops; with it, amplitudes from
   earlier partial files fill the methods not run.
7. Calibrated significance on ``--significance-n-mocks`` realisations.
8. Likelihood ratio.  Both models are refined to their maxima with
   :func:`~sys_mapping.inference.refine_to_mle` and compared with the maxima of
   :func:`~sys_mapping.model_selection.lrt_from_maxima`, keeping the higher likelihood
   per model; ``--lrt-null-mocks`` realisations calibrate the p-value.
9. Two-point correction.  :math:`w(\theta)` is measured with TreeCorr over 30 bins
   from 0.5 to 300 arcmin, the template correlation matrix of the rotated basis from at
   most ``--ct-max-galaxies`` galaxies, and every method's :math:`w_{\rm corr}` is
   computed.
10. Outputs: the weights FITS file, ``params.json``, ``wtheta_data.json`` and the
    figures, then ``summary_NSIDE<nside>.yaml`` once all samples are done.

Output JSON
~~~~~~~~~~~

``<sample>_NSIDE<nside>_params.json`` (full run) holds:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Key
     - Content
   * - ``sample_id``, ``nside``, ``n_sys``, ``template_names``
     - Run identity.
   * - ``template_basis``
     - ``standardised_on`` (``footprint`` or ``load-time``), ``rms_before``,
       ``mean_before``.
   * - ``significance``
     - ``method``, ``n_null``, ``p_value_floor``, ``significance``, ``p_values``,
       ``family_wise_p``, ``inflation`` (calibrated over independent-pixel error), or
       ``null``.
   * - ``a_hat_add``, ``b_hat_comb``, ``var_a_add``, ``var_b_comb``
     - MCMC amplitudes and their variances.
   * - ``lrt``
     - ``lambda_lr``, ``p_value``, ``n_dof``, ``reject_null``, ``calibration``
       (``chi2``, ``mock`` or ``failed``), ``p_chi2``, ``n_null``, ``null_cl_source``,
       ``null_cl_amplitude``, ``null_lambda_mean``, ``null_lambda_max``,
       ``null_lambda``.
   * - ``acceptance_fraction_add``, ``acceptance_fraction_comb``,
       ``sigma_hat_add``, ``sigma_hat_comb``
     - Sampler summaries.
   * - ``n_galaxies``, ``n_good_pix``
     - Sample size and footprint pixels.
   * - ``methods``
     - Per method: ``a_hat``, ``b_hat``, ``sigma_hat``, ``elapsed_s``, ``rms_a_hat``;
       for ISD also ``isd_poly_order``, ``n_steps``, ``stopped_on``, ``calibrated``,
       ``n_floored``, ``significance`` and ``steps`` (template, significance and
       coefficients of each accepted step).

A run whose ``--only-methods`` omits MCMC-comb writes
``<sample>_NSIDE<nside>_partial_<methods>.json``, with the sorted method names joined
by underscores.  It holds ``schema_version`` (1), ``sample_id``, ``methods_run``,
``timestamp_utc``, ``n_sys``, ``n_good``, ``n_galaxies``, ``template_names``,
``template_basis``, and one entry per method with ``elapsed_s``, ``a_hat``,
``sigma_hat``, and for the MCMC methods ``var_a`` and ``acceptance_fraction``.

A run with ``--only-methods`` skips a sample whose output already exists: the partial
file when MCMC-comb is not listed, ``params.json`` when it is.  ``--force`` re-runs.

Results page sentinels
~~~~~~~~~~~~~~~~~~~~~~

Unless ``--no-rst`` or ``--figures-only`` is given, and when every sample ran at one
resolution, the script rewrites a section of ``docs/results_ls10.rst`` bounded by

.. code-block:: rst

   .. _auto-ls10-{TAG}-start:
   ...
   .. _auto-ls10-{TAG}-end:

where ``{TAG}`` is the method list with ``-`` and ``+`` replaced by ``_`` (for example
``MCMC_comb``).  An existing block is replaced and a missing one appended.  A run
tagged ``MCMC_comb`` also rewrites the per-sample figure block between ``.. _auto-ls10-figures-start:`` and
``.. _auto-ls10-figures-end:``.

Figure colours
~~~~~~~~~~~~~~

``sys_mapping.plotting`` fixes one colour and line style per method:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Method
     - Colour
     - Hex
     - Line style
   * - OLS
     - steel blue
     - ``#4E79A7``
     - dotted (``:``)
   * - ElasticNet
     - forest green
     - ``#59A14F``
     - dashed (``--``)
   * - ISD-1
     - amber
     - ``#F28E2B``
     - dash-dot (``-.``)
   * - ISD-3
     - coral red
     - ``#E15759``
     - densely dash-dot-dot
   * - MCMC-add
     - muted purple
     - ``#B07AA1``
     - loosely dashed
   * - MCMC-comb
     - teal
     - ``#76B7B2``
     - solid (``-``)
