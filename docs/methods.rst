Methods Reference
=================

This page maps every method implemented in ``sys_mapping`` to its source
publication, provides the key equations with all notation spelled out, and
gives physical motivation and practical guidance on when to use each method.
For notation conventions see :doc:`overview`; for bibliographic detail on each
reference see :doc:`bibliography`.

.. contents::
   :local:
   :depth: 2

----

Template pre-selection (Stage 1)
---------------------------------

When many candidate systematic maps are available, running the full Bayesian
decontamination on all of them is computationally expensive and
numerically ill-conditioned.  A fast Stage 1 pre-selection filters the
template pool before Stage 2 decontamination.

**Module:** :mod:`sys_mapping.diagnostics`, :mod:`sys_mapping.model_selection`

Three SNR ranking statistics are available
(:func:`~sys_mapping.diagnostics.snr_template_ranking`):

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Key
     - Speed
     - Description
   * - ``"data"``
     - < 1 ms (JAX)
     - Pearson :math:`|r|` between :math:`\delta_g` and each template :math:`t_i`.
       Purely linear; insensitive to the sign of the contamination amplitude.
   * - ``"template"``
     - < 1 ms (JAX)
     - Per-template OLS :math:`|\hat\alpha_i|/\sigma_{\hat\alpha_i}` (*t*-statistic).
       Equivalent to :math:`|r|` up to a scaling; useful when template variances
       differ substantially.
   * - ``"isd"``
     - ~10 ms (JAX)
     - ISD :math:`\Delta\chi^2` (Rodríguez-Monroy et al. 2025, Sec. IV.A.1):
       pixels are binned by template value; a linear model :math:`f(s)` is fit
       to the binned galaxy density; :math:`\Delta\chi^2 =
       \chi^2_{\rm null} - \chi^2_{\rm model}` measures the contamination.

For a rigorous significance test,
:func:`~sys_mapping.diagnostics.isd_template_significance` compares the data
:math:`\Delta\chi^2` against a distribution from GLASS systematic-free mocks
on the same footprint, giving mock-based *p*-values.  The mocks are trimmed to
the survey footprint via ``good_pixels`` and generated with the correct surface
density by scaling ``n_total_footprint → n_total_footprint × N_{\rm full} /
N_{\rm good}`` before drawing on the full sky.

**Recommended two-stage workflow:**

.. code-block:: python

   import sys_mapping as sm

   # Stage 1 — pre-selection with ISD mocks (built into run_decontamination)
   result = sm.run_decontamination(
       "ISD-1", delta_g, delta_t,
       preselect=True,
       preselect_method="isd",
       preselect_n_mocks=100,
       preselect_p_threshold=0.05,
       good_pixels=good_pix,
       n_total_footprint=len(ra_gal),
       z_edges=z_edges, nz=nz, nside=nside,
   )
   print("Selected templates:", result["preselect_indices"])

   # Stage 2 results use only the pre-selected templates
   print("a_hat:", result["a_hat"])

See :doc:`results_snr_preselection` for a full validation on a 15 M galaxy
simulation with 20 templates at three contamination levels.

**Rule of thumb for N_mocks:**

.. math::

   N_{\rm mocks} \;\ge\; \max\!\left(20,\; \left\lceil 5 / \alpha \right\rceil\right)

where :math:`\alpha` is the target significance level (0.05 → 100 mocks).

----

Core contamination model
-------------------------

**Reference:** `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_
(MNRAS 531, 4954), Eq. 11–13.

**Module:** :mod:`sys_mapping.contamination`

**Physical motivation.** A photometric galaxy survey assigns each galaxy
to a HEALPix pixel based on its observed position.  The measured number
count :math:`n_g(p)` per pixel can differ from the true number count
:math:`n_g^{\rm true}(p)` for two physically distinct reasons:

* *Additive* contamination: a contaminant contributes galaxy-like objects
  to the catalog, or masks real galaxies over a spatially coherent pattern.
  Examples: stellar contamination (stars misclassified as galaxies add to
  counts), fibre allocation inefficiency (sparse regions are under-observed).
  These shift :math:`\delta_g` by a fixed amount proportional to :math:`t_i`.

* *Multiplicative* contamination: a varying selection efficiency scales
  galaxy counts up or down relative to the true density.  Examples: depth
  variations (seeing, sky background, airmass) modulate the survey
  completeness as a function of position.  These multiply :math:`1+\delta_g`
  by a factor proportional to :math:`1 + b_i t_i`.

The combined forward model is:

.. math::

   \hat{\delta}_{g}(p) = \delta_{g}(p)\!\left(1 + \sum_{i=1}^{n_s} b_i\,t_i(p)\right)
                        + \sum_{i=1}^{n_s} a_i\,t_i(p)

where :math:`\delta_g(p) = n_g^{\rm true}(p)/\bar{n}_g - 1` is the true
overdensity, :math:`\hat\delta_g(p)` is the observed overdensity, and
:math:`t_i(p)` is systematic template :math:`i` normalised to zero mean and
unit variance over the footprint.  The amplitudes :math:`a_i` (additive) and
:math:`b_i` (multiplicative) are the free parameters to be inferred.

**Three nested models** are obtained by constraining these amplitudes:

.. list-table::
   :header-rows: 1
   :widths: 20 20 22 38

   * - Model
     - Free parameters
     - Constraint
     - Use case
   * - ``additive``
     - :math:`a_i`, :math:`\sigma`
     - :math:`b_i = 0`
     - Stellar contamination, photometric bias
   * - ``multiplicative``
     - :math:`a_i`, :math:`\sigma`
     - :math:`b_i = a_i`
     - Depth / completeness variations
   * - ``combined``
     - :math:`a_i, b_i`, :math:`\sigma`
     - none
     - Unknown contamination type; general case

**Inverting the model.** The clean overdensity can be recovered from the
observed one via:

.. math::

   \delta_g(p) = \frac{\hat{\delta}_g(p) - \sum_i a_i\,t_i(p)}
                      {1 + \sum_i b_i\,t_i(p)}

This inversion is valid only when :math:`1 + \sum_i b_i t_i(p) > 0` in
every pixel (i.e. the multiplicative modulation never entirely suppresses the
galaxy density).  For typical values :math:`|b_i| \lesssim 0.3` and
unit-normalised templates this condition is satisfied almost everywhere.

**Weight-based correction for linear methods.**
For all additive-type methods (OLS, ElasticNet, ISD, MCMC-additive), the
corrected field is computed as (Weaverdyck & Huterer 2020, Eq. 46-47):

.. math::

   \delta_g^{\rm corr}(p) = w(p)\,(1 + \hat{\delta}_g(p)) - 1, \qquad
   w(p) = \frac{1}{1 + \hat{\boldsymbol\alpha}\cdot\mathbf{t}(p)}

This is exact for multiplicative contamination and equals the simpler
subtraction :math:`\hat\delta_g - \hat{\boldsymbol\alpha}\cdot\mathbf{t}` at
first order in :math:`\hat{\boldsymbol\alpha}\cdot\mathbf{t}`.  Using the
weight form preserves positivity (:math:`1 + \delta_g^{\rm corr} > 0`) and
is the convention in the reference literature (Weaverdyck & Huterer 2020;
Rodríguez-Monroy et al. 2025); it is applied uniformly throughout this
package.  For MCMC-combined the full inversion formula above is used instead.

**Technical implementation.**
All arithmetic is performed on **JAX arrays** (``jax.Array``) to enable
JIT compilation and automatic differentiation.  The forward and inverse
models are implemented as pure functions with no mutable state:

*Forward model* (:func:`~sys_mapping.contamination.apply_contamination`):

.. code-block:: python

   mult_term = jnp.einsum("i,ij->j", b, delta_t)   # Σ b_i t_i(p)
   add_term  = jnp.einsum("i,ij->j", a, delta_t)   # Σ a_i t_i(p)
   delta_g_obs = delta_g * (1.0 + mult_term) + add_term

Both Einstein sums are single BLAS-1 operations of cost
:math:`O(n_s \times N)`.  At ``n_pix = 10\,000``, ``n_sys = 5`` the
forward call runs in approximately **830 μs** after JIT warm-up
(float32 accumulation; max absolute roundtrip error < 1e-5).

*Inverse model* (:func:`~sys_mapping.contamination.invert_contamination`):
adds an element-wise division (≈ **1 730 μs** at the same size).

*Parameter vector layout* (:func:`~sys_mapping.contamination.pack_params`,
:func:`~sys_mapping.contamination.unpack_params`):
the flat ``float64`` vector fed to emcee / scipy optimisers has layout

.. code-block:: text

   [a_0, …, a_{n_s-1},   ← always present
    b_0, …, b_{n_s-1},   ← combined model only
    sigma,                ← always
    gamma]                ← skew-normal only

Slice operations preserve values to machine precision.  The
``match``/``case`` dispatch (Python 3.10+) has zero runtime cost.

**Functions:**
:func:`~sys_mapping.contamination.apply_contamination` (forward model),
:func:`~sys_mapping.contamination.invert_contamination` (recover clean field),
:func:`~sys_mapping.contamination.pack_params` / :func:`~sys_mapping.contamination.unpack_params`
(parameter vector layout),
:func:`~sys_mapping.contamination.n_free_params`,
:func:`~sys_mapping.contamination.compute_two_point_correction`

----

Gaussian and skew-normal likelihood
-------------------------------------

**Reference:** `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_,
Eq. 17–18.

**Module:** :mod:`sys_mapping.likelihood`

**Physical motivation.** To infer the contamination amplitudes
:math:`(a_i, b_i)` from data, we need a likelihood model for the
distribution of the clean overdensity field :math:`\delta_g(p)`.
At the pixel level (with :math:`\bar{n} \gtrsim 30` galaxies per pixel),
Poisson fluctuations in the count map are approximately Gaussian.
The central limit theorem ensures that the overdensity in each pixel
approaches a Gaussian as :math:`\bar{n} \to \infty`.

**Gaussian likelihood.** Assuming :math:`\delta_g(p) \stackrel{\rm iid}{\sim} \mathcal{N}(0, \sigma^2)`
for each unmasked pixel :math:`p`, the log-likelihood is (Eq. 17):

.. math::

   \ln\mathcal{L}(\boldsymbol\Theta) =
   -\frac{N}{2}\ln(2\pi\sigma^2)
   - \underbrace{\sum_p \ln\!\left|1 + \sum_i b_i\,t_i(p)\right|}_{\text{Jacobian of the change of variables}}
   - \frac{1}{2\sigma^2}\sum_p
     \underbrace{\left[\frac{\hat\delta_g(p) - \sum_i a_i t_i(p)}{1+\sum_i b_i t_i(p)}\right]^2}_{\delta_g(p)^2}

Here :math:`\boldsymbol\Theta = (a_1,\ldots,a_{n_s},b_1,\ldots,b_{n_s},\sigma)`,
:math:`N` is the number of unmasked pixels, and the Jacobian term
accounts for the change of variable from :math:`\hat\delta_g` to :math:`\delta_g`.
For the additive model (:math:`b_i=0`) the Jacobian term vanishes.

**Skew-normal likelihood.** For a lognormal galaxy field,
:math:`\delta_g = e^{G-\sigma_G^2/2}-1` has positive skewness
:math:`s \approx 3\sigma_G + \sigma_G^3 > 0`.
Fitting a Gaussian likelihood to a positively-skewed field introduces a
systematic bias in the inferred amplitudes.  The skew-normal extension
(Eq. 18) corrects for this by adding a log-CDF term:

.. math::

   \ln\mathcal{L}_{\rm skew} = \ln\mathcal{L}_{\rm Gauss}
       + N\ln 2
       + \sum_p \ln\Phi\!\left(\gamma\,\frac{\delta_g(p) - \xi}{\sigma}\right)

where :math:`\Phi(z) = \tfrac{1}{2}[1+{\rm erf}(z/\sqrt{2})]` is the
standard-normal CDF, :math:`\gamma > 0` is the skewness shape parameter
(fitted alongside :math:`a_i, b_i, \sigma`), and
:math:`\xi = -\sigma\delta\sqrt{2/\pi}` with
:math:`\delta = \gamma/\sqrt{1+\gamma^2}` is a mean-shift that ensures
:math:`\mathbb{E}[\delta_g]=0` is maintained.
Setting :math:`\gamma=0` recovers the Gaussian exactly.

**Technical implementation.**
:func:`~sys_mapping.likelihood.make_log_likelihood` is a *factory*: it
captures ``n_sys``, ``model``, and ``use_skewed`` as closed-over Python
integers/booleans that become compile-time constants during JAX tracing.
The returned function is decorated with ``@jax.jit`` and compiled on the
first call (~**227 ms** compile latency; reuse the function object across
all MCMC iterations).

*Gaussian path* (``use_skewed=False``):

.. code-block:: python

   mult_term   = jnp.einsum("i,ij->j", b, delta_t)
   add_term    = jnp.einsum("i,ij->j", a, delta_t)
   log_jac     = jnp.sum(jnp.log(jnp.abs(1.0 + mult_term)))   # Jacobian
   delta_g_clean = (delta_g_obs - add_term) / (1.0 + mult_term)
   log_gauss   = -0.5*N*log(2πσ²) - (1/2σ²)*Σ delta_g_clean²
   return log_gauss - log_jac

Execution time after JIT warm-up: **~284 μs/call** at
:math:`N = 10\,000`, :math:`n_s = 5` on CPU.
At zero contamination, matches ``scipy.stats.norm.logpdf`` sum to
``rtol = 1e-5``.

*Skew-normal path* (``use_skewed=True``):

.. code-block:: python

   delta_param = gamma / jnp.sqrt(1.0 + gamma**2)   # δ = γ/√(1+γ²)
   xi          = -sigma * delta_param * jnp.sqrt(2.0 / jnp.pi)
   residual    = delta_g_clean - xi
   z           = gamma * residual / sigma
   log_skew    = N*log(2) + jnp.sum(log_ndtr(z))     # Σ log Φ(γ r/σ)
   return log_gauss + log_skew - log_jac

The log-CDF :math:`\ln\Phi(z)` is evaluated via
``jax.scipy.special.log_ndtr``, which is numerically stable for large
negative :math:`z` (where ``log(1+erf(z/√2))`` would underflow).
Execution time: **~592 μs/call** (extra ``log_ndtr`` sum over :math:`N`
pixels).  Setting ``gamma = 0`` recovers the Gaussian to ``atol = 1e-6``.

**Function:**
:func:`~sys_mapping.likelihood.make_log_likelihood`

----

MCMC inference
--------------

**Reference:** `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_,
Sec. 4; `Goodman & Weare 2010
<https://msp.org/camcos/2010/5-1/p04.xhtml>`_ (emcee sampler).

**Module:** :mod:`sys_mapping.inference`

**Physical motivation.** Given the likelihood :math:`\mathcal{L}(\boldsymbol\Theta)`
and a prior :math:`\pi(\boldsymbol\Theta)`, Bayes' theorem gives the
posterior:

.. math::

   p(\boldsymbol\Theta \mid \hat{\boldsymbol\delta}_g, \mathbf{T}) \propto
   \mathcal{L}(\boldsymbol\Theta)\,\pi(\boldsymbol\Theta)

The prior on each :math:`a_i` and :math:`b_i` is a zero-mean Gaussian
with unit variance, :math:`a_i \sim \mathcal{N}(0, 1)`,
reflecting the expectation that contamination amplitudes are at most
of order unity.  The noise parameter :math:`\sigma > 0` has a half-normal
prior.  In the rotated template basis (see :ref:`Template PCA rotation`) the
prior factorises across parameters, improving MCMC mixing.

``sys_mapping`` uses the **affine-invariant ensemble sampler**
(`emcee <https://emcee.readthedocs.io>`_, Foreman-Mackey et al. 2013).
This sampler uses :math:`n_{\rm walkers}` parallel chains (walkers) that
share information via stretch and walk moves.  Key properties:

* Works efficiently for posteriors with correlated parameters.
* Automatically adapts to the posterior shape.
* The only tuning parameter is the number of walkers
  (recommendation: :math:`n_{\rm walkers} \geq 4\,n_{\rm dim}`, typical
  choice :math:`n_{\rm walkers} = 150` for :math:`n_{\rm dim} = 7`).

The MAP estimate is taken as the **posterior median** of the flat chain
(after discarding the burn-in phase).  The posterior variance is estimated
from the chain covariance matrix.

**Recommended settings:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Default / recommendation
   * - ``n_walkers``
     - 150 (>= 4 x n_dim)
   * - ``n_steps``
     - 1500 (beyond convergence)
   * - ``n_burn``
     - 300 (discard early transient)
   * - Convergence check
     - ``emcee`` autocorrelation tau

**Technical implementation.**

*Log-probability wrapper* (:func:`~sys_mapping.inference.make_log_prob`):
converts the JAX log-likelihood into an emcee-compatible Python callable.
``delta_g_obs`` and ``delta_t`` are converted to ``jnp.asarray`` once at
construction time to avoid repeated host–device copies.  A hard
``sigma <= sigma_min`` (default ``1e-6``) guard returns ``-inf``
without calling JAX, short-circuiting the dispatch overhead.

*Vectorised mode* (``vectorize=True``): builds a single
``jax.jit(jax.vmap(log_likelihood))`` kernel that accepts the full
``(n_walkers, n_dim)`` position matrix and returns ``(n_walkers,)``
log-probabilities in one GPU dispatch per step.  This is the primary
source of GPU speedup and requires ``emcee >= 3.0``.

*Walker initialisation* in :func:`~sys_mapping.inference.run_mcmc`:

.. code-block:: python

   p0[:, :n_cont] = rng.normal(0.0, 1e-3, (n_walkers, n_cont))
   p0[:, n_cont]  = std(delta_g_obs) * (1 + rng.normal(0.0, 0.01, n_walkers))
   # gamma (skew) initialised to N(0, 0.1) when use_skewed=True

The :math:`10^{-3}` scatter in contamination amplitudes places walkers
near zero (the prior centre) while providing enough diversity to explore
the posterior within the first ~50 steps.  ``sigma`` is initialised near
the observed overdensity standard deviation, which is a good prior for the
noise level.

*Point estimate*: :func:`~sys_mapping.inference.get_mle_params` returns
``np.median(flat_chain, axis=0)`` — a robust estimator that is
insensitive to outlier walkers and converges to the true parameter at
:math:`O(1/\sqrt{n_{\rm samples}})`.

*Covariance* (:func:`~sys_mapping.inference.get_param_covariance_from_chain`):
slices the flat chain by column rather than unpacking sample by sample:

.. code-block:: python

   a_arr = flat_chain[:, :n_sys]                   # (n_samples, n_sys)
   b_arr = flat_chain[:, n_sys : 2*n_sys]          # combined only
   cov_a = np.cov(a_arr, rowvar=False)              # (n_sys, n_sys)

The full covariance (not just the diagonal) is needed to correctly
propagate uncertainty through the PCA back-transformation
``cov_orig = R.T @ cov_rot @ R``.

**Performance.** Total wall time scales as
:math:`O(n_{\rm walkers} \times n_{\rm steps} \times N_{\rm pix} \times n_s)`.
A typical run with ``n_walkers = 250``, ``n_steps = 1500``,
:math:`N_{\rm pix} \approx 50\,000`, :math:`n_s = 5` takes a few minutes on CPU;
the JAX likelihood is compiled once (~227 ms) then runs at ~284 μs/eval.

**Functions:**
:func:`~sys_mapping.inference.make_log_prob`,
:func:`~sys_mapping.inference.run_mcmc`,
:func:`~sys_mapping.inference.get_mle_params`,
:func:`~sys_mapping.inference.get_param_variance_from_chain`,
:func:`~sys_mapping.inference.get_param_covariance_from_chain`

----

.. _Template PCA rotation:

Template PCA rotation
----------------------

**Reference:** `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_,
Appendix A.

**Module:** :mod:`sys_mapping.correction`

**Physical motivation.** Systematic template maps are often correlated with
each other (e.g. seeing correlates with depth because long exposures under
good seeing produce deeper images).  When templates are correlated, the
contamination amplitudes :math:`a_i` cannot be determined independently —
the likelihood has ridges in parameter space, MCMC chains mix slowly, and
the inferred amplitudes in the original basis are unreliable.

**PCA diagonalisation.** The template covariance matrix is:

.. math::

   C_{ij} = \frac{1}{N}\sum_{p=1}^N t_i(p)\,t_j(p)

(a symmetric, positive semi-definite :math:`n_s \times n_s` matrix).
Its eigendecomposition is :math:`C = V D V^\top` where :math:`V` is
the :math:`n_s \times n_s` orthogonal matrix of eigenvectors and
:math:`D = \mathrm{diag}(\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_{n_s})`
contains the eigenvalues (template variances in the rotated basis).

The **rotated templates** are:

.. math::

   \mathbf{T}' = V^\top \mathbf{T}, \qquad
   t'_j(p) = \sum_i V_{ji}\,t_i(p)

so that :math:`C' = V^\top C V = D` is diagonal.  The rotated templates are
orthogonal by construction:

.. math::

   \frac{1}{N}\sum_p t'_i(p)\,t'_j(p) = \lambda_i\,\delta_{ij}

In the rotated basis, the likelihood factorises across parameters and MCMC
convergence is faster.  The rotation is applied *before* MCMC and the
back-transformation is applied *after*:

.. math::

   a_i = (V\,\mathbf{a}')_i = \sum_j V_{ij}\,a'_j, \qquad
   b_i = (V\,\mathbf{b}')_i = \sum_j V_{ij}\,b'_j

The eigenvalues :math:`\lambda_j` measure the variance of each rotated
template and indicate how many independent systematic modes are present.
Large gaps in the eigenvalue spectrum suggest that some templates are
essentially redundant.

**Technical implementation.**

:func:`~sys_mapping.correction.rotate_templates` performs the following
steps on a ``(n_sys, n_pix)`` NumPy array:

.. code-block:: python

   cov = delta_t @ delta_t.T / n_pix          # (n_sys, n_sys), O(n_sys² × n_pix)
   eigenvalues, eigenvectors = np.linalg.eigh(cov)   # symmetric → faster than eig
   idx = np.argsort(eigenvalues)[::-1]         # descending order
   rotation_matrix = eigenvectors[:, idx].T   # V^T, rows are eigenvectors
   delta_t_rot = rotation_matrix @ delta_t    # (n_sys, n_pix)

``numpy.linalg.eigh`` exploits the symmetry and positive semi-definiteness
of the covariance matrix (LAPACK ``dsyevd`` driver), making it faster and
more accurate than the general ``eig`` decomposition.  Cost:
:math:`O(n_s^3 + n_s \times N)`.  At ``n_sys = 5``, ``n_pix = 10\,000``
the call takes approximately **177 μs**.

Numerical guarantees:

* Off-diagonal elements of the rotated covariance ``delta_t_rot @ delta_t_rot.T / n_pix``
  are :math:`< 10^{-10}` (orthogonality).
* Rotation matrix satisfies ``R @ R.T = I`` to ``atol < 1e-12``.
* Sum of eigenvalues equals ``trace(cov)`` to ``rtol < 1e-10`` (variance conservation).

:func:`~sys_mapping.correction.transform_params_from_rotated` applies the
inverse transformation ``a_orig = V @ a_rot`` where ``V = R.T``:

.. code-block:: python

   V = rotation_matrix.T   # (n_sys, n_sys)
   a_orig = V @ a_rot      # two small (n_sys, n_sys) × (n_sys,) products
   b_orig = V @ b_rot

Cost: two :math:`n_s \times n_s` matrix–vector multiplies (**~4 μs**).
Roundtrip precision (rotate → infer → back-transform) is ``atol < 1e-12``
in float64.

In :func:`~sys_mapping.regression.run_decontamination` (MCMC methods),
the covariance matrix is propagated back as:

.. code-block:: python

   cov_a = R.T @ cov_a_rot @ R   # (n_sys, n_sys) sandwich product

so that uncertainty in the rotated basis is correctly mapped to the
original template basis without approximation.

**Functions:**
:func:`~sys_mapping.correction.rotate_templates`,
:func:`~sys_mapping.correction.transform_params_from_rotated`

----

Noise debiasing
---------------

**Reference:** `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_,
Eq. 21.

**Module:** :mod:`sys_mapping.correction`

**Physical motivation.** When the true contamination amplitude :math:`a_i`
is small (close to zero), the posterior median :math:`\hat{a}_i` fluctuates
around zero due to noise.  The squared amplitude :math:`\hat{a}_i^2` is
therefore always positive even when the true amplitude is zero, and the
expectation is:

.. math::

   \mathbb{E}[\hat{a}_i^2] = a_i^2 + \mathrm{Var}[\hat{a}_i]

This is the classical *noise bias* familiar from power spectrum estimation
(where the noise power spectrum must be subtracted from the measured pseudo-
:math:`C_\ell`).

**Debiased estimator.** The noise-debiased squared amplitude is:

.. math::

   \tilde{a}_i^2 = \max\!\bigl(\hat{a}_i^2 - \mathrm{Var}[\hat{a}_i],\; 0\bigr)

and analogously:

.. math::

   \tilde{b}_i^2 = \max\!\bigl(\hat{b}_i^2 - \mathrm{Var}[\hat{b}_i],\; 0\bigr)

The :math:`\max(\cdot, 0)` clip enforces the physical constraint
:math:`a_i^2 \geq 0` (squared amplitudes are non-negative).  When the
true amplitude is zero, the expected debiased estimate is zero on average.

The debiased amplitudes :math:`\tilde{a}_i^2, \tilde{b}_i^2` are the
inputs to the two-point correction formula (see below).  Using the raw
(biased) squared amplitudes would over-subtract the template auto-correlation,
introducing a spurious dip in the corrected :math:`w(\theta)`.

**Technical implementation.**
:func:`~sys_mapping.correction.debias_params` is a pure NumPy function
operating on ``(n_sys,)`` float64 arrays:

.. code-block:: python

   a_sq = np.maximum(a_hat**2 - var_a, 0.0)   # clip at 0 (squared amp ≥ 0)
   b_sq = np.maximum(b_hat**2 - var_b, 0.0)

Cost: two element-wise squares, two subtractions, two ``np.maximum`` calls —
approximately **4 μs** regardless of :math:`n_s`.  When the exact posterior
variance is supplied, the debiased estimate recovers the true
:math:`a_i^2` to ``atol < 1e-10`` (float64 round-off limited).

The ``var_a`` and ``var_b`` inputs are the *diagonal* of the MCMC posterior
covariance returned by
:func:`~sys_mapping.inference.get_param_variance_from_chain`.  In the
PCA-rotated basis the off-diagonal covariance elements are small (templates
are orthogonal after rotation), so the diagonal approximation is accurate.

**Function:**
:func:`~sys_mapping.correction.debias_params`

----

Two-point function correction
------------------------------

**Reference:** `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_,
Eq. 15–16.

**Module:** :mod:`sys_mapping.correction`

**Derivation sketch.** The angular two-point correlation function of the
observed field is:

.. math::

   \hat{w}(\theta) = \bigl\langle \hat\delta_g(\hat{n})\,\hat\delta_g(\hat{n}')\bigr\rangle_\theta

Inserting the contamination model and expanding to first order in
:math:`a_i^2` and :math:`b_i^2` (all cross-terms between different templates
:math:`i \neq j` vanish on average because templates are uncorrelated with
:math:`\delta_g` and with each other after normalisation):

.. math::

   \hat{w}(\theta) \approx
     w_g(\theta) + \sum_i b_i^2\,w_g(\theta)\,w_{t_i t_i}(\theta)
     + \sum_i a_i^2\,w_{t_i t_i}(\theta)
   = w_g(\theta)\!\left[1 + \sum_i b_i^2\,w_{t_i t_i}(\theta)\right]
     + \sum_i a_i^2\,w_{t_i t_i}(\theta)

where :math:`w_{t_i t_i}(\theta) = \langle t_i(\hat{n})\,t_i(\hat{n}')\rangle_\theta`
is the template auto-correlation at angular separation :math:`\theta`.

Solving for :math:`w_g(\theta)` and replacing :math:`a_i^2 \to \tilde{a}_i^2`,
:math:`b_i^2 \to \tilde{b}_i^2` (debiased estimates):

.. math::

   \hat{w}_{\rm corr}(\theta) =
   \frac{\hat{w}(\theta) - \sum_i \tilde{a}_i^2\,w_{t_i t_i}(\theta)}
        {1 + \sum_i \tilde{b}_i^2\,w_{t_i t_i}(\theta)}

**When to use.** Apply this correction *after* measuring the raw
:math:`w(\theta)` from the galaxy catalog and the template auto-correlations
:math:`w_{t_i t_i}(\theta)` from the templates.  The correction requires
the debiased amplitudes from :func:`~sys_mapping.correction.debias_params`.
The corrected :math:`\hat{w}_{\rm corr}(\theta)` is an unbiased estimate
of :math:`w_g(\theta)` to first order in the contamination amplitudes.

For the harmonic-space (power spectrum) equivalent, see
:func:`~sys_mapping.correction.correct_power_spectrum_harmonic`.

**Technical implementation.**

:func:`~sys_mapping.contamination.compute_two_point_correction` operates
on JAX arrays and is the inner kernel:

.. code-block:: python

   add_bias  = jnp.einsum("i,ij->j", a_sq, template_correlations)  # (n_bins,)
   mult_bias = jnp.einsum("i,ij->j", b_sq, template_correlations)  # (n_bins,)
   w_corr    = (w_obs - add_bias) / (1.0 + mult_bias)

Two Einstein sums over ``(n_sys, n_bins)`` produce ``(n_bins,)`` vectors;
cost is :math:`O(n_s \times n_{\rm bins})` — negligible.  Execution time:
approximately **576 μs** at ``n_sys = 5``, ``n_bins = 15`` (dominated by
JAX dispatch overhead).  Matches the analytic formula to ``rtol = 1e-6``
(single float32 einsum).

The convenience wrapper
:func:`~sys_mapping.correction.correct_two_point_function` chains the
NumPy debiasing step with the JAX correction kernel and converts the result
back to a NumPy array:

.. code-block:: python

   a_sq, b_sq = debias_params(a_hat, b_hat, var_a, var_b)   # NumPy, ~4 μs
   w_corr = compute_two_point_correction(                    # JAX, ~576 μs
       jnp.asarray(w_obs), jnp.asarray(a_sq),
       jnp.asarray(b_sq), jnp.asarray(template_correlations))
   return np.asarray(w_corr)                                 # back to NumPy

The ``template_correlations`` array (shape ``(n_sys, n_bins)``) is the
auto-correlation :math:`w_{t_i t_i}(\theta)` of each template map,
measured with the same angular binning as the galaxy :math:`w(\theta)`.

**Functions:**
:func:`~sys_mapping.correction.correct_two_point_function`,
:func:`~sys_mapping.correction.correct_power_spectrum_harmonic`

----

Likelihood ratio test (model selection)
-----------------------------------------

**Reference:** `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_,
Eq. 19; Wilks 1938 (asymptotic theorem).

**Module:** :mod:`sys_mapping.model_selection`

**Physical motivation.** Given data, we want to know whether the
multiplicative contamination terms :math:`b_i` are statistically necessary,
or whether the simpler additive model with :math:`b_i = 0` provides an
adequate description.  The LRT is the standard frequentist tool for
comparing nested models.

**The LRT statistic.** For two nested models
:math:`\mathcal{M}_0 \subset \mathcal{M}_1` with MLEs
:math:`\hat{\Theta}_0` and :math:`\hat{\Theta}_1`:

.. math::

   \lambda_{\rm LR} = 2\!\left[\ln\mathcal{L}(\hat{\Theta}_1) -
                               \ln\mathcal{L}(\hat{\Theta}_0)\right]
   \;\xrightarrow{N\to\infty}\; \chi^2(r), \quad r = k_1 - k_0

where :math:`k_0, k_1` are the numbers of free parameters in each model.
For the additive (:math:`n_s + 1`) vs combined (:math:`2n_s + 1`)
comparison, :math:`r = n_s`.

**Interpretation.** A large :math:`\lambda_{\rm LR} \gg \chi^2_{r,0.95}`
(the 95th percentile of the :math:`\chi^2(r)` distribution) means the
combined model is strongly preferred — the data require multiplicative
contamination.  A small :math:`\lambda_{\rm LR} \approx 0` means the
simpler additive model is sufficient.

**Note on the sign convention.** The LRT statistic is always
:math:`\lambda_{\rm LR} \geq 0` because adding parameters cannot decrease
the maximised log-likelihood.  If the two likelihoods are nearly equal,
:math:`\lambda_{\rm LR} \approx 0` and the data cannot distinguish the
models.

**Technical implementation.**
The LRT uses the same MCMC posterior median as the point estimate for both
models.  The test statistic is computed by evaluating
:func:`~sys_mapping.likelihood.make_log_likelihood` at the two parameter
vectors and differencing:

.. code-block:: python

   lrt = 2.0 * (log_lik_combined(theta_1) - log_lik_additive(theta_0))

Both log-likelihood calls use the JIT-compiled function objects, so the
evaluation cost is two forward passes through the likelihood
(~284 μs each).  The p-value is obtained from the :math:`\chi^2(r)`
CDF via ``scipy.stats.chi2.sf(lrt, df=r)`` where
``r = n_sys`` for the additive-vs-combined comparison.

Because the MCMC posterior median — not a true MLE from gradient
optimisation — is used as the parameter estimate, the LRT statistic is a
conservative approximation: the posterior median is inside the posterior
but may not be exactly at the likelihood maximum, so
:math:`\lambda_{\rm LR}` may be slightly underestimated.  In practice,
for well-sampled posteriors (``n_steps >= 1500``) the bias is
negligible.

**Function:**
:func:`~sys_mapping.model_selection.likelihood_ratio_test`

----

Pseudo-:math:`C_\ell` power spectrum
--------------------------------------

**References:**
`Elsner, Leistedt & Peiris 2016 <https://ui.adsabs.harvard.edu/abs/2016MNRAS.456.2095E/abstract>`_
(MNRAS 456, 2095), Sec. 2–3;
`Elsner, Leistedt & Peiris 2017 <https://ui.adsabs.harvard.edu/abs/2017MNRAS.465.1847E/abstract>`_
(MNRAS 465, 1847);
`Leistedt & Peiris 2014 <https://ui.adsabs.harvard.edu/abs/2014MNRAS.444....2L/abstract>`_
(MNRAS 444, 2);
`Ho et al. 2012 <https://ui.adsabs.harvard.edu/abs/2012ApJ...761...14H/abstract>`_
(ApJ 761, 14).

**Module:** :mod:`sys_mapping.power_spectrum`

**Physical motivation.** Instead of working with the real-space two-point
function :math:`w(\theta)`, one can characterise galaxy clustering in
harmonic space via the angular power spectrum :math:`C_\ell`.  Each
multipole :math:`\ell` probes a characteristic angular scale
:math:`\theta \sim \pi/\ell` (e.g. :math:`\ell = 100` corresponds to
:math:`\approx 1.8°`).  Harmonic-space estimators allow easy comparison
with theoretical predictions computed via Boltzmann codes (CAMB, CLASS).

**Pseudo-:math:`C_\ell` estimator.** Because the galaxy survey covers only
a fraction of the sky, spherical harmonic modes are mixed by the survey mask
:math:`W(p)`.  The *pseudo-:math:`C_\ell`* estimator simply applies the mask
and computes the power spectrum of the masked map:

.. math::

   \tilde{C}_\ell = \frac{1}{2\ell+1}\sum_{m=-\ell}^{\ell}
   \left|\sum_p W(p)\,\hat\delta_g(p)\,Y_{\ell m}^*(p)\,\Omega_p\right|^2

where :math:`Y_{\ell m}(p)` are the spherical harmonic basis functions and
:math:`\Omega_p = 4\pi/N_{\rm pix}` is the pixel solid angle.
In ``sys_mapping`` this is computed by passing the masked map to
``healpy.anafast``.

The pseudo-:math:`C_\ell` is related to the true power spectrum :math:`C_\ell`
via the **MASTER equation** (Mode-coupling Matrix in Coupled Harmonics
for Angular Spectra Techniques, Hivon et al. 2002):

.. math::

   \langle\tilde{C}_\ell\rangle = \sum_{\ell'} M_{\ell\ell'}\,C_{\ell'}

where the mode-coupling matrix :math:`M_{\ell\ell'}` depends only on the
mask power spectrum :math:`\tilde{C}_\ell^{WW}`.

**Template subtraction (TS)** in harmonic space (Elsner+2016, Eq. 1–2).
Systematic contamination adds a spurious contribution to the measured
pseudo-:math:`C_\ell`.  For an additive contaminant with amplitude
:math:`\hat\alpha_i`, the cleaned power spectrum is:

.. math::

   \tilde{C}_\ell^{\rm TS} = \hat{C}_\ell^{dd}
   - \sum_i \hat\alpha_i\,\hat{C}_\ell^{t_i t_i}

where :math:`\hat{C}_\ell^{dd}` is the measured pseudo-:math:`C_\ell` of
the galaxy map and :math:`\hat{C}_\ell^{t_i t_i}` is the pseudo-:math:`C_\ell`
of template :math:`i`.

**Harmonic bias** (Elsner+2016, Eq. 8). Template subtraction introduces
a residual additive bias:

.. math::

   b_\ell = -\frac{n_s}{2\ell+1}

This arises because subtracting a linear combination of templates removes
:math:`n_s` degrees of freedom from the sky, reducing the effective number
of modes at each :math:`\ell` by :math:`n_s`.  At low multipoles
(:math:`\ell \lesssim 10`) and with many templates (:math:`n_s \gtrsim 10`)
this bias can be significant and must be corrected.

**Mode Projection (BMP/EMP)**. An alternative to template subtraction is
to *project out* the template modes from the covariance matrix before
power spectrum estimation.  In the **Basic Mode Projection** (BMP), the
pixel-pixel covariance matrix is modified to marginalise over the
template amplitudes.  The **Extended Mode Projection** (EMP, Leistedt+2014)
additionally discards multipoles where the template signal exceeds a
threshold :math:`k` times the noise, with an analytical bias correction.

**Technical implementation.**

*Pseudo-Cℓ measurement* (:func:`~sys_mapping.power_spectrum.measure_pseudo_cl`):

.. code-block:: python

   masked_map = delta_g * mask_float          # element-wise multiplication
   cl = hp.anafast(masked_map, lmax=lmax,
                   use_pixel_weights=True)    # healpy spherical harmonic transform
   ell = np.arange(len(cl))

The mask is applied as a float weight map (``True`` → 1.0, ``False`` → 0.0).
``use_pixel_weights=True`` enables healpy's pixel-area weighting scheme, which
corrects for the geometric distortion of the HEALPix equal-area pixels at
high Galactic latitude and improves accuracy at :math:`\ell \gtrsim N_{\rm side}`.
The returned ``pseudo_cl`` is the power of the masked map and must be divided
by :math:`W_2 = N_{\rm pix}^{-1}\sum_p W^2(p)` (the mask power) for a
first-order sky-fraction correction.

*Template subtraction* (:func:`~sys_mapping.power_spectrum.subtract_template_cl`):
calls ``hp.anafast`` once per template to compute
:math:`\tilde{C}_\ell^{t_i t_i}`, then subtracts the weighted contribution:

.. code-block:: python

   cl_cleaned = pseudo_cl.copy()
   for a_i, t_i in zip(alpha, delta_t):
       masked_template = t_i * mask_float
       cl_t = hp.anafast(masked_template, lmax=lmax, use_pixel_weights=True)
       cl_cleaned -= a_i * cl_t

Cost: :math:`O(n_s)` harmonic transforms.  Each transform has complexity
:math:`O(N_{\rm pix} \log N_{\rm pix})` (the ``healpy`` SHT).

*Harmonic bias correction* (:func:`~sys_mapping.power_spectrum.harmonic_bias`):
a closed-form expression requiring no iteration:

.. code-block:: python

   bias = -n_templates / (2.0 * ell + 1.0)   # (n_ell,)

*Mode Projection* (:func:`~sys_mapping.power_spectrum.mode_projection_bias`):
both BMP and EMP use the same ``harmonic_bias`` formula; EMP additionally
applies a per-multipole mask based on the template-to-signal ratio:

.. code-block:: python

   template_ratio = |harmonic_bias(1, ell)| / (cl_signal / n_templates)
   project_mask   = template_ratio > threshold
   n_projected    = project_mask.sum()
   bias = np.where(project_mask, harmonic_bias(n_projected, ell), 0.0)

When ``coupling_matrix`` is not ``None``, the scalar bias vector is
propagated through the mode-coupling matrix
:math:`b_\ell^{\rm pseudo} = M_{\ell\ell'} b_{\ell'}` before subtraction.
Passing ``coupling_matrix=None`` uses the identity (full-sky approximation).

**Functions:**
:func:`~sys_mapping.power_spectrum.measure_pseudo_cl`,
:func:`~sys_mapping.power_spectrum.subtract_template_cl`,
:func:`~sys_mapping.power_spectrum.harmonic_bias`,
:func:`~sys_mapping.power_spectrum.mode_projection_bias`

----

Block bootstrap covariance
---------------------------

**References:**
`Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_, Sec. 6.2;
`Ross et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.417.1350R/abstract>`_.

**Module:** :mod:`sys_mapping.bootstrap`

**Physical motivation.** Galaxy overdensity maps have spatially correlated
structure on scales of tens to hundreds of Mpc (the same clustering signal
we are trying to measure).  This means that the uncertainty in any
estimator computed from these maps cannot be estimated by treating pixels
as independent — the effective number of degrees of freedom is far smaller
than the pixel count.  A *spatial block bootstrap* preserves the spatial
covariance structure by resampling *patches* rather than individual pixels.

**Algorithm.** The survey footprint is divided into :math:`K` approximately
equal-area spatial patches by coarsening to a lower-resolution HEALPix
grid.  Each bootstrap resample draws :math:`K` patches with replacement,
then evaluates the estimator :math:`\hat\theta` on the resampled dataset.
The variance estimate is:

.. math::

   \widehat{\rm Var}[\hat\theta] = \frac{1}{B-1}\sum_{b=1}^B
   \bigl(\hat\theta_b - \bar{\hat\theta}\bigr)^2, \qquad
   \bar{\hat\theta} = \frac{1}{B}\sum_{b=1}^B \hat\theta_b

where :math:`B` is the number of bootstrap resamples (typical:
:math:`B = 100–500`).

**Typical settings.** Use :math:`K \approx 10–30` patches and
:math:`B \approx 100–200` resamples.  Fewer patches reduce the spatial
resolution of the covariance estimate; more patches increase computational
cost linearly.

**Technical implementation.**

*Patch assignment* (:func:`~sys_mapping.bootstrap._assign_patches`):
the footprint is partitioned by coarsening to a lower HEALPix NSIDE:

.. code-block:: python

   nside_patch = max(1, nside // int(sqrt(nside2npix(nside) / n_patches)))
   theta, phi  = hp.pix2ang(nside, pixel_indices)   # unmasked pixels only
   coarse_pix  = hp.ang2pix(nside_patch, theta, phi)
   patch_ids   = np.unique(coarse_pix, return_inverse=True)[1]  # 0..K-1

This ensures patches are approximately equal area (HEALPix is equal-area at
each resolution) and spatially contiguous, preserving large-scale galaxy
clustering modes within each patch.  The coarse NSIDE is chosen so that
there are :math:`\approx n_{\rm patches}` unique coarse pixels.

*Bootstrap loop* (:func:`~sys_mapping.bootstrap.block_bootstrap_variance`):

.. code-block:: python

   for _ in range(n_bootstrap):
       sampled_patches = rng.choice(unique_patches, size=K, replace=True)
       pixel_mask = np.concatenate(
           [np.where(patch_ids == p)[0] for p in sampled_patches])
       dg_boot = delta_g_obs[pixel_mask]
       dt_boot = delta_t[:, pixel_mask]
       bootstrap_estimates.append(estimator(dg_boot, dt_boot))
   variance = np.var(np.stack(bootstrap_estimates), axis=0, ddof=1)

Resampling ``K`` patches with replacement means that some patches are
included multiple times and others are omitted, which mimics the sampling
variance of the survey area.  The ``ddof=1`` divisor gives an unbiased
estimate of the variance of the estimator.

**Performance.** At ``n_boot = 50``, ``n_patches = 8``, NSIDE 32, the
entire bootstrap loop takes approximately **32 ms** for a simple mean
estimator.  Total cost scales linearly as
:math:`O(n_{\rm boot} \times \text{cost}(\hat\theta))`.

**Function:**
:func:`~sys_mapping.bootstrap.block_bootstrap_variance`

----

Jack-knife covariance
----------------------

**References:**
`Ross et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.417.1350R/abstract>`_
(MNRAS 417, 1350), Sec. 4.3;
`Ho et al. 2012 <https://ui.adsabs.harvard.edu/abs/2012ApJ...761...14H/abstract>`_
(ApJ 761, 14).

**Module:** :mod:`sys_mapping.bootstrap`

**Physical motivation.** The jack-knife is a *deterministic* leave-one-out
estimator that does not require random resampling.  It is more stable than
the bootstrap for a fixed number of patches because it uses all :math:`K`
leave-one-out datasets rather than a random subset.

**Algorithm.** For each of the :math:`K` patches, remove that patch and
evaluate the estimator on the remaining :math:`K-1` patches.
The jack-knife covariance matrix is:

.. math::

   \widehat{\rm Cov}[\hat\theta] = \frac{K-1}{K}
   \sum_{k=1}^K \bigl(\hat\theta_{(-k)} - \bar{\hat\theta}\bigr)
                \bigl(\hat\theta_{(-k)} - \bar{\hat\theta}\bigr)^\top

where :math:`\hat\theta_{(-k)}` is the estimator with patch :math:`k`
removed, and :math:`\bar{\hat\theta} = K^{-1}\sum_k \hat\theta_{(-k)}`
is the mean over jack-knife samples.
The prefactor :math:`(K-1)/K` ensures the estimator is unbiased for the
true covariance in the limit of large :math:`K`.

**Comparison with block bootstrap.** For a well-designed spatial
decomposition, the jack-knife and block bootstrap covariances should agree
to within :math:`\sim 20\%`.  The jack-knife is faster (deterministic,
:math:`K` estimator calls vs :math:`B` for bootstrap) but more sensitive
to the choice of patch geometry.  The bootstrap can capture asymmetric
distributions that the jack-knife misses.

**Technical implementation.**
The jack-knife uses the same ``_assign_patches`` helper as the bootstrap,
so patch geometry is identical.  The loop is fully deterministic:

.. code-block:: python

   leave_one_out = []
   for k in unique_patches:
       keep = patch_ids != k            # boolean mask
       estimate_k = estimator(delta_g_obs[keep], delta_t[:, keep])
       leave_one_out.append(estimate_k)
   estimates = np.stack(leave_one_out)  # (K, n_out)
   mean_est  = estimates.mean(axis=0)
   residuals = estimates - mean_est
   cov = (K-1)/K * (residuals.T @ residuals)

The outer product ``residuals.T @ residuals`` computes the full
``(n_out, n_out)`` covariance matrix in one BLAS-3 call.
For scalar estimators (``n_out = 1``) this reduces to a sum of squared
deviations.  A ``UserWarning`` is issued if ``n_patches < 5`` (the
covariance estimate is unreliable when few patches are available).

**Cost**: :math:`K` evaluations of the estimator (compared to
:math:`B` for bootstrap, with :math:`K \ll B` typically).  For ``K = 10``
patches the jack-knife is 10–20× faster than 100–200 bootstrap resamples.

**Function:**
:func:`~sys_mapping.bootstrap.jackknife_covariance`

----

ElasticNet regression
----------------------

**Reference:**
`Weaverdyck & Huterer 2021 <https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.5061W/abstract>`_
(MNRAS 503, 5061), Eq. 9–10.

**Module:** :mod:`sys_mapping.regression`

**Physical motivation.** When there are many systematic templates
(:math:`n_s \gg 1`) and only a few carry genuine contamination, standard
OLS tends to over-fit: it assigns non-zero amplitudes to all templates,
including those that are noise-dominated.  Regularised regression shrinks
unimportant amplitudes toward zero while retaining the dominant ones.

**ElasticNet loss function.** ElasticNet minimises:

.. math::

   \mathcal{L}(\boldsymbol\alpha) = \frac{1}{2N}
   \left\|\hat{\boldsymbol\delta}_g - \mathbf{T}^\top\boldsymbol\alpha\right\|_2^2
   + \underbrace{\lambda_1\|\boldsymbol\alpha\|_1}_{\rm Lasso\;(L1)}
   + \underbrace{\lambda_2\|\boldsymbol\alpha\|_2^2}_{\rm Ridge\;(L2)}

where :math:`\boldsymbol\alpha = (a_1,\ldots,a_{n_s})^\top` are the
additive contamination amplitudes, :math:`\mathbf{T}` is the
:math:`n_s \times N` template matrix, and the regularisation strengths
satisfy :math:`\lambda_1 = \lambda\,\rho` and :math:`\lambda_2 = \lambda(1-\rho)/2`
for a total regularisation :math:`\lambda` and L1 fraction
:math:`\rho \in [0,1]` (``l1_ratio`` in the code).

The L1 term (**Lasso** regularisation) promotes sparsity — small
amplitudes are shrunk exactly to zero, automatically selecting the
dominant templates.  The L2 term (**Ridge** regularisation) stabilises
correlated templates by sharing the coefficient equally.  The ElasticNet
combination is preferable when templates are both numerous and correlated.

When ``alpha_reg=None`` the optimal :math:`\lambda` is found by
:math:`k`-fold cross-validation on the pixel grid.

**Pixel weights.** After fitting, systematic weights are assigned as:

.. math::

   w(p) = \frac{1}{\max\!\left(1 + \hat{\boldsymbol\alpha}\cdot\mathbf{t}(p),\;\epsilon\right)}

with :math:`\epsilon = 10^{-6}` to prevent division by zero.

**Requires:** ``scikit-learn >= 1.3``  (``pip install sys_mapping[regression]``)

**Technical implementation.**

:func:`~sys_mapping.regression.elasticnet_contamination_fit` transposes
the template matrix so that scikit-learn receives the standard design matrix
convention (samples × features):

.. code-block:: python

   X = delta_t.T           # (n_pix, n_sys) — samples × features for sklearn
   y = delta_g_obs         # (n_pix,)

When ``alpha_reg=None`` (auto-tune), ``sklearn.linear_model.ElasticNetCV``
runs ``cv_folds``-fold cross-validation on the pixel grid.  The pixel grid
is split into contiguous chunks of roughly :math:`N/k` pixels each
(scikit-learn's default stratified-free split), so folds may mix spatially.
For strongly spatially correlated overdensity maps this can underestimate the
optimal :math:`\lambda`; in practice the effect is small relative to the
statistical noise.  The CV-selected ``alpha_`` is stored in ``cv_info``.

.. code-block:: python

   model = ElasticNetCV(l1_ratio=l1_ratio, cv=cv_folds, n_jobs=-1)
   model.fit(X, y)
   alpha_reg = float(model.alpha_)           # optimal λ from CV
   cv_scores = model.mse_path_.mean(axis=-1) # mean MSE vs λ grid

The fitted coefficients are extracted as ``model.coef_`` (shape
``(n_sys,)``).  Per-pixel weights are then computed and clipped to
:math:`[1/20, 20]` to prevent numerical blow-up in pixels where
:math:`1 + \hat{\boldsymbol\alpha}\cdot\mathbf{t}(p) \approx 0`:

.. code-block:: python

   denominator = np.maximum(1.0 + alpha_hat @ delta_t, 1e-6)
   weights = np.clip(1.0 / denominator, 1.0/20.0, 20.0)

The clip bound ``_ISD_MAX_WEIGHT = 20`` is shared with the ISD pipeline;
it corresponds to a denominator floor of 0.05 — generous enough not to
bias well-behaved scenarios.

**Functions:**
:func:`~sys_mapping.regression.elasticnet_contamination_fit`,
:func:`~sys_mapping.regression.method_comparison`

----

Iterative Systematics Decontamination (ISD)
--------------------------------------------

**References:**
`Rodríguez-Monroy et al. 2025 <https://ui.adsabs.harvard.edu/abs/2025arXiv250907943R/abstract>`_;
`Rezaie et al. 2020 <https://arxiv.org/abs/1907.11355>`_.

**Module:** :mod:`sys_mapping.regression`

**Physical motivation.** Standard OLS fits a linear model
:math:`\hat\delta_g \approx \sum_i \alpha_i t_i`.
When contamination is multiplicative, the correct model is
:math:`\hat\delta_g \approx \delta_g(1 + \sum_i b_i t_i)`, which is
non-linear in the templates.  ISD approximates this non-linearity by
expanding the template basis with polynomial cross-terms and iterating
the OLS solution.

**Algorithm.** Starting with the original templates
:math:`\{t_i\}_{i=1}^{n_s}`, the expanded basis at polynomial order
:math:`d` includes all products :math:`t_i^{k_1} t_j^{k_2}\cdots` with
:math:`k_1 + k_2 + \cdots \leq d`.  At order :math:`d=1` this is just
the original templates (standard OLS).  At order :math:`d=3` the cross
terms :math:`t_i t_j` (for :math:`i \leq j`) and triple products
:math:`t_i t_j t_k` are added.  For :math:`n_s` templates the expanded
basis has :math:`\binom{n_s + d}{d} - 1` columns (e.g. 55 columns for
:math:`n_s = 5, d = 3`).

The iterative loop:

1. Form the weighted normal equations using per-pixel weights
   :math:`w^{(k)}(p)`:

   .. math::

      \bigl(\mathbf{X}^\top \mathbf{W}^{(k)} \mathbf{X} + \mathbf{\Lambda}\bigr)
      \hat{\boldsymbol\alpha}^{(k)} = \mathbf{X}^\top \mathbf{W}^{(k)} \hat{\boldsymbol\delta}_g

   where :math:`\mathbf{X}` is the :math:`(N_{\rm pix} \times N_{\rm exp})`
   expanded design matrix (constant across iterations),
   :math:`\mathbf{W}^{(k)} = \mathrm{diag}(w^{(k)})`, and
   :math:`\mathbf{\Lambda}` is an optional ridge penalty matrix
   (see the ridge regularisation note below).

2. Update weights from the **linear coefficients only** (first :math:`n_s`
   entries of :math:`\hat{\boldsymbol\alpha}^{(k)}`):

   .. math::

      w^{(k+1)}(p) = \frac{1}{1 + \hat{\boldsymbol\alpha}^{(k)}_{\rm lin}
                                   \cdot \mathbf{t}(p)}

3. Repeat until convergence:
   :math:`\|\mathbf{w}^{(k+1)} - \mathbf{w}^{(k)}\|_2 / \|\mathbf{w}^{(k)}\|_2 < \varepsilon = 10^{-5}`.

The polynomial cross-terms improve the linear coefficient estimates but
are not propagated into the weight formula, which keeps the same form as
plain OLS.  The cleaned field is recovered as
:math:`\hat\delta_g^{\rm clean}(p) = w(p)(1 + \hat\delta_g^{\rm obs}(p)) - 1`.

.. _ISD normal equations speedup:

.. note::

   **Speed (v0.2.4).** Since :math:`\mathbf{X}` is constant across
   iterations, it is precomputed once outside the loop as
   :math:`\mathbf{X}^\top` (shape :math:`N_{\rm exp} \times N_{\rm pix}`).
   The per-iteration cost is then one BLAS-3 matrix multiply

   .. math::

      \mathbf{X}^\top \mathbf{W}^{(k)} \mathbf{X}
      = \underbrace{(\mathbf{X}^\top \odot \mathbf{w}^{(k)})}_{\text{broadcast scale}}
        \mathbf{X}

   producing an :math:`(N_{\rm exp} \times N_{\rm exp})` matrix — tiny
   regardless of :math:`N_{\rm pix}` — followed by ``lstsq`` on this
   small system.  This replaces the previous approach of computing the SVD
   of the full :math:`(N_{\rm pix} \times N_{\rm exp})` matrix at each
   iteration, giving a **5–15× speed-up** per iteration for realistic pixel
   counts (:math:`N_{\rm pix} \gtrsim 10^4`).

.. _ISD-3 ridge regularisation:

.. note::

   **ISD-3 accuracy: ridge regularisation on polynomial columns (v0.2.4).**
   A key source of ISD-3 instability is a *fixed-point inconsistency*: the
   regression minimises residuals over all :math:`N_{\rm exp}` columns
   (including polynomial cross-terms), but only the first :math:`n_s`
   linear coefficients drive the weight update.  When the polynomial
   columns are correlated with the linear ones — which is always the case
   for survey systematics — they absorb a share of the linear signal.
   The linear coefficients :math:`\hat{\boldsymbol\alpha}_{\rm lin}` are
   left systematically small, and the iteration converges to a fixed point
   where the contamination has not actually been removed.

   The fix is a ridge penalty :math:`\lambda_{\rm poly}` applied
   *exclusively* to the polynomial-only columns (indices :math:`n_s:` of
   the expanded basis):

   .. math::

      \mathbf{\Lambda} = \mathrm{diag}\!\underbrace{(0,\ldots,0}_{n_s},
                                \underbrace{\lambda_{\rm poly},\ldots,\lambda_{\rm poly}}_{N_{\rm exp}-n_s})

   The linear columns are **never** penalised, so the weight update is not
   biased.  The penalty suppresses the polynomial terms just enough to
   prevent them from over-fitting noise or absorbing linear signal.

   **Recommended value:** :math:`\lambda_{\rm poly} \approx 10^{-3}\,{\rm Var}(\hat\delta_g)`.
   This is small enough to leave the result unaffected when the polynomial
   terms are genuinely needed, and large enough to stabilise ISD-3 when
   templates are strongly correlated.  Exposed as the ``lambda_poly``
   argument to :func:`~sys_mapping.regression.iterative_systematics_decontamination`
   and as ``isd_lambda_poly`` in :func:`~sys_mapping.regression.run_decontamination`.
   Default is ``0.0`` (plain OLS, backward-compatible).

**Convergence.** ISD-1 reliably converges within ≲ 25 iterations for all
tested configurations.  ISD-3 with ``lambda_poly = 0`` may still oscillate
when templates are strongly correlated; setting
``lambda_poly = 1e-3 * np.var(delta_g_obs)`` resolves this in practice.
In the worst case the function returns weights at ``max_iter``, which
remain a useful (if not fully converged) correction.

**When to use.**

* **ISD-1** (``poly_order=1``): equivalent to OLS; fastest option; use as
  a baseline.
* **ISD-3** (``poly_order=3``): partially corrects multiplicative
  contamination without a full MCMC run.  Set
  ``isd_lambda_poly = 1e-3 * np.var(delta_g_obs)`` when ISD-3 produces
  implausible solutions or fails to converge.
* **MCMC-combined**: preferred for high-precision analyses with known
  multiplicative contamination — it jointly infers additive and
  multiplicative amplitudes with full posterior uncertainties.

**Technical implementation — polynomial expansion.**
The expanded template basis is built with
``itertools.combinations_with_replacement``, which generates all
multi-indices of degree :math:`\leq d` in lexicographic order:

.. code-block:: python

   rows = [delta_t]   # (n_sys, n_pix) — linear terms always first
   for deg in range(2, poly_order + 1):
       for combo in combinations_with_replacement(range(n_sys), deg):
           product = ones(n_pix)
           for idx in combo:
               product *= delta_t[idx]
           rows.append(product[newaxis, :])
   delta_t_expanded = vstack(rows)  # (n_expanded, n_pix)

For ``n_sys = 5``, ``poly_order = 3`` this yields
:math:`\binom{5+3}{3} - 1 = 55` columns.  If ``n_pix < 10 × n_expanded``
the system would be poorly conditioned; ``poly_order`` is reduced
automatically with a ``UserWarning``.

**Technical implementation — weighted normal equations.**
The design matrix ``X = delta_t_expanded.T`` (shape ``(n_pix, n_expanded)``)
is computed *once* before the iteration loop.  Each iteration updates only
the diagonal weight matrix:

.. code-block:: python

   Xt = delta_t_expanded            # (n_expanded, n_pix), constant
   X  = delta_t_expanded.T          # (n_pix, n_expanded), constant
   ridge_diag = zeros(n_expanded)
   ridge_diag[n_sys:] = lambda_poly  # penalty on polynomial columns only
   for iteration in range(1, max_iter + 1):
       Xw   = Xt * weights           # (n_expanded, n_pix): broadcast scale
       XtWX = Xw @ X                 # (n_expanded, n_expanded): BLAS-3
       XtWX.flat[::n_expanded+1] += ridge_diag   # in-place diagonal add
       XtWy = Xw @ delta_g_obs       # (n_expanded,): BLAS-1
       alpha_hat_all, *_ = lstsq(XtWX, XtWy)    # small system

The key insight is that ``Xt * weights`` is a row-broadcast (scale each row
of ``Xt`` by the corresponding weight), which is a single NumPy ufunc call,
not a loop.  The resulting ``(n_expanded, n_expanded)`` matrix (e.g. 55×55)
is tiny and the subsequent ``lstsq`` is negligible.  This replaces the
previous approach of computing the SVD of the full
``(n_pix × n_expanded)`` matrix at each iteration, yielding a
**5–15× speed-up** for :math:`N_{\rm pix} \gtrsim 10^4`.

**Weight update and clipping.**
Only the first ``n_sys`` (linear) coefficients drive the weight update:

.. code-block:: python

   weights = 1.0 / max(1.0 + alpha_hat_all[:n_sys] @ delta_t, 1e-6)
   weights = np.clip(weights, 1.0/20.0, 20.0)   # hard bound

Pixels where ``1 + a@t → 0`` would otherwise receive weights up to
:math:`10^6`, making all subsequent weighted OLS fits numerically
meaningless.  The clip to :math:`[1/20, 20]` keeps the iteration stable.

**Convergence criterion.**

.. code-block:: python

   rel_change = ||w_new - w_old|| / (||w_old|| + 1e-30)
   if rel_change < tol (default 1e-5): break

ISD-1 typically converges in :math:`\lesssim 25` iterations.  ISD-3 with
``lambda_poly = 0`` may oscillate when templates are strongly correlated;
setting ``lambda_poly = 1e-3 × Var(δ_g)`` resolves this.

**Two-pass outlier handling** (in :func:`~sys_mapping.regression.run_decontamination`).
After pass 1, pixels that hit the weight clip boundary are flagged as
outliers:

.. code-block:: python

   boundary     = 20.0 * (1 - 1e-6)
   outlier_mask = (weights >= boundary) | (weights <= 1.0/boundary)

If any outliers exist *and* there are at least :math:`\max(10 n_s, 50)`
clean pixels, pass 2 refits ISD on the clean subset only, then applies the
clean coefficients to all pixels (including outliers).  This two-pass scheme
prevents the handful of pathological pixels from biasing the overall fit.
The ``isd_outlier_mask`` and ``isd_masked_fraction`` fields in the result
dict report how many pixels were flagged.

**Functions:**
:func:`~sys_mapping.regression.iterative_systematics_decontamination`,
:func:`~sys_mapping.regression.run_decontamination`

----

Null tests and SNR ranking
---------------------------

**References:**
`Ross et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.417.1350R/abstract>`_,
Sec. 3.2 (null tests);
`Tanidis et al. 2026 <https://ui.adsabs.harvard.edu/abs/2026MNRAS.547ag537T/abstract>`_
(SNR ranking);
`Weaverdyck & Huterer 2021 <https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.5061W/abstract>`_
(method comparison);
`Rodríguez-Monroy et al. 2025 <https://ui.adsabs.harvard.edu/abs/2025arXiv250907943R/abstract>`_
(footprint masking).

**Module:** :mod:`sys_mapping.diagnostics`

**Null test.** After applying the systematic correction, the per-pixel
weights :math:`w(p)` should be statistically uncorrelated with all template
maps.  If the correction is complete, the Pearson correlation:

.. math::

   r_i = \frac{\sum_p \bigl(w(p) - \bar{w}\bigr)\bigl(t_i(p) - \bar{t}_i\bigr)}
              {\sqrt{\sum_p(w(p)-\bar{w})^2}\,\sqrt{\sum_p(t_i(p)-\bar{t}_i)^2}}

should satisfy :math:`|r_i| \approx 0` for all :math:`i`.  A significant
residual correlation indicates that template :math:`i` has not been
adequately corrected.  Permutation p-values are computed by shuffling
:math:`w(p)` and recomputing :math:`r_i` a large number of times.

**SNR ranking.** Three estimators rank templates by their contaminating power:

* ``"template"`` — :math:`{\rm SNR}_i = |\hat\alpha_i| / \sigma_{\hat\alpha_i}`,
  where :math:`\sigma_{\hat\alpha_i}` is the OLS standard error.  This is
  the standard frequentist significance of each template in the linear fit.

* ``"data"`` — :math:`{\rm SNR}_i = |\mathrm{Corr}(\hat\delta_g, t_i)|`,
  the absolute Pearson correlation between the observed overdensity and
  template :math:`i`.  Fast to compute; does not require fitting.

* ``"peak"`` — the peak value of the cross pseudo-:math:`C_\ell`
  :math:`\hat{C}_\ell^{dt_i}` divided by the noise level.  Identifies
  templates that contaminate on specific angular scales.

**Footprint masking sensitivity.** Cutting the survey at different
mask thresholds (progressively removing pixels with extreme template values)
and measuring how the fitted amplitudes :math:`\hat\alpha_i` change is a
powerful sanity check: genuinely contaminated fields should show stable
or decreasing amplitudes as the worst-affected pixels are removed
(Rodríguez-Monroy+2025).

**Technical implementation.**

*Null test* (:func:`~sys_mapping.diagnostics.null_test_cross_correlations`):
the Pearson correlation is computed in a numerically stable form using
centred vectors:

.. code-block:: python

   w_centered = weights - weights.mean()
   w_norm     = sqrt(sum(w_centered**2))
   for i, t_i in enumerate(delta_t):
       t_centered = t_i - t_i.mean()
       t_norm     = sqrt(sum(t_centered**2))
       r[i] = sum(w_centered * t_centered) / (w_norm * t_norm)

Division by zero is guarded: if either norm is ``< 1e-30``,
``r[i] = 0.0`` is returned rather than raising an exception.

Permutation p-values shuffle only the weight vector (not the templates),
preserving the spatial structure of the templates:

.. code-block:: python

   for _ in range(n_bootstrap):
       w_shuffled = rng.permutation(w_centered)
       r_perm = abs(sum(w_shuffled * t_centered) / (w_norm * t_norm))
       if r_perm >= |r_obs|: count_extreme += 1
   p_value = (count_extreme + 1) / (n_bootstrap + 1)   # Laplace smoothing

The ``+1`` in numerator and denominator is the standard conservative
correction that ensures ``p_value > 0`` even when no permutation exceeds
the observed statistic.

*SNR ranking* (:func:`~sys_mapping.diagnostics.snr_template_ranking`):

* **``"template"``** — fits OLS on the full template matrix
  ``X = delta_t.T``, estimates residual variance
  :math:`\hat\sigma^2 = \text{RSS}/(N-n_s)`, and computes the OLS t-statistic
  per parameter:

  .. code-block:: python

     alpha_hat, *_ = lstsq(X, delta_g_obs)
     sigma2        = sum(residual**2) / max(n_pix - n_sys, 1)
     XtX_inv       = pinv(X.T @ X)
     param_var     = sigma2 * diag(XtX_inv)
     snr           = |alpha_hat| / sqrt(maximum(param_var, 1e-30))

  Uses ``np.linalg.pinv`` for numerical stability when templates are
  nearly collinear.

* **``"data"``** — computes ``|Corr(delta_g, t_i)|`` in a loop; cost
  :math:`O(n_s \times N)`.

* **``"peak"``** — calls ``hp.anafast`` twice per template (once for the
  galaxy map, once cross with each template), then normalises by the median
  power of the galaxy auto-spectrum as a noise-level proxy:

  .. code-block:: python

     cl_g        = hp.anafast(delta_g_obs, lmax=lmax)
     noise_level = median(|cl_g|) + 1e-30
     for t_i in delta_t:
         cl_cross = hp.anafast(delta_g_obs, map2=t_i, lmax=lmax)
         snr[i]   = max(|cl_cross|) / noise_level

*Footprint masking diagnostics*
(:func:`~sys_mapping.diagnostics.footprint_mask_diagnostics`):
pixels are ranked by their RMS template value:

.. code-block:: python

   rms_per_pixel = sqrt(mean(delta_t**2, axis=0))  # (n_pix,): ∑ t²_i / n_s
   rank_order    = argsort(rms_per_pixel)           # ascending

For each masking level the lowest-``rms`` pixels are dropped first, ensuring
that the surviving pixels have progressively higher systematic leverage.
OLS is refitted on the surviving pixels at each level, and the amplitude
scatter across levels quantifies sensitivity to the masking threshold:

.. code-block:: python

   alpha_all[k], *_ = lstsq(delta_t[:, keep].T, delta_g_obs[keep])
   scatter = std(alpha_all, axis=0)   # (n_sys,): near zero → robust

**Functions:**
:func:`~sys_mapping.diagnostics.null_test_cross_correlations`,
:func:`~sys_mapping.diagnostics.snr_template_ranking`,
:func:`~sys_mapping.diagnostics.footprint_mask_diagnostics`

----

Mock catalog generation
------------------------

**References:** `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_
(validation strategy);
`Coles & Jones 1991 <https://academic.oup.com/mnras/article/248/1/1/1033817>`_
(lognormal field model);
`Weaverdyck & Huterer 2021 <https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.5061W/abstract>`_
(mock injection tests).

**Module:** :mod:`sys_mapping.mocks`

**Physical motivation.** Validating a systematic decontamination method
requires knowing the true underlying galaxy field and the true contamination
amplitudes.  Synthetic (mock) catalogs are generated with a known ground
truth, so parameter recovery and decontamination quality can be measured
exactly — something impossible with real data.

**Lognormal galaxy field.** The true galaxy overdensity is:

.. math::

   \delta_g^{\rm true}(p) = e^{G(p) - \sigma^2/2} - 1

where :math:`G(p)` is a zero-mean Gaussian random field drawn from a
power spectrum :math:`C_\ell^G \propto (\ell+1)^{-2}`, normalised so that
the pixel-to-pixel variance equals :math:`\sigma^2`.  This form ensures:

* :math:`\delta_g^{\rm true}(p) > -1` everywhere (physically: number density
  :math:`n_g \propto 1 + \delta_g > 0`).
* :math:`\mathbb{E}[\delta_g] = 0` by the normalisation of the exponential.
* The distribution is positively skewed (characteristic of the real galaxy
  distribution).

**Poisson sampling.** Galaxy counts are drawn as:

.. math::

   n_g(p) \sim {\rm Poisson}\!\bigl[\bar{n}\,(1 + \hat\delta_g(p))\bigr]

where :math:`\bar{n}` is the mean galaxy count per pixel (set by the
``n_mean`` parameter) and :math:`\hat\delta_g(p)` is the contaminated
overdensity.  A random catalog of :math:`N_r = 8\bar{n}N` objects is
generated uniformly over the unmasked footprint.

**Technical implementation.**

*Lognormal field generation* (:func:`~sys_mapping.mocks.generate_lognormal_field`):
a Gaussian random field :math:`G(p)` is drawn with ``healpy.synfast`` from
a power-law angular power spectrum :math:`C_\ell^G \propto (\ell+1)^{-2}`,
normalised so that :math:`\mathrm{Var}(G) = \sigma^2`:

.. code-block:: python

   ell  = np.arange(lmax + 1)
   cl_g = (ell + 1.0)**(-2)
   cl_g /= cl_g.sum()       # normalise to unit variance
   cl_g *= sigma**2          # scale to target variance
   G    = hp.synfast(cl_g, nside=nside, lmax=lmax)
   delta_g_true = np.exp(G - sigma**2 / 2.0) - 1.0   # ensures E[δ] = 0

The :math:`-\sigma^2/2` shift is the standard log-normal mean correction
that ensures :math:`\mathbb{E}[\delta_g] = 0` exactly (when
:math:`G \sim \mathcal{N}(0, \sigma^2)` the expectation of
:math:`e^G` is :math:`e^{\sigma^2/2}`).

*Contamination injection*: the true overdensity is contaminated using
:func:`~sys_mapping.contamination.apply_contamination` with the known
ground-truth amplitudes ``(a_true, b_true)``, so that parameter recovery
tests compare inferred amplitudes directly against the injected values.

*Poisson sampling* (:func:`~sys_mapping.mocks.make_mock_catalog`):

.. code-block:: python

   n_g = rng.poisson(n_mean * (1.0 + delta_g_obs))   # (n_pix,) integer counts
   # Random catalog: N_r = 8 * n_mean * n_pix objects uniformly over footprint
   N_r = int(8 * n_mean * good_pixels.sum())

The ratio :math:`N_r / N_g \approx 8` ensures that the random catalog shot
noise contributes negligibly to the Landy–Szalay estimator variance
(Landy & Szalay 1993 recommend :math:`N_r / N_g \geq 5`).

**Functions:**
:func:`~sys_mapping.mocks.generate_lognormal_field`,
:func:`~sys_mapping.mocks.make_galactic_mask`,
:func:`~sys_mapping.mocks.make_mock_catalog`,
:func:`~sys_mapping.mocks.make_mock_suite`,
:func:`~sys_mapping.mocks.MockCatalog`

----

HEALPix map utilities
----------------------

**Module:** :mod:`sys_mapping.maps`

`HEALPix <https://healpix.sourceforge.io>`_ (Hierarchical Equal Area
isoLatitude Pixelisation) divides the sphere into :math:`N_{\rm pix} = 12\,{\rm NSIDE}^2`
equal-area pixels.  Common NSIDE values and their properties:

.. list-table::
   :header-rows: 1
   :widths: 15 25 30 30

   * - NSIDE
     - :math:`N_{\rm pix}`
     - Pixel area (deg²)
     - Angular resolution (arcmin)
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

For LS10 BGS analyses NSIDE = 64–128 is used.

**Workflow** to go from catalogs to overdensity and template arrays:

1. :func:`~sys_mapping.maps.pixelize_catalog` — bin RA/Dec positions into
   a HEALPix count map.
2. :func:`~sys_mapping.maps.compute_overdensity` — form :math:`\hat\delta_g(p)
   = n_g(p)/(f_r n_r(p)) - 1` and return the boolean unmasked-pixel array.
3. :func:`~sys_mapping.maps.assign_template_values` — extract template values
   at unmasked pixels, returning a ``(n_sys, n_good_pix)`` array.

**Technical implementation.**

The three-step catalog-to-overdensity pipeline executes as follows:

*Step 1 — pixelise* (:func:`~sys_mapping.maps.pixelize_catalog`):

.. code-block:: python

   pix_ids = hp.ang2pix(nside, ra, dec, lonlat=True)   # RA/Dec → HEALPix ring index
   count_map = np.bincount(pix_ids, minlength=12*nside**2).astype(float)

``hp.ang2pix`` with ``lonlat=True`` accepts RA/Dec in degrees.  The result
is a ``(12×NSIDE²,)`` integer count map covering the full sky (zeros outside
the footprint).  The bincount is :math:`O(N_{\rm gal})`.

*Step 2 — overdensity* (:func:`~sys_mapping.maps.compute_overdensity`):

.. code-block:: python

   f_r          = n_gal.sum() / n_rand.sum()    # galaxy-to-random normalisation
   good_pixels  = (n_rand > 0) & mask           # boolean, (n_pix,)
   delta_g_full = np.zeros(n_pix)
   delta_g_full[good_pixels] = (
       n_gal[good_pixels] / (f_r * n_rand[good_pixels]) - 1.0)

The normalisation factor :math:`f_r = N_g / N_r` accounts for the different
total sizes of the galaxy and random catalogs.  Pixels with zero random
count are masked to avoid division by zero.

*Step 3 — template extraction* (:func:`~sys_mapping.maps.assign_template_values`):

.. code-block:: python

   delta_t = template_full_sky[:, good_pixels]   # (n_sys, n_good_pix)
   # Centre and normalise each template to zero mean, unit variance
   delta_t -= delta_t.mean(axis=1, keepdims=True)
   delta_t /= delta_t.std(axis=1, keepdims=True)

The centring and normalisation ensure that template amplitudes
:math:`a_i, b_i` are dimensionless and comparable across templates of
different physical origins.

**Functions:**
:func:`~sys_mapping.maps.pixelize_catalog`,
:func:`~sys_mapping.maps.compute_overdensity`,
:func:`~sys_mapping.maps.assign_template_values`,
:func:`~sys_mapping.maps.generate_systematic_map`,
:func:`~sys_mapping.maps.generate_systematic_maps`,
:func:`~sys_mapping.maps.systematic_power_spectrum`

----

Two-point function measurement
--------------------------------

**Module:** :mod:`sys_mapping.utils`

The angular two-point correlation function :math:`w(\theta)` is measured
using the **Landy-Szalay estimator**:

.. math::

   w(\theta) = \frac{DD(\theta) - 2\,DR(\theta) + RR(\theta)}{RR(\theta)}

where :math:`DD(\theta)`, :math:`DR(\theta)`, :math:`RR(\theta)` are the
normalised pair counts in angular separation bin :math:`\theta` for
galaxy-galaxy, galaxy-random, and random-random pairs respectively.
The estimator is unbiased for large random catalogs and has minimum
variance for a Poisson-distributed galaxy field
(Landy & Szalay 1993, ApJ 412, 64).

``sys_mapping`` provides wrappers for `TreeCorr
<https://rmjarvis.github.io/TreeCorr>`_ (which uses a ball-tree for
efficient pair counting, :math:`\mathcal{O}(N\log N)`) and for `Corrfunc
<https://corrfunc.readthedocs.io>`_ (highly optimised C implementation for
large catalogs).

**Technical implementation.**

*TreeCorr backend* (:func:`~sys_mapping.utils.measure_two_point_function`):
wraps ``treecorr.NNCorrelation`` with a ball-tree pair counter.  The angular
bins are specified in log-space (``min_sep``, ``max_sep`` in degrees,
``nbins`` bins), and the catalog is passed once for DD and once for DR:

.. code-block:: python

   cat_g = treecorr.Catalog(ra=ra_g, dec=dec_g, ra_units="deg", dec_units="deg")
   cat_r = treecorr.Catalog(ra=ra_r, dec=dec_r, ra_units="deg", dec_units="deg")
   nn = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep,
                                nbins=nbins, sep_units="deg")
   nn.process(cat_g)         # DD counts
   nr = nn.copy(); nr.process(cat_g, cat_r)   # DR counts
   rr = nn.copy(); rr.process(cat_r)          # RR counts
   xi, varxi = nn.calculateXi(rr=rr, dr=nr)  # Landy–Szalay

TreeCorr's ball-tree algorithm has :math:`O(N \log N)` pair-count
complexity; it applies the Landy–Szalay formula internally.

*Corrfunc backend* (:func:`~sys_mapping.utils.measure_two_point_function_corrfunc`):
calls ``Corrfunc.mocks.DDtheta_mocks`` directly on RA/Dec arrays.  Corrfunc
is a highly optimised C library that vectorises the inner pair loop with AVX2
SIMD instructions and outperforms TreeCorr by 2–5× for
:math:`N \gtrsim 10^5`:

.. code-block:: python

   DD = Corrfunc.mocks.DDtheta_mocks(autocorr=True, nthreads=n_threads,
           binfile=theta_bins, RA=ra_g, DEC=dec_g)
   DR = Corrfunc.mocks.DDtheta_mocks(autocorr=False, nthreads=n_threads,
           binfile=theta_bins, RA1=ra_g, DEC1=dec_g, RA2=ra_r, DEC2=dec_r)
   RR = Corrfunc.mocks.DDtheta_mocks(autocorr=True, nthreads=n_threads,
           binfile=theta_bins, RA=ra_r, DEC=dec_r)
   w_theta = (DD - 2*DR + RR) / RR   # Landy–Szalay (normalised pair counts)

Both backends return the same ``(n_bins,)`` array ``w_theta`` and are
interchangeable.  The Corrfunc backend should be preferred for production
runs with :math:`N > 10^6` galaxies.

**Functions:**
:func:`~sys_mapping.utils.measure_two_point_function`,
:func:`~sys_mapping.utils.measure_two_point_function_corrfunc`,
:func:`~sys_mapping.utils.measure_kk_correlation_treecorr`,
:func:`~sys_mapping.utils.measure_kk_correlation_corrfunc`,
:func:`~sys_mapping.utils.compute_covariance_matrix`,
:func:`~sys_mapping.utils.compute_amplitude_bias`

----

Method comparison table
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 28 14 28

   * - Method
     - Reference(s)
     - Status
     - Module
   * - Additive contamination model (MCMC)
     - Berlfein+2024
     - Implemented
     - ``likelihood``, ``inference``
   * - Multiplicative contamination model (MCMC)
     - Berlfein+2024
     - Implemented
     - ``likelihood``, ``inference``
   * - Combined contamination model (MCMC)
     - Berlfein+2024
     - Implemented
     - ``likelihood``, ``inference``
   * - PCA template rotation
     - Berlfein+2024
     - Implemented
     - ``correction``
   * - Noise debiasing
     - Berlfein+2024
     - Implemented
     - ``correction``
   * - Two-point function correction
     - Berlfein+2024
     - Implemented
     - ``contamination``, ``correction``
   * - Likelihood ratio model selection
     - Berlfein+2024
     - Implemented
     - ``model_selection``
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
   * - Harmonic template subtraction
     - Elsner+2016
     - Implemented
     - ``power_spectrum``
   * - Harmonic bias correction
     - Elsner+2016
     - Implemented
     - ``power_spectrum``
   * - Basic / Extended mode projection (BMP/EMP)
     - Elsner+2016, Leistedt+2014
     - Implemented
     - ``power_spectrum``
   * - Harmonic power spectrum correction pipeline
     - Elsner+2016, Elsner+2017
     - Implemented
     - ``correction``
   * - ElasticNet regression
     - Weaverdyck+2021
     - Implemented
     - ``regression``
   * - Iterative Systematics Decontamination (ISD)
     - Rodríguez-Monroy+2025
     - Implemented
     - ``regression``
   * - Multi-method comparison framework
     - Weaverdyck+2021
     - Implemented
     - ``regression``
   * - SNR-based template ranking
     - Tanidis+2026
     - Implemented
     - ``diagnostics``
   * - Footprint masking diagnostics
     - Rodríguez-Monroy+2025
     - Implemented
     - ``diagnostics``
   * - NaMaster coupling matrix (exact)
     - Alonso+2019, Elsner+2017
     - Optional backend (planned)
     - ``power_spectrum``
   * - Catalogue-based PCL deprojection
     - Cornish+2026
     - Deferred
     - ``deprojection`` (future)
   * - Transfer function calibration
     - Cornish+2026
     - Deferred
     - ``deprojection`` (future)
   * - ANN / deep-learning weights
     - Rezaie+2020
     - Not planned
     - External pre-processing
   * - Optimal quadratic estimator (QMV)
     - Ho+2012
     - Not planned
     - External (NaMaster)
   * - Shear E/B quadratic estimator
     - Tanidis+2026
     - Not planned
     - Weak-lensing specific
