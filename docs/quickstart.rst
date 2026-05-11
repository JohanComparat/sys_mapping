Quickstart
==========

This page walks through the complete ``sys_mapping`` inference pipeline on a
synthetic lognormal mock.  All six decontamination methods are demonstrated so
that you can see their outputs side by side.  The setup (templates, mock field,
overdensity) is identical to the three-template scenario of
`Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_.
For notation conventions see :doc:`overview`; for the mathematics behind each
step see :doc:`methods`.

----

Methods overview
-----------------

``sys_mapping`` implements six decontamination methods through a unified
``run_decontamination()`` interface.  They differ in the contamination model
they assume and in how they estimate the amplitudes.

.. list-table::
   :header-rows: 1
   :widths: 14 30 24 16 16

   * - Method
     - Reference
     - Model
     - Parameters
     - Typical runtime
   * - **OLS**
     - `Ross et al. 2011 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.417.1350R/abstract>`_
     - additive
     - :math:`a_i` only
     - < 1 s
   * - **ElasticNet**
     - `Weaverdyck & Huterer 2021 <https://doi.org/10.1093/mnras/stab818>`_
     - additive
     - :math:`a_i` only (L1+L2 penalised)
     - 1–5 s (5-fold CV)
   * - **ISD-1**
     - `Rodríguez-Monroy et al. 2025 <https://arxiv.org/abs/2509.07943>`_
     - additive (poly-1)
     - :math:`a_i` only, iterative
     - 1–10 s
   * - **ISD-3**
     - `Rodríguez-Monroy et al. 2025 <https://arxiv.org/abs/2509.07943>`_
     - additive (poly-3)
     - :math:`a_i` only, iterative
     - 2–30 s
   * - **MCMC-add**
     - `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_
     - additive
     - :math:`a_i`, :math:`\sigma`
     - 2–20 min
   * - **MCMC-comb**
     - `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_
     - combined (additive + multiplicative)
     - :math:`a_i`, :math:`b_i`, :math:`\sigma`
     - 5–60 min

**Weight convention.**  All methods produce a per-pixel weight
:math:`w(p)` that corrects for additive systematics:

.. math::

   w(p) = \frac{1}{1 + \sum_i \hat a_i\,t_i(p)}.

For ``MCMC-comb`` an additional multiplicative weight
:math:`w^{\rm mult}(p) = 1/(1 + \sum_i \hat b_i\,t_i(p))` is also available.

----

The pipeline has eight stages:

1. Generate HEALPix template maps.
2. Simulate a lognormal galaxy field and inject known contamination.
3. Measure the observed overdensity per pixel.
4. Rotate templates into a PCA basis to remove degeneracies.
5. Run all six decontamination methods.
6. Extract MAP estimates and diagnostics from the MCMC chains (MCMC methods only).
7. Perform a likelihood ratio test to decide whether multiplicative contamination is significant (MCMC-comb only).
8. Correct the two-point function for residual systematics.

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   import healpy as hp
   import sys_mapping as sm
   from sys_mapping.contamination import apply_contamination, pack_params
   from sys_mapping.correction import (
       rotate_templates, transform_params_from_rotated, correct_two_point_function
   )
   from sys_mapping.regression import run_decontamination

   rng = np.random.default_rng(0)

----

Step 1 — Generate systematic template maps
-------------------------------------------

Systematic template maps :math:`t_i(p)` are HEALPix maps (one scalar per
pixel) that trace known observational artefacts — seeing, Galactic extinction,
stellar density, sky brightness, survey depth, and so on.  Each is normalised
to zero mean and unit variance over the survey footprint so that the inferred
amplitudes :math:`a_i, b_i` are dimensionless and directly comparable across
templates (see :doc:`methods`, section "Systematic template maps").

``generate_systematic_maps`` returns an array of shape
:math:`(n_s, N_{\rm pix})` where :math:`N_{\rm pix} = 12 \times \texttt{NSIDE}^2`.
We use :math:`\texttt{NSIDE} = 64` (:math:`N_{\rm pix} = 49152` pixels,
mean pixel area :math:`\approx 0.84\,\text{deg}^2`) for speed;
the `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_ paper uses
:math:`\texttt{NSIDE} = 512`.  Three templates (:math:`n_s = 3`) are enough to
illustrate the combined additive-plus-multiplicative inference.

.. code-block:: python

   # ── 1. Generate template maps ──────────────────────────────────────────────
   NSIDE = 64
   templates = sm.generate_systematic_maps(NSIDE, families=[0, 1, 2], seed=0)
   N_PIX = hp.nside2npix(NSIDE)
   # templates.shape == (3, 49152)

----

Step 2 — Simulate the galaxy field and inject contamination
-----------------------------------------------------------

**True galaxy overdensity.**  We draw a Gaussian random field :math:`G(p)`
with power spectrum :math:`C_\ell^G \propto (\ell+1)^{-2}`, normalised so that
the map variance equals :math:`\sigma^2 = 0.25`.  The lognormal transformation

.. math::

   \delta_g^{\rm true}(p) = \exp\!\left(G(p) - \tfrac12\sigma^2\right) - 1

(`Coles & Jones 1991 <https://ui.adsabs.harvard.edu/abs/1991MNRAS.248....1C/abstract>`_) produces a non-negative density field with realistic
one-point statistics and positive correlation function.

**Contamination model.**  The observed overdensity is (see :doc:`methods`,
"Contamination model"):

.. math::

   \hat\delta_g(p) = \delta_g^{\rm true}(p)
                     \!\left(1 + \sum_{i=1}^{n_s} b_i\,t_i(p)\right)
                   + \sum_{i=1}^{n_s} a_i\,t_i(p).

We choose true amplitudes :math:`\mathbf{a}^{\rm true} = (0.05, -0.03, 0.08)`
and :math:`\mathbf{b}^{\rm true} = (0.04, 0.00, -0.06)`.  Template 2 has
:math:`b_2 = 0` (purely additive), while templates 1 and 3 have non-zero
multiplicative amplitudes — so the full ``combined`` model is required to fit
them correctly.

**Poisson sampling.**  The expected count per pixel is
:math:`\lambda(p) = \bar{n}_g \left(1 + \hat\delta_g(p)\right)` with
:math:`\bar{n}_g = 40` galaxies per pixel.  The ``random_counts`` array
simulates a random catalog ten times denser than the galaxy catalog, giving a
:math:`\sqrt{N_r/N_g} \approx 0.32` shot-noise reduction in the Landy-Szalay
estimator.

.. code-block:: python

   # ── 2. Simulate a lognormal galaxy field ──────────────────────────────────
   # True contamination parameters
   a_true = np.array([0.05, -0.03, 0.08])
   b_true = np.array([0.04,  0.00, -0.06])

   # Lognormal overdensity: delta = exp(G - 0.5*sigma_G^2) - 1
   np.random.seed(rng.integers(0, 2**31))
   lmax = 3 * NSIDE - 1
   ell = np.arange(lmax + 1, dtype=float)
   cl_G = (ell + 1.0) ** (-2)
   cl_G[0] = 0.0
   cl_G *= 0.5**2 / np.sum((2*ell+1)/(4*np.pi) * cl_G)   # normalise to sigma=0.5
   G = hp.synfast(cl_G, nside=NSIDE, lmax=lmax)
   delta_g_true = np.exp(G - 0.5 * 0.5**2) - 1.0

   # Apply contamination (combined model, Eq. 13 of Berlfein+2024)
   delta_g_obs = np.asarray(
       apply_contamination(
           jnp.asarray(delta_g_true),
           jnp.asarray(templates),
           jnp.asarray(a_true),
           jnp.asarray(b_true),
       )
   )

   # Poisson-sample galaxy and random counts
   N_MEAN = 40                                     # mean galaxies per pixel
   lam = np.maximum(N_MEAN * (1 + delta_g_obs), 0.0)
   galaxy_counts = rng.poisson(lam).astype(float)
   random_counts = np.full(N_PIX, float(N_MEAN * 10))   # 10× denser random catalog

----

Step 3 — Compute the observed overdensity
------------------------------------------

``compute_overdensity`` turns raw pixel counts into the observed overdensity
field.  At the pixel level the Landy-Szalay estimate simplifies to:

.. math::

   \hat\delta_g(p) = \frac{n_g(p)/\bar{n}_r}{n_r(p)/\bar{n}_g} - 1,

where :math:`n_g(p)` and :math:`n_r(p)` are the galaxy and random counts in
pixel :math:`p` (see :doc:`overview`, section "The observed galaxy overdensity").
Pixels with zero random counts are excluded; ``good_pix`` is a boolean mask of
the :math:`N_{\rm good}` pixels that pass this cut.

``assign_template_values`` samples each template map at the good pixels,
returning an array of shape :math:`(N_{\rm good}, n_s)`.  This is the design
matrix that enters the likelihood.

.. code-block:: python

   # ── 3. Compute overdensity and template values ────────────────────────────
   delta_g_obs_pix, good_pix = sm.compute_overdensity(galaxy_counts, random_counts)
   # delta_g_obs_pix.shape == (N_good,)   good_pix.shape == (N_PIX,), dtype bool
   delta_t = sm.assign_template_values(templates, good_pix)
   # delta_t.shape == (N_good, 3)   — rows = pixels, columns = templates

----

Step 4 — PCA rotation of templates
------------------------------------

When templates are correlated (e.g. Galactic extinction and stellar density both
peak toward the plane), the joint likelihood has a degenerate ridge and the MCMC
chain mixes poorly.  PCA rotation diagonalises the template covariance matrix
:math:`C = V D V^\top` and replaces the original templates by the rotated ones

.. math::

   \tilde{t}_j(p) = \sum_i V_{ij}\,t_i(p),

which are uncorrelated by construction (see :doc:`methods`, section "PCA
template rotation").  The regression-based methods (OLS, ElasticNet, ISD)
are rotation-invariant; ``run_decontamination()`` handles the PCA rotation
internally for the MCMC methods.  Running it here serves as a diagnostic:
a small eigenvalue signals a near-degenerate template pair that is essentially
unconstrained by the data.

.. code-block:: python

   # ── 4. PCA rotation (diagnostic; MCMC methods rotate internally) ─────────
   delta_t_rot, R, eigenvalues = rotate_templates(delta_t)
   print('Template eigenvalues:', eigenvalues.round(3))
   # Example output: [1.412  0.988  0.600]

----

Step 5 — Run all decontamination methods
-----------------------------------------

The unified ``run_decontamination()`` entry point accepts the overdensity
vector and the template matrix in their original (non-rotated) basis and
dispatches to the appropriate method.  For the MCMC methods the PCA rotation
is performed internally; for regression methods (OLS, ElasticNet, ISD) it is
not needed.

.. note::

   ``delta_t`` from Step 3 has shape ``(N_good, n_sys)`` — rows are pixels.
   ``run_decontamination()`` expects shape ``(n_sys, N_good)`` — rows are
   templates — so we pass ``delta_t.T``.

.. code-block:: python

   # ── 5. Run all six decontamination methods ────────────────────────────────
   METHODS = ["OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"]
   results = {}
   for method in METHODS:
       results[method] = run_decontamination(
           method,
           delta_g_obs_pix,
           delta_t.T,           # shape (n_sys, N_good)
           n_walkers=150,
           n_steps=500,
           n_burn=100,
           seed=42,
       )
       sig = results[method]["sigma_hat"]
       sig_str = f"  σ = {sig:.3f}" if sig is not None else ""
       print(
           f"{method:12s}  a_hat = {results[method]['a_hat'].round(3)}"
           f"{sig_str}  elapsed = {results[method]['elapsed_s']:.1f}s"
       )

Expected console output:

.. code-block:: text

   OLS           a_hat = [ 0.051 -0.029  0.081]             elapsed = 0.0s
   ElasticNet    a_hat = [ 0.050 -0.028  0.079]             elapsed = 2.1s
   ISD-1         a_hat = [ 0.051 -0.029  0.081]             elapsed = 0.3s
   ISD-3         a_hat = [ 0.050 -0.028  0.079]             elapsed = 1.2s
   MCMC-add      a_hat = [ 0.049 -0.030  0.080]  σ = 0.243  elapsed = 187.4s
   MCMC-comb     a_hat = [ 0.049 -0.031  0.079]  σ = 0.261  elapsed = 312.8s

Comparing parameter recovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All six methods should recover :math:`\mathbf{a}^{\rm true} = (0.05,\,-0.03,\,0.08)`
to within :math:`\sim 1\sigma`.  Only ``MCMC-comb`` estimates the multiplicative
amplitudes :math:`b_i`; the four regression methods and ``MCMC-add`` set
:math:`\hat b_i = 0` by construction.  The noise parameter :math:`\sigma`
(pixel-level overdensity scatter, :math:`\sigma_G^2 = 0.25` from Step 2) is
inferred only by the two MCMC methods.

.. code-block:: text

   Method        a₁      a₂      a₃     b₁     b₂     b₃       σ
   ──────────────────────────────────────────────────────────────────
   True       +0.050  −0.030  +0.080  +0.040  0.000  −0.060   0.25
   OLS        +0.051  −0.029  +0.081     —      —       —        —
   ElasticNet +0.050  −0.028  +0.079     —      —       —        —
   ISD-1      +0.051  −0.029  +0.081     —      —       —        —
   ISD-3      +0.050  −0.028  +0.079     —      —       —        —
   MCMC-add   +0.049  −0.030  +0.080     —      —       —      0.243
   MCMC-comb  +0.049  −0.031  +0.079  +0.038  +0.001  −0.059  0.261

The ``—`` entries are zero by construction (additive-only models do not fit
multiplicative amplitudes).  The ``MCMC-comb`` recovery of :math:`b_2 \approx 0`
correctly identifies that template 2 has no multiplicative contamination.
Both MCMC methods recover :math:`\sigma` close to the injected value of 0.25.

----

Step 6 — MCMC diagnostics: extract MAP estimates
--------------------------------------------------

.. note::

   Steps 6 and 7 are specific to the MCMC methods.  If you only need systematic
   weights, the ``weights`` key in the results dict from Step 5 is sufficient
   for all methods.

The maximum *a posteriori* (MAP) estimate is the marginal posterior median
(robust to asymmetric tails).  ``run_decontamination()`` back-rotates the
parameters to the original template basis automatically and stores the full
covariance matrix.

.. code-block:: python

   # ── 6. Extract MAP estimates from MCMC-comb ───────────────────────────────
   res_comb = results["MCMC-comb"]

   a_hat   = res_comb["a_hat"]          # (n_sys,) — original basis
   b_hat   = res_comb["b_hat"]          # (n_sys,) — original basis
   var_a   = np.diag(res_comb["cov_a"]) # per-parameter posterior variance
   var_b   = np.diag(res_comb["cov_b"])

   print('True a:', a_true, '  Recovered:', a_hat.round(3))
   print('True b:', b_true, '  Recovered:', b_hat.round(3))
   # Expected (approximate):
   #   True a: [ 0.05 -0.03  0.08]   Recovered: [ 0.049 -0.031  0.079]
   #   True b: [ 0.04  0.00 -0.06]   Recovered: [ 0.038  0.001 -0.059]

**Noise debiasing.**  The squared MAP estimates are biased upward by the
posterior variance (see :doc:`methods`, section "Noise debiasing"):

.. math::

   \mathbb{E}[\hat a_i^2] = a_i^2 + \mathrm{Var}[\hat a_i].

The debiased amplitudes :math:`\tilde a_i^2 = \max(\hat a_i^2 - \mathrm{Var}[\hat a_i],\; 0)`
are used in the two-point correction (Step 8).

----

Step 7 — Likelihood ratio test
--------------------------------

The likelihood ratio test (LRT) decides whether the ``combined`` model
(free :math:`a_i` and :math:`b_i`) is significantly better than the
``additive`` model (:math:`b_i = 0` for all :math:`i`).  The test statistic
is (see :doc:`methods`, section "Likelihood ratio test"):

.. math::

   \lambda_{\rm LR} = 2\!\left[\ln\mathcal{L}(\hat\Theta_1) - \ln\mathcal{L}(\hat\Theta_0)\right]
   \;\overset{H_0}{\sim}\; \chi^2(r),

where :math:`r = 8 - 5 = 3` is the number of extra free parameters
(:math:`b_1, b_2, b_3`).  The :math:`p = 0.05` critical value is
:math:`\chi^2_{0.05}(3) = 7.81`.  Because we injected non-zero
:math:`b_1, b_3`, we expect :math:`\lambda_{\rm LR} \gg 7.81` and ``reject_null = True``.

The LRT compares chains in the *rotated* basis; we retrieve the PCA-rotated
templates from Step 4.

.. code-block:: python

   # ── 7. Likelihood ratio test ─────────────────────────────────────────────
   theta_add  = sm.get_mle_params(results["MCMC-add"]["flat_chain"])
   theta_comb = sm.get_mle_params(results["MCMC-comb"]["flat_chain"])

   result = sm.likelihood_ratio_test(
       delta_g_obs_pix, delta_t_rot, theta_add, theta_comb,
       null_model='additive', alt_model='combined',
   )
   print(f'λ_LR = {result.lambda_lr:.2f},  p = {result.p_value:.4f},  '
         f'reject additive null: {result.reject_null}')
   # Expected (approximate):
   #   λ_LR = 48.3,  p = 0.0000,  reject additive null: True

----

Step 8 — Two-point function correction
----------------------------------------

The two-point correlation function :math:`w(\theta)` measured from the
contaminated catalog is biased.  After inferring the noise-debiased amplitudes
:math:`\tilde a_i^2` and :math:`\tilde b_i^2`, the first-order corrected
two-point function is (see :doc:`methods`, section "Two-point function correction"):

.. math::

   w_{\rm corr}(\theta) =
   \frac{\hat w(\theta) - \sum_{i=1}^{n_s} \tilde a_i^2\,w_{t_i}(\theta)}
        {1 + \sum_{i=1}^{n_s} \tilde b_i^2\,w_{t_i}(\theta)},

where :math:`w_{t_i}(\theta)` is the auto-correlation of template :math:`i`.
The numerator removes the additive bias from spurious template–template
clustering; the denominator corrects the multiplicative suppression of the true
signal.

In the toy example below, :math:`\hat w(\theta)` is a power-law stand-in; in
real analyses this is the Landy-Szalay estimator measured from galaxy and random
catalogs.  The fractional correction at small scales is typically 10–30% for
the amplitudes used here.

.. code-block:: python

   # ── 8. Two-point correction ──────────────────────────────────────────────
   lmax = 3 * NSIDE - 1
   n_theta = 12
   theta_deg = np.logspace(-1, 1, n_theta)
   cos_t = np.cos(np.radians(theta_deg))

   # Legendre-sum two-point function from a power spectrum C_ell
   def w_from_cl(cl, cos_vals):
       ell = np.arange(len(cl), dtype=float)
       p0, p1 = np.ones(len(cos_vals)), cos_vals.copy()
       w = cl[0] / (4*np.pi) * p0
       if len(cl) > 1:
           w += cl[1] * 3 / (4*np.pi) * p1
       for l in range(2, len(cl)):
           p2 = ((2*l-1)*cos_vals*p1 - (l-1)*p0) / l
           w += cl[l] * (2*l+1) / (4*np.pi) * p2
           p0, p1 = p1, p2
       return w

   # Template auto-correlations w_{t_i}(theta)
   tcorr = np.zeros((3, n_theta))
   for i in range(3):
       cl_i = hp.anafast(templates[i], lmax=lmax)
       tcorr[i] = w_from_cl(cl_i, cos_t)

   # Toy observed w(theta) — replace with your Landy-Szalay estimate
   w_obs = 0.02 * (theta_deg / 1.0) ** (-0.7)

   w_corr = correct_two_point_function(w_obs, a_hat, b_hat, var_a, var_b, tcorr)
   print(f'Fractional correction at smallest scale: '
         f'{(w_corr[0] - w_obs[0]) / w_obs[0] * 100:.1f}%')
   # Expected (approximate):  Fractional correction at smallest scale: -18.4%

----

Expected output summary
------------------------

Running the full script above should produce output close to:

.. code-block:: text

   Template eigenvalues: [1.412  0.988  0.600]
   OLS           a_hat = [ 0.051 -0.029  0.081]             elapsed = 0.0s
   ElasticNet    a_hat = [ 0.050 -0.028  0.079]             elapsed = 2.1s
   ISD-1         a_hat = [ 0.051 -0.029  0.081]             elapsed = 0.3s
   ISD-3         a_hat = [ 0.050 -0.028  0.079]             elapsed = 1.2s
   MCMC-add      a_hat = [ 0.049 -0.030  0.080]  σ = 0.243  elapsed = 187.4s
   MCMC-comb     a_hat = [ 0.049 -0.031  0.079]  σ = 0.261  elapsed = 312.8s
   True a: [ 0.05 -0.03  0.08]   Recovered: [ 0.049 -0.031  0.079]
   True b: [ 0.04  0.00 -0.06]   Recovered: [ 0.038  0.001 -0.059]
   True σ: 0.25                  Recovered (MCMC-comb): 0.261
   λ_LR = 48.3,  p = 0.0000,  reject additive null: True
   Fractional correction at smallest scale: -18.4%

The recovered amplitudes should lie within :math:`\sim 1\sigma` of the true
values for most random seeds.  The LRT statistic will vary across realisations
but should remain well above 7.81 (the :math:`p = 0.05` threshold) whenever
the injected :math:`|b_i| \gtrsim 0.03`.  The noise parameter :math:`\sigma`
should recover close to the injected value of 0.25 across realisations.

----

Results on 100 mock realisations
----------------------------------

The figures below summarise the pipeline output over 100 independent lognormal
mock realisations (same parameters, different random seeds) as produced by
``scripts/run_mock_analysis.py --synthetic --n-mocks 100 --nside 64 --n-sys 3``.

.. raw:: html

   <figure style="text-align:center;margin:1.5em 0;">
     <a href="_static/results_mock_analysis/mock_parameter_recovery_all_methods.png" target="_blank">
       <img src="_static/results_mock_analysis/mock_parameter_recovery_all_methods.png"
            style="width:98%;max-width:980px;">
     </a>
     <figcaption style="font-size:0.9em;color:#555;margin-top:0.5em;">
       <strong>Additive parameter recovery across all six methods (100 realisations).</strong>
       Each panel shows one method.  Scatter points are individual mock
       realisations (one per template); box-plots show the inter-quartile
       range.  The horizontal dashed line marks zero bias.  All methods
       recover <em>a</em><sub>i</sub> with negligible median bias; the
       MAD (median absolute deviation) reported in each title quantifies
       the scatter.
     </figcaption>
   </figure>

.. raw:: html

   <figure style="text-align:center;margin:1.5em 0;">
     <a href="_static/results_mock_analysis/mock_sigma_recovery.png" target="_blank">
       <img src="_static/results_mock_analysis/mock_sigma_recovery.png"
            style="width:70%;max-width:700px;">
     </a>
     <figcaption style="font-size:0.9em;color:#555;margin-top:0.5em;">
       <strong>Intrinsic noise parameter &sigma; recovery (MCMC methods only).</strong>
       Histograms of the recovered &sigma; across 100 realisations for
       MCMC-add (left) and MCMC-comb (right).  The vertical dashed line
       marks the injected value &sigma;<sub>G</sub><sup>2</sup>&nbsp;=&nbsp;0.25.
       Both methods recover the intrinsic field variance accurately with
       low scatter.
     </figcaption>
   </figure>

.. raw:: html

   <figure style="text-align:center;margin:1.5em 0;">
     <a href="_static/results_mock_analysis/mock_lrt_statistics.png" target="_blank">
       <img src="_static/results_mock_analysis/mock_lrt_statistics.png"
            style="width:60%;max-width:600px;">
     </a>
     <figcaption style="font-size:0.9em;color:#555;margin-top:0.5em;">
       <strong>LRT statistic distribution.</strong>
       :math:`\lambda_{\rm LR}` across 100 realisations.  The vertical dashed
       line marks the :math:`\chi^2(3)` critical value at :math:`p = 0.05`
       (:math:`\lambda = 7.81`).  All realisations correctly reject the
       additive null in favour of the combined model.
     </figcaption>
   </figure>

