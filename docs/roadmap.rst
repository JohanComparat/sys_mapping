Roadmap
=======

Prioritised next steps.  Every entry corresponds to an open finding in the
``sys_mapping_paper`` document, whose §9.3 table gives the same list ordered by cost
to the science; this page adds the implementation detail.  Priorities reflect impact
on *scientific output*, not implementation effort.

**What a next version needs**, in order:

1. Finish re-issuing the LS10 products on the footprint-standardised basis (P0
   below).  The code is in place and one cell is verified; the remaining 35 decide
   whether any of them still overshoots.
2. Give the per-template significances a covariance that holds on a correlated field
   (P1 below).  The iid likelihood has a 3-sigma false-positive rate of 76--96 % on
   clean simulations, so a significance quoted from it is not a detection.

.. contents:: On this page
   :local:
   :depth: 1

----

P0 — correctness of the published products
------------------------------------------

**Standardise the template basis on the footprint.**  *(implemented, re-run running)*
``load_templates_from_dir`` normalises each survey-property map over that map's own
valid region.  The analysis footprint is a subset of it, and the basis is not
standardised there: on the LS10 footprint at NSIDE 64 the per-template rms spans
0.108 to 1.015 and the relative mean reaches 0.496.  Fitted amplitudes are then not
in units of one template standard deviation, ``compute_covariance_matrix`` is not a
covariance, and the template auto-correlations the two-point correction subtracts
carry the same scaling.

:func:`~sys_mapping.maps.standardise_on_footprint` is applied after masking by both
production scripts, which record the means and rms they divided out in
``params.json`` and stamp ``TPLBASIS`` into the FITS header.  Measured on the
fiducial cell, the linear methods' ``rms|a_hat|`` drops from 0.138 to 0.018 while ISD
moves from 0.0049 to 0.0040, since its marginal fit bins by template value and
equal-occupancy bins are invariant under a monotone rescaling.  The likelihood ratio
is unchanged, 179 against 183.

*Remaining:* the 36-cell re-issue at ``WEIGHTVER = 3``.

**Measure the template two-point functions from the galaxies.**  *(implemented)*
The template correlations were measured with TreeCorr at the HEALPix pixel centres.
No pair of distinct pixels is separated by less than the pixel scale, so they came
back as exactly zero below it while the galaxy correlation is measured from the
catalogue down to 0.5 arcmin, and the correction did nothing over 21 of 30 bins at
NSIDE 64 and 24 of 30 at NSIDE 32.  A template is constant within a pixel, so its
correct correlation at sub-pixel separations is its variance, not zero.

Each galaxy now carries the rotated-template value of its pixel and the correlation
runs on the galaxy positions.  On the fiducial cell the correction acts in all 30
bins, and the corrected function stays between 0.97 and 0.99 of the observed one for
every method, where ``ISD-3`` previously reached -4.8.

**Guard the corrected** :math:`w(\theta)`.  *(done)*
:func:`~sys_mapping.correction.correct_two_point_function` warns where
:math:`\sum_i \tilde a_i^2 \xi_i(\theta)` exceeds :math:`\hat w(\theta)` and leaves a
negative correlation function.  Nothing bounds the subtracted term by the
measurement, and the failure was silent.

**Give** ``lambda_LR`` **a real MLE.**  *(done)*
:func:`~sys_mapping.inference.refine_to_mle` maximises the log-likelihood by L-BFGS-B
with analytic gradients, from the posterior median and from the ordinary
least-squares solution, keeping whichever start reaches the higher likelihood.  On
the nine NSIDE-32 LS10 cells the negative null draws went from 21 to 0, every null
minimum is positive, and the data statistic rose in all ten refinements.  Refining
raises the null draws too, so the nulls widen: one NSIDE-64 cell moved from a null
maximum of 65.0 to 81.3 and :math:`p` from 0.020 to 0.039.
``get_mle_params`` is deprecated in favour of ``posterior_median_params``.

**Fix the EMP mode count.**  *(done)*
``mode_projection_bias`` passed ``n_projected`` --- a count of *multipoles* --- into
``harmonic_bias``, whose first argument is a count of *templates*.  It now passes
``n_templates``; ``project_mask`` still selects which multipoles are corrected.  The
harmonic template subtraction also now removes the debiased *square*
:math:`\tilde a_i^2 C_\ell^{t_i}`, so it is the Hankel transform of the
configuration-space estimator.

**Re-run the LS10 ISD columns.**  *(superseded by the basis re-run)*
The products carry the library's own per-pixel weight, taken from
``result["weights"]``, and record the convention in ``WEIGHTVER``, ``WEIGHTCON``,
``WMAXCLIP`` and ``TPLBASIS``.  ``WEIGHT_ISD3`` is non-trivial in every cell and
``WEIGHT_ISD1`` in all but the two densest NSIDE-32 samples, where the stopping
rule accepts no template.  All four resolutions are being re-issued at
``WEIGHTVER = 3`` together with the footprint-standardised basis.

**Unify the weight definition.**  *(done)*
Both production scripts read ``result["weights"]`` rather than recomputing
:math:`1/(1 + \hat a \cdot t)` from the amplitudes, which reproduces neither ISD's
cumulative product nor ``MCMC-comb``'s exact inverse.  One denominator floor,
:math:`1/20`, everywhere.  A reader warns on ``WEIGHTVER = 1``.

**Align the likelihood between the two scripts.**  *(done)*
Both take the same ``--skewed`` flag, defaulting off.  It is opt-in because enabling
the skew-normal also moves the additive model off its exact analytic posterior onto
NUTS.

**Diagnose the Uchuu simulation result.**
``nside0064/summary_table.csv`` shows 37 of 45 Uchuu method-cells with an
improvement factor below 1 — the correction makes the :math:`w(\theta)` bias worse.
Two candidate explanations:

#. the injected amplitude is below the :math:`w(\theta)` detection threshold, so the
   variance of :math:`\hat a` exceeds the bias removed (the detectability law
   predicts exactly this, since :math:`w(\theta)` contamination grows as
   :math:`A^2`); or
#. the ``multiplicative`` *fit* model did not match its injector: it set
   :math:`b = a` while retaining :math:`a`, where the injector uses the pure
   form :math:`a = 0` with :math:`b` free.  The fit model now unpacks the same
   way, so this explanation is closed.

*Resolved:* hypothesis (1) is confirmed and (2) is excluded as the driver — see
:ref:`char-break-even`.  The ``multiplicative`` unpacking is still a genuine bug (it
gives that scenario the worst rule accuracy of the three, 71.7 %) and should be fixed,
but it is not what makes the Uchuu numbers bad.

----

P1 — statistical rigour
------------------------

**Finish the mock-calibrated LRT against the matched spectra.**
NSIDE 32 is complete (9/9 samples, :math:`N = 50`).  NSIDE 64 has 6 of 9 at a measured
cost of :math:`\approx 14` h per cell; the three outstanding are all in the group whose
estimator is broken, so they cannot move the conclusion.  The Monte-Carlo :math:`p`
floor is :math:`1/(N+1)`, so :math:`N = 50` reports :math:`p \ge 0.020` and every
detection above sits exactly on it --- the test cannot currently distinguish a
:math:`3\sigma` result from a :math:`10\sigma` one.

*Action:* raise :math:`N` for the cells that reject at the floor, so the detections
acquire a magnitude rather than only a direction.  ``--resume-null`` adds mocks without
re-fitting the data.

----

P2 — method completeness
-------------------------

**Cross-template terms in the two-point correction.**  *(now measured —* 
:ref:`char-crossterms` *)*  In the PCA-rotated basis the pipeline actually uses, the
auto-only approximation carries a 7–17 % median error (up to 65 %); in the unrotated
basis it would be wrong by a factor 14–20.  The rotation is therefore load-bearing for
the correction, not just for MCMC mixing, and should be documented as such.
Both the :math:`w(\theta)` correction and the amplitude-bias estimator retain only
auto-terms (:math:`\xi_{ii}`, :math:`C_{ii}`).  The PCA rotation diagonalises the
template covariance at **zero lag** only; it does not make
:math:`\xi_{ij}(\theta) = 0` for :math:`\theta > 0`.

*Action:* extend to the full :math:`\sum_{ij}\tilde a_i \tilde a_j \xi_{ij}(\theta)`
form, or quantify the residual bias on mocks and document it as a known limitation.

**Reconcile harmonic and configuration space.**
:func:`~sys_mapping.power_spectrum.subtract_template_cl` subtracts
:math:`\hat\alpha_i C_\ell^{t_i}` (linear in the amplitude) while
:func:`~sys_mapping.correction.correct_two_point_function` subtracts
:math:`\tilde a_i^2 \xi_i` (the debiased square).  The two are not transforms of one
another.

**Calibration defaults.**
Three changes that the measurements support but that have not been applied, each
because it needs one confirmation run first:

* ``n_bins``: the ISD sweep shows ``n_bins = 20`` strictly dominating the shipped
  ``10`` at ``poly_order = 3`` --- better residual (0.028 vs 0.032), precision (0.925
  vs 0.870) *and* recall (0.835 vs 0.808).  Untested on the sparse LS10 samples at
  NSIDE >= 128, where the ``N_beta >= 2`` bin-admission rule could start rejecting
  bins.  *Action:* run the sweep at NSIDE 128, then change the default.
* ``cl_amplitude``: the scalar parametric null under-clusters LS10 by ~25x at the
  ``5e-4`` default and no single value corrects it --- the fitted amplitude spans two
  orders of magnitude across samples and resolutions.  *Action:* require ``cl_input``
  for any calibrated statistic rather than falling back to a scalar.
* The ``cl_amplitude`` grid has been regenerated under the corrected shot-noise
  subtraction.  Every accepted amplitude rose by an amount tracking the leak,
  +0.9 % at a leak of 0.7 % and +23 % at 15 %, and six of the 36 cells are now
  refused.

**Reporting.**
Three places where a number is quoted that should not be:

* Per-template significances from the iid likelihood have a 3-sigma false-positive
  rate of 76--96 % on clean simulations.  *Action:* quote sandwich-calibrated
  significances only, and make the iid path warn.
* ``max_i |r(w, t_i)|`` is used as a scalar goodness-of-fit and is degenerate: it
  depends on the *support* of ``a_hat``, not its size, so a single-template correction
  scores 1.0 at any amplitude and a method that fits nothing scores 0.  *Action:*
  report per template; regenerate ``null_tests.png`` without the summary.
* The docs quote a basis condition number of ``~1e8`` with no derivation; the measured
  value on the standardised basis the pipeline fits is ``1.4e3`` (NSIDE 64).
  *Action:* replace it.

**Release hygiene.**
``pyproject.toml`` is at 1.3.0 and ``CHANGELOG.md`` carries a dated 1.3.0 section.  The
tag is cut from the commit that re-issues the LS10 products at ``WEIGHTVER = 3``.

----

P3 — infrastructure
--------------------

* **API coverage check in CI** — assert that every module reachable from
  ``sys_mapping.__all__`` has a page under ``docs/api/``.  ``nuts`` and ``plotting``
  were both missing until this pass.
* **Guard the GLS + skew-normal combination**, which is not a normalised density,
  either by raising or by documenting it as a heuristic.
* **Harmonise estimator defaults** — the auto- and cross-2PCF use different default
  metrics (``Euclidean`` vs ``Arc``), and :math:`w(\theta)` and the
  :math:`\kappa\kappa` correlators use different default binning, so mixing them
  silently yields incomparable :math:`\theta` grids.


----

P2 — from the 2025/2026 literature
-----------------------------------

Four papers added to the technical paper's bibliography in September 2026 name
capabilities this package does not have.  Four of their proposals are now implemented
(see :doc:`bibliography` for the per-paper status); what follows is what is not,
each attributed to where it comes from and to what it would change here.

**Hernández-Monteagudo et al. 2025** (J-PLUS, OJAp 8, 93)

*Power-law template linearisation.*  Before standardising a template, fit
:math:`n_g^{\rm obs}/\langle n_g^{\rm obs}\rangle \propto (M_j/\langle M_j\rangle)^{\alpha_j}`
and replace :math:`M_j \rightarrow M_j^{1/|\alpha_j|}` when :math:`|\alpha_j| > 1`.
This makes the *linear* forward model of Eq. 3 a better approximation before any
fitting happens, which is cheaper than adding freedom to the fit and does not cost
degrees of freedom.  Lands in ``maps``, alongside the existing standardisation.

*Variance-minimisation estimator for* :math:`b`.  Subtract the OLS linear part, then
find the :math:`\beta_i` that minimise the variance of
:math:`\delta_g^{[1]} / \prod_i (1 + \beta_i t_i)`.  This is a genuinely independent
estimator of the multiplicative amplitude — additive contamination shifts the mean,
multiplicative contamination modulates the variance — so it is a cross-check on the
likelihood-ratio test rather than another way of running one.  Would be a seventh
method in ``run_decontamination``.

*The monopole.*  ``maps.compute_overdensity`` enforces
:math:`\langle\hat\delta_g\rangle = 0` through the single global normalisation
:math:`f_r`, so the package cannot see the additive contribution
:math:`\boldsymbol\alpha\cdot\bar{\mathbf M}` to :math:`\bar n_g` — the one part of
an additive contamination that rescales every :math:`C_\ell` by a constant.  Recovering
it requires an external :math:`\bar n_g`, e.g. from a purity measurement, and the
useful deliverable is to accept one and report the induced amplitude offset rather
than to pretend the bias is absent.

*Residual forecast per angular scale.*  They predict the residual on :math:`C_\ell`
as a function of :math:`\ell` for a given contamination configuration, finding a
:math:`\propto \ell^{-1}` decline.  ``docs/detectability_law`` gives the scalar
version of this; extending it to a per-:math:`\ell` curve is a small step from what is
already measured.

**Kong et al. 2026** (PRD 113, 043538)

*Sub-sample systematics diagnostic.*  Split the sample by an observable (magnitude,
colour), fit :math:`f_k` per sub-sample against the same templates, and evaluate
:math:`\sum_{kk'} h_k h_{k'} \langle f_k, f_{k'}\rangle \langle \delta_k, \delta_{k'}\rangle`.
This term survives a *perfect* first-order correction, is multiplicative in
:math:`w(\theta)`, and does not correlate with the templates — so
``diagnostics.null_test_cross_correlations`` cannot see it by construction, however
well it is calibrated.

*Window-decomposed estimator.*
:math:`w_{\rm obs}(\theta) = \sum_{AB} {\rm Win}_{AB}(\theta)\, w_{AB}(\theta)` with
:math:`{\rm Win}_{AB} = R_A R_B / R_{\rm tot} R_{\rm tot}`, the randoms binned by
systematics-map value.  ``utils.measure_cross_two_point_function`` already reuses
``dr``/``rr`` pair counts, which is most of the machinery.

*Spatially varying* :math:`n(z, {\rm sys})`, :math:`b(z, {\rm sys})`.  With the
four-parameter :math:`n(z)` model of DeRose et al. (shift, stretch, outlier fraction
and location), and the propagation of Baleato Lizancos & White 2023 and Hang et al.
2024.  The package models galaxy density with no redshift dependence at all today, so
this is the largest of the items here.

**Weaverdyck et al. 2026** (DES Y6 MagLim++, arXiv:2601.14484)

*Residual null test with a mock covariance.*  :math:`\chi^2` of the weighted density
in deciles of each template against a covariance from ~1000 contamination-free mocks,
the precision matrix built with Ledoit–Wolf optimal shrinkage (100 elements estimated
from 1000 mocks is noisy even when unbiased), then a KS test of those :math:`\chi^2`
against :math:`\chi^2_{10}`.  This is the direct fix for the null test the technical
paper reports failing with iid errors: the problem there is the error model, and a
mock covariance is the error model.

*Leverage mask.*  Cut pixels with high leverage — the diagonal of the hat matrix of
the template design matrix — so that the pixels with the most influence over the fit
are not also the ones where a perturbative contamination model is least likely to
hold.  Complements ``diagnostics.footprint_mask_diagnostics``, which looks at one
template at a time; leverage is the high-dimensional version.

*Data-split consistency test.*  Split the footprint in two, measure
:math:`w(\theta)` in each half, and test
:math:`\chi^2 = \Delta^{\rm T} (C/4)^{-1} \Delta`.

**DeRose et al. 2026** (arXiv:2603.10113)

*Fisher-bias propagation.*  Given the residual :math:`\Delta C_\ell` left by an
imperfect correction, compute the induced parameter bias
:math:`\delta\theta = F^{-1} \partial\mu^{\rm T} C^{-1} \Delta\mu`.  The likelihood
here is already differentiable under ``jax`` (``likelihood.make_log_likelihood``,
``nuts.build_logdensity``), so ``jax.jacfwd`` supplies the derivatives; what is missing
is a theory data vector to differentiate.  This is what turns the break-even rule from
a statement about :math:`w(\theta)` into a statement about :math:`\Omega_m` and
:math:`b\sigma_8` — which is the form in which a survey can act on it.
