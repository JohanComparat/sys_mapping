Roadmap
=======

Prioritised next steps.  Every entry corresponds to an open finding in the
``sys_mapping_paper`` document, whose §9.3 table gives the same list ordered by cost
to the science; this page adds the implementation detail.  Priorities reflect impact
on *scientific output*, not implementation effort.

**The two that a next version should contain**, in order:

1. Standardise the template basis on the footprint (P0 below).  The maps are
   normalised over each map's own valid region, so on the pixels the fit uses the
   eleven LS10 templates at NSIDE 64 have rms spanning 0.905 to 5.77.  Every
   amplitude, the condition number, and the template auto-correlations the two-point
   correction subtracts are read in units that do not hold there.
2. Measure the template two-point functions from the galaxies (P0 below).  They are
   measured on the pixel grid, so they are exactly zero inside one pixel and the
   correction does nothing over 21 of 30 bins at NSIDE 64 and 24 of 30 at NSIDE 32.

.. contents:: On this page
   :local:
   :depth: 1

----

P0 — correctness of the published products
------------------------------------------

**Standardise the template basis on the footprint.**  *(highest value in this list)*
``load_templates_from_dir`` standardises each survey-property map over that map's own
valid region.  The analysis footprint is a subset of it, and the basis is not
standardised there: at NSIDE 64 the eleven LS10 maps have per-template rms spanning
0.905 to 5.77 and a relative mean of 0.729, so the covariance eigenvalues sum to 44.4
rather than ``n_sys = 11`` with a leading eigenvalue of 37.5.

Three quantities are read in units that do not hold.  Fitted amplitudes are not in
units of one standard deviation of the template, so they are not comparable between
templates.  :func:`~sys_mapping.utils.compute_covariance_matrix` is an uncentred
second moment and is a covariance only for a zero-mean basis.  And
:math:`\xi_i(\theta)` in the two-point correction carries the same scaling, which is
what drives ``ISD-3``'s corrected :math:`w(\theta)` negative in all 18 shipped LS10
cells, to :math:`-39\times \hat w` at NSIDE 64 and :math:`-232\times` at NSIDE 32.

*Action:* re-standardise after masking, where the footprint is known, rather than at
load time.  ``compute_covariance_matrix`` now warns on both conditions, so the defect
is loud; closing it needs every LS10 amplitude re-fitted.
*Blast radius:* every published amplitude, the condition number, and the corrected
:math:`w(\theta)`.  A full campaign.

**Measure the template two-point functions from the galaxies.**
``run_ls10_analysis.py`` measures :math:`\xi_i(\theta)` with TreeCorr on the rotated
templates at the HEALPix pixel centres.  No pair of distinct pixels is separated by
less than the pixel scale, so :math:`\xi_i` comes back as exactly zero below it, while
:math:`\hat w(\theta)` is measured from the catalogue down to :math:`0.5'`.  The
correction is therefore identically zero over the bins carrying most of the signal:
21 of 30 at NSIDE 64 (below :math:`49'`) and 24 of 30 at NSIDE 32 (below
:math:`94'`).  The first corrected bin tracks the pixel scale in both.

*Action:* attach each galaxy the template value of its pixel and run the KK
correlation on the galaxy positions, at the same separations as :math:`\hat w`.
*Blast radius:* the corrected :math:`w(\theta)` at every separation below the pixel
scale, which is currently uncorrected rather than wrong.

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

**Re-run the LS10 ISD columns.**  *(done at NSIDE 32 and 64)*
The 18 shipped products at NSIDE 32 and 64 carry the library's own per-pixel weight,
taken from ``result["weights"]``, and record ``WEIGHTVER = 2``, ``WEIGHTCON`` and
``WMAXCLIP``.  ``WEIGHT_ISD3`` is non-trivial in all 18 and ``WEIGHT_ISD1`` in 16,
being identically unity for the two densest NSIDE-32 samples, where the stopping rule
accepts no template.  NSIDE 128 and 256 are still ``WEIGHTVER = 1`` and are being
regenerated.

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
#. the ``multiplicative`` *fit* model sets :math:`b = a` while retaining :math:`a`,
   giving :math:`\hat\delta_g = \delta_g(1+\sum a_i t_i) + \sum a_i t_i`, whereas the
   *injector* uses the pure form :math:`a = 0`, :math:`b` free — so the fit model
   cannot represent the field it is validated against.

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
* The family-B amplitude grid predates the ``--min-probes`` / ``--max-shot-leak``
  guards and the corrected shot-noise subtraction in ``measure()``.  *Action:* re-run
  it; expect ~8 % shifts where the leak was small and non-convergence where it was
  not.

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
``pyproject.toml`` is at 1.2.0, the tag ``v1.2.0`` exists with 20 commits after it,
and ``CHANGELOG.md`` stops at 1.1.0.  Cut 1.3.0 covering the covariance module,
mock-calibrated LRT, :math:`w(\theta)` covariance, detectability law, NUTS
``chain_method``, and the sweep runner.

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
