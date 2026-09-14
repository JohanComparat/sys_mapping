Roadmap
=======

Open work, ordered by impact on scientific output rather than by implementation
effort.  Every entry corresponds to an open finding in the ``sys_mapping_paper``
document, whose resolution table gives the same list ordered by cost; this page adds
the implementation detail.

.. contents:: On this page
   :local:
   :depth: 1

----

P0 — correctness of the published products
------------------------------------------

**Include the cross-template terms at** :math:`\nside=64`.
The two-point correction and the amplitude-bias estimator retain only the auto terms.
The PCA rotation diagonalises the template covariance at zero lag, which does not make
:math:`\xi_{ij}(\theta)` vanish at :math:`\theta > 0`.  Measured with the fitted
amplitudes over the nine samples, the neglected terms are 1.9 % of the correction at
NSIDE 32 and 14.7 % at NSIDE 64, reaching 36.9 % on one sample.  The correction itself
reaches 18 % of :math:`\hat w` in the widest bins, so at NSIDE 64 the neglected term
is about 3 % of the corrected :math:`w(\theta)`.

The share is a property of the rotated basis rather than of the amplitudes' size:
``ISD-1`` gives 2.5 % on the same cells with amplitudes differing by a factor 2.5 in
norm.  :func:`~sys_mapping.contamination.compute_two_point_correction` accepts the full
:math:`(\nsys, \nsys, n_\theta)` matrix; building it costs
:math:`\nsys(\nsys{+}1)/2 = 66` cross-spectra at :math:`\nsys = 11`, minutes per
analysis.  *Action:* re-issue the products through that path.

**Complete the product grid at NSIDE 256.**
``MCMC-comb`` takes 6.1 h at NSIDE 128 and NSIDE 256 has four times the pixels, so the
cell needs roughly 24 h against the 12 h the campaign allows.  No calibrated result
depends on that resolution.  *Action:* submit with a 36 h walltime, or state that the
supported resolutions are 32, 64 and 128.

----

P1 — statistical rigour
------------------------

**Give the per-template significances a covariance that holds on a correlated field.**
The iid likelihood has a 3-sigma false-positive rate of 76--96 % on clean simulations,
so a significance quoted from it is not a detection.  A full-rank GLS precision is the
fix rather than a repair: :math:`R^{-1}v` on the cut sky is a conjugate-gradient solve
wrapping a spherical-harmonic operator, differentiable under ``jax``.  Until it exists,
:func:`~sys_mapping.diagnostics.snr_template_ranking` warns and the calibrated errors
come from :func:`~sys_mapping.covariance.mock_sandwich_covariance`.

**Give the rejections a magnitude.**
The Monte-Carlo :math:`p` floor is :math:`1/(N+1)`, so :math:`N = 50` reports
:math:`p \ge 0.020` and every rejection sits exactly on it: the test cannot
distinguish a :math:`3\sigma` result from a :math:`10\sigma` one.  *Action:* raise
:math:`N` for the cells that reject at the floor.  ``--resume-null`` adds mocks without
re-fitting the data.

**Require a matched spectrum for any calibrated statistic.**
The scalar parametric null under-clusters LS10 by a factor of about 25 in variance at
the ``5e-4`` default, and no single value corrects it: the fitted amplitude spans two
orders of magnitude across samples and resolutions.  *Action:* require ``cl_input``
rather than falling back to a scalar.

----

P2 — method completeness
-------------------------

**Confirm the ISD bin count at high resolution.**
The sweep shows ``n_bins = 20`` dominating the shipped ``10`` at ``poly_order = 3`` on
residual (0.028 against 0.032), precision (0.925 against 0.870) and recall (0.835
against 0.808).  It is untested on the sparse LS10 samples at NSIDE >= 128, where the
:math:`N_\beta \ge 2` bin-admission rule could start rejecting bins.  *Action:* run
the sweep at NSIDE 128, then change the default.

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

Four papers in the technical paper's bibliography name capabilities this package does
not have.  :doc:`bibliography` carries the per-paper status; what follows is what is
absent, each attributed to where it comes from and to what it would change here.

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
