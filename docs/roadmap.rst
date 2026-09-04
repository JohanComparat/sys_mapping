Roadmap
=======

Prioritised next steps, grounded in the verification pass documented in
the ``sys_mapping_paper`` document (§9) and in the open items in ``TODO``.
Priorities reflect impact on *scientific output*, not implementation effort.

.. contents:: On this page
   :local:
   :depth: 1

----

P0 — correctness of the published products
------------------------------------------

**Unify the weight definition.**
Three conventions are live (see the weighting table in the README).
:func:`~sys_mapping.regression.run_decontamination` computes an exact-inverse weight
and stores it in ``result["weights"]``; **neither production script reads that
field**.  Both recompute from the fitted coefficients with a different denominator
floor (0.01, i.e. a maximum weight of 100, against the library's clip of
:math:`[1/20, 20]`), and ``WEIGHT_COMB`` is built from :math:`\hat b` alone, which
equals the exact inverse only when :math:`\hat a \approx 0`.

*Action:* pick the canonical definition (recommended: the library's W2 exact
inverse), make ``compute_sys_weights.py`` and ``run_ls10_analysis.py`` consume
``result["weights"]`` rather than reimplementing it, and re-issue any
``data/sys_weights/`` product built with the other convention.
*Blast radius:* changes published weight values; requires a re-run.

**Align the likelihood between the two scripts.**
``compute_sys_weights.py`` fits the **skew-normal** likelihood by default
(``use_skewed = not args.no_skewed``); ``run_ls10_analysis.py`` never passes
``use_skewed`` and is always **Gaussian**.  Both write a column named
``WEIGHT_SYS`` consumed by the same downstream package.

*Action:* choose one default, make it explicit in both CLIs, and record it in the
FITS header alongside the weight formula.

**Recalibrate the GLASS mocks to the data's clustering.**  *(new, and it outranks
everything else here)*  The GLASS null is used by Stage-1 ISD p-values, the
mock-calibrated LRT and the sandwich covariance.  Its shot noise matches the data
exactly, but its **clustering variance is 25x too low** (``cl_amplitude=5e-4``
gives :math:`\sigma_{\rm clus}=0.078` against the data's :math:`0.387`), so all three
"calibrated" nulls are too narrow and remain overconfident.  See
:ref:`char-glass-underclustered`.

*Action:* fit ``cl_amplitude`` per sample so the mock reproduces the measured
:math:`\hat\sigma` (:math:`3.8\times10^{-2}` for the fiducial sample, a factor 76 above
the default), then re-run the mock-calibrated LRT.  Expect the detections to survive —
the data statistic sits 30–60x above the null maximum — but with a smaller margin.
``characterisation/calibrate_glass_clustering.py`` in the
`sys_mapping_benchmark <https://github.com/JohanComparat/sys_mapping_benchmark>`_
repository does the fitting.

**Adopt the break-even rule.**  *(resolved — see* :ref:`char-break-even` *)*  The
question of why the Uchuu simulations degrade is settled: it is the detection
threshold, not the ``multiplicative`` model mismatch.  Additive-only runs, where the
fitted model is exactly the injected one, still show 0 % improvement at low and medium
amplitude.  Correcting helps only when :math:`A/\Delta A > 1`.

*Action:* compute :math:`\Delta A` from the fit covariance inside
``run_decontamination`` and expose :math:`A/\Delta A` in the result dict, so callers
can refuse to apply weights below break-even instead of silently degrading
:math:`w(\theta)`.

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

**Give** :math:`\lambda_{\rm LR}` **a real MLE.**
:func:`~sys_mapping.inference.get_mle_params` returns the marginal posterior
*median* despite its name.  Differencing two medians is not a likelihood ratio: the
nesting guarantee :math:`\lambda_{\rm LR} \ge 0` is lost and the statistic is
observed to go negative at NSIDE 64.  The mock-calibrated test stays valid (data and
null share the estimator), but the magnitude of :math:`\lambda_{\rm LR}` is
meaningless.

*Action:* add a gradient refinement — the JAX log-likelihood is already
differentiable, so ``jax.grad`` plus ``scipy.optimize.minimize`` from the posterior
median converges in a few tens of iterations.  Keep the mock calibration regardless;
it is correcting for pixel correlations, not for the estimator.

**Finish the mock-calibrated LRT.**
NSIDE 32 is complete at :math:`N = 30` (9/9 samples).  NSIDE 64 has 1 of 9, at a
measured cost of :math:`\approx 10` h per sample-resolution cell.  The Monte-Carlo
:math:`p`-value floor is :math:`1/(N+1)`, so :math:`N = 30` can only ever report
:math:`p \le 0.032`.

*Action:* run the 8 missing NSIDE-64 cells (:math:`\approx 80` h, best on a cluster),
then top up all cells to :math:`N \ge 50` with ``--resume-null``, which adds only the
missing mocks without re-fitting the data.

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

**Release hygiene.**
``pyproject.toml`` is at 1.2.0, the tag ``v1.2.0`` exists with 20 commits after it,
and ``CHANGELOG.md`` stops at 1.1.0.  Cut 1.3.0 covering the covariance module,
mock-calibrated LRT, :math:`w(\theta)` covariance, detectability law, NUTS
``chain_method``, and the sweep runner.

----

P1 — the two failing tests
---------------------------

``pytest -m "not slow"`` gives **430 passed, 2 failed, 16 skipped**.  Both failures are
in ``tests/test_snr_preselection.py::TestMethodComparison``
(``test_all_snr_methods_rank_contaminant2_in_top2``,
``test_isd_snr_contaminants_larger_than_noise``) and both are long-standing — a
numerical edge case in the ``poly_order=1`` ``_one_isd`` path, already failing at the
v1.1.0 release.

They matter for two reasons: CI runs the full suite with no marker filter, so the
build is red; and the ISD :math:`\Delta\chi^2` is precisely the statistic that Stage-1
pre-selection relies on, so a ranking edge case there is not cosmetic.

*Action:* reproduce with a fixed seed, determine whether the contaminated template
genuinely fails to rank in the top two (a method limitation, in which case the test
encodes an unrealistic expectation) or whether the binned :math:`\Delta\chi^2` is
mis-normalised at ``poly_order=1``.  Either fix the statistic or re-state the test
against what the method can actually deliver — but do not simply mark it ``xfail``
without establishing which.

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
