Algorithm characterisation: when does correcting actually help?
================================================================

.. note::
   Three studies that together answer *when the decontamination is worth applying* and
   *how far its stated uncertainties can be trusted*.  The first re-analyses existing
   simulation output; the second and third are new measurements.

   The scripts live in the separate
   `sys_mapping_benchmark <https://github.com/JohanComparat/sys_mapping_benchmark>`_
   repository, under ``characterisation/``; point them at a checkout of this package
   with ``SYS_MAPPING_ROOT``.  The data behind every table below is committed here
   under ``_static/characterisation/``, so this page renders standalone.

.. contents:: On this page
   :local:
   :depth: 1

----

.. _char-break-even:

1. The break-even condition
---------------------------

Two large simulation campaigns already existed and had never been joined:
``results/detectability_sweep_*.csv`` (14 400 rows) measures *field-level detection*
against amplitude, while ``data/simulations/nside*/results_summary.json`` (18
configurations) measures *w(θ) correction quality*.  The first says when a systematic
is detectable; the second says whether correcting helped.

**The mechanism.**  Correcting removes a bias but injects the noise of the fitted
amplitudes.  Writing

.. math::

   A = {\rm rms}\Bigl(\sum_i a_i^{\rm true} t_i\Bigr), \qquad
   \Delta A = {\rm rms}\Bigl(\sum_i (\hat a_i - a_i^{\rm true})\, t_i\Bigr)

for the field the correction must remove and the mis-fit field it adds back, the
correction can only help when the recovered field is closer to the truth than to
zero, i.e. when :math:`A/\Delta A \gtrsim 1`.  Both quantities are computable from
what is already stored (``a_true``, ``a_hat``, and the named templates).

**Result.**  :math:`A/\Delta A` predicts the sign of the effect across all 180
method–configuration cells:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * -
     - correction helped
     - correction hurt
   * - :math:`A/\Delta A > 1`
     - **57**
     - 15
   * - :math:`A/\Delta A < 1`
     - 23
     - **85**

Rank correlation between :math:`\log(A/\Delta A)` and :math:`\log` improvement is
**+0.73**; the simple rule "correct only if :math:`A/\Delta A>1`" is right in 79 % of
cells.

**This explains the Uchuu results.**  The Uchuu configurations sit at median
:math:`A/\Delta A = 0.76` (below break-even) and helped in only 16.7 % of cells; the
GLASS ones sit at 1.28 and helped in 72.2 %.

**And it identifies the cause.**  Splitting by scenario isolates it, because the
``multiplicative`` fit/inject mismatch cannot apply to additive-only runs:

.. list-table::
   :header-rows: 1
   :widths: 20 12 16 16 16

   * - scenario
     - n
     - helped
     - median :math:`A/\Delta A`
     - rule accuracy
   * - additive
     - 60
     - 60.0 %
     - 1.39
     - 80.0 %
   * - multiplicative
     - 60
     - 23.3 %
     - 0.98
     - 71.7 %
   * - combined
     - 60
     - 50.0 %
     - 1.15
     - 85.0 %

Restricting further to **additive-only on Uchuu** — where the fitted model is exactly
the injected one — gives 0 % helped at low and medium amplitude
(:math:`A/\Delta A = 0.45` and :math:`0.61`) and 70 % at high
(:math:`A/\Delta A = 0.72`).  The failure therefore is **not** caused by the
multiplicative model mismatch; it is the detection threshold.  Correcting a
systematic you cannot measure makes the answer worse.

.. admonition:: Practical rule
   :class: important

   Estimate :math:`\Delta A` from the fit covariance and correct only when
   :math:`A/\Delta A > 1`.  Below that, applying weights degrades :math:`w(\theta)`.

----

.. _char-variance-inflation:

2. How wrong the iid error bars are
------------------------------------

The recorded null test reports a 3σ false-positive rate of 76–96 % where 0.27 % is
expected, but only as a flag.  ``run_variance_inflation.py`` measures the
inflation directly on uncontaminated mocks, comparing the empirical scatter of
:math:`\hat a` across realisations against the analytic OLS error the pipeline
reports:

.. math::

   \kappa_i = \frac{\sigma^{\rm emp}[\hat a_i]}{\sigma^{\rm iid}[\hat a_i]}

**Control.**  On pure-Poisson maps — the regime where the iid likelihood is valid —
:math:`\kappa` lands in :math:`[0.987, 1.021]` across **all twenty** cells and the
per-template FPR(3σ) has median 0.26 % against a nominal 0.27 %.  The measurement is
sound, which is what licenses the rest of this section.

**Clustered fields**, at the package default spectrum (:math:`\alpha = 2`) and
:math:`\sigma_{\rm clus} = 0.4`, from 2000 realisations per cell:

.. list-table::
   :header-rows: 1
   :widths: 14 14 14 14 14 14

   * - NSIDE
     - :math:`n_s=3`
     - :math:`n_s=5`
     - :math:`n_s=7`
     - :math:`n_s=9`
     - :math:`n_s=11`
   * - 32
     - 7.41
     - 6.53
     - 6.50
     - 5.88
     - 5.30
   * - 64
     - 12.94
     - 11.09
     - 10.75
     - 9.80
     - 8.76

(:math:`\kappa` field.)  So the reported error is **5–13× too tight** at the default
spectrum, and 2–23× across the full range of spectra tested.  Two trends: the
inflation **grows with resolution** (finer pixels resolve more of the clustering the
iid model calls noise) and **shrinks with template count** (more templates absorb more
of it into the model).  The FPR for *any* of the :math:`n_s` templates is 96–100 % at
NSIDE 32 and 99–100 % at NSIDE 64.

**It is almost independent of the clustering amplitude.**  Doubling
:math:`\sigma_{\rm clus}` from 0.2 to 0.4 moves :math:`\kappa` by under 5 % in every
cell (7.09 → 7.41 at NSIDE 32, :math:`n_s=3`; 10.29 → 10.75 at NSIDE 64,
:math:`n_s=7`).  What sets the inflation is the *shape* of the clustering, not how
much of it there is.

**It depends strongly on that shape.**  Scanning
:math:`C_\ell \propto (\ell+1)^{-\alpha}` at NSIDE 64, :math:`n_s=11`:

.. list-table::
   :header-rows: 1
   :widths: 20 16 16 16 16 16

   * - :math:`\alpha`
     - 1.0
     - 1.5
     - 2.0
     - 2.5
     - 3.0
   * - :math:`\kappa` field
     - 2.73
     - 5.10
     - 8.76
     - 12.80
     - 16.10

so :math:`\kappa` must always be quoted with the spectrum it was measured on.  The
FPR exceeds 96 % at every slope tested, though — that conclusion is robust.

.. note::

   These numbers supersede an earlier six-cell grid measured at
   :math:`n_{\rm real} = 150`, whose source data was overwritten before it could be
   committed.  The present 100-cell run reproduces it where the two overlap
   (:math:`\alpha = 2`, :math:`\sigma_{\rm clus} = 0.4`) to within 1–4 % at every
   shared cell.  The earlier grid also quoted :math:`n_s = 1`, which the new grid does
   not cover, so the old headline of ":math:`\kappa` up to 18.4" is not reproduced
   here and has been dropped rather than carried forward unverified.

----

.. _char-glass-underclustered:

3. The mocks used for calibration are under-clustered
------------------------------------------------------

.. warning::

   The GLASS mocks are the null hypothesis for **three** separate calibrations —
   Stage-1 ISD p-values, the mock-calibrated LRT, and the sandwich covariance.  The
   footprint-matching rule guarantees they reproduce the data's *surface density*, and
   therefore its shot noise, but **nothing constrains their clustering**.

Generating a mock exactly as the pipeline does for the fiducial LS10 sample
(``calibrate_glass_clustering.py``):

.. list-table::
   :header-rows: 1
   :widths: 30 22 22 22

   * -
     - :math:`\bar n` (gal/pix)
     - :math:`\hat\sigma`
     - :math:`\sigma_{\rm clus}`
   * - LS10, log M\ :sub:`*` ≥ 10.0, NSIDE 64
     - 127.35
     - 0.3969
     - 0.3869
   * - GLASS mock (default ``cl_amplitude=5e-4``)
     - 127.32
     - 0.1179
     - 0.0777

The shot noise matches to four digits — the footprint matching works exactly as
designed.  But for this cell at the package default the **clustering variance is 25×
too low** (5× in :math:`\sigma`), and
the mock's total scatter is 3.4× smaller than the data's.  The mocks therefore sit
close to the shot-noise-dominated regime **where the iid likelihood is valid** —
precisely the regime the real data is not in.

The consequence is that all three "calibrated" quantities are calibrated against a
null that is too narrow, and remain overconfident.  This does not overturn the LS10
detections (the data :math:`\lambda_{\rm LR}` sits 30–60× above the null maximum, a
wide margin) but it does mean the margin is overstated.

**The fix.**  Scanning ``cl_amplitude`` until the mock reproduces the measured
:math:`\hat\sigma`:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - ``cl_amplitude``
     - :math:`\hat\sigma`
     - :math:`\sigma_{\rm clus}`
     - ratio to data
   * - 5.0e-4 *(default)*
     - 0.1179
     - 0.0777
     - 0.20×
   * - 1.0e-2
     - 0.2234
     - 0.2051
     - 0.53×
   * - **3.8e-2**
     - **0.3965**
     - **0.3864**
     - **1.00×**

The scan above is a single cell.  Root-finding the amplitude properly — over all nine
samples at four resolutions with five seeds each, 180 fits — shows it is **not one
number to correct once**:

.. list-table:: Fitted ``cl_amplitude`` (median over converged seeds)
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - :math:`\log M_\star \ge`
     - NSIDE 32
     - NSIDE 64
     - NSIDE 128
     - NSIDE 256
   * - 9.00
     - 0.1038
     - 0.0922
     - 0.0791
     - —
   * - 9.50
     - 0.0813
     - 0.0603
     - 0.0692
     - —
   * - 10.00
     - 0.0534
     - 0.0363
     - 0.0432
     - 0.0395
   * - 10.25
     - 0.0431
     - 0.0269
     - 0.0327
     - 0.0313
   * - 10.50
     - 0.0386
     - 0.0212
     - 0.0261
     - 0.0244
   * - 10.75
     - 0.0343
     - 0.0168
     - 0.0203
     - —
   * - 11.00
     - 0.0314
     - 0.0169
     - 0.0195
     - —
   * - 11.25
     - 0.0351
     - 0.0198
     - —
     - —
   * - 11.50
     - 0.0431
     - 0.0005
     - —
     - —

Median over the 143 converged fits is 3.4e-2, a factor **69** above the default, with
1–5 % seed-to-seed scatter — so the spread across the table is real.  Three things
matter more than the median:

* The amplitude tracks how much of each sample's variance is clustering rather than
  shot noise, not the stellar mass as such.
* The :math:`\log M_\star \ge 11.5`, NSIDE 64 cell needs **no** correction: at
  :math:`\bar n = 5.6` galaxies per pixel the default already reproduces the measured
  :math:`\sigma_{\rm clus}` to 1 %.  A single global rescaling would have
  over-clustered exactly the samples that were already right.
* The dashes are a limit of the method, not gaps in the run: all 37 non-converged fits
  lie at NSIDE ≥ 128, where these samples fall below roughly five galaxies per pixel
  and the mock's shot noise alone exceeds the target scatter, so no amplitude
  reproduces it.  At NSIDE 32 and 64 — where the calibrated statistics are actually
  evaluated — the grid is complete.

:func:`~sys_mapping.diagnostics.isd_template_significance` and the LRT null builder
both accept ``cl_amplitude``, so the fitted value can be passed per sample.

----

.. _char-crossterms:

4. The dropped cross-terms
---------------------------

The two-point correction keeps only template *auto*-correlations, while the
contamination an additive field imprints is its full autocorrelation
:math:`\sum_{ij} a_i a_j \xi_{ij}(\theta)`.  ``run_crossterm_bias.py``
evaluates both forms using the **amplitudes actually fitted to LS10**:

.. list-table::
   :header-rows: 1
   :widths: 16 24 30 30

   * - NSIDE
     - basis
     - median share
     - range over the nine samples
   * - 32
     - original
     - ~870 %
     - 650–1040 %
   * - 32
     - PCA-rotated
     - **1.9 %**
     - 0.1–3.4 %
   * - 64
     - original
     - ~193 %
     - 63–533 %
   * - 64
     - PCA-rotated
     - **14.7 %**
     - 5.7–36.9 %

**The auto-only form needs the rotated basis.**  In the original correlated basis the
approximation is wrong by a factor of a few to ten.  The pipeline applies the
correction in the rotated basis (``run_ls10_analysis.py`` passes
``a_rot``/``b_rot``/``ct_rot``), which is what brings the share to the level tabulated
above, beyond the rotation's effect on MCMC mixing.

The residual is 1.9 % at NSIDE 32, where the auto-only form is defensible, and 14.7 %
at NSIDE 64, where it is not.  The share depends on the method through the orientation
of its amplitude vector: ``ISD-1`` gives 2.5 % on the same nine cells against 14.7 %
for ``OLS`` and ``MCMC-add``.  ``compute_two_point_correction`` accepts the
``(n_sys, n_sys, n_theta)`` matrix, which costs ``n_sys(n_sys+1)/2 = 66`` cross-spectra
to build at ``n_sys = 11``, minutes per analysis.

.. warning::
   This quantity **cannot** be measured with stand-in amplitudes.  Random amplitudes of
   the same typical size give 6.3 % and 4.9 % — three times too large at NSIDE 32, and
   inverting the ordering between the two resolutions.  ``sum_ij a_i a_j xi_ij`` depends
   on how the amplitude *vector* is oriented relative to the off-diagonal
   ``xi_ij``, not on its length, and a fit in the rotated basis lands in an orientation
   that suppresses the cross terms.  ``run_crossterm_bias.py`` therefore takes
   ``--params-json`` and warns when it falls back to random draws.
