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
:math:`\kappa = 0.95`–:math:`1.05` and FPR(3σ) = 0–3 % across the whole grid.  The
measurement is sound.

**Clustered fields** (:math:`\sigma_{\rm clus}=0.4`, matching LS10):

.. list-table::
   :header-rows: 1
   :widths: 12 12 16 16 16

   * - NSIDE
     - :math:`n_s`
     - :math:`\kappa` per template
     - :math:`\kappa` field
     - FPR (3σ)
   * - 32
     - 1
     - 8.9
     - 8.9
     - 73.5 %
   * - 32
     - 5
     - 5.4
     - 6.3
     - 99.0 %
   * - 32
     - 11
     - 3.4
     - 5.2
     - 100 %
   * - 64
     - 1
     - 18.4
     - 18.4
     - 84.0 %
   * - 64
     - 5
     - 9.1
     - 10.9
     - 100 %
   * - 64
     - 11
     - 5.3
     - 8.8
     - 100 %

Two trends: the inflation **grows with resolution** (finer pixels resolve more of the
clustering the iid model calls noise) and **shrinks with template count** (more
templates absorb more of it into the model).

**It depends strongly on the assumed clustering spectrum.**  Scanning
:math:`C_\ell \propto (\ell+1)^{-\alpha}` at NSIDE 64, :math:`n_s=11`:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - :math:`\alpha`
     - 1.5
     - 2.5
     - 4.0
   * - :math:`\kappa` per template
     - 3.2
     - 7.2
     - 13.1

so :math:`\kappa` must always be quoted with the spectrum it was measured on.  The
FPR exceeds 99 % at every slope tested, though — that conclusion is robust.

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
designed.  But the **clustering variance is 25× too low** (5× in :math:`\sigma`), and
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

``cl_amplitude = 3.8e-2`` — a factor **76** above the package default — reproduces the
LS10 field to better than 1 %.  The amplitude is sample- and resolution-dependent, so
it should be fitted per sample from the measured :math:`\hat\sigma` rather than
hard-coded.

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
     - median \|error\|
     - max \|error\|
   * - 32
     - original
     - 1952 %
     - 3459 %
   * - 32
     - PCA-rotated
     - **7.1 %**
     - 64.5 %
   * - 64
     - original
     - 1369 %
     - 1973 %
   * - 64
     - PCA-rotated
     - **17.3 %**
     - 33.7 %

**The PCA rotation is load-bearing.**  In the original correlated basis the auto-only
approximation is wrong by a factor of 14–20 — it would be unusable.  The pipeline
applies the correction in the rotated basis
(``run_ls10_analysis.py`` passes ``a_rot``/``b_rot``/``ct_rot``), which reduces the
error to 7–17 % typical.  The rotation is therefore not merely an MCMC-mixing
convenience: it is what makes the correction viable at all, and that should be stated
wherever the rotation is described as optional.

A residual 7–17 % systematic on the correction term nevertheless remains, and is
worth either fixing (the full :math:`\sum_{ij}` form costs one
:math:`n_s \times n_s` contraction) or quoting as a systematic floor.
