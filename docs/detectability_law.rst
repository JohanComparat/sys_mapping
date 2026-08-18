Systematic-detectability law — survey design rules of thumb
===========================================================

.. note::
   Rapid, analytic + reuse-only answer (Stage 1) to: *given a survey setup
   (sky coverage, pixel size, galaxy density), what is the smallest systematic
   affecting galaxy number density that can be detected, and at what
   significance?* No simulations are run; the law is analytic and its
   normalisation is read from existing LS10 fit outputs. The corresponding
   analysis for other surveys lives in their own repositories
   (``results_glass_detectability_law``); the cross-survey combination is on
   :doc:`survey_design_synthesis`.

The law
-------

Model the observed overdensity per pixel as a systematic field plus noise,
:math:`\delta_g = f + \varepsilon`, with :math:`f=\sum_i a_i t_i` (templates
mean-subtracted, unit-variance) and per-pixel noise variance
:math:`\hat\sigma^2` — the fit's residual scatter, which is **shot ⊕ clustering**:
:math:`\hat\sigma^2 = 1/\bar n_{\rm pix} + \sigma_{\rm clus}^2`. With
:math:`N_{\rm gal}=\bar n_{\rm pix} N_{\rm pix}` and field RMS :math:`A={\rm rms}(f)`:

.. math::

   {\rm SNR}_{\rm field} = \frac{\lVert f\rVert}{\hat\sigma}
   = A\,\frac{\sqrt{N_{\rm pix}}}{\hat\sigma} = A\sqrt{N_{\rm eff}},
   \qquad
   \boxed{\,A_{\min}(\nu\sigma) = \nu\,\frac{\hat\sigma}{\sqrt{N_{\rm pix}}}\,},
   \qquad
   N_{\rm eff}\equiv\frac{N_{\rm pix}}{\hat\sigma^2}\le N_{\rm gal}.

The single substitution :math:`\sqrt{N_{\rm gal}}\to\sqrt{N_{\rm eff}}` carries
the shot-noise idealisation :math:`A_{\min}=\nu/\sqrt{N_{\rm gal}}` into the
correlated-field reality. The field statistic is **VIF-free** (the recovered
combination is well-constrained even when individual templates are collinear —
the LS10 basis has condition number :math:`\sim10^8`), whereas the
per-template amplitude carries a variance-inflation factor
:math:`{\rm VIF}_i=1/\sqrt{1-R_i^2}`. This is the quantitative form of *judge by
the field, not the name*.

Rules of thumb
--------------

* **More galaxies help only until clustering dominates.** In the shot regime
  (:math:`\bar n_{\rm pix}<1/\sigma_{\rm clus}^2`) :math:`A_{\min}\propto
  1/\sqrt{N_{\rm gal}}`; in the clustering regime it saturates at a
  cosmic-variance floor set by the number of independent modes,
  :math:`\propto 1/\sqrt{f_{\rm sky}}` — **more area, not more depth**, lowers it.
* **Finer pixels resolve more modes.** At fixed :math:`N_{\rm gal}`,
  :math:`A_{\min}=\nu\hat\sigma/\sqrt{N_{\rm pix}}` keeps falling as pixels
  refine (more independent measurements of the same smooth systematic) until the
  shot floor :math:`\nu/\sqrt{N_{\rm gal}}` — refine to just below the
  systematic's coherence scale, no finer.
* **The honest error bar is the sandwich.** The per-template iid
  :math:`\sigma_i` is :math:`\sim2\times` too tight for every method; multiply
  detection SNRs by :math:`\approx1/1.9` (or use
  ``sys_mapping.mock_sandwich_covariance``).
* **w(θ) is a weaker detector** — its contamination signal grows as
  :math:`A^2`, so the direct field regression sees fainter systematics.

LS10 worked example (log :math:`M_*\ge` 10.0)
------------------------------------------------

The fiducial LS10 volume-limited sample (:math:`N_{\rm gal}=`\ 2,759,238,
:math:`f_{\rm sky}\approx`\ 0.441) is **clustering-limited**: at the recommended
NSIDE 64 the per-pixel noise :math:`\hat\sigma=`\ 0.397 is dominated by
clustering (:math:`\bar n_{\rm pix}=`\ 127, shot
:math:`1/\bar n_{\rm pix}=`\ 0.0079), so only :math:`N_{\rm eff}/N_{\rm
gal}=`\ 0.050 of the galaxies count toward detection. The smallest detectable
systematic field RMS is :math:`A_{\min}(3\sigma)=`\ 8.09e-03 (:math:`5\sigma`:
1.35e-02), versus the shot-floor 1.81e-03. The dominant real detection is
**GAIA_nstar_faint** at iid SNR 7.0 (:math:`\approx`\ 3.7 calibrated) — the
known LS10 BGS stellar-density systematic.

.. figure:: /_static/detectability_law/fig1_Amin_vs_Ngal.png
   :width: 88%

   Smallest detectable systematic vs galaxy count across the nine LS10
   stellar-mass samples (real fits). :math:`A_{\min}\propto\hat\sigma` at fixed
   footprint, so sensitivity peaks for the intermediate-mass samples (lowest
   :math:`\hat\sigma`), not the most numerous.

.. figure:: /_static/detectability_law/fig2_Amin_vs_nside.png
   :width: 88%

   Pixel size at fixed :math:`N_{\rm gal}`: refining NSIDE 32→256 lowers
   :math:`A_{\min}` toward the shot floor (more resolved modes).

.. figure:: /_static/detectability_law/fig3_crossover.png
   :width: 88%

   Shot vs clustering per-pixel variance. LS10 sits deep in the
   clustering-limited regime at every NSIDE tested.

.. figure:: /_static/detectability_law/fig4_Neff_fraction.png
   :width: 88%

   The usable fraction :math:`N_{\rm eff}/N_{\rm gal}` (1 = pure shot noise).

.. figure:: /_static/detectability_law/fig5_per_template_snr.png
   :width: 92%

   Per-template detection SNR (iid vs sandwich-calibrated); stellar density
   dominates. Individual templates are collinear (VIF-inflated); the *field* is
   robust.

.. figure:: /_static/detectability_law/fig6_detection_vs_amplitude.png
   :width: 88%

   Fast-method (OLS/ISD-1/ElasticNet + LRT) detection vs amplitude on the
   progressive mocks.

The per-sample (NSIDE 64) and per-NSIDE (fiducial sample) numbers are tabulated in
``_static/detectability_law/ls10_detectability_scorecard.csv``.

Empirical sweep on the remote (Stage 2 — run)
---------------------------------------------

The analytic curves above are anchored at the *measured* operating points. They
have been traced empirically over the full ``nside × density × f_sky × amplitude``
grid by a controlled sweep, which **has been run** on the compute host (14 400
LS10-geometry fits, 8 100 Euclid-geometry fits, 240 MCMC-add anchors, 30
simulations per cell). The fitted exponents, and the two levers that do *not*
come out at :math:`-1/2`, are reported on the :doc:`survey_design_synthesis`
page; this page is unaffected by that run, since its numbers come from the
measured LS10 weight maps rather than from mocks.

.. code-block:: bash

   # dry-run / input check only (zero compute):
   python scripts/run_detectability_sweep.py --check
   bash bash/run_remote_full.sh check
   # the run itself (compute host; ~7 core-h at the committed defaults):
   bash bash/run_remote_full.sh sweep_ls10 sweep_euclid mcmc_anchors
   # continue after a wall-clock timeout:
   RESUME=1 bash bash/run_remote_full.sh sweep_euclid

See ``scripts/run_detectability_sweep.py`` for the knobs (it adds the missing
``--fsky`` footprint-fraction dial) and ``bash/run_remote_full.sh`` for the
staged recipe and resource budget.

Reproduce
---------

.. code-block:: bash

   python scripts/analyze_detectability_law.py
   bash bash/build_docs.sh
