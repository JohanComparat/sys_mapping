Systematic-detectability law
============================

This page gives the smallest systematic in galaxy density that a survey of given
sky coverage, pixel size and galaxy density can detect, and at what significance.
The law is analytic; its normalisation comes from the LS10 fit outputs and the
calibrated significances. The GLASS-mock sweep that tests the exponents is on
:doc:`survey_design_synthesis`.

The law
-------

We model the overdensity per pixel as a systematic field plus noise,
:math:`\delta_g = f + \varepsilon`, with :math:`f=\sum_i a_i t_i` for templates of
zero mean and unit variance. The per-pixel noise variance :math:`\hat\sigma^2` is the
residual scatter of the fit, shot noise plus clustering:
:math:`\hat\sigma^2 = 1/\bar n_{\rm pix} + \sigma_{\rm clus}^2`. With
:math:`N_{\rm gal}=\bar n_{\rm pix} N_{\rm pix}` and field rms :math:`A={\rm rms}(f)`,

.. math::

   {\rm SNR}_{\rm field} = \frac{\lVert f\rVert}{\hat\sigma}
   = A\,\frac{\sqrt{N_{\rm pix}}}{\hat\sigma} = A\sqrt{N_{\rm eff}},
   \qquad
   A_{\min}(\nu\sigma) = \nu\,\frac{\hat\sigma}{\sqrt{N_{\rm pix}}},
   \qquad
   N_{\rm eff}\equiv\frac{N_{\rm pix}}{\hat\sigma^2}\le N_{\rm gal}.

In the shot-noise limit this reduces to :math:`A_{\min}=\nu/\sqrt{N_{\rm gal}}`.
The field statistic does not depend on how collinear the templates are, whereas the
amplitude of template :math:`i` carries the variance-inflation factor
:math:`{\rm VIF}_i=1/\sqrt{1-R_i^2}`; the standardised LS10 basis at NSIDE 64 has
condition number 108.

Rules of thumb
--------------

* In the shot regime, :math:`\bar n_{\rm pix}<1/\sigma_{\rm clus}^2`,
  :math:`A_{\min}\propto 1/\sqrt{N_{\rm gal}}`. In the clustering regime it
  saturates at a floor set by the number of independent modes,
  :math:`\propto 1/\sqrt{f_{\rm sky}}`, which more area lowers and more depth does not.
* At fixed :math:`N_{\rm gal}`, :math:`A_{\min}=\nu\hat\sigma/\sqrt{N_{\rm pix}}` falls
  as pixels refine, down to the shot floor :math:`\nu/\sqrt{N_{\rm gal}}`. Pixels
  finer than the coherence scale of the systematic add nothing.
* The independent-pixel error :math:`\sigma_i` ignores the correlation of the
  clustered field between pixels. Against 400 uncontaminated GLASS
  realisations carrying each sample's matched spectrum, the median over templates of
  the amplitude scatter divided by :math:`\sigma_i` is 1.3 to 3.0 per
  sample and rises with resolution (NSIDE 16: 1.3; NSIDE 32: 1.8; NSIDE 64: 1.8 to 2.4; NSIDE 128: 2.6 to 3.0). Single templates span
  0.9 to 5.6. ``sys_mapping.calibrated_template_significance`` measures
  the factor per template.
* A search over :math:`n_{\rm sys}` templates reports the largest significance, so
  its p-value is read from the largest significance of each null realisation (the
  family-wise p).
* The contamination of :math:`w(\theta)` grows as :math:`A^2`, so the field
  regression detects fainter systematics than :math:`w(\theta)` does.

LS10 worked example (log :math:`M_*\ge` 10.0)
------------------------------------------------

The fiducial sample has :math:`N_{\rm gal}=`\ 2,759,238 and
:math:`f_{\rm sky}\approx`\ 0.441, and is clustering-limited. At NSIDE 64 the per-pixel noise
is :math:`\hat\sigma=`\ 0.397 for :math:`\bar n_{\rm pix}=`\ 127 (shot term
0.0079), so :math:`N_{\rm eff}/N_{\rm gal}=`\ 0.050. The smallest detectable field
rms is :math:`A_{\min}(3\sigma)=`\ 8.09e-03 (:math:`5\sigma`: 1.35e-02), against a
shot floor of 1.81e-03. The leading template at NSIDE 64 is LS10_GALDEPTH_R, at
independent-pixel SNR 6.8.

Occupancy puts this sample at NSIDE 128 (32.5 galaxies per pixel,
floor 25). There the leading template is GAIA_phot_rp_mean_flux, at calibrated significance
4.02 (independent-pixel 4.2, inflation 1.03) and family-wise
:math:`p` ≤ 0.0025 from 400 realisations. It traces the Gaia stellar density.

.. csv-table:: Calibrated significance per sample, each at the resolution its occupancy supports.
   The family-wise p is bounded below by 1/(N+1) for N realisations.
   :file: _static/detectability_law/ls10_calibrated_significance.csv
   :header-rows: 1

.. figure:: /_static/detectability_law/fig1_Amin_vs_Ngal.png
   :width: 88%

   Smallest detectable systematic against galaxy count for the nine LS10
   stellar-mass samples at NSIDE 64. At fixed footprint
   :math:`A_{\min}\propto\hat\sigma`, so the intermediate-mass samples, with the
   lowest :math:`\hat\sigma`, are the most sensitive.

.. figure:: /_static/detectability_law/fig2_Amin_vs_nside.png
   :width: 88%

   :math:`A_{\min}` against NSIDE (32 to 128) for the fiducial sample, with the shot
   floor.

.. figure:: /_static/detectability_law/fig3_crossover.png
   :width: 88%

   Shot and clustering per-pixel variance. LS10 is clustering-limited at every NSIDE
   tested.

.. figure:: /_static/detectability_law/fig4_Neff_fraction.png
   :width: 88%

   :math:`N_{\rm eff}/N_{\rm gal}`, equal to 1 for pure shot noise.

.. figure:: /_static/detectability_law/fig5_per_template_snr.png
   :width: 92%

   Per-template significance of the fiducial sample at NSIDE 128,
   independent-pixel against calibrated on matched realisations.

.. figure:: /_static/detectability_law/fig6_detection_vs_amplitude.png
   :width: 88%

   Detection fraction against injected amplitude for OLS, ISD-1, ElasticNet and the
   likelihood-ratio test on the progressive mocks.

The per-sample (NSIDE 64) and per-NSIDE (fiducial sample) numbers are in
``_static/detectability_law/ls10_detectability_scorecard.csv``, the calibrated
significances in ``_static/detectability_law/ls10_calibrated_significance.csv``.

Reproduce
---------

.. code-block:: bash

   python scripts/analyze_detectability_law.py
   bash bash/build_docs.sh

The GLASS-mock sweep over ``nside × density × f_sky × amplitude`` is run with
``scripts/run_detectability_sweep.py`` (``--check`` validates the inputs; ``--fskys``
sets the footprint fractions) and staged by ``bash/run_remote_full.sh``:

.. code-block:: bash

   bash bash/run_remote_full.sh check
   bash bash/run_remote_full.sh sweep_ls10 sweep_euclid mcmc_anchors
   RESUME=1 bash bash/run_remote_full.sh sweep_euclid
