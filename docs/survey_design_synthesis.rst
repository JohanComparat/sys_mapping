Survey design and the detectability law
=======================================

This page turns the LS10 worked example of :doc:`detectability_law` into rules of
thumb for dimensioning a survey, and compares the scaling laws with a GLASS-mock
sweep.

Where LS10 sits in the design space
-----------------------------------

The law :math:`A_{\min}(\nu\sigma)=\nu\hat\sigma/\sqrt{N_{\rm pix}}`, with
:math:`N_{\rm eff}=N_{\rm pix}/\hat\sigma^2`, places a survey by its measured
per-pixel scatter and its pixel count.

.. list-table:: Design-space scorecard (fiducial configuration)
   :header-rows: 1
   :widths: 26 8 8 8 16 20 12

   * - survey
     - NSIDE
     - :math:`f_{\rm sky}`
     - :math:`\bar n_{\rm pix}`
     - :math:`N_{\rm gal}`
     - limiting factor
     - :math:`A_{\min}(3\sigma)`
   * - LS10 (log$M_*\geq$10.0)
     - 64
     - 0.44
     - 127
     - 2,759,238
     - clustering (area-limited)
     - 8.1e-03

LS10 is clustering-limited at every NSIDE in the scorecard. At NSIDE 64 the
per-pixel scatter :math:`\hat\sigma=0.397` of the fiducial sample is 4.5 times the
shot term :math:`1/\sqrt{\bar n_{\rm pix}}`, so :math:`N_{\rm eff}/N_{\rm gal}=0.05`.

.. figure:: /_static/survey_design_synthesis/fig1_master_Amin_vs_Ngal.png
   :width: 90%

   Smallest detectable systematic against galaxy count for the LS10 samples at
   NSIDE 64 (:math:`4.5\text{–}13.8\times10^{-3}`). At fixed :math:`N_{\rm gal}` the
   scatter follows :math:`\hat\sigma`: :math:`\log M_*\geq11.0` reaches
   :math:`6.1\times10^{-3}` on 1.6 M galaxies with :math:`\hat\sigma=0.297`.

.. figure:: /_static/survey_design_synthesis/fig2_Amin_vs_fsky.png
   :width: 90%

   In the clustering regime :math:`A_{\min}\propto f_{\rm sky}^{-1/2}`. LS10 covers
   :math:`f_{\rm sky}=0.44`.

.. figure:: /_static/survey_design_synthesis/fig3_design_space.png
   :width: 90%

   :math:`N_{\rm eff}/N_{\rm gal}` across pixel size and density; 1 is pure shot
   noise.

Rules of thumb
--------------

#. Sensitivity is set by :math:`N_{\rm eff}=N_{\rm pix}/\hat\sigma^2`. Measure
   :math:`\hat\sigma` as the residual scatter of the fit; then
   :math:`A_{\min}(\nu\sigma)=\nu\hat\sigma/\sqrt{N_{\rm pix}}`.
#. In the shot regime, :math:`\bar n_{\rm pix}\lesssim1/\sigma_{\rm clus}^{2}`
   (about 7 for LS10 at NSIDE 64), :math:`A_{\min}\propto1/\sqrt{N_{\rm gal}}` and
   depth helps. At higher density :math:`A_{\min}` saturates at a floor
   :math:`\propto1/\sqrt{f_{\rm sky}}` and area helps.
#. Refine pixels down to the shot floor or the coherence scale of the systematic,
   whichever is reached first. Pixels coarser than the systematic average its
   signal away.
#. Detect the field rather than individual templates. The field statistic is
   insensitive to collinearity, while individual templates are poorly identified
   (the standardised LS10 basis on the fiducial footprint at NSIDE 64 has second-moment
   condition number 108). Calibrate per-template significance on matched mocks with
   ``sys_mapping.calibrated_template_significance``: on LS10 the independent-pixel
   error is short by a median factor of 1.3 to 2.9 per sample, rising with
   resolution, and by 0.9 to 5.6 for single templates.
#. The contamination of :math:`w(\theta)` grows as :math:`A^2` and the field
   regression signal as :math:`A`, so the field regression detects fainter
   systematics.

Dimensioning a run
------------------

* A wide, shallow survey such as LS10 is area-limited: it reaches the smallest
  amplitudes because area adds modes, and more depth adds little.
* A deep, narrow survey lowers its :math:`A_{\min}` floor fastest by enlarging the
  footprint. Its fine pixels already sample the available modes.
* At NSIDE 64 the fiducial LS10 sample reaches :math:`A_{\min}(3\sigma)=8.1\times10^{-3}`
  field rms; across all samples and resolutions the range is :math:`4.3\times10^{-3}`
  to :math:`1.5\times10^{-2}`. The systematics in the NSIDE-64 fiducial weight map
  have rms 3.2 % for the OLS weights and 4.9 % for the combined model, about
  :math:`12\sigma`.

GLASS-mock sweep
----------------

We ran ``run_detectability_sweep.py`` (OLS and ISD-1 on GLASS mocks) over the
``nside × density × f_sky × amplitude`` grid: 14 400 LS10-geometry fits, 8 100
Euclid-geometry fits and 240 MCMC-add anchor fits, with 30 simulations per cell.
The exponents are fitted in log-log at injected amplitude :math:`A=0.05`.

Pixel refinement gives :math:`-0.469` (LS10 geometry) and :math:`-0.479` (Euclid
geometry), and sky area :math:`-0.493`, against the analytic :math:`-1/2`.
Density gives :math:`-0.375` over the full range. Fitting
:math:`A_{\min}\propto\sqrt{1/\bar n_{\rm pix}+\sigma_{\rm clus}^{2}}` to the density
lever gives :math:`\sigma_{\rm clus}=0.063`, against 0.064 measured on the GLASS
map at NSIDE 256; clustering supplies 34 % of the pixel variance at
:math:`\bar n_{\rm pix}=127` and 67 % at 490.

Panel (a) holds :math:`\bar n_{\rm pix}` fixed, so :math:`N_{\rm gal}\propto N_{\rm pix}`
along it. Raising NSIDE while lowering the density so that :math:`N_{\rm gal}` stays
at :math:`2.7\times10^{6}` gives a slope of :math:`-0.055`, on the shot floor
:math:`3/\sqrt{N_{\rm gal}}=1.8\times10^{-3}`.

.. figure:: /_static/survey_design_synthesis/fig4_sweep_validation.png
   :width: 98%

   Fitted exponents from the sweep. (a) and (b) with the analytic
   :math:`\propto x^{-1/2}` reference (dashed); (c) with pure shot noise (dashed) and
   the shot-plus-clustering curve (green); (d) at fixed :math:`N_{\rm gal}`, with the
   shot floor.

.. caution::

   The fitted exponents depend on the injected amplitude. ``field_snr`` is built
   from :math:`\lVert T\hat a\rVert`, which is biased high at low signal, so the
   same :math:`f_{\rm sky}` slice fits :math:`-0.388` at :math:`A=0.005` and
   :math:`-0.493` at :math:`A=0.05`.

   The mock per-pixel scatter at :math:`\bar n_{\rm pix}=127`, NSIDE 64 is
   :math:`\hat\sigma\approx0.10`, against 0.397 measured on LS10, so mock
   :math:`A_{\min}` values are about four times lower than the survey reaches. The
   survey numbers are the scorecard values above.

The MCMC-add anchors agree with OLS to better than 0.1 % in all
240 shared cells. For the additive Gaussian model the posterior mean is
the OLS solution, so this is a consistency check.

Reproduce
---------

.. code-block:: bash

   python scripts/make_survey_design_synthesis.py
   bash bash/build_docs.sh
