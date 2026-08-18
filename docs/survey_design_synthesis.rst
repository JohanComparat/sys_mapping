Survey design & the detectability law
=====================================

.. note::
   Takes the LS10 worked example (:doc:`detectability_law`) into generalised
   rules of thumb and a design-space picture, then checks the scaling laws
   against the Stage-2 GLASS-mock sweep. Interpretive; no new fit.

Where LS10 sits in the design space
-----------------------------------

The law :math:`A_{\min}(\nu\sigma)=\nu\hat\sigma/\sqrt{N_{\rm pix}}`
(:math:`N_{\rm eff}=N_{\rm pix}/\hat\sigma^2`) places a survey by its measured
per-pixel scatter and its pixel count:

.. list-table:: Design-space scorecard (fiducial configurations)
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

LS10 is **clustering- (cosmic-variance-) limited** on the real sky: the measured
per-pixel scatter :math:`\hat\sigma\sim0.4\text{–}0.8` exceeds the shot term
:math:`1/\sqrt{\bar n_{\rm pix}}` by :math:`4.5\times` in RMS at the fiducial
NSIDE 64, so only a small fraction :math:`N_{\rm eff}/N_{\rm gal}` of the
galaxies count toward detection. It sits **deep in the clustering regime** —
wide, dense, coarse pixels — at every NSIDE in the scorecard.

.. figure:: /_static/survey_design_synthesis/fig1_master_Amin_vs_Ngal.png
   :width: 90%

   Smallest detectable systematic vs galaxy count across the LS10 samples
   (:math:`4.5\text{–}13.8\times10^{-3}`). Scatter at fixed
   :math:`N_{\rm gal}` is driven by :math:`\hat\sigma`, not by counts:
   :math:`\log M_*\geq11.0` reaches :math:`6.1\times10^{-3}` on 1.6 M galaxies
   thanks to an unusually low :math:`\hat\sigma=0.297`, better than samples
   twice its size.

.. figure:: /_static/survey_design_synthesis/fig2_Amin_vs_fsky.png
   :width: 90%

   In the clustering regime :math:`A_{\min}\propto f_{\rm sky}^{-1/2}`: **more
   area, not more depth, lowers the floor.** LS10 already exploits most of the
   available sky, so its remaining gain from area is modest.

.. figure:: /_static/survey_design_synthesis/fig3_design_space.png
   :width: 90%

   The usable fraction :math:`N_{\rm eff}/N_{\rm gal}` across pixel size and
   density; 1 is the pure-shot ceiling.

Generalised rules of thumb
--------------------------

#. **Sensitivity is set by** :math:`N_{\rm eff}=N_{\rm pix}/\hat\sigma^2`, not
   :math:`N_{\rm gal}`. Measure :math:`\hat\sigma` (the fit residual scatter);
   then :math:`A_{\min}(\nu\sigma)=\nu\hat\sigma/\sqrt{N_{\rm pix}}`.
#. **Shot regime** (:math:`\bar n_{\rm pix}\lesssim1/\sigma_{\rm clus}^{2}`, which is
   :math:`\approx7` for LS10 at NSIDE 64 — not "a few tens"):
   :math:`A_{\min}\propto1/\sqrt{N_{\rm gal}}` — add galaxies/depth.
   **Clustering regime** (denser): :math:`A_{\min}` saturates at a floor
   :math:`\propto1/\sqrt{f_{\rm sky}}` — add area.
#. **Pixel size:** refine until the shot floor or the systematic's coherence
   scale, whichever comes first; finer resolves more modes but buys nothing once
   shot-limited, and coarser than the systematic washes out its signal.
#. **Detect the field, not the map:** the field statistic is VIF-free; individual
   collinear templates (cond :math:`\sim10^8` for the LS10 basis) are not
   identifiable. Calibrate per-template SNR with the sandwich (iid is
   :math:`\sim2\times` optimistic).
#. **Use the field regression, not** :math:`w(\theta)`, **to detect:** the
   :math:`w(\theta)` contamination signal grows as :math:`A^2` while the linear
   field regression grows as :math:`A`, so the field regression detects far
   fainter systematics at the amplitudes actually present.

Bottom line for dimensioning a run
----------------------------------

* **A wide shallow survey (LS10-like)** detects the *smallest amplitudes* because
  area buys modes; it is already area-limited, so deeper imaging barely helps —
  push :math:`f_{\rm sky}`.
* **A deep narrow survey** is limited by area; at fixed depth its
  :math:`A_{\min}` floor drops fastest by enlarging the footprint. Its fine
  pixels already extract the available modes, so coarser pixelisation would not
  hurt and finer would not help.
* At its fiducial configuration LS10 reaches
  :math:`A_{\min}(3\sigma)=8.1\times10^{-3}` field RMS; across all samples and
  resolutions the range is :math:`4.3\times10^{-3}` to :math:`1.5\times10^{-2}`,
  so it is not uniformly sub-percent. That comfortably detects the real
  systematics present in the data, whose amplitude is 3.2 % RMS for the
  additive/OLS weights and 4.9 % for the combined model (measured from the
  NSIDE-64 fiducial weight map) — about :math:`12\sigma`.

Empirical validation (Stage 2 — RUN, full remote grid)
--------------------------------------------------------

The sweep (``run_detectability_sweep.py``, OLS/ISD-1 on GLASS mocks) has now been
run over the **full** ``nside × density × f_sky × amplitude`` grid on the compute
host: 14 400 LS10-geometry fits, 8 100 Euclid-geometry fits and 240 MCMC-add
anchor fits, 30 simulations per cell. Exponents below are fitted in log-log,
read at injected amplitude :math:`A=0.05`.

**Two of the three levers reproduce the analytic law.** Pixel refinement gives
:math:`-0.469` (LS10) and :math:`-0.479` (Euclid), and sky area gives
:math:`-0.493` — both consistent with :math:`-1/2`.

**The density lever does not, and that is the interesting part.** Over the full
range it fits :math:`-0.375`, not :math:`-1/2`, because the mock is *not*
shot-limited at high density: fitting
:math:`A_{\min}\propto\sqrt{1/\bar n_{\rm pix}+\sigma_{\rm clus}^{2}}` recovers
:math:`\sigma_{\rm clus}=0.063`, which matches the toy GLASS spectrum's own
per-pixel clustering scatter at NSIDE 256 (0.064 measured directly from the map).
Clustering supplies 34 % of the pixel variance at :math:`\bar n_{\rm pix}=127`
and 67 % at 490, so the lever flattens exactly where it should. The sweep
therefore *measures* the clustering floor rather than merely assuming it.

**Pixelisation on its own buys nothing.** Panel (a) holds
:math:`\bar n_{\rm pix}` fixed, so :math:`N_{\rm gal}\propto N_{\rm pix}` and the
two are degenerate along it. Breaking the degeneracy — raising NSIDE while
lowering the density so :math:`N_{\rm gal}` stays at :math:`2.7\times10^{6}` —
gives a slope of :math:`-0.055`, i.e. flat, sitting on the shot floor
:math:`3/\sqrt{N_{\rm gal}}=1.8\times10^{-3}`. In the shot regime
:math:`A_{\min}` is set by the galaxy count alone.

.. figure:: /_static/survey_design_synthesis/fig4_sweep_validation.png
   :width: 98%

   Stage-2 sweep with **fitted** exponents. (a) and (b) follow the analytic
   :math:`\propto x^{-1/2}` reference (dashed); (c) departs from pure shot noise
   (dashed) and is tracked instead by the shot+clustering curve (green); (d) is
   flat at the shot floor. GLASS mocks — a scaling diagnostic, not a survey
   forecast.

.. caution::

   Two caveats a reader should carry away from this figure.

   *The fitted exponents depend on the injected amplitude.* ``field_snr`` is
   built from :math:`\lVert T\hat a\rVert`, which is biased upward at low signal,
   so the same :math:`f_{\rm sky}` slice fits :math:`-0.388` at :math:`A=0.005`
   and :math:`-0.493` at :math:`A=0.05`. The panels are read at :math:`A=0.05`,
   where the bias has died away; quoting the law at the detection threshold
   itself would understate every exponent.

   *These are mock numbers, not forecasts.* At matched
   :math:`\bar n_{\rm pix}=127`, NSIDE 64 the mock's per-pixel scatter is
   :math:`\hat\sigma\approx0.10` against the measured LS10 :math:`0.397`, so its
   :math:`A_{\min}` is a factor :math:`\sim4` lower than anything the real survey
   can reach. The real-survey numbers are the scorecard values above.

The MCMC-add anchors ran (240 fits) and agree with OLS to better than 0.1 %
in every one of the 240 shared cells. That is a consistency check rather than an independent measurement:
for the additive Gaussian model the posterior mean *is* the OLS solution, so
``MCMC-add`` cannot pin the floor any more tightly than ``OLS`` already does.

Reproduce
---------

.. code-block:: bash

   python scripts/make_survey_design_synthesis.py
   bash bash/build_docs.sh
