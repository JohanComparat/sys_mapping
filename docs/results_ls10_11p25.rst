.. _sample-11p25:

BGS VLIM log M\ :sub:`*` ≥ 11.25,  z < 0.35 — detailed systematic analysis
===============================================================================

High-mass BGS VLIM sample (541,855 galaxies, z < 0.35).  Shot noise dominates per-pixel statistics at both resolutions.  The LRT rejects the additive null at NSIDE 64 (:math:`\lambda_{\rm LR} = 123.4`).

.. contents:: On this page
   :local:
   :depth: 1

.. seealso::

   :doc:`results_ls10` — summary tables and figures for all nine samples.

----

Sample statistics
-----------------

.. csv-table::
   :header: "Parameter", "NSIDE 32", "NSIDE 64", "NSIDE 128", "NSIDE 256"
   :widths: 36, 16, 16, 16, 16

   "Stellar-mass threshold", "log M* ≥ 11.25", "log M* ≥ 11.25", "log M* ≥ 11.25", "log M* ≥ 11.25"
   "Redshift limit", "z < 0.35", "z < 0.35", "z < 0.35", "z < 0.35"
   "N\ :sub:`gal`", "541,855", "541,855", "541,855", "541,855"
   "N\ :sub:`pix` (good footprint)", "5,609", "21,555", "84,131", "325,324"
   "N\ :sub:`templates`", "11", "11", "11", "11"
   "MCMC walkers", "210", "210", "210", "210"
   "MCMC steps after burn-in", "1500", "1500", "1500", "1500"

----

Goodness-of-fit: :math:`\hat{\sigma}` by method and resolution
------------------------------------------------------------------

The noise parameter :math:`\hat{\sigma}` measures residual scatter after systematic subtraction — lower is better.  Results are shown for NSIDE 32, 64, 128, and 256.  ISD-3 is unavailable at NSIDE 128 and 256 (no partial files generated at those resolutions).

.. csv-table::
   :header: "Method", ":math:`\hat{\sigma}` (N32)", ":math:`\hat{\sigma}` (N64)", ":math:`\hat{\sigma}` (N128)", ":math:`\hat{\sigma}` (N256)", "Notes"
   :widths: 14, 12, 12, 12, 12, 38

   "OLS", "0.3308", "0.3842", "0.6405", "1.2602", "closed-form least-squares"
   "ElasticNet", "0.3319", "0.3846", "0.6406", "1.2603", "L1+L2 regularised; 3-fold CV"
   "ISD-1", "0.3308", "0.3842", "0.6405", "1.2602", "iterative self-calibration; poly order 1"
   "ISD-3 †", "0.7031", "0.6196", "0.6454", "1.3131", "† degree-3 polynomial; unavailable at NSIDE 128/256"
   "MCMC-add", "0.3313", "0.3843", "0.6406", "1.2602", "MCMC additive; acc N32=0.386 N64=0.389"
   "MCMC-comb", "0.4062", "0.3930", "0.6329", "1.2117", "MCMC combined; acc N32=0.293 N64=0.288"

† ISD-3 uses a degree-3 polynomial expansion and is ill-conditioned with correlated templates.

Likelihood Ratio Test (additive vs combined model)
--------------------------------------------------

.. csv-table::
   :header: "Resolution", ":math:`\lambda_{\rm LR}`", "dof", "p-value", "Reject H₀"
   :widths: 14, 14, 8, 22, 12

   "NSIDE 32", "613.5", "11", "< 10\ :sup:`-124`", "**Yes**"
   "NSIDE 64", "123.4", "11", "< 10\ :sup:`-21`", "**Yes**"
   "NSIDE 128", "140.8", "11", "< 10\ :sup:`-24`", "**Yes**"
   "NSIDE 256", "597.3", "11", "< 10\ :sup:`-121`", "**Yes**"

MCMC acceptance fractions:
NSIDE 32: add 0.386, comb 0.293  NSIDE 64: add 0.389, comb 0.288  NSIDE 128: add 0.387, comb 0.287  NSIDE 256: add 0.390, comb 0.301.  Healthy range: 0.15–0.50.

----

Template amplitude ranking — additive model (MCMC-add, NSIDE 64)
------------------------------------------------------------------

All 11 templates sorted by absolute MCMC-add additive amplitude :math:`|\hat{a}_i|`.  OLS shown for comparison.

.. csv-table::
   :header: "Rank", "Template", "NSIDE", ":math:`\hat{a}_i` (MCMC-add)", ":math:`\hat{a}_i` (OLS)", "Physical meaning"
   :widths: 5, 28, 6, 14, 14, 50
   :stub-columns: 1

   "1", "**GAIA_nstar_faint**", "64", "+0.2755", "+0.2785", "GAIA faint stellar density (photometric mis-classification of faint stars as galaxies)"
   "2", "**GAIA_nstar_medium**", "64", "-0.1944", "-0.1970", "GAIA medium stellar density (crowding and deblending near bright stars)"
   "3", "**GAIA_phot_g_mean_flux**", "64", "+0.0530", "+0.0522", "GAIA mean stellar flux in G band (scattered-light / sky-background variations)"
   "4", "**GAIA_phot_rp_mean_flux**", "64", "-0.0511", "-0.0504", "GAIA mean stellar flux in RP band (red scattered light)"
   "5", "**GAIA_phot_bp_mean_flux**", "64", "-0.0334", "-0.0331", "GAIA mean stellar flux in BP band (blue scattered light)"
   "6", "LS10_GALDEPTH_G_NSIDE_0064", "?", "+0.0132", "+0.0131", "LS10_GALDEPTH_G_NSIDE_0064"
   "7", "LS10_PSFSIZE_R_NSIDE_0064", "?", "-0.0079", "-0.0079", "LS10_PSFSIZE_R_NSIDE_0064"
   "8", "LS10_GALDEPTH_Z_NSIDE_0064", "?", "+0.0054", "+0.0054", "LS10_GALDEPTH_Z_NSIDE_0064"
   "9", "LS10_NOBS_R_NSIDE_0064", "?", "+0.0044", "+0.0045", "LS10_NOBS_R_NSIDE_0064"
   "10", "LS10_GALDEPTH_R_NSIDE_0064", "?", "+0.0034", "+0.0034", "LS10_GALDEPTH_R_NSIDE_0064"
   "11", "LS10_EBV_NSIDE_0064", "?", "-0.0029", "-0.0028", "LS10_EBV_NSIDE_0064"

----

Template amplitude ranking — multiplicative model (MCMC-comb, NSIDE 64)
--------------------------------------------------------------------------

All 11 templates sorted by absolute MCMC-comb multiplicative amplitude :math:`|\hat{b}_i|`.

.. csv-table::
   :header: "Rank", "Template", "NSIDE", ":math:`\hat{b}_i` (MCMC-comb)", "Physical meaning"
   :widths: 5, 30, 6, 16, 50
   :stub-columns: 1

   "1", "**GAIA_nstar_faint**", "64", "+0.3944", "GAIA faint stellar density (photometric mis-classification of faint stars as galaxies)"
   "2", "**GAIA_nstar_medium**", "64", "-0.2422", "GAIA medium stellar density (crowding and deblending near bright stars)"
   "3", "**LS10_EBV_NSIDE_0064**", "?", "+0.0513", "LS10_EBV_NSIDE_0064"
   "4", "**GAIA_phot_bp_mean_flux**", "64", "-0.0343", "GAIA mean stellar flux in BP band (blue scattered light)"
   "5", "**GAIA_phot_g_mean_flux**", "64", "+0.0288", "GAIA mean stellar flux in G band (scattered-light / sky-background variations)"
   "6", "GAIA_phot_rp_mean_flux", "64", "-0.0168", "GAIA mean stellar flux in RP band (red scattered light)"
   "7", "LS10_GALDEPTH_G_NSIDE_0064", "?", "+0.0137", "LS10_GALDEPTH_G_NSIDE_0064"
   "8", "LS10_GALDEPTH_R_NSIDE_0064", "?", "+0.0098", "LS10_GALDEPTH_R_NSIDE_0064"
   "9", "LS10_GALDEPTH_Z_NSIDE_0064", "?", "-0.0063", "LS10_GALDEPTH_Z_NSIDE_0064"
   "10", "LS10_NOBS_R_NSIDE_0064", "?", "+0.0054", "LS10_NOBS_R_NSIDE_0064"
   "11", "LS10_PSFSIZE_R_NSIDE_0064", "?", "-0.0016", "LS10_PSFSIZE_R_NSIDE_0064"

----

Per-galaxy weight statistics (NSIDE 64)
----------------------------------------

From the ``*_NSIDE0064_WEIGHTS.fits`` file.  Mean ≈ 1 and small std indicate a well-behaved weight distribution.

.. csv-table::
   :header: "Method", "N", "mean", "std", "p1", "p5", "p50", "p95", "p99"
   :widths: 14, 12, 8, 8, 8, 8, 8, 8, 8

   "OLS", "541,855", "0.9914", "0.0352", "0.8993", "0.9443", "0.9892", "1.0495", "1.0841"
   "ElasticNet", "541,855", "0.9906", "0.0252", "0.9133", "0.9564", "0.9927", "1.0248", "1.0468"
   "ISD-1", "541,855", "0.9914", "0.0355", "0.8980", "0.9439", "0.9893", "1.0500", "1.0844"
   "ISD-3 †", "541,855", "1.8450", "9.0095", "0.7238", "0.8593", "0.9633", "1.2506", "5.9101"
   "MCMC-add", "541,855", "0.9914", "0.0352", "0.8990", "0.9445", "0.9892", "1.0493", "1.0843"
   "MCMC-comb", "541,855", "1.0292", "0.0470", "0.9159", "0.9462", "1.0363", "1.0932", "1.1294"

----

Systematic weight maps
----------------------

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Mollweide weight maps — all six methods</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0032_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0032_weight_map.png" style="width:100%" alt="weight maps NSIDE 32 weight maps">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32 weight maps</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0064_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0064_weight_map.png" style="width:100%" alt="weight maps NSIDE 64 weight maps">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64 weight maps</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0128_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0128_weight_map.png" style="width:100%" alt="weight maps NSIDE 128 weight maps">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128 weight maps</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0256_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0256_weight_map.png" style="width:100%" alt="weight maps NSIDE 256 weight maps">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 256 weight maps</figcaption>
     </figure>
   </div>


Systematic weight distributions
-------------------------------

Narrow peaks near 1 indicate stable weight estimates.  ElasticNet weights may be exactly 1 when cross-validation selects zero amplitudes.

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Per-galaxy weight distributions — all six methods</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0032_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0032_weight_hist.png" style="width:100%" alt="weight distributions NSIDE 32 weight distributions">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32 weight distributions</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0064_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0064_weight_hist.png" style="width:100%" alt="weight distributions NSIDE 64 weight distributions">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64 weight distributions</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0128_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0128_weight_hist.png" style="width:100%" alt="weight distributions NSIDE 128 weight distributions">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128 weight distributions</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0256_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0256_weight_hist.png" style="width:100%" alt="weight distributions NSIDE 256 weight distributions">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 256 weight distributions</figcaption>
     </figure>
   </div>


Angular clustering w(θ) before and after correction
-----------------------------------------------------

Each panel shows the observed angular two-point correlation function (solid black) and the corrected :math:`w(\theta)` for all six methods.  A well-corrected sample shows suppressed excess clustering at all scales.  Each panel corresponds to one map resolution.

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">w(θ): observed vs corrected — all six methods</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0032_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0032_wtheta.png" style="width:100%" alt="wtheta NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0064_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0064_wtheta.png" style="width:100%" alt="wtheta NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0128_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0128_wtheta.png" style="width:100%" alt="wtheta NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0256_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0256_wtheta.png" style="width:100%" alt="wtheta NSIDE 256">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 256</figcaption>
     </figure>
   </div>


----

Cosmological analysis verdict
-----------------------------

Sub-degree scales (:math:`\theta < 30'`): regime is **moderately contaminated** (:math:`\delta w/w \approx +2.2\%` at 30 arcmin).

* Without correction: **borderline** without correction.
* After correction: **suitable** after applying ``WEIGHT_COMB``.
* **Large-angle warning** (:math:`\theta > 2°`): max correction 17.4% at 181 arcmin — any analysis using angular scales > 1° **must** apply ``WEIGHT_COMB``.

LRT (NSIDE 64): :math:`\lambda_{\rm LR} = 123.4` (dof = 11), p = 3.7e-21 → **Reject H₀** — multiplicative contamination is statistically detected.

**Recommendation**: use ``WEIGHT_COMB`` (``WEIGHT_SYS``) for all science-grade analyses.

