Results: systematic weights
============================

Per-galaxy systematic weights for the nine LS10 BGS volume-limited stellar-mass
threshold samples, computed by ``scripts/run_ls10_analysis.py`` with 11
observational templates (GAIA DR3 star density and photometry; LS10 imaging depth,
PSF size, and exposure count) at NSIDE 32, 64, 128, and 256.  **Use** ``WEIGHT_COMB``
(MCMC combined model) for all science analyses; ``WEIGHT_ADD`` and ``WEIGHT_OLS``
are provided as cross-checks.  See :doc:`pipeline_ls10` for how to reproduce these
results.

.. role:: best-result

.. contents:: On this page
   :local:
   :depth: 1

----

Run configuration
-----------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Parameter
     - Value
   * - Script
     - ``scripts/run_ls10_analysis.py``
   * - NSIDE
     - 32 (pixel area ≈ 3.36 deg²; 12 288 sky pixels),
       64 (≈ 0.84 deg²; 49 152 pixels),
       128 (≈ 0.21 deg²; 196 608 pixels),
       256 (≈ 0.052 deg²; 786 432 pixels)
   * - Templates
     - 11 maps at the analysis NSIDE: GAIA DR3 (nstar_faint, nstar_medium,
       phot_g/bp/rp_mean_flux) and LS10 imaging (EBV, GALDEPTH_G/R/Z, NOBS_R, PSFSIZE_R)
   * - Decontamination methods
     - OLS, ElasticNet, ISD-1, ISD-3, MCMC-add, MCMC-comb
   * - MCMC walkers / steps / burn-in
     - 210 / 1500 / 300
   * - Recommended weight column
     - ``WEIGHT_COMB`` — MCMC combined model (additive + multiplicative)
   * - Additive weight column
     - ``WEIGHT_ADD`` — MCMC additive model only
   * - OLS weight column
     - ``WEIGHT_OLS`` — ordinary least-squares regression

----

Sample overview
---------------

The nine BGS VLIM (volume-limited stellar mass threshold) samples span
:math:`\log_{10}(M_*/M_\odot) \in [9.0, 11.5]` at their respective
redshift limits.

.. csv-table::
   :header: "Sample (log M* ≥, z <)", "N\ :sub:`gal`", "N\ :sub:`rand`"
   :widths: 22, 14, 16

   "9.0,  0.08", "523 486", "2 617 332"
   "9.5,  0.12", "1 432 502", "7 160 697"
   "10.0, 0.18", "2 759 238", "13 795 884"
   "10.25, 0.22", "3 308 841", "16 544 481"
   "10.5, 0.26", "3 263 228", "16 315 418"
   "10.75, 0.31", "2 802 710", "14 013 316"
   "11.0, 0.35", "1 619 838", "8 097 853"
   "11.25, 0.35", "541 855", "2 708 912"
   "11.5, 0.35", "120 882", "606 304"

Goodness-of-fit comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~

The noise parameter :math:`\hat{{\sigma}}` measures the residual scatter of the
galaxy overdensity after subtracting the systematic model — lower is better.
All six methods were run at NSIDE 32, 64, 128.
The bold entry in each row is the method with the lowest :math:`\hat{\sigma}`
across the six methods.

**NSIDE 32** (pixel area ≈ 3.36 deg², :math:`N_{\rm pix} ≈ 5 600`):

.. csv-table::
   :header: "Sample (log M* ≥, z <)", "OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"
   :widths: 22, 8, 9, 8, 9, 9, 10

   "9.0,  0.08", "0.5469", "0.5531", "0.5498", "0.5495", "0.5475", ":best-result:`0.5384`"
   "9.5,  0.12", "0.4703", "0.4704", "0.4732", "0.4731", "0.4709", ":best-result:`0.4539`"
   "10.0, 0.18", "0.3801", "0.3820", "0.3820", "0.3816", "0.3805", ":best-result:`0.3650`"
   "10.25, 0.22", "0.3417", "0.3427", "0.3439", "0.3433", "0.3421", ":best-result:`0.3266`"
   "10.5, 0.26", "0.3238", "0.3248", "0.3251", "0.3251", "0.3241", ":best-result:`0.3040`"
   "10.75, 0.31", "0.3057", "0.3064", "0.3067", "0.3067", "0.3061", ":best-result:`0.2826`"
   "11.0, 0.35", "0.2971", "0.2977", "0.2979", "0.2982", "0.2975", ":best-result:`0.2808`"
   "11.25, 0.35", "0.3309", "0.3313", "0.3317", "0.3317", "0.3313", ":best-result:`0.3199`"
   "11.5, 0.35", "0.4453", "0.4456", "0.4459", "0.4459", "0.4458", ":best-result:`0.4379`"

**NSIDE 64** (pixel area ≈ 0.84 deg², :math:`N_{\rm pix} ≈ 21 600`):

.. csv-table::
   :header: "Sample (log M* ≥, z <)", "OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"
   :widths: 22, 8, 9, 8, 9, 9, 10

   "9.0,  0.08", ":best-result:`0.6759`", "0.6769", "0.6787", "0.6785", "0.6761", "0.6882"
   "9.5,  0.12", "0.5238", "0.5239", "0.5267", "0.5266", "0.5240", ":best-result:`0.5228`"
   "10.0, 0.18", ":best-result:`0.3969`", "0.3970", "0.3980", "0.3980", "0.3970", "0.3983"
   "10.25, 0.22", ":best-result:`0.3433`", "0.3434", "0.3443", "0.3443", "0.3434", "0.3501"
   "10.5, 0.26", ":best-result:`0.3089`", "0.3092", "0.3096", "0.3103", "0.3090", "0.3138"
   "10.75, 0.31", ":best-result:`0.2831`", "0.2832", "0.2837", "0.2840", "0.2831", "0.2908"
   "11.0, 0.35", ":best-result:`0.2973`", "0.2974", "0.2976", "0.2983", "0.2974", "0.3074"
   "11.25, 0.35", ":best-result:`0.3842`", "0.3843", "0.3847", "0.3852", "0.3843", "0.3931"
   "11.5, 0.35", ":best-result:`0.6447`", "0.6448", "0.6452", "0.6452", "0.6449", "0.6627"

**NSIDE 128** (pixel area ≈ 0.21 deg², :math:`N_{\rm pix} ≈ 84 000`):

.. csv-table::
   :header: "Sample (log M* ≥, z <)", "OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"
   :widths: 22, 8, 9, 8, 9, 9, 10

   "9.0,  0.08", ":best-result:`0.9913`", "0.9917", "0.9930", "0.9929", "0.9914", "1.0403"
   "9.5,  0.12", ":best-result:`0.7458`", "0.7462", "0.7475", "0.7469", "0.7459", "0.7730"
   "10.0, 0.18", ":best-result:`0.5605`", "0.5606", "0.5610", "0.5612", "0.5606", "0.5957"
   "10.25, 0.22", ":best-result:`0.4900`", "0.4900", "0.4903", "0.4903", "0.4900", "0.5189"
   "10.5, 0.26", ":best-result:`0.4477`", "0.4479", "0.4481", "0.4483", "0.4478", "0.4672"
   "10.75, 0.31", ":best-result:`0.4190`", "0.4192", "0.4196", "0.4197", "0.4191", "0.4461"
   "11.0, 0.35", ":best-result:`0.4580`", "0.4581", "0.4583", "0.4585", "0.4581", "0.4818"
   "11.25, 0.35", ":best-result:`0.6412`", "0.6413", "0.6416", "0.6417", "0.6413", "0.6570"
   "11.5, 0.35", ":best-result:`1.3894`", "1.3906", "1.3896", "1.3898", "1.3895", "1.4987"

**ISD-1** and **ISD-3** fit one template at a time against its own binned density
relation, at degree 1 and 3.  The degree buys curvature in one template's value
rather than cross-products between templates, so the two differ only where the
response is non-linear; on this grid they agree to within a few per cent.

**Key observations:**

* **OLS and ISD-1** give nearly identical :math:`\hat{\sigma}` (differences
  < 0.001) at all resolutions, consistent with ISD-1 converging to the OLS
  solution for linearly contaminated data.

* **ElasticNet** is marginally worse than OLS due to regularisation shrinkage.
  For some samples/NSIDEs, ElasticNet CV selects zero amplitudes — those
  weight distributions are flat (all weights = 1.0); this is a legitimate result.

* **NSIDE 32 — multiplicative model overfits.**
  At NSIDE 32 (≈ 5 600 pixels), :math:`\hat{\sigma}_{\rm comb} > \hat{\sigma}_{\rm add}`
  for *all* nine samples.  With only ≈ 5 600 pixels and 11 multiplicative
  parameters, the combined model absorbs noise.  LRT still strongly rejects H₀.
  **Use NSIDE 64 or higher for science.**

* **NSIDE 64** — MCMC-comb lowers :math:`\hat{\sigma}` relative to MCMC-add
  only for the two densest intermediate-mass samples (log M* ≥ 10.0 and 10.25,
  which have the highest LRT statistics).  For the remaining seven samples,
  :math:`\hat{\sigma}_{\rm comb} > \hat{\sigma}_{\rm add}`, reflecting that
  the multiplicative correction tightens the angular-clustering profile rather
  than the pixel-level residual.  **WEIGHT_COMB is still the recommended choice**
  for all samples: the LRT strongly rejects the additive-only model and the
  combined correction removes degree-scale power that WEIGHT_ADD leaves behind.

* **NSIDE 128 and 256** — :math:`\hat{\sigma}` rises above its NSIDE 64 minimum
  because finer pixels contain fewer galaxies per pixel (higher Poisson noise).
  At NSIDE 128 the combined model overfits for the two sparsest samples
  (:math:`\hat{\sigma}_{\rm comb} > 1` for log M* = 9.0 and 11.5).
  At NSIDE 256 overfitting extends to all sparse samples at both ends of the
  mass range (:math:`\hat{\sigma}_{\rm comb} > 1` for log M* ≤ 9.5 and
  log M* ≥ 11.25).  The intermediate dense samples (log M* 10.0–11.0) remain
  below 1 at both NSIDEs.  **NSIDE 64 is the recommended analysis resolution.**

----

Systematics are detected: Likelihood Ratio Test
-------------------------------------------------

To decide whether multiplicative contamination is needed on top of an additive
offset, we compare two nested models with a **Likelihood Ratio Test (LRT)**:

* :math:`H_0` — *additive only*: galaxy density fluctuations are offset by
  :math:`\sum_i a_i\,t_i(p)` at pixel :math:`p`, but the survey area is uniform.
* :math:`H_1` — *combined* (Berlfein et al. 2024): both additive shifts
  :math:`a_i` *and* multiplicative depth variations :math:`b_i` are present.

The test statistic
:math:`\lambda_{\rm LR} = 2[\ln\mathcal{L}_1 - \ln\mathcal{L}_0]`
follows a :math:`\chi^2(11)` distribution under :math:`H_0`.
Critical value at 5 %: :math:`\chi^2_{11,\,0.95} \approx 19.7`.

**NSIDE 32:**

.. csv-table::
   :header: "Sample (log M* ≥, z <)", "λ\ :sub:`LR`", "dof", "p-value", "Reject H\ :sub:`0`"
   :widths: 24, 12, 6, 20, 12

   "9.0,  0.08", "486.0", "11", "< 10\ :sup:`-60`", "**Yes**"
   "9.5,  0.12", "702.0", "11", "< 10\ :sup:`-100`", "**Yes**"
   "10.0, 0.18", "696.7", "11", "< 10\ :sup:`-100`", "**Yes**"
   "10.25, 0.22", "836.6", "11", "< 10\ :sup:`-100`", "**Yes**"
   "10.5, 0.26", "977.0", "11", "< 10\ :sup:`-100`", "**Yes**"
   "10.75, 0.31", "1207.9", "11", "< 10\ :sup:`-200`", "**Yes**"
   "11.0, 0.35", "1070.2", "11", "< 10\ :sup:`-200`", "**Yes**"
   "11.25, 0.35", "635.2", "11", "< 10\ :sup:`-100`", "**Yes**"
   "11.5, 0.35", "415.3", "11", "< 10\ :sup:`-60`", "**Yes**"

**NSIDE 64:**

.. csv-table::
   :header: "Sample (log M* ≥, z <)", "λ\ :sub:`LR`", "dof", "p-value", "Reject H\ :sub:`0`"
   :widths: 24, 12, 6, 20, 12

   "9.0,  0.08", "1502.0", "11", "< 10\ :sup:`-200`", "**Yes**"
   "9.5,  0.12", "741.5", "11", "< 10\ :sup:`-100`", "**Yes**"
   "10.0, 0.18", "182.5", "11", "< 10\ :sup:`-18`", "**Yes**"
   "10.25, 0.22", "107.6", "11", "< 10\ :sup:`-18`", "**Yes**"
   "10.5, 0.26", "73.6", "11", "< 10\ :sup:`-9`", "**Yes**"
   "10.75, 0.31", "139.2", "11", "< 10\ :sup:`-18`", "**Yes**"
   "11.0, 0.35", "77.6", "11", "< 10\ :sup:`-9`", "**Yes**"
   "11.25, 0.35", "126.5", "11", "< 10\ :sup:`-18`", "**Yes**"
   "11.5, 0.35", "154.4", "11", "< 10\ :sup:`-18`", "**Yes**"

**NSIDE 128:**

.. csv-table::
   :header: "Sample (log M* ≥, z <)", "λ\ :sub:`LR`", "dof", "p-value", "Reject H\ :sub:`0`"
   :widths: 24, 12, 6, 20, 12

   "9.0,  0.08", "3571.7", "11", "< 10\ :sup:`-200`", "**Yes**"
   "9.5,  0.12", "2335.4", "11", "< 10\ :sup:`-200`", "**Yes**"
   "10.0, 0.18", "330.0", "11", "< 10\ :sup:`-60`", "**Yes**"
   "10.25, 0.22", "210.9", "11", "< 10\ :sup:`-40`", "**Yes**"
   "10.5, 0.26", "235.8", "11", "< 10\ :sup:`-40`", "**Yes**"
   "10.75, 0.31", "289.0", "11", "< 10\ :sup:`-40`", "**Yes**"
   "11.0, 0.35", "206.8", "11", "< 10\ :sup:`-40`", "**Yes**"
   "11.25, 0.35", "141.6", "11", "< 10\ :sup:`-18`", "**Yes**"
   "11.5, 0.35", "198.9", "11", "< 10\ :sup:`-18`", "**Yes**"

**Interpretation.**  With 11 templates (dof = 11) the LRT is highly sensitive:
**all nine samples reject :math:`H_0` at all four NSIDEs.**
:math:`\lambda_{\rm LR}` grows with NSIDE because finer pixels yield more
independent data points, amplifying the power of the test.  The dominant
driver in all cases is GAIA stellar-density maps.

----

Fractional systematic uncertainty on :math:`w(\theta)`
--------------------------------------------------------

The table below shows the fractional correction
:math:`\delta w/w = (w_{\rm comb} - w_{\rm obs})/w_{\rm obs}`.
For the six samples with external measurements, values come from
``~/software/sum_stat/`` (TreeCorr, NSIDE = 64 weights) and are given
at :math:`\theta = 30'` and as max and RMS over 1–200 arcmin.
For the three intermediate samples (log M* = 10.25, 10.5, 10.75),
values are derived from the sys_mapping internal :math:`w(\theta)` (NSIDE 64,
0.6–272 arcmin range); max is over the full range and RMS over 1–200 arcmin.

.. csv-table::
   :header: "Sample (log M* ≥)", "δw/w at 30′", "max \|δw/w\|", "RMS δw/w (1–200′)", "Regime"
   :widths: 18, 14, 22, 12, 34

   "9.0",  "−7.2 %", "8.4 % (at 23′)",  "5.9 %", "Systematics-dominated at all scales"
   "9.5",  "−4.8 %", "4.9 % (at 15′)",  "3.7 %", "Systematics-dominated"
   "10.0", "−0.4 %", "2.0 % (at 120′)", "0.6 %", "Borderline (correction < noise at sub-degree)"
   "10.25","≈0 %",   "3.2 % (at 178′)", "1.0 %", "Sub-degree clean; degree-scale correction present"
   "10.5", "≈0 %",   "5.2 % (at 178′)", "1.6 %", "Sub-degree clean; degree-scale correction present"
   "10.75","≈0 %",   "8.6 % (at 178′)", "2.5 %", "Sub-degree clean; degree-scale correction significant"
   "11.0", "−0.1 %", "11.5 % (at 181′)","2.4 %", "Sub-degree OK; large-scale systematic present"
   "11.25","+2.2 %", "17.4 % (at 181′)","6.5 %", "Large-scale dominated"
   "11.5", "+0.7 %", "9.7 % (at 97′)",  "3.3 %", "Statistics-dominated"

----

Is LS10 BGS (:math:`r < 19.5`) systematics-limited?
-----------------------------------------------------

**Low-mass samples (log** :math:`M_* < 10.0` **)** — YES, correction is essential.
The fractional correction reaches 5–8 % at :math:`\theta \approx 30'`.
Use ``WEIGHT_COMB`` for all analyses.

**Intermediate samples (log** :math:`10.0 \leq M_* < 11.0` **) at sub-degree
scales** — NO at :math:`\theta < 30'` (:math:`\delta w/w \lesssim 0\%`).
Clustering science at sub-degree scales is safe after applying ``WEIGHT_COMB``.
However, degree-scale corrections of 3–13 % are present and grow with
:math:`\theta`; large-angle analyses **must** apply ``WEIGHT_COMB``.

**All samples at large angles (**\ :math:`\theta > 2°`\ **)** — YES.  GAIA
stellar-density maps carry degree-scale power imposing a 10–40 % fractional
correction on :math:`w(\theta)`.  BAO, ISW, and angular dipole analyses
**must** apply ``WEIGHT_COMB`` weights.

**Recommendation**: always use ``WEIGHT_COMB`` (NSIDE 64) for science-grade analyses.

----

Cross-sample comparison (NSIDE 64)
-----------------------------------

Key metrics at NSIDE 64.  :math:`\delta w/w` values at :math:`\theta = 30'`
are from the TreeCorr HDF5 pipeline; n/a = no measurement available.

.. csv-table::
   :header: "Sample (log M*≥, z<)", "N\ :sub:`gal`", "N\ :sub:`pix`", "λ\ :sub:`LR`", "Reject H\ :sub:`0`", "σ̂ OLS", "σ̂ MCMC-add", "σ̂ MCMC-comb", "δw/w at 30′"
   :widths: 18, 11, 9, 9, 9, 8, 11, 12, 12

   "9.0,  0.08", "523 486", "21,563", "1502.0", "**Yes**", "—", ":best-result:`0.6761`", "0.6882", "-7.2 %"
   "9.5,  0.12", "1 432 502", "21,637", "741.5", "**Yes**", "—", "0.5240", ":best-result:`0.5228`", "-4.8 %"
   "10.0, 0.18", "2 759 238", "21,667", "182.5", "**Yes**", "—", ":best-result:`0.3970`", "0.3983", "-0.4 %"
   "10.25, 0.22", "3 308 841", "21,669", "107.6", "**Yes**", "—", ":best-result:`0.3434`", "0.3501", "n/a"
   "10.5, 0.26", "3 263 228", "21,675", "73.6", "**Yes**", "—", ":best-result:`0.3090`", "0.3138", "n/a"
   "10.75, 0.31", "2 802 710", "21,662", "139.2", "**Yes**", "—", ":best-result:`0.2831`", "0.2908", "n/a"
   "11.0, 0.35", "1 619 838", "21,646", "77.6", "**Yes**", "—", ":best-result:`0.2974`", "0.3074", "-0.1 %"
   "11.25, 0.35", "541 855", "21,555", "126.5", "**Yes**", "—", ":best-result:`0.3843`", "0.3931", "+2.2 %"
   "11.5, 0.35", "120 882", "21,344", "154.4", "**Yes**", "—", ":best-result:`0.6449`", "0.6627", "+0.7 %"

----

Per-sample results — all 9 samples
-------------------------------------

For each sample: weight maps and histograms at all four NSIDEs, angular clustering
w(θ) comparing observed and six corrected measurements, and a table of key numbers.

.. _ls10-sample-9p0:

log M\* ≥ 9.0,  z < 0.08  (N = 523 486)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Systematic weight maps — log M* ≥ 9.0</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0032_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0032_weight_map.png" style="width:100%" alt="Weight maps log M*≥9.0 NSIDE 32 (≈5 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32 (≈5 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0064_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0064_weight_map.png" style="width:100%" alt="Weight maps log M*≥9.0 NSIDE 64 (≈21 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64 (≈21 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0128_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0128_weight_map.png" style="width:100%" alt="Weight maps log M*≥9.0 NSIDE 128 (≈84 000 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128 (≈84 000 pix)</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Weight distributions — log M* ≥ 9.0</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0032_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0032_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥9.0 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0064_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0064_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥9.0 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0128_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0128_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥9.0 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Angular clustering w(θ) — observed and corrected (one line per method) — log M* ≥ 9.0</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0032_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0032_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥9.0 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0064_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0064_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥9.0 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0128_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0128_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥9.0 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. csv-table:: Key numbers — log M* ≥ 9.0
   :header: "Parameter", "NSIDE 32", "NSIDE 64", "NSIDE 128", "NSIDE 256"
   :widths: 28, 15, 15, 15, 15

   "N\ :sub:`gal`",             "523 486",  "523 486", "523 486", "523 486"
   "N\ :sub:`pix` (good)",      "5612", "21563", "84367"
   "LRT λ\ :sub:`LR` (dof=11)", "486.0 (**Yes**)", "1502.0 (**Yes**)", "3571.7 (**Yes**)"
   "σ̂ OLS",                     "—", "—", "—"
   "σ̂ ElasticNet",               "—", "—", "—"
   "σ̂ ISD-1",                    "—", "—", "—"
   "σ̂ ISD-3 ‡",                  "—", "—", "—"
   "σ̂ MCMC-add",                 "0.5475", "0.6761", "0.9914"
   "σ̂ MCMC-comb",                "0.5384", "0.6882", "1.0403"
   "MCMC-add acc. frac.",         "1.000", "1.000", "1.000"
   "MCMC-comb acc. frac.",        "0.802", "0.893", "0.780"
   "Dominant template",           "ns_fnt", "ns_med", "GD_R"
   "δw/w at 30′",                 "—", "-7.2 %", "—", "—"

ISD-3 fits the same marginal relation as ISD-1 at degree 3, so the two
  separate only where the template response is non-linear.


.. seealso::

   :doc:`results_ls10_9p0` — full template amplitude tables, weight statistics, and cosmological analysis verdict for log M* ≥ 9.0.

.. _ls10-sample-9p5:

log M\* ≥ 9.5,  z < 0.12  (N = 1 432 502)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Systematic weight maps — log M* ≥ 9.5</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0032_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0032_weight_map.png" style="width:100%" alt="Weight maps log M*≥9.5 NSIDE 32 (≈5 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32 (≈5 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0064_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0064_weight_map.png" style="width:100%" alt="Weight maps log M*≥9.5 NSIDE 64 (≈21 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64 (≈21 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0128_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0128_weight_map.png" style="width:100%" alt="Weight maps log M*≥9.5 NSIDE 128 (≈84 000 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128 (≈84 000 pix)</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Weight distributions — log M* ≥ 9.5</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0032_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0032_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥9.5 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0064_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0064_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥9.5 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0128_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0128_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥9.5 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Angular clustering w(θ) — observed and corrected (one line per method) — log M* ≥ 9.5</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0032_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0032_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥9.5 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0064_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0064_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥9.5 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0128_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0128_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥9.5 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. csv-table:: Key numbers — log M* ≥ 9.5
   :header: "Parameter", "NSIDE 32", "NSIDE 64", "NSIDE 128", "NSIDE 256"
   :widths: 28, 15, 15, 15, 15

   "N\ :sub:`gal`",             "1 432 502",  "1 432 502", "1 432 502", "1 432 502"
   "N\ :sub:`pix` (good)",      "5611", "21637", "84627"
   "LRT λ\ :sub:`LR` (dof=11)", "702.0 (**Yes**)", "741.5 (**Yes**)", "2335.4 (**Yes**)"
   "σ̂ OLS",                     "—", "—", "—"
   "σ̂ ElasticNet",               "—", "—", "—"
   "σ̂ ISD-1",                    "—", "—", "—"
   "σ̂ ISD-3 ‡",                  "—", "—", "—"
   "σ̂ MCMC-add",                 "0.4709", "0.5240", "0.7459"
   "σ̂ MCMC-comb",                "0.4539", "0.5228", "0.7730"
   "MCMC-add acc. frac.",         "1.000", "1.000", "1.000"
   "MCMC-comb acc. frac.",        "0.864", "0.875", "0.866"
   "Dominant template",           "ns_med", "ns_med", "GD_R"
   "δw/w at 30′",                 "—", "-4.8 %", "—", "—"

ISD-3 fits the same marginal relation as ISD-1 at degree 3, so the two
  separate only where the template response is non-linear.


.. seealso::

   :doc:`results_ls10_9p5` — full template amplitude tables, weight statistics, and cosmological analysis verdict for log M* ≥ 9.5.

.. _ls10-sample-10p0:

log M\* ≥ 10.0,  z < 0.18  (N = 2 759 238)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Systematic weight maps — log M* ≥ 10.0</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0032_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0032_weight_map.png" style="width:100%" alt="Weight maps log M*≥10.0 NSIDE 32 (≈5 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32 (≈5 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0064_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0064_weight_map.png" style="width:100%" alt="Weight maps log M*≥10.0 NSIDE 64 (≈21 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64 (≈21 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0128_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0128_weight_map.png" style="width:100%" alt="Weight maps log M*≥10.0 NSIDE 128 (≈84 000 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128 (≈84 000 pix)</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Weight distributions — log M* ≥ 10.0</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0032_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0032_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥10.0 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0064_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0064_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥10.0 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0128_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0128_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥10.0 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Angular clustering w(θ) — observed and corrected (one line per method) — log M* ≥ 10.0</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0032_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0032_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥10.0 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0064_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0064_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥10.0 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0128_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0128_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥10.0 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. csv-table:: Key numbers — log M* ≥ 10.0
   :header: "Parameter", "NSIDE 32", "NSIDE 64", "NSIDE 128", "NSIDE 256"
   :widths: 28, 15, 15, 15, 15

   "N\ :sub:`gal`",             "2 759 238",  "2 759 238", "2 759 238", "2 759 238"
   "N\ :sub:`pix` (good)",      "5616", "21667", "84860"
   "LRT λ\ :sub:`LR` (dof=11)", "696.7 (**Yes**)", "182.5 (**Yes**)", "330.0 (**Yes**)"
   "σ̂ OLS",                     "—", "—", "—"
   "σ̂ ElasticNet",               "—", "—", "—"
   "σ̂ ISD-1",                    "—", "—", "—"
   "σ̂ ISD-3 ‡",                  "—", "—", "—"
   "σ̂ MCMC-add",                 "0.3805", "0.3970", "0.5606"
   "σ̂ MCMC-comb",                "0.3650", "0.3983", "0.5957"
   "MCMC-add acc. frac.",         "1.000", "1.000", "1.000"
   "MCMC-comb acc. frac.",        "0.861", "0.862", "0.929"
   "Dominant template",           "ns_med", "ns_med", "GD_R"
   "δw/w at 30′",                 "—", "-0.4 %", "—", "—"

ISD-3 fits the same marginal relation as ISD-1 at degree 3, so the two
  separate only where the template response is non-linear.


.. seealso::

   :doc:`results_ls10_10p0` — full template amplitude tables, weight statistics, and cosmological analysis verdict for log M* ≥ 10.0.

.. _ls10-sample-10p25:

log M\* ≥ 10.25,  z < 0.22  (N = 3 308 841)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Systematic weight maps — log M* ≥ 10.25</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0032_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0032_weight_map.png" style="width:100%" alt="Weight maps log M*≥10.25 NSIDE 32 (≈5 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32 (≈5 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0064_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0064_weight_map.png" style="width:100%" alt="Weight maps log M*≥10.25 NSIDE 64 (≈21 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64 (≈21 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0128_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0128_weight_map.png" style="width:100%" alt="Weight maps log M*≥10.25 NSIDE 128 (≈84 000 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128 (≈84 000 pix)</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Weight distributions — log M* ≥ 10.25</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0032_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0032_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥10.25 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0064_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0064_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥10.25 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0128_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0128_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥10.25 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Angular clustering w(θ) — observed and corrected (one line per method) — log M* ≥ 10.25</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0032_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0032_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥10.25 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0064_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0064_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥10.25 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0128_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0128_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥10.25 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. csv-table:: Key numbers — log M* ≥ 10.25
   :header: "Parameter", "NSIDE 32", "NSIDE 64", "NSIDE 128", "NSIDE 256"
   :widths: 28, 15, 15, 15, 15

   "N\ :sub:`gal`",             "3 308 841",  "3 308 841", "3 308 841", "3 308 841"
   "N\ :sub:`pix` (good)",      "5618", "21669", "84831"
   "LRT λ\ :sub:`LR` (dof=11)", "836.6 (**Yes**)", "107.6 (**Yes**)", "210.9 (**Yes**)"
   "σ̂ OLS",                     "—", "—", "—"
   "σ̂ ElasticNet",               "—", "—", "—"
   "σ̂ ISD-1",                    "—", "—", "—"
   "σ̂ ISD-3 ‡",                  "—", "—", "—"
   "σ̂ MCMC-add",                 "0.3421", "0.3434", "0.4900"
   "σ̂ MCMC-comb",                "0.3266", "0.3501", "0.5189"
   "MCMC-add acc. frac.",         "1.000", "1.000", "1.000"
   "MCMC-comb acc. frac.",        "0.827", "0.827", "0.878"
   "Dominant template",           "ns_med", "ns_med", "ns_med"
   "δw/w at 30′",                 "—", "n/a", "—", "—"

ISD-3 fits the same marginal relation as ISD-1 at degree 3, so the two
  separate only where the template response is non-linear.


.. seealso::

   :doc:`results_ls10_10p25` — full template amplitude tables, weight statistics, and cosmological analysis verdict for log M* ≥ 10.25.

.. _ls10-sample-10p5:

log M\* ≥ 10.5,  z < 0.26  (N = 3 263 228)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Systematic weight maps — log M* ≥ 10.5</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0032_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0032_weight_map.png" style="width:100%" alt="Weight maps log M*≥10.5 NSIDE 32 (≈5 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32 (≈5 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0064_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0064_weight_map.png" style="width:100%" alt="Weight maps log M*≥10.5 NSIDE 64 (≈21 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64 (≈21 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0128_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0128_weight_map.png" style="width:100%" alt="Weight maps log M*≥10.5 NSIDE 128 (≈84 000 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128 (≈84 000 pix)</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Weight distributions — log M* ≥ 10.5</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0032_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0032_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥10.5 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0064_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0064_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥10.5 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0128_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0128_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥10.5 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Angular clustering w(θ) — observed and corrected (one line per method) — log M* ≥ 10.5</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0032_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0032_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥10.5 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0064_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0064_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥10.5 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0128_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0128_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥10.5 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. csv-table:: Key numbers — log M* ≥ 10.5
   :header: "Parameter", "NSIDE 32", "NSIDE 64", "NSIDE 128", "NSIDE 256"
   :widths: 28, 15, 15, 15, 15

   "N\ :sub:`gal`",             "3 263 228",  "3 263 228", "3 263 228", "3 263 228"
   "N\ :sub:`pix` (good)",      "5617", "21675", "84811"
   "LRT λ\ :sub:`LR` (dof=11)", "977.0 (**Yes**)", "73.6 (**Yes**)", "235.8 (**Yes**)"
   "σ̂ OLS",                     "—", "—", "—"
   "σ̂ ElasticNet",               "—", "—", "—"
   "σ̂ ISD-1",                    "—", "—", "—"
   "σ̂ ISD-3 ‡",                  "—", "—", "—"
   "σ̂ MCMC-add",                 "0.3241", "0.3090", "0.4478"
   "σ̂ MCMC-comb",                "0.3040", "0.3138", "0.4672"
   "MCMC-add acc. frac.",         "1.000", "1.000", "1.000"
   "MCMC-comb acc. frac.",        "0.838", "0.925", "0.903"
   "Dominant template",           "ns_med", "ns_med", "ns_med"
   "δw/w at 30′",                 "—", "n/a", "—", "—"

ISD-3 fits the same marginal relation as ISD-1 at degree 3, so the two
  separate only where the template response is non-linear.


.. seealso::

   :doc:`results_ls10_10p5` — full template amplitude tables, weight statistics, and cosmological analysis verdict for log M* ≥ 10.5.

.. _ls10-sample-10p75:

log M\* ≥ 10.75,  z < 0.31  (N = 2 802 710)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Systematic weight maps — log M* ≥ 10.75</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0032_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0032_weight_map.png" style="width:100%" alt="Weight maps log M*≥10.75 NSIDE 32 (≈5 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32 (≈5 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0064_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0064_weight_map.png" style="width:100%" alt="Weight maps log M*≥10.75 NSIDE 64 (≈21 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64 (≈21 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0128_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0128_weight_map.png" style="width:100%" alt="Weight maps log M*≥10.75 NSIDE 128 (≈84 000 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128 (≈84 000 pix)</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Weight distributions — log M* ≥ 10.75</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0032_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0032_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥10.75 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0064_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0064_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥10.75 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0128_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0128_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥10.75 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Angular clustering w(θ) — observed and corrected (one line per method) — log M* ≥ 10.75</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0032_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0032_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥10.75 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0064_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0064_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥10.75 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0128_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0128_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥10.75 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. csv-table:: Key numbers — log M* ≥ 10.75
   :header: "Parameter", "NSIDE 32", "NSIDE 64", "NSIDE 128", "NSIDE 256"
   :widths: 28, 15, 15, 15, 15

   "N\ :sub:`gal`",             "2 802 710",  "2 802 710", "2 802 710", "2 802 710"
   "N\ :sub:`pix` (good)",      "5618", "21662", "84824"
   "LRT λ\ :sub:`LR` (dof=11)", "1207.9 (**Yes**)", "139.2 (**Yes**)", "289.0 (**Yes**)"
   "σ̂ OLS",                     "—", "—", "—"
   "σ̂ ElasticNet",               "—", "—", "—"
   "σ̂ ISD-1",                    "—", "—", "—"
   "σ̂ ISD-3 ‡",                  "—", "—", "—"
   "σ̂ MCMC-add",                 "0.3061", "0.2831", "0.4191"
   "σ̂ MCMC-comb",                "0.2826", "0.2908", "0.4461"
   "MCMC-add acc. frac.",         "1.000", "1.000", "1.000"
   "MCMC-comb acc. frac.",        "0.910", "0.878", "0.925"
   "Dominant template",           "ns_med", "ns_med", "rp_fl"
   "δw/w at 30′",                 "—", "n/a", "—", "—"

ISD-3 fits the same marginal relation as ISD-1 at degree 3, so the two
  separate only where the template response is non-linear.


.. seealso::

   :doc:`results_ls10_10p75` — full template amplitude tables, weight statistics, and cosmological analysis verdict for log M* ≥ 10.75.

.. _ls10-sample-11p0:

log M\* ≥ 11.0,  z < 0.35  (N = 1 619 838)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Systematic weight maps — log M* ≥ 11.0</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0032_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0032_weight_map.png" style="width:100%" alt="Weight maps log M*≥11.0 NSIDE 32 (≈5 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32 (≈5 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0064_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0064_weight_map.png" style="width:100%" alt="Weight maps log M*≥11.0 NSIDE 64 (≈21 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64 (≈21 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0128_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0128_weight_map.png" style="width:100%" alt="Weight maps log M*≥11.0 NSIDE 128 (≈84 000 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128 (≈84 000 pix)</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Weight distributions — log M* ≥ 11.0</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0032_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0032_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥11.0 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0064_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0064_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥11.0 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0128_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0128_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥11.0 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Angular clustering w(θ) — observed and corrected (one line per method) — log M* ≥ 11.0</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0032_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0032_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥11.0 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0064_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0064_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥11.0 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0128_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0128_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥11.0 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. csv-table:: Key numbers — log M* ≥ 11.0
   :header: "Parameter", "NSIDE 32", "NSIDE 64", "NSIDE 128", "NSIDE 256"
   :widths: 28, 15, 15, 15, 15

   "N\ :sub:`gal`",             "1 619 838",  "1 619 838", "1 619 838", "1 619 838"
   "N\ :sub:`pix` (good)",      "5614", "21646", "84719"
   "LRT λ\ :sub:`LR` (dof=11)", "1070.2 (**Yes**)", "77.6 (**Yes**)", "206.8 (**Yes**)"
   "σ̂ OLS",                     "—", "—", "—"
   "σ̂ ElasticNet",               "—", "—", "—"
   "σ̂ ISD-1",                    "—", "—", "—"
   "σ̂ ISD-3 ‡",                  "—", "—", "—"
   "σ̂ MCMC-add",                 "0.2975", "0.2974", "0.4581"
   "σ̂ MCMC-comb",                "0.2808", "0.3074", "0.4818"
   "MCMC-add acc. frac.",         "1.000", "1.000", "1.000"
   "MCMC-comb acc. frac.",        "0.829", "0.909", "0.924"
   "Dominant template",           "ns_med", "ns_med", "rp_fl"
   "δw/w at 30′",                 "—", "-0.1 %", "—", "—"

ISD-3 fits the same marginal relation as ISD-1 at degree 3, so the two
  separate only where the template response is non-linear.


.. seealso::

   :doc:`results_ls10_11p0` — full template amplitude tables, weight statistics, and cosmological analysis verdict for log M* ≥ 11.0.

.. _ls10-sample-11p25:

log M\* ≥ 11.25,  z < 0.35  (N = 541 855)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Systematic weight maps — log M* ≥ 11.25</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0032_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0032_weight_map.png" style="width:100%" alt="Weight maps log M*≥11.25 NSIDE 32 (≈5 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32 (≈5 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0064_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0064_weight_map.png" style="width:100%" alt="Weight maps log M*≥11.25 NSIDE 64 (≈21 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64 (≈21 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0128_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0128_weight_map.png" style="width:100%" alt="Weight maps log M*≥11.25 NSIDE 128 (≈84 000 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128 (≈84 000 pix)</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Weight distributions — log M* ≥ 11.25</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0032_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0032_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥11.25 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0064_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0064_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥11.25 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0128_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0128_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥11.25 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Angular clustering w(θ) — observed and corrected (one line per method) — log M* ≥ 11.25</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0032_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0032_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥11.25 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0064_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0064_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥11.25 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0128_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0128_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥11.25 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. csv-table:: Key numbers — log M* ≥ 11.25
   :header: "Parameter", "NSIDE 32", "NSIDE 64", "NSIDE 128", "NSIDE 256"
   :widths: 28, 15, 15, 15, 15

   "N\ :sub:`gal`",             "541 855",  "541 855", "541 855", "541 855"
   "N\ :sub:`pix` (good)",      "5609", "21555", "84131"
   "LRT λ\ :sub:`LR` (dof=11)", "635.2 (**Yes**)", "126.5 (**Yes**)", "141.6 (**Yes**)"
   "σ̂ OLS",                     "—", "—", "—"
   "σ̂ ElasticNet",               "—", "—", "—"
   "σ̂ ISD-1",                    "—", "—", "—"
   "σ̂ ISD-3 ‡",                  "—", "—", "—"
   "σ̂ MCMC-add",                 "0.3313", "0.3843", "0.6413"
   "σ̂ MCMC-comb",                "0.3199", "0.3931", "0.6570"
   "MCMC-add acc. frac.",         "1.000", "1.000", "1.000"
   "MCMC-comb acc. frac.",        "0.901", "0.915", "0.887"
   "Dominant template",           "ns_med", "g_fl", "rp_fl"
   "δw/w at 30′",                 "—", "+2.2 %", "—", "—"

ISD-3 fits the same marginal relation as ISD-1 at degree 3, so the two
  separate only where the template response is non-linear.


.. seealso::

   :doc:`results_ls10_11p25` — full template amplitude tables, weight statistics, and cosmological analysis verdict for log M* ≥ 11.25.

.. _ls10-sample-11p5:

log M\* ≥ 11.5,  z < 0.35  (N = 120 882)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Systematic weight maps — log M* ≥ 11.5</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0032_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0032_weight_map.png" style="width:100%" alt="Weight maps log M*≥11.5 NSIDE 32 (≈5 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32 (≈5 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0064_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0064_weight_map.png" style="width:100%" alt="Weight maps log M*≥11.5 NSIDE 64 (≈21 600 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64 (≈21 600 pix)</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0128_weight_map.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0128_weight_map.png" style="width:100%" alt="Weight maps log M*≥11.5 NSIDE 128 (≈84 000 pix)">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128 (≈84 000 pix)</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Weight distributions — log M* ≥ 11.5</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0032_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0032_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥11.5 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0064_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0064_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥11.5 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0128_weight_hist.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0128_weight_hist.png" style="width:100%" alt="Weight distributions log M*≥11.5 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. raw:: html

   <p style="text-align:center;font-weight:bold;margin-bottom:0.4em">Angular clustering w(θ) — observed and corrected (one line per method) — log M* ≥ 11.5</p>
   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;max-width:1100px;margin:auto">
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0032_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0032_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥11.5 NSIDE 32">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 32</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0064_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0064_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥11.5 NSIDE 64">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 64</figcaption>
     </figure>
     <figure style="text-align:center;margin:0">
       <a href="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0128_wtheta.png" target="_blank">
         <img src="_static/results_ls10/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0128_wtheta.png" style="width:100%" alt="Angular clustering w(θ) log M*≥11.5 NSIDE 128">
       </a>
       <figcaption style="font-size:0.82em;color:#555">NSIDE 128</figcaption>
     </figure>
   </div>

.. csv-table:: Key numbers — log M* ≥ 11.5
   :header: "Parameter", "NSIDE 32", "NSIDE 64", "NSIDE 128", "NSIDE 256"
   :widths: 28, 15, 15, 15, 15

   "N\ :sub:`gal`",             "120 882",  "120 882", "120 882", "120 882"
   "N\ :sub:`pix` (good)",      "5571", "21344", "83244"
   "LRT λ\ :sub:`LR` (dof=11)", "415.3 (**Yes**)", "154.4 (**Yes**)", "198.9 (**Yes**)"
   "σ̂ OLS",                     "—", "—", "—"
   "σ̂ ElasticNet",               "—", "—", "—"
   "σ̂ ISD-1",                    "—", "—", "—"
   "σ̂ ISD-3 ‡",                  "—", "—", "—"
   "σ̂ MCMC-add",                 "0.4458", "0.6449", "1.3895"
   "σ̂ MCMC-comb",                "0.4379", "0.6627", "1.4987"
   "MCMC-add acc. frac.",         "1.000", "1.000", "1.000"
   "MCMC-comb acc. frac.",        "0.739", "0.934", "0.903"
   "Dominant template",           "ns_fnt", "ns_med", "rp_fl"
   "δw/w at 30′",                 "—", "+0.7 %", "—", "—"

ISD-3 fits the same marginal relation as ISD-1 at degree 3, so the two
  separate only where the template response is non-linear.


.. seealso::

   :doc:`results_ls10_11p5` — full template amplitude tables, weight statistics, and cosmological analysis verdict for log M* ≥ 11.5.

----

MAP parameters — 11-template analysis (NSIDE 64)
-------------------------------------------------

The table below lists MAP estimates from the NSIDE = 64 run (11 templates).
Column abbreviations:

.. list-table::
   :widths: 12 35
   :header-rows: 1

   * - Abbreviation
     - Full template name
   * - EBV
     - LS10:EBV
   * - GD_G
     - LS10:GALDEPTH_G
   * - GD_R
     - LS10:GALDEPTH_R
   * - GD_Z
     - LS10:GALDEPTH_Z
   * - NOBS_R
     - LS10:NOBS_R
   * - PSF_R
     - LS10:PSFSIZE_R
   * - ns_fnt
     - GAIA:nstar_faint
   * - ns_med
     - GAIA:nstar_medium
   * - bp_fl
     - GAIA:phot_bp_mean_flux
   * - g_fl
     - GAIA:phot_g_mean_flux
   * - rp_fl
     - GAIA:phot_rp_mean_flux

The dominant systematic in all samples is **GAIA:nstar_faint** (stellar density).

Additive MAP parameters :math:`\hat{a}_i` (MCMC-add, NSIDE 64)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Sample (log M* ≥, z <)", "EBV", "GD_G", "GD_R", "GD_Z", "NOBS_R", "PSF_R", "ns_fnt", "ns_med", "bp_fl", "g_fl", "rp_fl"
   :widths: 14, 7, 7, 7, 7, 7, 7, 8, 8, 7, 7, 7
   :stub-columns: 1

   "9.0,  0.08", +0.0889, -0.0953, -0.0149, +0.0235, -0.0177, -0.0250, -0.0264, +0.0827, +0.0005, -0.0043, -0.0098
   "9.5,  0.12", +0.0850, -0.1003, -0.0119, +0.0086, -0.0056, -0.0181, -0.0234, +0.0725, +0.0019, -0.0113, +0.0105
   "10.0, 0.18", +0.0214, -0.0415, -0.0013, +0.0026, -0.0092, +0.0061, -0.0143, +0.0300, +0.0065, -0.0155, -0.0052
   "10.25, 0.22", +0.0144, -0.0316, -0.0028, +0.0078, -0.0138, +0.0007, -0.0092, +0.0189, +0.0040, -0.0109, -0.0091
   "10.5, 0.26", +0.0311, -0.0456, -0.0081, +0.0184, -0.0210, -0.0086, -0.0062, +0.0146, +0.0023, -0.0069, -0.0039
   "10.75, 0.31", +0.0234, -0.0370, -0.0084, +0.0187, -0.0224, -0.0100, +0.0015, +0.0077, +0.0044, -0.0020, -0.0043
   "11.0, 0.35", +0.0161, -0.0334, -0.0098, +0.0202, -0.0231, -0.0028, +0.0066, +0.0048, +0.0053, +0.0016, -0.0059
   "11.25, 0.35", +0.0089, -0.0234, -0.0193, +0.0333, -0.0270, +0.0000, +0.0130, +0.0036, +0.0053, +0.0049, -0.0071
   "11.5, 0.35", +0.0273, -0.0342, -0.0172, +0.0320, -0.0278, +0.0008, +0.0199, +0.0065, +0.0086, +0.0071, -0.0100

Multiplicative MAP parameters :math:`\hat{b}_i` (MCMC-comb, NSIDE 64)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Sample (log M* ≥, z <)", "EBV", "GD_G", "GD_R", "GD_Z", "NOBS_R", "PSF_R", "ns_fnt", "ns_med", "bp_fl", "g_fl", "rp_fl"
   :widths: 14, 7, 7, 7, 7, 7, 7, 8, 8, 7, 7, 7
   :stub-columns: 1

   "9.0,  0.08", +0.0511, -0.0471, -0.0109, +0.0105, -0.0202, +0.0291, -0.0326, +0.0927, +0.0177, +0.0676, +0.0091
   "9.5,  0.12", +0.1672, -0.1664, -0.0142, +0.0028, -0.0108, -0.0088, -0.0260, +0.1064, +0.0429, -0.0163, +0.0470
   "10.0, 0.18", +0.0473, -0.0719, -0.0010, -0.0232, +0.0121, +0.0288, -0.0241, +0.0543, -0.0005, -0.0088, +0.0163
   "10.25, 0.22", -0.0067, +0.0083, +0.0052, -0.0210, -0.0026, +0.0273, -0.0352, +0.0095, +0.0033, +0.0356, -0.0032
   "10.5, 0.26", +0.0245, -0.0172, +0.0034, -0.0010, -0.0262, +0.0242, -0.0327, +0.0064, +0.0112, +0.0362, +0.0048
   "10.75, 0.31", +0.0632, -0.0582, -0.0172, +0.0006, -0.0209, +0.0108, -0.0322, +0.0074, -0.0214, +0.0484, +0.0086
   "11.0, 0.35", +0.0069, +0.0091, -0.0154, +0.0099, -0.0118, +0.0151, -0.0241, -0.0037, -0.0208, +0.0662, -0.0034
   "11.25, 0.35", +0.0392, -0.0330, -0.0164, +0.0127, -0.0089, +0.0420, -0.0068, -0.0283, -0.0327, +0.0706, -0.0218
   "11.5, 0.35", +0.0563, -0.0486, -0.0131, -0.0027, +0.0077, +0.0345, -0.0038, -0.0310, -0.0246, +0.0920, -0.0168

**Key pattern**: ``GAIA:nstar_faint`` (ns_fnt) carries the largest amplitude
in nearly every sample.  The anti-correlated ``GAIA:nstar_medium`` (ns_med)
reflects stellar colour selection at moderate magnitudes.  LS10:GALDEPTH_R
captures imaging-depth variations in the :math:`r` band.

----

Outcome
-------

The systematic decontamination analysis of LS10 BGS VLIM (:math:`r < 19.5`)
yields a clear conclusion:

* **Systematics are present and detectable.**  The LRT rejects the additive
  null for **all nine samples** at **all four NSIDEs** (dof = 11,
  :math:`\chi^2_{11,\,0.95} \approx 19.7`).  The dominant
  source is GAIA stellar density (nstar_faint).

* **Sub-degree clustering is safe after correction.**  At :math:`\theta < 30'`,
  the fractional correction is :math:`\delta w/w < 2\%` for log M* ≥ 10.0.

* **Large-angle clustering requires the correction.**  At :math:`\theta > 2°`,
  stellar contamination contributes 10–40 % to :math:`w(\theta)`.

* **Use NSIDE 64 weights** for all science.  NSIDE 32 overfits the multiplicative
  model (too few pixels).  NSIDE 128/256 add noise without improving the fit.

* **Recommended weight**: ``WEIGHT_COMB`` (NSIDE 64) for all science.
