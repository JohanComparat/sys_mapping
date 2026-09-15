.. _sample-11p25:

BGS VLIM log M* ≥ 11.25, z < 0.35
=================================

541,855 galaxies, issued at NSIDE 64 (25.1 galaxies per pixel over 21,555 pixels). Leading template: ``GAIA_phot_g_mean_flux`` at 3.80 calibrated, family-wise p ≤ 0.0025. Recommended column: ``WEIGHT_SYS``; a template is detected (family-wise p ≤ 0.0025) and the NSIDE 64 likelihood ratio requires the multiplicative term (p = 0.020).

.. contents:: On this page
   :local:
   :depth: 1

.. seealso::

   :doc:`results_ls10` — all nine samples.

Template significance
---------------------

.. csv-table::
   :header: "template", "S cal", "S iid", "p", "κ"

   "GAIA_phot_g_mean_flux", "3.80", "4.08", "0.0025", "1.07"
   "GAIA_phot_rp_mean_flux", "3.58", "3.75", "0.0025", "1.05"
   "GAIA_phot_bp_mean_flux", "3.40", "3.52", "0.0025", "1.04"
   "LS10_GALDEPTH_G", "2.13", "3.95", "0.0274", "1.85"
   "LS10_PSFSIZE_R", "1.21", "2.54", "0.2095", "2.09"
   "LS10_GALDEPTH_Z", "0.93", "1.65", "0.3317", "1.78"
   "GAIA_nstar_medium", "0.92", "2.22", "0.3766", "2.42"
   "LS10_NOBS_R", "0.84", "1.42", "0.4040", "1.70"
   "LS10_GALDEPTH_R", "0.48", "0.83", "0.6384", "1.70"
   "GAIA_nstar_faint", "0.39", "0.85", "0.7082", "2.17"
   "LS10_EBV", "0.00", "0.00", "1.0000", "2.91"

Calibrated on 400 realisations; p is per template, with floor 0.0025.

Fitted amplitudes (NSIDE 64)
----------------------------

.. csv-table::
   :header: "template", "a OLS", "a ElasticNet", "a ISD-1", "a ISD-3", "a MCMC-add", "a MCMC-comb", "b MCMC-comb"

   "GAIA_nstar_faint", "+0.0090", "-0.0000", "+0.0000", "+0.0000", "+0.0090", "+0.0067", "+0.0523"
   "GAIA_nstar_medium", "-0.0235", "-0.0159", "+0.0000", "+0.0000", "-0.0235", "-0.0206", "-0.0466"
   "GAIA_phot_bp_mean_flux", "-0.0192", "-0.0079", "-0.0150", "+0.0000", "-0.0193", "-0.0193", "-0.0025"
   "GAIA_phot_g_mean_flux", "+0.0333", "+0.0015", "+0.0000", "-0.0163", "+0.0333", "+0.0271", "+0.0021"
   "GAIA_phot_rp_mean_flux", "-0.0270", "-0.0050", "+0.0000", "+0.0000", "-0.0270", "-0.0189", "-0.0076"
   "LS10_EBV", "-0.0000", "-0.0000", "+0.0000", "+0.0000", "-0.0000", "-0.0007", "+0.0381"
   "LS10_GALDEPTH_G", "+0.0130", "+0.0120", "+0.0000", "+0.0000", "+0.0130", "+0.0049", "-0.0133"
   "LS10_GALDEPTH_R", "+0.0035", "+0.0037", "+0.0169", "+0.0187", "+0.0035", "+0.0031", "-0.0358"
   "LS10_GALDEPTH_Z", "+0.0050", "+0.0040", "+0.0000", "+0.0000", "+0.0050", "+0.0073", "-0.0274"
   "LS10_NOBS_R", "+0.0051", "+0.0041", "+0.0000", "+0.0000", "+0.0051", "+0.0133", "+0.0711"
   "LS10_PSFSIZE_R", "-0.0071", "-0.0060", "+0.0000", "+0.0000", "-0.0071", "-0.0037", "-0.0226"

Amplitudes are per unit template standard deviation on the footprint.

Weights
-------

.. csv-table::
   :header: "column", "min", "max", "1 %", "99 %", "clipped", "identically 1"

   "WEIGHT_OLS", "0.570", "1.280", "0.906", "1.099", "0.000 %", "no"
   "WEIGHT_ENET", "0.651", "1.235", "0.915", "1.088", "0.000 %", "no"
   "WEIGHT_ISD1", "0.922", "1.077", "0.928", "1.073", "0.000 %", "no"
   "WEIGHT_ISD3", "0.918", "1.109", "0.926", "1.097", "0.000 %", "no"
   "WEIGHT_ADD", "0.050", "1.952", "0.897", "1.117", "0.003 %", "no"
   "WEIGHT_COMB", "0.050", "20.000", "0.871", "1.124", "0.023 %", "no"

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0064_weight_map.png
   :width: 95%

   Weight maps at NSIDE 64, one panel per method.

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0064_weight_hist.png
   :width: 70%

   Weight distributions.

Angular correlation function
----------------------------

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855_NSIDE0064_wtheta.png
   :width: 80%

   Observed and corrected w(θ), full cross-template correction.

.. csv-table:: Corrected over observed w(θ).
   :header: "method", "at 30′", "smallest ratio", "at θ"

   "OLS", "0.982", "0.785", "220′"
   "ElasticNet", "0.985", "0.819", "220′"
   "ISD-1", "0.994", "0.960", "220′"
   "ISD-3", "0.993", "0.954", "220′"
   "MCMC-add", "0.981", "0.778", "220′"
   "MCMC-comb", "0.979", "0.803", "220′"

Likelihood ratio
----------------

.. csv-table::
   :header: "NSIDE", "λ LR", "mock p", "null mean / max", "N"

   "32", "636.5", "0.020", "34.4 / 106.5", "50"
   "64", "127.4", "0.020", "47.2 / 108.9", "50"

Resolution comparison
---------------------

.. csv-table:: Residual scatter :math:`\hat\sigma`.
   :header: "method", "NSIDE 32", "NSIDE 64", "NSIDE 128"

   "OLS", "0.3309", "0.3842", "0.6412"
   "ElasticNet", "0.3313", "0.3843", "0.6413"
   "ISD-1", "0.3317", "0.3847", "0.6416"
   "ISD-3", "0.3317", "0.3852", "0.6417"
   "MCMC-add", "0.3313", "0.3843", "0.6413"
   "MCMC-comb", "0.3199", "0.3931", "0.6570"
