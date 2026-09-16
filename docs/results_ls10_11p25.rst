.. _sample-11p25:

BGS VLIM log M* ≥ 11.25, z < 0.35
=================================

541,855 galaxies, issued at NSIDE 64 (25.1 galaxies per pixel over 21,555 pixels). Leading template: ``GAIA_phot_rp_mean_flux`` at 3.84 calibrated, family-wise p 0.0050. Recommended column: ``WEIGHT_SYS``; a template is detected (family-wise p 0.0050) and the NSIDE 64 likelihood ratio requires the multiplicative term (p = 0.020).

.. contents:: On this page
   :local:
   :depth: 1

.. seealso::

   :doc:`results_ls10` — all nine samples.

Template significance
---------------------

.. csv-table::
   :header: "template", "S cal", "S iid", "p", "κ"

   "GAIA_phot_rp_mean_flux", "3.84", "3.75", "0.0025", "0.98"
   "GAIA_phot_g_mean_flux", "3.81", "4.08", "0.0025", "1.07"
   "GAIA_phot_bp_mean_flux", "3.20", "3.52", "0.0050", "1.10"
   "LS10_GALDEPTH_G", "2.13", "3.95", "0.0374", "1.85"
   "LS10_PSFSIZE_R", "1.25", "2.54", "0.2195", "2.03"
   "LS10_GALDEPTH_Z", "0.93", "1.65", "0.3416", "1.77"
   "GAIA_nstar_medium", "0.93", "2.22", "0.3716", "2.40"
   "LS10_NOBS_R", "0.82", "1.42", "0.4065", "1.73"
   "LS10_GALDEPTH_R", "0.47", "0.83", "0.6783", "1.74"
   "GAIA_nstar_faint", "0.39", "0.85", "0.6983", "2.14"
   "LS10_EBV", "0.00", "0.00", "1.0000", "2.89"

Calibrated on 400 realisations; p is per template, with floor 0.0025.

Fitted amplitudes (NSIDE 64)
----------------------------

.. csv-table::
   :header: "template", "a OLS", "a ElasticNet", "a ISD-1", "a ISD-3", "a MCMC-add", "a MCMC-comb", "b MCMC-comb"

   "GAIA_nstar_faint", "+0.0090", "-0.0000", "+0.0000", "+0.0000", "+0.0090", "+0.0166", "+0.0231"
   "GAIA_nstar_medium", "-0.0235", "-0.0159", "+0.0000", "+0.0000", "-0.0235", "-0.0137", "+0.0155"
   "GAIA_phot_bp_mean_flux", "-0.0192", "-0.0079", "+0.0000", "+0.0000", "-0.0193", "+0.0011", "-0.0009"
   "GAIA_phot_g_mean_flux", "+0.0333", "+0.0015", "+0.0000", "+0.0000", "+0.0333", "-0.0151", "-0.0470"
   "GAIA_phot_rp_mean_flux", "-0.0270", "-0.0050", "-0.0148", "-0.0173", "-0.0270", "-0.0053", "-0.0186"
   "LS10_EBV", "-0.0000", "-0.0000", "+0.0000", "+0.0000", "-0.0000", "+0.0000", "+0.0177"
   "LS10_GALDEPTH_G", "+0.0130", "+0.0120", "+0.0000", "+0.0000", "+0.0130", "-0.0060", "-0.0110"
   "LS10_GALDEPTH_R", "+0.0035", "+0.0037", "+0.0169", "+0.0190", "+0.0035", "+0.0412", "+0.0368"
   "LS10_GALDEPTH_Z", "+0.0050", "+0.0040", "+0.0000", "+0.0000", "+0.0050", "-0.0257", "-0.0371"
   "LS10_NOBS_R", "+0.0051", "+0.0041", "+0.0000", "+0.0000", "+0.0051", "+0.0393", "-0.0013"
   "LS10_PSFSIZE_R", "-0.0071", "-0.0060", "+0.0000", "+0.0000", "-0.0071", "+0.0012", "-0.0114"

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

   "OLS", "0.986", "0.828", "220′"
   "ElasticNet", "0.989", "0.862", "220′"
   "ISD-1", "0.998", "0.998", "32′"
   "ISD-3", "0.996", "0.993", "220′"
   "MCMC-add", "0.986", "0.828", "220′"
   "MCMC-comb", "0.949", "0.741", "220′"

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
