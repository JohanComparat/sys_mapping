.. _sample-10p25:

BGS VLIM log M* ≥ 10.25, z < 0.22
=================================

3,308,841 galaxies, issued at NSIDE 128 (39.0 galaxies per pixel over 84,831 pixels). Leading template: ``GAIA_phot_rp_mean_flux`` at 4.81 calibrated, family-wise p ≤ 0.0025. Recommended column: ``WEIGHT_SYS``; a template is detected (family-wise p ≤ 0.0025); the NSIDE 64 likelihood ratio does not require the multiplicative term (p = 0.078), so WEIGHT_ADD is the simpler alternative.

.. contents:: On this page
   :local:
   :depth: 1

.. seealso::

   :doc:`results_ls10` — all nine samples.

Template significance
---------------------

.. csv-table::
   :header: "template", "S cal", "S iid", "p", "κ"

   "GAIA_phot_rp_mean_flux", "4.81", "4.95", "0.0025", "1.03"
   "LS10_GALDEPTH_R", "2.53", "6.87", "0.0175", "2.72"
   "GAIA_phot_g_mean_flux", "2.19", "2.33", "0.0299", "1.06"
   "GAIA_phot_bp_mean_flux", "1.96", "2.13", "0.0474", "1.09"
   "LS10_NOBS_R", "1.88", "5.12", "0.0798", "2.73"
   "GAIA_nstar_medium", "1.78", "5.02", "0.0823", "2.82"
   "LS10_GALDEPTH_G", "1.44", "4.25", "0.1446", "2.95"
   "LS10_PSFSIZE_R", "1.16", "3.76", "0.2369", "3.23"
   "LS10_GALDEPTH_Z", "0.48", "1.34", "0.6160", "2.78"
   "GAIA_nstar_faint", "0.29", "0.80", "0.7781", "2.71"
   "LS10_EBV", "0.08", "0.37", "0.9551", "4.88"

Calibrated on 400 realisations; p is per template, with floor 0.0025.

Fitted amplitudes (NSIDE 128)
-----------------------------

.. csv-table::
   :header: "template", "a OLS", "a ElasticNet", "a ISD-1", "a ISD-3", "a MCMC-add", "a MCMC-comb", "b MCMC-comb"

   "GAIA_nstar_faint", "+0.0037", "-0.0000", "+0.0000", "+0.0000", "+0.0037", "+0.0035", "-0.0132"
   "GAIA_nstar_medium", "-0.0228", "-0.0186", "+0.0000", "+0.0000", "-0.0228", "-0.0240", "-0.0125"
   "GAIA_phot_bp_mean_flux", "-0.0071", "-0.0028", "+0.0000", "+0.0000", "-0.0072", "-0.0056", "+0.0510"
   "GAIA_phot_g_mean_flux", "+0.0119", "+0.0000", "+0.0000", "+0.0000", "+0.0119", "+0.0082", "+0.0217"
   "GAIA_phot_rp_mean_flux", "-0.0221", "-0.0136", "-0.0122", "-0.0123", "-0.0221", "-0.0190", "+0.0020"
   "LS10_EBV", "+0.0008", "-0.0000", "+0.0000", "+0.0000", "+0.0008", "-0.0006", "-0.0050"
   "LS10_GALDEPTH_G", "-0.0089", "-0.0059", "+0.0000", "+0.0000", "-0.0089", "-0.0037", "+0.0019"
   "LS10_GALDEPTH_R", "+0.0186", "+0.0139", "+0.0000", "+0.0000", "+0.0186", "+0.0161", "+0.0102"
   "LS10_GALDEPTH_Z", "+0.0025", "+0.0014", "+0.0000", "+0.0000", "+0.0025", "-0.0001", "-0.0358"
   "LS10_NOBS_R", "-0.0117", "-0.0081", "+0.0000", "+0.0000", "-0.0117", "-0.0123", "-0.0326"
   "LS10_PSFSIZE_R", "-0.0067", "-0.0059", "+0.0000", "+0.0000", "-0.0067", "-0.0066", "+0.0136"

Amplitudes are per unit template standard deviation on the footprint.

Weights
-------

.. csv-table::
   :header: "column", "min", "max", "1 %", "99 %", "clipped", "identically 1"

   "WEIGHT_OLS", "0.654", "1.710", "0.944", "1.104", "0.000 %", "no"
   "WEIGHT_ENET", "0.844", "2.229", "0.953", "1.092", "0.000 %", "no"
   "WEIGHT_ISD1", "0.978", "1.069", "0.978", "1.069", "0.000 %", "no"
   "WEIGHT_ISD3", "0.962", "1.060", "0.962", "1.060", "0.000 %", "no"
   "WEIGHT_ADD", "0.050", "4.530", "0.929", "1.120", "0.000 %", "no"
   "WEIGHT_COMB", "0.050", "5.597", "0.886", "1.153", "0.028 %", "no"

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0128_weight_map.png
   :width: 95%

   Weight maps at NSIDE 128, one panel per method.

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0128_weight_hist.png
   :width: 70%

   Weight distributions.

Angular correlation function
----------------------------

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_10.25_Mstar_12.0_0.05_z_0.22_N_3308841_NSIDE0128_wtheta.png
   :width: 80%

   Observed and corrected w(θ), full cross-template correction.

.. csv-table:: Corrected over observed w(θ).
   :header: "method", "at 30′", "smallest ratio", "at θ"

   "OLS", "0.992", "0.943", "272′"
   "ElasticNet", "0.993", "0.949", "272′"
   "ISD-1", "1.000", "1.000", "272′"
   "ISD-3", "1.000", "1.000", "272′"
   "MCMC-add", "0.991", "0.942", "272′"
   "MCMC-comb", "0.990", "0.937", "272′"

Likelihood ratio
----------------

.. csv-table::
   :header: "NSIDE", "λ LR", "mock p", "null mean / max", "N"

   "32", "836.9", "0.020", "42.4 / 107.6", "50"
   "64", "116.3", "0.078", "61.3 / 135.9", "50"

Resolution comparison
---------------------

.. csv-table:: Residual scatter :math:`\hat\sigma`.
   :header: "method", "NSIDE 32", "NSIDE 64", "NSIDE 128"

   "OLS", "0.3417", "0.3433", "0.4900"
   "ElasticNet", "0.3427", "0.3434", "0.4900"
   "ISD-1", "0.3439", "0.3443", "0.4903"
   "ISD-3", "0.3433", "0.3443", "0.4903"
   "MCMC-add", "0.3421", "0.3434", "0.4900"
   "MCMC-comb", "0.3266", "0.3501", "0.5189"
