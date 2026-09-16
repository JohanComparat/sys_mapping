.. _sample-10p25:

BGS VLIM log M* ≥ 10.25, z < 0.22
=================================

3,308,841 galaxies, issued at NSIDE 128 (39.0 galaxies per pixel over 84,831 pixels). Leading template: ``GAIA_phot_rp_mean_flux`` at 4.99 calibrated, family-wise p ≤ 0.0025. Recommended column: ``WEIGHT_SYS``; a template is detected (family-wise p ≤ 0.0025); the NSIDE 64 likelihood ratio does not require the multiplicative term (p = 0.078), so WEIGHT_ADD is the simpler alternative.

.. contents:: On this page
   :local:
   :depth: 1

.. seealso::

   :doc:`results_ls10` — all nine samples.

Template significance
---------------------

.. csv-table::
   :header: "template", "S cal", "S iid", "p", "κ"

   "GAIA_phot_rp_mean_flux", "4.99", "4.95", "0.0025", "0.99"
   "LS10_GALDEPTH_R", "2.55", "6.87", "0.0150", "2.70"
   "GAIA_phot_g_mean_flux", "2.17", "2.33", "0.0274", "1.07"
   "GAIA_phot_bp_mean_flux", "1.87", "2.13", "0.0549", "1.13"
   "LS10_NOBS_R", "1.86", "5.12", "0.0873", "2.75"
   "GAIA_nstar_medium", "1.81", "5.02", "0.0773", "2.78"
   "LS10_GALDEPTH_G", "1.45", "4.25", "0.1596", "2.94"
   "LS10_PSFSIZE_R", "1.17", "3.76", "0.2344", "3.20"
   "LS10_GALDEPTH_Z", "0.48", "1.34", "0.6035", "2.79"
   "GAIA_nstar_faint", "0.30", "0.80", "0.7556", "2.65"
   "LS10_EBV", "0.08", "0.37", "0.9426", "4.84"

Calibrated on 400 realisations; p is per template, with floor 0.0025.

Fitted amplitudes (NSIDE 128)
-----------------------------

.. csv-table::
   :header: "template", "a OLS", "a ElasticNet", "a ISD-1", "a ISD-3", "a MCMC-add", "a MCMC-comb", "b MCMC-comb"

   "GAIA_nstar_faint", "+0.0037", "-0.0000", "+0.0000", "+0.0000", "+0.0037", "+0.0011", "-0.0052"
   "GAIA_nstar_medium", "-0.0228", "-0.0186", "+0.0000", "+0.0000", "-0.0228", "-0.0250", "-0.0270"
   "GAIA_phot_bp_mean_flux", "-0.0071", "-0.0028", "+0.0000", "+0.0000", "-0.0072", "-0.0009", "+0.0213"
   "GAIA_phot_g_mean_flux", "+0.0119", "+0.0000", "-0.0102", "-0.0105", "+0.0119", "-0.0146", "+0.0200"
   "GAIA_phot_rp_mean_flux", "-0.0221", "-0.0136", "-0.0049", "+0.0000", "-0.0221", "+0.0010", "-0.0167"
   "LS10_EBV", "+0.0008", "-0.0000", "+0.0000", "+0.0000", "+0.0008", "+0.0056", "+0.0146"
   "LS10_GALDEPTH_G", "-0.0089", "-0.0059", "+0.0000", "+0.0000", "-0.0089", "-0.0141", "-0.0091"
   "LS10_GALDEPTH_R", "+0.0186", "+0.0139", "+0.0000", "+0.0000", "+0.0186", "+0.0149", "+0.0107"
   "LS10_GALDEPTH_Z", "+0.0025", "+0.0014", "+0.0000", "+0.0000", "+0.0025", "+0.0165", "-0.0133"
   "LS10_NOBS_R", "-0.0117", "-0.0081", "+0.0000", "+0.0000", "-0.0117", "-0.0134", "-0.0458"
   "LS10_PSFSIZE_R", "-0.0067", "-0.0059", "+0.0000", "+0.0000", "-0.0067", "-0.0067", "+0.0155"

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

   "OLS", "0.995", "0.961", "272′"
   "ElasticNet", "0.996", "0.967", "272′"
   "ISD-1", "1.003", "1.000", "1′"
   "ISD-3", "1.003", "1.000", "1′"
   "MCMC-add", "0.995", "0.961", "272′"
   "MCMC-comb", "0.992", "0.955", "272′"

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
