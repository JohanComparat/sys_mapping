.. _sample-10p0:

BGS VLIM log M* ≥ 10.0, z < 0.18
================================

2,759,238 galaxies, issued at NSIDE 128 (32.5 galaxies per pixel over 84,860 pixels). Leading template: ``GAIA_phot_rp_mean_flux`` at 4.02 calibrated, family-wise p ≤ 0.0025. Recommended column: ``WEIGHT_SYS``; a template is detected (family-wise p ≤ 0.0025) and the NSIDE 64 likelihood ratio requires the multiplicative term (p = 0.020).

.. contents:: On this page
   :local:
   :depth: 1

.. seealso::

   :doc:`results_ls10` — all nine samples.

Template significance
---------------------

.. csv-table::
   :header: "template", "S cal", "S iid", "p", "κ"

   "GAIA_phot_rp_mean_flux", "4.02", "4.16", "0.0025", "1.03"
   "LS10_GALDEPTH_R", "3.38", "9.54", "0.0050", "2.82"
   "LS10_NOBS_R", "2.23", "6.63", "0.0349", "2.98"
   "GAIA_phot_g_mean_flux", "1.86", "1.95", "0.0549", "1.05"
   "LS10_GALDEPTH_G", "1.79", "5.61", "0.0723", "3.13"
   "GAIA_nstar_medium", "1.65", "4.86", "0.1097", "2.96"
   "GAIA_phot_bp_mean_flux", "1.51", "1.64", "0.1372", "1.09"
   "LS10_GALDEPTH_Z", "0.85", "2.44", "0.3741", "2.86"
   "LS10_EBV", "0.56", "2.92", "0.6010", "5.17"
   "GAIA_nstar_faint", "0.27", "0.80", "0.7731", "2.96"
   "LS10_PSFSIZE_R", "0.26", "1.01", "0.7955", "3.83"

Calibrated on 400 realisations; p is per template, with floor 0.0025.

Fitted amplitudes (NSIDE 128)
-----------------------------

.. csv-table::
   :header: "template", "a OLS", "a ElasticNet", "a ISD-1", "a ISD-3", "a MCMC-add", "a MCMC-comb", "b MCMC-comb"

   "GAIA_nstar_faint", "+0.0043", "-0.0000", "+0.0000", "+0.0000", "+0.0043", "-0.0175", "-0.0096"
   "GAIA_nstar_medium", "-0.0252", "-0.0191", "+0.0000", "+0.0000", "-0.0252", "-0.0060", "-0.0240"
   "GAIA_phot_bp_mean_flux", "-0.0063", "-0.0020", "+0.0000", "+0.0000", "-0.0063", "-0.0003", "+0.0249"
   "GAIA_phot_g_mean_flux", "+0.0114", "-0.0000", "-0.0101", "-0.0114", "+0.0114", "-0.0145", "+0.0150"
   "GAIA_phot_rp_mean_flux", "-0.0212", "-0.0128", "+0.0000", "+0.0000", "-0.0212", "+0.0009", "-0.0093"
   "LS10_EBV", "+0.0069", "+0.0037", "+0.0000", "+0.0000", "+0.0069", "+0.0089", "+0.0119"
   "LS10_GALDEPTH_G", "-0.0135", "-0.0095", "+0.0000", "+0.0000", "-0.0135", "-0.0113", "-0.0468"
   "LS10_GALDEPTH_R", "+0.0296", "+0.0232", "+0.0000", "+0.0000", "+0.0296", "+0.0211", "+0.0186"
   "LS10_GALDEPTH_Z", "+0.0053", "+0.0037", "+0.0000", "+0.0005", "+0.0053", "+0.0182", "-0.0026"
   "LS10_NOBS_R", "-0.0173", "-0.0126", "+0.0000", "+0.0000", "-0.0173", "-0.0214", "-0.0145"
   "LS10_PSFSIZE_R", "-0.0021", "-0.0007", "+0.0000", "+0.0000", "-0.0021", "-0.0010", "+0.0133"

Amplitudes are per unit template standard deviation on the footprint.

Weights
-------

.. csv-table::
   :header: "column", "min", "max", "1 %", "99 %", "clipped", "identically 1"

   "WEIGHT_OLS", "0.665", "1.671", "0.915", "1.108", "0.000 %", "no"
   "WEIGHT_ENET", "0.759", "1.993", "0.929", "1.089", "0.000 %", "no"
   "WEIGHT_ISD1", "0.977", "1.063", "0.977", "1.063", "0.000 %", "no"
   "WEIGHT_ISD3", "0.935", "1.083", "0.944", "1.069", "0.000 %", "no"
   "WEIGHT_ADD", "0.050", "9.681", "0.914", "1.128", "0.001 %", "no"
   "WEIGHT_COMB", "0.050", "9.042", "0.903", "1.157", "0.015 %", "no"

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0128_weight_map.png
   :width: 95%

   Weight maps at NSIDE 128, one panel per method.

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0128_weight_hist.png
   :width: 70%

   Weight distributions.

Angular correlation function
----------------------------

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238_NSIDE0128_wtheta.png
   :width: 80%

   Observed and corrected w(θ), full cross-template correction.

.. csv-table:: Corrected over observed w(θ).
   :header: "method", "at 30′", "smallest ratio", "at θ"

   "OLS", "0.996", "0.987", "272′"
   "ElasticNet", "0.998", "0.991", "272′"
   "ISD-1", "1.004", "1.000", "1′"
   "ISD-3", "1.004", "1.000", "1′"
   "MCMC-add", "0.996", "0.987", "272′"
   "MCMC-comb", "0.996", "0.985", "272′"

Likelihood ratio
----------------

.. csv-table::
   :header: "NSIDE", "λ LR", "mock p", "null mean / max", "N"

   "32", "696.7", "0.020", "43.6 / 106.0", "50"
   "64", "179.0", "0.020", "64.4 / 133.4", "50"

Resolution comparison
---------------------

.. csv-table:: Residual scatter :math:`\hat\sigma`.
   :header: "method", "NSIDE 32", "NSIDE 64", "NSIDE 128"

   "OLS", "0.3801", "0.3969", "0.5605"
   "ElasticNet", "0.3820", "0.3970", "0.5606"
   "ISD-1", "0.3820", "0.3980", "0.5610"
   "ISD-3", "0.3816", "0.3980", "0.5612"
   "MCMC-add", "0.3805", "0.3970", "0.5606"
   "MCMC-comb", "0.3650", "0.3983", "0.5957"
