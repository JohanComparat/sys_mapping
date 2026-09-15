.. _sample-11p0:

BGS VLIM log M* ≥ 11.0, z < 0.35
================================

1,619,838 galaxies, issued at NSIDE 64 (74.8 galaxies per pixel over 21,646 pixels). Leading template: ``GAIA_phot_rp_mean_flux`` at 4.37 calibrated, family-wise p ≤ 0.0025. Recommended column: ``WEIGHT_SYS``; a template is detected (family-wise p ≤ 0.0025); the NSIDE 64 likelihood ratio does not require the multiplicative term (p = 0.078), so WEIGHT_ADD is the simpler alternative.

.. contents:: On this page
   :local:
   :depth: 1

.. seealso::

   :doc:`results_ls10` — all nine samples.

Template significance
---------------------

.. csv-table::
   :header: "template", "S cal", "S iid", "p", "κ"

   "GAIA_phot_rp_mean_flux", "4.37", "4.15", "0.0025", "0.95"
   "GAIA_phot_g_mean_flux", "3.02", "3.19", "0.0025", "1.06"
   "GAIA_phot_bp_mean_flux", "2.21", "2.32", "0.0299", "1.05"
   "GAIA_nstar_medium", "1.39", "4.08", "0.1546", "2.94"
   "LS10_GALDEPTH_G", "1.33", "2.60", "0.1796", "1.96"
   "LS10_PSFSIZE_R", "1.23", "2.72", "0.2269", "2.20"
   "LS10_GALDEPTH_Z", "1.13", "2.21", "0.2519", "1.96"
   "GAIA_nstar_faint", "0.77", "1.96", "0.4140", "2.53"
   "LS10_GALDEPTH_R", "0.73", "1.40", "0.4938", "1.92"
   "LS10_NOBS_R", "0.33", "0.60", "0.7581", "1.84"
   "LS10_EBV", "0.29", "1.09", "0.7681", "3.69"

Calibrated on 400 realisations; p is per template, with floor 0.0025.

Fitted amplitudes (NSIDE 64)
----------------------------

.. csv-table::
   :header: "template", "a OLS", "a ElasticNet", "a ISD-1", "a ISD-3", "a MCMC-add", "a MCMC-comb", "b MCMC-comb"

   "GAIA_nstar_faint", "+0.0161", "-0.0000", "+0.0000", "+0.0000", "+0.0161", "+0.0038", "+0.0131"
   "GAIA_nstar_medium", "-0.0335", "-0.0185", "-0.0171", "+0.0000", "-0.0334", "-0.0217", "+0.0058"
   "GAIA_phot_bp_mean_flux", "-0.0098", "-0.0023", "+0.0000", "+0.0000", "-0.0098", "-0.0102", "-0.0179"
   "GAIA_phot_g_mean_flux", "+0.0202", "-0.0000", "+0.0000", "-0.0171", "+0.0202", "+0.0172", "+0.0110"
   "GAIA_phot_rp_mean_flux", "-0.0231", "-0.0084", "-0.0159", "+0.0000", "-0.0231", "-0.0181", "-0.0111"
   "LS10_EBV", "-0.0027", "-0.0015", "+0.0000", "+0.0000", "-0.0027", "-0.0038", "+0.0156"
   "LS10_GALDEPTH_G", "+0.0066", "+0.0053", "+0.0000", "+0.0000", "+0.0066", "+0.0029", "-0.0246"
   "LS10_GALDEPTH_R", "+0.0046", "+0.0050", "+0.0116", "+0.0139", "+0.0046", "+0.0057", "-0.0040"
   "LS10_GALDEPTH_Z", "+0.0051", "+0.0038", "+0.0000", "+0.0000", "+0.0051", "-0.0031", "-0.0203"
   "LS10_NOBS_R", "+0.0017", "+0.0004", "+0.0000", "+0.0000", "+0.0017", "+0.0106", "+0.0672"
   "LS10_PSFSIZE_R", "-0.0059", "-0.0043", "+0.0000", "+0.0000", "-0.0059", "-0.0026", "-0.0032"

Amplitudes are per unit template standard deviation on the footprint.

Weights
-------

.. csv-table::
   :header: "column", "min", "max", "1 %", "99 %", "clipped", "identically 1"

   "WEIGHT_OLS", "0.683", "1.217", "0.930", "1.101", "0.000 %", "no"
   "WEIGHT_ENET", "0.760", "1.237", "0.940", "1.089", "0.000 %", "no"
   "WEIGHT_ISD1", "0.925", "1.132", "0.939", "1.099", "0.000 %", "no"
   "WEIGHT_ISD3", "0.932", "1.109", "0.939", "1.099", "0.000 %", "no"
   "WEIGHT_ADD", "0.050", "1.931", "0.922", "1.112", "0.000 %", "no"
   "WEIGHT_COMB", "0.050", "9.885", "0.912", "1.114", "0.004 %", "no"

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0064_weight_map.png
   :width: 95%

   Weight maps at NSIDE 64, one panel per method.

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0064_weight_hist.png
   :width: 70%

   Weight distributions.

Angular correlation function
----------------------------

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_11.0_Mstar_12.0_0.05_z_0.35_N_1619838_NSIDE0064_wtheta.png
   :width: 80%

   Observed and corrected w(θ), full cross-template correction.

.. csv-table:: Corrected over observed w(θ).
   :header: "method", "at 30′", "smallest ratio", "at θ"

   "OLS", "0.981", "0.763", "220′"
   "ElasticNet", "0.986", "0.817", "220′"
   "ISD-1", "0.987", "0.858", "220′"
   "ISD-3", "0.994", "0.968", "220′"
   "MCMC-add", "0.980", "0.756", "220′"
   "MCMC-comb", "0.977", "0.765", "220′"

Likelihood ratio
----------------

.. csv-table::
   :header: "NSIDE", "λ LR", "mock p", "null mean / max", "N"

   "32", "1022.0", "0.020", "39.8 / 103.9", "50"
   "64", "79.6", "0.078", "37.2 / 125.8", "50"

Resolution comparison
---------------------

.. csv-table:: Residual scatter :math:`\hat\sigma`.
   :header: "method", "NSIDE 32", "NSIDE 64", "NSIDE 128"

   "OLS", "0.2971", "0.2973", "0.4580"
   "ElasticNet", "0.2977", "0.2974", "0.4581"
   "ISD-1", "0.2979", "0.2976", "0.4583"
   "ISD-3", "0.2982", "0.2983", "0.4585"
   "MCMC-add", "0.2975", "0.2974", "0.4581"
   "MCMC-comb", "0.2808", "0.3074", "0.4818"
