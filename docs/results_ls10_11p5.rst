.. _sample-11p5:

BGS VLIM log M* ≥ 11.5, z < 0.35
================================

120,882 galaxies, issued at NSIDE 16 (82.5 galaxies per pixel over 1,465 pixels). Leading template: ``LS10_GALDEPTH_G`` at 1.23 calibrated, family-wise p 0.9002. Recommended column: ``WEIGHT_SYS``; the NSIDE 32 likelihood ratio requires the multiplicative term (p = 0.020).

.. contents:: On this page
   :local:
   :depth: 1

.. seealso::

   :doc:`results_ls10` — all nine samples.

Template significance
---------------------

.. csv-table::
   :header: "template", "S cal", "S iid", "p", "κ"

   "LS10_GALDEPTH_G", "1.23", "1.57", "0.2170", "1.28"
   "LS10_GALDEPTH_Z", "1.00", "1.31", "0.3342", "1.31"
   "GAIA_nstar_faint", "0.96", "1.53", "0.2993", "1.60"
   "GAIA_nstar_medium", "0.86", "1.46", "0.3641", "1.69"
   "LS10_PSFSIZE_R", "0.54", "0.73", "0.5761", "1.36"
   "LS10_EBV", "0.40", "0.86", "0.7232", "2.15"
   "GAIA_phot_g_mean_flux", "0.26", "0.26", "0.8105", "1.03"
   "LS10_NOBS_R", "0.17", "0.22", "0.8579", "1.29"
   "GAIA_phot_rp_mean_flux", "0.15", "0.15", "0.8803", "1.03"
   "LS10_GALDEPTH_R", "0.14", "0.18", "0.8978", "1.29"
   "GAIA_phot_bp_mean_flux", "0.03", "0.04", "0.9800", "1.15"

Calibrated on 400 realisations; p is per template, with floor 0.0025.

Fitted amplitudes (NSIDE 16)
----------------------------

.. csv-table::
   :header: "template", "a OLS", "a ElasticNet", "a ISD-1", "a ISD-3", "a MCMC-add", "a MCMC-comb", "b MCMC-comb"

   "GAIA_nstar_faint", "+0.0471", "-0.0000", "+0.0000", "+0.0000", "+0.0471", "+0.0110", "+0.3590"
   "GAIA_nstar_medium", "-0.0501", "-0.0019", "+0.0000", "+0.0000", "-0.0501", "-0.0230", "-0.2943"
   "GAIA_phot_bp_mean_flux", "-0.0009", "-0.0053", "+0.0000", "+0.0000", "-0.0010", "+0.0052", "+0.0988"
   "GAIA_phot_g_mean_flux", "-0.0089", "-0.0044", "+0.0000", "+0.0000", "-0.0089", "-0.0189", "-0.1404"
   "GAIA_phot_rp_mean_flux", "-0.0050", "-0.0056", "+0.0000", "+0.0000", "-0.0049", "-0.0054", "+0.0029"
   "LS10_EBV", "-0.0078", "-0.0057", "+0.0000", "+0.0000", "-0.0078", "+0.0014", "-0.0645"
   "LS10_GALDEPTH_G", "+0.0143", "+0.0102", "+0.0200", "+0.0000", "+0.0143", "+0.0129", "-0.0089"
   "LS10_GALDEPTH_R", "-0.0022", "+0.0000", "+0.0000", "+0.0000", "-0.0022", "-0.0056", "+0.0156"
   "LS10_GALDEPTH_Z", "+0.0117", "+0.0080", "+0.0000", "+0.0263", "+0.0117", "+0.0159", "-0.0395"
   "LS10_NOBS_R", "-0.0023", "+0.0000", "+0.0000", "+0.0000", "-0.0023", "+0.0079", "-0.0455"
   "LS10_PSFSIZE_R", "-0.0057", "-0.0030", "+0.0000", "+0.0000", "-0.0057", "-0.0039", "+0.0215"

Amplitudes are per unit template standard deviation on the footprint.

Weights
-------

.. csv-table::
   :header: "column", "min", "max", "1 %", "99 %", "clipped", "identically 1"

   "WEIGHT_OLS", "0.817", "1.112", "0.907", "1.080", "0.000 %", "no"
   "WEIGHT_ENET", "0.850", "1.104", "0.930", "1.070", "0.000 %", "no"
   "WEIGHT_ISD1", "0.930", "1.024", "0.930", "1.024", "0.000 %", "no"
   "WEIGHT_ISD3", "0.924", "1.016", "0.926", "1.016", "0.000 %", "no"
   "WEIGHT_ADD", "0.751", "1.191", "0.895", "1.092", "0.000 %", "no"
   "WEIGHT_COMB", "0.547", "1.411", "0.853", "1.126", "0.000 %", "no"

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0016_weight_map.png
   :width: 95%

   Weight maps at NSIDE 16, one panel per method.

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0016_weight_hist.png
   :width: 70%

   Weight distributions.

Angular correlation function
----------------------------

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882_NSIDE0016_wtheta.png
   :width: 80%

   Observed and corrected w(θ), full cross-template correction.

.. csv-table:: Corrected over observed w(θ).
   :header: "method", "at 30′", "smallest ratio", "at θ"

   "OLS", "0.997", "0.918", "272′"
   "ElasticNet", "1.003", "1.000", "1′"
   "ISD-1", "1.007", "1.000", "1′"
   "ISD-3", "1.004", "1.000", "1′"
   "MCMC-add", "0.997", "0.918", "272′"
   "MCMC-comb", "0.989", "0.872", "272′"

Likelihood ratio
----------------

.. csv-table::
   :header: "NSIDE", "λ LR", "mock p", "null mean / max", "N"

   "32", "397.5", "0.020", "33.2 / 87.1", "50"
   "64", "154.8", "0.451", "163.1 / 546.3", "50"

Resolution comparison
---------------------

.. csv-table:: Residual scatter :math:`\hat\sigma`.
   :header: "method", "NSIDE 32", "NSIDE 64", "NSIDE 128"

   "OLS", "0.4453", "0.6447", "1.3894"
   "ElasticNet", "0.4456", "0.6448", "1.3906"
   "ISD-1", "0.4459", "0.6452", "1.3896"
   "ISD-3", "0.4459", "0.6452", "1.3898"
   "MCMC-add", "0.4458", "0.6449", "1.3895"
   "MCMC-comb", "0.4379", "0.6627", "1.4987"
