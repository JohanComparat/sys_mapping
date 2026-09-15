.. _sample-10p5:

BGS VLIM log M* ≥ 10.5, z < 0.26
================================

3,263,228 galaxies, issued at NSIDE 128 (38.5 galaxies per pixel over 84,811 pixels). Leading template: ``GAIA_phot_rp_mean_flux`` at 6.33 calibrated, family-wise p ≤ 0.0025. Recommended column: ``WEIGHT_SYS``; a template is detected (family-wise p ≤ 0.0025); the NSIDE 64 likelihood ratio does not require the multiplicative term (p = 0.373), so WEIGHT_ADD is the simpler alternative.

.. contents:: On this page
   :local:
   :depth: 1

.. seealso::

   :doc:`results_ls10` — all nine samples.

Template significance
---------------------

.. csv-table::
   :header: "template", "S cal", "S iid", "p", "κ"

   "GAIA_phot_rp_mean_flux", "6.33", "6.18", "0.0025", "0.98"
   "GAIA_phot_g_mean_flux", "3.62", "3.80", "0.0025", "1.05"
   "GAIA_phot_bp_mean_flux", "3.32", "3.56", "0.0025", "1.07"
   "GAIA_nstar_medium", "2.47", "7.59", "0.0100", "3.07"
   "LS10_GALDEPTH_R", "2.13", "5.63", "0.0299", "2.65"
   "LS10_NOBS_R", "1.43", "3.74", "0.1771", "2.61"
   "GAIA_nstar_faint", "1.23", "3.36", "0.2269", "2.73"
   "LS10_GALDEPTH_G", "1.04", "2.94", "0.3092", "2.83"
   "LS10_EBV", "0.83", "4.63", "0.3965", "5.59"
   "LS10_PSFSIZE_R", "0.31", "1.01", "0.7332", "3.21"
   "LS10_GALDEPTH_Z", "0.22", "0.60", "0.8254", "2.78"

Calibrated on 400 realisations; p is per template, with floor 0.0025.

Fitted amplitudes (NSIDE 128)
-----------------------------

.. csv-table::
   :header: "template", "a OLS", "a ElasticNet", "a ISD-1", "a ISD-3", "a MCMC-add", "a MCMC-comb", "b MCMC-comb"

   "GAIA_nstar_faint", "+0.0143", "-0.0000", "+0.0000", "+0.0000", "+0.0143", "+0.0147", "+0.0012"
   "GAIA_nstar_medium", "-0.0315", "-0.0176", "-0.0171", "+0.0000", "-0.0315", "-0.0321", "-0.0250"
   "GAIA_phot_bp_mean_flux", "-0.0109", "-0.0036", "+0.0000", "+0.0000", "-0.0109", "-0.0097", "+0.0105"
   "GAIA_phot_g_mean_flux", "+0.0177", "-0.0000", "+0.0000", "+0.0000", "+0.0177", "+0.0117", "-0.0093"
   "GAIA_phot_rp_mean_flux", "-0.0252", "-0.0115", "-0.0130", "-0.0132", "-0.0252", "-0.0175", "-0.0026"
   "LS10_EBV", "-0.0087", "-0.0058", "+0.0000", "+0.0000", "-0.0087", "-0.0078", "+0.0018"
   "LS10_GALDEPTH_G", "-0.0056", "-0.0000", "+0.0000", "+0.0000", "-0.0056", "-0.0061", "+0.0186"
   "LS10_GALDEPTH_R", "+0.0139", "+0.0032", "+0.0000", "+0.0000", "+0.0139", "+0.0173", "+0.0025"
   "LS10_GALDEPTH_Z", "+0.0010", "+0.0000", "+0.0000", "+0.0000", "+0.0010", "-0.0099", "-0.0327"
   "LS10_NOBS_R", "-0.0078", "-0.0000", "+0.0000", "+0.0000", "-0.0078", "-0.0066", "-0.0206"
   "LS10_PSFSIZE_R", "-0.0016", "-0.0000", "+0.0000", "+0.0000", "-0.0016", "-0.0011", "+0.0190"

Amplitudes are per unit template standard deviation on the footprint.

Weights
-------

.. csv-table::
   :header: "column", "min", "max", "1 %", "99 %", "clipped", "identically 1"

   "WEIGHT_OLS", "0.554", "1.749", "0.947", "1.104", "0.000 %", "no"
   "WEIGHT_ENET", "0.911", "1.984", "0.965", "1.085", "0.000 %", "no"
   "WEIGHT_ISD1", "0.959", "1.126", "0.959", "1.088", "0.000 %", "no"
   "WEIGHT_ISD3", "0.956", "1.063", "0.956", "1.063", "0.000 %", "no"
   "WEIGHT_ADD", "0.050", "3.038", "0.926", "1.121", "0.000 %", "no"
   "WEIGHT_COMB", "0.050", "2.477", "0.920", "1.120", "0.003 %", "no"

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0128_weight_map.png
   :width: 95%

   Weight maps at NSIDE 128, one panel per method.

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0128_weight_hist.png
   :width: 70%

   Weight distributions.

Angular correlation function
----------------------------

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228_NSIDE0128_wtheta.png
   :width: 80%

   Observed and corrected w(θ), full cross-template correction.

.. csv-table:: Corrected over observed w(θ).
   :header: "method", "at 30′", "smallest ratio", "at θ"

   "OLS", "0.989", "0.904", "272′"
   "ElasticNet", "0.993", "0.931", "272′"
   "ISD-1", "0.996", "0.958", "272′"
   "ISD-3", "1.000", "0.999", "272′"
   "MCMC-add", "0.989", "0.903", "272′"
   "MCMC-comb", "0.987", "0.914", "272′"

Likelihood ratio
----------------

.. csv-table::
   :header: "NSIDE", "λ LR", "mock p", "null mean / max", "N"

   "32", "1029.2", "0.020", "37.9 / 110.3", "50"
   "64", "74.1", "0.373", "57.3 / 120.7", "50"

Resolution comparison
---------------------

.. csv-table:: Residual scatter :math:`\hat\sigma`.
   :header: "method", "NSIDE 32", "NSIDE 64", "NSIDE 128"

   "OLS", "0.3238", "0.3089", "0.4477"
   "ElasticNet", "0.3248", "0.3092", "0.4479"
   "ISD-1", "0.3251", "0.3096", "0.4481"
   "ISD-3", "0.3251", "0.3103", "0.4483"
   "MCMC-add", "0.3241", "0.3090", "0.4478"
   "MCMC-comb", "0.3040", "0.3138", "0.4672"
