.. _sample-9p0:

BGS VLIM log M* ≥ 9.0, z < 0.08
===============================

523,486 galaxies, issued at NSIDE 32 (93.3 galaxies per pixel over 5,612 pixels). Leading template: ``LS10_GALDEPTH_R`` at 2.84 calibrated, family-wise p 0.0698. Recommended column: ``WEIGHT_SYS``; the NSIDE 32 likelihood ratio requires the multiplicative term (p = 0.020).

.. contents:: On this page
   :local:
   :depth: 1

.. seealso::

   :doc:`results_ls10` — all nine samples.

Template significance
---------------------

.. csv-table::
   :header: "template", "S cal", "S iid", "p", "κ"

   "LS10_GALDEPTH_R", "2.84", "4.86", "0.0100", "1.71"
   "GAIA_nstar_faint", "1.61", "4.57", "0.0973", "2.84"
   "LS10_EBV", "1.59", "5.55", "0.1122", "3.50"
   "GAIA_nstar_medium", "1.43", "4.34", "0.1446", "3.04"
   "LS10_GALDEPTH_G", "1.40", "2.58", "0.1746", "1.84"
   "GAIA_phot_rp_mean_flux", "1.16", "1.07", "0.2369", "0.93"
   "GAIA_phot_g_mean_flux", "0.93", "0.89", "0.3666", "0.96"
   "LS10_GALDEPTH_Z", "0.83", "1.52", "0.4165", "1.84"
   "GAIA_phot_bp_mean_flux", "0.70", "0.71", "0.5062", "1.01"
   "LS10_PSFSIZE_R", "0.36", "0.86", "0.6983", "2.36"
   "LS10_NOBS_R", "0.33", "0.60", "0.7581", "1.82"

Calibrated on 400 realisations; p is per template, with floor 0.0025.

Fitted amplitudes (NSIDE 32)
----------------------------

.. csv-table::
   :header: "template", "a OLS", "a ElasticNet", "a ISD-1", "a ISD-3", "a MCMC-add", "a MCMC-comb", "b MCMC-comb"

   "GAIA_nstar_faint", "+0.1544", "+0.0000", "+0.0000", "+0.0000", "+0.1543", "+0.1328", "+0.4492"
   "GAIA_nstar_medium", "-0.1494", "+0.0000", "+0.0000", "+0.0000", "-0.1494", "-0.1303", "-0.4111"
   "GAIA_phot_bp_mean_flux", "-0.0128", "+0.0000", "+0.0000", "+0.0000", "-0.0127", "-0.0208", "-0.0318"
   "GAIA_phot_g_mean_flux", "+0.0235", "+0.0000", "+0.0000", "+0.0000", "+0.0235", "+0.0279", "+0.0518"
   "GAIA_phot_rp_mean_flux", "-0.0259", "+0.0000", "+0.0000", "+0.0000", "-0.0260", "-0.0203", "-0.0799"
   "LS10_EBV", "-0.0510", "+0.0000", "-0.0637", "-0.0618", "-0.0510", "-0.0506", "-0.0830"
   "LS10_GALDEPTH_G", "-0.0239", "+0.0000", "+0.0000", "+0.0000", "-0.0239", "-0.0058", "-0.0351"
   "LS10_GALDEPTH_R", "+0.0600", "+0.0000", "+0.0000", "+0.0000", "+0.0600", "+0.0447", "+0.0975"
   "LS10_GALDEPTH_Z", "-0.0132", "+0.0000", "+0.0000", "+0.0000", "-0.0132", "-0.0141", "+0.0545"
   "LS10_NOBS_R", "-0.0062", "+0.0000", "+0.0000", "+0.0000", "-0.0062", "+0.0037", "-0.0360"
   "LS10_PSFSIZE_R", "-0.0068", "+0.0000", "+0.0000", "+0.0000", "-0.0068", "-0.0063", "+0.0738"

Amplitudes are per unit template standard deviation on the footprint.

Weights
-------

.. csv-table::
   :header: "column", "min", "max", "1 %", "99 %", "clipped", "identically 1"

   "WEIGHT_OLS", "0.563", "1.604", "0.793", "1.183", "0.000 %", "no"
   "WEIGHT_ENET", "1.000", "1.000", "1.000", "1.000", "0.000 %", "yes"
   "WEIGHT_ISD1", "0.918", "1.133", "0.918", "1.133", "0.000 %", "no"
   "WEIGHT_ISD3", "0.942", "1.144", "0.942", "1.144", "0.000 %", "no"
   "WEIGHT_ADD", "0.050", "3.705", "0.760", "1.229", "0.003 %", "no"
   "WEIGHT_COMB", "0.050", "4.176", "0.737", "1.268", "0.006 %", "no"

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0032_weight_map.png
   :width: 95%

   Weight maps at NSIDE 32, one panel per method.

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0032_weight_hist.png
   :width: 70%

   Weight distributions.

Angular correlation function
----------------------------

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486_NSIDE0032_wtheta.png
   :width: 80%

   Observed and corrected w(θ), full cross-template correction.

.. csv-table:: Corrected over observed w(θ).
   :header: "method", "at 30′", "smallest ratio", "at θ"

   "OLS", "0.983", "0.941", "272′"
   "ElasticNet", "1.010", "1.001", "1′"
   "ISD-1", "0.999", "0.975", "272′"
   "ISD-3", "0.999", "0.979", "272′"
   "MCMC-add", "0.983", "0.941", "272′"
   "MCMC-comb", "0.970", "0.935", "272′"

Likelihood ratio
----------------

.. csv-table::
   :header: "NSIDE", "λ LR", "mock p", "null mean / max", "N"

   "32", "477.5", "0.020", "85.5 / 208.6", "50"
   "64", "1503.4", "0.020", "152.1 / 396.9", "50"

Resolution comparison
---------------------

.. csv-table:: Residual scatter :math:`\hat\sigma`.
   :header: "method", "NSIDE 32", "NSIDE 64", "NSIDE 128"

   "OLS", "0.5469", "0.6759", "0.9913"
   "ElasticNet", "0.5531", "0.6769", "0.9917"
   "ISD-1", "0.5498", "0.6787", "0.9930"
   "ISD-3", "0.5495", "0.6785", "0.9929"
   "MCMC-add", "0.5475", "0.6761", "0.9914"
   "MCMC-comb", "0.5384", "0.6882", "1.0403"
