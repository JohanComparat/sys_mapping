.. _sample-9p0:

BGS VLIM log M* ≥ 9.0, z < 0.08
===============================

523,486 galaxies, issued at NSIDE 32 (93.3 galaxies per pixel over 5,612 pixels). Leading template: ``LS10_GALDEPTH_R`` at 2.88 calibrated, family-wise p 0.0524. Recommended column: ``WEIGHT_SYS``; the NSIDE 32 likelihood ratio requires the multiplicative term (p = 0.020).

.. contents:: On this page
   :local:
   :depth: 1

.. seealso::

   :doc:`results_ls10` — all nine samples.

Template significance
---------------------

.. csv-table::
   :header: "template", "S cal", "S iid", "p", "κ"

   "LS10_GALDEPTH_R", "2.88", "4.86", "0.0075", "1.69"
   "GAIA_nstar_faint", "1.60", "4.57", "0.1072", "2.86"
   "LS10_EBV", "1.58", "5.55", "0.1172", "3.50"
   "GAIA_nstar_medium", "1.43", "4.34", "0.1421", "3.04"
   "LS10_GALDEPTH_G", "1.41", "2.58", "0.1671", "1.83"
   "GAIA_phot_rp_mean_flux", "1.17", "1.07", "0.2419", "0.92"
   "GAIA_phot_g_mean_flux", "0.93", "0.89", "0.3591", "0.96"
   "LS10_GALDEPTH_Z", "0.83", "1.52", "0.3965", "1.82"
   "GAIA_phot_bp_mean_flux", "0.68", "0.71", "0.5187", "1.03"
   "LS10_PSFSIZE_R", "0.36", "0.86", "0.7057", "2.37"
   "LS10_NOBS_R", "0.33", "0.60", "0.7307", "1.82"

Calibrated on 400 realisations; p is per template, with floor 0.0025.

Fitted amplitudes (NSIDE 32)
----------------------------

.. csv-table::
   :header: "template", "a OLS", "a ElasticNet", "a ISD-1", "a ISD-3", "a MCMC-add", "a MCMC-comb", "b MCMC-comb"

   "GAIA_nstar_faint", "+0.1544", "+0.0000", "+0.0000", "+0.0000", "+0.1543", "+0.1407", "+0.4726"
   "GAIA_nstar_medium", "-0.1494", "+0.0000", "+0.0000", "+0.0000", "-0.1494", "-0.1331", "-0.4381"
   "GAIA_phot_bp_mean_flux", "-0.0128", "+0.0000", "+0.0000", "+0.0000", "-0.0127", "-0.0156", "-0.0310"
   "GAIA_phot_g_mean_flux", "+0.0235", "+0.0000", "+0.0000", "+0.0000", "+0.0235", "+0.0354", "+0.0671"
   "GAIA_phot_rp_mean_flux", "-0.0259", "+0.0000", "+0.0000", "+0.0000", "-0.0260", "-0.0325", "-0.0922"
   "LS10_EBV", "-0.0510", "+0.0000", "-0.0637", "-0.0618", "-0.0510", "-0.0518", "-0.0857"
   "LS10_GALDEPTH_G", "-0.0239", "+0.0000", "+0.0000", "+0.0000", "-0.0239", "-0.0097", "-0.0382"
   "LS10_GALDEPTH_R", "+0.0600", "+0.0000", "+0.0000", "+0.0000", "+0.0600", "+0.0479", "+0.0971"
   "LS10_GALDEPTH_Z", "-0.0132", "+0.0000", "+0.0000", "+0.0000", "-0.0132", "-0.0122", "+0.0614"
   "LS10_NOBS_R", "-0.0062", "+0.0000", "+0.0000", "+0.0000", "-0.0062", "+0.0079", "-0.0412"
   "LS10_PSFSIZE_R", "-0.0068", "+0.0000", "+0.0000", "+0.0000", "-0.0068", "-0.0024", "+0.0735"

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

   "OLS", "0.973", "0.904", "272′"
   "ElasticNet", "1.000", "1.000", "1′"
   "ISD-1", "0.989", "0.938", "272′"
   "ISD-3", "0.989", "0.941", "272′"
   "MCMC-add", "0.972", "0.902", "272′"
   "MCMC-comb", "0.947", "0.891", "272′"

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
