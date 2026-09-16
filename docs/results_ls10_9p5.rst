.. _sample-9p5:

BGS VLIM log M* ≥ 9.5, z < 0.12
===============================

1,432,502 galaxies, issued at NSIDE 64 (66.2 galaxies per pixel over 21,637 pixels). Leading template: ``LS10_GALDEPTH_R`` at 5.63 calibrated, family-wise p ≤ 0.0025. Recommended column: ``WEIGHT_SYS``; a template is detected (family-wise p ≤ 0.0025) and the NSIDE 64 likelihood ratio requires the multiplicative term (p = 0.020).

.. contents:: On this page
   :local:
   :depth: 1

.. seealso::

   :doc:`results_ls10` — all nine samples.

Template significance
---------------------

.. csv-table::
   :header: "template", "S cal", "S iid", "p", "κ"

   "LS10_GALDEPTH_R", "5.63", "12.46", "0.0025", "2.21"
   "LS10_GALDEPTH_G", "2.15", "5.25", "0.0299", "2.44"
   "GAIA_nstar_medium", "2.03", "6.96", "0.0524", "3.44"
   "GAIA_nstar_faint", "1.79", "5.87", "0.0748", "3.28"
   "GAIA_phot_bp_mean_flux", "1.59", "1.59", "0.1172", "1.00"
   "LS10_EBV", "1.00", "4.21", "0.3591", "4.21"
   "LS10_NOBS_R", "0.99", "2.36", "0.3167", "2.38"
   "LS10_PSFSIZE_R", "0.84", "2.73", "0.4140", "3.27"
   "GAIA_phot_g_mean_flux", "0.77", "0.77", "0.4364", "1.00"
   "GAIA_phot_rp_mean_flux", "0.65", "0.58", "0.5387", "0.89"
   "LS10_GALDEPTH_Z", "0.19", "0.41", "0.8354", "2.20"

Calibrated on 400 realisations; p is per template, with floor 0.0025.

Fitted amplitudes (NSIDE 64)
----------------------------

.. csv-table::
   :header: "template", "a OLS", "a ElasticNet", "a ISD-1", "a ISD-3", "a MCMC-add", "a MCMC-comb", "b MCMC-comb"

   "GAIA_nstar_faint", "+0.0853", "+0.0600", "+0.0000", "+0.0000", "+0.0853", "+0.0196", "+0.0720"
   "GAIA_nstar_medium", "-0.1004", "-0.0764", "+0.0000", "+0.0000", "-0.1004", "-0.0449", "-0.0217"
   "GAIA_phot_bp_mean_flux", "-0.0119", "-0.0085", "+0.0000", "+0.0000", "-0.0119", "-0.0152", "+0.0164"
   "GAIA_phot_g_mean_flux", "+0.0086", "+0.0000", "+0.0000", "+0.0000", "+0.0086", "+0.0094", "-0.0110"
   "GAIA_phot_rp_mean_flux", "-0.0056", "-0.0000", "+0.0000", "+0.0000", "-0.0056", "-0.0085", "-0.0242"
   "LS10_EBV", "-0.0186", "-0.0170", "+0.0000", "-0.0347", "-0.0186", "-0.0026", "+0.0075"
   "LS10_GALDEPTH_G", "-0.0235", "-0.0216", "+0.0000", "+0.0000", "-0.0235", "-0.0049", "-0.0380"
   "LS10_GALDEPTH_R", "+0.0724", "+0.0689", "+0.0000", "+0.0000", "+0.0724", "+0.0298", "+0.0583"
   "LS10_GALDEPTH_Z", "+0.0017", "+0.0007", "+0.0000", "+0.0000", "+0.0017", "+0.0224", "+0.0377"
   "LS10_NOBS_R", "-0.0115", "-0.0087", "+0.0000", "+0.0000", "-0.0115", "-0.0153", "+0.0039"
   "LS10_PSFSIZE_R", "+0.0104", "+0.0095", "+0.0000", "+0.0000", "+0.0104", "+0.0080", "+0.0201"

Amplitudes are per unit template standard deviation on the footprint.

Weights
-------

.. csv-table::
   :header: "column", "min", "max", "1 %", "99 %", "clipped", "identically 1"

   "WEIGHT_OLS", "0.497", "1.754", "0.747", "1.144", "0.000 %", "no"
   "WEIGHT_ENET", "0.505", "1.501", "0.752", "1.130", "0.000 %", "no"
   "WEIGHT_ISD1", "1.000", "1.000", "1.000", "1.000", "0.000 %", "yes"
   "WEIGHT_ISD3", "0.951", "1.097", "0.951", "1.097", "0.000 %", "no"
   "WEIGHT_ADD", "0.050", "2.571", "0.762", "1.173", "0.014 %", "no"
   "WEIGHT_COMB", "0.050", "6.334", "0.751", "1.158", "0.000 %", "no"

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0064_weight_map.png
   :width: 95%

   Weight maps at NSIDE 64, one panel per method.

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0064_weight_hist.png
   :width: 70%

   Weight distributions.

Angular correlation function
----------------------------

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502_NSIDE0064_wtheta.png
   :width: 80%

   Observed and corrected w(θ), full cross-template correction.

.. csv-table:: Corrected over observed w(θ).
   :header: "method", "at 30′", "smallest ratio", "at θ"

   "OLS", "0.963", "0.949", "272′"
   "ElasticNet", "0.967", "0.955", "272′"
   "ISD-1", "1.005", "1.000", "1′"
   "ISD-3", "1.000", "0.991", "272′"
   "MCMC-add", "0.963", "0.949", "272′"
   "MCMC-comb", "0.991", "0.981", "272′"

Likelihood ratio
----------------

.. csv-table::
   :header: "NSIDE", "λ LR", "mock p", "null mean / max", "N"

   "32", "691.1", "0.020", "65.7 / 149.8", "50"
   "64", "741.3", "0.020", "100.6 / 267.8", "50"

Resolution comparison
---------------------

.. csv-table:: Residual scatter :math:`\hat\sigma`.
   :header: "method", "NSIDE 32", "NSIDE 64", "NSIDE 128"

   "OLS", "0.4703", "0.5238", "0.7458"
   "ElasticNet", "0.4704", "0.5239", "0.7462"
   "ISD-1", "0.4732", "0.5267", "0.7475"
   "ISD-3", "0.4731", "0.5266", "0.7469"
   "MCMC-add", "0.4709", "0.5240", "0.7459"
   "MCMC-comb", "0.4539", "0.5228", "0.7730"
