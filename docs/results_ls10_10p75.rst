.. _sample-10p75:

BGS VLIM log M* ≥ 10.75, z < 0.31
=================================

2,802,710 galaxies, issued at NSIDE 128 (33.0 galaxies per pixel over 84,824 pixels). Leading template: ``GAIA_phot_rp_mean_flux`` at 7.69 calibrated, family-wise p ≤ 0.0025. Recommended column: ``WEIGHT_SYS``; a template is detected (family-wise p ≤ 0.0025) and the NSIDE 64 likelihood ratio requires the multiplicative term (p = 0.020).

.. contents:: On this page
   :local:
   :depth: 1

.. seealso::

   :doc:`results_ls10` — all nine samples.

Template significance
---------------------

.. csv-table::
   :header: "template", "S cal", "S iid", "p", "κ"

   "GAIA_phot_rp_mean_flux", "7.69", "7.43", "0.0025", "0.97"
   "GAIA_phot_g_mean_flux", "5.18", "5.25", "0.0025", "1.01"
   "GAIA_phot_bp_mean_flux", "4.53", "4.88", "0.0025", "1.08"
   "GAIA_nstar_medium", "2.10", "6.04", "0.0399", "2.88"
   "LS10_GALDEPTH_R", "1.37", "3.47", "0.1845", "2.53"
   "LS10_EBV", "1.05", "5.72", "0.2893", "5.42"
   "LS10_GALDEPTH_Z", "0.82", "2.19", "0.3940", "2.68"
   "GAIA_nstar_faint", "0.75", "1.91", "0.4414", "2.54"
   "LS10_PSFSIZE_R", "0.62", "1.85", "0.5112", "2.96"
   "LS10_NOBS_R", "0.62", "1.52", "0.5237", "2.46"
   "LS10_GALDEPTH_G", "0.15", "0.40", "0.8703", "2.66"

Calibrated on 400 realisations; p is per template, with floor 0.0025.

Fitted amplitudes (NSIDE 128)
-----------------------------

.. csv-table::
   :header: "template", "a OLS", "a ElasticNet", "a ISD-1", "a ISD-3", "a MCMC-add", "a MCMC-comb", "b MCMC-comb"

   "GAIA_nstar_faint", "+0.0076", "-0.0000", "+0.0000", "+0.0000", "+0.0076", "+0.0086", "-0.0071"
   "GAIA_nstar_medium", "-0.0234", "-0.0158", "-0.0175", "+0.0000", "-0.0234", "-0.0242", "-0.0281"
   "GAIA_phot_bp_mean_flux", "-0.0140", "-0.0050", "+0.0000", "+0.0000", "-0.0140", "-0.0132", "-0.0101"
   "GAIA_phot_g_mean_flux", "+0.0229", "-0.0000", "-0.0111", "-0.0146", "+0.0229", "+0.0213", "+0.0194"
   "GAIA_phot_rp_mean_flux", "-0.0284", "-0.0113", "+0.0000", "+0.0000", "-0.0284", "-0.0251", "-0.0058"
   "LS10_EBV", "-0.0101", "-0.0083", "+0.0000", "-0.0160", "-0.0101", "-0.0102", "-0.0089"
   "LS10_GALDEPTH_G", "+0.0007", "+0.0000", "+0.0000", "+0.0000", "+0.0007", "+0.0019", "+0.0332"
   "LS10_GALDEPTH_R", "+0.0080", "+0.0047", "+0.0000", "+0.0000", "+0.0080", "+0.0117", "-0.0033"
   "LS10_GALDEPTH_Z", "+0.0036", "+0.0014", "+0.0000", "+0.0000", "+0.0036", "-0.0061", "-0.0402"
   "LS10_NOBS_R", "-0.0030", "+0.0000", "+0.0000", "+0.0000", "-0.0030", "-0.0027", "-0.0333"
   "LS10_PSFSIZE_R", "-0.0028", "-0.0007", "+0.0000", "+0.0000", "-0.0028", "-0.0028", "+0.0261"

Amplitudes are per unit template standard deviation on the footprint.

Weights
-------

.. csv-table::
   :header: "column", "min", "max", "1 %", "99 %", "clipped", "identically 1"

   "WEIGHT_OLS", "0.490", "1.780", "0.943", "1.105", "0.000 %", "no"
   "WEIGHT_ENET", "0.882", "2.101", "0.958", "1.086", "0.000 %", "no"
   "WEIGHT_ISD1", "0.958", "1.126", "0.958", "1.084", "0.000 %", "no"
   "WEIGHT_ISD3", "0.939", "1.138", "0.939", "1.105", "0.000 %", "no"
   "WEIGHT_ADD", "0.332", "5.878", "0.924", "1.123", "0.000 %", "no"
   "WEIGHT_COMB", "0.050", "4.551", "0.902", "1.132", "0.021 %", "no"

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0128_weight_map.png
   :width: 95%

   Weight maps at NSIDE 128, one panel per method.

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0128_weight_hist.png
   :width: 70%

   Weight distributions.

Angular correlation function
----------------------------

.. figure:: /_static/results_ls10/issued/LS10_VLIM_ANY_10.75_Mstar_12.0_0.05_z_0.31_N_2802710_NSIDE0128_wtheta.png
   :width: 80%

   Observed and corrected w(θ), full cross-template correction.

.. csv-table:: Corrected over observed w(θ).
   :header: "method", "at 30′", "smallest ratio", "at θ"

   "OLS", "0.985", "0.846", "272′"
   "ElasticNet", "0.989", "0.891", "272′"
   "ISD-1", "0.995", "0.943", "272′"
   "ISD-3", "0.995", "0.952", "272′"
   "MCMC-add", "0.984", "0.845", "272′"
   "MCMC-comb", "0.981", "0.855", "272′"

Likelihood ratio
----------------

.. csv-table::
   :header: "NSIDE", "λ LR", "mock p", "null mean / max", "N"

   "32", "1209.0", "0.020", "36.0 / 111.1", "50"
   "64", "114.1", "0.020", "42.9 / 102.9", "50"

Resolution comparison
---------------------

.. csv-table:: Residual scatter :math:`\hat\sigma`.
   :header: "method", "NSIDE 32", "NSIDE 64", "NSIDE 128"

   "OLS", "0.3057", "0.2831", "0.4190"
   "ElasticNet", "0.3064", "0.2832", "0.4192"
   "ISD-1", "0.3067", "0.2837", "0.4196"
   "ISD-3", "0.3067", "0.2840", "0.4197"
   "MCMC-add", "0.3061", "0.2831", "0.4191"
   "MCMC-comb", "0.2826", "0.2908", "0.4461"
