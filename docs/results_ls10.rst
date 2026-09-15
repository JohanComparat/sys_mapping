Results: systematic weights
===========================

Per-galaxy systematic weights for the nine LS10 BGS volume-limited stellar-mass
threshold samples, computed by ``scripts/run_ls10_analysis.py`` with 11 templates
(Gaia DR3 star counts and fluxes; LS10 extinction, depth, exposure count and PSF
size). Each sample is issued at the finest NSIDE whose mean occupancy reaches 25
galaxies per pixel. :doc:`results_ls10_recommendations` gives the column to use per
sample, and :doc:`pipeline_ls10` how to reproduce the products.

.. contents:: On this page
   :local:
   :depth: 1

Issued products
---------------

.. csv-table::
   :header: "log M* ≥", "z <", "N gal", "NSIDE", "N pix", "galaxies / pixel", "WEIGHT_SYS 1–99 %", "clipped"

   "9.0", "0.08", "523,486", "32", "5,612", "93.3", "0.737–1.268", "0.006 %"
   "9.5", "0.12", "1,432,502", "64", "21,637", "66.2", "0.751–1.158", "0.000 %"
   "10.0", "0.18", "2,759,238", "128", "84,860", "32.5", "0.903–1.157", "0.015 %"
   "10.25", "0.22", "3,308,841", "128", "84,831", "39.0", "0.886–1.153", "0.028 %"
   "10.5", "0.26", "3,263,228", "128", "84,811", "38.5", "0.920–1.120", "0.003 %"
   "10.75", "0.31", "2,802,710", "128", "84,824", "33.0", "0.902–1.132", "0.021 %"
   "11.0", "0.35", "1,619,838", "64", "21,646", "74.8", "0.912–1.114", "0.004 %"
   "11.25", "0.35", "541,855", "64", "21,555", "25.1", "0.871–1.124", "0.023 %"
   "11.5", "0.35", "120,882", "16", "1,465", "82.5", "0.853–1.126", "0.000 %"

Every file records ``WEIGHTVER`` = 3 and ``TPLBASIS`` = ``footprint``. *Clipped* is the fraction of galaxies at the weight clip [1/20, 20].

The two-point correction uses the full cross-template matrix
:math:`\xi_{ij}(\theta)`, measured from the template values the galaxies carry.

Detection of systematics
------------------------

Each template amplitude is scored against its scatter over uncontaminated GLASS
realisations drawn with the sample's matched spectrum
(``sys_mapping.calibrated_template_significance``). The family-wise p compares the
largest significance over the templates with the largest of each realisation, and
is bounded below by 1/(N+1). κ is the calibrated error over the independent-pixel
error.

.. csv-table::
   :header: "log M* ≥", "NSIDE", "leading template", "S cal", "S iid", "family-wise p", "κ median", "κ range", "N"

   "9.0", "32", "LS10_GALDEPTH_R", "2.88", "4.86", "0.0524", "1.8", "0.9–3.5", "400"
   "9.5", "64", "LS10_GALDEPTH_R", "5.55", "12.46", "≤ 0.0025", "2.4", "0.9–4.2", "400"
   "10.0", "128", "GAIA_phot_rp_mean_flux", "4.19", "4.16", "≤ 0.0025", "2.9", "1.0–5.2", "400"
   "10.25", "128", "GAIA_phot_rp_mean_flux", "4.81", "4.95", "≤ 0.0025", "2.7", "1.0–4.9", "400"
   "10.5", "128", "GAIA_phot_rp_mean_flux", "6.33", "6.18", "≤ 0.0025", "2.7", "1.0–5.6", "400"
   "10.75", "128", "GAIA_phot_rp_mean_flux", "7.69", "7.43", "≤ 0.0025", "2.5", "1.0–5.4", "400"
   "11.0", "64", "GAIA_phot_rp_mean_flux", "4.37", "4.15", "≤ 0.0025", "2.0", "1.0–3.7", "400"
   "11.25", "64", "GAIA_phot_g_mean_flux", "3.80", "4.08", "≤ 0.0025", "1.8", "1.0–2.9", "400"
   "11.5", "16", "LS10_GALDEPTH_G", "1.23", "1.57", "0.8903", "1.3", "1.0–2.2", "400"

Additive or combined model: likelihood ratio
--------------------------------------------

Both models are refined to their likelihood maxima, and the statistic is ranked
against 50 uncontaminated realisations of the sample's matched spectrum fitted the
same way. The Wilks p assumes independent pixels and is shown for contrast.

.. csv-table::
   :header: "log M* ≥", "NSIDE", "λ LR", "Wilks p", "mock p", "null min / mean / max", "rejects additive"

   "9.0", "32", "477.5", "2.0e-95", "0.020", "20.7 / 85.5 / 208.6", "**yes**"
   "9.5", "32", "691.1", "4.4e-141", "0.020", "17.4 / 65.7 / 149.8", "**yes**"
   "10.0", "32", "696.7", "2.7e-142", "0.020", "12.8 / 43.6 / 106.0", "**yes**"
   "10.25", "32", "836.9", "2.3e-172", "0.020", "8.7 / 42.4 / 107.6", "**yes**"
   "10.5", "32", "1029.2", "9.9e-214", "0.020", "12.3 / 37.9 / 110.3", "**yes**"
   "10.75", "32", "1209.0", "1.9e-252", "0.020", "7.6 / 36.0 / 111.1", "**yes**"
   "11.0", "32", "1022.0", "3.6e-212", "0.020", "7.1 / 39.8 / 103.9", "**yes**"
   "11.25", "32", "636.5", "2.1e-129", "0.020", "10.5 / 34.4 / 106.5", "**yes**"
   "11.5", "32", "397.5", "2.1e-78", "0.020", "5.4 / 33.2 / 87.1", "**yes**"
   "9.0", "64", "1503.4", "0.0e+00", "0.020", "60.3 / 152.1 / 396.9", "**yes**"
   "9.5", "64", "741.3", "7.3e-152", "0.020", "37.6 / 100.6 / 267.8", "**yes**"
   "10.0", "64", "179.0", "1.7e-32", "0.020", "15.2 / 64.4 / 133.4", "**yes**"
   "10.25", "64", "116.3", "1.0e-19", "0.078", "9.7 / 61.3 / 135.9", "no"
   "10.5", "64", "74.1", "2.0e-11", "0.373", "17.1 / 57.3 / 120.7", "no"
   "10.75", "64", "114.1", "2.8e-19", "0.020", "9.1 / 42.9 / 102.9", "**yes**"
   "11.0", "64", "79.6", "1.8e-12", "0.078", "12.1 / 37.2 / 125.8", "no"
   "11.25", "64", "127.4", "5.8e-22", "0.020", "16.4 / 47.2 / 108.9", "**yes**"
   "11.5", "64", "154.8", "1.5e-27", "0.451", "15.2 / 163.1 / 546.3", "no"

The likelihood-ratio grids use templates standardised over each map's own valid
region rather than over the footprint; on the footprint basis the NSIDE 64 statistic
changes by a median of 0.7 %.

Corrected angular correlation function
--------------------------------------

.. figure:: /_static/results_ls10/wtheta_ratio_occupancy.png
   :width: 95%

   Corrected over observed w(θ), each sample at its issued resolution.

.. csv-table::
   :header: "log M* ≥", "NSIDE", "MCMC-comb at 30′", "OLS min", "ElasticNet min", "ISD-1 min", "ISD-3 min", "MCMC-add min", "MCMC-comb min"

   "9.0", "32", "0.947", "0.904 (272′)", "1.000 (1′)", "0.938 (272′)", "0.941 (272′)", "0.902 (272′)", "0.891 (272′)"
   "9.5", "64", "0.954", "0.934 (272′)", "0.940 (272′)", "1.000 (1′)", "0.976 (272′)", "0.933 (272′)", "0.940 (272′)"
   "10.0", "128", "0.991", "0.972 (272′)", "0.976 (272′)", "1.000 (272′)", "1.000 (272′)", "0.972 (272′)", "0.959 (272′)"
   "10.25", "128", "0.990", "0.943 (272′)", "0.949 (272′)", "1.000 (272′)", "1.000 (272′)", "0.942 (272′)", "0.937 (272′)"
   "10.5", "128", "0.987", "0.904 (272′)", "0.931 (272′)", "0.958 (272′)", "0.999 (272′)", "0.903 (272′)", "0.914 (272′)"
   "10.75", "128", "0.981", "0.846 (272′)", "0.891 (272′)", "0.943 (272′)", "0.952 (272′)", "0.845 (272′)", "0.855 (272′)"
   "11.0", "64", "0.977", "0.763 (220′)", "0.817 (220′)", "0.858 (220′)", "0.968 (220′)", "0.756 (220′)", "0.765 (220′)"
   "11.25", "64", "0.979", "0.785 (220′)", "0.819 (220′)", "0.960 (220′)", "0.954 (220′)", "0.778 (220′)", "0.803 (220′)"
   "11.5", "16", "0.957", "0.784 (272′)", "0.872 (272′)", "0.955 (272′)", "0.947 (272′)", "0.704 (272′)", "0.583 (272′)"

*min* is the smallest corrected/observed ratio over the 30 bins and the separation
it occurs at.

Resolution comparison
---------------------

The same fits at NSIDE 32, 64 and 128. Coarser pixels hold more galaxies, so
:math:`\hat\sigma` tracks shot noise and does not select a resolution.

.. csv-table:: Residual scatter :math:`\hat\sigma` of the additive model.
   :header: "log M* ≥", "NSIDE 32", "NSIDE 64", "NSIDE 128"

   "9.0", "0.5475", "0.6761", "0.9914"
   "9.5", "0.4709", "0.5240", "0.7459"
   "10.0", "0.3805", "0.3970", "0.5606"
   "10.25", "0.3421", "0.3434", "0.4900"
   "10.5", "0.3241", "0.3090", "0.4478"
   "10.75", "0.3061", "0.2831", "0.4191"
   "11.0", "0.2975", "0.2974", "0.4581"
   "11.25", "0.3313", "0.3843", "0.6413"
   "11.5", "0.4458", "0.6449", "1.3895"

.. csv-table:: Residual scatter :math:`\hat\sigma` of the combined model.
   :header: "log M* ≥", "NSIDE 32", "NSIDE 64", "NSIDE 128"

   "9.0", "0.5384", "0.6882", "1.0403"
   "9.5", "0.4539", "0.5228", "0.7730"
   "10.0", "0.3650", "0.3983", "0.5957"
   "10.25", "0.3266", "0.3501", "0.5189"
   "10.5", "0.3040", "0.3138", "0.4672"
   "10.75", "0.2826", "0.2908", "0.4461"
   "11.0", "0.2808", "0.3074", "0.4818"
   "11.25", "0.3199", "0.3931", "0.6570"
   "11.5", "0.4379", "0.6627", "1.4987"

Per-sample pages
----------------

* :doc:`results_ls10_9p0`
* :doc:`results_ls10_9p5`
* :doc:`results_ls10_10p0`
* :doc:`results_ls10_10p25`
* :doc:`results_ls10_10p5`
* :doc:`results_ls10_10p75`
* :doc:`results_ls10_11p0`
* :doc:`results_ls10_11p25`
* :doc:`results_ls10_11p5`
