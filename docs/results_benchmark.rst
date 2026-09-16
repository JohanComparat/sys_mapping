Benchmarks
==========

sys_mapping 1.4.0 (commit ``6f43dcc``), dahu node dahu122, Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz, 2026-09-15.

We time 280 cases of the pipeline with ``benchmark/benchmark_pipeline.py`` of the
`sys_mapping_benchmark <https://github.com/JohanComparat/sys_mapping_benchmark>`_
repository; ``docs/generate_benchmark_page.py`` renders
``docs/_static/benchmark/benchmarks.csv`` into this page.
Each entry is the median over repeated calls, with the JIT compilation of the first call
excluded.
The run used 8 cores, Python 3.11.16, JAX 0.10.2 (``cpu``, 64-bit on), NumPy 2.4.6, SciPy 1.17.1, healpy 1.20.0 and BlackJAX 1.5; the one-minute load average at the start was 24.23.
MCMC-add is the analytic posterior and MCMC-comb NUTS with two chains, 400 warmup steps and 400 draws per chain.


Per-function costs
------------------

Public functions on the hot path.

.. csv-table::
   :header: "Operation", "NSIDE 16, n_s=5", "NSIDE 16, n_s=11", "NSIDE 32, n_s=5", "NSIDE 32, n_s=11", "NSIDE 64, n_s=5", "NSIDE 64, n_s=11", "NSIDE 128, n_s=5", "NSIDE 128, n_s=11", "NSIDE 256, n_s=5", "NSIDE 256, n_s=11"
   :widths: 34, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11

   "``apply_contamination``", "1.22 ms", "1.31 ms", "1.38 ms", "1.36 ms", "1.5 ms", "1.6 ms", "1.88 ms", "1.74 ms", "3.57 ms", "4.49 ms"
   "``invert_contamination``", "1.18 ms", "1.28 ms", "1.36 ms", "1.35 ms", "1.52 ms", "1.61 ms", "1.73 ms", "1.82 ms", "3.97 ms", "5.05 ms"
   "``log_likelihood[gauss]``", "113 µs", "55.5 µs", "139 µs", "136 µs", "474 µs", "760 µs", "910 µs", "1.12 ms", "2.14 ms", "2.73 ms"
   "``log_likelihood[skew]``", "201 µs", "157 µs", "393 µs", "442 µs", "1.68 ms", "1.73 ms", "4.48 ms", "4.25 ms", "15.2 ms", "16 ms"
   "``rotate_templates``", "109 µs", "149 µs", "315 µs", "517 µs", "894 µs", "2.31 ms", "4.12 ms", "4.78 ms", "9.31 ms", "24.1 ms"
   "``debias_params``", "10.3 µs", "8.21 µs", "12.4 µs", "12.1 µs", "12.8 µs", "15.1 µs", "12.9 µs", "7.93 µs", "8.02 µs", "7.24 µs"
   "``compute_two_point_correction``", "1.46 ms", "642 µs", "1.08 ms", "1.08 ms", "1.08 ms", "1.16 ms", "1.03 ms", "620 µs", "622 µs", "607 µs"
   "``compute_covariance_matrix``", "389 µs", "270 µs", "923 µs", "1.43 ms", "3.29 ms", "4.29 ms", "8.22 ms", "13.5 ms", "31 ms", "62.2 ms"
   "``likelihood_ratio_test``", "1.15 ms", "690 µs", "1.51 ms", "1.39 ms", "2.06 ms", "2.17 ms", "3.49 ms", "6.71 ms", "13.6 ms", "29.8 ms"
   "``block_bootstrap_variance[B=20,K=8]``", "11.4 ms", "11.2 ms", "19.9 ms", "32.6 ms", "63.8 ms", "131 ms", "273 ms", "533 ms", "1.25 s", "3.06 s"
   "``jackknife_covariance[K=8]``", "3.14 ms", "5.2 ms", "9.32 ms", "17.3 ms", "32.8 ms", "68.9 ms", "140 ms", "276 ms", "625 ms", "1.58 s"
   "``measure_pseudo_cl``", "518 µs", "477 µs", "1.21 ms", "1.28 ms", "4.82 ms", "4.79 ms", "23.7 ms", "122 ms", "133 ms", "131 ms"


HEALPix map utilities
---------------------

``pixelize_catalog`` bins :math:`10^5` galaxies.

.. csv-table::
   :header: "Operation", "NSIDE 16, n_s=5", "NSIDE 16, n_s=11", "NSIDE 32, n_s=5", "NSIDE 32, n_s=11", "NSIDE 64, n_s=5", "NSIDE 64, n_s=11", "NSIDE 128, n_s=5", "NSIDE 128, n_s=11", "NSIDE 256, n_s=5", "NSIDE 256, n_s=11"
   :widths: 34, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11

   "``systematic_power_spectrum``", "18.6 µs", "18.4 µs", "17.7 µs", "17.3 µs", "19.1 µs", "19.4 µs", "21.2 µs", "22 µs", "26.5 µs", "26.4 µs"
   "``generate_systematic_map``", "330 µs", "312 µs", "735 µs", "747 µs", "2.33 ms", "2.32 ms", "9.16 ms", "9.28 ms", "44.4 ms", "45.2 ms"
   "``generate_systematic_maps``", "1.68 ms", "1.6 ms", "3.93 ms", "3.99 ms", "12.1 ms", "12.1 ms", "48.4 ms", "48.3 ms", "243 ms", "245 ms"
   "``pixelize_catalog[1e5 gal]``", "5.77 ms", "5 ms", "5.06 ms", "5.06 ms", "5.17 ms", "5.18 ms", "5.49 ms", "5.34 ms", "8.32 ms", "8.8 ms"
   "``compute_overdensity``", "37 µs", "35.8 µs", "78.6 µs", "77.3 µs", "331 µs", "357 µs", "1.84 ms", "1.9 ms", "13.9 ms", "14.1 ms"
   "``assign_template_values``", "81.1 µs", "93.3 µs", "268 µs", "329 µs", "1.07 ms", "1.51 ms", "4.26 ms", "5.67 ms", "15.1 ms", "20.4 ms"


Stage-1 pre-selection
---------------------

The ranking statistics of :func:`~sys_mapping.diagnostics.snr_template_ranking`.

.. csv-table::
   :header: "Operation", "NSIDE 16, n_s=5", "NSIDE 16, n_s=11", "NSIDE 32, n_s=5", "NSIDE 32, n_s=11", "NSIDE 64, n_s=5", "NSIDE 64, n_s=11", "NSIDE 128, n_s=5", "NSIDE 128, n_s=11", "NSIDE 256, n_s=5", "NSIDE 256, n_s=11"
   :widths: 34, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11

   "``snr_template_ranking[template]``", "306 µs", "569 µs", "631 µs", "967 µs", "1.29 ms", "1.81 ms", "3.45 ms", "4.78 ms", "11.2 ms", "33.2 ms"
   "``snr_template_ranking[data]``", "484 µs", "395 µs", "809 µs", "1.08 ms", "1.18 ms", "1.41 ms", "3.95 ms", "7.21 ms", "12.4 ms", "42.3 ms"
   "``snr_template_ranking[isd]``", "461 µs", "809 µs", "1.02 ms", "2.37 ms", "3.86 ms", "7.04 ms", "14.2 ms", "40.7 ms", "70.2 ms", "151 ms"
   "``snr_template_ranking[isd,poly3]``", "1.06 ms", "1.19 ms", "1.53 ms", "2.59 ms", "6.79 ms", "13.1 ms", "29.2 ms", "59.8 ms", "100 ms", "201 ms"


Stage-2 decontamination
-----------------------

One call to :func:`~sys_mapping.regression.run_decontamination` per method.

.. csv-table::
   :header: "Operation", "NSIDE 16, n_s=5", "NSIDE 16, n_s=11", "NSIDE 32, n_s=5", "NSIDE 32, n_s=11", "NSIDE 64, n_s=5", "NSIDE 64, n_s=11", "NSIDE 128, n_s=5", "NSIDE 128, n_s=11", "NSIDE 256, n_s=5", "NSIDE 256, n_s=11"
   :widths: 34, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11

   "``OLS``", "314 µs", "401 µs", "559 µs", "1.27 ms", "2.11 ms", "7.08 ms", "8.59 ms", "20.5 ms", "41.5 ms", "122 ms"
   "``ISD-1``", "25.6 ms", "156 ms", "283 ms", "662 ms", "996 ms", "3.13 s", "5.02 s", "17.5 s", "18.9 s", "81.2 s"
   "``ElasticNet``", "116 ms", "180 ms", "144 ms", "177 ms", "187 ms", "387 ms", "524 ms", "499 ms", "843 ms", "2.27 s"
   "``ISD-3``", "64.9 ms", "179 ms", "383 ms", "899 ms", "622 ms", "2.39 s", "4.63 s", "17.3 s", "19 s", "74.1 s"
   "``MCMC-add``", "118 ms", "178 ms", "125 ms", "199 ms", "115 ms", "174 ms", "126 ms", "211 ms", "154 ms", "280 ms"
   "``MCMC-comb``", "10.8 s", "30.2 s", "54.3 s", "98.9 s", "192 s", "326 s", "585 s", "4.05e+03 s", "3.03e+03 s", "1.23e+04 s"


Ranges
------

Stage-2 calls span 7.6 decades, from 314 µs to 1.23e+04 s.
A Stage-1 ranking call takes 306 µs to 201 ms; the cost of pre-selection is the GLASS null, which runs in parallel over ``preselect_n_jobs``.

