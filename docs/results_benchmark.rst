Benchmarks: how long each stage takes
======================================

.. note::
   Generated from ``docs/_static/benchmark/benchmarks.csv`` by
   ``docs/generate_benchmark_page.py``.  The measurements are produced by the
   `sys_mapping_benchmark <https://github.com/JohanComparat/sys_mapping_benchmark>`_
   repository, which is kept separate so this package's CI does not carry the
   168 timing cases below.

Provenance
----------

Measured on 11th Gen Intel(R) Core(TM) i9-11900H @ 2.50GHz (16 logical cores, 62.5 GB RAM), Python 3.11.15, JAX 0.10.0 on the ``cpu`` backend with 64-bit precision enabled; NumPy 2.4.3, SciPy 1.17.1, healpy 1.19.0, BlackJAX 1.5, emcee 3.1.6.  Commit ``cf6b12a``, 2026-09-04T12:31:21+00:00.

One-minute load average at the start of the run was 1.32 on 16 cores — the machine was effectively idle.

Each entry is the **median** over repeated calls with JIT warm-up excluded;
median rather than mean because JIT stragglers and scheduler noise are
one-sided and inflate the mean.


Per-function costs
------------------

Numerically hot public API, JIT warm-up excluded.

.. csv-table::
   :header: "Operation", "NSIDE 16, n_s=5", "NSIDE 16, n_s=11", "NSIDE 32, n_s=5", "NSIDE 32, n_s=11", "NSIDE 64, n_s=5", "NSIDE 64, n_s=11"
   :widths: 34, 11, 11, 11, 11, 11, 11

   "``apply_contamination``", "241 µs", "325 µs", "281 µs", "392 µs", "351 µs", "382 µs"
   "``invert_contamination``", "316 µs", "355 µs", "317 µs", "382 µs", "474 µs", "481 µs"
   "``log_likelihood[gauss]``", "30.6 µs", "33.5 µs", "91.5 µs", "120 µs", "579 µs", "515 µs"
   "``log_likelihood[skew]``", "172 µs", "119 µs", "248 µs", "248 µs", "1.03 ms", "1.11 ms"
   "``rotate_templates``", "32.3 µs", "89 µs", "90.5 µs", "185 µs", "388 µs", "625 µs"
   "``debias_params``", "2.43 µs", "4.52 µs", "2.33 µs", "2.85 µs", "3.03 µs", "2.7 µs"
   "``compute_two_point_correction``", "215 µs", "327 µs", "220 µs", "265 µs", "270 µs", "239 µs"
   "``compute_covariance_matrix``", "13.5 µs", "35.8 µs", "51.5 µs", "117 µs", "253 µs", "469 µs"
   "``likelihood_ratio_test``", "98.3 ms", "107 ms", "108 ms", "122 ms", "113 ms", "123 ms"
   "``block_bootstrap_variance[B=20,K=8]``", "2.92 ms", "4.59 ms", "8.71 ms", "16.1 ms", "33.5 ms", "70.7 ms"
   "``jackknife_covariance[K=8]``", "1.28 ms", "2.16 ms", "4.24 ms", "8.21 ms", "17.1 ms", "37.3 ms"
   "``measure_pseudo_cl``", "453 µs", "369 µs", "1.35 ms", "1.46 ms", "12.1 ms", "7.14 ms"


HEALPix map utilities
---------------------

``pixelize_catalog`` uses :math:`10^5` galaxies.

.. csv-table::
   :header: "Operation", "NSIDE 16, n_s=5", "NSIDE 16, n_s=11", "NSIDE 32, n_s=5", "NSIDE 32, n_s=11", "NSIDE 64, n_s=5", "NSIDE 64, n_s=11"
   :widths: 34, 11, 11, 11, 11, 11, 11

   "``systematic_power_spectrum``", "7.11 µs", "6.74 µs", "7.11 µs", "7.65 µs", "8.13 µs", "8.52 µs"
   "``generate_systematic_map``", "173 µs", "156 µs", "446 µs", "470 µs", "1.98 ms", "1.96 ms"
   "``generate_systematic_maps``", "822 µs", "850 µs", "2.22 ms", "2.49 ms", "9.55 ms", "9.94 ms"
   "``pixelize_catalog[1e5 gal]``", "3.14 ms", "3.01 ms", "3.07 ms", "3.35 ms", "3.32 ms", "3.55 ms"
   "``compute_overdensity``", "17.1 µs", "17.7 µs", "37.8 µs", "41.3 µs", "168 µs", "181 µs"
   "``assign_template_values``", "35.8 µs", "37.8 µs", "146 µs", "187 µs", "595 µs", "814 µs"


Stage-1 pre-selection
---------------------

All four ranking statistics of :func:`~sys_mapping.diagnostics.snr_template_ranking`.

.. csv-table::
   :header: "Operation", "NSIDE 16, n_s=5", "NSIDE 16, n_s=11", "NSIDE 32, n_s=5", "NSIDE 32, n_s=11", "NSIDE 64, n_s=5", "NSIDE 64, n_s=11"
   :widths: 34, 11, 11, 11, 11, 11, 11

   "``snr_template_ranking[template]``", "138 µs", "163 µs", "222 µs", "449 µs", "546 µs", "1.06 ms"
   "``snr_template_ranking[data]``", "137 µs", "192 µs", "239 µs", "335 µs", "796 µs", "952 µs"
   "``snr_template_ranking[isd]``", "332 µs", "370 µs", "477 µs", "1.03 ms", "1.68 ms", "3.2 ms"
   "``snr_template_ranking[isd,poly3]``", "731 µs", "628 µs", "923 µs", "1.62 ms", "2.25 ms", "4.26 ms"


Stage-2 decontamination
-----------------------

End-to-end per call to :func:`~sys_mapping.regression.run_decontamination`.

.. csv-table::
   :header: "Operation", "NSIDE 16, n_s=5", "NSIDE 16, n_s=11", "NSIDE 32, n_s=5", "NSIDE 32, n_s=11", "NSIDE 64, n_s=5", "NSIDE 64, n_s=11"
   :widths: 34, 11, 11, 11, 11, 11, 11

   "``OLS``", "71.9 µs", "159 µs", "231 µs", "643 µs", "1 ms", "2.91 ms"
   "``ISD-1``", "1.48 ms", "1.96 ms", "2.52 ms", "7.05 ms", "9.12 ms", "28.5 ms"
   "``ElasticNet``", "45 ms", "72.7 ms", "67.5 ms", "95 ms", "75.5 ms", "123 ms"
   "``ISD-3``", "22 ms", "25.6 ms", "99.5 ms", "2.32 s", "361 ms", "2.26 s"
   "``MCMC-add``", "56.7 ms", "80.4 ms", "52.4 ms", "75.9 ms", "44.8 ms", "72.4 ms"
   "``MCMC-comb``", "6.52 s", "35 s", "22.2 s", "33.8 s", "115 s", "169 s"


Reading these
-------------

* The JAX kernels are **dispatch-dominated** at these sizes: the forward and
  inverse contamination models differ by one element-wise division yet cost
  almost the same, because both are microseconds of arithmetic behind a fixed
  dispatch overhead.
* ``likelihood_ratio_test`` is far more expensive than two likelihood
  evaluations because **each call compiles two new likelihood functions**.
  Build them once with :func:`~sys_mapping.likelihood.make_log_likelihood` and
  difference them directly if you are testing repeatedly.
* Stage 2 spans **6.4 orders of magnitude**, from the fastest
  method to the slowest.  This is why
  :download:`run_ls10_analysis.py <../scripts/run_ls10_analysis.py>` runs
  the methods fastest-first and can checkpoint after the fast phase.
* Stage 1 ranking costs between 137 µs and 4.26 ms per
  call, so pre-selection cost is dominated by the GLASS mock null, which is
  embarrassingly parallel (``preselect_n_jobs``).

