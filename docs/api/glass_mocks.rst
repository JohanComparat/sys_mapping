sys_mapping.glass_mocks
=======================

Lognormal GLASS realisations of the galaxy field (Tessore et al. 2023), used as the
null of every calibrated statistic.

* :func:`~sys_mapping.glass_mocks.load_matched_cl` — the validated matched
  spectrum for a sample, at the nearest resolution at or above the one requested
  (the finest below when none is finer). A null built without a matched spectrum or
  an explicit ``cl_amplitude`` warns and uses a power law.
* :func:`~sys_mapping.glass_mocks.draw_null_overdensity` and
  :func:`~sys_mapping.glass_mocks.generate_glass_null_overdensity` — overdensity
  realisations on the footprint pixels, with galaxy counts
  :math:`N_g\sim{\rm Poisson}(\bar n(1+\delta))` and random counts
  :math:`N_r\sim{\rm Poisson}(r\bar n)` per pixel, reduced as
  :func:`~sys_mapping.maps.compute_overdensity` reduces the data. Realisation
  :math:`k` uses seed ``seed + k``. At NSIDE 64 a realisation takes 0.01 s,
  against 0.97 s for a pixelised catalogue, with the same pixel variance.
* :func:`~sys_mapping.glass_mocks.generate_glass_delta_map` — the GLASS
  :math:`\delta` map alone.
* :func:`~sys_mapping.glass_mocks.generate_glass_fullsky_mock` — a full-sky
  galaxy and random catalogue: one tophat redshift shell, positions from
  ``glass.positions_from_delta`` and redshifts from an :math:`n(z)` measured with
  :func:`~sys_mapping.glass_mocks.measure_nz`. It returns a dict with keys ``ra``,
  ``dec``, ``z``, ``ra_rand``, ``dec_rand``, ``n_total``, ``nside``, ``seed``.

.. automodule:: sys_mapping.glass_mocks
   :members:
   :show-inheritance:
