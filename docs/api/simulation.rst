sys_mapping.simulation
======================

Systematic contamination injection, FITS I/O, and w(θ) recovery.

Provides the end-to-end infrastructure for simulation-based validation:

1. **Load catalogs** — :func:`~sys_mapping.simulation.load_uchuu_mock` reads
   the Uchuu lightcone FITS files; GLASS catalogs come from
   :mod:`sys_mapping.glass_mocks`.
2. **Load templates** — :func:`~sys_mapping.simulation.load_systematic_maps`
   reads LSDR10 HEALPix systematic maps.
3. **Build contamination grid** — :func:`~sys_mapping.simulation.make_contamination_grid`
   returns 9 :class:`~sys_mapping.simulation.ContaminationConfig` objects
   spanning 3 amplitude levels × 3 contamination scenarios.
4. **Inject contamination** — :func:`~sys_mapping.simulation.inject_systematics`
   computes per-galaxy weights
   :math:`\texttt{WEIGHT\_CONT}(p) = (1+\delta_{\rm cont}(p))/(1+\delta_g(p))`.
5. **Save / load FITS** — :func:`~sys_mapping.simulation.save_simulation_catalog`
   and :func:`~sys_mapping.simulation.load_simulation_catalog`.
6. **Full recovery pipeline** — :func:`~sys_mapping.simulation.run_wtheta_recovery`
   measures truth, contaminated, and recovered :math:`w(\theta)` for OLS,
   ISD-1, and ElasticNet.

.. automodule:: sys_mapping.simulation
   :members:
   :show-inheritance:
