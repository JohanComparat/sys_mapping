sys_mapping.simulation
======================

``sys_mapping.simulation`` injects systematic contamination into catalogues, reads and writes them as FITS, and measures the recovery of :math:`w(\theta)`,
for simulation-based validation:

1. Catalogues: :func:`~sys_mapping.simulation.load_uchuu_mock` reads the Uchuu lightcone FITS files;
   GLASS catalogues come from :mod:`sys_mapping.glass_mocks`.
2. Templates: :func:`~sys_mapping.simulation.load_systematic_maps` reads the LSDR10 HEALPix systematic maps.
3. Contamination grid: :func:`~sys_mapping.simulation.make_contamination_grid` returns 9 :class:`~sys_mapping.simulation.ContaminationConfig` objects
   spanning 3 amplitude levels × 3 contamination scenarios.
4. Injection: :func:`~sys_mapping.simulation.inject_systematics` computes the per-galaxy weights
   :math:`\texttt{WEIGHT\_CONT}(p) = (1+\delta_{\rm cont}(p))/(1+\delta_g(p))`.
5. FITS input and output: :func:`~sys_mapping.simulation.save_simulation_catalog` and :func:`~sys_mapping.simulation.load_simulation_catalog`.
6. Recovery: :func:`~sys_mapping.simulation.run_wtheta_recovery` measures the true, contaminated and recovered :math:`w(\theta)` for OLS, ISD-1 and ElasticNet.

.. automodule:: sys_mapping.simulation
   :members:
   :show-inheritance:
   :exclude-members: level,scenario,a_true,b_true
