# Performance measurements behind 1.4.0

Measured on the development laptop (16 cores, CPU JAX, 64-bit), 2026-09-15, on prototypes
of the changes listed; the scripts compare against the package as it was at `a8f0304`.
Interleave comparisons when the machine is shared.

| Change | Script | Result | Parity |
|---|---|---|---|
| `lrt_from_maxima`, the default LRT null of `run_ls10_analysis.py` | `proto_batched_lrt.py 32 8 1000 11` | 8 mocks, NSIDE 32, 11 templates: 3.7 s first call, 0.6 s repeat, against 49.5 s for NUTS 1000+1000 per model and refinement | lambda equal to 8e-7 relative; final gradient 4e-12 |
| `draw_null_overdensity`, the default draw of every GLASS null | `proto_pixel_null.py 64 300` | 0.01 s per mock against 0.97 s (123x) | pixel variance 0.03521 vs 0.03523; amplitude scatter ratio median 0.98, range 0.90-1.15 over 11 templates (sampling error 0.08) |
| `template_correlation_matrix` by the polarisation identity | `proto_polarisation_ct.py 300000 11` | 32.9 s against 55.6 s (1.69x) | 2e-8 off-diagonal, 3e-14 diagonal |
| `run_nuts(dense_mass_matrix=True)`, the default | `proto_nuts_dense.py 32 1000` | LS10 log M* >= 10.0, NSIDE 32, 4 chains: wall 44.6 vs 60.5 s; min ESS 531 vs 285; min ESS/s 11.9 vs 4.7; leapfrog steps 14.2 vs 28.4 | medians agree to 4e-3; R-hat 1.007 vs 1.005; 7% divergences in both |
| Array-namespace dispatch (`sys_mapping/_array.py`) for NumPy functions | `tests/test_jax_transformability.py` | `debias_params`, `debias_params_matrix`, `standardise_on_footprint` now trace; 27 of 46 transform cases pass (18 before) | full fast suite unchanged (747 passed) |
| ISD step loop on device (not adopted) | profile of one LS10 cell | ISD-1 0.3 s, ISD-3 < 0.1 s per fit | nothing to gain |

For reference, a profiled LS10 cell (NSIDE 32, 40 significance and 10 ISD realisations,
NUTS 500+500) spends 68% of 728 s in `template_correlation_matrix` and 16% in NUTS;
see `docs/coverage.rst`.
