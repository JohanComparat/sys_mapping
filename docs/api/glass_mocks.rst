sys_mapping.glass_mocks
=======================

Full-sky lognormal galaxy mock generation using the GLASS package
(Tessore et al. 2023).

Generates systematic-free galaxy catalogs matched to the BGS LS10
redshift distribution and surface density.  The workflow is:

1. **Measure n(z)** from a reference catalog (e.g. Uchuu) with
   :func:`~sys_mapping.glass_mocks.measure_nz`.
2. **Generate GLASS mock** with
   :func:`~sys_mapping.glass_mocks.generate_glass_fullsky_mock`:
   a single tophat redshift shell is used; galaxy positions are drawn from a
   lognormal density field via ``glass.positions_from_delta``, and redshifts
   are sampled from the measured :math:`n(z)`.

**Inputs:** NSIDE, target :math:`N_{\rm gal}`, measured :math:`n(z)`, seed.

**Outputs:** dict with keys ``ra``, ``dec``, ``z``, ``ra_rand``,
``dec_rand``, ``n_total``, ``nside``, ``seed``.

.. automodule:: sys_mapping.glass_mocks
   :members:
   :show-inheritance:
