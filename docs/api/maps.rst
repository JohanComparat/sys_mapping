sys_mapping.maps
================

HEALPix map utilities: pixelisation, overdensity estimation, and
systematic template generation.

**Inputs:** RA/Dec galaxy and random catalog arrays, or HEALPix NSIDE parameter.

**Outputs:** HEALPix count maps, overdensity arrays ``delta_g`` at unmasked
pixels, and template arrays ``delta_t`` of shape ``(n_sys, n_good_pix)``.

Synthetic template families
---------------------------

Five spectral families (0–4) are available through :func:`generate_systematic_map`
and :func:`generate_systematic_maps`.  Each map is drawn from a Gaussian random
field with power spectrum :math:`C_\ell` normalised so that
:math:`\sum_\ell \frac{2\ell+1}{4\pi} C_\ell = 1` (unit variance over the full
sky), then the map itself is rescaled to zero mean and unit standard deviation.

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - Family
     - Name
     - Power spectrum
   * - 0
     - ``exp_linear``
     - :math:`C_\ell \propto e^{-\ell/500}` — large-scale coherent structure
   * - 1
     - ``exp_quadratic``
     - :math:`C_\ell \propto e^{-(\ell/250)^2}` — intermediate-scale artefact
   * - 2
     - ``power_neg2``
     - :math:`C_\ell \propto (\ell+1)^{-2}` — steep power law
   * - 3
     - ``power_neg1``
     - :math:`C_\ell \propto (\ell+1)^{-1}` — shallow power law
   * - 4
     - ``white_noise``
     - :math:`C_\ell = \text{const}` — pixel-scale noise

Real observational templates
-----------------------------

Two loaders bring in physically motivated templates from external FITS files:

:func:`load_real_template`
    Generic loader for any HEALPix FITS ``BinTableHDU``.  Reads the requested
    column, optionally degrades or upgrades the resolution via
    :func:`healpy.ud_grade`, and normalises the map to zero mean and unit
    standard deviation over valid pixels.  Pixels where the raw value is
    :math:`\leq` ``valid_min`` (default 0) are set to exactly 0 in the output
    and excluded from the normalisation; they are not ``NaN``.

:func:`load_real_templates`
    Convenience wrapper that loads the two templates used in the real-data
    validation tests:

    * **synth_5** — GAIA DR3 faint-star surface density
      (file: ``GAIA_nstar_faint_NSIDE_NNNNN.fits``, column ``nstar_faint``).
      A proxy for stellar contamination: misclassified stars inflate galaxy
      counts in proportion to the local stellar density.
    * **synth_6** — Legacy Survey DR10 galaxy depth in the z band
      (file: ``LS10_GALDEPTH_Z_NSIDE_NNNN.fits``, column ``GALDEPTH_Z``).
      A proxy for depth-dependent selection: survey completeness decreases
      in shallow regions, modulating galaxy counts multiplicatively.

    The combined valid mask is the *intersection* of the two individual
    footprints (GAIA is full-sky; the LS10 mask determines the survey area).
    At NSIDE = 32 this yields approximately 5 954 valid pixels
    (48.4 % of the sky), matching the LS10 South Galactic Cap footprint.

    Files are expected in ``~/data/legacysurvey/dr10/systematics/`` by
    default (see the :ref:`real-template-tests` section of :doc:`../testing`).

.. automodule:: sys_mapping.maps
   :members:
   :show-inheritance:
