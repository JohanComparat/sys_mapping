Running the real-data pipeline
==============================

This page describes the input data and how to run ``sys_mapping`` on the
Legacy Survey DR10 Bright Galaxy Survey (LS10 BGS), following the DESI-BGS
selection described by `Hahn et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023AJ....165..253H/abstract>`_.
The galaxy and random samples are those used in
`Comparat et al. 2025a <https://arxiv.org/abs/2503.19796>`_
and are available on
`Zenodo record 15111974 <https://zenodo.org/records/15111974>`_.
For the mathematical background see :doc:`methods`; for the synthetic-mock
tutorial see :doc:`quickstart`.  Results are documented in :doc:`results_ls10`.

.. warning::

   The script uses **5 synthetic template families by default** when
   ``--template-dir`` is omitted.  Always pass ``--template-dir`` pointing to
   the real GAIA + LS10 FITS maps to get scientifically meaningful results::

      --template-dir ~/data/legacysurvey/dr10/systematics/0128

----

Input data
----------

BGS VLIM samples
~~~~~~~~~~~~~~~~

Nine volume-limited stellar-mass threshold samples spanning
:math:`0.08 < z < 0.35`.  Each is a galaxy + random FITS pair located under
``--catalog-dir`` (default ``~/data/legacysurvey/dr10/sweep/BGS_VLIM_Mstar``).

.. list-table::
   :header-rows: 1
   :widths: 16 12 14 16

   * - log M\ :sub:`*` ≥
     - z\ :sub:`max`
     - N\ :sub:`gal`
     - N\ :sub:`rand`
   * - 9.0
     - 0.08
     - 523 486
     - 2 617 332
   * - 9.5
     - 0.12
     - 1 432 502
     - 7 160 697
   * - 10.0
     - 0.18
     - 2 759 238
     - 13 795 884
   * - 10.25
     - 0.22
     - 3 308 841
     - 16 544 481
   * - 10.5
     - 0.26
     - 3 263 228
     - 16 315 418
   * - 10.75
     - 0.31
     - 2 802 710
     - 14 013 316
   * - 11.0
     - 0.35
     - 1 619 838
     - 8 097 853
   * - 11.25
     - 0.35
     - 541 855
     - 2 708 912
   * - 11.5
     - 0.35
     - 120 882
     - 606 304

Systematic templates
~~~~~~~~~~~~~~~~~~~~

Eleven HEALPix maps from Legacy Survey imaging metadata and Gaia DR3, stored once per
resolution under ``systematics/<NSIDE>/`` (NSIDE 32, 64, 128 and 256).  Pass the
NSIDE 128 directory; the script downgrades the maps to each sample's resolution and
standardises them to zero mean and unit variance over that sample's footprint.

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Template
     - Source
     - Physical quantity
   * - ``LS10_EBV``
     - SFD98
     - Galactic dust extinction :math:`E(B-V)`
   * - ``LS10_GALDEPTH_{G,R,Z}``
     - LS10 imaging
     - 5σ galaxy detection depth per band
   * - ``LS10_NOBS_R``
     - LS10 imaging
     - Number of :math:`r`-band exposures
   * - ``LS10_PSFSIZE_R``
     - LS10 imaging
     - :math:`r`-band PSF FWHM
   * - ``GAIA_nstar_{faint,medium}``
     - Gaia DR3
     - Surface density of faint and medium stars
   * - ``GAIA_phot_{g,bp,rp}_mean_flux``
     - Gaia DR3
     - Mean stellar flux per band

----

Running the pipeline
--------------------

The issued products
~~~~~~~~~~~~~~~~~~~

Each sample is analysed at the finest NSIDE, up to 128, at which its mean footprint
occupancy reaches 25 galaxies per pixel.  Every calibrated statistic draws its null
from the sample's matched GLASS spectrum; the script refuses to build a null without
one unless ``--allow-parametric-null`` is given.

.. code-block:: bash

   python scripts/run_ls10_analysis.py \
       --catalog-dir ~/data/legacysurvey/dr10/sweep/BGS_VLIM_Mstar \
       --sample LS10_VLIM_ANY_10.0_Mstar_12.0_0.05_z_0.18_N_2759238 \
       --template-dir ~/data/legacysurvey/dr10/systematics/0128 \
       --nside 128 --min-per-pixel 25 \
       --null-cl-file <sys_mapping_benchmark>/matched_spectra \
       --isd-n-mocks 30 --significance-n-mocks 400 \
       --sampler auto --no-rst \
       --output-dir data/sys_weights_auto/

``--nside`` is the finest resolution tried; ``--min-per-pixel`` halves it until the
occupancy floor is met, and the chosen NSIDE is written into every output file name.
Without ``--min-per-pixel`` the sample is analysed at ``--nside`` as given.

The matched spectra are produced and validated by
``characterisation/match_glass_to_data.py`` in the ``sys_mapping_benchmark``
repository, one ``*_match.json`` per sample and resolution.  ``load_matched_cl`` uses
the file at the requested resolution when it passed validation, otherwise the validated
file nearest at or above it, otherwise the finest below.

The additive-versus-combined likelihood ratio is calibrated with ``--lrt-null-mocks``
(50 in the published grid); each realisation is a full additive and combined fit, so
this is a cluster job.

**Runtime.** OLS, ElasticNet and ISD take seconds to minutes per sample.  With
``--sampler auto``, ``MCMC-add`` uses the exact Normal-Inverse-Gamma posterior
(milliseconds) and ``MCMC-comb`` uses BlackJAX NUTS, about six hours at NSIDE 128 on
eight cores.  The 400 significance realisations take about ten minutes at NSIDE 128.

Key command-line options
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Flag
     - Default
     - Description
   * - ``--catalog-dir``
     - *(required)*
     - Directory containing ``*_DATA.fits`` / ``*_RAND.fits`` pairs
   * - ``--sample``
     - all
     - One sample, named without the ``_DATA.fits`` suffix
   * - ``--template-dir``
     - *(none: synthetic)*
     - Directory of HEALPix FITS maps; required for real results
   * - ``--nside``
     - 64
     - Resolution, or the finest tried with ``--min-per-pixel``
   * - ``--min-per-pixel``
     - off
     - Occupancy floor that chooses the resolution per sample
   * - ``--null-cl-file``
     - *(none)*
     - Matched spectrum file or directory for every GLASS null
   * - ``--significance-n-mocks``
     - 0
     - Realisations for the calibrated template significance
   * - ``--isd-n-mocks``
     - 30
     - Realisations calibrating the ISD threshold
   * - ``--lrt-null-mocks``
     - 0
     - Realisations for the mock-calibrated likelihood ratio
   * - ``--sampler``
     - ``auto``
     - ``analytic`` for MCMC-add, ``nuts`` for MCMC-comb
   * - ``--only-methods``
     - all
     - Restrict to a subset, e.g. ``OLS ElasticNet``
   * - ``--ct-auto-only``
     - off
     - Drop the cross-template terms of the two-point correction
   * - ``--figures-only``
     - off
     - Redraw figures from saved JSON without refitting
   * - ``--force``
     - off
     - Re-run even if output JSON already exists
   * - ``--output-dir``
     - ``data/sys_weights/``
     - Output directory

----

Output files
------------

::

   data/sys_weights_auto/
   ├── <sample_id>_NSIDE<nside>_WEIGHTS.fits       # per-galaxy weights, all methods
   ├── <sample_id>_NSIDE<nside>_params.json        # amplitudes, significance, LRT, σ̂
   ├── <sample_id>_NSIDE<nside>_wtheta_data.json   # observed and corrected w(θ)
   ├── <sample_id>_NSIDE<nside>_weight_map.png     # Mollweide weight maps
   ├── <sample_id>_NSIDE<nside>_weight_hist.png    # weight distributions
   └── <sample_id>_NSIDE<nside>_wtheta.png         # w(θ) before and after correction

The FITS header records ``WEIGHTVER`` (3), ``TPLBASIS`` (``footprint``), ``WEIGHTCON``
and ``WMAXCLIP``.  The table holds one column per method plus ``WEIGHT_SYS``:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Column
     - Description
   * - ``WEIGHT_OLS``
     - OLS additive correction
   * - ``WEIGHT_ENET``
     - ElasticNet additive correction
   * - ``WEIGHT_ISD1``
     - ISD, degree 1
   * - ``WEIGHT_ISD3``
     - ISD, degree 3
   * - ``WEIGHT_ADD``
     - MCMC additive model
   * - ``WEIGHT_COMB``
     - MCMC combined (additive and multiplicative) model
   * - ``WEIGHT_SYS``
     - Alias for ``WEIGHT_COMB``

Documentation pages
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python scripts/generate_results_ls10_summary.py     # results pages
   python scripts/plot_ls10_occupancy_products.py      # corrected w(θ) figure
   python scripts/analyze_detectability_law.py         # detectability page
   make -C docs html

.. seealso::

   :doc:`results_ls10` for the issued products and
   :doc:`results_ls10_recommendations` for the column to use per sample.
