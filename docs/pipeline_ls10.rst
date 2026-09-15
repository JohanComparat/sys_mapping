Running the real-data pipeline
==============================

This page describes the inputs and the command that produce the systematic weights of
the Legacy Survey DR10 Bright Galaxy Survey (LS10 BGS) samples, selected following
`Hahn et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023AJ....165..253H/abstract>`_.
The galaxy and random samples are those of
`Comparat et al. 2025a <https://arxiv.org/abs/2503.19796>`_, available on
`Zenodo record 15111974 <https://zenodo.org/records/15111974>`_.  The phases of
``scripts/run_ls10_analysis.py`` and its JSON outputs are described on
:doc:`overview`, the mathematics on :doc:`methods`, and the results on
:doc:`results_ls10`.

.. warning::

   Without ``--template-dir`` the script fits five synthetic template families.  Pass
   the directory of the Gaia and LS10 maps for real results::

      --template-dir ~/data/legacysurvey/dr10/systematics/0128

----

Input data
----------

BGS VLIM samples
~~~~~~~~~~~~~~~~

Nine volume-limited stellar-mass threshold samples with :math:`0.05 < z < z_{\max}`.
Each is a ``<sample>_DATA.fits`` and ``<sample>_RAND.fits`` pair under
``--catalog-dir``.  The script reads ``RA``, ``DEC`` and, when present, the galaxy
weight ``WEIGHT_COMP``.

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

Eleven HEALPix maps, six from Legacy Survey imaging metadata and five from Gaia DR3,
stored per resolution under ``systematics/0032``, ``0064``, ``0128`` and ``0256``.  The
script reads every ``*.fits`` file in ``--template-dir``, downgrades each map to the
sample's resolution, normalises it over its valid pixels, and standardises the basis
again over the sample's footprint.

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

Each sample runs at the finest NSIDE, from 128 down, at which its mean footprint
occupancy reaches 25 galaxies per pixel; the nine samples land at NSIDE 16 to 128.
Every GLASS null is drawn from the sample's matched spectrum.

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

``--nside`` is the finest resolution tried, and ``--min-per-pixel`` halves it until the
occupancy floor is met (NSIDE 8 at the coarsest).  Without ``--min-per-pixel`` the
sample runs at ``--nside``.  The chosen NSIDE enters every output file name.

The matched spectra are produced and validated by
``characterisation/match_glass_to_data.py`` in the ``sys_mapping_benchmark``
repository, one ``<sample>_NSIDE<nside>_match.json`` per sample and resolution.
:func:`~sys_mapping.glass_mocks.load_matched_cl` takes the file at the run's resolution
if it passed validation, otherwise the nearest validated file above it, otherwise the
finest validated file below.  When the run builds a null (ISD threshold, pre-selection,
calibrated significance or LRT null) and no spectrum is found, the script stops unless
``--allow-parametric-null`` is given.

The additive-versus-combined likelihood ratio is calibrated with ``--lrt-null-mocks``
(50 in the published LRT grid at NSIDE 32 and 64).  With the default
``--lrt-null-method maxima`` the realisations are maximised together by
:func:`~sys_mapping.model_selection.lrt_from_maxima`; ``nuts`` fits each with NUTS and
refines to the maximum.

Command-line options
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Flag
     - Default
     - Description
   * - ``--catalog-dir``
     - *(required)*
     - Directory of ``*_DATA.fits`` / ``*_RAND.fits`` pairs
   * - ``--sample``
     - all
     - One sample, named without ``_DATA.fits``
   * - ``--template-dir``
     - synthetic
     - Directory of HEALPix FITS maps
   * - ``--nside``
     - 64
     - Resolution, or the finest tried with ``--min-per-pixel``
   * - ``--min-per-pixel``
     - off
     - Mean galaxies per pixel that chooses the resolution per sample
   * - ``--null-cl-file``
     - none
     - Matched spectrum file or directory for every GLASS null
       (alias ``--lrt-null-cl-file``)
   * - ``--allow-parametric-null``
     - off
     - Build nulls from a power law when no matched spectrum is found
   * - ``--lrt-null-cl-amplitude``
     - library default
     - Amplitude of that power law
   * - ``--null-draw``
     - ``pixel``
     - Null realisations drawn per footprint pixel, or ``catalogue``
   * - ``--isd-n-mocks``
     - 30
     - Realisations calibrating the ISD threshold; 0 leaves it uncalibrated
   * - ``--significance-n-mocks``
     - 0
     - Realisations for the calibrated template significance; 370 resolve 0.0027
   * - ``--significance-seed``
     - 70000
     - Base seed of those realisations
   * - ``--lrt-null-mocks``
     - 0
     - Realisations for the mock-calibrated likelihood ratio
   * - ``--lrt-null-method``
     - ``maxima``
     - ``maxima`` (batched maximisation) or ``nuts``
   * - ``--lrt-null-seed``
     - 90000
     - Base seed of the LRT realisations
   * - ``--resume-null``
     - off
     - Add realisations to an existing LRT null in ``params.json`` without refitting
   * - ``--sampler``
     - ``auto``
     - ``auto``, ``analytic``, ``nuts`` or ``emcee``
   * - ``--n-chains``
     - 4 on CPU, 8 on GPU
     - NUTS chains
   * - ``--nuts-warmup``, ``--nuts-samples``
     - 1000, 1000
     - NUTS adaptation steps and draws per chain
   * - ``--n-walkers``, ``--n-steps``, ``--n-burn``
     - 210, 1500, 300
     - emcee settings
   * - ``--skewed``
     - off
     - Skew-normal likelihood for the combined model
   * - ``--preselect``
     - off
     - Template pre-selection, with ``--preselect-method`` (``isd``),
       ``--preselect-n-top``, ``--preselect-p-threshold`` (0.05),
       ``--preselect-n-mocks`` (100), ``--preselect-n-jobs`` (1)
   * - ``--only-methods``
     - all
     - Subset of methods, e.g. ``OLS ElasticNet``
   * - ``--ct-max-galaxies``
     - 1 000 000
     - Galaxies drawn for the template correlation matrix
   * - ``--ct-auto-only``
     - off
     - Drop the cross-template terms of the two-point correction
   * - ``--ct-from-pixels``
     - off
     - Measure template correlations on pixel centres, as ``WEIGHTVER`` ≤ 2 products did
   * - ``--no-footprint-standardise``
     - off
     - Keep the load-time template normalisation; writes ``WEIGHTVER`` 2
   * - ``--figures-only``
     - off
     - Redraw figures and ``wtheta_data.json`` from saved JSON without refitting
   * - ``--force``
     - off
     - Re-run a sample whose output exists
   * - ``--no-rst``
     - off
     - Leave ``docs/results_ls10.rst`` and ``docs/_static`` untouched
   * - ``--output-dir``
     - ``data/sys_weights/``
     - Output directory

----

Output files
------------

::

   data/sys_weights_auto/
   ├── <sample>_NSIDE<nside>_WEIGHTS.fits       # per-galaxy weights, all methods
   ├── <sample>_NSIDE<nside>_params.json        # amplitudes, significance, LRT, σ̂
   ├── <sample>_NSIDE<nside>_wtheta_data.json   # observed and corrected w(θ)
   ├── <sample>_NSIDE<nside>_weight_map.png     # Mollweide weight maps
   ├── <sample>_NSIDE<nside>_weight_hist.png    # weight distributions
   ├── <sample>_NSIDE<nside>_wtheta.png         # w(θ) before and after correction
   └── summary_NSIDE<nside>.yaml                # samples run at that resolution

The FITS header records ``WEIGHTVER`` (3), ``TPLBASIS`` (``footprint``), ``WEIGHTCON``
(``library``), ``WMAXCLIP`` (20), ``SAMPLE``, ``NSIDE``, ``N_SYS`` and the likelihood
ratio (``LRT_LAM``, ``LRT_P``, ``LRT_REJ``, ``LRT_CAL``).  The table holds one column
per method and ``WEIGHT_SYS``:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Column
     - Description
   * - ``WEIGHT_OLS``
     - OLS
   * - ``WEIGHT_ENET``
     - ElasticNet
   * - ``WEIGHT_ISD1``
     - ISD, degree 1
   * - ``WEIGHT_ISD3``
     - ISD, degree 3
   * - ``WEIGHT_ADD``
     - MCMC additive model
   * - ``WEIGHT_COMB``
     - MCMC combined model
   * - ``WEIGHT_SYS``
     - Copy of ``WEIGHT_COMB``

Weights lie in :math:`[1/20, 20]`, and galaxies outside the fitted footprint have
weight 1.

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
