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

      --template-dir ~/data/legacysurvey/dr10/systematics

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

44 HEALPix maps at NSIDE ∈ {32, 64, 128, 256} from Legacy Survey imaging
metadata and GAIA DR3 stellar catalogues.  Templates are standardised to
zero mean and unit variance over the survey footprint before fitting.

.. list-table::
   :header-rows: 1
   :widths: 22 16 62

   * - Template family
     - Source
     - Physical quantity
   * - ``LS10:EBV``
     - SFD98
     - Galactic dust extinction :math:`E(B-V)`
   * - ``LS10:GALDEPTH_{G,R,Z}``
     - LS10 imaging
     - 5σ galaxy detection depth (per band)
   * - ``LS10:PSFSIZE_{G,R,Z}``
     - LS10 imaging
     - PSF FWHM (per band)
   * - ``LS10:NOBS_{G,R,Z}``
     - LS10 imaging
     - Number of exposures (per band)
   * - ``GAIA:nstar_faint/medium``
     - GAIA DR3
     - Surface density of faint / medium stars
   * - ``GAIA:phot_{g,bp,rp}_mean_flux``
     - GAIA DR3
     - Mean stellar flux (per photometric band)

Each family appears at all four NSIDE values, giving 44 maps in total.
The dominant systematic in LS10 BGS is stellar density
(``GAIA:nstar_faint``), which correlates with galaxy counts at the
5–70 % per-pixel level depending on mass threshold.

Representative figures for all nine samples are in :doc:`results_ls10`.

----

Running the pipeline
--------------------

Full run with real templates (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda activate sys_map
   python scripts/run_ls10_analysis.py \
       --catalog-dir ~/data/legacysurvey/dr10/sweep/BGS_VLIM_Mstar \
       --template-dir ~/data/legacysurvey/dr10/systematics \
       --nside 64 \
       --n-walkers 210 --n-steps 1500 --n-burn 300 \
       --output-dir data/sys_weights/ \
       --force \
       2>&1 | tee logs/ls10_run.log

**Runtime**: OLS/ElasticNet/ISD run in seconds–minutes per sample; MCMC-add and
MCMC-comb take 30–90 minutes per sample on CPU (9 samples × 2 MCMC methods).
Use ``--only-methods MCMC-comb`` to run only the combined model.

Regenerate figures without re-running MCMC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python scripts/run_ls10_analysis.py \
       --catalog-dir ~/data/legacysurvey/dr10/sweep/BGS_VLIM_Mstar \
       --template-dir ~/data/legacysurvey/dr10/systematics \
       --figures-only \
       --output-dir data/sys_weights/

This reloads the saved ``*_params.json`` files, redraws all figures, and copies
them to ``docs/_static/results_ls10/``.

Key command-line options
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Flag
     - Default
     - Description
   * - ``--catalog-dir``
     - *(required)*
     - Directory containing ``*_DATA.fits`` / ``*_RAND.fits`` pairs
   * - ``--template-dir``
     - *(none — synthetic)*
     - Directory of real HEALPix FITS maps; **required for real results**
   * - ``--nside``
     - 64
     - HEALPix resolution
   * - ``--n-walkers``
     - 210
     - emcee walkers per MCMC run
   * - ``--n-steps``
     - 1500
     - MCMC steps after burn-in
   * - ``--n-burn``
     - 300
     - MCMC burn-in steps
   * - ``--only-methods``
     - all
     - Restrict to a subset, e.g. ``OLS ElasticNet``
   * - ``--force``
     - off
     - Re-run even if output JSON already exists
   * - ``--figures-only``
     - off
     - Regenerate figures from saved JSON without MCMC
   * - ``--output-dir``
     - ``data/sys_weights/``
     - Root output directory

----

Output files
------------

All results are written to ``--output-dir`` (default ``data/sys_weights/``).
Figures are copied to ``docs/_static/results_ls10/``.

::

   data/sys_weights/
   ├── <sample_id>_NSIDE0064_WEIGHTS.fits          # per-galaxy weights, all methods
   ├── <sample_id>_NSIDE0064_params.json           # MCMC amplitudes, LRT, σ_hat
   ├── <sample_id>_NSIDE0064_partial_OLS.json      # partial results per method
   ├── <sample_id>_NSIDE0064_weight_map.png        # 2×3 Mollweide weight maps
   ├── <sample_id>_NSIDE0064_weight_hist.png       # log-scale weight distributions
   ├── <sample_id>_NSIDE0064_wtheta.png            # w(θ) before/after correction
   └── summary_NSIDE0064.yaml                      # cross-sample YAML summary

The FITS weight table contains one column per method plus ``WEIGHT_SYS``:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Column
     - Description
   * - ``WEIGHT_OLS``
     - OLS additive correction weights
   * - ``WEIGHT_ENET``
     - ElasticNet additive correction weights
   * - ``WEIGHT_ISD1``
     - ISD (order 1) additive correction weights
   * - ``WEIGHT_ISD3``
     - ISD (order 3) additive correction weights
   * - ``WEIGHT_ADD``
     - MCMC-additive correction weights
   * - ``WEIGHT_COMB``
     - MCMC-combined (additive + multiplicative) weights
   * - ``WEIGHT_SYS``
     - Alias for ``WEIGHT_COMB`` — **recommended default**

After the run, rebuild the HTML documentation::

   make -C docs html

.. note::

   **:math:`w(\theta)` figure** — ``wtheta_corrected_nside64.png`` is produced by
   ``scripts/plot_ls10_wtheta_corrected.py`` using the analytical correction
   (Eq. 15–16) from ``data/sys_weights/*_wtheta_data.json``.  It shows all 6
   decontamination methods across all 9 samples.  To regenerate after a new run::

      python scripts/plot_ls10_wtheta_corrected.py

.. seealso::

   :doc:`results_ls10` — per-sample results tables, LRT statistics, and
   fractional systematic uncertainty on :math:`w(\theta)`.
