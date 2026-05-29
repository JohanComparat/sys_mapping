GLASS simulation tests: galaxy-count scaling at NSIDE = 64
===========================================================

This page reports the validation of the ``sys_mapping`` decontamination pipeline
on GLASS full-sky mocks at two galaxy counts (1 000 000 and 2 000 000) with
NSIDE = 64.  The goal is to quantify how w(θ) recovery quality scales with
the number of galaxies for a fixed map resolution, independently of the Uchuu
*N*-body mock.

The script used here is ``scripts/run_glass_simulation_tests.py``, a
GLASS-only variant of the full ``scripts/run_simulation_tests.py`` that loads
no Uchuu data.  Each (NSIDE, N_glass) combination writes to its own
subdirectory so runs never overwrite each other::

    data/simulations_glass/nside0064_N1000000/
    data/simulations_glass/nside0064_N2000000/

----

How to reproduce
----------------

Run the two configurations sequentially:

.. code-block:: bash

   python scripts/run_glass_simulation_tests.py \
       --nside 64 \
       --n-glass 1000000 \
       --methods OLS ISD-1 ElasticNet \
       --output-dir data/simulations_glass \
       --syst-dir ~/data/legacysurvey/dr10/systematics/ && \
   python scripts/run_glass_simulation_tests.py \
       --nside 64 \
       --n-glass 2000000 \
       --methods OLS ISD-1 ElasticNet \
       --output-dir data/simulations_glass \
       --syst-dir ~/data/legacysurvey/dr10/systematics/

Generate figures and tables after each run:

.. code-block:: bash

   python scripts/plot_simulation_tests.py \
       --nside 64 \
       --results-json data/simulations_glass/nside0064_N1000000/results_summary.json \
       --output-dir docs/_static/results_glass_simulation_tests/N1000000

   python scripts/plot_simulation_tests.py \
       --nside 64 \
       --results-json data/simulations_glass/nside0064_N2000000/results_summary.json \
       --output-dir docs/_static/results_glass_simulation_tests/N2000000

Build the documentation:

.. code-block:: bash

   cd docs && make html

----

Mock catalogs
-------------

GLASS full-sky mock
^^^^^^^^^^^^^^^^^^^

GLASS generates correlated lognormal HEALPix density fields using the
algorithm of Tessore et al. 2023.  A single tophat redshift shell covering
:math:`0.05 \le z \le 0.26` is used, with a power spectrum
:math:`C_\ell \propto (\ell+1)^{-1.5}`.  Galaxy positions are drawn from the
density field using ``glass.positions_from_delta``.

The redshift distribution is a synthetic uniform draw over :math:`[0.05, 0.26]`
(20 bins), the same functional form used for GLASS in the combined-mock test
when no Uchuu reference is available.  Both runs use the same redshift
distribution and random seed so that any difference in recovery quality is
attributable solely to the galaxy count.

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Property
     - N = 1 000 000
     - N = 2 000 000
   * - Number of galaxies
     - 1 000 000
     - 2 000 000
   * - NSIDE
     - 64
     - 64
   * - Pixel scale
     - ≈ 55 arcmin
     - ≈ 55 arcmin
   * - Redshift range
     - :math:`0.05 \le z \le 0.26`
     - :math:`0.05 \le z \le 0.26`
   * - Random seed
     - 0
     - 0

----

Systematic template maps
------------------------

Five LSDR10 imaging-systematic maps are used, all normalised to zero mean
and unit standard deviation over valid pixels:

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Map
     - Column
     - Physical meaning
   * - ``LS10_EBV``
     - ``EBV``
     - Galactic dust reddening (Schlegel et al. 1998)
   * - ``LS10_GALDEPTH_Z``
     - ``GALDEPTH_Z``
     - z-band galaxy depth (selection completeness proxy)
   * - ``LS10_PSFSIZE_R``
     - ``PSFSIZE_R``
     - Seeing PSF size in r band (affects star–galaxy separation)
   * - ``LS10_NOBS_R``
     - ``NOBS_R``
     - Number of r-band exposures (depth uniformity)
   * - ``GAIA_nstar_faint``
     - ``nstar_faint``
     - Faint stellar surface density (stellar contamination proxy)

All maps are loaded at NSIDE = 64 from
``~/data/legacysurvey/dr10/systematics/0064/``.

.. figure:: _static/results_glass_simulation_tests/N1000000/templates_overview.png
   :width: 100%
   :align: center
   :alt: Mollweide projections of the 5 LSDR10 systematic template maps at NSIDE=64

   **LSDR10 systematic template maps (NSIDE = 64)** shown in Mollweide
   projection.  Red–blue colour scale: ±2 standard deviations from the mean.
   Identical for both runs (maps are independent of galaxy count).

----

Contamination injection
-----------------------

Contamination is injected as per-galaxy weights.  For a galaxy in pixel
:math:`p`:

.. math::

   \texttt{WEIGHT\_CONT}(p) = \frac{1 + \delta_{\rm cont}(p)}{1 + \delta_g(p)}

where :math:`\delta_{\rm cont}(p)` follows the forward model:

.. math::

   \delta_{\rm cont}(p)
   = \delta_g(p)\,\Bigl(1 + \textstyle\sum_i b_i\,t_i(p)\Bigr)
   + \sum_i a_i\,t_i(p)

Nine contamination configurations are tested (3 amplitude levels ×
3 scenarios):

.. list-table::
   :widths: 15 35 50
   :header-rows: 1

   * - Level
     - Amplitudes
     - Interpretation
   * - Low
     - :math:`|a_i| = |b_i| = 0.02`
     - Sub-percent modulation
   * - Medium
     - :math:`|a_i| = |b_i| = 0.05`
     - Few-percent modulation
   * - High
     - :math:`|a_i| = |b_i| = 0.10`
     - Ten-percent modulation

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Scenario
     - Forward model applied
   * - Additive
     - :math:`\delta_{\rm cont} = \delta_g + \sum_i a_i\,t_i`
   * - Multiplicative
     - :math:`\delta_{\rm cont} = \delta_g\,(1 + \sum_i b_i\,t_i)`
   * - Combined
     - :math:`\delta_{\rm cont} = \delta_g\,(1 + \sum_i b_i\,t_i) + \sum_i a_i\,t_i`

----

w(θ) recovery results — N = 1 000 000
--------------------------------------

.. figure:: _static/results_glass_simulation_tests/N1000000/wtheta_recovery_grid_glass.png
   :width: 100%
   :align: center
   :alt: w(θ) recovery grid — GLASS mock, N=1e6, NSIDE=64

   **w(θ) recovery on the GLASS mock (N = 1 000 000, NSIDE = 64).**  Rows:
   contamination amplitude level (low / medium / high).  Columns:
   contamination scenario (additive / multiplicative / combined).  In each
   panel: black solid = truth; grey dashed = contaminated; coloured lines =
   recovered by OLS (blue), ISD-1 (orange), ElasticNet (green).

Recovery metric summary — N = 1 000 000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :file: _static/results_glass_simulation_tests/N1000000/summary_table.csv
   :header-rows: 1

Heatmap of recovery bias — N = 1 000 000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: _static/results_glass_simulation_tests/N1000000/recovery_bias_heatmap.png
   :width: 100%
   :align: center
   :alt: Heatmap of mean fractional w(θ) bias — N=1e6, NSIDE=64

   **Mean fractional bias** :math:`\mathcal{B}` **per method (row) and
   configuration (column), N = 1 000 000, NSIDE = 64.**  Green = low bias;
   red = high bias.

----

w(θ) recovery results — N = 2 000 000
--------------------------------------

.. figure:: _static/results_glass_simulation_tests/N2000000/wtheta_recovery_grid_glass.png
   :width: 100%
   :align: center
   :alt: w(θ) recovery grid — GLASS mock, N=2e6, NSIDE=64

   **w(θ) recovery on the GLASS mock (N = 2 000 000, NSIDE = 64).**
   Layout as above.  The doubled galaxy count halves the shot-noise variance
   per pixel, which is expected to tighten the recovered w(θ) around the truth
   for additive contamination and to improve multiplicative bias estimation
   where the mode-coupling signal-to-noise is shot-noise limited.

Recovery metric summary — N = 2 000 000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :file: _static/results_glass_simulation_tests/N2000000/summary_table.csv
   :header-rows: 1

Heatmap of recovery bias — N = 2 000 000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: _static/results_glass_simulation_tests/N2000000/recovery_bias_heatmap.png
   :width: 100%
   :align: center
   :alt: Heatmap of mean fractional w(θ) bias — N=2e6, NSIDE=64

   **Mean fractional bias** :math:`\mathcal{B}` **per method (row) and
   configuration (column), N = 2 000 000, NSIDE = 64.**  Comparing with the
   N = 1 000 000 heatmap above directly shows the impact of doubling the galaxy
   count on decontamination performance.

----

Bias vs contamination amplitude — comparison
--------------------------------------------

.. rubric:: N = 1 000 000

.. figure:: _static/results_glass_simulation_tests/N1000000/contamination_amplitude_scan.png
   :width: 100%
   :align: center
   :alt: Bias vs amplitude per scenario — N=1e6, NSIDE=64

   **Recovery bias vs contamination amplitude (N = 1 000 000, NSIDE = 64)**,
   for each scenario.  Dashed black = contaminated baseline.  Coloured lines:
   OLS (blue), ISD-1 (orange), ElasticNet (green).

.. rubric:: N = 2 000 000

.. figure:: _static/results_glass_simulation_tests/N2000000/contamination_amplitude_scan.png
   :width: 100%
   :align: center
   :alt: Bias vs amplitude per scenario — N=2e6, NSIDE=64

   **Same scan at N = 2 000 000.**  Lower shot noise is expected to reduce
   residual bias at medium and high amplitudes, most visibly for the additive
   scenario where the OLS estimator variance is directly shot-noise limited.

----

Discussion
----------

**Shot noise and recovery quality.**  The GLASS mock is a Poisson realisation
of a lognormal density field; its true :math:`w(\theta)` is near zero at
NSIDE = 64 (shot-noise dominated).  Doubling the galaxy count from 1 × 10\ :sup:`6`
to 2 × 10\ :sup:`6` halves the shot-noise variance per pixel.  For the
*additive* scenario this directly tightens the regression estimate
:math:`\hat{a}_i` and reduces the residual :math:`\mathcal{B}`.

**Multiplicative scenario.**  OLS, ISD-1, and ElasticNet fit an additive-only
model; they provide little correction for purely multiplicative contamination
regardless of galaxy count.  Any improvement with galaxy count seen in this
scenario reflects a better estimation of :math:`\delta_g` rather than a change
in the fundamental limitation of the additive model.

**Combined scenario.**  At high amplitude the additive-only methods can
overcorrect the additive component while leaving the multiplicative term
untouched, producing a net residual larger than the contaminated baseline.
Doubling the galaxy count may not help here; the fundamental issue is model
mismatch, not noise.

**Practical implications for GLASS-matched BGS samples.**  The BGS
bright-sample surface density in :math:`0.05 \le z \le 0.26` is
approximately :math:`500\,\text{deg}^{-2}`, corresponding to
:math:`\sim 2 \times 10^7` galaxies over the full sky.  The runs here at
1 × 10\ :sup:`6` and 2 × 10\ :sup:`6` are therefore at
:math:`\sim 5\%` and :math:`\sim 10\%` of the expected full-sky count.
Recovery performance at the full BGS density is expected to be substantially
better than shown here, particularly for the additive scenario.

----

References
----------

* Tessore et al. 2023, OJAp 6, 11 (GLASS).  `arXiv:2302.01942 <https://arxiv.org/abs/2302.01942>`_
* GLASS code: https://github.com/glass-dev/glass
* Berlfein et al. 2024, MNRAS 531, 4954.  `arXiv:2401.12293 <https://arxiv.org/abs/2401.12293>`_
* Weaverdyck & Huterer 2021, MNRAS 503, 5061.
