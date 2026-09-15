Simulation tests: GLASS and Uchuu mocks with LSDR10 systematics
================================================================

The figures and tables on this page come from cells ``simtests64`` (node dahu154, 2026-09-15 15:36–17:28, 1 h 52 min) and ``simtests32`` (node dahu155, 2026-09-15 15:36–16:32, 56 min) of campaign 20260915r on the GRICAD dahu cluster, each on 8 cores of an Intel Xeon Gold 5318Y, running ``sys_mapping`` 1.4.0 (commit 6f43dcc with the working-tree changes of 2026-09-15).

We inject contamination built from five LSDR10 imaging-systematic maps into two mock catalogues, correct it with six decontamination methods, and compare the recovered angular correlation function :math:`w(\theta)` with that of the uncontaminated mock.
We run the test at NSIDE 64 and NSIDE 32 on one realisation (seed 0) per configuration; the metrics are in Tables 3 and 4 and the curves in the figures below.

Mock catalogues
---------------

The GLASS mock (Tessore et al. 2023) is a lognormal field in one tophat shell over :math:`0 \le z \le 0.26`, drawn from the parametric spectrum :math:`C_\ell = 5\times10^{-4}\,(\ell+1)^{-1.5}` with a full-sky target of 500 000 galaxies.
Galaxy positions are Poisson draws from the field, redshifts are drawn from the Uchuu :math:`n(z)` (20 bins over :math:`0.05 \le z \le 0.26`), and the randoms are uniform on the sphere at ten times the target.
The Uchuu catalogue is the lightcone sample ``MOCK_VLIM_ANY_10.65_Mstar_12.0_0.05_z_0.26_N_0923373``, with 923 373 galaxies and 6 249 378 randoms over :math:`0.05 \le z \le 0.26`; its randoms cover one octant, :math:`180° \le \mathrm{RA} \le 270°` and :math:`-90° \le \mathrm{Dec} \le 0°`.

Both catalogues are cut to the pixels where all five templates are valid, see Table 1.

.. list-table:: Table 1. Footprint and catalogue sizes after the cut.
   :widths: 30 35 35
   :header-rows: 1

   * -
     - NSIDE 64
     - NSIDE 32
   * - Footprint pixels
     - 22 641 of 49 152 (46.1 %)
     - 5 954 of 12 288 (48.5 %)
   * - GLASS generated (full sky)
     - 498 874
     - 500 094
   * - GLASS galaxies / randoms
     - 229 382 / 2 303 760
     - 242 124 / 2 422 610
   * - Uchuu galaxies / randoms
     - 330 748 / 2 256 239
     - 358 597 / 2 450 717

.. figure:: _static/results_simulation_tests/nside0064/nz_comparison.png
   :width: 70%
   :align: center
   :alt: Redshift distribution of the Uchuu catalogue and the GLASS mock

   Redshift distribution of the Uchuu catalogue (bars) and of the GLASS mock (step line), normalised to unit area.

Systematic templates
--------------------

The five templates are normalised to zero mean and unit standard deviation over valid pixels and read from ``~/data/legacysurvey/dr10/systematics/{NSIDE:04d}/``.

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Map
     - Column
     - Quantity
   * - ``LS10_EBV``
     - ``EBV``
     - Galactic reddening (Schlegel et al. 1998)
   * - ``LS10_GALDEPTH_Z``
     - ``GALDEPTH_Z``
     - z-band galaxy depth
   * - ``LS10_PSFSIZE_R``
     - ``PSFSIZE_R``
     - r-band PSF size
   * - ``LS10_NOBS_R``
     - ``NOBS_R``
     - number of r-band exposures
   * - ``GAIA_nstar_faint``
     - ``nstar_faint``
     - faint Gaia stellar density

.. figure:: _static/results_simulation_tests/nside0064/templates_overview.png
   :width: 100%
   :align: center
   :alt: Mollweide projections of the five LSDR10 templates at NSIDE 64

   The five templates at NSIDE 64 in Mollweide projection; the colour scale spans ±2 standard deviations.

Contamination
-------------

Contamination enters as a per-galaxy weight :math:`\texttt{WEIGHT\_CONT}(p) = (1 + \delta_{\rm cont}(p))/(1 + \delta_g(p))`, where :math:`\delta_g` is the overdensity of the mock in pixel :math:`p` and :math:`\delta_{\rm cont}` follows the forward model of Berlfein et al. (2024, Eqs. 11–13):

.. math::

   \delta_{\rm cont}(p) = \delta_g(p)\,\Bigl(1 + \textstyle\sum_i b_i\,t_i(p)\Bigr) + \sum_i a_i\,t_i(p).

The grid has nine configurations: three levels, :math:`|a_i| = |b_i| = 0.02` (low), 0.05 (medium) and 0.10 (high), times three scenarios, additive (:math:`b_i = 0`), multiplicative (:math:`a_i = 0`) and combined.
The signs of :math:`a_i` and :math:`b_i` are drawn once from the seed and shared by all levels, and all five templates are contaminated.

Recovery and methods
--------------------

For each configuration we compute :math:`\delta_g^{\rm obs}` from the catalogue weighted by :math:`\texttt{WEIGHT\_CONT}` and run each method; the per-galaxy weight is :math:`\texttt{WEIGHT\_CONT}` times the correction weight, and :math:`w(\theta)` is measured with the TreeCorr Landy–Szalay estimator in 10 logarithmic bins over :math:`0.1° \le \theta \le 10°`.
The truth uses uniform weights.
The six methods are:

* OLS and ElasticNet (:math:`\ell_1 + \ell_2` regularised) regress :math:`\delta_g^{\rm obs}` on the templates, with weight :math:`w = 1/(1 + \hat{a}\cdot t)`.
* ISD-1 and ISD-3 fit one template at a time in 10 equal-occupancy bins with a polynomial of degree 1 or 3, weight by the inverse of the most significant fit and repeat until :math:`\max S = \Delta\chi^2/\Delta\chi^2_{68} < 2`, with at most :math:`4 n_{\rm sys}` steps and three uses per template; the weight is the product of the per-step corrections.
* MCMC-add is the analytic Normal-Inverse-Gamma posterior of :math:`a_i` with :math:`b_i = 0`; MCMC-comb samples :math:`(a_i, b_i)` with BlackJAX NUTS (dense mass matrix, 4 chains, 1000 warm-up steps and 1000 draws per chain).
  Both use the inverse weight :math:`w = (1 + \hat{\delta}_{\rm clean})/(1 + \delta_g^{\rm obs})`.

All weights are clipped to :math:`[1/20, 20]`.

ISD threshold and steps
^^^^^^^^^^^^^^^^^^^^^^^

We calibrate :math:`\Delta\chi^2_{68}` per template, per degree and per mock source as the 68th percentile over 30 uncontaminated GLASS realisations, see Table 2.
The null spectrum is the parametric one at :math:`5\times10^{-4}` for the GLASS source and the matched spectrum of ``LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228`` at the same NSIDE for the Uchuu source (``isd_chi2_68_glass.json`` and ``isd_chi2_68_uchuu.json`` in each ``_static/results_simulation_tests/nside00NN/`` directory).

.. list-table:: Table 2. ISD :math:`\Delta\chi^2_{68}` per template.
   :widths: 10 12 10 13 13 13 13 13
   :header-rows: 1

   * - NSIDE
     - Source
     - ISD
     - EBV
     - GALDEPTH_Z
     - PSFSIZE_R
     - NOBS_R
     - nstar_faint
   * - 64
     - GLASS
     - 1
     - 1.7
     - 2.4
     - 1.2
     - 1.6
     - 1.1
   * - 64
     - GLASS
     - 3
     - 4.8
     - 4.7
     - 3.9
     - 5.5
     - 5.5
   * - 64
     - Uchuu
     - 1
     - 6.9
     - 3.5
     - 7.6
     - 4.2
     - 6.8
   * - 64
     - Uchuu
     - 3
     - 17.0
     - 8.6
     - 12.7
     - 13.2
     - 42.3
   * - 32
     - GLASS
     - 1
     - 1.6
     - 3.5
     - 1.5
     - 2.6
     - 1.6
   * - 32
     - GLASS
     - 3
     - 5.6
     - 6.4
     - 5.7
     - 7.4
     - 5.3
   * - 32
     - Uchuu
     - 1
     - 4.6
     - 1.7
     - 4.8
     - 2.7
     - 8.0
   * - 32
     - Uchuu
     - 3
     - 10.3
     - 7.3
     - 9.2
     - 9.0
     - 20.7

Every ISD run stopped on the threshold, and no step floored the correction :math:`1 + F(t)` at :math:`1/20` in any pixel.
The number of steps per configuration, in the order low, medium, high and within each level additive, multiplicative, combined, was:

* NSIDE 64, GLASS: ISD-1 6, 1, 6, 9, 1, 9, 10, 1, 10; ISD-3 4, 0, 4, 9, 0, 9, 10, 0, 10.
* NSIDE 64, Uchuu: ISD-1 3, 2, 3, 6, 2, 6, 8, 1, 8; ISD-3 4, 1, 4, 4, 1, 4, 7, 4, 7.
* NSIDE 32, GLASS: ISD-1 4, 0, 4, 8, 0, 9, 12, 0, 12; ISD-3 2, 0, 2, 9, 0, 9, 10, 0, 10.
* NSIDE 32, Uchuu: ISD-1 2, 0, 2, 4, 0, 6, 6, 0, 8; ISD-3 1, 0, 1, 3, 0, 3, 3, 0, 5.

Recovery metric
---------------

The mean fractional bias is

.. math::

   \mathcal{B}(w) = \left\langle \frac{|w(\theta) - w_{\rm true}(\theta)|}{\max\bigl(10^{-3},\,\max_\theta |w_{\rm true}(\theta)|\bigr)} \right\rangle_{\!\theta},

averaged over the 10 angular bins, and the improvement factor is :math:`\mathcal{B}(w_{\rm contaminated})/\mathcal{B}(w_{\rm recovered})`.
The :math:`10^{-3}` floor is not reached in any configuration.

Table 3 (NSIDE 64):

.. csv-table::
   :file: _static/results_simulation_tests/nside0064/summary_table.csv
   :header-rows: 1

Table 4 (NSIDE 32):

.. csv-table::
   :file: _static/results_simulation_tests/nside0032/summary_table.csv
   :header-rows: 1

The contaminated :math:`\mathcal{B}` spans 0.11–4.71 on GLASS and 0.006–0.040 on Uchuu at NSIDE 64, and 0.036–3.63 on GLASS and 0.004–0.061 on Uchuu at NSIDE 32.
Table 5 counts, per method, the configurations in which the corrected :math:`\mathcal{B}` exceeds the contaminated one, with the median improvement factor over the nine configurations.

.. list-table:: Table 5. Configurations (of 9) with corrected bias above the contaminated bias, and median improvement factor in brackets.
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Method
     - GLASS, NSIDE 64
     - Uchuu, NSIDE 64
     - GLASS, NSIDE 32
     - Uchuu, NSIDE 32
   * - OLS
     - 0 (2.09)
     - 8 (0.72)
     - 3 (3.25)
     - 7 (0.78)
   * - ISD-1
     - 3 (3.17)
     - 8 (0.66)
     - 0 (2.74)
     - 4 (1.00)
   * - ISD-3
     - 0 (2.99)
     - 8 (0.55)
     - 0 (4.25)
     - 4 (1.00)
   * - ElasticNet
     - 0 (2.44)
     - 5 (0.79)
     - 0 (3.07)
     - 4 (1.00)
   * - MCMC-add
     - 0 (2.91)
     - 9 (0.70)
     - 1 (3.95)
     - 8 (0.71)
   * - MCMC-comb
     - 1 (1.72)
     - 8 (0.33)
     - 5 (0.87)
     - 8 (0.21)
   * - Total (of 54)
     - 4
     - 46
     - 9
     - 35

The corrected :math:`\mathcal{B}` equals the contaminated one in the three multiplicative configurations for ISD-3 on GLASS and for ElasticNet on Uchuu at NSIDE 64, and for ISD-1, ISD-3 and ElasticNet on both sources at NSIDE 32.

w(θ) residuals
--------------

Each grid has contamination level in rows and scenario in columns, and shows :math:`(w - w_{\rm true})/|w_{\rm true}|` per angular bin for the contaminated measurement (grey dashed) and each method.

.. figure:: _static/results_simulation_tests/nside0064/wtheta_recovery_grid_glass.png
   :width: 100%
   :align: center
   :alt: w(θ) residual grid, GLASS mock, NSIDE 64

   GLASS mock, NSIDE 64.

.. figure:: _static/results_simulation_tests/nside0032/wtheta_recovery_grid_glass.png
   :width: 100%
   :align: center
   :alt: w(θ) residual grid, GLASS mock, NSIDE 32

   GLASS mock, NSIDE 32.

.. figure:: _static/results_simulation_tests/nside0064/wtheta_recovery_grid_uchuu.png
   :width: 100%
   :align: center
   :alt: w(θ) residual grid, Uchuu mock, NSIDE 64

   Uchuu mock, NSIDE 64.

.. figure:: _static/results_simulation_tests/nside0032/wtheta_recovery_grid_uchuu.png
   :width: 100%
   :align: center
   :alt: w(θ) residual grid, Uchuu mock, NSIDE 32

   Uchuu mock, NSIDE 32.

.. figure:: _static/results_simulation_tests/nside0064/wtheta_recovery_by_source.png
   :width: 100%
   :align: center
   :alt: GLASS and Uchuu residuals at medium combined contamination, NSIDE 64

   GLASS (left) and Uchuu (right) at medium combined contamination, NSIDE 64.

.. figure:: _static/results_simulation_tests/nside0032/wtheta_recovery_by_source.png
   :width: 100%
   :align: center
   :alt: GLASS and Uchuu residuals at medium combined contamination, NSIDE 32

   The same at NSIDE 32.

Bias by configuration
---------------------

The heatmaps and amplitude scans show :math:`\mathcal{B}` averaged over the GLASS and Uchuu rows of Tables 3 and 4.

.. figure:: _static/results_simulation_tests/nside0064/recovery_bias_heatmap.png
   :width: 100%
   :align: center
   :alt: Heatmap of the mean fractional bias, NSIDE 64

   :math:`\mathcal{B}` per method (rows) and configuration (columns), NSIDE 64; the colour scale saturates at 0.5.

.. figure:: _static/results_simulation_tests/nside0032/recovery_bias_heatmap.png
   :width: 100%
   :align: center
   :alt: Heatmap of the mean fractional bias, NSIDE 32

   The same at NSIDE 32.

.. figure:: _static/results_simulation_tests/nside0064/contamination_amplitude_scan.png
   :width: 100%
   :align: center
   :alt: Bias against contamination amplitude per scenario, NSIDE 64

   :math:`\mathcal{B}` against contamination amplitude for each scenario, NSIDE 64; black dashed is the contaminated measurement.

.. figure:: _static/results_simulation_tests/nside0032/contamination_amplitude_scan.png
   :width: 100%
   :align: center
   :alt: Bias against contamination amplitude per scenario, NSIDE 32

   The same at NSIDE 32.

Amplitude recovery
------------------

We compare the recovered amplitudes with the injected ones over both sources, all templates and all configurations with a non-zero injected value (60 values of :math:`a_i` and 60 of :math:`b_i` per NSIDE), see Table 6.
For ISD-1 and ISD-3, :math:`\hat{a}_i` is the linear projection of the fitted curves and is zero for a template that is never selected; only MCMC-comb estimates :math:`b_i`.

.. list-table:: Table 6. Relative error :math:`|\hat{a}_i - a_i|/|a_i|` (and :math:`|\hat{b}_i - b_i|/|b_i|` for MCMC-comb), mean and median.
   :widths: 28 18 18 18 18
   :header-rows: 1

   * - Method
     - NSIDE 64 mean
     - NSIDE 64 median
     - NSIDE 32 mean
     - NSIDE 32 median
   * - OLS
     - 0.453
     - 0.101
     - 0.466
     - 0.092
   * - ISD-1
     - 0.605
     - 0.397
     - 0.521
     - 0.440
   * - ISD-3
     - 0.538
     - 0.495
     - 0.628
     - 0.650
   * - ElasticNet
     - 0.363
     - 0.133
     - 0.370
     - 0.114
   * - MCMC-add
     - 0.453
     - 0.101
     - 0.466
     - 0.092
   * - MCMC-comb, :math:`a_i`
     - 0.473
     - 0.090
     - 0.470
     - 0.099
   * - MCMC-comb, :math:`b_i`
     - 0.585
     - 0.241
     - 1.294
     - 0.383

The figures show the relative error against the injected value, coloured by level (blue low, orange medium, green high), with the mean and standard deviation in each panel; the rows are also written separately as ``parameter_recovery_a.png`` and ``parameter_recovery_b.png``.

.. figure:: _static/results_simulation_tests/nside0064/parameter_recovery_ab.png
   :width: 100%
   :align: center
   :alt: Relative error of recovered contamination amplitudes, NSIDE 64

   Relative error of the recovered :math:`a_i` (top) and :math:`b_i` (bottom), NSIDE 64.

.. figure:: _static/results_simulation_tests/nside0032/parameter_recovery_ab.png
   :width: 100%
   :align: center
   :alt: Relative error of recovered contamination amplitudes, NSIDE 32

   The same at NSIDE 32.

Run time
--------

Table 7 gives the wall time of each method per configuration on the dahu nodes, as recorded in ``results_summary.json``.

.. list-table:: Table 7. Method wall time per configuration in seconds, median and [min, max] over the nine configurations.
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Method
     - GLASS, NSIDE 64
     - Uchuu, NSIDE 64
     - GLASS, NSIDE 32
     - Uchuu, NSIDE 32
   * - OLS
     - 0.002 [0.002, 0.002]
     - < 0.001 [< 0.001, 0.012]
     - < 0.001 [< 0.001, 0.006]
     - < 0.001 [< 0.001, < 0.001]
   * - ISD-1
     - 0.49 [0.08, 4.1]
     - 0.03 [0.01, 3.1]
     - 0.10 [0.01, 3.1]
     - 0.009 [0.002, 2.5]
   * - ISD-3
     - 0.30 [0.03, 0.55]
     - 0.025 [0.009, 0.040]
     - 0.05 [0.009, 0.13]
     - 0.004 [0.002, 0.009]
   * - ElasticNet
     - 6.9 [2.5, 7.1]
     - 7.4 [4.6, 7.6]
     - 5.6 [2.0, 5.6]
     - 5.9 [4.2, 6.1]
   * - MCMC-add
     - 0.12 [0.09, 1.8]
     - 0.13 [0.11, 5.3]
     - 0.12 [0.08, 1.6]
     - 0.11 [0.07, 4.5]
   * - MCMC-comb
     - 432 [274, 1296]
     - 22 [5.9, 108]
     - 195 [22, 526]
     - 7.9 [2.4, 128]

Reproduce
---------

Each NSIDE writes to its own subdirectory of ``--output-dir``; a missing Uchuu file is an error unless ``--glass-only`` is given.

.. code-block:: bash

   UCHUU=~/data/Uchuu/FullSky/mock_catalogues/MOCK_VLIM_ANY_10.65_Mstar_12.0_0.05_z_0.26_N_0923373/MOCK_VLIM_ANY_10.65_Mstar_12.0_0.05_z_0.26_N_0923373_DATA.fits.gz
   CL=<sys_mapping_benchmark>/matched_spectra/LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228

   for NS in 64 32; do
     python scripts/run_simulation_tests.py --nside ${NS} --n-glass 500000 \
         --methods OLS ISD-1 ISD-3 ElasticNet MCMC-add MCMC-comb \
         --cl-amplitude 5e-4 --uchuu-data "$UCHUU" \
         --uchuu-null-cl-file "${CL}_NSIDE$(printf '%04d' ${NS})_match.json" \
         --syst-dir ~/data/legacysurvey/dr10/systematics/ \
         --output-dir data/simulations
   done

The figures and ``summary_table.csv`` go to ``docs/_static/results_simulation_tests/nside{NSIDE:04d}/`` (both plot commands also accept ``--results-json``, ``--output-dir`` and ``--cl-amplitude``); the two ``isd_chi2_68.json`` files are in ``data/simulations/nside{NSIDE:04d}/{glass,uchuu}/``.

.. code-block:: bash

   python scripts/plot_simulation_tests.py --nside 64
   python scripts/plot_simulation_tests.py --nside 32 --no-templates

References
----------

* Berlfein et al. 2024, MNRAS 531, 4954. `arXiv:2401.12293 <https://arxiv.org/abs/2401.12293>`_
* Tessore et al. 2023, OJAp 6, 11 (GLASS). `arXiv:2302.01942 <https://arxiv.org/abs/2302.01942>`_
* GLASS code: https://github.com/glass-dev/glass
* Schlegel, Finkbeiner & Davis 1998, ApJ 500, 525.
* Weaverdyck & Huterer 2021, MNRAS 503, 5061.
* Rodríguez-Monroy et al. 2025, arXiv:2509.07943.
