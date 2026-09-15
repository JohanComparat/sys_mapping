Model test matrix
=================

Run: sys_mapping 1.4.0 (commit ``6f43dcc`` plus the uncommitted working-tree changes of
2026-09-15), GRICAD dahu campaign 20260915r, cell ``systests`` (node dahu148, Intel Xeon Gold
5318Y, 8 cores), 2026-09-15; the run took 32 min.

We run the six methods of ``sm.run_decontamination`` on 32 contaminated mocks and report, for
each, the pixel standard deviation of

.. math::

   \mathcal{R}(p) = \frac{1 + \delta_g^{\rm corr}(p)}{1 + \delta_g^{\rm true}(p)}

over the footprint, after discarding pixels outside the 0.5th–99.5th percentiles of
:math:`\mathcal{R}`.
A perfect correction gives :math:`\mathcal{R} = 1`; Poisson noise sets the floor of
:math:`\sigma[\mathcal{R}]`.

Mocks
-----

Each mock is a lognormal field with :math:`\sigma_G = 0.5` and
:math:`C_\ell \propto (\ell+1)^{-2}`, contaminated as
:math:`\delta_{\rm obs} = \delta_{\rm true}(1 + \mathbf{b}\cdot\mathbf{t}) + \mathbf{a}\cdot\mathbf{t}`
and Poisson-sampled with 50 galaxies per pixel on the :math:`|\text{latitude}| > 20°` footprint
(8 064 pixels at NSIDE 32).
The random map holds eight times the mean galaxy count in every footprint pixel.
Every active template carries an amplitude of 0.10.

The seven templates are unit-variance Gaussian maps drawn by ``sm.generate_systematic_map``:

=======  ======  ======================================
Label    Family  :math:`C_\ell`
=======  ======  ======================================
synth_0  0       :math:`e^{-\ell/500}`
synth_1  1       :math:`e^{-(\ell/250)^2}`
synth_2  2       :math:`(\ell+1)^{-2}`
synth_3  3       :math:`(\ell+1)^{-1}`
synth_4  4       constant
synth_5  0       :math:`e^{-\ell/500}`, second seed
synth_6  2       :math:`(\ell+1)^{-2}`, second seed
=======  ======  ======================================

On the footprint, synth_0, synth_1 and synth_4 have pairwise correlation coefficients above
0.9997, synth_3 correlates with each of them at 0.95, synth_2 with synth_0, 1, 3 and 4 at
0.67–0.86, and synth_5 with synth_6 at 0.70; synth_5 and synth_6 correlate with synth_0..4 at
below 0.04.

The 32 configurations are:

* Tier 1 additive (:math:`\mathbf{b} = 0`): each template alone (7) and the cumulative sets
  synth_0..k−1 for k = 2..7 (6);
* Tier 1 multiplicative (:math:`\mathbf{a} = 0`): the same 13 template sets;
* Tier 2 mixed: all seven templates with :math:`a_i = 0.10`, of which the first
  :math:`n_{\rm mult}` also carry :math:`b_i = 0.10`, for :math:`n_{\rm mult} = 1..6`.

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Method
     - Fit and weight
   * - OLS
     - Least-squares fit of :math:`\delta_{\rm obs}` on the templates;
       :math:`w = 1/(1 + \hat{\mathbf{a}}\cdot\mathbf{t})`.
   * - ElasticNet
     - L1+L2-penalised fit (``l1_ratio`` 0.5, penalty from 5-fold cross-validation); same weight
       as OLS.
   * - ISD-1, ISD-3
     - Iterative marginal fits, one template at a time, of a degree-1 or degree-3 polynomial in
       that template's value over 10 equal-occupancy bins.
       Each step weights by the inverse of the most significant fit and stops when
       :math:`\max S = \Delta\chi^2/\Delta\chi^2_{68} < 2`, with at most :math:`4\,n_{\rm sys}`
       steps and 3 uses per template; the weight is the cumulative product of the per-step
       corrections.
   * - MCMC-add
     - Exact Normal-Inverse-Gamma posterior of the additive model (:math:`\mathbf{b} = 0`);
       :math:`w = (1 + \delta_{\rm clean})/(1 + \delta_{\rm obs})`.
   * - MCMC-comb
     - BlackJAX NUTS on the combined model (:math:`\mathbf{a}`, :math:`\mathbf{b}` free), dense
       mass matrix, 4 chains, 1000 warm-up steps and 1000 draws per chain.

All weights are clipped to :math:`[1/20, 20]`.
The corrected field is :math:`w(1 + \delta_{\rm obs}) - 1`, except for MCMC-comb, for which it is
:math:`(\delta_{\rm obs} - \hat{\mathbf{a}}\cdot\mathbf{t})/(1 + \hat{\mathbf{b}}\cdot\mathbf{t})`.

ISD calibration
^^^^^^^^^^^^^^^

We set :math:`\Delta\chi^2_{68}` to the 68th percentile, per template and per order, of the
first-step marginal :math:`\Delta\chi^2` over 50 uncontaminated mocks from the same generator
(``isd_chi2_68.json``); the table lists the values.

=========  =============  =============
Template   ISD-1          ISD-3
=========  =============  =============
synth_0    1.53           4.42
synth_1    1.53           4.40
synth_2    56.83          65.34
synth_3    9.50           11.79
synth_4    1.38           4.27
synth_5    2.15           5.03
synth_6    59.28          78.16
=========  =============  =============

All 64 ISD runs stopped on the threshold.
ISD-1 and ISD-3 took 0 or 1 steps for single templates (0 for synth_6 additive and for every
single multiplicative template), 2–9 (ISD-1) and 1–8 (ISD-3) for the cumulative additive sets,
0–1 for the cumulative multiplicative sets, and 6–12 (ISD-1) and 6–8 (ISD-3) in Tier 2, where
the cap is 28.

Correction quality
------------------

The three tables list :math:`\sigma[\mathcal{R}]` from ``systematic_test_summary.csv``.

.. table:: Tier 1 additive, :math:`\sigma[\mathcal{R}]`.

   ==========  =====  ==========  =====  =====  ========  =========
   Templates   OLS    ElasticNet  ISD-1  ISD-3  MCMC-add  MCMC-comb
   ==========  =====  ==========  =====  =====  ========  =========
   synth_0     0.156  0.156       0.157  0.157  0.145     0.145
   synth_1     0.158  0.158       0.158  0.158  0.144     0.144
   synth_2     0.155  0.155       0.156  0.158  0.144     0.144
   synth_3     0.156  0.156       0.156  0.156  0.145     0.144
   synth_4     0.157  0.157       0.158  0.158  0.145     0.145
   synth_5     0.155  0.155       0.157  0.156  0.145     0.145
   synth_6     0.160  0.161       0.189  0.189  0.151     0.151
   synth_0..1  0.189  0.189       0.189  0.190  0.146     0.144
   synth_0..2  0.244  0.244       0.259  0.249  0.187     0.147
   synth_0..3  0.306  0.304       0.315  0.310  0.229     0.157
   synth_0..4  0.371  0.368       0.373  0.371  0.273     0.176
   synth_0..5  0.378  0.373       0.380  0.377  0.279     0.180
   synth_0..6  0.386  0.382       0.394  0.391  0.288     0.191
   ==========  =====  ==========  =====  =====  ========  =========

.. table:: Tier 1 multiplicative, :math:`\sigma[\mathcal{R}]`.

   ==========  =====  ==========  =====  =====  ========  =========
   Templates   OLS    ElasticNet  ISD-1  ISD-3  MCMC-add  MCMC-comb
   ==========  =====  ==========  =====  =====  ========  =========
   synth_0     0.153  0.153       0.153  0.153  0.154     0.145
   synth_1     0.154  0.154       0.154  0.154  0.155     0.146
   synth_2     0.157  0.156       0.156  0.156  0.158     0.147
   synth_3     0.155  0.154       0.154  0.154  0.156     0.146
   synth_4     0.154  0.154       0.154  0.154  0.154     0.146
   synth_5     0.152  0.152       0.153  0.153  0.150     0.145
   synth_6     0.157  0.157       0.155  0.155  0.150     0.148
   synth_0..1  0.181  0.180       0.180  0.180  0.182     0.154
   synth_0..2  0.209  0.208       0.208  0.208  0.212     0.164
   synth_0..3  0.258  0.256       0.258  0.256  0.261     0.188
   synth_0..4  0.298  0.291       0.294  0.295  0.302     0.227
   synth_0..5  0.299  0.293       0.296  0.297  0.302     0.229
   synth_0..6  0.319  0.308       0.310  0.311  0.317     0.244
   ==========  =====  ==========  =====  =====  ========  =========

.. table:: Tier 2 mixed, seven templates, :math:`\sigma[\mathcal{R}]`.

   ======  =====  ==========  =====  =====  ========  =========
   n_mult  OLS    ElasticNet  ISD-1  ISD-3  MCMC-add  MCMC-comb
   ======  =====  ==========  =====  =====  ========  =========
   1       0.358  0.354       0.363  0.374  0.281     0.187
   2       0.326  0.323       0.346  0.354  0.283     0.181
   3       0.308  0.304       0.331  0.340  0.289     0.178
   4       0.289  0.285       0.321  0.332  0.301     0.185
   5       0.282  0.278       0.324  0.341  0.323     0.202
   6       0.262  0.258       0.306  0.327  0.322     0.200
   ======  =====  ==========  =====  =====  ========  =========

For single additive templates, MCMC-add and MCMC-comb give 0.144–0.151 and the other four
methods 0.155–0.161, except ISD-1 and ISD-3 on synth_6 (0.189).
For single multiplicative templates, MCMC-comb gives 0.145–0.148 and the other five 0.150–0.158.
For synth_0..6, MCMC-comb gives 0.191 (additive) and 0.244 (multiplicative), MCMC-add 0.288 and
0.317, and OLS, ElasticNet, ISD-1 and ISD-3 0.382–0.394 and 0.308–0.319.
In Tier 2, MCMC-comb gives 0.178–0.202, MCMC-add 0.281–0.323, OLS and ElasticNet 0.258–0.358,
and ISD-1 and ISD-3 0.306–0.374.

.. figure:: _static/results_systematic_tests/summary_additive.png
   :width: 85%
   :align: center
   :alt: sigma of the ratio against the number of additive templates, six methods

   :math:`\sigma[\mathcal{R}]` against the number of templates for Tier 1 additive contamination.
   The points at one template are the seven single-template configurations; the others are the
   cumulative sets synth_0..k−1.

.. figure:: _static/results_systematic_tests/summary_multiplicative.png
   :width: 85%
   :align: center
   :alt: sigma of the ratio against the number of multiplicative templates, six methods

   As above for Tier 1 multiplicative contamination.

.. figure:: _static/results_systematic_tests/summary_combined_3mult.png
   :width: 85%
   :align: center
   :alt: sigma of the ratio for the Tier 2 configuration with three multiplicative templates

   :math:`\sigma[\mathcal{R}]` per method for the Tier 2 configuration with
   :math:`n_{\rm mult} = 3`.

.. figure:: _static/results_systematic_tests/histograms/T1_add_s0.png
   :width: 80%
   :align: center
   :alt: Histogram of the ratio per method, synth_0 additive

   Distribution of :math:`\mathcal{R}` per method for synth_0 alone with additive contamination:
   60 bins over [0.5, 1.5], dashed line at :math:`\mathcal{R} = 1`.

.. figure:: _static/results_systematic_tests/histograms/T1_mul_m7.png
   :width: 80%
   :align: center
   :alt: Histogram of the ratio per method, seven multiplicative templates

   As above for the seven templates with multiplicative contamination.

.. figure:: _static/results_systematic_tests/histograms/T2_comb_3m.png
   :width: 80%
   :align: center
   :alt: Histogram of the ratio per method, Tier 2 with three multiplicative templates

   As above for the Tier 2 configuration with :math:`n_{\rm mult} = 3`.

Amplitude bias
--------------

The column ``rms_a_bias`` is
:math:`\sqrt{\langle(\hat{a}_i - a_i^{\rm true})^2\rangle}` over the active templates; for ISD,
:math:`\hat{a}_i` is the sum, over the steps that selected template :math:`i`, of the
least-squares projection of the fitted correction onto :math:`t_i`.
For single templates it is at most 0.042 for every method, except ISD-1 and ISD-3 on synth_6
additive (0.100, zero steps).
The table lists the multi-template configurations.

.. table:: ``rms_a_bias`` for the multi-template configurations.

   ===============  ======  ==========  =====  =====  ========  =========
   Configuration    OLS     ElasticNet  ISD-1  ISD-3  MCMC-add  MCMC-comb
   ===============  ======  ==========  =====  =====  ========  =========
   add synth_0..1   1.860   0.099       0.097  0.093  1.858     1.921
   add synth_0..2   1.950   0.013       0.111  0.088  1.947     2.041
   add synth_0..3   1.075   0.043       0.125  0.095  1.083     1.313
   add synth_0..4   13.389  0.034       0.162  0.140  13.380    14.240
   add synth_0..5   11.862  0.035       0.147  0.127  11.860    12.592
   add synth_0..6   13.017  0.045       0.142  0.124  13.030    13.961
   mult synth_0..1  1.622   0.000       0.000  0.000  1.620     2.181
   mult synth_0..2  1.413   0.000       0.000  0.000  1.410     1.347
   mult synth_0..3  0.225   0.000       0.005  0.000  0.217     1.216
   mult synth_0..4  11.873  0.000       0.005  0.005  11.864    8.696
   mult synth_0..5  11.368  0.000       0.005  0.004  11.366    10.015
   mult synth_0..6  12.947  0.008       0.004  0.004  12.961    11.566
   mixed n_mult=1   13.551  0.051       0.143  0.126  13.563    14.570
   mixed n_mult=2   13.392  0.053       0.147  0.115  13.405    13.920
   mixed n_mult=3   13.059  0.055       0.133  0.118  13.073    12.266
   mixed n_mult=4   12.628  0.062       0.136  0.122  12.642    11.291
   mixed n_mult=5   12.572  0.056       0.138  0.123  12.586    10.316
   mixed n_mult=6   12.748  0.063       0.151  0.123  12.762    9.696
   ===============  ======  ==========  =====  =====  ========  =========

OLS, MCMC-add and MCMC-comb give 0.22–2.18 for two to four templates and 8.7–14.6 from five
templates on; ElasticNet gives at most 0.099, and ISD-1 and ISD-3 at most 0.162.

Compute time
------------

The column ``time_s`` holds the wall-clock time of each method call on dahu148.
Means over the 32 configurations are 0.002 s (OLS), 0.16 s (ElasticNet), 0.20 s (ISD-1 and
ISD-3), 0.49 s (MCMC-add) and 58 s (MCMC-comb).
MCMC-comb takes 5.5–134 s per configuration: 5.5–6.2 s for single templates after the first
call, 55–127 s for the cumulative sets and 97–134 s in Tier 2.

The first configuration (synth_0 additive) includes one-time compilation and set-up: 13.9 s for
MCMC-comb, 2.4 s for MCMC-add, 1.4 s for ElasticNet and 0.3 s for ISD-1 and ISD-3.
MCMC-add also takes 1.2–2.0 s at the first configuration with each larger template count (the
cumulative additive sets), against 0.09–0.17 s in every other configuration.
Without the first configuration the means are 59 s (MCMC-comb), 0.43 s (MCMC-add) and 0.12 s
(ElasticNet), with the other methods unchanged.
Median timings with compilation excluded, across NSIDE and template counts, are on
:doc:`results_benchmark`.

.. figure:: _static/results_systematic_tests/timing_vs_ntemplates.png
   :width: 85%
   :align: center
   :alt: Wall-clock time per method against the number of templates, log scale

   Mean wall-clock time per method call against the number of templates, averaged over the
   configurations with that template count (log scale).

.. figure:: _static/results_systematic_tests/timing_mean_per_method.png
   :width: 75%
   :align: center
   :alt: Mean wall-clock time per method over all configurations, log scale

   Mean wall-clock time per method call over the 32 configurations (log scale).

Reproduce
---------

::

    python scripts/run_systematic_tests.py --nside 32 --output-dir results/systematic_tests/

``--sampler`` (``auto``, ``analytic``, ``nuts`` or ``emcee``; default ``auto``) selects the MCMC
backend, ``--nuts-warmup`` and ``--nuts-samples`` (default 1000 each) set the NUTS chains, and
``--isd-n-mocks`` (default 50) sets the number of uncontaminated mocks calibrating
:math:`\Delta\chi^2_{68}`; ``--n-walkers``, ``--n-steps`` and ``--n-burn`` apply only to
``--sampler emcee``.
``--tier2-only`` skips Tier 1 and ``--fast`` runs OLS and MCMC-comb only.
The script writes ``systematic_test_summary.csv``, ``isd_chi2_68.json``, the ``summary_*.png``
and ``timing_*.png`` figures, and one histogram per configuration in ``histograms/``; these are
copied unchanged into ``docs/_static/results_systematic_tests/``.
