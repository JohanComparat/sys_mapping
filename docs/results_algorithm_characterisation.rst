Algorithm characterisation
==========================

Provenance, with the data in ``_static/characterisation/``: :ref:`char-break-even` from
campaign F, tag 20260907c (sys_mapping 1.2.0, dahu, 2026-09-07 to 09-08);
:ref:`char-variance-inflation` from campaign C, tag 20260904 (dahu, 2026-09-04; healpy and
NumPy only); :ref:`char-null-spectra` from campaign B, tag 20260904b (1.2.0, dahu,
2026-09-04 to 09-05) and campaign M, tags 20260906m and 20260907m (1.2.0, dahu, 2026-09-06
to 09-07); :ref:`char-crossterms` from the LS10 amplitudes of campaign P, tag 20260912p
(1.2.0, dahu, 2026-09-11 to 09-12), evaluated with 1.4.0 on a laptop on 2026-09-15. Scripts:
`sys_mapping_benchmark <https://github.com/JohanComparat/sys_mapping_benchmark>`_,
``characterisation/``.

.. contents:: On this page
   :local:
   :depth: 1

----

.. _char-break-even:

1. The break-even condition
---------------------------

We measure when a correction improves :math:`w(\theta)` on the family-F simulations, in the
three tables below. Each cell is a GLASS mock of 500 000 galaxies from the parametric
spectrum :math:`5\times10^{-4}\,(\ell+1)^{-1.5}`, contaminated through five LS10 and Gaia
templates (additive, multiplicative and combined, per-template amplitudes 0.02, 0.05 and
0.10) and corrected by the six methods, ISD with a :math:`\Delta\chi^2_{68}` calibrated on
30 clean mocks. Fifty seeds at NSIDE 32 and 64 and 34 at NSIDE 128 give 134 cells × 9
configurations × 6 methods = 7236 method–configuration cells.

.. math::

   A = {\rm rms}\Bigl(\sum_i a_i^{\rm true} t_i\Bigr), \qquad
   \Delta A = {\rm rms}\Bigl(\sum_i (\hat a_i - a_i^{\rm true})\, t_i\Bigr)

are the additive field the correction removes and the mis-fit field it adds back, so
correcting helps once :math:`A/\Delta A \gtrsim 1`. The improvement factor is
:math:`B_{\rm cont}/B_{\rm corr}`, with
:math:`B = \lVert w - w_{\rm true}\rVert_2 / \lVert w_{\rm true}\rVert_2` over ten bins from
0.13 to 8.2 deg; a cell is helped when it exceeds 1.

.. list-table:: Additive and combined cells (4824)
   :header-rows: 1
   :widths: 40 30 30

   * -
     - correction helped
     - correction hurt
   * - :math:`A/\Delta A > 1`
     - 3971
     - 351
   * - :math:`A/\Delta A < 1`
     - 236
     - 266

"Correct only if :math:`A/\Delta A > 1`" is right in 87.8 % of these cells; the Spearman
correlation between :math:`\log(A/\Delta A)` and the log improvement factor is +0.42
(n = 4824). In the 2412 multiplicative cells :math:`A = 0`: ISD-1, ISD-3 and ElasticNet
apply no additive correction in 562, and correcting helped in 749 of the other 1850.

.. list-table:: Per method, additive and combined cells (804 each)
   :header-rows: 1
   :widths: 25 25 25 25

   * - method
     - median :math:`A/\Delta A`
     - helped
     - rule right
   * - OLS
     - 2.26
     - 88.9 %
     - 78.0 %
   * - ISD-1
     - 1.89
     - 95.5 %
     - 95.5 %
   * - ISD-3
     - 1.88
     - 94.3 %
     - 94.3 %
   * - ElasticNet
     - 2.23
     - 89.1 %
     - 79.0 %
   * - MCMC-add
     - 2.26
     - 82.0 %
     - 94.2 %
   * - MCMC-comb
     - 2.28
     - 73.5 %
     - 86.1 %

.. list-table:: Per resolution, additive and combined cells
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - NSIDE
     - cells
     - median :math:`A/\Delta A`
     - helped
     - rule right
   * - 32
     - 1800
     - 2.26
     - 96.7 %
     - 96.7 %
   * - 64
     - 1800
     - 2.19
     - 88.7 %
     - 88.1 %
   * - 128
     - 1224
     - 1.48
     - 71.1 %
     - 74.5 %

Every ISD cell lies above break-even. Per-cell values are in
``breakeven_20260907c_joined.csv``, counts in ``breakeven_20260907c_summary.json``.

----

.. _char-variance-inflation:

2. Inflation of the independent-pixel error
-------------------------------------------

We measure how far the analytic OLS error underestimates the scatter of the fitted
amplitudes on uncontaminated maps, in the two tables below, through

.. math::

   \kappa_i = \frac{\sigma^{\rm emp}[\hat a_i]}{\sigma^{\rm iid}[\hat a_i]}, \qquad
   \kappa_{\rm field} = \Bigl(\frac{\langle A^2\rangle}{s^2\, n_s / n_{\rm pix}}\Bigr)^{1/2},
   \quad A^2 = {\rm mean}\bigl[(\hat a \cdot t)^2\bigr],

the denominator of :math:`\kappa_{\rm field}` being :math:`\langle A^2\rangle` under the
independent-pixel model. Each cell fits the first :math:`n_s \in \{3, 5, 7, 9, 11\}` of
eleven LS10 and Gaia maps to 2000 realisations at 127 galaxies per pixel, at NSIDE 32 and
64, for a lognormal field with :math:`C_\ell \propto (\ell+1)^{-\alpha}`,
:math:`\alpha \in \{1, 1.5, 2, 2.5, 3\}`, :math:`\sigma_{\rm clus} \in \{0.2, 0.4\}`, and
for a Poisson-only control; tag 20260907c repeats the grid with identical values.

In the control :math:`\kappa_{\rm field}` lies in [0.987, 1.021] over the 100 fits and the
per-template 3σ false-positive rate has median 0.26 % against a nominal 0.27 %.

.. list-table:: :math:`\kappa_{\rm field}` at :math:`\alpha = 2`, :math:`\sigma_{\rm clus} = 0.4`
   :header-rows: 1
   :widths: 14 14 14 14 14 14

   * - NSIDE
     - :math:`n_s=3`
     - :math:`n_s=5`
     - :math:`n_s=7`
     - :math:`n_s=9`
     - :math:`n_s=11`
   * - 32
     - 7.41
     - 6.53
     - 6.50
     - 5.88
     - 5.30
   * - 64
     - 12.94
     - 11.09
     - 10.75
     - 9.80
     - 8.76

:math:`\kappa_{\rm field}` is 5.3–12.9 at this spectrum and 2.1–22.7 over the grid, larger at
NSIDE 64 than at 32 in every cell. Some template exceeds 3σ in 96–100 % of realisations at
NSIDE 32 and 99–100 % at NSIDE 64 at this spectrum, and in 55–100 % over the grid. Doubling
:math:`\sigma_{\rm clus}` from 0.2 to 0.4 changes :math:`\kappa_{\rm field}` by 2–7 %.

.. list-table:: Spectral slope at NSIDE 64, :math:`n_s = 11`, :math:`\sigma_{\rm clus} = 0.4`
   :header-rows: 1
   :widths: 30 14 14 14 14 14

   * - :math:`\alpha`
     - 1.0
     - 1.5
     - 2.0
     - 2.5
     - 3.0
   * - :math:`\kappa_{\rm field}`
     - 2.73
     - 5.10
     - 8.76
     - 12.80
     - 16.10
   * - any template > 3σ
     - 88.6 %
     - 99.4 %
     - 100 %
     - 100 %
     - 100 %

:func:`~sys_mapping.diagnostics.calibrated_template_significance` and
:func:`~sys_mapping.covariance.mock_sandwich_covariance` take the amplitude scatter from null
realisations. All rows are in ``variance_inflation_20260904.csv``.

----

.. _char-null-spectra:

3. Clustering of the calibration nulls
--------------------------------------

Footprint matching fixes the density of a null mock but not its clustering. We compare
:math:`\sigma_{\rm clus} = (\hat\sigma^2 - 1/\bar n)^{1/2}` of the LS10 samples with that of
two nulls in the table below. In campaign B the power-law null
:math:`5\times10^{-4}\,(\ell+1)^{-1.5}` for :math:`\log M_\star \ge 10.0` at NSIDE 64 holds
127.3 galaxies per pixel against the data's 127.4 and has :math:`\hat\sigma = 0.118`
against 0.397; with the galaxy and random-catalogue shot terms removed,
:math:`\sigma_{\rm clus} = 0.045` against 0.387 (median of five seeds).

The data columns are from campaign B. The null columns are from ``match_glass_to_data.py``
(campaign M), which measures a mock through the data's estimator on the sample's
footprint: the power law is its first iteration, the matched spectrum the fit that
:func:`~sys_mapping.glass_mocks.load_matched_cl` serves, measured on eight seeds the fit
did not use. The large-scale ratio is the mock-to-data band power over the multipoles of
:math:`r_p = 5`–20 :math:`h^{-1}` Mpc, or the largest scales the map resolves; a spectrum
is validated when it is within 0.10 of 1 and the density within 3 % plus its uncertainty.

.. csv-table:: Clustering of the data and of the two nulls
   :header: "log M* ≥", "NSIDE", ":math:`\bar n`", ":math:`\sigma_{\rm clus}` data", ":math:`\sigma_{\rm clus}` power law", ":math:`\sigma_{\rm clus}` matched", "large-scale ratio", ":math:`\ell` range"
   :widths: 10 8 10 12 14 12 12 12

   "9.00", "32", "96.6", "0.538", "0.037", "0.510", "1.016", "31–63"
   "9.50", "32", "264.5", "0.467", "0.036", "0.436", "1.016", "45–64"
   "10.00", "32", "509.4", "0.378", "0.036", "0.349", "0.997", "57–64"
   "10.25", "32", "610.9", "0.339", "0.035", "0.315", "1.020", "16–64"
   "10.50", "32", "602.5", "0.321", "0.036", "0.297", "1.015", "16–64"
   "10.75", "32", "517.4", "0.302", "0.035", "0.278", "1.017", "16–64"
   "11.00", "32", "299.1", "0.291", "0.036", "0.268", "1.012", "16–64"
   "11.25", "32", "100.0", "0.315", "0.037", "0.287", "1.006", "16–64"
   "11.50", "32", "22.3", "0.392", "0.035", "0.349", "1.017", "16–64"
   "9.00", "64", "24.2", "0.645", "0.042", "0.606", "0.972", "31–126"
   "9.50", "64", "66.1", "0.509", "0.043", "0.471", "0.927", "45–89"
   "10.00", "64", "127.4", "0.387", "0.043", "0.370", "1.016", "64–128"
   "10.25", "64", "152.7", "0.334", "0.044", "0.320", "0.997", "77–128"
   "10.50", "64", "150.6", "0.298", "0.043", "0.280", "0.950", "90–128"
   "10.75", "64", "129.4", "0.269", "0.044", "0.251", "0.957", "106–128"
   "11.00", "64", "74.8", "0.274", "0.044", "0.256", "0.912", "117–128"
   "11.25", "64", "25.0", "0.328", "0.041", "0.307", "0.962", "118–128"
   "11.50", "64", "5.6", "0.485", "—", "—", "—", "—"

The power law gives :math:`\sigma_{\rm clus} = 0.035`–0.044, 0.07–0.16 of the data's, a
clustering variance 38 to 230 times too low. The matched spectra give 0.89–0.96 of the
data's :math:`\sigma_{\rm clus}`, large-scale ratios of 0.912–1.020 and densities within
2.0 %. For :math:`\log M_\star \ge 11.5` at NSIDE 64 (5.6 galaxies per pixel) no fit is
validated and :func:`~sys_mapping.glass_mocks.load_matched_cl` serves the NSIDE 32 fit; an
NSIDE 128 analysis uses the NSIDE 64 fit. Per-cell details are in
``null_spectra_nside32_64.csv``, the spectra in ``matched_spectra/`` of the benchmark
repository.

A null for data draws from the sample's matched spectrum (``--null-cl-file`` in
``run_ls10_analysis.py``):

.. code-block:: python

   cl = sm.load_matched_cl("matched_spectra", sample=sample_id, nside=nside)
   null = sm.generate_glass_null_overdensity(
       400, nside, good_pixels, n_galaxies_footprint, z_max, seed=0, cl_input=cl)
   significance = sm.calibrated_template_significance(delta_g, delta_t, null)

``cl_amplitude`` is for synthetic studies whose truth is that power law, as in
:ref:`char-break-even`; a mock given neither ``cl_input`` nor ``cl_amplitude`` warns.

----

.. _char-crossterms:

4. Cross-template terms of the two-point correction
---------------------------------------------------

A two-point correction removes :math:`\sum_{ij} a_i a_j \xi_{ij}(\theta)`. The LS10 products
apply the full :math:`(n_{\rm sys}, n_{\rm sys}, n_\theta)` matrix in the PCA-rotated basis,
66 correlations at :math:`n_{\rm sys} = 11` measured on the galaxies
(``--ct-auto-only`` in ``run_ls10_analysis.py`` keeps the diagonal). We measure the share of
the correction carried by the off-diagonal terms,
:math:`|\Delta w_{\rm full} - \Delta w_{\rm auto}| / |\Delta w_{\rm full}|`, in the table
below.

We take the OLS amplitudes of the nine samples at NSIDE 32 and 64, standardise the templates
on each sample's footprint (reproducing the basis recorded in the products), compute
:math:`\xi_{ij}` from pixel cross-spectra on that footprint with the estimator of
``run_crossterm_bias.py`` in ten bins from 0.5 to 5 deg, and rotate the amplitudes with the
templates, :math:`a_{\rm rot} = R\,a`, which leaves the full correction unchanged to
:math:`10^{-13}`. Each sample's share is its median over :math:`\theta`.

.. list-table:: Cross-term share of the correction, OLS amplitudes
   :header-rows: 1
   :widths: 14 22 22 22 20

   * - NSIDE
     - basis
     - median of samples
     - range of samples
     - largest bin
   * - 32
     - unrotated
     - 606 %
     - 340–1258 %
     -
   * - 32
     - PCA-rotated
     - 23.9 %
     - 18.5–25.9 %
     - 31.9 %
   * - 64
     - unrotated
     - 121 %
     - 4.4–648 %
     -
   * - 64
     - PCA-rotated
     - 22.5 %
     - 12.2–31.5 %
     - 34.0 %

In the rotated basis MCMC-add matches OLS to 0.05 percentage points, MCMC-comb gives 18.4 %
and 26.6 % and ElasticNet 25.2 % and 25.7 % at NSIDE 32 and 64. All six methods are in
``crossterm_share_20260912p.csv``.
