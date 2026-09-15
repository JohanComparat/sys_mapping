.. _ls10-recommendations:

LS10 systematic-correction recommendations
==========================================

The weight column to use per sample, from two tests on the issued products: whether
any template is detected by the calibrated significance, and whether the
mock-calibrated likelihood ratio on the grid nearest the issued resolution (NSIDE 32
below NSIDE 64, NSIDE 64 otherwise) requires the multiplicative term. Both at
α = 0.05.

.. csv-table::
   :header: "log M* ≥", "NSIDE", "family-wise p", "likelihood-ratio p", "column", "reason"
   :widths: 8, 7, 11, 13, 13, 48

   "9.0", "32", "0.0524", "0.020 (NSIDE 32)", "``WEIGHT_SYS``", "the NSIDE 32 likelihood ratio requires the multiplicative term (p = 0.020)"
   "9.5", "64", "≤ 0.0025", "0.020 (NSIDE 64)", "``WEIGHT_SYS``", "a template is detected (family-wise p ≤ 0.0025) and the NSIDE 64 likelihood ratio requires the multiplicative term (p = 0.020)"
   "10.0", "128", "≤ 0.0025", "0.020 (NSIDE 64)", "``WEIGHT_SYS``", "a template is detected (family-wise p ≤ 0.0025) and the NSIDE 64 likelihood ratio requires the multiplicative term (p = 0.020)"
   "10.25", "128", "≤ 0.0025", "0.078 (NSIDE 64)", "``WEIGHT_SYS``", "a template is detected (family-wise p ≤ 0.0025); the NSIDE 64 likelihood ratio does not require the multiplicative term (p = 0.078), so WEIGHT_ADD is the simpler alternative"
   "10.5", "128", "≤ 0.0025", "0.373 (NSIDE 64)", "``WEIGHT_SYS``", "a template is detected (family-wise p ≤ 0.0025); the NSIDE 64 likelihood ratio does not require the multiplicative term (p = 0.373), so WEIGHT_ADD is the simpler alternative"
   "10.75", "128", "≤ 0.0025", "0.020 (NSIDE 64)", "``WEIGHT_SYS``", "a template is detected (family-wise p ≤ 0.0025) and the NSIDE 64 likelihood ratio requires the multiplicative term (p = 0.020)"
   "11.0", "64", "≤ 0.0025", "0.078 (NSIDE 64)", "``WEIGHT_SYS``", "a template is detected (family-wise p ≤ 0.0025); the NSIDE 64 likelihood ratio does not require the multiplicative term (p = 0.078), so WEIGHT_ADD is the simpler alternative"
   "11.25", "64", "≤ 0.0025", "0.020 (NSIDE 64)", "``WEIGHT_SYS``", "a template is detected (family-wise p ≤ 0.0025) and the NSIDE 64 likelihood ratio requires the multiplicative term (p = 0.020)"
   "11.5", "16", "0.8903", "0.020 (NSIDE 32)", "``WEIGHT_SYS``", "the NSIDE 32 likelihood ratio requires the multiplicative term (p = 0.020)"

``WEIGHT_SYS`` is ``WEIGHT_COMB``, the combined additive and multiplicative model.
``WEIGHT_ISD3`` is recommended only where neither test finds contamination, following
the break-even condition measured on :doc:`results_algorithm_characterisation`.
