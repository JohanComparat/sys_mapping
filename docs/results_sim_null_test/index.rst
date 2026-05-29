Null test — uncontaminated simulations
=======================================

Summary over up to 50 realizations per bin.
No systematics are injected; the pipeline should detect nothing.
A method **passes** the null test if its false positive rate (FPR) at 3σ is ≤ 5%.

.. toctree::
   :hidden:

   bin_1
   bin_2
   bin_3
   bin_4
   bin_5
   bin_6

Summary table
-------------

.. list-table::
   :widths: 10 10 15 15 15 15 25
   :header-rows: 1

   * - Bin
     - N sims
     - FPR MCMC (3σ)
     - FPR OLS (3σ)
     - ratio @ 10'
     - ratio @ 200'
     - Verdict (MCMC-comb)
   * - :doc:`bin_1`
     - 50
     - 92.0%
     - 96.0%
     - 0.9442
     - -1.3462
     - FAILS (FPR > 5%)
   * - :doc:`bin_2`
     - 50
     - 86.0%
     - 88.0%
     - 0.9357
     - -0.0682
     - FAILS (FPR > 5%)
   * - :doc:`bin_3`
     - 50
     - 80.0%
     - 88.0%
     - 0.9474
     - 0.8734
     - FAILS (FPR > 5%)
   * - :doc:`bin_4`
     - 50
     - 80.0%
     - 82.0%
     - 0.9598
     - 1.0455
     - FAILS (FPR > 5%)
   * - :doc:`bin_5`
     - 50
     - 76.0%
     - 82.0%
     - 0.9493
     - 0.7083
     - FAILS (FPR > 5%)
   * - :doc:`bin_6`
     - 50
     - 80.0%
     - 82.0%
     - 0.9590
     - 0.3125
     - FAILS (FPR > 5%)

