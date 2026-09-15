nuts — gradient-based NUTS sampler
==================================

BlackJAX No-U-Turn sampler for the combined, multiplicative and skew-normal models.
Window adaptation tunes the step size and a dense mass matrix by default
(``dense_mass_matrix=True``); the runner is compiled once per configuration and
takes the data as arguments. On the LS10 combined fit at NSIDE 32 the dense matrix
halves the leapfrog steps per iteration and gives 2.5 times the minimum effective
samples per second of the diagonal one.

.. automodule:: sys_mapping.nuts
   :members:
   :undoc-members:
   :show-inheritance:
