sys_mapping.nuts
================

``sys_mapping.nuts`` runs the BlackJAX No-U-Turn sampler for the combined, multiplicative and skew-normal models.
Window adaptation tunes the step size and the mass matrix, and the runner is compiled once per configuration and takes the data as arguments.
We use a dense mass matrix by default (``dense_mass_matrix=True``):
on the LS10 combined fit at NSIDE 32, it halves the leapfrog steps per iteration and gives 2.5 times the minimum effective samples per second of the diagonal one.

.. automodule:: sys_mapping.nuts
   :members:
   :undoc-members:
   :show-inheritance:
