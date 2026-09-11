sys_mapping.contamination
=========================

Forward and inverse contamination model (Berlfein et al. 2024, Eq. 11–13).

Implements the three nested models (additive, multiplicative, combined),
parameter packing/unpacking for the emcee/scipy interface, and the
pixel-level two-point correction formula.

Also implements a second, injection-only forward model:
:func:`~sys_mapping.contamination.apply_nonlinear_contamination` applies a
per-template *selection efficiency*
``1 + delta_obs = (1 + delta_true) * prod_i (1 + F_i(t_i))`` in which ``F`` need
not be linear.  That is the model ISD's weight inverts, and it is not Eq. 13 --
the two coincide only when every ``F`` is linear and ``a = b``.  It exists
because a contamination linear in every template is one a linear marginal fit
already suffices for, so it cannot distinguish ``ISD-1`` from ``ISD-3``.

**Key paper:** `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_ — see also :doc:`../methods`.

.. automodule:: sys_mapping.contamination
   :members:
   :show-inheritance:
