sys_mapping.contamination
=========================

Forward and inverse contamination model (Berlfein et al. 2024, Eq. 11–13).

The forward model :math:`\delta_g^{\rm obs} = \delta_g(1 + \mathbf b\cdot\mathbf t) +
\mathbf a\cdot\mathbf t` in three nested forms: additive (:math:`\mathbf b=0`),
multiplicative (:math:`\mathbf a=0`) and combined (both free). The module packs and
unpacks the parameter vector shared by the likelihood and the samplers, and computes
the two-point correction with scalar, per-template or full-matrix amplitudes.

Also implements a second, injection-only forward model:
:func:`~sys_mapping.contamination.apply_nonlinear_contamination` applies a
per-template *selection efficiency*
``1 + delta_obs = (1 + delta_true) * prod_i (1 + F_i(t_i))`` with ``F`` not
necessarily linear. This is the model the ISD weight inverts. It reduces to Eq. 13
to first order when every ``F_i`` is linear, with ``a = b``. A non-linear ``F``
separates ``ISD-1`` from ``ISD-3``, which a linear contamination cannot.

**Key paper:** `Berlfein et al. 2024 <https://arxiv.org/abs/2401.12293>`_ — see also :doc:`../methods`.

.. automodule:: sys_mapping.contamination
   :members:
   :show-inheritance:
