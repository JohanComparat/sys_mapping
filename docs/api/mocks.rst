sys_mapping.mocks
=================

``sys_mapping.mocks`` generates synthetic galaxy catalogues with controlled contamination, for the validation of this package.
It takes the NSIDE, the number of templates, the true amplitudes (optional), the scenario, the mean galaxy density and a seed,
and returns a :class:`~sys_mapping.mocks.MockCatalog` with RA/Dec arrays, template maps, true and observed overdensity fields, the Galactic mask, and the true contamination parameters.

A mock is built in four steps:

1. A lognormal galaxy field (Coles & Jones 1991): the overdensity :math:`\delta_g = \exp(G - \sigma^2/2) - 1`
   from a Gaussian random field with :math:`C_\ell \propto (\ell+1)^{-2}`.
2. A Galactic mask at a configurable latitude cut.
3. Contamination injection with a chosen scenario (additive, multiplicative, combined or none)
   and known amplitudes :math:`a_i^{\rm true}`, :math:`b_i^{\rm true}`.
4. Poisson sampling of the galaxy and random catalogues.

The :class:`~sys_mapping.mocks.MockCatalog` dataclass stores all inputs and outputs.

.. automodule:: sys_mapping.mocks
   :members:
   :show-inheritance:
   :exclude-members: scenario,nside,ra_gal,dec_gal,ra_rand,dec_rand,templates,delta_true,delta_obs,mask,a_true,b_true,n_mean,sigma,seed
