sys_mapping.mocks
=================

Synthetic galaxy catalog generation for package validation.

Generates physically motivated mock catalogs with controlled contamination:

1. **Lognormal galaxy field** (Coles & Jones 1991): overdensity
   :math:`\delta_g = \exp(G - \sigma^2/2) - 1` from a Gaussian random
   field with :math:`C_\ell \propto (\ell+1)^{-2}`.
2. **Galactic mask** at configurable latitude cut.
3. **Contamination injection** with chosen scenario (additive / multiplicative
   / combined / none) and known amplitudes :math:`a_i^{\rm true}`,
   :math:`b_i^{\rm true}`.
4. **Poisson sampling** of galaxy and random catalogs.

The :class:`~sys_mapping.mocks.MockCatalog` dataclass stores all inputs and
outputs for convenient downstream use.

**Inputs:** NSIDE, number of templates, true amplitudes (optional), scenario,
mean galaxy density, seed.

**Outputs:** :class:`~sys_mapping.mocks.MockCatalog` with RA/Dec arrays,
template maps, true and observed overdensity fields, galactic mask, and
the true contamination parameters.

.. automodule:: sys_mapping.mocks
   :members:
   :show-inheritance:
   :exclude-members: scenario,nside,ra_gal,dec_gal,ra_rand,dec_rand,templates,delta_true,delta_obs,mask,a_true,b_true,n_mean,sigma,seed
