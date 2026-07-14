"""JAX JIT-compiled log-likelihood factory (Eq. 17-18, Berlfein et al. 2024).

Gaussian log-likelihood (Eq. 17):
    ln L = -(N/2) ln(2πσ²) - Σ ln|1 + b·δ_{t,j}|
           - (1/2σ²) Σ δ_{g,j}²

where δ_{g,j} = (δ_g_obs,j - Σ a_i t_ij) / (1 + Σ b_i t_ij) is the clean overdensity.

Skew-normal log-likelihood (Eq. 18):
    ln L_skew = ln L_gauss + N·ln 2 + Σ log_ndtr(γ·δ_{g,j}/σ)

where ξ = -σ·δ·√(2/π), δ = γ/√(1+γ²) is the mean correction for zero mean,
and log_ndtr(z) = log Φ(z) is the standard-normal log-CDF (numerically stable).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import log_ndtr

from .contamination import unpack_params


def make_log_likelihood(
    n_sys: int,
    model: str,
    use_skewed: bool = False,
    precision=None,
) -> Callable[[Array, Array, Array], Array]:
    """Return a JIT-compiled log-likelihood function for the given model.

    The returned function is decorated with ``@jax.jit`` and traces on
    first call. Reuse it across MCMC iterations to amortize compilation cost.

    Parameters
    ----------
    n_sys : int
        Number of systematic templates.
    model : str
        One of ``'additive'``, ``'multiplicative'``, or ``'combined'``.
    use_skewed : bool
        If True, use skew-normal likelihood (Eq. 18); otherwise Gaussian (Eq. 17).
    precision : LowRankPrecision, optional
        Fixed unit-diagonal pixel **correlation** operator ``R`` (see
        :mod:`sys_mapping.covariance`).  When given, the white quadratic form
        :math:`\\sum_j r_j^2` is replaced by the generalized (GLS) form
        :math:`r^\\top R^{-1} r` and the constant :math:`-\\tfrac12 \\ln|R|` is
        added, i.e. the covariance is :math:`C = \\sigma^2 R` instead of
        :math:`\\sigma^2 I`.  ``None`` (default) keeps the exact white behavior.
        ``R`` must not depend on the sampled parameters, so gradients w.r.t.
        ``(a, b, sigma, gamma)`` are unchanged in cost.

    Returns
    -------
    log_likelihood(theta, delta_g_obs, delta_t) -> scalar
        ``theta``        : flat parameter vector (see :func:`~contamination.pack_params`)
        ``delta_g_obs``  : (n_pix,) observed overdensity
        ``delta_t``      : (n_sys, n_pix) template values at unmasked pixels

    Performance
    -----------
    Measured on CPU (n_pix=10_000, n_sys=5):

    - **First call (JIT compile): ~227 ms** — call once before MCMC loops.
    - **Subsequent calls (Gaussian): ~284 μs/call**.
    - **Subsequent calls (skew-normal): ~592 μs/call** (extra ``log_ndtr`` sum).

    Precision
    ---------
    At zero contamination (a=b=0), matches ``scipy.stats.norm.logpdf`` sum to
    ``rtol=1e-5``. JAX gradient at the additive OLS MLE is < 1e-4 for n_pix ≥ 10_000.
    Setting ``gamma=0`` in skew mode recovers the Gaussian to ``atol=1e-6``.

    Examples
    --------
    >>> import numpy as np, jax.numpy as jnp
    >>> from sys_mapping import make_log_likelihood, pack_params
    >>> rng = np.random.default_rng(0)
    >>> n_pix, n_sys = 5000, 3
    >>> delta_g = rng.standard_normal(n_pix) * 0.1
    >>> delta_t = rng.standard_normal((n_sys, n_pix))
    >>> a = np.zeros(n_sys); b = np.zeros(n_sys); sigma = 0.1
    >>> log_lik = make_log_likelihood(n_sys, "combined")   # compile once
    >>> theta = jnp.asarray(pack_params(a, b, sigma, model="combined"))
    >>> ll = float(log_lik(theta, jnp.asarray(delta_g), jnp.asarray(delta_t)))
    >>> ll  # typical value: large negative number
    """
    _model = model
    _n_sys = n_sys
    _use_skewed = use_skewed
    _precision = precision

    def _quad_and_logdet(residual: Array) -> tuple[Array, Array]:
        """Return ``(r^T R^{-1} r, ln|R|)``.  White (``R = I``) unless a precision is given."""
        if _precision is None:
            return jnp.sum(residual**2), 0.0
        return _precision.quad(residual), _precision.logdet

    @jax.jit
    def log_likelihood(
        theta: Array,
        delta_g_obs: Array,
        delta_t: Array,
    ) -> Array:
        a, b, sigma, gamma = unpack_params(theta, _n_sys, _model, _use_skewed)

        mult_term = jnp.einsum("i,ij->j", b, delta_t)
        add_term = jnp.einsum("i,ij->j", a, delta_t)

        # Jacobian factor from the change of variables (Eq. 17)
        log_jac = jnp.sum(jnp.log(jnp.abs(1.0 + mult_term)))

        # Residual: observed minus additive contamination, then scale by (1+b*t)
        # Under the model delta_g_obs = delta_g*(1+b*t) + a*t, so
        # delta_g = (delta_g_obs - a*t) / (1+b*t)
        delta_g_clean = (delta_g_obs - add_term) / (1.0 + mult_term)
        n_pix = delta_g_obs.shape[0]

        if _use_skewed:
            # Mean correction so that E[X] = 0 for skew-normal
            delta_param = gamma / jnp.sqrt(1.0 + gamma**2)
            xi = -sigma * delta_param * jnp.sqrt(2.0 / jnp.pi)
            residual = delta_g_clean - xi
            # Gaussian part.  With a correlated-noise precision R (C = sigma^2 R) the white
            # sum-of-squares becomes r^T R^{-1} r and the -0.5*ln|R| constant is added.
            quad, logdet_R = _quad_and_logdet(residual)
            log_gauss = (
                -0.5 * n_pix * jnp.log(2.0 * jnp.pi * sigma**2)
                - 0.5 * logdet_R
                - 0.5 / sigma**2 * quad
            )
            # Skewness correction: Σ log Φ(γ·r/σ) = Σ log_ndtr(z)
            # The 2/σ in the PDF contributes N·log(2); Φ contributes Σ log_ndtr(z).
            # Derivation: log Φ(z) = log_ndtr(z); using identity
            #   log(1+erf(w)) = log(2) + log_ndtr(√2·w),
            # set w = z/√2 → log_ndtr(z) = log(1+erf(z/√2)) − log(2). ✓
            z = gamma * residual / sigma
            log_skew = n_pix * jnp.log(2.0) + jnp.sum(log_ndtr(z))
            return log_gauss + log_skew - log_jac
        else:
            residual = delta_g_clean
            quad, logdet_R = _quad_and_logdet(residual)
            log_gauss = (
                -0.5 * n_pix * jnp.log(2.0 * jnp.pi * sigma**2)
                - 0.5 * logdet_R
                - 0.5 / sigma**2 * quad
            )
            return log_gauss - log_jac

    return log_likelihood
