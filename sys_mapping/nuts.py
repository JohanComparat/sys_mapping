"""Gradient-based NUTS inference (BlackJAX) for contamination parameters.

Drop-in alternative to the emcee sampler in :mod:`sys_mapping.inference` for the
*non-linear* models (``combined``, ``multiplicative``, and the skew-normal
likelihood).  The JAX log-likelihood from :mod:`sys_mapping.likelihood` is already
fully differentiable, so the No-U-Turn Sampler explores the posterior with
gradients — far fewer evaluations per effective sample than emcee's gradient-free
stretch move — and BlackJAX runs the entire chain under ``jax.lax.scan`` (no Python
per-step loop, no host↔device sync per walker).

Reparameterization
------------------
Only ``σ > 0`` is constrained.  We sample an *unconstrained* vector ``u`` and map
``σ = exp(u_σ)`` (identity for ``a``, ``b``, ``γ``), adding the transform
log-Jacobian ``log|dσ/du_σ| = u_σ``.  With the flat-in-σ prior this reproduces the
posterior emcee samples (see :func:`sys_mapping.inference.run_mcmc`), now free of the
hard ``σ > σ_min`` discontinuity that is incompatible with HMC trajectories.

Device portability
------------------
Multiple chains are run with a single :func:`jax.vmap`, so the same code runs on
CPU (few chains) or GPU (many chains); :func:`default_n_chains` picks a sensible
count from the detected backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

import blackjax

from .likelihood import make_log_likelihood
from .contamination import n_free_params


def default_n_chains() -> int:
    """Pick a chain count from the JAX backend (CPU: 4, GPU/TPU: 8)."""
    return 8 if jax.default_backend() in ("gpu", "tpu") else 4


@dataclass
class _NutsSampler:
    """Stand-in for :class:`emcee.EnsembleSampler` returned by :func:`run_nuts`.

    Exposes ``acceptance_fraction`` (read by existing callers) plus the NUTS
    convergence diagnostics ``rhat`` (max over parameters), ``ess`` (min over
    parameters) and ``num_divergences``.
    """

    acceptance_fraction: float
    rhat: float
    ess: float
    num_divergences: int
    n_chains: int
    n_samples: int


def build_logdensity(
    n_sys: int,
    model: str,
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    use_skewed: bool = False,
    prior_scale_a: float | None = None,
    prior_scale_b: float | None = None,
    precision=None,
):
    """Return ``(logdensity_fn, n_dim, idx_sigma)`` for NUTS in unconstrained space.

    ``logdensity_fn(u)`` maps the unconstrained vector ``u`` to the packed
    parameter vector (``σ = exp(u_σ)``), evaluates the JAX log-likelihood and adds
    the ``exp`` transform Jacobian.  Optional wide Gaussian priors on ``a`` / ``b``
    (``prior_scale_*``) regularize the fit and guard the ``ln|1+b·t|`` singularity;
    ``None`` leaves them flat (matching emcee).

    ``precision`` (a :class:`sys_mapping.covariance.LowRankPrecision`) switches the
    Gaussian likelihood to the correlated-noise (GLS) form :math:`C = \\sigma^2 R`;
    ``None`` keeps the white :math:`\\sigma^2 I` likelihood.
    """
    log_likelihood = make_log_likelihood(n_sys, model, use_skewed, precision=precision)
    _delta_g = jnp.asarray(delta_g_obs, dtype=jnp.float64)
    _delta_t = jnp.asarray(delta_t, dtype=jnp.float64)
    n_cont = n_free_params(n_sys, model)
    idx_sigma = n_cont
    n_dim = n_cont + 1 + (1 if use_skewed else 0)

    def logdensity_fn(u: jnp.ndarray) -> jnp.ndarray:
        u = jnp.asarray(u)
        u_sigma = u[idx_sigma]
        # map unconstrained -> constrained: sigma = exp(u_sigma), rest identity
        theta = u.at[idx_sigma].set(jnp.exp(u_sigma))
        lp = log_likelihood(theta, _delta_g, _delta_t)
        lp = lp + u_sigma  # log|d sigma / d u_sigma| = u_sigma
        if prior_scale_a is not None:
            a = u[:n_sys]
            lp = lp - 0.5 * jnp.sum((a / prior_scale_a) ** 2)
        if prior_scale_b is not None and model == "combined":
            b = u[n_sys : 2 * n_sys]
            lp = lp - 0.5 * jnp.sum((b / prior_scale_b) ** 2)
        return lp

    return logdensity_fn, n_dim, idx_sigma


def run_nuts(
    n_sys: int,
    *,
    model: str = "combined",
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    use_skewed: bool = False,
    n_chains: int | None = None,
    n_warmup: int = 1000,
    n_samples: int = 1000,
    seed: int = 42,
    target_acceptance_rate: float = 0.8,
    prior_scale_a: float | None = None,
    prior_scale_b: float | None = None,
    precision=None,
    chain_method: str = "vmap",
    progress: bool = False,
) -> tuple[np.ndarray, _NutsSampler]:
    """Run BlackJAX NUTS to infer contamination parameters.

    Window adaptation tunes the step size and (diagonal) mass matrix during
    ``n_warmup`` steps, then ``n_samples`` NUTS steps are drawn per chain under
    ``lax.scan``.  ``n_chains`` chains run in parallel via :func:`jax.vmap`.

    Parameters
    ----------
    n_sys : int
    model : str  ``'combined'`` (default), ``'multiplicative'``, or ``'additive'``
    delta_g_obs : (n_pix,) observed overdensity (keyword-only)
    delta_t : ``(n_sys, n_pix)`` template values (keyword-only)
    use_skewed : bool
    n_chains : int or None  parallel chains; defaults to :func:`default_n_chains`
    n_warmup : int  window-adaptation steps (discarded)
    n_samples : int  post-warmup NUTS steps *per chain*
    seed : int
    target_acceptance_rate : float  NUTS dual-averaging target (default 0.8)
    prior_scale_a, prior_scale_b : float or None  optional wide Gaussian prior scales
    precision : LowRankPrecision or None  correlated-noise (GLS) pixel correlation ``R``
        (``C = σ² R``); ``None`` keeps the white ``σ² I`` likelihood
    chain_method : str  how the chains are executed — ``'vmap'`` (default) runs them
        all at once (fastest, peak memory ∝ ``n_chains``), ``'sequential'``
        (:func:`jax.lax.map`) runs one at a time for ~``1/n_chains`` of the peak
        memory.  Use ``'sequential'`` when a multi-chain ``'vmap'`` run OOMs on a
        large footprint — it keeps R-hat available (which needs ``n_chains >= 2``)
        instead of forcing ``n_chains=1``.
    progress : bool  accepted for signature parity with :func:`run_mcmc` (unused)

    Returns
    -------
    flat_chain : (n_chains × n_samples, n_dim) posterior draws in the standard
        packed layout ``[a, b, σ, (γ)]`` (σ mapped back from ``exp(u_σ)``)
    sampler : :class:`_NutsSampler`  with ``acceptance_fraction``, ``rhat``,
        ``ess``, ``num_divergences``
    """
    if n_chains is None:
        n_chains = default_n_chains()

    logdensity_fn, n_dim, idx_sigma = build_logdensity(
        n_sys, model, delta_g_obs, delta_t, use_skewed,
        prior_scale_a=prior_scale_a, prior_scale_b=prior_scale_b,
        precision=precision,
    )
    n_cont = n_free_params(n_sys, model)

    # Initial positions (unconstrained): mirror the emcee init scheme, but with
    # u_sigma = log(sigma0) since sigma = exp(u_sigma).
    rng = np.random.default_rng(seed)
    sigma0 = max(float(np.std(delta_g_obs)), 1e-6)
    u0 = np.zeros((n_chains, n_dim))
    u0[:, :n_cont] = rng.normal(0.0, 0.05, (n_chains, n_cont))
    u0[:, idx_sigma] = np.log(sigma0) + rng.normal(0.0, 0.1, n_chains)
    if use_skewed:
        u0[:, idx_sigma + 1] = rng.normal(0.0, 0.1, n_chains)
    u0 = jnp.asarray(u0, dtype=jnp.float64)

    def run_one_chain(key, init_position):
        warmup = blackjax.window_adaptation(
            blackjax.nuts, logdensity_fn,
            target_acceptance_rate=target_acceptance_rate,
            progress_bar=False,
        )
        warmup_key, sample_key = jax.random.split(key)
        (last_state, parameters), _ = warmup.run(warmup_key, init_position, num_steps=n_warmup)
        kernel = blackjax.nuts(logdensity_fn, **parameters).step

        def one_step(state, k):
            state, info = kernel(k, state)
            return state, (state.position, info.is_divergent, info.acceptance_rate)

        keys = jax.random.split(sample_key, n_samples)
        _, (positions, divergent, accept) = jax.lax.scan(one_step, last_state, keys)
        return positions, divergent, accept

    chain_keys = jax.random.split(jax.random.PRNGKey(seed + 1), n_chains)
    # positions: (n_chains, n_samples, n_dim); divergent/accept: (n_chains, n_samples)
    #
    # chain_method: "vmap" runs every chain at once — fastest, but it holds all
    # chains' NUTS trajectories live simultaneously, which OOMs on large-n_pix
    # footprints (the ~94k-pixel Euclid TR1 combined fit) and forced callers down
    # to n_chains=1, where R-hat is undefined.  "sequential" (lax.map) runs one
    # chain at a time for ~1/n_chains of the peak memory at similar total work, so
    # multi-chain R-hat becomes affordable on the full footprint.
    if chain_method == "sequential":
        positions, divergent, accept = jax.lax.map(
            lambda xs: run_one_chain(*xs), (chain_keys, u0)
        )
    elif chain_method == "vmap":
        positions, divergent, accept = jax.vmap(run_one_chain)(chain_keys, u0)
    else:
        raise ValueError("chain_method must be 'vmap' or 'sequential', "
                         f"got {chain_method!r}")

    # Convergence diagnostics across chains (on the unconstrained samples).
    # R-hat needs >= 2 chains; with a single chain it is undefined (nan).
    ess = blackjax.diagnostics.effective_sample_size(
        positions, chain_axis=0, sample_axis=1
    )
    if n_chains > 1:
        rhat = blackjax.diagnostics.potential_scale_reduction(
            positions, chain_axis=0, sample_axis=1
        )
        rhat_val = float(jnp.max(rhat))
    else:
        rhat_val = float("nan")

    # Map sigma back to the constrained space, then flatten chains.
    positions = positions.at[:, :, idx_sigma].set(jnp.exp(positions[:, :, idx_sigma]))
    flat_chain = np.asarray(positions.reshape(-1, n_dim))

    sampler = _NutsSampler(
        acceptance_fraction=float(jnp.mean(accept)),
        rhat=rhat_val,
        ess=float(jnp.min(ess)),
        num_divergences=int(jnp.sum(divergent)),
        n_chains=n_chains,
        n_samples=n_samples,
    )
    return flat_chain, sampler
