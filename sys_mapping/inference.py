"""MCMC inference using emcee for systematic contamination parameters.

Uses the JIT-compiled log-likelihood from likelihood.py.
emcee walkers are initialized near reasonable starting values and
the JAX computation is called from within a numpy-compatible wrapper.

GPU / vectorised mode
---------------------
Pass ``vectorize=True`` to :func:`run_mcmc` (or :func:`make_log_prob`) to
enable batched walker evaluation.  emcee then calls the log-prob once per step
with the full ``(n_walkers, n_dim)`` position matrix, and a single
``jax.vmap``-over-``jax.jit`` kernel evaluates all walkers simultaneously.
This turns ``n_walkers`` sequential JAX dispatches into one batched GPU kernel
per step — the primary source of GPU speedup.  Requires emcee ≥ 3.0 (which
supports ``EnsembleSampler(vectorize=True)``).
"""

from __future__ import annotations

import numpy as np
import emcee
import jax
import jax.numpy as jnp
from jax import Array

from .likelihood import make_log_likelihood
from .contamination import n_free_params, pack_params


def make_log_prob(
    n_sys: int,
    model: str,
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    use_skewed: bool = False,
    sigma_min: float = 1e-6,
    vectorize: bool = False,
) -> tuple[callable, int]:
    """Build an emcee-compatible log_prob function with a sigma > 0 prior.

    Wraps the JIT-compiled log-likelihood from :func:`~likelihood.make_log_likelihood`
    in a Python callable that emcee can call. Returns ``-inf`` for ``sigma ≤ sigma_min``.

    Parameters
    ----------
    n_sys : int
    model : str  ``'additive'``, ``'multiplicative'``, or ``'combined'``
    delta_g_obs : (n_pix,) observed overdensity
    delta_t : (n_sys, n_pix) template values
    use_skewed : bool
    sigma_min : float  lower bound on sigma (hard prior); default ``1e-6``
    vectorize : bool
        If True, return a *batched* log_prob that accepts ``(n_walkers, n_dim)``
        and returns ``(n_walkers,)``.  Uses ``jax.vmap`` so all walkers are
        evaluated in a single GPU kernel.  Pass ``vectorize=True`` to
        ``emcee.EnsembleSampler`` as well.

    Returns
    -------
    log_prob : callable(theta) -> float  (scalar mode)
               callable(thetas: (n_walkers, n_dim)) -> (n_walkers,)  (vectorized mode)
    n_dim : int  dimensionality of the parameter space

    Precision
    ---------
    All floating-point arithmetic delegated to the JAX log-likelihood
    (see :func:`~likelihood.make_log_likelihood` for precision details).

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import make_log_prob
    >>> rng = np.random.default_rng(0)
    >>> delta_g = rng.standard_normal(2000) * 0.1
    >>> delta_t = rng.standard_normal((3, 2000))
    >>> log_prob, n_dim = make_log_prob(3, "combined", delta_g, delta_t)
    >>> n_dim  # 3a + 3b + 1sigma = 7
    7
    >>> import numpy as np
    >>> theta0 = np.zeros(n_dim); theta0[-1] = 0.1  # sigma = 0.1
    >>> log_prob(theta0)  # finite negative number
    """
    log_likelihood = make_log_likelihood(n_sys, model, use_skewed)
    _delta_g = jnp.asarray(delta_g_obs)
    _delta_t = jnp.asarray(delta_t)
    n_cont = n_free_params(n_sys, model)
    n_dim = n_cont + 1 + (1 if use_skewed else 0)

    if vectorize:
        # Single vmap+jit kernel: evaluates all walkers in one GPU dispatch.
        _batched_ll = jax.jit(
            jax.vmap(lambda theta: log_likelihood(theta, _delta_g, _delta_t))
        )

        def log_prob_vec(thetas: np.ndarray) -> np.ndarray:
            lps = np.array(_batched_ll(jnp.asarray(thetas, dtype=jnp.float64)))
            lps[thetas[:, n_cont] <= sigma_min] = -np.inf
            lps[~np.isfinite(lps)] = -np.inf
            return lps

        return log_prob_vec, n_dim

    def log_prob(theta: np.ndarray) -> float:
        sigma = theta[n_cont]
        if sigma <= sigma_min:
            return -np.inf
        lp = float(log_likelihood(jnp.asarray(theta), _delta_g, _delta_t))
        if not np.isfinite(lp):
            return -np.inf
        return lp

    return log_prob, n_dim


def run_mcmc(
    n_sys: int,
    *,
    model: str = "combined",
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    use_skewed: bool = False,
    n_walkers: int = 250,
    n_steps: int = 1500,
    n_burn: int = 300,
    seed: int = 42,
    progress: bool = True,
    vectorize: bool = False,
) -> tuple[np.ndarray, emcee.EnsembleSampler]:
    """Run emcee MCMC to infer contamination parameters.

    Walkers are initialized near zero contamination with sigma near
    ``std(delta_g_obs)``. The first ``n_burn`` steps are discarded as burn-in.

    The **combined** model (joint :math:`a_i, b_i` inference) is the default
    and the recommended choice; it is strictly superior to the additive-only
    model across all contamination regimes (see :doc:`/results_systematic_tests`).

    Parameters
    ----------
    n_sys : int
    model : str  ``'combined'`` (default), ``'additive'``, or ``'multiplicative'``
    delta_g_obs : (n_pix,) observed overdensity (keyword-only)
    delta_t : (n_sys, n_pix) template values (keyword-only)
    use_skewed : bool
    n_walkers : int  number of emcee ensemble walkers (≥ 2 × n_dim)
    n_steps : int   total MCMC steps (burn-in included)
    n_burn : int    steps to discard; ``flat_chain`` has ``n_walkers*(n_steps-n_burn)`` rows
    seed : int
    progress : bool  show tqdm progress bar
    vectorize : bool
        Evaluate all walkers in one batched JAX/GPU call per step (see
        :func:`make_log_prob`).  Requires emcee ≥ 3.0.  Recommended on GPU.

    Returns
    -------
    flat_chain : (n_walkers × (n_steps − n_burn), n_dim) posterior samples
    sampler : emcee.EnsembleSampler

    Performance
    -----------
    Wall time scales as O(n_walkers × n_steps × n_pix × n_sys).
    Typical runs: **minutes** for n_pix ~ 50_000, n_walkers=250, n_steps=1500.
    The JAX likelihood is JIT-compiled on the first walker evaluation (~227 ms),
    then executes at ~284 μs/eval (Gaussian) or ~592 μs/eval (skewed).

    Precision
    ---------
    Posterior median of the chain is taken as the point estimate via
    :func:`get_mle_params`. Posterior variance is estimated via
    :func:`get_param_variance_from_chain`.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import run_mcmc
    >>> rng = np.random.default_rng(0)
    >>> delta_g = rng.standard_normal(500) * 0.1
    >>> delta_t = rng.standard_normal((2, 500))
    >>> chain, sampler = run_mcmc(
    ...     n_sys=2, model="additive",
    ...     delta_g_obs=delta_g, delta_t=delta_t,
    ...     n_walkers=50, n_steps=200, n_burn=50,
    ...     progress=False,
    ... )
    >>> chain.shape  # (50 walkers * 150 steps, 3 params: a0, a1, sigma)
    (7500, 3)
    """
    log_prob, n_dim = make_log_prob(
        n_sys, model, delta_g_obs, delta_t, use_skewed, vectorize=vectorize
    )

    rng = np.random.default_rng(seed)
    n_cont = n_free_params(n_sys, model)
    sigma0 = float(np.std(delta_g_obs))

    # Initial walker positions: small perturbations around zero contamination
    p0 = np.zeros((n_walkers, n_dim))
    p0[:, :n_cont] = rng.normal(0.0, 1e-3, (n_walkers, n_cont))
    p0[:, n_cont] = sigma0 * (1.0 + rng.normal(0.0, 0.01, n_walkers))
    if use_skewed:
        p0[:, n_cont + 1] = rng.normal(0.0, 0.1, n_walkers)

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob, vectorize=vectorize)
    sampler.run_mcmc(p0, n_steps, progress=progress)

    flat_chain = sampler.get_chain(discard=n_burn, flat=True)
    return flat_chain, sampler


def get_mle_params(flat_chain: np.ndarray) -> np.ndarray:
    """Return the posterior median as a robust point estimate.

    Parameters
    ----------
    flat_chain : (n_samples, n_dim) posterior samples from :func:`run_mcmc`

    Returns
    -------
    (n_dim,) parameter vector (median of the flat chain)

    Performance
    -----------
    O(n_samples × n_dim) numpy median; negligible compared to MCMC runtime.

    Precision
    ---------
    Median is a robust estimator; for symmetric posteriors it converges to
    the true parameter at rate O(1/√n_samples).

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import get_mle_params
    >>> rng = np.random.default_rng(0)
    >>> flat_chain = rng.standard_normal((5000, 4))  # mock chain
    >>> theta_hat = get_mle_params(flat_chain)
    >>> theta_hat.shape
    (4,)
    """
    # Use posterior median as a robust point estimate
    return np.median(flat_chain, axis=0)


def get_param_variance_from_chain(
    flat_chain: np.ndarray,
    n_sys: int,
    model: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate variance of contamination parameters from MCMC chain.

    Returns only the diagonal (per-parameter variance). For the full covariance
    matrix needed to propagate uncertainty through a basis rotation, use
    :func:`get_param_covariance_from_chain`.

    Parameters
    ----------
    flat_chain : (n_samples, n_dim) posterior samples from :func:`run_mcmc`
    n_sys : int
    model : str  ``'additive'``, ``'multiplicative'``, or ``'combined'``

    Returns
    -------
    var_a : (n_sys,) posterior variance of additive parameters
    var_b : (n_sys,) posterior variance of multiplicative parameters
              (all zeros for ``'additive'`` model)

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import get_param_variance_from_chain
    >>> rng = np.random.default_rng(0)
    >>> n_sys, n_samples = 3, 2000
    >>> flat_chain = rng.standard_normal((n_samples, 2 * n_sys + 1)) * 0.01
    >>> var_a, var_b = get_param_variance_from_chain(flat_chain, n_sys, "combined")
    >>> var_a.shape
    (3,)
    """
    cov_a, cov_b = get_param_covariance_from_chain(flat_chain, n_sys, model)
    return np.diag(cov_a), np.diag(cov_b)


def get_param_covariance_from_chain(
    flat_chain: np.ndarray,
    n_sys: int,
    model: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the full covariance matrix of contamination parameters.

    Slices the flat chain directly rather than unpacking sample by sample.
    The returned (n_sys, n_sys) matrices are needed for correct variance
    propagation when back-transforming parameters from the PCA-rotated basis
    to the original template basis via ``R.T @ cov_rot @ R``.

    Parameters
    ----------
    flat_chain : (n_samples, n_dim) posterior samples from :func:`run_mcmc`
    n_sys : int
    model : str  ``'additive'``, ``'multiplicative'``, or ``'combined'``

    Returns
    -------
    cov_a : (n_sys, n_sys) posterior covariance of additive parameters
    cov_b : (n_sys, n_sys) posterior covariance of multiplicative parameters

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import get_param_covariance_from_chain
    >>> rng = np.random.default_rng(0)
    >>> n_sys, n_samples = 3, 2000
    >>> flat_chain = rng.standard_normal((n_samples, 2 * n_sys + 1)) * 0.01
    >>> cov_a, cov_b = get_param_covariance_from_chain(flat_chain, n_sys, "combined")
    >>> cov_a.shape
    (3, 3)
    """
    a_arr = flat_chain[:, :n_sys]
    if model == 'combined':
        b_arr = flat_chain[:, n_sys:2 * n_sys]
    elif model == 'multiplicative':
        b_arr = a_arr
    else:  # additive
        b_arr = np.zeros_like(a_arr)
    cov_a = np.cov(a_arr, rowvar=False) if n_sys > 1 else np.atleast_2d(np.var(a_arr))
    cov_b = np.cov(b_arr, rowvar=False) if n_sys > 1 else np.atleast_2d(np.var(b_arr))
    return cov_a, cov_b
