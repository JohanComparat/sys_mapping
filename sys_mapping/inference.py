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

import warnings

import numpy as np
import emcee
import jax
import jax.numpy as jnp
from jax import Array

from .likelihood import MIN_EFFICIENCY, make_log_likelihood
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
    delta_t : ``(n_sys, n_pix)`` template values
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
    >>> bool(np.isfinite(log_prob(theta0)))
    True
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
    delta_t : ``(n_sys, n_pix)`` template values (keyword-only)
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
    :func:`posterior_median_params`. Posterior variance is estimated via
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

    # Initial walker positions: spread over a range that covers typical
    # contamination amplitudes (0.02–0.10).  A scale of 0.05 ensures walkers
    # explore the full posterior from the start rather than having to diffuse
    # away from an overly tight ball near zero.
    p0 = np.zeros((n_walkers, n_dim))
    p0[:, :n_cont] = rng.normal(0.0, 0.05, (n_walkers, n_cont))
    p0[:, n_cont] = sigma0 * (1.0 + rng.normal(0.0, 0.1, n_walkers))
    if use_skewed:
        p0[:, n_cont + 1] = rng.normal(0.0, 0.1, n_walkers)

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob, vectorize=vectorize)
    sampler.run_mcmc(p0, n_steps, progress=progress)

    flat_chain = sampler.get_chain(discard=n_burn, flat=True)
    return flat_chain, sampler


class _AnalyticSampler:
    """Lightweight stand-in for :class:`emcee.EnsembleSampler`.

    Exposes the attributes downstream code reads from a real sampler.  The
    convergence diagnostics are ``None`` rather than perfect scores: these draws
    are exact and i.i.d., so there is nothing to converge and no quantity to
    measure.  Reporting ``rhat = 1.0`` and ``ess = n_samples`` was worse than
    reporting nothing, because a diagnostic that cannot fail reads, in a summary
    table beside real NUTS values, as evidence that it passed.
    """

    def __init__(self, n_samples: int, n_dim: int):
        self.acceptance_fraction = 1.0   # exact draws: not a measurement, but true
        self.num_divergences = 0         # no trajectory to diverge
        self.rhat = None                 # not applicable to i.i.d. draws
        self.ess = None                  # not applicable to i.i.d. draws
        self.n_samples = n_samples
        self.n_dim = n_dim


def run_additive_analytic(
    n_sys: int,
    *,
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    n_samples: int = 100_000,
    seed: int = 42,
) -> tuple[np.ndarray, _AnalyticSampler]:
    """Draw the exact analytic posterior for the additive (linear-Gaussian) model.

    The additive model ``δ_g_obs = Σ a_i t_i + ε``, ``ε ~ N(0, σ²)`` is ordinary
    linear regression, so under the *same* flat priors :func:`run_mcmc` uses (flat
    on ``a``, flat on ``σ > 0``) the posterior is Normal–Inverse-Gamma in closed
    form:

        σ² | data ~ Inv-Gamma(α = (n_pix − n_sys − 1)/2, β = RSS/2)
        a  | σ², data ~ N(a_hat, σ² (XᵀX)⁻¹)

    where ``X = delta_t.T``, ``a_hat`` is the OLS solution and ``RSS`` the residual
    sum of squares.  Sampling this hierarchy (σ² then a|σ²) yields the exact
    multivariate-t marginal posterior on ``a`` — with **no** Monte-Carlo
    autocorrelation — in milliseconds, replacing the ~80 s emcee ``MCMC-add`` run.

    Parameters
    ----------
    n_sys : int
    delta_g_obs : (n_pix,) observed overdensity (keyword-only)
    delta_t : ``(n_sys, n_pix)`` template values (keyword-only)
    n_samples : int  number of independent posterior draws (default 100_000)
    seed : int

    Returns
    -------
    flat_chain : ``(n_samples, n_sys + 1)`` posterior draws, layout ``[a_0..a_{n-1}, σ]``
    sampler : :class:`_AnalyticSampler`

    Notes
    -----
    The ``(n_pix − n_sys − 1)/2`` shape reproduces the flat-in-σ prior that emcee
    implicitly samples (uniform on ``σ``, not ``log σ``); for ``n_pix ≫ n_sys`` the
    distinction from a Jeffreys prior is negligible.  Matches an emcee ``MCMC-add``
    chain's posterior mean and covariance to Monte-Carlo error, but is exact.
    """
    delta_g = jnp.asarray(delta_g_obs, dtype=jnp.float64)
    n_pix = int(delta_g.shape[0])
    key_sigma, key_a = jax.random.split(jax.random.PRNGKey(seed))

    if n_sys == 0:
        rss = jnp.dot(delta_g, delta_g)
        alpha = max((n_pix - 1) / 2.0, 1e-3)
        g = jax.random.gamma(key_sigma, alpha, shape=(n_samples,))
        sigma = np.asarray(jnp.sqrt((rss / 2.0) / g))
        return sigma[:, None], _AnalyticSampler(n_samples, 1)

    X = jnp.asarray(delta_t, dtype=jnp.float64).T              # (n_pix, n_sys)
    XtX = X.T @ X                                              # (n_sys, n_sys)
    Xty = X.T @ delta_g                                        # (n_sys,)
    # Cholesky of the (small) normal-equations matrix: XᵀX = L Lᵀ
    L = jax.scipy.linalg.cholesky(XtX, lower=True)
    a_hat = jax.scipy.linalg.cho_solve((L, True), Xty)         # OLS solution
    resid = delta_g - X @ a_hat
    rss = jnp.dot(resid, resid)

    # σ² ~ Inv-Gamma(α, β):  σ² = β / Gamma(α, 1)
    alpha = max((n_pix - n_sys - 1) / 2.0, 1e-3)
    beta = rss / 2.0
    g = jax.random.gamma(key_sigma, alpha, shape=(n_samples,))
    sigma = jnp.sqrt(beta / g)                                 # (n_samples,)

    # a | σ² ~ N(a_hat, σ² (XᵀX)⁻¹).  With XᵀX = L Lᵀ, a draw with covariance
    # (XᵀX)⁻¹ is L⁻ᵀ z for z ~ N(0, I): Cov(L⁻ᵀ z) = L⁻ᵀ L⁻¹ = (XᵀX)⁻¹.
    z = jax.random.normal(key_a, shape=(n_samples, n_sys))
    y = jax.scipy.linalg.solve_triangular(L.T, z.T, lower=False).T   # L⁻ᵀ z
    a_samples = a_hat[None, :] + sigma[:, None] * y

    flat_chain = jnp.concatenate([a_samples, sigma[:, None]], axis=1)
    return np.asarray(flat_chain), _AnalyticSampler(n_samples, n_sys + 1)


def posterior_median_params(flat_chain: np.ndarray) -> np.ndarray:
    """Per-parameter posterior median of a chain.

    A robust point estimate, and the right one for reporting an amplitude.  It is
    **not** a maximum-likelihood point: the median is taken coordinate by
    coordinate, so on a correlated posterior the result need not lie near the
    mode, and for two *nested* models the two medians are not guaranteed to
    satisfy :math:`\\ell_{\\rm alt} \\ge \\ell_{\\rm null}`.  Anything that
    differences two log-likelihoods --- a likelihood ratio above all --- must use
    :func:`refine_to_mle` instead.

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

    See Also
    --------
    refine_to_mle : maximises the likelihood, for statistics that need a maximum.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import posterior_median_params
    >>> rng = np.random.default_rng(0)
    >>> flat_chain = rng.standard_normal((5000, 4))  # mock chain
    >>> theta_hat = posterior_median_params(flat_chain)
    >>> theta_hat.shape
    (4,)
    """
    # Use posterior median as a robust point estimate
    return np.median(flat_chain, axis=0)


_INFEASIBLE_VALUE = 1e30
_REFINE_CACHE: dict = {}


def _neg_log_lik_value_and_grad(n_sys: int, model: str, use_skewed: bool, precision=None):
    """Compiled ``(u, delta_g, delta_t) -> (-log L, gradient)`` with ``sigma = exp(u_sigma)``.

    The data are arguments, so refinements of same-shaped fields (a mock null)
    share one compilation.  Not cached when a precision operator is given.
    """
    key = (int(n_sys), str(model), bool(use_skewed))
    if precision is None and key in _REFINE_CACHE:
        return _REFINE_CACHE[key]
    log_lik = make_log_likelihood(n_sys, model, use_skewed, precision=precision)
    i_sigma = n_free_params(n_sys, model)
    i_b = {"combined": n_sys, "multiplicative": 0}.get(model)

    def neg(u, dg, dt):
        if i_b is None:
            return -log_lik(u.at[i_sigma].set(jnp.exp(u[i_sigma])), dg, dt)
        # The likelihood is unbounded above as an efficiency 1 + b.t vanishes; outside the
        # region where every efficiency is at least MIN_EFFICIENCY the objective is a large
        # constant.  SciPy's line search needs finite values to backtrack from.
        feasible = jnp.min(1.0 + u[i_b:i_b + n_sys] @ dt) >= MIN_EFFICIENCY
        u_safe = jnp.where(feasible, u, u.at[i_b:i_b + n_sys].set(0.0))
        value = -log_lik(u_safe.at[i_sigma].set(jnp.exp(u_safe[i_sigma])), dg, dt)
        return jnp.where(feasible, value, _INFEASIBLE_VALUE)

    fn = jax.jit(jax.value_and_grad(neg))
    if precision is None:
        _REFINE_CACHE[key] = fn
    return fn


def refine_to_mle(
    theta0: np.ndarray,
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    *,
    model: str = "combined",
    use_skewed: bool = False,
    precision=None,
    max_iter: int = 500,
) -> np.ndarray:
    """Maximise the likelihood, starting from ``theta0``.

    A likelihood ratio differences two log-likelihoods, so it is only guaranteed
    non-negative between *nested* models when both are evaluated at their maxima.
    Evaluated at posterior medians instead, the guarantee is lost and the
    statistic goes negative: on the LS10 grid at :math:`NSIDE 64` four of eight
    cells return a majority of negative null draws, with minima of order
    :math:`-10^{6}`.  This function is what makes that statistic a likelihood
    ratio again.

    The optimiser works on :math:`\\log\\sigma` rather than :math:`\\sigma`, which
    keeps the scale positive without a bound and conditions the problem; the
    returned vector is in the ordinary :func:`~sys_mapping.pack_params` layout.

    Parameters
    ----------
    theta0:
        Starting point, in the packed layout for ``model``.  A posterior median
        is a good one.
    delta_g_obs:
        Observed overdensity at the fitted pixels, shape ``(n_pix,)``.
    delta_t:
        Templates, shape ``(n_sys, n_pix)``.
    model:
        ``"additive"``, ``"multiplicative"`` or ``"combined"``.
    use_skewed:
        Whether ``theta0`` carries a trailing skewness parameter.
    precision:
        Optional pixel correlation operator, passed through to
        :func:`~sys_mapping.likelihood.make_log_likelihood`.
    max_iter:
        Cap on L-BFGS-B iterations.

    Returns
    -------
    ``(n_dim,)`` parameter vector at the maximum.  Falls back to ``theta0`` with a
    warning if the optimiser does not improve on it, so a caller never silently
    receives a worse point than it supplied.

    Notes
    -----
    This maximises the *likelihood*, not the NUTS log-density of
    :func:`~sys_mapping.nuts.build_logdensity` --- that adds the
    :math:`+\\log\\sigma` Jacobian of its own reparametrisation and any priors, so
    its maximiser is a MAP in the transformed variable rather than an MLE.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import refine_to_mle, pack_params
    >>> rng = np.random.default_rng(0)
    >>> n_pix, n_sys = 4000, 2
    >>> delta_t = rng.standard_normal((n_sys, n_pix))
    >>> a = np.array([0.05, -0.03])
    >>> dg = a @ delta_t + rng.standard_normal(n_pix) * 0.1
    >>> theta0 = pack_params(np.zeros(n_sys), None, 0.5, model="additive")
    >>> theta = refine_to_mle(theta0, dg, delta_t, model="additive")
    >>> bool(np.allclose(theta[:n_sys], a, atol=0.01))
    True
    """
    from scipy.optimize import minimize

    theta0 = np.asarray(theta0, dtype=float)
    n_sys = int(np.asarray(delta_t).shape[0])
    log_lik = make_log_likelihood(n_sys, model, use_skewed, precision=precision)

    _dg = jnp.asarray(delta_g_obs)
    _dt = jnp.asarray(delta_t)
    # sigma sits immediately after the amplitude block; gamma, when present, last.
    i_sigma = n_free_params(n_sys, model)

    def _to_u(theta: np.ndarray) -> np.ndarray:
        u = np.array(theta, dtype=float)
        u[i_sigma] = np.log(max(float(theta[i_sigma]), 1e-12))
        return u

    def _to_theta(u):
        return u.at[i_sigma].set(jnp.exp(u[i_sigma]))

    value_and_grad = _neg_log_lik_value_and_grad(n_sys, model, use_skewed, precision)

    def _fun(u):
        v, g = value_and_grad(jnp.asarray(u), _dg, _dt)
        return float(v), np.asarray(g, dtype=float)

    # Two starts, because one is not reliably enough.  A posterior median of a
    # near-degenerate 23-parameter posterior can sit in a region where L-BFGS-B
    # stalls, and a stalled refinement silently leaves the statistic median-based
    # -- which is the whole defect this function exists to remove.  The second
    # start is analytic: for the additive model OLS *is* the MLE, and for the
    # combined model (a = OLS, b = 0) is on the ridge by construction.
    starts = [np.asarray(theta0, dtype=float)]
    gamma0 = float(theta0[-1]) if use_skewed else None
    try:
        if model == "multiplicative":
            # No additive term to fit: b = 0 is the uncontaminated field itself.
            sig = max(float(np.std(np.asarray(delta_g_obs))), 1e-9)
            analytic = pack_params(None, np.zeros(n_sys), sig, gamma=gamma0, model=model)
        else:
            a_ols, *_ = np.linalg.lstsq(np.asarray(delta_t).T,
                                        np.asarray(delta_g_obs), rcond=None)
            resid = np.asarray(delta_g_obs) - a_ols @ np.asarray(delta_t)
            sig = max(float(np.std(resid)), 1e-9)
            analytic = pack_params(
                a_ols, np.zeros(n_sys) if model == "combined" else None, sig,
                gamma=gamma0, model=model,
            )
        if analytic.shape == theta0.shape:
            starts.append(np.asarray(analytic, dtype=float))
    except np.linalg.LinAlgError:
        pass

    if use_skewed:
        # gamma = 0 is a stationary point of the skew-normal log-likelihood, so a start
        # that sits on it cannot move off it; offer starts on both sides of it.
        extra = []
        for start in starts:
            if abs(float(start[-1])) < 1e-12:
                for g0 in (1.5, -1.5):
                    alt = np.array(start, dtype=float)
                    alt[-1] = g0
                    extra.append(alt)
        starts = starts + extra

    ll0 = float(log_lik(jnp.asarray(theta0), _dg, _dt))
    best_theta, best_ll, messages = theta0, ll0, []
    i_b = {"combined": n_sys, "multiplicative": 0}.get(model)

    def _feasible(theta):
        if i_b is None:
            return True
        return float(np.min(1.0 + np.asarray(theta[i_b:i_b + n_sys]) @ np.asarray(delta_t))) \
            >= MIN_EFFICIENCY

    # A model carrying b has an unbounded likelihood as an efficiency vanishes; its search
    # runs in JAX, whose line search backtracks from the +inf outside the region where every
    # efficiency is at least MIN_EFFICIENCY.  SciPy's cannot.
    floored = None
    if i_b is not None and precision is None:
        from .model_selection import _refine_with_floor
        floored = _refine_with_floor(n_sys, model, use_skewed, int(max_iter))

    for start in starts:
        if not _feasible(start):
            continue
        if floored is not None:
            u_opt, _ = floored(jnp.asarray(_to_u(start)), _dg, _dt)
            cand = np.asarray(_to_theta(u_opt), dtype=float)
            messages.append("jax L-BFGS with the efficiency floor")
        else:
            res = minimize(_fun, _to_u(start), jac=True, method="L-BFGS-B",
                           options={"maxiter": int(max_iter)})
            cand = np.asarray(_to_theta(jnp.asarray(res.x)), dtype=float)
            messages.append(res.message)
        if not _feasible(cand):
            messages.append("left the efficiency floor")
            continue
        ll = float(log_lik(jnp.asarray(cand), _dg, _dt))
        if np.isfinite(ll) and ll > best_ll:
            best_theta, best_ll = cand, ll

    if best_ll <= ll0:
        warnings.warn(
            f"refine_to_mle did not improve the log-likelihood from any of "
            f"{len(starts)} starts ({ll0:.6g}); returning the starting point. "
            f"Optimiser messages: {messages}",
            RuntimeWarning,
            stacklevel=2,
        )
        return theta0
    return best_theta


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
    var_a : ``(n_sys,)`` posterior variance of additive parameters
    var_b : ``(n_sys,)`` posterior variance of multiplicative parameters
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
    cov_a : ``(n_sys, n_sys)`` posterior covariance of additive parameters
    cov_b : ``(n_sys, n_sys)`` posterior covariance of multiplicative parameters

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
    if model == 'combined':
        a_arr = flat_chain[:, :n_sys]
        b_arr = flat_chain[:, n_sys:2 * n_sys]
    elif model == 'multiplicative':
        # The free parameters are the multiplicative coefficients; there is no
        # additive component in this model.
        b_arr = flat_chain[:, :n_sys]
        a_arr = np.zeros_like(b_arr)
    else:  # additive
        a_arr = flat_chain[:, :n_sys]
        b_arr = np.zeros_like(a_arr)
    # ddof=1 in both branches: np.cov defaults to ddof=1, np.var to ddof=0, so
    # taking the defaults made the n_sys == 1 variance smaller by (N-1)/N than the
    # same quantity for n_sys > 1.  axis=0 is explicit because a bare np.var on a
    # (n_samples, 1) slice flattens, which is only harmless while the slice is one
    # column wide.
    cov_a = (np.cov(a_arr, rowvar=False) if n_sys > 1
             else np.atleast_2d(np.var(a_arr, axis=0, ddof=1)))
    cov_b = (np.cov(b_arr, rowvar=False) if n_sys > 1
             else np.atleast_2d(np.var(b_arr, axis=0, ddof=1)))
    return cov_a, cov_b
