"""Likelihood ratio test for model selection (Eq. 19, Berlfein et al. 2024).

The likelihood ratio statistic:
    λ_LR = 2 [ln L(Θ̂) - ln L(Θ̂_0)]

is asymptotically χ²(r) distributed under the null hypothesis, where
r = dim(Θ̂) - dim(Θ̂_0) is the number of additional free parameters.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .likelihood import MIN_EFFICIENCY, make_log_likelihood
from .contamination import n_free_params, pack_params
from .diagnostics import snr_template_ranking
import jax.numpy as jnp


@dataclass
class LikelihoodRatioResult:
    lambda_lr: float
    n_dof: int
    p_value: float
    reject_null: bool
    null_model: str
    alt_model: str
    calibration: str = "chi2"


def likelihood_ratio_test(
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    theta_null: np.ndarray,
    theta_alt: np.ndarray,
    null_model: str,
    alt_model: str,
    use_skewed: bool = False,
    significance: float = 0.05,
    null_lambda: np.ndarray | None = None,
) -> LikelihoodRatioResult:
    """Perform a likelihood ratio test comparing null vs. alternative model (Eq. 19).

    Computes ``λ_LR = 2 [ln L(θ̂_alt) − ln L(θ̂_null)]``, which under the null
    hypothesis is asymptotically ``χ²(r)`` where
    ``r = dim(alt) − dim(null)`` is the extra degrees of freedom.

    .. warning::
       The asymptotic ``χ²(r)`` null (Wilks) assumes the log-likelihood is built from
       **independent** observations.  The pixel likelihood is not — it uses the ``σ²I`` model on a
       spatially-**correlated** clustering field — so under the null ``λ_LR`` is *inflated* relative
       to ``χ²(r)`` and the default p-value is **too small** (overconfident detection).  Pass
       ``null_lambda`` (an ensemble of ``λ_LR`` values from *uncontaminated* mocks fit the same way)
       to read a **mock-calibrated** p-value from the empirical null tail instead.

    .. note::
       Each call creates new JIT-compiled likelihood functions internally.
       For repeated calls with the same data, precompute the log-likelihoods
       using :func:`~likelihood.make_log_likelihood` to avoid recompilation.

    Parameters
    ----------
    delta_g_obs : (n_pix,) observed overdensity
    delta_t : ``(n_sys, n_pix)`` template values
    theta_null : flat parameter vector for the null model, at its maximum
        (see :func:`~sys_mapping.inference.refine_to_mle`)
    theta_alt  : flat parameter vector for the alternative model
    null_model : str  ``'additive'`` or ``'multiplicative'`` (must be nested in alt_model)
    alt_model  : str  ``'combined'`` (must have more free parameters than null_model)
    use_skewed : bool
    significance : float  p-value threshold for rejecting the null; default 0.05
    null_lambda : (n_mock,) array or None
        Empirical null distribution of ``λ_LR`` — the statistic evaluated on an ensemble of
        *uncontaminated* mock reconstructions fit with the same two models.  When given, the p-value
        is the Monte-Carlo tail ``(1 + #{λ_null ≥ λ_LR}) / (1 + n_mock)`` (Davison & Hinkley) and
        ``calibration='mock'``; when ``None`` (default) the asymptotic ``χ²(r)`` is used
        (``calibration='chi2'``).

    Returns
    -------
    LikelihoodRatioResult
        ``lambda_lr`` : test statistic (>= 0 only when both theta are at their MLEs;
        a warning is issued when it is not)
        ``n_dof``     : degrees of freedom ``r = n_free_params(alt) − n_free_params(null)``
        ``p_value``   : null-tail probability (``χ²`` or mock-calibrated per ``null_lambda``)
        ``reject_null``: True when ``p_value < significance``
        ``null_model``, ``alt_model``: model names
        ``calibration``: ``'chi2'`` or ``'mock'``

    Performance
    -----------
    Measured on CPU (n_pix=10_000, n_sys=3): **~481 ms/call** (includes JIT
    compilation of two likelihood functions, ~227 ms each on first call).
    In a pipeline where likelihoods are already compiled, evaluation is ~1 ms.

    Precision
    ---------
    p-value is in [0, 1] by construction.
    ``lambda_lr = 0`` when ``theta_alt`` uses b=0 (equivalent to the null).
    For large n_pix, ``lambda_lr`` is χ²-distributed under the null asymptotically.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import likelihood_ratio_test, pack_params
    >>> rng = np.random.default_rng(0)
    >>> n_pix, n_sys = 5000, 3
    >>> delta_g = rng.standard_normal(n_pix) * 0.1
    >>> delta_t = rng.standard_normal((n_sys, n_pix))
    >>> a_hat = np.zeros(n_sys)
    >>> b_hat = np.zeros(n_sys)
    >>> sigma  = float(delta_g.std())
    >>> theta_add  = pack_params(a_hat, None,  sigma, model="additive")
    >>> theta_comb = pack_params(a_hat, b_hat, sigma, model="combined")
    >>> result = likelihood_ratio_test(delta_g, delta_t,
    ...                                theta_add, theta_comb,
    ...                                "additive", "combined")
    >>> result.n_dof   # combined has n_sys extra b parameters
    3
    >>> 0.0 <= result.p_value <= 1.0
    True
    """
    n_sys = delta_t.shape[0]
    _delta_g = jnp.asarray(delta_g_obs)
    _delta_t = jnp.asarray(delta_t)

    log_L_null = make_log_likelihood(n_sys, null_model, use_skewed)
    log_L_alt = make_log_likelihood(n_sys, alt_model, use_skewed)

    ll_null = float(log_L_null(jnp.asarray(theta_null), _delta_g, _delta_t))
    ll_alt = float(log_L_alt(jnp.asarray(theta_alt), _delta_g, _delta_t))

    lambda_lr = 2.0 * (ll_alt - ll_null)

    # Between nested models a likelihood ratio cannot be negative -- but only when
    # both points are maxima.  Evaluated at posterior medians it can be, and has
    # been by six orders of magnitude, which is a statement about the estimator and
    # not about the data.  Warn rather than clip: clipping would hide it, and the
    # magnitude is meaningless either way.  Refine with
    # :func:`~sys_mapping.inference.refine_to_mle` before calling this.
    if lambda_lr < 0.0:
        warnings.warn(
            f"lambda_LR = {lambda_lr:.6g} < 0 between nested models "
            f"('{null_model}' in '{alt_model}'). The supplied theta are not both "
            "at their maxima; pass points from refine_to_mle. The mock-calibrated "
            "p-value stays valid (data and null share the estimator) but the "
            "magnitude of lambda_LR does not.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Degrees of freedom = difference in number of free parameters
    n_dof_null = n_free_params(n_sys, null_model) + 1 + (1 if use_skewed else 0)
    n_dof_alt = n_free_params(n_sys, alt_model) + 1 + (1 if use_skewed else 0)
    r = n_dof_alt - n_dof_null

    if r <= 0:
        raise ValueError(
            f"Alternative model '{alt_model}' must have more free params than null '{null_model}'. "
            f"Got r={r}."
        )

    if null_lambda is not None:
        null = np.asarray(null_lambda, dtype=float)
        null = null[np.isfinite(null)]
        if null.size == 0:
            raise ValueError("null_lambda has no finite values")
        # Monte-Carlo p-value with the +1 correction (Davison & Hinkley 1997): unbiased and never 0.
        p_value = float((1.0 + np.sum(null >= lambda_lr)) / (1.0 + null.size))
        calibration = "mock"
    else:
        p_value = float(chi2.sf(lambda_lr, df=r))
        calibration = "chi2"
    reject_null = p_value < significance

    return LikelihoodRatioResult(
        lambda_lr=lambda_lr,
        n_dof=r,
        p_value=p_value,
        reject_null=reject_null,
        null_model=null_model,
        alt_model=alt_model,
        calibration=calibration,
    )


_MAXIMA_CACHE: dict = {}
_MAXIMA_GRAD_STOP = 1e-9  # largest gradient component at which an L-BFGS run stops
_MAXIMA_RESTARTS = 3      # L-BFGS runs per maximum, each started from the best point so far
_MAXIMA_GAMMA_STARTS = (1.5, -1.5)  # skewness starts; gamma = 0 is a stationary point


def _lbfgs_maximise(neg, u0, n_iter):
    """Minimise ``neg`` from ``u0`` with L-BFGS, robust to ``+inf`` outside a feasible region.

    Returns ``(u, max|grad|)``.  Each run keeps its best finite point and stops at
    convergence, at a non-finite value or gradient, or after ``n_iter`` iterations; it is
    restarted from its best point ``_MAXIMA_RESTARTS`` times.  Traceable: runs under
    ``jit`` and ``vmap``.
    """
    import jax
    import optax
    opt = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(neg)

    def step(carry):
        u, state, u_best, f_best, _ = carry
        value, grad = value_and_grad(u, state=state)
        # Keep the best finite point: near a pixel where 1 + b.t = 0 the gradient
        # explodes and a step can throw the iterate onto the flat ridge |b| -> inf.
        # The value the line search leaves in the state is not always the value at u
        # after a failed search, so the bookkeeping evaluates u itself.
        value_here = neg(u)
        ok = (jnp.isfinite(value) & jnp.isfinite(value_here)
              & jnp.all(jnp.isfinite(grad)))
        improved = ok & (value_here < f_best)
        u_best = jnp.where(improved, u, u_best)
        f_best = jnp.where(improved, value_here, f_best)
        updates, state = opt.update(grad, state, u, value=value, grad=grad, value_fn=neg)
        return optax.apply_updates(u, updates), state, u_best, f_best, ok

    def running(carry):
        # Stop at convergence (past it the line search spends its full step budget on
        # every iteration without moving) or once the value or gradient is not finite.
        _, state, _, _, ok = carry
        count = optax.tree_utils.tree_get(state, "count")
        grad = optax.tree_utils.tree_get(state, "grad")
        return (count == 0) | (ok & (count < n_iter)
                               & (jnp.max(jnp.abs(grad)) > _MAXIMA_GRAD_STOP))

    def run_from(u_start, _):
        f0 = neg(u_start)
        carry = (u_start, opt.init(u_start), u_start, f0, jnp.asarray(True))
        u, _, u_best, f_best, _ = jax.lax.while_loop(running, step, carry)
        f_end = neg(u)
        u_best = jnp.where(jnp.isfinite(f_end) & (f_end <= f_best), u, u_best)
        return u_best, None

    # Restart from the best point with a fresh curvature memory; a run that converged
    # stops again after one iteration.
    u, _ = jax.lax.scan(run_from, u0, None, length=_MAXIMA_RESTARTS)
    return u, jnp.max(jnp.abs(jax.grad(neg)(u)))


def _batched_maxima(n_sys: int, use_skewed: bool, n_iter: int):
    """Compiled ``(G, T) -> (lambda, theta_add, theta_comb, max|grad|)`` over rows of ``G``."""
    key = (int(n_sys), bool(use_skewed), int(n_iter))
    if key in _MAXIMA_CACHE:
        return _MAXIMA_CACHE[key]
    import jax
    import optax

    ll_add = make_log_likelihood(n_sys, "additive", use_skewed)
    ll_comb = make_log_likelihood(n_sys, "combined", use_skewed)
    i_add, i_comb = n_sys, 2 * n_sys

    def one(g, T):
        gram = T @ T.T
        a = jnp.linalg.solve(gram, T @ g)
        log_sig = 0.5 * jnp.log(jnp.mean((g - a @ T) ** 2))
        gamma0 = jnp.zeros(1) if use_skewed else jnp.zeros(0)

        def neg_add(u):
            return -ll_add(u.at[i_add].set(jnp.exp(u[i_add])), g, T)

        def neg_comb(u):
            # The likelihood is unbounded above as a pixel's efficiency 1 + b.t goes to
            # zero, so the maximum is sought where every efficiency is at least
            # MIN_EFFICIENCY, the region that holds b = 0; the line search backtracks
            # from +inf.
            feasible = jnp.min(1.0 + u[n_sys:2 * n_sys] @ T) >= MIN_EFFICIENCY
            u_safe = jnp.where(feasible, u, u.at[n_sys:2 * n_sys].set(0.0))
            value = -ll_comb(u_safe.at[i_comb].set(jnp.exp(u_safe[i_comb])), g, T)
            return jnp.where(feasible, value, jnp.inf)

        u_add = jnp.concatenate([a, log_sig[None], gamma0])
        g_add = jnp.asarray(0.0)
        if use_skewed:
            # gamma = 0 is a stationary point of the skew-normal log-likelihood (its
            # information for the skewness vanishes there), so an optimiser started on it
            # never leaves it and the fit stays Gaussian.  Start off it, both ways.
            cands = [_lbfgs_maximise(neg_add, u_add.at[-1].set(g0), n_iter)
                     for g0 in _MAXIMA_GAMMA_STARTS]
            values = jnp.stack([neg_add(u) for u, _ in cands])
            pick = jnp.argmin(values)
            u_add = jnp.stack([u for u, _ in cands])[pick]
            g_add = jnp.stack([g for _, g in cands])[pick]
        u_comb0 = jnp.concatenate([u_add[:n_sys], jnp.zeros(n_sys), u_add[n_sys:]])
        u_comb, g_comb = _lbfgs_maximise(neg_comb, u_comb0, n_iter)
        # The start is feasible and finite, so a maximum that is neither is replaced by it;
        # the gradient there is large, and the field is flagged as not converged.
        good_comb = jnp.isfinite(neg_comb(u_comb))
        u_comb = jnp.where(good_comb, u_comb, u_comb0)
        g_comb = jnp.where(good_comb, g_comb, jnp.inf)
        lam = 2.0 * (neg_add(u_add) - neg_comb(u_comb))
        th_add = u_add.at[i_add].set(jnp.exp(u_add[i_add]))
        th_comb = u_comb.at[i_comb].set(jnp.exp(u_comb[i_comb]))
        return lam, th_add, th_comb, jnp.maximum(g_add, g_comb)

    fn = jax.jit(jax.vmap(one, in_axes=(0, None)))
    _MAXIMA_CACHE[key] = fn
    return fn


_REFINE_B_CACHE: dict = {}


def _refine_with_floor(n_sys: int, model: str, use_skewed: bool, n_iter: int = 500):
    """Compiled ``(u0, g, T) -> (u, max|grad|)``: the maximum of a model carrying ``b``.

    ``u`` has ``sigma`` on a log scale; the search stays where every efficiency ``1 + b.t``
    is at least ``MIN_EFFICIENCY``, where the likelihood is bounded.
    """
    key = (int(n_sys), str(model), bool(use_skewed), int(n_iter))
    if key in _REFINE_B_CACHE:
        return _REFINE_B_CACHE[key]
    import jax
    ll = make_log_likelihood(n_sys, model, use_skewed)
    i_sigma = n_free_params(n_sys, model)
    i_b = n_sys if model == "combined" else 0

    def run(u0, g, T):
        def neg(u):
            feasible = jnp.min(1.0 + u[i_b:i_b + n_sys] @ T) >= MIN_EFFICIENCY
            u_safe = jnp.where(feasible, u, u.at[i_b:i_b + n_sys].set(0.0))
            value = -ll(u_safe.at[i_sigma].set(jnp.exp(u_safe[i_sigma])), g, T)
            return jnp.where(feasible, value, jnp.inf)
        return _lbfgs_maximise(neg, u0, n_iter)

    fn = jax.jit(run)
    _REFINE_B_CACHE[key] = fn
    return fn


def lrt_from_maxima(
    delta_g: np.ndarray,
    delta_t: np.ndarray,
    *,
    use_skewed: bool = False,
    n_iter: int = 300,
    batch_size: int = 64,
    grad_tol: float = 1e-3,
) -> dict[str, np.ndarray]:
    r"""Additive-versus-combined :math:`\lambda_{\rm LR}` of many fields at once, from their maxima.

    For each row of ``delta_g`` the additive maximum is the least-squares solution (found by
    L-BFGS as well when ``use_skewed``), and the combined maximum is found by L-BFGS started
    from it with ``b = 0``, a point on the combined model's ridge; each L-BFGS run stops when
    its largest gradient component falls below 1e-9, or after ``n_iter`` iterations.  All rows are optimised
    together under ``jax.vmap``, in batches of ``batch_size``, with one compilation per
    ``(n_sys, use_skewed, n_iter)``.

    This is the null :func:`lrt_null_distribution` builds when ``fit_theta`` returns maxima,
    without fitting a posterior per mock: on 7 040 pixels with 11 templates, 8 mocks take
    0.15 s after compilation, against about 50 s for NUTS fits refined to their maxima,
    with :math:`\lambda_{\rm LR}` equal to a relative 8e-7.

    Parameters
    ----------
    delta_g : ``(n_field, n_pix)`` or ``(n_pix,)``
        Overdensity fields on the fit pixels, e.g. uncontaminated mocks.
    delta_t : ``(n_sys, n_pix)``
        Templates on the same pixels.
    use_skewed : bool
        Skew-normal likelihood for both models.
    n_iter : int
        L-BFGS iterations per optimisation.
    batch_size : int
        Fields optimised together; bounds the memory of the batched optimisation.
    grad_tol : float
        A field counts as converged when the largest gradient component of the negative
        log-likelihood (in the parameters with :math:`\sigma` on a log scale) is below
        this at the returned points.

    Returns
    -------
    dict with ``"lambda"`` ``(n_field,)``; ``"theta_null"`` and ``"theta_alt"``, the maxima in
    the packed layout of :func:`~sys_mapping.contamination.pack_params`; ``"max_grad"``
    ``(n_field,)``; ``"min_efficiency"``, the smallest ``1 + b.t`` of each combined maximum;
    ``"at_efficiency_floor"``, whether that maximum sits on ``MIN_EFFICIENCY``; and
    ``"converged"`` ``(n_field,)`` booleans, true for a vanishing gradient or a maximum on
    the floor.  A warning names the fields that did not converge.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping.model_selection import lrt_from_maxima
    >>> rng = np.random.default_rng(0)
    >>> T = rng.standard_normal((3, 2000))
    >>> mocks = rng.standard_normal((4, 2000)) * 0.2
    >>> out = lrt_from_maxima(mocks, T)
    >>> out["lambda"].shape, bool(np.all(out["lambda"] >= -1e-6)), bool(out["converged"].all())
    ((4,), True, True)
    """
    G = np.atleast_2d(np.asarray(delta_g, dtype=float))
    T = np.asarray(delta_t, dtype=float)
    if T.ndim != 2 or G.shape[1] != T.shape[1]:
        raise ValueError(f"delta_g must be (n_field, n_pix) matching delta_t (n_sys, n_pix); "
                         f"got {G.shape} and {T.shape}")
    n_sys = T.shape[0]
    # Optimise in the whitened basis T_w = W T, whose second moment is the identity:
    # a . t = a_w . t_w with a = W^T a_w (the same for b), so the likelihood, the
    # efficiency 1 + b.t and lambda are unchanged, while L-BFGS no longer has to cross
    # the ill-conditioned valleys of a collinear basis.
    evals, evecs = np.linalg.eigh(T @ T.T / T.shape[1])
    W = (evecs / np.sqrt(np.clip(evals, 1e-300, None))).T
    fn = _batched_maxima(n_sys, use_skewed, n_iter)
    Tj = jnp.asarray(W @ T)
    parts = [fn(jnp.asarray(G[i:i + batch_size]), Tj) for i in range(0, len(G), int(batch_size))]
    lam, th0, th1, gmax = (np.concatenate([np.asarray(p[j]) for p in parts]) for j in range(4))
    th0[:, :n_sys] = th0[:, :n_sys] @ W
    th1[:, :n_sys] = th1[:, :n_sys] @ W
    th1[:, n_sys:2 * n_sys] = th1[:, n_sys:2 * n_sys] @ W
    # A maximum on the efficiency floor is a constrained maximum: its unconstrained
    # gradient need not vanish there.
    min_eff = (1.0 + th1[:, n_sys:2 * n_sys] @ T).min(axis=1)
    at_floor = min_eff <= MIN_EFFICIENCY * (1.0 + 1e-3)
    converged = (gmax < grad_tol) | at_floor
    if not converged.all():
        warnings.warn(
            f"lrt_from_maxima: {int((~converged).sum())} of {len(G)} fields did not converge "
            f"(largest gradient {gmax.max():.3g} >= {grad_tol}); raise n_iter.",
            RuntimeWarning, stacklevel=2)
    return {"lambda": lam, "theta_null": th0, "theta_alt": th1,
            "max_grad": gmax, "converged": converged, "at_efficiency_floor": at_floor,
            "min_efficiency": min_eff}


def lrt_null_distribution(
    mock_delta_g: np.ndarray,
    delta_t: np.ndarray,
    fit_theta,
    null_model: str = "additive",
    alt_model: str = "combined",
    use_skewed: bool = False,
) -> np.ndarray:
    """Empirical null distribution of :math:`\\lambda_{\\rm LR}` from *uncontaminated* mock fields.

    Feed the resulting array to :func:`likelihood_ratio_test` as ``null_lambda`` to obtain a
    **mock-calibrated** p-value instead of the (overconfident on a correlated field) Wilks
    :math:`\\chi^2`. Each mock is fit with the same two models the data uses, so the null captures
    the correlated-field inflation of :math:`\\lambda_{\\rm LR}`.

    Parameters
    ----------
    mock_delta_g : (n_pix, n_mock)
        Column-stacked *uncontaminated* overdensity reconstructions on the fit pixels.
    delta_t : ``(n_sys, n_pix)``
        Templates on the fit pixels (same basis as the data fit; rotate first if the data used a
        rotated basis).
    fit_theta : callable(model, delta_g, delta_t) -> theta
        Returns the flat parameter vector at the likelihood maximum for ``model`` on one mock — e.g. a
        wrapper around :func:`~sys_mapping.regression.run_decontamination` (matching how the data
        were fit) or an OLS surrogate.
    null_model, alt_model, use_skewed :
        Passed through to :func:`likelihood_ratio_test`.

    Returns
    -------
    (n_mock,) array of :math:`\\lambda_{\\rm LR}` values under the null.
    """
    mock_delta_g = np.asarray(mock_delta_g, dtype=float)
    if mock_delta_g.ndim != 2:
        raise ValueError(f"mock_delta_g must be 2-D (n_pix, n_mock); got {mock_delta_g.shape}")
    lam = np.empty(mock_delta_g.shape[1])
    for k in range(mock_delta_g.shape[1]):
        dg = mock_delta_g[:, k]
        theta_null = fit_theta(null_model, dg, delta_t)
        theta_alt = fit_theta(alt_model, dg, delta_t)
        res = likelihood_ratio_test(dg, delta_t, theta_null, theta_alt,
                                    null_model, alt_model, use_skewed=use_skewed)
        lam[k] = res.lambda_lr
    return lam


# ── Greedy forward template selection ────────────────────────────────────────

# Cache JIT-compiled additive log-likelihood functions by (n_sys, use_skewed)
# so the forward-selection loop doesn't recompile on every round.
_ll_fwd_cache: dict[tuple[int, bool], object] = {}


def _ols_mle_theta(
    delta_g: np.ndarray,
    delta_t: np.ndarray,
    use_skewed: bool,
) -> np.ndarray:
    """OLS MLE theta for the additive model (fast surrogate for MCMC MLE).

    Returns the flat parameter vector ``[a_0, …, a_{n-1}, sigma, (gamma)]``
    where *a* are OLS coefficients and *sigma* is the residual std.
    For an empty template set the vector is ``[sigma]`` (or ``[sigma, 0]``).
    """
    n = delta_t.shape[0]
    if n == 0:
        sigma = max(float(np.std(delta_g)), 1e-9)
        a = np.empty(0)
    else:
        a, _, _, _ = np.linalg.lstsq(delta_t.T, delta_g, rcond=None)
        resid = delta_g - a @ delta_t
        sigma = max(float(np.std(resid)), 1e-9)
    return pack_params(
        a, None, sigma,
        gamma=0.0 if use_skewed else None,
        model='additive',
    )


def _get_ll_fn_fwd(n_sys: int, use_skewed: bool):
    """Return (and cache) the JIT-compiled additive log-likelihood for *n_sys*."""
    key = (n_sys, use_skewed)
    if key not in _ll_fwd_cache:
        _ll_fwd_cache[key] = make_log_likelihood(n_sys, 'additive', use_skewed)
    return _ll_fwd_cache[key]


def _lrt_p_add_one(
    delta_g: np.ndarray,
    delta_t_current: np.ndarray,
    t_candidate: np.ndarray,
    use_skewed: bool,
) -> float:
    """χ²(1) LRT p-value for adding *t_candidate* to the current template set.

    Null: additive model with *delta_t_current* (*n_null* templates).
    Alt:  additive model with *delta_t_current* ∪ *t_candidate*.
    Both thetas are computed via OLS MLE.
    """
    n_null      = delta_t_current.shape[0]
    delta_t_alt = np.vstack([delta_t_current, t_candidate[np.newaxis]])

    theta_null = _ols_mle_theta(delta_g, delta_t_current, use_skewed)
    theta_alt  = _ols_mle_theta(delta_g, delta_t_alt,    use_skewed)

    ll_null_fn = _get_ll_fn_fwd(n_null,     use_skewed)
    ll_alt_fn  = _get_ll_fn_fwd(n_null + 1, use_skewed)

    _dg = jnp.asarray(delta_g,          dtype=jnp.float64)
    _tc = jnp.asarray(delta_t_current,  dtype=jnp.float64)
    _ta = jnp.asarray(delta_t_alt,      dtype=jnp.float64)

    lv_null = float(ll_null_fn(jnp.asarray(theta_null, dtype=jnp.float64), _dg, _tc))
    lv_alt  = float(ll_alt_fn( jnp.asarray(theta_alt,  dtype=jnp.float64), _dg, _ta))

    # theta here are exact OLS maxima, so lambda_lr >= 0 analytically; the clip
    # only absorbs float noise at the 1e-12 level.  Anything larger is a real
    # regression and must not be silently flattened.
    lambda_lr = 2.0 * (lv_alt - lv_null)
    if lambda_lr < -1e-6:
        warnings.warn(
            f"forward selection: lambda_LR = {lambda_lr:.6g} < 0 between nested "
            "OLS fits, which is analytically impossible; check _ols_mle_theta.",
            RuntimeWarning,
            stacklevel=2,
        )
    return float(chi2.sf(max(lambda_lr, 0.0), df=1))


@dataclass
class ForwardSelectionRound:
    """Diagnostics for one round of greedy forward template selection."""
    round_num: int
    added_index: int
    p_value: float


@dataclass
class GreedyForwardSelectionResult:
    """Result of :func:`greedy_forward_select`.

    Attributes
    ----------
    selected_indices : list[int]
        Indices into the input *delta_t* of accepted templates, in selection order.
    rounds : list[ForwardSelectionRound]
        Per-round diagnostics: which template was added and at what p-value.
    p_threshold : float
        The LRT gate used.
    n_initial : int
        Number of candidate templates passed in.
    """
    selected_indices: list[int]
    rounds: list[ForwardSelectionRound]
    p_threshold: float
    n_initial: int


def greedy_forward_select(
    delta_g: np.ndarray,
    delta_t: np.ndarray,
    p_threshold: float = 0.05,
    use_skewed: bool = False,
) -> GreedyForwardSelectionResult:
    """Greedy LRT-based forward template selection.

    Starting from an empty model, iteratively adds the template whose inclusion
    yields the smallest χ²(1) LRT p-value, stopping when no remaining candidate
    achieves ``p_value < p_threshold``.  OLS is used to compute the MLE at each
    step, avoiding the cost of full MCMC inference during selection.

    Parameters
    ----------
    delta_g : (n_pix,) observed galaxy overdensity.
    delta_t : (n_cand, n_pix) candidate template pixel values.
    p_threshold : float
        Significance gate; a candidate is added only if its p-value is strictly
        below this threshold.  Default 0.05.
    use_skewed : bool
        Whether to include a skewness parameter in the likelihood.  Default False.

    Returns
    -------
    GreedyForwardSelectionResult
        ``.selected_indices`` lists the row indices into *delta_t* for the
        accepted subset, in the order they were added.

    Notes
    -----
    JIT-compiled likelihood functions are cached at module level the first time
    each (n_sys, use_skewed) combination is encountered, so subsequent calls
    with the same sizes are fast.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import greedy_forward_select
    >>> rng = np.random.default_rng(0)
    >>> n_pix = 2000
    >>> delta_t = rng.standard_normal((5, n_pix))
    >>> # Pure noise — no template should be significant at very tight threshold
    >>> result = greedy_forward_select(rng.standard_normal(n_pix), delta_t,
    ...                                p_threshold=1e-6)
    >>> len(result.selected_indices) == 0
    True
    """
    n_cand = delta_t.shape[0]
    if n_cand == 0:
        return GreedyForwardSelectionResult(
            selected_indices=[], rounds=[], p_threshold=p_threshold, n_initial=0,
        )

    cand_idx   = list(range(n_cand))
    sel_idx: list[int] = []
    dt_sel     = np.empty((0, delta_t.shape[1]), dtype=delta_t.dtype)
    rounds: list[ForwardSelectionRound] = []

    while cand_idx:
        p_vals = [
            _lrt_p_add_one(delta_g, dt_sel, delta_t[i], use_skewed)
            for i in cand_idx
        ]
        best_local  = int(np.argmin(p_vals))
        best_p      = p_vals[best_local]
        if best_p >= p_threshold:
            break
        best_global = cand_idx[best_local]
        sel_idx.append(best_global)
        cand_idx.remove(best_global)
        dt_sel = delta_t[np.array(sel_idx)]
        rounds.append(ForwardSelectionRound(
            round_num=len(rounds) + 1,
            added_index=best_global,
            p_value=best_p,
        ))

    return GreedyForwardSelectionResult(
        selected_indices=sel_idx,
        rounds=rounds,
        p_threshold=p_threshold,
        n_initial=n_cand,
    )


# ── SNR-based template pre-selection ─────────────────────────────────────────

@dataclass
class SnrPreselectionResult:
    """Result of :func:`snr_preselect`.

    Attributes
    ----------
    selected_indices : list[int]
        Indices into *delta_t*, sorted from highest to lowest SNR.
    snr_values : np.ndarray
        SNR/Δχ² for **all** ``n_initial`` templates (shape ``(n_initial,)``).
    method : str
        SNR estimator used (``"data"``, ``"template"``, ``"peak"``, or ``"isd"``).
    snr_min : float or None
        Minimum SNR threshold applied, or ``None`` if not used.
    n_top : int or None
        Top-K cap applied, or ``None`` if not used.
    n_initial : int
        Total number of candidate templates passed in.
    """
    selected_indices: list[int]
    snr_values: np.ndarray
    method: str
    snr_min: float | None
    n_top: int | None
    n_initial: int


def snr_preselect(
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    *,
    method: str = "data",
    snr_min: float | None = None,
    n_top: int | None = None,
    n_bins: int = 10,
    poly_order: int = 1,
    fracdet: np.ndarray | None = None,
) -> SnrPreselectionResult:
    """Rank and pre-select templates by SNR of their cross-correlation with data.

    Step 1 of the two-stage decontamination pipeline: compute a cheap SNR metric
    for every candidate template and keep only those above a threshold and/or in
    the top-K by SNR.  Pass the ``selected_indices`` output to
    :func:`greedy_forward_select` (or direct MCMC) for Step 2.

    For mock-based significance (the paper method), call
    :func:`~diagnostics.isd_template_significance` directly and threshold its
    ``"p_values"`` output.

    Parameters
    ----------
    delta_g_obs : (n_pix,) observed galaxy overdensity.
    delta_t : (n_cand, n_pix) candidate template pixel values.
    method :
        SNR estimator; one of ``"data"``, ``"template"``, ``"peak"``,
        ``"isd"``.  Default ``"data"`` (Pearson cross-correlation).
    snr_min :
        Keep only templates with ``snr >= snr_min``.  If ``None``, no
        threshold is applied.
    n_top :
        Keep at most the ``n_top`` highest-SNR templates.  Applied after
        ``snr_min``.  If ``None``, no cap is applied.
    n_bins :
        Number of equal-width bins for ``method="isd"``.
    poly_order :
        Polynomial degree for the 1D fit in ``method="isd"``.
    fracdet :
        Per-pixel fractional coverage weights for ``method="isd"``.

    Returns
    -------
    SnrPreselectionResult
        ``.selected_indices`` lists the accepted template indices in
        descending SNR order.  ``.snr_values`` contains the SNR for all
        ``n_initial`` templates (useful for inspection).

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import snr_preselect, greedy_forward_select
    >>> rng = np.random.default_rng(0)
    >>> n_pix, n_cand = 4000, 8
    >>> delta_t = rng.standard_normal((n_cand, n_pix))
    >>> delta_g = 0.6 * delta_t[3] + rng.standard_normal(n_pix) * 0.1
    >>> pre = snr_preselect(delta_g, delta_t, method="data", n_top=4)
    >>> len(pre.selected_indices)
    4
    >>> pre.selected_indices[0]  # template 3 should rank first
    3
    >>> pre.snr_values.shape
    (8,)
    """
    n_cand = delta_t.shape[0]

    snr = snr_template_ranking(
        delta_g_obs, delta_t,
        method=method,
        n_bins=n_bins,
        poly_order=poly_order,
        fracdet=fracdet,
    )

    order = np.argsort(snr)[::-1]  # descending SNR

    if snr_min is not None:
        order = order[snr[order] >= snr_min]

    if n_top is not None:
        order = order[:n_top]

    return SnrPreselectionResult(
        selected_indices=order.tolist(),
        snr_values=snr,
        method=method,
        snr_min=snr_min,
        n_top=n_top,
        n_initial=n_cand,
    )
