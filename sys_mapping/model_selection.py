"""Likelihood ratio test for model selection (Eq. 19, Berlfein et al. 2024).

The likelihood ratio statistic:
    λ_LR = 2 [ln L(Θ̂) - ln L(Θ̂_0)]

is asymptotically χ²(r) distributed under the null hypothesis, where
r = dim(Θ̂) - dim(Θ̂_0) is the number of additional free parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .likelihood import make_log_likelihood
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
    theta_null : flat parameter vector for the null model (from :func:`~inference.get_mle_params`)
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
        ``lambda_lr`` : test statistic (≥ 0 only when both theta are at their MLEs)
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
        Returns the MLE/posterior-median flat parameter vector for ``model`` on one mock — e.g. a
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

    lambda_lr = max(2.0 * (lv_alt - lv_null), 0.0)
    return float(chi2.sf(lambda_lr, df=1))


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
