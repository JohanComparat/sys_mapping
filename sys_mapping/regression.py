"""Regression-based systematic decontamination methods.

Implements:
  - ElasticNet regularised regression (Weaverdyck & Huterer 2021)
  - Iterative Systematics Decontamination via polynomial OLS (Rodríguez-Monroy+2025)
  - Multi-method comparison framework (Weaverdyck & Huterer 2021)
  - Unified run_decontamination() interface for all 6 methods
"""

from __future__ import annotations

import itertools
import time
import warnings
from typing import Callable

import numpy as np

_VALID_METHODS = frozenset({"OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"})

# Maximum weight allowed during ISD iterations and in the final ISD output.
# Pixels where 1 + a@t ≈ 0 would otherwise produce weights → 1/eps = 1e6,
# making all subsequent re-weighting steps meaningless.  A cap of 20 corresponds
# to a denominator floor of 0.05 — generous enough not to bias well-behaved
# scenarios while preventing numerical blow-up under strong contamination.
_ISD_MAX_WEIGHT: float = 20.0


def _compute_weights(
    alpha: np.ndarray,
    delta_t: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-pixel systematic weights w(p) = 1 / (1 + alpha · t(p)).

    Parameters
    ----------
    alpha:
        Regression coefficients (shape ``(n_sys,)``).
    delta_t:
        Template values (shape ``(n_sys, n_pix)``).
    eps:
        Minimum denominator to prevent division by zero.

    Returns
    -------
    weights:
        Per-pixel weights (shape ``(n_pix,)``), all positive.
    """
    systematic_field = alpha @ delta_t  # (n_pix,)
    denominator = np.maximum(1.0 + systematic_field, eps)
    return 1.0 / denominator


def elasticnet_contamination_fit(
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    *,
    l1_ratio: float = 0.5,
    alpha_reg: float | None = None,
    cv_folds: int = 5,
    max_iter: int = 10_000,
    fit_intercept: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit systematic contamination amplitudes via ElasticNet regression.

    Minimises the penalised least-squares loss (Weaverdyck & Huterer 2021, Eq. 9):

    .. math::

        \\mathcal{L} = \\frac{1}{2N_{\\rm pix}}
        \\left\\|\\delta_g - \\sum_i\\alpha_i\\,t_i\\right\\|_2^2
        + \\lambda_1\\|\\boldsymbol\\alpha\\|_1
        + \\lambda_2\\|\\boldsymbol\\alpha\\|_2^2

    where the total regularisation strength and L1 fraction are controlled by
    ``alpha_reg`` and ``l1_ratio``.

    Parameters
    ----------
    delta_g_obs:
        Observed galaxy overdensity (shape ``(n_pix,)``).
    delta_t:
        Template maps (shape ``(n_sys, n_pix)``).
    l1_ratio:
        ElasticNet mixing parameter: 1.0 = Lasso, 0.0 = Ridge.
    alpha_reg:
        Regularisation strength.  If ``None``, tuned automatically via
        ``cv_folds``-fold cross-validation (``ElasticNetCV``).
    cv_folds:
        Number of cross-validation folds used when ``alpha_reg=None``.
    max_iter:
        Maximum coordinate descent iterations.
    fit_intercept:
        Whether to fit a constant offset.  Typically ``False`` because
        the overdensity field has zero mean by construction.

    Returns
    -------
    alpha_hat:
        Fitted contamination amplitudes (shape ``(n_sys,)``).
    weights:
        Per-pixel systematic weights ``w(p) = 1 / (1 + alpha_hat · t(p))``
        (shape ``(n_pix,)``).
    cv_info:
        Dictionary with keys ``alpha_reg`` (used value), ``l1_ratio``,
        and ``cv_scores`` (array of cross-validation MSE scores, or ``None``
        if ``alpha_reg`` was provided explicitly).

    Notes
    -----
    Requires ``scikit-learn >= 1.3``.  Install via::

        pip install "sys_mapping[regression]"

    The per-pixel weight corresponds to the DES-Y1 / Weaverdyck+2021 form
    :math:`w(p) = (1 + \\hat{\\boldsymbol\\alpha}\\cdot\\mathbf{t}(p))^{-1}`.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import elasticnet_contamination_fit
    >>> rng = np.random.default_rng(42)
    >>> n_pix, n_sys = 5000, 3
    >>> delta_t = rng.standard_normal((n_sys, n_pix))
    >>> true_alpha = np.array([0.2, -0.1, 0.05])
    >>> delta_g = true_alpha @ delta_t + rng.standard_normal(n_pix) * 0.3
    >>> alpha_hat, weights, info = elasticnet_contamination_fit(
    ...     delta_g, delta_t, alpha_reg=1e-4)
    >>> weights.shape
    (5000,)
    >>> np.all(weights > 0)
    True

    References
    ----------
    Weaverdyck & Huterer 2021, MNRAS 503, 5061.
    """
    try:
        from sklearn.linear_model import ElasticNet, ElasticNetCV
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for elasticnet_contamination_fit. "
            "Install it via: pip install 'sys_mapping[regression]'"
        ) from exc

    X = delta_t.T  # (n_pix, n_sys)
    y = delta_g_obs  # (n_pix,)

    cv_scores = None
    if alpha_reg is None:
        model = ElasticNetCV(
            l1_ratio=l1_ratio,
            cv=cv_folds,
            max_iter=max_iter,
            fit_intercept=fit_intercept,
            n_jobs=-1,
        )
        model.fit(X, y)
        alpha_reg = float(model.alpha_)
        cv_scores = model.mse_path_.mean(axis=-1)
    else:
        model = ElasticNet(
            alpha=alpha_reg,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            fit_intercept=fit_intercept,
        )
        model.fit(X, y)

    alpha_hat = model.coef_  # (n_sys,)
    weights = np.clip(
        _compute_weights(alpha_hat, delta_t),
        1.0 / _ISD_MAX_WEIGHT, _ISD_MAX_WEIGHT,
    )
    cv_info = {
        "alpha_reg": alpha_reg,
        "l1_ratio": l1_ratio,
        "cv_scores": cv_scores,
    }
    return alpha_hat, weights, cv_info


def iterative_systematics_decontamination(
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    *,
    poly_order: int = 3,
    max_iter: int = 20,
    tol: float = 1e-5,
    lambda_poly: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Iterative Systematics Decontamination (ISD) via polynomial OLS.

    Expands templates to include cross-products up to ``poly_order``, then
    iteratively fits OLS weights until convergence.  Implements the approach
    of Rodríguez-Monroy et al. 2025 (DES Y6), Sec. 3.2.

    The ISD cleansed overdensity after convergence is:

    .. math::

        \\delta_g^{\\rm clean}(p) =
        \\frac{\\delta_g(p) - f_{\\rm add}(p)}{1 + f_{\\rm mult}(p)}

    where :math:`f_{\\rm add}` and :math:`f_{\\rm mult}` are the additive and
    multiplicative systematic components estimated from polynomial template
    regression (here approximated by a single linear OLS pass per iteration).

    Parameters
    ----------
    delta_g_obs:
        Observed galaxy overdensity (shape ``(n_pix,)``).
    delta_t:
        Template maps (shape ``(n_sys, n_pix)``).
    poly_order:
        Maximum polynomial order for template expansion.  Order 1 = linear
        templates only; order 2 adds pairwise products; order 3 adds triple
        products.  Automatically reduced if the expanded template matrix is
        underdetermined.
    max_iter:
        Maximum number of OLS iterations.
    tol:
        Convergence tolerance on the relative change in weights
        (``||w_new - w_old|| / ||w_old||``).
    lambda_poly:
        Ridge penalty applied exclusively to the polynomial (non-linear)
        expansion columns — the first ``n_sys`` linear columns are never
        penalised.  A small positive value (e.g. ``1e-3 * np.var(delta_g_obs)``)
        prevents the polynomial terms from absorbing signal that belongs to the
        linear coefficients, which are the only ones used for the weight update.
        Defaults to ``0.0`` (plain OLS, backward-compatible).

    Returns
    -------
    weights:
        Per-pixel systematic weights (shape ``(n_pix,)``), all positive.
    alpha_hat_all:
        Fitted coefficients for the expanded template matrix at the final
        iteration (shape ``(n_expanded,)``).
    n_iterations:
        Number of iterations performed (1 if converged on first pass).

    Notes
    -----
    Template expansion uses
    ``itertools.combinations_with_replacement`` — no new dependencies.
    For ``n_sys=5``, ``poly_order=3`` the expanded set has
    :math:`\\binom{n+k}{k} - 1 = 55` columns.  If ``n_pix < 10 × n_expanded``,
    ``poly_order`` is reduced with a warning.

    **Speed:** the design matrix ``X`` is constant across iterations; only the
    diagonal weight matrix changes.  The per-iteration cost is therefore one
    BLAS-3 call ``X^T diag(w) X`` of shape ``(n_expanded, n_expanded)`` plus a
    small ``lstsq`` solve — roughly 5–15× faster than SVD of the full
    ``(n_pix × n_expanded)`` matrix that the previous sqrt-weight formulation
    required.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import iterative_systematics_decontamination
    >>> rng = np.random.default_rng(0)
    >>> n_pix, n_sys = 8000, 3
    >>> delta_t = rng.standard_normal((n_sys, n_pix))
    >>> true_alpha = np.array([0.15, -0.08, 0.04])
    >>> delta_g = true_alpha @ delta_t + rng.standard_normal(n_pix) * 0.4
    >>> weights, alpha_all, n_it = iterative_systematics_decontamination(
    ...     delta_g, delta_t, poly_order=2, max_iter=10)
    >>> weights.shape
    (8000,)
    >>> np.all(weights > 0)
    True

    References
    ----------
    Rodríguez-Monroy et al. 2025, arXiv:2509.07943.
    """
    n_sys, n_pix = delta_t.shape

    # Build polynomial template expansion
    def _build_expanded(t: np.ndarray, order: int) -> np.ndarray:
        rows = [t]
        for deg in range(2, order + 1):
            for combo in itertools.combinations_with_replacement(range(n_sys), deg):
                product = np.ones(n_pix)
                for idx in combo:
                    product *= t[idx]
                rows.append(product[np.newaxis, :])
        return np.vstack(rows)  # (n_expanded, n_pix)

    # Reduce poly_order if problem is underdetermined
    current_order = poly_order
    while current_order > 1:
        test = _build_expanded(delta_t, current_order)
        n_expanded = test.shape[0]
        if n_pix >= 10 * n_expanded:
            break
        warnings.warn(
            f"poly_order={current_order} gives {n_expanded} columns but n_pix={n_pix}. "
            f"Reducing poly_order to {current_order - 1}.",
            stacklevel=2,
        )
        current_order -= 1

    delta_t_expanded = _build_expanded(delta_t, current_order)
    n_expanded = delta_t_expanded.shape[0]

    weights = np.ones(n_pix)
    alpha_hat_all = np.zeros(n_expanded)

    # X is (n_pix, n_expanded) and never changes between iterations; precompute
    # its transpose so the per-iteration cost is one BLAS-3 matmul rather than a
    # full SVD of the (n_pix × n_expanded) matrix.
    X = delta_t_expanded.T   # (n_pix, n_expanded)
    Xt = delta_t_expanded    # (n_expanded, n_pix)

    # Ridge penalty on polynomial-only columns: prevents them from absorbing
    # signal that belongs to the linear coefficients used in the weight update.
    # Linear columns (first n_sys) are never penalised.
    ridge_diag = np.zeros(n_expanded)
    ridge_diag[n_sys:] = lambda_poly

    for iteration in range(1, max_iter + 1):
        weights_old = weights.copy()

        # Weighted normal equations: (X^T W X + diag(ridge)) α = X^T W y.
        # Xw = X^T diag(w) is computed via broadcasting: (n_expanded, n_pix).
        # XtWX and XtWy are (n_expanded, n_expanded) and (n_expanded,) — small
        # regardless of n_pix. lstsq on this small system is SVD-stable even
        # when lambda_poly=0.
        Xw = Xt * weights          # (n_expanded, n_pix): row i scaled by w
        XtWX = Xw @ X              # (n_expanded, n_expanded)
        XtWX.flat[:: n_expanded + 1] += ridge_diag   # add ridge to diagonal in-place
        XtWy = Xw @ delta_g_obs    # (n_expanded,)
        alpha_hat_all, *_ = np.linalg.lstsq(XtWX, XtWy, rcond=None)

        weights = _compute_weights(alpha_hat_all[:n_sys], delta_t)
        # Clip to prevent divergent re-weighting. Without this, pixels where
        # 1 + a@t → 0 get weight 1/eps = 1e6, which makes all subsequent
        # weighted OLS fits numerically meaningless. Clip to [1/w_max, w_max]
        # so the iteration stays in a stable regime.
        weights = np.clip(weights, 1.0 / _ISD_MAX_WEIGHT, _ISD_MAX_WEIGHT)

        rel_change = np.linalg.norm(weights - weights_old) / (np.linalg.norm(weights_old) + 1e-30)
        if rel_change < tol:
            return weights, alpha_hat_all, iteration

    return weights, alpha_hat_all, max_iter


def method_comparison(
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    good_pixels: np.ndarray,
    nside: int,
    methods: tuple[str, ...] = ("combined_mcmc", "additive_mcmc", "elasticnet", "ols"),
    seed: int = 42,
    mcmc_kwargs: dict | None = None,
) -> dict[str, dict]:
    """Run multiple decontamination methods on the same data and compare.

    Implements the multi-method comparison framework of Weaverdyck & Huterer
    2021 (Sec. 4).  Each method returns coefficients, per-pixel weights, and
    a goodness-of-fit metric, enabling direct comparison.

    Parameters
    ----------
    delta_g_obs:
        Observed overdensity (shape ``(n_pix,)``).
    delta_t:
        Template maps (shape ``(n_sys, n_pix)``).
    good_pixels:
        Boolean mask (shape ``(n_full_pix,)``).  Passed through for spatial
        context but not used in computation directly (templates already masked).
    nside:
        HEALPix NSIDE of the full map.
    methods:
        Subset of ``{"combined_mcmc", "additive_mcmc", "elasticnet", "ols"}``.
        ``"combined_mcmc"`` (the default first entry) is the recommended method;
        it infers both additive and multiplicative amplitudes jointly.
    seed:
        Random seed for MCMC initialisation.
    mcmc_kwargs:
        Keyword arguments forwarded to :func:`~sys_mapping.inference.run_mcmc`
        for the ``"additive_mcmc"`` method.

    Returns
    -------
    results:
        Dictionary keyed by method name.  Each value is a dict with:

        - ``alpha_hat`` or ``a_hat``: contamination amplitudes
        - ``weights``: per-pixel weights (shape ``(n_pix,)``)
        - ``chi2_residual``: sum-of-squared residuals / n_pix

    Notes
    -----
    ``"additive_mcmc"`` requires no optional dependencies (uses the built-in
    MCMC pipeline).  ``"elasticnet"`` requires ``scikit-learn``.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import method_comparison
    >>> rng = np.random.default_rng(0)
    >>> n_pix, n_sys = 3000, 2
    >>> delta_t = rng.standard_normal((n_sys, n_pix))
    >>> delta_g = 0.1 * delta_t[0] + rng.standard_normal(n_pix) * 0.5
    >>> good = np.ones(n_pix, dtype=bool)
    >>> results = method_comparison(delta_g, delta_t, good, nside=16,
    ...                             methods=("elasticnet", "ols"))
    >>> set(results.keys()) == {"elasticnet", "ols"}
    True
    >>> "weights" in results["ols"]
    True

    References
    ----------
    Weaverdyck & Huterer 2021, MNRAS 503, 5061.
    """
    results: dict[str, dict] = {}
    n_pix = delta_g_obs.shape[0]

    def _chi2(alpha: np.ndarray) -> float:
        residual = delta_g_obs - alpha @ delta_t
        return float(np.sum(residual**2) / n_pix)

    if "ols" in methods:
        X = delta_t.T  # (n_pix, n_sys)
        alpha_ols, *_ = np.linalg.lstsq(X, delta_g_obs, rcond=None)
        weights_ols = _compute_weights(alpha_ols, delta_t)
        results["ols"] = {
            "alpha_hat": alpha_ols,
            "weights": weights_ols,
            "chi2_residual": _chi2(alpha_ols),
        }

    if "elasticnet" in methods:
        alpha_en, weights_en, cv_info = elasticnet_contamination_fit(
            delta_g_obs, delta_t, alpha_reg=None
        )
        results["elasticnet"] = {
            "alpha_hat": alpha_en,
            "weights": weights_en,
            "chi2_residual": _chi2(alpha_en),
            "cv_info": cv_info,
        }

    if "combined_mcmc" in methods or "additive_mcmc" in methods:
        from .inference import run_mcmc, get_mle_params
        from .contamination import unpack_params

        kw = {"seed": seed}
        if mcmc_kwargs:
            kw.update(mcmc_kwargs)

    if "combined_mcmc" in methods:
        flat_chain, _ = run_mcmc(
            n_sys=delta_t.shape[0],
            model="combined",
            delta_g_obs=delta_g_obs,
            delta_t=delta_t,
            **kw,
        )
        theta_hat = get_mle_params(flat_chain)
        n_sys = delta_t.shape[0]
        a_hat, b_hat, _, _ = unpack_params(theta_hat, n_sys, "combined", use_skewed=False)
        weights_comb = _compute_weights(b_hat, delta_t)
        results["combined_mcmc"] = {
            "a_hat": np.asarray(a_hat),
            "b_hat": np.asarray(b_hat),
            "weights": weights_comb,
            "chi2_residual": _chi2(np.asarray(a_hat)),
        }

    if "additive_mcmc" in methods:
        flat_chain, _ = run_mcmc(
            n_sys=delta_t.shape[0],
            model="additive",
            delta_g_obs=delta_g_obs,
            delta_t=delta_t,
            **kw,
        )
        theta_hat = get_mle_params(flat_chain)
        n_sys = delta_t.shape[0]
        a_hat, _, _, _ = unpack_params(theta_hat, n_sys, "additive", use_skewed=False)
        weights_mcmc = _compute_weights(a_hat, delta_t)
        results["additive_mcmc"] = {
            "a_hat": np.asarray(a_hat),
            "weights": weights_mcmc,
            "chi2_residual": _chi2(np.asarray(a_hat)),
        }

    return results


def run_decontamination(
    method: str,
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    *,
    n_walkers: int = 100,
    n_steps: int = 600,
    n_burn: int = 100,
    seed: int = 42,
    use_skewed: bool = False,
    progress: bool = False,
    cv_folds: int = 5,
    isd_max_iter: int = 50,
    isd_lambda_poly: float = 0.0,
) -> dict:
    """Run a single decontamination method and return a standardised result dict.

    This is the unified entry point used by both validation scripts and
    production pipelines.  All 6 methods share the same interface so callers
    can iterate over a list of method names without any method-specific
    boilerplate.

    Parameters
    ----------
    method:
        One of ``"OLS"``, ``"ElasticNet"``, ``"ISD-1"``, ``"ISD-3"``,
        ``"MCMC-add"``, ``"MCMC-comb"``.
    delta_g_obs:
        Observed galaxy overdensity at unmasked pixels, shape ``(n_pix,)``.
    delta_t:
        Template maps in the *original* (non-rotated) basis,
        shape ``(n_sys, n_pix)``.  MCMC methods rotate internally.
    n_walkers, n_steps, n_burn:
        MCMC sampler settings.
    seed:
        Random seed for MCMC walker initialisation.
    use_skewed:
        Use skew-normal likelihood for MCMC-comb.  Ignored for all other
        methods.
    progress:
        Show emcee progress bar (MCMC methods only).
    cv_folds:
        Cross-validation folds for ElasticNet (when ``alpha_reg`` is auto).
    isd_max_iter:
        Maximum iterations for ISD methods.
    isd_lambda_poly:
        Ridge penalty on the polynomial expansion columns for ISD-3 (see
        :func:`iterative_systematics_decontamination`).  A value of
        ``1e-3 * np.var(delta_g_obs)`` is recommended when ISD-3 converges
        to implausible solutions.  Has no effect for ISD-1 (poly_order=1 has
        no polynomial-only columns).  Default ``0.0`` (plain OLS).

    Returns
    -------
    dict with keys:

    All methods
        ``a_hat`` (n_sys,), ``b_hat`` (n_sys,), ``weights`` (n_pix,),
        ``elapsed_s`` (float).
    MCMC methods additionally
        ``flat_chain``, ``sampler``, ``a_rot``, ``b_rot``,
        ``cov_a_rot``, ``cov_b_rot``, ``cov_a``, ``cov_b``,
        ``R``, ``acceptance_fraction``, ``sigma_hat``.
    ISD methods additionally
        ``n_iterations`` (int).
    ElasticNet additionally
        ``cv_info`` (dict).
    Non-applicable keys are ``None``.

    Notes
    -----
    **Weight convention**

    * OLS / ElasticNet / ISD-1 / ISD-3 / MCMC-add:
      ``w(p) = 1 / (1 + a_hat @ t(p))`` — additive weight.
    * MCMC-comb:
      ``w(p) = 1 / (1 + b_hat @ t(p))`` — multiplicative weight, matching
      the production weight-map convention used in real-data pipelines.

    References
    ----------
    Berlfein et al. 2024, MNRAS 531, 4954.
    Weaverdyck & Huterer 2021, MNRAS 503, 5061.
    Rodríguez-Monroy et al. 2025, arXiv:2509.07943.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_VALID_METHODS)}, got {method!r}"
        )

    n_sys, n_pix = delta_t.shape
    t0 = time.perf_counter()

    # ── Default return scaffold (all None for method-specific extras) ─────────
    result: dict = {
        "a_hat": None, "b_hat": None, "weights": None, "elapsed_s": None,
        # MCMC-only
        "flat_chain": None, "sampler": None,
        "a_rot": None, "b_rot": None,
        "cov_a_rot": None, "cov_b_rot": None,
        "cov_a": None, "cov_b": None,
        "R": None,
        "acceptance_fraction": None, "sigma_hat": None,
        # ISD-only
        "n_iterations": None,
        "isd_outlier_mask": None,    # bool (n_pix,): True = pixel excluded in pass 2
        "isd_masked_fraction": None, # float: fraction of pixels excluded in pass 2
        # ElasticNet-only
        "cv_info": None,
    }

    # ── OLS ───────────────────────────────────────────────────────────────────
    if method == "OLS":
        X = delta_t.T  # (n_pix, n_sys)
        a_hat, *_ = np.linalg.lstsq(X, delta_g_obs, rcond=None)
        b_hat = np.zeros(n_sys)
        weights = _compute_weights(a_hat, delta_t)
        result.update({"a_hat": a_hat, "b_hat": b_hat, "weights": weights})

    # ── ElasticNet ────────────────────────────────────────────────────────────
    elif method == "ElasticNet":
        a_hat, weights, cv_info = elasticnet_contamination_fit(
            delta_g_obs, delta_t, cv_folds=cv_folds
        )
        b_hat = np.zeros(n_sys)
        result.update({"a_hat": a_hat, "b_hat": b_hat, "weights": weights,
                        "cv_info": cv_info})

    # ── ISD-1 / ISD-3 ─────────────────────────────────────────────────────────
    elif method in ("ISD-1", "ISD-3"):
        poly_order = 1 if method == "ISD-1" else 3

        # Pass 1 — run on all pixels (weights clipped to [1/W, W] internally).
        weights_p1, alpha_all_p1, n_iter_p1 = iterative_systematics_decontamination(
            delta_g_obs, delta_t, poly_order=poly_order, max_iter=isd_max_iter,
            lambda_poly=isd_lambda_poly,
        )

        # Pixels that hit the clip boundary are pathological: 1 + a@t ≈ 0.
        # Mask them and refit so the final coefficients are not biased by them.
        _boundary = _ISD_MAX_WEIGHT * (1.0 - 1e-6)
        outlier_mask = (weights_p1 >= _boundary) | (weights_p1 <= 1.0 / _boundary)
        n_outlier = int(outlier_mask.sum())
        masked_fraction = n_outlier / n_pix

        if n_outlier > 0 and (n_pix - n_outlier) >= max(10 * n_sys, 50):
            # Pass 2 — refit on pixels that stayed away from the clip boundary.
            good = ~outlier_mask
            weights_p2, alpha_all_p2, n_iter_p2 = iterative_systematics_decontamination(
                delta_g_obs[good], delta_t[:, good],
                poly_order=poly_order, max_iter=isd_max_iter,
                lambda_poly=isd_lambda_poly,
            )
            # Apply clean coefficients to ALL pixels (outliers included).
            a_hat = np.asarray(alpha_all_p2)[:n_sys]
            weights = _compute_weights(a_hat, delta_t)
            weights = np.clip(weights, 1.0 / _ISD_MAX_WEIGHT, _ISD_MAX_WEIGHT)
            n_iterations = n_iter_p1 + n_iter_p2
            warnings.warn(
                f"{method}: {n_outlier}/{n_pix} outlier pixels masked for pass 2 "
                f"(pass1={n_iter_p1} iters, pass2={n_iter_p2} iters).",
                UserWarning,
                stacklevel=2,
            )
        else:
            a_hat = np.asarray(alpha_all_p1)[:n_sys]
            weights = weights_p1
            n_iterations = n_iter_p1

        b_hat = np.zeros(n_sys)
        # Guard: non-finite coefficients (residual edge case).
        if not np.all(np.isfinite(a_hat)):
            warnings.warn(
                f"{method} produced non-finite coefficients; "
                "a_hat zeroed, weights set to ones.",
                stacklevel=2,
            )
            a_hat = np.zeros(n_sys)
            weights = np.ones(n_pix)
            outlier_mask = np.zeros(n_pix, dtype=bool)
            masked_fraction = 0.0

        result.update({
            "a_hat": a_hat, "b_hat": b_hat, "weights": weights,
            "n_iterations": n_iterations,
            "isd_outlier_mask": outlier_mask,
            "isd_masked_fraction": masked_fraction,
        })

    # ── MCMC-add / MCMC-comb ──────────────────────────────────────────────────
    else:
        from .inference import run_mcmc, get_mle_params, get_param_covariance_from_chain
        from .contamination import unpack_params
        from .correction import rotate_templates, transform_params_from_rotated

        mcmc_model = "additive" if method == "MCMC-add" else "combined"
        _use_skewed = (use_skewed if mcmc_model == "combined" else False)

        # PCA rotation — deterministic for fixed delta_t
        delta_t_rot, R, _eigenvalues = rotate_templates(delta_t)

        # Emcee requires n_walkers >= 2*n_dim+2
        n_dim = n_sys + 1 if mcmc_model == "additive" else 2 * n_sys + 1
        nw = max(n_walkers, 2 * n_dim + 2)

        flat_chain, sampler = run_mcmc(
            n_sys=n_sys, model=mcmc_model,
            delta_g_obs=delta_g_obs, delta_t=delta_t_rot,
            n_walkers=nw, n_steps=n_steps, n_burn=n_burn,
            seed=seed, progress=progress,
            use_skewed=_use_skewed,
        )

        theta_hat = get_mle_params(flat_chain)
        a_rot_raw, b_rot_raw, sigma_hat_val, _ = unpack_params(
            theta_hat, n_sys, mcmc_model, use_skewed=_use_skewed
        )
        a_hat, b_hat = transform_params_from_rotated(
            np.asarray(a_rot_raw), np.asarray(b_rot_raw), R
        )

        cov_a_rot, cov_b_rot = get_param_covariance_from_chain(
            flat_chain, n_sys, mcmc_model
        )
        cov_a = R.T @ cov_a_rot @ R
        cov_b = R.T @ cov_b_rot @ R

        # Weight convention: additive for MCMC-add; multiplicative for MCMC-comb
        weights = _compute_weights(
            b_hat if mcmc_model == "combined" else a_hat, delta_t
        )

        result.update({
            "a_hat": a_hat, "b_hat": b_hat, "weights": weights,
            "flat_chain": flat_chain, "sampler": sampler,
            "a_rot": np.asarray(a_rot_raw), "b_rot": np.asarray(b_rot_raw),
            "cov_a_rot": cov_a_rot, "cov_b_rot": cov_b_rot,
            "cov_a": cov_a, "cov_b": cov_b,
            "R": R,
            "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
            "sigma_hat": float(sigma_hat_val),
        })

    result["elapsed_s"] = time.perf_counter() - t0
    return result
