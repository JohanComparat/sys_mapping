"""Regression-based systematic decontamination methods.

Implements:
  - ElasticNet regularised regression (Weaverdyck & Huterer 2021)
  - Iterative Systematics Decontamination (Elvin-Poole+2018, Rodríguez-Monroy+2022,
    Weaverdyck+2026): marginal binned polynomial fits, greedy, mock-calibrated stop
  - Iteratively reweighted polynomial OLS (the v1.2 method formerly named ISD)
  - Multi-method comparison framework (Weaverdyck & Huterer 2021)
  - Unified run_decontamination() interface for all 6 methods
"""

from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import dataclass
from typing import Callable

import re

import numpy as np

_VALID_METHODS = frozenset({"OLS", "ElasticNet", "ISD-1", "ISD-3", "MCMC-add", "MCMC-comb"})

# ISD at an arbitrary marginal-polynomial degree.  "ISD-1" and "ISD-3" are the
# DES Y1/Y3 and Y6 choices and stay in _VALID_METHODS as named methods; the
# pattern admits the rest so the degree can be swept without inventing a method
# name per degree.  Degree 0 is a constant, which is no correction at all.
_ISD_METHOD_RE = re.compile(r"^ISD-([1-9]\d*)$")

# Maximum weight allowed during ISD iterations and in the final ISD output.
# Pixels where 1 + a@t ≈ 0 would otherwise produce weights → 1/eps = 1e6,
# making all subsequent re-weighting steps meaningless.  A cap of 20 corresponds
# to a denominator floor of 0.05 — generous enough not to bias well-behaved
# scenarios while preventing numerical blow-up under strong contamination.
_ISD_MAX_WEIGHT: float = 20.0


def _as_float_or_none(value):
    """``float(value)``, or ``None`` when the quantity is not defined.

    A sampler diagnostic that does not apply must not be reported as a number:
    ``rhat = 1.0`` from an exact analytic draw is indistinguishable in a results
    table from ``rhat = 1.0`` measured on a converged chain.
    """
    if value is None:
        return None
    v = float(value)
    return None if not np.isfinite(v) else v


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


from functools import partial as _partial

import jax as _jax
import jax.numpy as _jnp

@_partial(_jax.jit, static_argnums=(4, 5))
def _isd_iterate_jax(X, y, ridge_diag, delta_t_lin, n_sys, max_iter, tol, w_max):
    """Fully-jitted ISD reweighting loop (``lax.while_loop``).

    Numerically equivalent to the NumPy iteration in
    :func:`polynomial_ols_decontamination`: the constant design matrix
    ``X`` becomes a JIT constant, so the whole fixed-point iteration compiles
    into a single kernel with no Python-level loop overhead.
    """
    n_pix, n_expanded = X.shape
    Xt = X.T
    eye = _jnp.eye(n_expanded)

    def cond(state):
        it, _weights, _alpha, rel = state
        return (it < max_iter) & (rel >= tol)

    def body(state):
        it, weights, _alpha, _rel = state
        Xw = Xt * weights                              # (n_expanded, n_pix)
        XtWX = Xw @ X + eye * ridge_diag               # ridge on the diagonal
        XtWy = Xw @ y
        alpha_new, *_ = _jnp.linalg.lstsq(XtWX, XtWy, rcond=None)
        field = alpha_new[:n_sys] @ delta_t_lin
        w_new = 1.0 / _jnp.maximum(1.0 + field, 1e-6)
        w_new = _jnp.clip(w_new, 1.0 / w_max, w_max)
        rel_new = _jnp.linalg.norm(w_new - weights) / (_jnp.linalg.norm(weights) + 1e-30)
        return (it + 1, w_new, alpha_new, rel_new)

    init = (0, _jnp.ones(n_pix), _jnp.zeros(n_expanded), _jnp.asarray(tol + 1.0))
    it, weights, alpha, _rel = _jax.lax.while_loop(cond, body, init)
    return weights, alpha, it


def elasticnet_contamination_fit(
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    *,
    l1_ratio: float = 0.5,
    alpha_reg: float | None = None,
    cv_folds: int = 5,
    max_iter: int = 10_000,
    fit_intercept: bool = False,
    patch_ids: np.ndarray | None = None,
    pixel_weights: np.ndarray | None = None,
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
    patch_ids:
        Spatial patch label per pixel (shape ``(n_pix,)``), from
        :func:`~sys_mapping.bootstrap.assign_spatial_patches`.  When given, the
        cross-validation folds are built from whole patches
        (``sklearn.model_selection.GroupKFold``) instead of arbitrary index
        chunks, so a held-out fold is spatially disjoint from the training set.
        This matters on a clustered field: with folds that mix spatially, the
        held-out pixels are correlated with the training pixels, the prediction
        error is under-estimated, and the cross-validation selects too weak a
        penalty.  DES Y6 uses ~200 compact patches of about 5 degrees diameter
        (Weaverdyck et al. 2026, Sec. III B).
    pixel_weights:
        Per-pixel weight in the likelihood (shape ``(n_pix,)``), passed to
        ``sample_weight``.  Use :func:`inverse_variance_pixel_weights` to build
        the DES Y6 form :math:`A_k^2 / (N_k + 2)`, which accounts for both partial
        coverage and Poisson scatter in the pixel counts.
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
    sw = None if pixel_weights is None else np.asarray(pixel_weights, dtype=float)

    cv = cv_folds
    if patch_ids is not None:
        from sklearn.model_selection import GroupKFold

        patch_ids = np.asarray(patch_ids)
        n_groups = int(np.unique(patch_ids).size)
        if n_groups < 2:
            warnings.warn(
                "patch_ids defines fewer than two patches; falling back to "
                f"{cv_folds}-fold index-chunk cross-validation.",
                UserWarning, stacklevel=2,
            )
        else:
            n_splits = min(cv_folds, n_groups)
            cv = list(GroupKFold(n_splits=n_splits).split(X, y, groups=patch_ids))

    cv_scores = None
    if alpha_reg is None:
        model = ElasticNetCV(
            l1_ratio=l1_ratio,
            cv=cv,
            max_iter=max_iter,
            fit_intercept=fit_intercept,
            n_jobs=-1,
        )
        model.fit(X, y, sample_weight=sw)
        alpha_reg = float(model.alpha_)
        cv_scores = model.mse_path_.mean(axis=-1)
    else:
        model = ElasticNet(
            alpha=alpha_reg,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            fit_intercept=fit_intercept,
        )
        model.fit(X, y, sample_weight=sw)

    alpha_hat = model.coef_  # (n_sys,)
    weights = np.clip(
        _compute_weights(alpha_hat, delta_t),
        1.0 / _ISD_MAX_WEIGHT, _ISD_MAX_WEIGHT,
    )
    cv_info = {
        "alpha_reg": alpha_reg,
        "l1_ratio": l1_ratio,
        "cv_scores": cv_scores,
        "cv_spatial": patch_ids is not None and not isinstance(cv, int),
    }
    return alpha_hat, weights, cv_info


def polynomial_ols_decontamination(
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    *,
    poly_order: int = 3,
    max_iter: int = 20,
    tol: float = 1e-5,
    lambda_poly: float = 0.0,
    backend: str = "numpy",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Iteratively reweighted OLS on a multivariate polynomial template basis.

    .. warning::
       **This is not ISD.**  Up to and including v1.2 this function was named
       ``iterative_systematics_decontamination`` and cited as the method of
       Rodriguez-Monroy et al.; it is a different algorithm.  It expands the
       templates into *all* monomials :math:`t_{i_1}\cdots t_{i_k}` with
       :math:`k \le` ``poly_order`` --- cross-products between different templates
       included, :math:`\\binom{n_s + k}{k} - 1` columns --- and fits them
       *simultaneously*.  Published ISD fits one template at a time against its own
       binned density relation; see
       :func:`iterative_systematics_decontamination` for that algorithm.

       The distinction is not cosmetic.  Only the first ``n_sys`` linear
       coefficients drive the weight, so at ``poly_order=3`` fifty-odd strongly
       collinear columns are fitted and discarded, inflating the variance of the
       coefficients that are actually used.  On LS10 this produced
       :math:`{\\rm rms}|\\hat a| \\simeq 7.6`, some 35x the OLS solution on the
       same data, and a weight map saturated across the whole footprint.

       It is kept because the v1.2 benchmark and timing results were produced with
       it and remain reproducible.  Do not use it for new analyses.

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
    backend:
        ``"numpy"`` (default) or ``"jax"``.  The ``"jax"`` path compiles the
        whole reweighting fixed-point into a single ``lax.while_loop`` kernel
        (numerically identical to ``"numpy"`` — verified to ``~1e-14``).  On CPU
        the NumPy BLAS path is typically faster (the per-iteration cost is a
        small BLAS-3 ``XᵀWX``); ``"jax"`` is provided for GPU execution and
        device-portable pipelines where the constant design matrix can stay
        resident on-device.

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
    >>> from sys_mapping import polynomial_ols_decontamination
    >>> rng = np.random.default_rng(0)
    >>> n_pix, n_sys = 8000, 3
    >>> delta_t = rng.standard_normal((n_sys, n_pix))
    >>> true_alpha = np.array([0.15, -0.08, 0.04])
    >>> delta_g = true_alpha @ delta_t + rng.standard_normal(n_pix) * 0.4
    >>> weights, alpha_all, n_it = polynomial_ols_decontamination(
    ...     delta_g, delta_t, poly_order=2, max_iter=10)
    >>> weights.shape
    (8000,)
    >>> np.all(weights > 0)
    True

    See Also
    --------
    iterative_systematics_decontamination : the published ISD algorithm.
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

    # Opt-in fully-jitted iteration (identical numerics, no Python loop overhead).
    if backend == "jax":
        w_j, a_j, it_j = _isd_iterate_jax(
            _jnp.asarray(X), _jnp.asarray(delta_g_obs),
            _jnp.asarray(ridge_diag), _jnp.asarray(delta_t_expanded[:n_sys]),
            int(n_sys), int(max_iter), float(tol), float(_ISD_MAX_WEIGHT),
        )
        return np.asarray(w_j), np.asarray(a_j), int(it_j)

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


@dataclass(frozen=True)
class ISDStep:
    """One accepted step of the ISD iteration.

    ``template``
        Index of the template selected at this step.
    ``significance``
        :math:`S = \\Delta\\chi^2 / \\Delta\\chi^2_{68}` at selection time.
    ``coeffs``
        Fitted polynomial coefficients, shape ``(poly_order + 1,)``, in ascending
        power order, so the intermediate weight is
        ``1 / (1 + sum_k coeffs[k] t**k)``.
    """

    template: int
    significance: float
    coeffs: np.ndarray


@dataclass(frozen=True)
class ISDResult:
    """Result of :func:`iterative_systematics_decontamination`.

    ``weights``
        Shape ``(n_pix,)``.  Product of the intermediate weights over all accepted steps.
    ``a_hat``
        Shape ``(n_sys,)``.  Linear contamination amplitude per template, summed
        over the steps that selected it.  Each contribution is the least-squares
        projection of the fitted curve onto the template,
        ``<F_i(t_i) t_i> / <t_i^2>`` -- the amplitude such that ``sum_i a_i t_i``
        reproduces the contamination actually removed, which is what the
        downstream two-point correction
        (:func:`~sys_mapping.correction.correct_two_point_function`) and amplitude
        bias (:func:`~sys_mapping.utils.compute_amplitude_bias`) assume.  For a
        linear fit on a standardised template it recovers the degree-1
        coefficient up to the effect of clipping the tails; for a cubic the two
        part company entirely, because the coefficients are large whenever the bin
        centres span a narrow range while the curve they describe is not.  Any
        curvature the fit found lives in ``steps``, not here.
    ``steps``
        The accepted :class:`ISDStep` records, in order.  Empty when no template ever crossed the
        threshold, in which case ``weights`` is identically one.
    ``significance``
        Shape ``(n_sys,)``.  Significance of every template at the *final* iteration, i.e. what is left
        after weighting.  All entries should be below ``threshold``.
    ``n_steps``
        ``len(steps)``.
    ``n_floored``
        Total number of pixel-steps where ``1 + F(t)`` had to be floored at
        ``1/w_max``.  Non-zero is normal in the tails; large means the template
        wants masking.
    ``stopped_on``
        ``"threshold"`` (every template insignificant), ``"max_steps"``, or
        ``"exhausted"`` (every template already selected ``max_reuse`` times).
    ``calibrated``
        Whether ``chi2_68`` came from mocks.  When ``False`` the threshold is in
        raw :math:`\\Delta\\chi^2` units and is not comparable to the DES value.
    """

    weights: np.ndarray
    a_hat: np.ndarray
    steps: list
    significance: np.ndarray
    n_steps: int
    stopped_on: str
    calibrated: bool
    n_floored: int = 0


def iterative_systematics_decontamination(
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    *,
    poly_order: int = 1,
    n_bins: int = 10,
    binning: str = "quantile",
    threshold: float = 2.0,
    chi2_68: np.ndarray | float | None = None,
    max_steps: int | None = None,
    max_reuse: int = 3,
    fracdet: np.ndarray | None = None,
    w_max: float = _ISD_MAX_WEIGHT,
    bad_pixel_frac: float = 0.01,
) -> ISDResult:
    """Iterative Systematics Decontamination (ISD).

    The published algorithm (Elvin-Poole et al. 2018 for DES Y1; Rodriguez-Monroy
    et al. 2022 for Y3; Weaverdyck et al. 2026 Sec. III B for Y6).  Each step fits
    the observed density *marginally* against one template at a time, weights the
    field by the inverse of the single most significant fit, and repeats on the
    reweighted field until no template is significant any more:

    .. math::

        \\hat F_i(t) = \\sum_{k \\le d} c_k^{(i)} t^k,
        \\qquad
        S_i = \\frac{\\Delta\\chi^2_i}{\\Delta\\chi^2_{68}},
        \\qquad
        w \\;\\leftarrow\\; \\frac{w}{1 + \\hat F_j(t_j)},
        \\quad j = \\arg\\max_i S_i,

    stopping when :math:`\\max_i S_i < ` ``threshold``.

    ``poly_order`` is the degree of the polynomial **in a single template's
    value** --- ``1`` reproduces DES Y1/Y3, ``3`` the DES Y6 choice.  It is not a
    multivariate polynomial order: the design matrix of each fit is
    ``(n_bins, poly_order+1)`` however many templates there are.  This is what
    makes the method well conditioned, and it is the difference from
    :func:`polynomial_ols_decontamination`, which is what this package computed
    under this name up to v1.2.

    Parameters
    ----------
    delta_g_obs:
        Observed galaxy overdensity at footprint pixels (shape ``(n_pix,)``).
    delta_t:
        Template maps at footprint pixels (shape ``(n_sys, n_pix)``).
    poly_order:
        Degree of the 1D marginal fit.  ``1`` (linear) or ``3`` (cubic).
    n_bins:
        Number of template-value bins per marginal fit.
    binning:
        ``"quantile"`` (default) or ``"width"``; see
        :func:`~sys_mapping.diagnostics.isd_marginal_fit`.
    threshold:
        Stopping threshold on :math:`S_i`.  DES Y6 uses ``2.0`` with a
        mock-calibrated ``chi2_68``.
    chi2_68:
        68th percentile of :math:`\\Delta\\chi^2` on contamination-free mocks,
        either one value per template (shape ``(n_sys,)``) or a scalar.  Obtain it
        from :func:`~sys_mapping.diagnostics.isd_template_significance` as
        ``np.percentile(result["delta_chi2_mocks"], 68, axis=0)``.  When ``None``
        the significance is raw :math:`\\Delta\\chi^2` and the threshold is
        **uncalibrated** --- a warning is issued, because the DES value of 2 then
        means nothing.
    max_steps:
        Hard cap on accepted steps.  ``None`` (default) uses ``4 * n_sys``.
    max_reuse:
        Maximum number of times one template may be selected.  A template that
        keeps being picked after this many corrections is not being fitted, it is
        being chased.
    fracdet:
        Per-pixel fractional coverage weights (shape ``(n_pix,)``).
    w_max:
        Clip for the running weight, ``[1/w_max, w_max]``.  Guards a fit that
        drives ``1 + F(t)`` through zero in some pixel.
    bad_pixel_frac:
        Tolerance on the fraction of pixels whose ``1 + F(t)`` falls below
        ``1/w_max``.  Below it the denominator is floored and the step applied;
        above it the step is refused and the template retired, because a fit that
        inverts the field over a percent of the footprint is not a perturbative
        correction.  Default ``0.01``.

    Returns
    -------
    :class:`ISDResult`

    Notes
    -----
    The fitted polynomial is evaluated at ``clip(t, t_lo, t_hi)``, where
    ``t_lo``/``t_hi`` are the outermost bin centres the fit was constrained by.
    Survey-property maps are strongly skewed --- LS10's ``GALDEPTH_Z`` reaches
    +26 standardised units while its outermost bin centre sits near +2 --- so an
    unclipped cubic would be evaluated three orders of magnitude beyond its
    support and diverge, which it does: without clipping the iteration on real
    templates re-selects the same templates with *rising* significance instead of
    converging.  Holding ``F`` constant outside the fitted range is the
    conservative reading of a binned fit.

    Precision
    ---------
    Each step is a weighted least-squares solve on a ``(n_bins, poly_order+1)``
    Vandermonde --- condition number of order ``n_bins**poly_order`` at worst,
    against the ``binom(n_s+3, 3)``-column collinear system the v1.2 code
    inverted.  The iteration is finite by construction: ``max_steps`` and
    ``max_reuse`` both bound it, independently of the threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import iterative_systematics_decontamination
    >>> rng = np.random.default_rng(0)
    >>> n_pix, n_sys = 20000, 3
    >>> delta_t = rng.standard_normal((n_sys, n_pix))
    >>> delta_g = 0.15 * delta_t[1] + rng.standard_normal(n_pix) * 0.3
    >>> res = iterative_systematics_decontamination(
    ...     delta_g, delta_t, chi2_68=50.0)
    >>> res.steps[0].template
    1
    >>> bool(abs(res.a_hat[1] - 0.15) < 0.02)
    True

    References
    ----------
    Elvin-Poole et al. 2018, PRD 98, 042006.
    Rodriguez-Monroy et al. 2022, MNRAS 511, 2665.
    Weaverdyck et al. 2026, arXiv:2601.14484, Sec. III B.

    See Also
    --------
    polynomial_ols_decontamination : the v1.2 algorithm this replaces.
    sys_mapping.diagnostics.isd_marginal_fit : the per-step fit.
    """
    from .diagnostics import isd_marginal_fit

    delta_g_obs = np.asarray(delta_g_obs, dtype=float)
    delta_t = np.atleast_2d(np.asarray(delta_t, dtype=float))
    n_sys, n_pix = delta_t.shape
    order = int(poly_order)

    calibrated = chi2_68 is not None
    if calibrated:
        norm = np.broadcast_to(np.asarray(chi2_68, dtype=float), (n_sys,)).copy()
        norm[norm <= 0] = np.inf     # an all-zero mock null can never be exceeded
    else:
        warnings.warn(
            "ISD: chi2_68 not supplied, so the stopping threshold is in raw "
            "Delta chi^2 units and is not the DES-calibrated statistic. Pass "
            "chi2_68 from isd_template_significance for a meaningful threshold.",
            UserWarning, stacklevel=2,
        )
        norm = np.ones(n_sys)

    if max_steps is None:
        max_steps = 4 * n_sys

    weights = np.ones(n_pix)
    a_hat = np.zeros(n_sys)
    steps: list[ISDStep] = []
    n_used = np.zeros(n_sys, dtype=int)
    n_floored_total = 0
    stopped_on = "max_steps"

    powers = np.arange(order + 1)
    significance = np.zeros(n_sys)

    for _ in range(max_steps):
        # The density as it currently stands, i.e. after every weight applied so
        # far.  ISD refits on the *corrected* field, which is what makes it
        # iterative rather than a one-shot marginal regression.
        delta_now = (1.0 + delta_g_obs) * weights - 1.0

        dchi2, coeffs, t_range = isd_marginal_fit(
            delta_now, delta_t, n_bins=n_bins, poly_order=order,
            fracdet=fracdet, binning=binning,
        )
        significance = dchi2 / norm
        # A template already corrected max_reuse times is out of the running.
        eligible = np.where(n_used < max_reuse, significance, -np.inf)

        if not np.any(np.isfinite(eligible)):
            stopped_on = "exhausted"
            break

        j = int(np.argmax(eligible))
        s_j = float(eligible[j])
        if s_j < threshold:
            stopped_on = "threshold"
            break

        c = np.asarray(coeffs[j], dtype=float)
        # Evaluate the fit only where it was fitted.  The polynomial is
        # constrained by ten bin centres; outside them it is an extrapolation,
        # and survey-property maps are skewed enough for that to matter a great
        # deal -- LS10's GALDEPTH_Z reaches +26 standardised units while its
        # outermost bin centre sits near +2, so an unclipped cubic is evaluated
        # 10^3 times beyond its support and diverges. Holding F constant outside
        # the fitted range is the conservative reading of a binned fit.
        t_lo, t_hi = float(t_range[j, 0]), float(t_range[j, 1])
        t_eval = np.clip(delta_t[j], t_lo, t_hi)
        f_j = (t_eval[:, None] ** powers[None, :]) @ c
        denom = 1.0 + f_j

        # Residual guard, for a fit that is bad *within* its own range -- the
        # clipping above has already removed the extrapolation.  Floor the
        # denominator so a handful of pixels cannot invert the field, but refuse
        # the step outright if a sizeable fraction needs it, because a
        # "correction" that inverts a percent of the footprint is not
        # perturbative and the template wants masking, not fitting.
        n_floored = int(np.sum(denom < 1.0 / w_max))
        frac_floored = n_floored / n_pix
        if frac_floored > bad_pixel_frac:
            warnings.warn(
                f"ISD: the fit for template {j} drives 1 + F(t) below "
                f"{1.0 / w_max:.3g} in {n_floored}/{n_pix} pixels "
                f"({100 * frac_floored:.2f}%), above the {100 * bad_pixel_frac:.2f}% "
                "tolerance; step refused and template retired. Mask the extreme "
                "values of this template (see footprint_mask_diagnostics) rather "
                "than fitting through them.",
                UserWarning, stacklevel=2,
            )
            n_used[j] = max_reuse
            continue
        if n_floored:
            n_floored_total += n_floored

        weights = np.clip(weights / np.maximum(denom, 1.0 / w_max),
                          1.0 / w_max, w_max)
        # Report the linear amplitude of the correction that was *applied*, as
        # the least-squares projection of the fitted curve onto the template,
        # rather than the degree-1 polynomial coefficient.  The two track each
        # other for a linear fit on a standardised template, but for a cubic they
        # do not: bin centres of a skewed template span a narrow range, so the
        # Vandermonde is poorly conditioned and the coefficients are large
        # (c_3 ~ 4 is routine) even though the curve itself is small.  The
        # projection is invariant to that conditioning, and it is the quantity
        # the downstream two-point correction assumes -- the amplitude a_i such
        # that sum_i a_i t_i reproduces the contamination removed.
        tt = float(np.dot(delta_t[j], delta_t[j]))
        a_hat[j] += float(np.dot(f_j, delta_t[j]) / tt) if tt > 0 else 0.0
        n_used[j] += 1
        steps.append(ISDStep(template=j, significance=s_j, coeffs=c))

    if steps and stopped_on == "max_steps":
        # The loop ran out of steps rather than breaking, so `significance` is
        # from before the last accepted step.  Recompute it, since callers read
        # it as "what is left after weighting".
        delta_now = (1.0 + delta_g_obs) * weights - 1.0
        significance = isd_marginal_fit(
            delta_now, delta_t, n_bins=n_bins, poly_order=order,
            fracdet=fracdet, binning=binning,
        )[0] / norm

    return ISDResult(
        weights=weights,
        a_hat=a_hat,
        steps=steps,
        significance=significance,
        n_steps=len(steps),
        stopped_on=stopped_on,
        calibrated=calibrated,
        n_floored=n_floored_total,
    )


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

    def _chi2_of_clean(delta_g_clean: np.ndarray) -> float:
        """Mean square of the recovered clean field: what the model leaves over."""
        return float(np.sum(np.asarray(delta_g_clean) ** 2) / n_pix)

    def _chi2(alpha: np.ndarray) -> float:
        """Additive-model residual.  Not valid for the combined model.

        For the combined model the clean field is
        ``(delta_g_obs - a.t) / (1 + b.t)``, not ``delta_g_obs - a.t``: using the
        additive form there ignores ``b`` entirely and reports a number that does
        not correspond to the fit it is labelling.  That branch calls
        :func:`_chi2_of_clean` with the field it already inverts.
        """
        return _chi2_of_clean(delta_g_obs - alpha @ delta_t)

    if "ols" in methods:
        X = delta_t.T  # (n_pix, n_sys)
        alpha_ols, *_ = np.linalg.lstsq(X, delta_g_obs, rcond=None)
        weights_ols = np.clip(
            _compute_weights(alpha_ols, delta_t),
            1.0 / _ISD_MAX_WEIGHT, _ISD_MAX_WEIGHT,
        )
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
        from .inference import run_mcmc, posterior_median_params
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
        theta_hat = posterior_median_params(flat_chain)
        n_sys = delta_t.shape[0]
        a_hat, b_hat, _, _ = unpack_params(theta_hat, n_sys, "combined", use_skewed=False)
        from .contamination import invert_contamination as _inv_cont
        import jax.numpy as _jnp
        _dg_clean = np.asarray(_inv_cont(
            _jnp.asarray(delta_g_obs), _jnp.asarray(delta_t),
            _jnp.asarray(a_hat), _jnp.asarray(b_hat),
        ))
        weights_comb = np.clip(
            (1.0 + _dg_clean) / np.maximum(1.0 + delta_g_obs, 1e-6),
            1.0 / _ISD_MAX_WEIGHT, _ISD_MAX_WEIGHT,
        )
        results["combined_mcmc"] = {
            "a_hat": np.asarray(a_hat),
            "b_hat": np.asarray(b_hat),
            "weights": weights_comb,
            # The combined model's own residual, using both a_hat and b_hat.
            "chi2_residual": _chi2_of_clean(_dg_clean),
        }

    if "additive_mcmc" in methods:
        flat_chain, _ = run_mcmc(
            n_sys=delta_t.shape[0],
            model="additive",
            delta_g_obs=delta_g_obs,
            delta_t=delta_t,
            **kw,
        )
        theta_hat = posterior_median_params(flat_chain)
        n_sys = delta_t.shape[0]
        a_hat, _, _, _ = unpack_params(theta_hat, n_sys, "additive", use_skewed=False)
        weights_mcmc = np.clip(
            _compute_weights(a_hat, delta_t),
            1.0 / _ISD_MAX_WEIGHT, _ISD_MAX_WEIGHT,
        )
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
    n_steps: int = 1200,
    n_burn: int = 200,
    seed: int = 42,
    use_skewed: bool = False,
    progress: bool = False,
    # Sampler backend for MCMC-add / MCMC-comb
    sampler: str = "auto",
    n_chains: int | None = None,
    nuts_n_warmup: int = 1000,
    nuts_n_samples: int = 1000,
    pixel_precision=None,
    cv_folds: int = 5,
    # ISD (see iterative_systematics_decontamination)
    isd_n_bins: int = 10,
    isd_binning: str = "quantile",
    isd_threshold: float = 2.0,
    isd_chi2_68: np.ndarray | float | None = None,
    isd_max_steps: int | None = None,
    isd_max_reuse: int = 3,
    isd_fracdet: np.ndarray | None = None,
    isd_poly_order: int | None = None,
    isd_w_max: float = _ISD_MAX_WEIGHT,
    isd_bad_pixel_frac: float = 0.01,
    # Pre-selection (ignored when preselect=False)
    preselect: bool = False,
    preselect_method: str = "isd",
    preselect_n_top: int | None = None,
    preselect_p_threshold: float | None = 0.05,
    preselect_n_mocks: int = 100,
    preselect_seed: int = 0,
    preselect_rand_factor: int = 2,
    preselect_n_jobs: int = 1,
    preselect_cl_input: np.ndarray | None = None,
    preselect_cl_amplitude: float | None = None,
    # Required when preselect=True and preselect_method="isd"
    good_pixels: np.ndarray | None = None,
    n_total_footprint: int | None = None,
    z_edges: np.ndarray | None = None,
    nz: np.ndarray | None = None,
    nside: int | None = None,
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
    sampler:
        Sampling backend for ``MCMC-add`` / ``MCMC-comb``.  One of:

        * ``"auto"`` (default): additive → exact analytic Normal-Inverse-Gamma
          posterior (:func:`~inference.run_additive_analytic`); combined / skew →
          gradient-based BlackJAX NUTS (:func:`~nuts.run_nuts`).
        * ``"analytic"``: force the analytic posterior (additive only; falls back
          to NUTS for combined / skew).
        * ``"nuts"``: force BlackJAX NUTS.
        * ``"emcee"``: force the legacy gradient-free emcee sampler
          (:func:`~inference.run_mcmc`) — kept as the validation baseline.
    n_chains:
        Number of parallel NUTS chains; ``None`` picks from the JAX backend
        (CPU: 4, GPU: 8).  Ignored for analytic / emcee.
    nuts_n_warmup, nuts_n_samples:
        NUTS window-adaptation steps and post-warmup draws per chain.
    pixel_precision:
        Optional :class:`~covariance.LowRankPrecision` correlated-noise (GLS) pixel
        correlation ``R`` (``C = σ² R``) for ``MCMC-add`` / ``MCMC-comb``.  The pixel
        likelihood otherwise assumes independent pixels (``σ² I``), which underestimates
        the posterior width when the clean field is spatially correlated.  ``None``
        (default) keeps the white likelihood.  Supplying it forces the NUTS sampler
        (the exact analytic additive posterior exists only for white noise).
    cv_folds:
        Cross-validation folds for ElasticNet (when ``alpha_reg`` is auto).
    isd_n_bins:
        Number of template-value bins per ISD marginal fit (default 10, the DES
        value).
    isd_binning:
        ``"quantile"`` (default, equal occupancy) or ``"width"`` for the ISD bins.
    isd_threshold:
        ISD stopping threshold on :math:`S = \\Delta\\chi^2/\\Delta\\chi^2_{68}`.
        Default ``2.0``, the DES Y6 value; meaningful only when ``isd_chi2_68`` is
        supplied.
    isd_chi2_68:
        Mock-calibrated normalisation of the ISD significance, per template or
        scalar.  When ``None`` and ``preselect=True`` with
        ``preselect_method="isd"``, it is taken from the GLASS null that
        pre-selection already generated; otherwise the ISD threshold is
        uncalibrated and a warning is issued.
    isd_max_steps:
        Cap on accepted ISD steps.  ``None`` uses ``4 * n_sys``.
    isd_fracdet:
        Per-pixel fractional coverage weights for the ISD binned means.
    preselect:
        If ``True``, run Stage 1 SNR pre-selection before decontamination.
        The ``delta_t`` passed to the decontamination method will be the
        filtered subset.
    preselect_method:
        SNR ranking method for Stage 1: ``"data"``, ``"template"``, or
        ``"isd"`` (default).
    preselect_n_top:
        Keep only the top-K templates.  ``None`` keeps all that pass the
        p-value threshold (ISD only).
    preselect_p_threshold:
        ISD mock p-value threshold (default 0.05).  Templates with
        ``p > threshold`` are removed.  Set to ``None`` to disable p-filtering.
    preselect_n_mocks:
        Number of GLASS mocks for ISD significance (default 100).
    preselect_seed:
        Random seed for GLASS mock generation.
    preselect_rand_factor:
        Ratio of randoms to galaxies in each GLASS mock (default 2).
    preselect_cl_input:
        The sample's matched spectrum for the ISD pre-selection null, typically from
        :func:`~sys_mapping.glass_mocks.load_matched_cl`.  Without it, and without
        ``preselect_cl_amplitude``, the null is the default power law and the mock
        generator warns that it is not calibrated.
    preselect_cl_amplitude:
        Amplitude of a parametric null spectrum, to choose one deliberately.
    preselect_n_jobs:
        Parallel worker processes for the GLASS mock loop in ISD pre-selection
        (default 1 = serial; -1 = all cores).
    good_pixels:
        Boolean footprint mask, shape ``(12 * nside**2,)``.  Required when
        ``preselect=True`` and ``preselect_method="isd"``.
    n_total_footprint:
        Number of galaxies in the survey footprint (e.g. ``len(ra_gal)``).
        Used to scale the GLASS mock surface density to match the data.
    z_edges, nz:
        Redshift histogram for GLASS mock generation.
    nside:
        HEALPix resolution.  Auto-derived from ``good_pixels`` if ``None``.

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
    Pre-selection (when ``preselect=True``)
        ``preselect_indices`` (list[int]), ``preselect_result``
        (:class:`~model_selection.SnrPreselectionResult`),
        ``preselect_isd`` (dict or ``None``).
    Non-applicable keys are ``None``.

    Notes
    -----
    **Weight convention**

    * OLS / ElasticNet / ISD-1 / ISD-3 / MCMC-add:
      ``w(p) = 1 / (1 + a_hat @ t(p))`` — additive weight.
    * MCMC-comb:
      ``w(p) = (1 + δ_g_clean(p)) / (1 + δ_g_obs(p))`` — exact pixel-level
      inverse of the contamination weight; cancels ``weight_cont`` exactly
      when (a_hat, b_hat) equal the true parameters.

    References
    ----------
    Berlfein et al. 2024, MNRAS 531, 4954.
    Weaverdyck & Huterer 2021, MNRAS 503, 5061.
    Rodríguez-Monroy et al. 2025, arXiv:2509.07943.
    """
    if method not in _VALID_METHODS and not _ISD_METHOD_RE.match(method):
        raise ValueError(
            f"method must be one of {sorted(_VALID_METHODS)} or 'ISD-<degree>', "
            f"got {method!r}"
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
        "acceptance_fraction": None, "sigma_hat": None, "gamma_hat": None,
        # ISD-only
        "n_iterations": None,
        "isd_outlier_mask": None,    # bool (n_pix,): True = pixel excluded in pass 2
        "isd_masked_fraction": None, # float: fraction of pixels excluded in pass 2
        # ElasticNet-only
        "cv_info": None,
        # Pre-selection (None when preselect=False)
        "preselect_indices": None,
        "preselect_result": None,
        "preselect_isd": None,
    }

    # ── Pre-selection (Stage 1) ───────────────────────────────────────────────
    isd_sig = None
    if preselect:
        from .diagnostics import isd_template_significance
        from .model_selection import snr_preselect

        pre = snr_preselect(delta_g_obs, delta_t,
                            method=preselect_method,
                            n_top=preselect_n_top)
        selected = list(pre.selected_indices)

        if preselect_method == "isd" and preselect_p_threshold is not None:
            _ns = nside or int(round(np.sqrt(len(good_pixels) / 12)))
            isd_sig = isd_template_significance(
                delta_g_obs, delta_t[selected], good_pixels, _ns,
                n_total=0,  # overridden by n_total_footprint below
                z_edges=z_edges, nz=nz,
                n_total_footprint=n_total_footprint,
                n_mocks=preselect_n_mocks, seed=preselect_seed,
                rand_factor=preselect_rand_factor,
                n_jobs=preselect_n_jobs,
                cl_input=preselect_cl_input,
                cl_amplitude=preselect_cl_amplitude,
            )
            kept_pos = [k for k, p in enumerate(isd_sig["p_values"])
                        if p <= preselect_p_threshold] or [0]
            selected = [selected[k] for k in kept_pos]
            result["preselect_isd"] = isd_sig
            # The null was drawn for every candidate; the ISD threshold below must
            # line up with the templates that survived the cut.
            isd_sig = dict(isd_sig,
                           delta_chi2_mocks=np.asarray(isd_sig["delta_chi2_mocks"])[:, kept_pos])
        else:
            isd_sig = None

        delta_t = delta_t[selected]
        n_sys   = delta_t.shape[0]
        result["preselect_indices"] = selected
        result["preselect_result"]  = pre

    # ── OLS ───────────────────────────────────────────────────────────────────
    if method == "OLS":
        X = delta_t.T  # (n_pix, n_sys)
        a_hat, *_ = np.linalg.lstsq(X, delta_g_obs, rcond=None)
        b_hat = np.zeros(n_sys)
        weights = np.clip(
            _compute_weights(a_hat, delta_t),
            1.0 / _ISD_MAX_WEIGHT, _ISD_MAX_WEIGHT,
        )
        result.update({"a_hat": a_hat, "b_hat": b_hat, "weights": weights})

    # ── ElasticNet ────────────────────────────────────────────────────────────
    elif method == "ElasticNet":
        a_hat, weights, cv_info = elasticnet_contamination_fit(
            delta_g_obs, delta_t, cv_folds=cv_folds
        )
        b_hat = np.zeros(n_sys)
        result.update({"a_hat": a_hat, "b_hat": b_hat, "weights": weights,
                        "cv_info": cv_info})

    # ── ISD-d ─────────────────────────────────────────────────────────────────
    elif method.startswith("ISD-"):
        # The order is carried by the method name so the six-method comparison
        # can name its columns, but `isd_poly_order` overrides it: a sweep over
        # the order should not have to invent a method name per degree.
        poly_order = int(method.split("-", 1)[1])
        if isd_poly_order is not None:
            poly_order = int(isd_poly_order)

        # Mock calibration of the stopping rule.  Stage-1 pre-selection, when it
        # ran with method="isd", already generated the contamination-free GLASS
        # null this needs, so reuse it rather than paying for it twice.
        chi2_68 = isd_chi2_68
        if chi2_68 is None and preselect and isd_sig is not None:
            chi2_68 = np.percentile(isd_sig["delta_chi2_mocks"], 68, axis=0)

        isd_res = iterative_systematics_decontamination(
            delta_g_obs, delta_t,
            poly_order=poly_order,
            n_bins=isd_n_bins,
            binning=isd_binning,
            threshold=isd_threshold,
            chi2_68=chi2_68,
            max_steps=isd_max_steps,
            max_reuse=isd_max_reuse,
            fracdet=isd_fracdet,
            w_max=isd_w_max,
            bad_pixel_frac=isd_bad_pixel_frac,
        )

        a_hat = np.asarray(isd_res.a_hat, dtype=float)
        weights = np.asarray(isd_res.weights, dtype=float)
        b_hat = np.zeros(n_sys)

        # Guard: non-finite coefficients (residual edge case).
        if not (np.all(np.isfinite(a_hat)) and np.all(np.isfinite(weights))):
            warnings.warn(
                f"{method} produced non-finite output; a_hat zeroed, weights set "
                "to ones.",
                stacklevel=2,
            )
            a_hat = np.zeros(n_sys)
            weights = np.ones(n_pix)

        result.update({
            "a_hat": a_hat, "b_hat": b_hat, "weights": weights,
            "n_iterations": isd_res.n_steps,
            "isd_steps": [
                {"template": st.template,
                 "significance": st.significance,
                 "coeffs": np.asarray(st.coeffs)}
                for st in isd_res.steps
            ],
            "isd_significance": isd_res.significance,
            "isd_stopped_on": isd_res.stopped_on,
            "isd_calibrated": isd_res.calibrated,
            "isd_n_floored": isd_res.n_floored,
            "isd_poly_order": poly_order,
        })

    # ── MCMC-add / MCMC-comb ──────────────────────────────────────────────────
    else:
        from .inference import posterior_median_params, get_param_covariance_from_chain
        from .contamination import unpack_params
        from .correction import rotate_templates, transform_params_from_rotated

        mcmc_model = "additive" if method == "MCMC-add" else "combined"
        _use_skewed = (use_skewed if mcmc_model == "combined" else False)

        # PCA rotation — deterministic for fixed delta_t
        delta_t_rot, R, _eigenvalues = rotate_templates(delta_t)

        # ── Sampler dispatch ──────────────────────────────────────────────
        # "auto": exact analytic posterior for the linear-Gaussian additive
        # model, gradient-based NUTS for the non-linear combined / skew models.
        _sampler = sampler
        if _sampler == "auto":
            _sampler = "analytic" if (mcmc_model == "additive" and not _use_skewed) else "nuts"
        elif _sampler == "analytic" and (mcmc_model != "additive" or _use_skewed):
            _sampler = "nuts"  # analytic posterior only exists for additive Gaussian
        if pixel_precision is not None and _sampler != "nuts":
            _sampler = "nuts"  # GLS correlated-noise likelihood is only wired through NUTS

        if _sampler == "analytic":
            from .inference import run_additive_analytic
            flat_chain, sampler_obj = run_additive_analytic(
                n_sys, delta_g_obs=delta_g_obs, delta_t=delta_t_rot, seed=seed,
            )
        elif _sampler == "nuts":
            from .nuts import run_nuts
            flat_chain, sampler_obj = run_nuts(
                n_sys, model=mcmc_model,
                delta_g_obs=delta_g_obs, delta_t=delta_t_rot,
                use_skewed=_use_skewed,
                n_chains=n_chains, n_warmup=nuts_n_warmup, n_samples=nuts_n_samples,
                seed=seed, progress=progress, precision=pixel_precision,
            )
        else:  # "emcee" — legacy gradient-free baseline
            from .inference import run_mcmc
            n_dim = n_sys + 1 if mcmc_model == "additive" else 2 * n_sys + 1
            nw = max(n_walkers, 2 * n_dim + 2)  # emcee requires >= 2*n_dim+2 walkers
            flat_chain, sampler_obj = run_mcmc(
                n_sys=n_sys, model=mcmc_model,
                delta_g_obs=delta_g_obs, delta_t=delta_t_rot,
                n_walkers=nw, n_steps=n_steps, n_burn=n_burn,
                seed=seed, progress=progress,
                use_skewed=_use_skewed,
            )

        theta_hat = posterior_median_params(flat_chain)
        a_rot_raw, b_rot_raw, sigma_hat_val, gamma_hat_val = unpack_params(
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

        # Exact pixel-level decontamination weight:
        #   w(p) = (1 + δ_g_clean(p)) / (1 + δ_g_obs(p))
        # where δ_g_clean = invert_contamination(δ_g_obs, δ_t, a_hat, b_hat).
        # This is the exact inverse of the contamination weight and cancels
        # weight_cont perfectly when (a_hat, b_hat) = (a_true, b_true).
        # The 1/(1+α·t) approximation overcorrects when noisy b_hat≠0 in
        # additive scenarios — the exact inverse avoids that bias.
        from .contamination import invert_contamination
        import jax.numpy as _jnp
        _delta_g_clean = np.asarray(invert_contamination(
            _jnp.asarray(delta_g_obs),
            _jnp.asarray(delta_t),
            _jnp.asarray(a_hat),
            _jnp.asarray(b_hat),
        ))
        _denom = np.maximum(1.0 + delta_g_obs, 1e-6)
        weights = np.clip(
            (1.0 + _delta_g_clean) / _denom,
            1.0 / _ISD_MAX_WEIGHT, _ISD_MAX_WEIGHT,
        )

        result.update({
            "a_hat": a_hat, "b_hat": b_hat, "weights": weights,
            "flat_chain": flat_chain, "sampler": sampler_obj,
            "a_rot": np.asarray(a_rot_raw), "b_rot": np.asarray(b_rot_raw),
            "cov_a_rot": cov_a_rot, "cov_b_rot": cov_b_rot,
            "cov_a": cov_a, "cov_b": cov_b,
            "R": R,
            "acceptance_fraction": float(np.mean(sampler_obj.acceptance_fraction)),
            "sigma_hat": float(sigma_hat_val),
            # Skewness, when the skew-normal likelihood was used.  A caller that
            # refines this fit to an MLE needs it to pack a starting point.
            "gamma_hat": (None if gamma_hat_val is None else float(gamma_hat_val)),
            "sampler_backend": _sampler,
            # None where the diagnostic does not apply (exact i.i.d. draws) rather
            # than a perfect score, which would read as a passed check.
            "rhat": _as_float_or_none(getattr(sampler_obj, "rhat", None)),
            "ess": _as_float_or_none(getattr(sampler_obj, "ess", None)),
            "num_divergences": int(getattr(sampler_obj, "num_divergences", 0)),
        })

    result["elapsed_s"] = time.perf_counter() - t0
    return result
