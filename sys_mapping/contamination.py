"""Contamination model for galaxy overdensity systematics.

Implements the forward model (Eq. 11-13 of Berlfein et al. 2024):
    delta_g_obs = delta_g * (1 + b·delta_t) + a·delta_t

Three nested models:
  - 'additive':       b = 0, free params: [a_0,...,a_{N-1}]
  - 'multiplicative': a = 0, free params: [b_0,...,b_{N-1}]
  - 'combined':       a, b free, params: [a_0,...,a_{N-1}, b_0,...,b_{N-1}]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import Array

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Parameter layout helpers
# ---------------------------------------------------------------------------

def n_free_params(n_sys: int, model: str) -> int:
    """Number of contamination parameters (excluding sigma / gamma).

    Parameters
    ----------
    n_sys : int
        Number of systematic templates.
    model : str
        One of ``'additive'``, ``'multiplicative'``, or ``'combined'``.

    Returns
    -------
    int
        Number of free contamination parameters (not counting sigma or gamma).

    Precision
    ---------
    Exact integer arithmetic; no floating-point error.

    Examples
    --------
    >>> from sys_mapping import n_free_params
    >>> n_free_params(4, "additive")
    4
    >>> n_free_params(4, "multiplicative")
    4
    >>> n_free_params(4, "combined")
    8
    """
    match model:
        case "additive":
            return n_sys
        case "multiplicative":
            return n_sys
        case "combined":
            return 2 * n_sys
        case _:
            raise ValueError(f"Unknown model '{model}'. Choose additive, multiplicative, or combined.")


def pack_params(
    a: np.ndarray | Array,
    b: np.ndarray | Array | None,
    sigma: float,
    gamma: float | None = None,
    model: str = "combined",
) -> np.ndarray:
    """Pack (a, b, sigma[, gamma]) into a flat parameter vector.

    Parameters
    ----------
    a : ``(n_sys,)`` additive coefficients
    b : ``(n_sys,)`` or None  multiplicative coefficients (required for
        ``combined`` and for ``multiplicative``, where they are the free parameters)
    sigma : float  noise standard deviation
    gamma : float or None  skewness parameter (appended last when provided)
    model : str  ``'additive'``, ``'multiplicative'``, or ``'combined'``

    Returns
    -------
    (n_dim,) flat numpy array suitable for emcee / scipy optimizers

    Precision
    ---------
    Pure concatenation; values are preserved to machine precision (``float64``).

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import pack_params
    >>> a = np.array([0.05, -0.03, 0.02])
    >>> b = np.array([0.04,  0.01, -0.02])
    >>> theta = pack_params(a, b, sigma=0.12, model="combined")
    >>> theta  # [a0, a1, a2, b0, b1, b2, sigma]
    array([ 0.05, -0.03,  0.02,  0.04,  0.01, -0.02,  0.12])
    >>> pack_params(a, None, 0.12, gamma=1.5, model="additive")
    array([ 0.05, -0.03,  0.02,  0.12,  1.5 ])
    """
    if model == "multiplicative":
        if b is None:
            raise ValueError(
                "multiplicative model requires b: its free parameters are the "
                "multiplicative coefficients, and a is identically zero")
        parts: list[np.ndarray] = [np.asarray(b)]
    else:
        parts = [np.asarray(a)]
        if model == "combined":
            if b is None:
                raise ValueError("combined model requires b")
            parts.append(np.asarray(b))
    parts.append(np.array([sigma]))
    if gamma is not None:
        parts.append(np.array([gamma]))
    return np.concatenate(parts)


def unpack_params(
    theta: np.ndarray | Array,
    n_sys: int,
    model: str,
    use_skewed: bool = False,
) -> tuple[Array, Array, Array, Array | None]:
    """Unpack flat parameter vector into (a, b, sigma, gamma).

    Parameters
    ----------
    theta : (n_dim,) flat parameter vector from :func:`pack_params`
    n_sys : int  number of systematic templates
    model : str  ``'additive'``, ``'multiplicative'``, or ``'combined'``
    use_skewed : bool  if True, extract gamma from the last element

    Returns
    -------
    a : ``(n_sys,)`` additive coefficients (JAX array)
    b : ``(n_sys,)`` multiplicative coefficients; zeros for additive
    sigma : scalar noise standard deviation
    gamma : scalar skewness parameter or None

    Precision
    ---------
    Slice/view operations; values are preserved to machine precision.

    Examples
    --------
    >>> import numpy as np, jax.numpy as jnp
    >>> from sys_mapping import pack_params, unpack_params
    >>> a = np.array([0.05, -0.03])
    >>> b = np.array([0.04,  0.01])
    >>> theta = pack_params(a, b, sigma=0.12, model="combined")
    >>> a2, b2, sigma2, _ = unpack_params(jnp.asarray(theta), n_sys=2, model="combined")
    >>> float(sigma2)
    0.12
    """
    theta = jnp.asarray(theta)
    match model:
        case "additive":
            a = theta[:n_sys]
            b = jnp.zeros(n_sys)
            rest = theta[n_sys:]
        case "multiplicative":
            # Purely multiplicative: the templates modulate the density and do
            # not also add to it.  The free parameters ARE the b's.
            a = jnp.zeros(n_sys)
            b = theta[:n_sys]
            rest = theta[n_sys:]
        case "combined":
            a = theta[:n_sys]
            b = theta[n_sys : 2 * n_sys]
            rest = theta[2 * n_sys :]
        case _:
            raise ValueError(f"Unknown model '{model}'.")

    sigma = rest[0]
    gamma = rest[1] if use_skewed else None
    return a, b, sigma, gamma


# ---------------------------------------------------------------------------
# Forward / inverse contamination
# ---------------------------------------------------------------------------

def apply_contamination(
    delta_g: Array,
    delta_t: Array,
    a: Array,
    b: Array,
) -> Array:
    """Apply contamination to clean overdensity (Eq. 12).

    Computes ``δ_g_obs = δ_g * (1 + Σ b_i δ_{t,i}) + Σ a_i δ_{t,i}``.

    Parameters
    ----------
    delta_g : (n_pix,) true galaxy overdensity
    delta_t : ``(n_sys, n_pix)`` systematic templates
    a : ``(n_sys,)`` additive coefficients
    b : ``(n_sys,)`` multiplicative coefficients

    Returns
    -------
    (n_pix,) observed (contaminated) overdensity

    Performance
    -----------
    Measured on CPU (JAX, n_pix=10_000, n_sys=5): **~830 μs/call** after warmup.
    Scales as O(n_sys × n_pix).

    Precision
    ---------
    Forward–inverse roundtrip recovers ``delta_g`` with max absolute error < 1e-5
    (float32 accumulation). RMS error < 1e-6 for n_pix ≥ 50_000.

    Examples
    --------
    >>> import numpy as np, jax.numpy as jnp
    >>> from sys_mapping import apply_contamination
    >>> rng = np.random.default_rng(0)
    >>> delta_g = jnp.asarray(rng.standard_normal(1000) * 0.1)
    >>> delta_t = jnp.asarray(rng.standard_normal((2, 1000)))
    >>> a = jnp.array([0.05, -0.03])
    >>> b = jnp.array([0.04,  0.01])
    >>> delta_g_obs = apply_contamination(delta_g, delta_t, a, b)
    >>> delta_g_obs.shape
    (1000,)
    """
    mult_term = jnp.einsum("i,ij->j", b, delta_t)
    add_term = jnp.einsum("i,ij->j", a, delta_t)
    return delta_g * (1.0 + mult_term) + add_term


def invert_contamination(
    delta_g_obs: Array,
    delta_t: Array,
    a: Array,
    b: Array,
) -> Array:
    """Remove contamination from observed overdensity (Eq. 12 inverted).

    Computes ``δ_g = (δ_g_obs − Σ a_i δ_{t,i}) / (1 + Σ b_i δ_{t,i})``.

    Parameters
    ----------
    delta_g_obs : (n_pix,) observed galaxy overdensity
    delta_t : ``(n_sys, n_pix)`` systematic templates
    a : ``(n_sys,)`` additive coefficients
    b : ``(n_sys,)`` multiplicative coefficients

    Returns
    -------
    (n_pix,) cleaned galaxy overdensity

    Performance
    -----------
    Measured on CPU (JAX, n_pix=10_000, n_sys=5): **~1730 μs/call** after warmup.
    Slightly slower than :func:`apply_contamination` due to the element-wise division.

    Precision
    ---------
    Roundtrip (apply → invert) max absolute error < 1e-5; RMS < 1e-6 for large n_pix.
    Assumes ``1 + b·δ_t ≠ 0`` at every pixel (satisfied for small \|b\|).

    Examples
    --------
    >>> import numpy as np, jax.numpy as jnp
    >>> from sys_mapping import apply_contamination, invert_contamination
    >>> rng = np.random.default_rng(0)
    >>> delta_g = jnp.asarray(rng.standard_normal(1000) * 0.1)
    >>> delta_t = jnp.asarray(rng.standard_normal((2, 1000)))
    >>> a = jnp.array([0.05, -0.03])
    >>> b = jnp.array([0.04,  0.01])
    >>> obs = apply_contamination(delta_g, delta_t, a, b)
    >>> recovered = invert_contamination(obs, delta_t, a, b)
    >>> float(jnp.max(jnp.abs(recovered - delta_g)))  # < 1e-5
    """
    mult_term = jnp.einsum("i,ij->j", b, delta_t)
    add_term = jnp.einsum("i,ij->j", a, delta_t)
    return (delta_g_obs - add_term) / (1.0 + mult_term)


# ---------------------------------------------------------------------------
# Two-point correction (Eq. 15-16)
# ---------------------------------------------------------------------------

def compute_two_point_correction(
    w_obs: Array,
    a_sq: Array,
    b_sq: Array,
    template_correlations: Array,
) -> Array:
    """Apply two-point function correction (Eq. 15-16).

    Computes ``w_corr = (w_obs − Σ a²_i ξ_i(θ)) / (1 + Σ b²_i ξ_i(θ))``,
    where ``ξ_i(θ) = ⟨δ_{t,i} δ_{t,i}⟩(θ)`` is the template auto-correlation.

    Parameters
    ----------
    w_obs : (n_bins,) observed angular correlation function
    a_sq : ``(n_sys,)`` debiased squared additive coefficients
    b_sq : ``(n_sys,)`` debiased squared multiplicative coefficients
    template_correlations : ``(n_sys, n_bins)`` template auto-correlations per
        angular bin, or ``(n_sys, n_sys, n_bins)`` for the full correlation matrix
        including cross terms.  With the 3-D form, ``a_sq``/``b_sq`` must be the
        debiased *matrices* from
        :func:`~sys_mapping.correction.debias_params_matrix`.

    Returns
    -------
    (n_bins,) corrected angular correlation function

    Performance
    -----------
    Measured on CPU (JAX, n_sys=5, n_bins=15): **~576 μs/call** after warmup.

    Precision
    ---------
    Matches the analytic formula to ``rtol=1e-6`` (single float32 einsum).

    Examples
    --------
    >>> import numpy as np, jax.numpy as jnp
    >>> from sys_mapping import compute_two_point_correction
    >>> n_sys, n_bins = 3, 15
    >>> rng = np.random.default_rng(0)
    >>> w_obs   = jnp.asarray(rng.standard_normal(n_bins) * 0.01)
    >>> a_sq    = jnp.asarray(np.abs(rng.standard_normal(n_sys)) * 1e-3)
    >>> b_sq    = jnp.asarray(np.abs(rng.standard_normal(n_sys)) * 1e-3)
    >>> tcorr   = jnp.asarray(np.abs(rng.standard_normal((n_sys, n_bins))) * 1e-3)
    >>> w_corr  = compute_two_point_correction(w_obs, a_sq, b_sq, tcorr)
    >>> w_corr.shape
    (15,)
    """
    tc = jnp.asarray(template_correlations)
    a_sq = jnp.asarray(a_sq)
    b_sq = jnp.asarray(b_sq)
    # The amplitude rank must match the correlation rank: vectors go with autos,
    # matrices with the full correlation matrix.  Mixing them is a silent
    # broadcast in numpy semantics and an opaque einsum error in jax, so name it.
    if a_sq.ndim + 1 != tc.ndim or b_sq.ndim + 1 != tc.ndim:
        raise ValueError(
            f"amplitude and correlation ranks disagree: a_sq{tuple(a_sq.shape)}, "
            f"b_sq{tuple(b_sq.shape)}, template_correlations{tuple(tc.shape)}. "
            "Use 1-D a_sq/b_sq (from debias_params) with (n_sys, n_bins) autos, "
            "or 2-D (from debias_params_matrix) with (n_sys, n_sys, n_bins)."
        )
    if tc.ndim == 3:
        # Full form: sum_ij A_ij xi_ij(theta), with A the debiased outer product
        # from debias_params_matrix.  The PCA rotation diagonalises the template
        # covariance at zero lag only, so xi_ij(theta) != 0 for theta > 0 and the
        # cross terms are a 7-17% effect on the LS10 basis.
        add_bias = jnp.einsum("ij,ijk->k", a_sq, tc)
        mult_bias = jnp.einsum("ij,ijk->k", b_sq, tc)
    elif tc.ndim == 2:
        # Auto-only form: keeps the diagonal alone.  Retained for compatibility;
        # pass the full (n_sys, n_sys, n_bins) matrix to include the cross terms.
        add_bias = jnp.einsum("i,ij->j", a_sq, tc)
        mult_bias = jnp.einsum("i,ij->j", b_sq, tc)
    else:
        raise ValueError(
            f"template_correlations must be (n_sys, n_bins) for the auto-only "
            f"correction or (n_sys, n_sys, n_bins) for the full one; got shape "
            f"{tuple(tc.shape)}"
        )
    return (w_obs - add_bias) / (1.0 + mult_bias)


# ---------------------------------------------------------------------------
# Non-linear template response (injection only)
# ---------------------------------------------------------------------------

#: Response shapes.  The first three lie inside the basis a degree-3 marginal
#: polynomial can represent exactly; the last three do not, and exist to find the
#: boundary of the method rather than to flatter it.
RESPONSE_KINDS: tuple[str, ...] = (
    "linear", "quadratic", "cubic", "tanh", "threshold", "exp",
)


@dataclass(frozen=True)
class TemplateResponse:
    """A per-template contamination response :math:`F(t)`.

    The response is applied as a *selection efficiency*,

    .. math::

        1 + \\delta_{\\rm obs} = (1 + \\delta_g)\\,\\prod_i \\bigl(1 + F_i(t_i)\\bigr),

    which is the model
    :func:`~sys_mapping.regression.iterative_systematics_decontamination` inverts,
    its weight being :math:`\\prod_j 1/(1 + \\hat F_j)`.  It is **not**
    :func:`apply_contamination`'s model; the two coincide only when every
    :math:`F` is linear and ``a = b``.  Measuring what that mismatch costs is one
    of the things this class exists for.

    ``kind`` is one of :data:`RESPONSE_KINDS`.  ``shape`` is the shape parameter,
    whose meaning it sets: the curvature :math:`\kappa` for
    ``quadratic``/``cubic``, the rate :math:`\alpha` for ``tanh``/``exp``, the cut
    :math:`t_0` for ``threshold``, and nothing for ``linear``.  ``amplitude`` is
    the target ``rms(F(t))`` over the footprint; every response is rescaled to hit
    it (see :func:`evaluate_response`), so shapes are compared at equal injected
    power rather than at equal nominal coefficient.  Without that normalisation a
    ranking of shapes is a ranking of amplitudes.

    Examples
    --------
    >>> import numpy as np
    >>> from sys_mapping import TemplateResponse, evaluate_response
    >>> rng = np.random.default_rng(0)
    >>> t = rng.standard_normal(20000)
    >>> f = evaluate_response(TemplateResponse("cubic", 0.5, 0.05), t)
    >>> bool(abs(float(np.sqrt(np.mean(f**2))) - 0.05) < 1e-12)
    True
    """

    kind: str
    shape: float
    amplitude: float

    def __post_init__(self) -> None:
        if self.kind not in RESPONSE_KINDS:
            raise ValueError(
                f"unknown response kind {self.kind!r}; expected one of "
                f"{', '.join(RESPONSE_KINDS)}"
            )

    def to_header_dict(self, i: int) -> dict:
        """FITS-header-safe record for template ``i``."""
        return {f"RESP{i}": self.kind, f"RSHP{i}": float(self.shape),
                f"RAMP{i}": float(self.amplitude)}


def _response_unnormalised(kind: str, shape: float, t: np.ndarray) -> np.ndarray:
    """Shape of ``F`` before the rms rescaling.  ``t`` is standardised."""
    if kind == "linear":
        return t
    if kind == "quadratic":
        return t + shape * t ** 2
    if kind == "cubic":
        return t + shape * t ** 3
    if kind == "tanh":
        # Saturating: a completeness that plateaus.  alpha -> 0 recovers `linear`.
        a = float(shape)
        return np.tanh(a * t) / a if a != 0.0 else t
    if kind == "threshold":
        # A depth cut: no response below t0, linear above.  The derivative is
        # discontinuous, so no polynomial represents it on the bin scale.
        return np.where(t > shape, t - shape, 0.0)
    if kind == "exp":
        a = float(shape)
        return np.expm1(a * t) / a if a != 0.0 else t
    raise ValueError(f"unknown response kind {kind!r}")


def evaluate_response(response: TemplateResponse, t: np.ndarray) -> np.ndarray:
    """Evaluate :math:`F(t)`, rescaled so ``rms(F) == response.amplitude``.

    Parameters
    ----------
    response:
        The response to evaluate.
    t:
        Standardised template values at the footprint pixels, shape ``(n_pix,)``.

    Returns
    -------
    ``(n_pix,)`` array with ``rms`` equal to ``response.amplitude``.

    Notes
    -----
    The rescaling is what makes the shapes comparable: ``cubic`` with
    :math:`\\kappa = 0.5` on a Gaussian template has roughly twice the raw
    variance of ``linear``, so an unnormalised comparison would report the cubic
    as harder to correct when it is merely larger.  Normalising the *injected*
    power leaves the shape as the only difference.

    A response is centred before rescaling, because a constant offset in ``F`` is
    absorbed by the overall density normalisation of
    :func:`~sys_mapping.maps.compute_overdensity` and is not contamination the
    fit can or should see.
    """
    t = np.asarray(t, dtype=float)
    f = _response_unnormalised(response.kind, float(response.shape), t)
    f = f - f.mean()
    rms = float(np.sqrt(np.mean(f ** 2)))
    if rms <= 0.0:
        return np.zeros_like(t)
    return f * (float(response.amplitude) / rms)


def apply_nonlinear_contamination(
    delta_g: np.ndarray,
    delta_t: np.ndarray,
    responses: "Sequence[TemplateResponse | None]",
    *,
    min_efficiency: float = 0.05,
) -> np.ndarray:
    """Inject a template response that need not be linear.

    Computes

    .. math::

        1 + \\delta_{\\rm obs} = (1 + \\delta_g)\\,\\prod_i \\bigl(1 + F_i(t_i)\\bigr)

    with each :math:`F_i` given by :func:`evaluate_response`.  This is the
    selection-efficiency model ISD inverts, and the reason it is a separate
    function from :func:`apply_contamination` rather than a generalisation of it:
    the flat parameter vector of :func:`pack_params`, the Jacobian in
    :mod:`sys_mapping.likelihood`, and the closed-form two-point correction of
    :func:`compute_two_point_correction` are all tied to one coefficient per
    template per branch, and none of them has an analogue for a general
    :math:`F`.  This function is for *injecting* contamination into mocks; the
    inference models are unchanged.

    Parameters
    ----------
    delta_g:
        Clean galaxy overdensity, shape ``(n_pix,)``.
    delta_t:
        Standardised templates, shape ``(n_sys, n_pix)``.
    responses:
        One :class:`TemplateResponse` per template, or ``None`` for a template
        left uncontaminated.  ``None`` entries are what make greedy template
        *selection* measurable: without a true negative there is nothing for a
        selection rule to get wrong.
    min_efficiency:
        Floor on the product :math:`\\prod_i (1 + F_i)`.  A ``cubic`` or ``exp``
        response on a skewed template drives the efficiency through zero in the
        tail, and :func:`~sys_mapping.glass_mocks.sample_positions_from_delta`
        requires :math:`1 + \\delta \\ge 0`.  Flooring is the conservative
        reading: an efficiency of zero is a masked pixel, not a negative count.

    Returns
    -------
    ``(n_pix,)`` contaminated overdensity.

    See Also
    --------
    apply_contamination : the linear model of Eq. 13, used by the inference path.
    evaluate_response : the per-template response, and its normalisation.
    """
    delta_g = np.asarray(delta_g, dtype=float)
    delta_t = np.atleast_2d(np.asarray(delta_t, dtype=float))
    if len(responses) != delta_t.shape[0]:
        raise ValueError(
            f"got {len(responses)} responses for {delta_t.shape[0]} templates"
        )

    efficiency = np.ones(delta_t.shape[1], dtype=float)
    for i, resp in enumerate(responses):
        if resp is None:
            continue
        efficiency *= 1.0 + evaluate_response(resp, delta_t[i])
    np.maximum(efficiency, float(min_efficiency), out=efficiency)
    return (1.0 + delta_g) * efficiency - 1.0
