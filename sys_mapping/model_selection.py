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
from .contamination import n_free_params
import jax.numpy as jnp


@dataclass
class LikelihoodRatioResult:
    lambda_lr: float
    n_dof: int
    p_value: float
    reject_null: bool
    null_model: str
    alt_model: str


def likelihood_ratio_test(
    delta_g_obs: np.ndarray,
    delta_t: np.ndarray,
    theta_null: np.ndarray,
    theta_alt: np.ndarray,
    null_model: str,
    alt_model: str,
    use_skewed: bool = False,
    significance: float = 0.05,
) -> LikelihoodRatioResult:
    """Perform a likelihood ratio test comparing null vs. alternative model (Eq. 19).

    Computes ``λ_LR = 2 [ln L(θ̂_alt) − ln L(θ̂_null)]``, which under the null
    hypothesis is asymptotically ``χ²(r)`` where
    ``r = dim(alt) − dim(null)`` is the extra degrees of freedom.

    .. note::
       Each call creates new JIT-compiled likelihood functions internally.
       For repeated calls with the same data, precompute the log-likelihoods
       using :func:`~likelihood.make_log_likelihood` to avoid recompilation.

    Parameters
    ----------
    delta_g_obs : (n_pix,) observed overdensity
    delta_t : (n_sys, n_pix) template values
    theta_null : flat parameter vector for the null model (from :func:`~inference.get_mle_params`)
    theta_alt  : flat parameter vector for the alternative model
    null_model : str  ``'additive'`` or ``'multiplicative'`` (must be nested in alt_model)
    alt_model  : str  ``'combined'`` (must have more free parameters than null_model)
    use_skewed : bool
    significance : float  p-value threshold for rejecting the null; default 0.05

    Returns
    -------
    LikelihoodRatioResult
        ``lambda_lr`` : test statistic (≥ 0 only when both theta are at their MLEs)
        ``n_dof``     : degrees of freedom ``r = n_free_params(alt) − n_free_params(null)``
        ``p_value``   : ``Pr(χ²(r) > λ_LR)`` under the null
        ``reject_null``: True when ``p_value < significance``
        ``null_model``, ``alt_model``: model names

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

    p_value = float(chi2.sf(lambda_lr, df=r))
    reject_null = p_value < significance

    return LikelihoodRatioResult(
        lambda_lr=lambda_lr,
        n_dof=r,
        p_value=p_value,
        reject_null=reject_null,
        null_model=null_model,
        alt_model=alt_model,
    )
