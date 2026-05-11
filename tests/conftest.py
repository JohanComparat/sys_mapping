"""Shared pytest fixtures for sys_mapping tests.

Uses small synthetic data (n_pix=200, n_sys=3) for fast tests.
All random state is seeded for reproducibility.
"""

import numpy as np
import pytest
import jax.numpy as jnp


N_PIX = 200
N_SYS = 3
RNG_SEED = 7


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(RNG_SEED)


@pytest.fixture(scope="session")
def delta_t(rng):
    """(n_sys, n_pix) template maps, zero-mean unit-variance."""
    t = rng.standard_normal((N_SYS, N_PIX))
    t -= t.mean(axis=1, keepdims=True)
    t /= t.std(axis=1, keepdims=True)
    return t


@pytest.fixture(scope="session")
def delta_g_clean(rng):
    """(n_pix,) clean galaxy overdensity."""
    return rng.standard_normal(N_PIX) * 0.1


@pytest.fixture(scope="session")
def true_a():
    return np.array([0.05, -0.03, 0.02])


@pytest.fixture(scope="session")
def true_b():
    return np.array([0.04, 0.01, -0.02])


@pytest.fixture(scope="session")
def delta_g_obs(delta_g_clean, delta_t, true_a, true_b):
    """(n_pix,) contaminated overdensity."""
    from sys_mapping.contamination import apply_contamination
    return np.asarray(
        apply_contamination(
            jnp.asarray(delta_g_clean),
            jnp.asarray(delta_t),
            jnp.asarray(true_a),
            jnp.asarray(true_b),
        )
    )


@pytest.fixture(scope="session")
def sigma_true():
    return 0.12


@pytest.fixture(scope="session")
def theta_combined(true_a, true_b, sigma_true):
    """Flat parameter vector for the combined model."""
    from sys_mapping.contamination import pack_params
    return pack_params(true_a, true_b, sigma_true, model="combined")


@pytest.fixture(scope="session")
def theta_additive(true_a, sigma_true):
    from sys_mapping.contamination import pack_params
    return pack_params(true_a, np.zeros(N_SYS), sigma_true, model="additive")


@pytest.fixture(scope="session")
def theta_multiplicative(true_a, sigma_true):
    from sys_mapping.contamination import pack_params
    return pack_params(true_a, true_a, sigma_true, model="multiplicative")
