"""Array-namespace dispatch for functions that serve both NumPy and JAX callers.

A function written against ``xp = namespace(*inputs)`` behaves exactly as NumPy code for
NumPy inputs, and traces under ``jax.jit``, ``jax.vmap`` and ``jax.grad`` for JAX inputs
(tracers are ``jax.Array`` instances).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def namespace(*arrays):
    """``jax.numpy`` if any argument is a JAX array or tracer, else ``numpy``."""
    return jnp if any(isinstance(a, jax.Array) for a in arrays) else np
