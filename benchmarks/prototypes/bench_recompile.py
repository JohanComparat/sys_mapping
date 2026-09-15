"""First- and repeat-call wall times of the JAX entry points, on new data of the same shape.

A repeat call that costs as much as the first is recompiling.
"""
import time
import warnings

import numpy as np

import sys_mapping as sm
from sys_mapping.nuts import run_nuts

warnings.filterwarnings("ignore")
N_PIX, N_SYS = 20000, 5


def field(seed):
    rng = np.random.default_rng(seed)
    t = rng.standard_normal((N_SYS, N_PIX))
    g = 0.02 * t[0] + rng.standard_normal(N_PIX) * 0.3
    return g, t


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    fn(*a, **k)
    return time.perf_counter() - t0


def lrt(seed):
    g, t = field(seed)
    a, *_ = np.linalg.lstsq(t.T, g, rcond=None)
    s = float(np.std(g - a @ t))
    sm.likelihood_ratio_test(g, t, sm.pack_params(a, None, s, model="additive"),
                             sm.pack_params(a, np.zeros(N_SYS), s, model="combined"),
                             "additive", "combined")


def refine(seed):
    g, t = field(seed)
    sm.refine_to_mle(sm.pack_params(np.zeros(N_SYS), np.zeros(N_SYS), 0.5, model="combined"),
                     g, t, model="combined", max_iter=50)


def nuts(seed):
    g, t = field(seed)
    run_nuts(N_SYS, model="combined", delta_g_obs=g, delta_t=t, n_chains=2,
             n_warmup=100, n_samples=100, seed=seed)


for name, fn in (("likelihood_ratio_test", lrt), ("refine_to_mle", refine), ("run_nuts", nuts)):
    times = [timed(fn, s) for s in range(4)]
    print(f"{name:22s} first {times[0]:7.3f} s   repeats {np.mean(times[1:]):7.3f} s "
          f"(min {min(times[1:]):.3f})")
