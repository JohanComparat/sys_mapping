"""Interleaved A/B of run_nuts: package version (A) vs cached whole-program jit (B)."""
import importlib.util, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sys_mapping.nuts import run_nuts as run_a
spec = importlib.util.spec_from_file_location("nuts_cached", sys.argv[1])
mod = importlib.util.module_from_spec(spec); sys.modules["nuts_cached"] = mod; spec.loader.exec_module(mod)
run_b = mod.run_nuts
N_PIX, N_SYS = 20000, 5
steps = int(sys.argv[2])

def field(seed):
    rng = np.random.default_rng(seed)
    t = rng.standard_normal((N_SYS, N_PIX))
    return 0.02 * t[0] + rng.standard_normal(N_PIX) * 0.3, t

res = {"A": [], "B": []}
for rep in range(4):
    for name, fn in ((("B", run_b), ("A", run_a)) if len(sys.argv) > 3 else (("A", run_a), ("B", run_b))):
        g, t = field(rep)
        t0 = time.perf_counter()
        chain, s = fn(N_SYS, model="combined", delta_g_obs=g, delta_t=t, n_chains=2,
                      n_warmup=steps, n_samples=steps, seed=rep)
        res[name].append(time.perf_counter() - t0)
        if rep == 0:
            print(name, "median posterior", np.round(np.median(chain, 0)[:3], 4), flush=True)
for name, ts in res.items():
    print(f"{name}: first {ts[0]:.2f} s, repeats {np.mean(ts[1:]):.2f} s (min {min(ts[1:]):.2f})", flush=True)
