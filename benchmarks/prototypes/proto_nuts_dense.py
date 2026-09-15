"""Prototype 5C.3: NUTS for the combined model with a dense mass matrix.

Runs the same window adaptation and sampling as sys_mapping.nuts.run_nuts on the fiducial
LS10 sample (log M* >= 10.0, NSIDE 32, footprint-standardised templates, rotated basis),
once with the diagonal mass matrix the package uses and once with a dense one.
"""
import glob
import importlib.util
import sys
import time
import warnings
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

import sys_mapping as sm
from sys_mapping.nuts import _logdensity_with_data
from sys_mapping.contamination import n_free_params

warnings.filterwarnings("ignore")
REPO = Path("/home/comparat/software/sys_mapping")
spec = importlib.util.spec_from_file_location("ls10", REPO / "scripts/run_ls10_analysis.py")
ls10 = importlib.util.module_from_spec(spec)
sys.modules["ls10"] = ls10
spec.loader.exec_module(ls10)

nside = int(sys.argv[1]) if len(sys.argv) > 1 else 32
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
n_chains = 4
cat = Path.home() / "data/legacysurvey/dr10/sweep/BGS_VLIM_Mstar"
data = glob.glob(str(cat / "LS10_VLIM_ANY_10.0_Mstar_*_DATA.fits"))[0]
tpl, _ = ls10.load_templates_from_dir(Path.home() / f"data/legacysurvey/dr10/systematics/{nside:04d}", nside)
rg, dg_ = ls10._read_positions(data)
rr, dr_ = ls10._read_positions(data.replace("_DATA", "_RAND"))
gal = sm.pixelize_catalog(rg, dg_, nside)
ran = sm.pixelize_catalog(rr, dr_, nside)
delta_g, good = sm.compute_overdensity(gal, ran)
delta_t = sm.standardise_on_footprint(sm.assign_template_values(tpl, good))
delta_t_rot, R, _ = sm.rotate_templates(delta_t)
n_sys = delta_t.shape[0]
print(f"NSIDE {nside}: {good.sum()} pixels, {n_sys} templates, {n_chains} chains, {steps}+{steps} steps",
      flush=True)

base, n_dim, idx_sigma = _logdensity_with_data(n_sys, "combined", False, None, None, None)
G, T = jnp.asarray(delta_g), jnp.asarray(delta_t_rot)
logdensity = lambda u: base(u, G, T)
n_cont = n_free_params(n_sys, "combined")
rng = np.random.default_rng(0)
u0 = np.zeros((n_chains, n_dim))
u0[:, :n_cont] = rng.normal(0, 0.05, (n_chains, n_cont))
u0[:, idx_sigma] = np.log(np.std(delta_g)) + rng.normal(0, 0.1, n_chains)
u0 = jnp.asarray(u0)
keys = jax.random.split(jax.random.PRNGKey(1), n_chains)


def run(diagonal):
    def one(key, init):
        warm = blackjax.window_adaptation(blackjax.nuts, logdensity, target_acceptance_rate=0.8,
                                          is_mass_matrix_diagonal=diagonal, progress_bar=False)
        wk, sk = jax.random.split(key)
        (state, params), _ = warm.run(wk, init, num_steps=steps)
        kernel = blackjax.nuts(logdensity, **params).step

        def step(s, k):
            s, info = kernel(k, s)
            return s, (s.position, info.is_divergent, info.num_integration_steps)

        _, out = jax.lax.scan(step, state, jax.random.split(sk, steps))
        return out

    t0 = time.perf_counter()
    pos, div, nint = jax.jit(jax.vmap(one))(keys, u0)
    pos = np.asarray(pos)
    wall = time.perf_counter() - t0
    ess = np.asarray(blackjax.diagnostics.effective_sample_size(pos, chain_axis=0, sample_axis=1))
    rhat = np.asarray(blackjax.diagnostics.potential_scale_reduction(pos, chain_axis=0, sample_axis=1))
    return dict(wall=wall, ess_min=float(ess.min()), ess_med=float(np.median(ess)),
                rhat=float(rhat.max()), div=int(np.sum(div)), steps=float(np.mean(nint)),
                median=np.median(pos.reshape(-1, n_dim), axis=0))


res = {}
for name, diag in (("diagonal", True), ("dense", False)):
    r = run(diag)
    res[name] = r
    print(f"{name:8s} wall {r['wall']:7.1f} s  min ESS {r['ess_min']:7.0f}  median ESS {r['ess_med']:7.0f}  "
          f"min ESS/s {r['ess_min'] / r['wall']:6.2f}  R-hat {r['rhat']:.3f}  divergences {r['div']}  "
          f"leapfrog steps/iteration {r['steps']:.1f}", flush=True)
d = np.abs(res["dense"]["median"] - res["diagonal"]["median"])
print(f"posterior medians agree to {d.max():.2e} (max over {n_dim} parameters)")
