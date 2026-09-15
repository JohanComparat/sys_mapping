"""Prototype 5C.2: the likelihood-ratio null from batched maxima instead of per-mock NUTS.

For each uncontaminated mock the null statistic is 2[max log L_comb - max log L_add].
The additive maximum is closed-form (OLS, sigma = rms residual).  The combined maximum is
found by L-BFGS (optax) started from (a_ols, b=0, sigma), run for a fixed number of
iterations under jit and vmapped over all mocks at once.

Compared against the path scripts/run_ls10_analysis.build_lrt_null takes today: a NUTS
fit per model per mock, then refine_to_mle of the posterior medians.
"""
import importlib.util
import sys
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax

import sys_mapping as sm

warnings.filterwarnings("ignore")


def batched_lrt_null(mock_delta_g, delta_t, n_iter=200):
    """(n_mock,) lambda_LR = 2[max L_comb - max L_add] for mocks of shape (n_mock, n_pix)."""
    n_sys, n_pix = delta_t.shape
    T = jnp.asarray(delta_t)
    G = jnp.asarray(mock_delta_g)
    ll_add = sm.make_log_likelihood(n_sys, "additive")
    ll_comb = sm.make_log_likelihood(n_sys, "combined")
    i_sig = 2 * n_sys

    proj = jnp.linalg.pinv(T.T)                                  # (n_sys, n_pix)

    def one(g):
        a = proj @ g
        sig = jnp.sqrt(jnp.mean((g - a @ T) ** 2))
        l_add = ll_add(jnp.concatenate([a, sig[None]]), g, T)
        u0 = jnp.concatenate([a, jnp.zeros(n_sys), jnp.log(sig)[None]])

        def neg(u):
            return -ll_comb(u.at[i_sig].set(jnp.exp(u[i_sig])), g, T)

        opt = optax.lbfgs()
        vg = optax.value_and_grad_from_state(neg)

        def step(carry, _):
            u, state = carry
            value, grad = vg(u, state=state)
            updates, state = opt.update(grad, state, u, value=value, grad=grad, value_fn=neg)
            return (optax.apply_updates(u, updates), state), None

        (u, _), _ = jax.lax.scan(step, (u0, opt.init(u0)), None, length=n_iter)
        l_comb = jnp.maximum(-neg(u), -neg(u0))
        return 2.0 * (l_comb - l_add), jnp.max(jnp.abs(jax.grad(neg)(u)))

    return jax.jit(jax.vmap(one))(G)


def reference_lrt_null(mock_delta_g, delta_t, nuts_warmup, nuts_samples):
    """Today's path: NUTS per model per mock, then refine_to_mle of the medians."""
    spec = importlib.util.spec_from_file_location(
        "ls10", "/home/comparat/software/sys_mapping/scripts/run_ls10_analysis.py")
    ls10 = importlib.util.module_from_spec(spec)
    sys.modules["ls10"] = ls10
    spec.loader.exec_module(ls10)

    def fit_theta(model, dg, dt):
        method = "MCMC-add" if model == "additive" else "MCMC-comb"
        res = sm.run_decontamination(method, dg, dt, sampler="auto", nuts_n_warmup=nuts_warmup,
                                     nuts_n_samples=nuts_samples, n_chains=2)
        if model == "additive":
            th = sm.pack_params(np.asarray(res["a_hat"]), None, float(res["sigma_hat"]),
                                model="additive")
            return sm.refine_to_mle(th, dg, dt, model="additive")
        th = sm.pack_params(np.asarray(res["a_hat"]), np.asarray(res["b_hat"]),
                            float(res["sigma_hat"]), model="combined")
        return sm.refine_to_mle(th, dg, dt, model="combined")

    return sm.lrt_null_distribution(np.asarray(mock_delta_g).T, delta_t, fit_theta)


if __name__ == "__main__":
    nside, n_mock = int(sys.argv[1]), int(sys.argv[2])
    warm = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    import healpy as hp
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(1)
    good = np.abs(hp.pix2ang(nside, np.arange(npix), lonlat=True)[1]) > 25
    n_sys = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    T = np.array([hp.smoothing(rng.standard_normal(npix), fwhm=np.radians(f))[good]
                  for f in np.linspace(4, 20, n_sys)])
    T = sm.standardise_on_footprint(T)
    mocks = np.array([hp.smoothing(rng.standard_normal(npix), fwhm=np.radians(6))[good] * 0.3
                      + rng.standard_normal(good.sum()) * 0.2 for _ in range(n_mock)])
    print(f"NSIDE {nside}: {good.sum()} pixels, {n_sys} templates, {n_mock} mocks", flush=True)

    t0 = time.perf_counter()
    lam_b, gmax = batched_lrt_null(mocks, T)
    lam_b = np.asarray(lam_b)
    t_b = time.perf_counter() - t0
    t0 = time.perf_counter()
    lam_b2, _ = batched_lrt_null(mocks[::-1], T)
    t_b2 = time.perf_counter() - t0
    print(f"batched: {t_b:.1f} s first, {t_b2:.1f} s repeat; max |grad| at the end "
          f"{float(np.max(gmax)):.2e}", flush=True)

    t0 = time.perf_counter()
    lam_r = reference_lrt_null(mocks, T, warm, warm)
    t_r = time.perf_counter() - t0
    print(f"reference (NUTS {warm}+{warm} + refine): {t_r:.1f} s", flush=True)
    print("lambda batched  :", np.round(lam_b, 3))
    print("lambda reference:", np.round(lam_r, 3))
    print("max |diff|:", float(np.max(np.abs(lam_b - lam_r))),
          " relative:", float(np.max(np.abs(lam_b - lam_r) / np.maximum(np.abs(lam_r), 1e-9))))
