"""Tests for the analytic additive posterior and BlackJAX NUTS samplers,
plus the sampler dispatch in run_decontamination.

The additive model is linear-Gaussian, so run_additive_analytic must reproduce
the OLS MLE and an emcee chain's posterior to Monte-Carlo error.  NUTS must
recover injected (a, b) and agree with emcee, with healthy convergence.
"""

import numpy as np
import pytest
import jax.numpy as jnp

from sys_mapping.inference import (
    run_additive_analytic,
    run_mcmc,
    get_mle_params,
    get_param_covariance_from_chain,
)
from sys_mapping.nuts import run_nuts, default_n_chains
from sys_mapping.contamination import apply_contamination
from sys_mapping import run_decontamination


N_PIX = 4000


@pytest.fixture(scope="module")
def additive_data():
    rng = np.random.default_rng(11)
    n_sys = 3
    dt = rng.standard_normal((n_sys, N_PIX))
    dt -= dt.mean(1, keepdims=True)
    a_true = np.array([0.05, -0.03, 0.02])
    dg = a_true @ dt + rng.normal(0, 0.1, N_PIX)
    return dg, dt, a_true


@pytest.fixture(scope="module")
def combined_data():
    rng = np.random.default_rng(12)
    n_sys = 3
    dt = rng.standard_normal((n_sys, N_PIX))
    dt -= dt.mean(1, keepdims=True)
    dt /= dt.std(1, keepdims=True)
    a_true = np.array([0.05, -0.03, 0.02])
    b_true = np.array([0.04, 0.0, -0.02])
    dg_clean = rng.normal(0, 0.1, N_PIX)
    dg_obs = np.asarray(apply_contamination(
        jnp.asarray(dg_clean), jnp.asarray(dt), jnp.asarray(a_true), jnp.asarray(b_true)))
    return dg_obs, dt, a_true, b_true


class TestAnalyticAdditive:
    def test_shape_and_layout(self, additive_data):
        dg, dt, _ = additive_data
        n_sys = dt.shape[0]
        chain, sampler = run_additive_analytic(n_sys, delta_g_obs=dg, delta_t=dt, n_samples=5000)
        assert chain.shape == (5000, n_sys + 1)
        assert np.all(chain[:, -1] > 0)  # sigma column strictly positive

    def test_diagnostics_are_perfect(self, additive_data):
        dg, dt, _ = additive_data
        _, sampler = run_additive_analytic(dt.shape[0], delta_g_obs=dg, delta_t=dt, n_samples=2000)
        assert float(np.mean(sampler.acceptance_fraction)) == 1.0
        assert sampler.rhat == 1.0
        assert sampler.num_divergences == 0
        assert sampler.ess == 2000.0

    def test_median_matches_ols(self, additive_data):
        dg, dt, _ = additive_data
        n_sys = dt.shape[0]
        chain, _ = run_additive_analytic(n_sys, delta_g_obs=dg, delta_t=dt, n_samples=50000)
        a_ols = np.linalg.lstsq(dt.T, dg, rcond=None)[0]
        a_med = get_mle_params(chain)[:n_sys]
        assert np.allclose(a_med, a_ols, atol=5e-4)

    def test_reproducible(self, additive_data):
        dg, dt, _ = additive_data
        c1, _ = run_additive_analytic(dt.shape[0], delta_g_obs=dg, delta_t=dt, n_samples=1000, seed=7)
        c2, _ = run_additive_analytic(dt.shape[0], delta_g_obs=dg, delta_t=dt, n_samples=1000, seed=7)
        assert np.array_equal(c1, c2)

    @pytest.mark.slow
    def test_matches_emcee_posterior(self, additive_data):
        dg, dt, _ = additive_data
        n_sys = dt.shape[0]
        chain_a, _ = run_additive_analytic(n_sys, delta_g_obs=dg, delta_t=dt, n_samples=60000, seed=1)
        chain_e, _ = run_mcmc(n_sys=n_sys, model="additive", delta_g_obs=dg, delta_t=dt,
                              n_walkers=40, n_steps=1500, n_burn=500, seed=1, progress=False)
        cov_a, _ = get_param_covariance_from_chain(chain_a, n_sys, "additive")
        cov_e, _ = get_param_covariance_from_chain(chain_e, n_sys, "additive")
        # posterior means agree to MC error, std agree to ~10%
        assert np.allclose(get_mle_params(chain_a)[:n_sys],
                           get_mle_params(chain_e)[:n_sys], atol=1e-3)
        assert np.allclose(np.sqrt(np.diag(cov_a)), np.sqrt(np.diag(cov_e)), rtol=0.15)


class TestNuts:
    def test_flat_chain_shape(self, combined_data):
        dg, dt, _, _ = combined_data
        n_sys = dt.shape[0]
        chain, sampler = run_nuts(n_sys, model="combined", delta_g_obs=dg, delta_t=dt,
                                  n_chains=2, n_warmup=300, n_samples=400, seed=3)
        assert chain.shape == (2 * 400, 2 * n_sys + 1)
        assert np.all(chain[:, -1] > 0)  # sigma positive

    def test_recovers_truth_and_converges(self, combined_data):
        dg, dt, a_true, b_true = combined_data
        n_sys = dt.shape[0]
        chain, sampler = run_nuts(n_sys, model="combined", delta_g_obs=dg, delta_t=dt,
                                  n_chains=2, n_warmup=500, n_samples=800, seed=3)
        mle = get_mle_params(chain)
        cov_a, cov_b = get_param_covariance_from_chain(chain, n_sys, "combined")
        # additive coefficients recovered within 4 sigma (b is weakly constrained)
        assert np.all(np.abs(mle[:n_sys] - a_true) < 4 * np.sqrt(np.diag(cov_a)) + 1e-3)
        assert sampler.rhat < 1.1
        assert sampler.num_divergences == 0
        assert 0.5 < sampler.acceptance_fraction <= 1.0

    @pytest.mark.slow
    def test_matches_emcee(self, combined_data):
        dg, dt, _, _ = combined_data
        n_sys = dt.shape[0]
        chain_n, _ = run_nuts(n_sys, model="combined", delta_g_obs=dg, delta_t=dt,
                              n_chains=4, n_warmup=800, n_samples=1500, seed=3)
        chain_e, _ = run_mcmc(n_sys=n_sys, model="combined", delta_g_obs=dg, delta_t=dt,
                              n_walkers=40, n_steps=3000, n_burn=1000, seed=3, progress=False)
        # a is tightly constrained: means agree to MC error
        assert np.allclose(get_mle_params(chain_n)[:n_sys],
                           get_mle_params(chain_e)[:n_sys], atol=2e-3)


class TestChainMethod:
    """`chain_method='sequential'` is the low-memory route to multi-chain R-hat.

    `vmap` holds every chain's NUTS trajectory live at once, which OOMs on large
    footprints and forces n_chains=1 (where R-hat is undefined).  `lax.map` runs
    one chain at a time; same maths, so results must be identical.
    """

    def test_sequential_matches_vmap_statistically(self, combined_data):
        """Both routes sample the same posterior (not the same draws).

        Draws are deliberately not compared elementwise: batching the NUTS
        trajectory ``while_loop`` under ``vmap`` changes XLA's floating-point
        reduction order, and NUTS is chaotic — an O(1e-16) gradient difference
        snowballs into a different trajectory within a few hundred leapfrog
        steps.  The posterior it targets is what must agree.
        """
        dg, dt, a_true, _ = combined_data
        n_sys = dt.shape[0]
        kw = dict(model="combined", delta_g_obs=dg, delta_t=dt,
                  n_chains=2, n_warmup=400, n_samples=800, seed=3)
        chain_v, samp_v = run_nuts(n_sys, chain_method="vmap", **kw)
        chain_s, samp_s = run_nuts(n_sys, chain_method="sequential", **kw)
        mle_v, mle_s = get_mle_params(chain_v), get_mle_params(chain_s)
        # tightly-constrained additive coefficients agree to Monte-Carlo error
        assert np.allclose(mle_v[:n_sys], mle_s[:n_sys], atol=3e-3)
        # sigma (last column) is very tightly constrained
        assert mle_s[-1] == pytest.approx(mle_v[-1], rel=5e-2)
        assert samp_s.num_divergences == 0 and samp_v.num_divergences == 0

    def test_sequential_gives_multichain_rhat(self, combined_data):
        dg, dt, _, _ = combined_data
        n_sys = dt.shape[0]
        chain, sampler = run_nuts(n_sys, model="combined", delta_g_obs=dg, delta_t=dt,
                                  n_chains=4, n_warmup=300, n_samples=400, seed=3,
                                  chain_method="sequential")
        assert chain.shape == (4 * 400, 2 * n_sys + 1)
        assert np.isfinite(sampler.rhat) and sampler.rhat < 1.1
        assert sampler.n_chains == 4

    def test_rejects_unknown_chain_method(self, combined_data):
        dg, dt, _, _ = combined_data
        with pytest.raises(ValueError, match="chain_method"):
            run_nuts(dt.shape[0], model="combined", delta_g_obs=dg, delta_t=dt,
                     n_chains=1, n_warmup=10, n_samples=10, chain_method="threads")


class TestSamplerDispatch:
    def test_auto_additive_is_analytic(self, additive_data):
        dg, dt, _ = additive_data
        r = run_decontamination("MCMC-add", dg, dt, sampler="auto")
        assert r["sampler_backend"] == "analytic"
        assert r["weights"].shape == (N_PIX,)

    def test_auto_combined_is_nuts(self, combined_data):
        dg, dt, _, _ = combined_data
        r = run_decontamination("MCMC-comb", dg, dt, sampler="auto",
                                nuts_n_warmup=300, nuts_n_samples=400, n_chains=2)
        assert r["sampler_backend"] == "nuts"
        assert np.isfinite(r["rhat"]) and r["num_divergences"] == 0

    def test_emcee_backend_still_available(self, additive_data):
        dg, dt, _ = additive_data
        r = run_decontamination("MCMC-add", dg, dt, sampler="emcee",
                                n_walkers=40, n_steps=400, n_burn=100)
        assert r["sampler_backend"] == "emcee"
        assert r["a_hat"].shape == (dt.shape[0],)

    def test_analytic_combined_falls_back_to_nuts(self, combined_data):
        dg, dt, _, _ = combined_data
        # analytic posterior only exists for additive; combined must fall back
        r = run_decontamination("MCMC-comb", dg, dt, sampler="analytic",
                                nuts_n_warmup=300, nuts_n_samples=400, n_chains=2)
        assert r["sampler_backend"] == "nuts"

    def test_backends_agree_on_additive(self, additive_data):
        dg, dt, _ = additive_data
        r_an = run_decontamination("MCMC-add", dg, dt, sampler="analytic")
        r_em = run_decontamination("MCMC-add", dg, dt, sampler="emcee",
                                   n_walkers=40, n_steps=1200, n_burn=400)
        assert np.allclose(r_an["a_hat"], r_em["a_hat"], atol=2e-3)

    def test_default_n_chains_positive(self):
        assert default_n_chains() >= 1
