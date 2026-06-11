"""Tests for model_selection.py: likelihood ratio test and greedy forward selection."""

import numpy as np
import pytest

from sys_mapping.model_selection import (
    likelihood_ratio_test,
    greedy_forward_select,
    GreedyForwardSelectionResult,
    ForwardSelectionRound,
    SnrPreselectionResult,
    snr_preselect,
)
from sys_mapping.contamination import pack_params


N_PIX = 300
N_SYS = 3


def _make_data(seed=0):
    rng = np.random.default_rng(seed)
    delta_t = rng.standard_normal((N_SYS, N_PIX))
    delta_t -= delta_t.mean(axis=1, keepdims=True)
    delta_g_obs = rng.standard_normal(N_PIX) * 0.1
    return delta_g_obs, delta_t


class TestLikelihoodRatioTest:
    def test_identical_params_lambda_zero(self):
        """When null and alt are at the same point (embedded), λ_LR >= 0."""
        dg, dt = _make_data()
        sigma = 0.1
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)

        # Additive null embedded in combined alt: b=0 in combined
        theta_null = pack_params(a, None, sigma, model="additive")
        theta_alt = pack_params(a, b, sigma, model="combined")

        result = likelihood_ratio_test(dg, dt, theta_null, theta_alt, "additive", "combined")
        assert result.lambda_lr >= -1e-6  # should be >= 0

    def test_better_fit_gives_positive_lambda(self):
        """Combined model at true params should have higher likelihood than additive null."""
        rng = np.random.default_rng(1)
        a_true = np.array([0.15, -0.1, 0.12])
        b_true = np.array([0.2, -0.15, 0.1])
        sigma = 0.05
        delta_t = rng.standard_normal((N_SYS, N_PIX))
        delta_t -= delta_t.mean(axis=1, keepdims=True)

        from sys_mapping.contamination import apply_contamination
        import jax.numpy as jnp
        dg_clean = rng.standard_normal(N_PIX) * 0.02
        dg_obs = np.asarray(
            apply_contamination(
                jnp.asarray(dg_clean), jnp.asarray(delta_t),
                jnp.asarray(a_true), jnp.asarray(b_true),
            )
        )

        theta_null = pack_params(a_true, None, sigma, model="additive")
        theta_alt = pack_params(a_true, b_true, sigma, model="combined")

        result = likelihood_ratio_test(dg_obs, delta_t, theta_null, theta_alt, "additive", "combined")
        assert result.lambda_lr > 0

    def test_degrees_of_freedom_additive_vs_combined(self):
        """Combined has N_SYS extra params over additive, so r = N_SYS."""
        dg, dt = _make_data()
        sigma = 0.1
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)
        theta_null = pack_params(a, None, sigma, model="additive")
        theta_alt = pack_params(a, b, sigma, model="combined")

        result = likelihood_ratio_test(dg, dt, theta_null, theta_alt, "additive", "combined")
        assert result.n_dof == N_SYS

    def test_degrees_of_freedom_mult_vs_combined(self):
        """Combined has N_SYS extra params over multiplicative, so r = N_SYS."""
        dg, dt = _make_data()
        sigma = 0.1
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)
        theta_null = pack_params(a, a, sigma, model="multiplicative")
        theta_alt = pack_params(a, b, sigma, model="combined")

        result = likelihood_ratio_test(dg, dt, theta_null, theta_alt, "multiplicative", "combined")
        assert result.n_dof == N_SYS

    def test_invalid_direction_raises(self):
        """Null more complex than alt should raise ValueError."""
        dg, dt = _make_data()
        sigma = 0.1
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)
        theta_null = pack_params(a, b, sigma, model="combined")
        theta_alt = pack_params(a, None, sigma, model="additive")

        with pytest.raises(ValueError):
            likelihood_ratio_test(dg, dt, theta_null, theta_alt, "combined", "additive")

    def test_p_value_range(self):
        dg, dt = _make_data()
        sigma = 0.1
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)
        theta_null = pack_params(a, None, sigma, model="additive")
        theta_alt = pack_params(a, b, sigma, model="combined")

        result = likelihood_ratio_test(dg, dt, theta_null, theta_alt, "additive", "combined")
        assert 0.0 <= result.p_value <= 1.0

    def test_reject_null_field_consistent(self):
        dg, dt = _make_data()
        sigma = 0.1
        a = np.zeros(N_SYS)
        b = np.zeros(N_SYS)
        theta_null = pack_params(a, None, sigma, model="additive")
        theta_alt = pack_params(a, b, sigma, model="combined")

        result = likelihood_ratio_test(dg, dt, theta_null, theta_alt, "additive", "combined",
                                       significance=0.05)
        assert result.reject_null == (result.p_value < 0.05)


class TestGreedyForwardSelect:
    """Tests for greedy_forward_select."""

    N_PIX = 3000

    def _pure_noise(self, seed=42):
        rng = np.random.default_rng(seed)
        delta_t = rng.standard_normal((5, self.N_PIX))
        delta_g = rng.standard_normal(self.N_PIX) * 0.05
        return delta_g, delta_t

    def _with_signal(self, seed=0):
        rng = np.random.default_rng(seed)
        delta_t = rng.standard_normal((4, self.N_PIX))
        # Strong contamination from template 2 only
        a_true = np.array([0.0, 0.0, 0.8, 0.0])
        delta_g = a_true @ delta_t + rng.standard_normal(self.N_PIX) * 0.02
        return delta_g, delta_t

    def test_returns_correct_type(self):
        dg, dt = self._pure_noise()
        result = greedy_forward_select(dg, dt, p_threshold=1e-6)
        assert isinstance(result, GreedyForwardSelectionResult)
        assert isinstance(result.selected_indices, list)
        assert isinstance(result.rounds, list)

    def test_pure_noise_tight_threshold_selects_nothing(self):
        dg, dt = self._pure_noise()
        result = greedy_forward_select(dg, dt, p_threshold=1e-6)
        assert len(result.selected_indices) == 0
        assert len(result.rounds) == 0

    def test_metadata_fields(self):
        dg, dt = self._pure_noise()
        result = greedy_forward_select(dg, dt, p_threshold=0.05)
        assert result.p_threshold == 0.05
        assert result.n_initial == dt.shape[0]

    def test_selects_injected_template(self):
        dg, dt = self._with_signal()
        result = greedy_forward_select(dg, dt, p_threshold=0.05)
        assert 2 in result.selected_indices

    def test_selected_indices_are_valid(self):
        dg, dt = self._with_signal()
        result = greedy_forward_select(dg, dt, p_threshold=0.05)
        assert all(0 <= i < dt.shape[0] for i in result.selected_indices)

    def test_round_diagnostics_match_selected(self):
        dg, dt = self._with_signal()
        result = greedy_forward_select(dg, dt, p_threshold=0.05)
        assert len(result.rounds) == len(result.selected_indices)
        for k, r in enumerate(result.rounds):
            assert isinstance(r, ForwardSelectionRound)
            assert r.round_num == k + 1
            assert r.added_index == result.selected_indices[k]
            assert 0.0 <= r.p_value < 0.05

    def test_empty_candidate_set(self):
        rng = np.random.default_rng(0)
        delta_g = rng.standard_normal(self.N_PIX)
        delta_t = np.empty((0, self.N_PIX))
        result = greedy_forward_select(delta_g, delta_t)
        assert result.selected_indices == []
        assert result.n_initial == 0

    def test_no_duplicate_indices(self):
        dg, dt = self._with_signal()
        result = greedy_forward_select(dg, dt, p_threshold=0.05)
        assert len(result.selected_indices) == len(set(result.selected_indices))


class TestSnrPreselect:
    """Tests for snr_preselect and SnrPreselectionResult."""

    N_PIX = 4000

    @pytest.fixture
    def noise_data(self):
        rng = np.random.default_rng(70)
        delta_t = rng.standard_normal((6, self.N_PIX))
        delta_g = rng.standard_normal(self.N_PIX) * 0.05
        return delta_g, delta_t

    @pytest.fixture
    def signal_data(self):
        rng = np.random.default_rng(71)
        n_sys = 5
        delta_t = rng.standard_normal((n_sys, self.N_PIX))
        # Strong contamination from template 2 only
        delta_g = 0.7 * delta_t[2] + rng.standard_normal(self.N_PIX) * 0.05
        return delta_g, delta_t

    def test_returns_correct_type(self, noise_data):
        dg, dt = noise_data
        result = snr_preselect(dg, dt)
        assert isinstance(result, SnrPreselectionResult)
        assert isinstance(result.selected_indices, list)
        assert isinstance(result.snr_values, np.ndarray)

    def test_snr_values_shape(self, noise_data):
        dg, dt = noise_data
        result = snr_preselect(dg, dt)
        assert result.snr_values.shape == (dt.shape[0],)

    def test_metadata_fields(self, noise_data):
        dg, dt = noise_data
        result = snr_preselect(dg, dt, method="data", snr_min=0.1, n_top=3)
        assert result.method == "data"
        assert result.snr_min == 0.1
        assert result.n_top == 3
        assert result.n_initial == dt.shape[0]

    def test_injected_template_ranks_first_data(self, signal_data):
        dg, dt = signal_data
        result = snr_preselect(dg, dt, method="data")
        assert result.selected_indices[0] == 2

    def test_injected_template_ranks_first_template(self, signal_data):
        dg, dt = signal_data
        result = snr_preselect(dg, dt, method="template")
        assert result.selected_indices[0] == 2

    def test_injected_template_ranks_first_isd(self, signal_data):
        dg, dt = signal_data
        result = snr_preselect(dg, dt, method="isd")
        assert result.selected_indices[0] == 2

    def test_no_filter_returns_all_sorted(self, noise_data):
        dg, dt = noise_data
        result = snr_preselect(dg, dt)
        assert len(result.selected_indices) == dt.shape[0]
        # Verify SNR-descending order
        snr_order = [result.snr_values[i] for i in result.selected_indices]
        assert snr_order == sorted(snr_order, reverse=True)

    def test_snr_min_tight_gives_empty(self, noise_data):
        dg, dt = noise_data
        result = snr_preselect(dg, dt, method="data", snr_min=1e6)
        assert len(result.selected_indices) == 0

    def test_n_top_limits_count(self, noise_data):
        dg, dt = noise_data
        result = snr_preselect(dg, dt, n_top=3)
        assert len(result.selected_indices) == 3

    def test_snr_min_and_n_top_combined(self, signal_data):
        dg, dt = signal_data
        result = snr_preselect(dg, dt, method="data", snr_min=0.0, n_top=2)
        assert len(result.selected_indices) <= 2
        for idx in result.selected_indices:
            assert result.snr_values[idx] >= 0.0

    def test_selected_indices_valid(self, signal_data):
        dg, dt = signal_data
        result = snr_preselect(dg, dt, n_top=3)
        assert all(0 <= i < dt.shape[0] for i in result.selected_indices)

    def test_no_duplicate_indices(self, signal_data):
        dg, dt = signal_data
        result = snr_preselect(dg, dt)
        assert len(result.selected_indices) == len(set(result.selected_indices))

    def test_isd_method_accepted(self, signal_data):
        dg, dt = signal_data
        result = snr_preselect(dg, dt, method="isd", n_bins=10, poly_order=1)
        assert isinstance(result, SnrPreselectionResult)
        assert result.method == "isd"
        assert np.all(result.snr_values >= 0)
