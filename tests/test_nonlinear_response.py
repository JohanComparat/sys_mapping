"""Non-linear template response: injection, normalisation, and what ISD-3 buys.

The existing simulation grid injects `delta_g*(1 + sum b_i t_i) + sum a_i t_i`,
which is linear in every template.  A linear marginal fit is already sufficient
for that, so it cannot separate ISD-1 from ISD-3 -- and in campaign 20260907c it
does not: the two recover the same amplitude in all 900 NSIDE-64 cells.  These
tests cover the injection path that makes the difference measurable.
"""

import numpy as np
import pytest

from sys_mapping import (
    RESPONSE_KINDS,
    TemplateResponse,
    apply_nonlinear_contamination,
    evaluate_response,
    run_decontamination,
)
from sys_mapping.simulation import RESPONSE_SHAPES, make_response_grid


@pytest.fixture(scope="module")
def templates():
    rng = np.random.default_rng(7)
    t = rng.standard_normal((5, 40000))
    return (t - t.mean(1, keepdims=True)) / t.std(1, keepdims=True)


class TestResponseNormalisation:
    @pytest.mark.parametrize("kind", RESPONSE_KINDS)
    @pytest.mark.parametrize("amp", [0.02, 0.05, 0.10])
    def test_rms_equals_requested_amplitude(self, kind, amp, templates):
        """Every shape is injected at the same power, or shapes are incomparable.

        A cubic with kappa=0.3 has more raw variance than a linear response at
        the same nominal coefficient, so without this an ordering of shapes is
        an ordering of amplitudes.
        """
        f = evaluate_response(
            TemplateResponse(kind, RESPONSE_SHAPES[kind], amp), templates[0])
        assert float(np.sqrt(np.mean(f ** 2))) == pytest.approx(amp, rel=1e-12)

    @pytest.mark.parametrize("kind", RESPONSE_KINDS)
    def test_response_is_centred(self, kind, templates):
        """A constant offset is absorbed by the density normalisation, not fitted."""
        f = evaluate_response(
            TemplateResponse(kind, RESPONSE_SHAPES[kind], 0.05), templates[0])
        assert abs(float(f.mean())) < 1e-12

    def test_unknown_kind_is_refused(self):
        with pytest.raises(ValueError, match="unknown response kind"):
            TemplateResponse("sigmoid", 1.0, 0.05)


class TestInjection:
    def test_linear_response_matches_the_multiplicative_branch(self, templates):
        """kind="linear" must reproduce the model the existing grid injects.

        With one template, `(1+d)(1+b t) - 1` is Eq. 13 with a = b, which is the
        selection-efficiency reading of the multiplicative branch.
        """
        rng = np.random.default_rng(3)
        d = rng.standard_normal(templates.shape[1]) * 0.3
        f = evaluate_response(TemplateResponse("linear", 0.0, 0.05), templates[0])
        got = apply_nonlinear_contamination(
            d, templates[:1], [TemplateResponse("linear", 0.0, 0.05)])
        np.testing.assert_allclose(got, (1.0 + d) * (1.0 + f) - 1.0, rtol=1e-12)

    @pytest.mark.parametrize("kind", RESPONSE_KINDS)
    def test_efficiency_never_goes_negative(self, kind, templates):
        """`sample_positions_from_delta` requires 1 + delta >= 0.

        A cubic or exponential tail on a skewed template reaches zero easily, so
        the floor is load-bearing rather than defensive.
        """
        d = np.zeros(templates.shape[1])
        out = apply_nonlinear_contamination(
            d, templates,
            [TemplateResponse(kind, RESPONSE_SHAPES[kind], 0.10)] * 5)
        assert np.all(1.0 + out >= 0.0)
        assert np.all(np.isfinite(out))

    def test_none_leaves_a_template_uncontaminated(self, templates):
        """A true negative is what makes greedy selection measurable."""
        rng = np.random.default_rng(4)
        d = rng.standard_normal(templates.shape[1]) * 0.3
        resp = [TemplateResponse("cubic", 0.3, 0.08), None, None, None, None]
        out = apply_nonlinear_contamination(d, templates, resp)
        # Only template 0 gains correlation with the injected field.
        r = [abs(np.corrcoef(out - d, templates[i])[0, 1]) for i in range(5)]
        assert r[0] > 0.3
        assert max(r[1:]) < 0.05

    def test_response_count_must_match_template_count(self, templates):
        with pytest.raises(ValueError, match="responses for"):
            apply_nonlinear_contamination(
                np.zeros(templates.shape[1]), templates, [None, None])


class TestWhatISD3Buys:
    """The campaign's headline question, at unit scale."""

    @staticmethod
    def _residual_dchi2(delta, t, order):
        from sys_mapping.diagnostics import isd_marginal_fit
        return float(isd_marginal_fit(delta, t, poly_order=order)[0][0])

    def test_cubic_injection_leaves_isd1_with_curvature_isd3_removes(self, templates):
        """On a curved response ISD-3 must beat ISD-1 where it is visible.

        Not in `a_hat` -- that is the linear projection of the fitted curve and
        is blind to this -- but in the residual Delta chi^2 the corrected field
        still shows against the template.
        """
        rng = np.random.default_rng(11)
        d = rng.standard_normal(templates.shape[1]) * 0.3
        resp = [TemplateResponse("cubic", 0.3, 0.10)] + [None] * 4
        obs = apply_nonlinear_contamination(d, templates, resp)

        r1 = run_decontamination("ISD-1", obs, templates, isd_chi2_68=50.0)
        r3 = run_decontamination("ISD-3", obs, templates, isd_chi2_68=50.0)

        # The corrected field is w*(1 + delta_obs) - 1, not w*delta_obs: the
        # weight multiplies the density, not the contrast.
        t0 = templates[:1]
        res1 = self._residual_dchi2(r1["weights"] * (1.0 + obs) - 1.0, t0, 3)
        res3 = self._residual_dchi2(r3["weights"] * (1.0 + obs) - 1.0, t0, 3)
        assert res3 < 0.5 * res1, (res1, res3)

    def test_the_amplitude_does_not_see_it(self, templates):
        """Guards the metric choice: a_hat must NOT be used to score this."""
        rng = np.random.default_rng(11)
        d = rng.standard_normal(templates.shape[1]) * 0.3
        obs = apply_nonlinear_contamination(
            d, templates, [TemplateResponse("cubic", 0.3, 0.10)] + [None] * 4)
        a1 = run_decontamination("ISD-1", obs, templates, isd_chi2_68=50.0)["a_hat"][0]
        a3 = run_decontamination("ISD-3", obs, templates, isd_chi2_68=50.0)["a_hat"][0]
        # Same linear content removed; the difference is not in the amplitude.
        assert abs(a1 - a3) < 0.5 * max(abs(a1), abs(a3))


class TestResponseGrid:
    def test_grid_shape_and_labels(self):
        cfgs = make_response_grid(n_sys=5, n_contaminated=2)
        assert len(cfgs) == 3 * len(RESPONSE_KINDS)
        assert {c.level for c in cfgs} == {"low", "medium", "high"}
        assert {c.shape for c in cfgs} == set(RESPONSE_KINDS)
        assert all(len(c.contaminated) == 2 for c in cfgs)

    def test_header_round_trip_carries_the_shape(self):
        cfg = make_response_grid(n_sys=5, n_contaminated=2)[0]
        h = cfg.to_header_dict()
        assert h["SCENARIO"] == "response"
        assert h["SHAPE"] in RESPONSE_KINDS
        assert any(k.startswith("RESP") for k in h)

    def test_all_templates_contaminated_by_default(self):
        cfgs = make_response_grid(n_sys=5)
        assert all(len(c.contaminated) == 5 for c in cfgs)
