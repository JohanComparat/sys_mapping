"""Tests for helpers in ``scripts/run_ls10_analysis.py``.

The script is not an installed module, so it is loaded by path.  Importing it is
cheap: the heavy work all lives behind ``main()``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import sys_mapping as sm

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_ls10_analysis.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("_ls10_analysis", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ls10_analysis"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ls10():
    return _load_script()


class TestExpandPreselectedParams:
    """Regression guard for the ``--preselect`` all-ones-weights bug.

    Pre-selection fits a subset of templates, so ``a_hat`` is shorter than the
    template basis used to build the weight map.  The old code discarded it, which
    made every weight column exactly 1.0 (no correction) with no warning.
    """

    def test_full_length_passes_through(self, ls10):
        p = np.array([0.1, -0.2, 0.3])
        out = ls10.expand_preselected_params(p, 3, None)
        np.testing.assert_array_equal(out, p)

    def test_preselected_subset_is_scattered_back(self, ls10):
        # 2 of 5 templates survived Stage 1, at positions 1 and 3.
        out = ls10.expand_preselected_params(np.array([0.4, -0.7]), 5, [1, 3])
        np.testing.assert_allclose(out, [0.0, 0.4, 0.0, -0.7, 0.0])

    def test_scattered_params_produce_non_unit_weights(self, ls10):
        """The actual failure mode: weights must not collapse to 1."""
        rng = np.random.default_rng(0)
        n_sys, n_pix = 5, 400
        templates = rng.standard_normal((n_sys, n_pix))
        a_hat_subset = np.array([0.30, -0.25])          # fitted on templates 1 and 3

        expanded = ls10.expand_preselected_params(a_hat_subset, n_sys, [1, 3])
        weights = 1.0 / np.maximum(1.0 + np.einsum("i,ij->j", expanded, templates), 0.01)

        assert not np.allclose(weights, 1.0), "correction collapsed to unit weights"
        assert weights.std() > 0.01

    def test_unusable_shape_warns_and_returns_zeros(self, ls10):
        with pytest.warns(UserWarning, match="matches neither"):
            out = ls10.expand_preselected_params(np.array([0.1, 0.2]), 5, [1, 2, 3])
        np.testing.assert_array_equal(out, np.zeros(5))

    def test_missing_method_stays_zero_without_warning(self, ls10):
        """An unfitted method legitimately yields w == 1; that must not warn."""
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            out = ls10.expand_preselected_params(np.zeros(4), 4, [0, 2])
        np.testing.assert_array_equal(out, np.zeros(4))


class TestWeightConvention:
    """The shipped weight must be the library's, not a linear reconstruction.

    For OLS the two agree by construction.  For ISD they do not: the library's
    weight is the cumulative product of the per-step corrections
    ``prod_j 1/(1 + F_j(t_j))``, which carries the curvature a cubic marginal fit
    found, while ``1/(1 + a_hat . t)`` is the linear projection of it.
    """

    @staticmethod
    def _setup(seed=0, n_sys=4, n_pix=6000):
        rng = np.random.default_rng(seed)
        dt = rng.standard_normal((n_sys, n_pix))
        dt = (dt - dt.mean(1, keepdims=True)) / dt.std(1, keepdims=True)
        # A curved response, so the two conventions genuinely differ.
        f = 0.12 * dt[0] + 0.06 * dt[0] ** 2 - 0.03 * dt[0] ** 3
        dg = (1.0 + rng.standard_normal(n_pix) * 0.3) * (1.0 + f) - 1.0
        return dg, dt

    def test_library_weight_differs_from_linear_reconstruction_for_isd(self):
        dg, dt = self._setup()
        res = sm.run_decontamination("ISD-3", dg, dt, isd_chi2_68=50.0)
        w_lib = np.asarray(res["weights"])
        w_lin = 1.0 / np.maximum(1.0 + np.asarray(res["a_hat"]) @ dt, 1.0 / 20.0)
        assert not np.allclose(w_lib, w_lin, rtol=1e-3), (
            "the two conventions must differ for ISD, or this fix is untested")

    def test_ols_weight_agrees_with_the_linear_form(self):
        """The control: where the model IS linear, the conventions coincide."""
        dg, dt = self._setup(seed=1)
        res = sm.run_decontamination("OLS", dg, dt)
        w_lib = np.asarray(res["weights"])
        w_lin = 1.0 / np.maximum(1.0 + np.asarray(res["a_hat"]) @ dt, 1.0 / 20.0)
        np.testing.assert_allclose(w_lib, w_lin, rtol=1e-6)

    @pytest.mark.parametrize("method", ["OLS", "ISD-1", "ISD-3", "ElasticNet"])
    def test_weights_respect_the_library_clip(self, method):
        """One floor everywhere: 1/20, not the scripts' old 1/100."""
        dg, dt = self._setup(seed=2)
        kw = {"isd_chi2_68": 50.0} if method.startswith("ISD") else {}
        w = np.asarray(sm.run_decontamination(method, dg, dt, **kw)["weights"])
        assert w.min() >= 1.0 / 20.0 - 1e-12
        assert w.max() <= 20.0 + 1e-12

    def test_weight_map_scatters_onto_the_footprint(self):
        mod = _load_script()

        n_pix = 100
        good = np.zeros(n_pix, dtype=bool)
        good[10:40] = True
        w = np.linspace(0.5, 1.5, 30)
        wm = mod.method_weight_map({"weights": w}, good, n_pix, label="t")
        np.testing.assert_allclose(wm[good], w)
        assert np.all(wm[~good] == 1.0)      # unfitted pixels are uncorrected

        with pytest.raises(ValueError, match="one per fitted pixel"):
            mod.method_weight_map({"weights": w[:5]}, good, n_pix, label="t")


class TestSkewedFlagParity:
    """Both production scripts expose the same opt-in skew-normal likelihood.

    ``compute_sys_weights.py`` used to default it *on* while
    ``run_ls10_analysis.py`` had no flag at all and was Gaussian always, so a
    column named ``WEIGHT_SYS`` meant a different model depending on which
    script wrote it.  Enabling the skew-normal also routes the additive model
    off its exact analytic posterior onto NUTS, which is why it is opt-in.
    """

    def test_flag_exists_and_defaults_off(self, ls10, monkeypatch):
        # The parser is built inside main(), so capture it as it is constructed.
        import argparse

        captured = {}
        real_parse = argparse.ArgumentParser.parse_args

        def _capture(self, *a, **kw):
            captured["parser"] = self
            raise SystemExit(0)

        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _capture)
        monkeypatch.setattr(sys, "argv", ["run_ls10_analysis.py"])
        with pytest.raises(SystemExit):
            ls10.main()
        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", real_parse)

        parser = captured["parser"]
        skewed = [a for a in parser._actions if "--skewed" in a.option_strings]
        assert skewed, "run_ls10_analysis.py has no --skewed flag"
        assert skewed[0].default is False
        assert skewed[0].dest == "skewed"

    def test_gamma_hat_is_reported_and_is_none_when_gaussian(self):
        rng = np.random.default_rng(4)
        delta_t = rng.standard_normal((2, 1500))
        delta_g = np.array([0.05, -0.02]) @ delta_t + rng.standard_normal(1500) * 0.1
        res = sm.run_decontamination("OLS", delta_g, delta_t)
        assert "gamma_hat" in res and res["gamma_hat"] is None


class TestNullSpectrumIsMandatory:
    """No calibrated null is built on a spectrum nobody chose.

    The default power law under-clusters an LS10 sample about 25x in variance, so a
    null built on it is too narrow and its p-values are anticonservative.  The script
    refuses to build one unless a matched spectrum resolves or the caller accepts an
    uncalibrated null explicitly.
    """

    @staticmethod
    def _args(**kw):
        import argparse

        base = dict(lrt_null_cl_file=None, allow_parametric_null=False,
                    lrt_null_cl_amplitude=None)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_refuses_without_a_spectrum(self, ls10):
        with pytest.raises(SystemExit, match="no --null-cl-file given"):
            ls10._resolve_null_cl(self._args(), "SAMPLE", 64)

    def test_refuses_when_the_directory_has_nothing_for_the_sample(self, ls10, tmp_path):
        with pytest.raises(SystemExit, match="no matched spectrum for SAMPLE NSIDE0064"):
            ls10._resolve_null_cl(self._args(lrt_null_cl_file=str(tmp_path)), "SAMPLE", 64)

    def test_parametric_only_when_asked(self, ls10, capsys):
        out = ls10._resolve_null_cl(self._args(allow_parametric_null=True), "SAMPLE", 64)
        assert out is None
        assert "NOT calibrated" in capsys.readouterr().out

    def test_a_validated_match_is_returned(self, ls10, tmp_path):
        import json

        cl = [0.0, 1e-3, 5e-4, 2e-4, 1e-4]
        (tmp_path / "SAMPLE_NSIDE0064_match.json").write_text(json.dumps({
            "sample": "SAMPLE", "nside": 64, "cl_matched": cl,
            "validation": {"passed": True, "large_scale_ratio": 1.0,
                           "l_range": [2, 32], "tol": 0.1},
        }))
        out = ls10._resolve_null_cl(self._args(lrt_null_cl_file=str(tmp_path)), "SAMPLE", 64)
        np.testing.assert_allclose(out, cl)

    def test_the_old_flag_name_still_parses(self, ls10, monkeypatch):
        import argparse

        captured = {}

        def _capture(self, *a, **kw):
            captured["parser"] = self
            raise SystemExit(0)

        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _capture)
        monkeypatch.setattr(sys, "argv", ["run_ls10_analysis.py"])
        with pytest.raises(SystemExit):
            ls10.main()
        opts = {o for a in captured["parser"]._actions for o in a.option_strings}
        assert {"--null-cl-file", "--lrt-null-cl-file", "--allow-parametric-null"} <= opts


@pytest.mark.filterwarnings("ignore:.*not matched to any sample.*:UserWarning")
class TestNullFields:
    """The GLASS null the script builds its calibrated statistics from."""

    def test_null_overdensity_fields_are_reproducible_by_index(self, ls10):
        pytest.importorskip("glass")
        nside = 8
        good = np.ones(12 * nside ** 2, dtype=bool)
        kw = dict(nside=nside, good_pix=good, n_total_footprint=20000,
                  z_edges=np.array([0.1, 0.3]), nz=np.array([1.0]), seed=7, cl_amplitude=5e-4)
        both = ls10.null_overdensity_fields(2, **kw)
        second = ls10.null_overdensity_fields(1, k_start=1, **kw)
        assert both.shape == (2, good.sum())
        np.testing.assert_array_equal(both[1], second[0])

    def test_build_lrt_null_with_an_ols_fit(self, ls10, monkeypatch):
        pytest.importorskip("glass")
        nside = 8
        npix = 12 * nside ** 2
        good = np.ones(npix, dtype=bool)
        rng = np.random.default_rng(0)
        t = rng.standard_normal((2, npix))

        def fake_decontamination(method, dg, dt, **kw):
            a, *_ = np.linalg.lstsq(dt.T, dg, rcond=None)
            return {"a_hat": a, "b_hat": np.zeros(dt.shape[0]),
                    "sigma_hat": float(np.std(dg - a @ dt))}

        monkeypatch.setattr(ls10.sm, "run_decontamination", fake_decontamination)
        lam = ls10.build_lrt_null(2, nside, good, t, np.array([0.1, 0.3]), np.array([1.0]),
                                  20000, seed=3, sampler="analytic", nuts_warmup=10,
                                  nuts_samples=10, n_chains=1, cl_amplitude=5e-4)
        assert lam.shape == (2,)
        assert np.all(lam >= -1e-6)
