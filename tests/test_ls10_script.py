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
