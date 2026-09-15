#!/usr/bin/env python
"""Generate ``docs/coverage.rst``: test coverage and JAX coverage of the package.

Inputs:

* a coverage JSON report of the test suite (``pytest --cov --cov-branch
  --cov-report=json:PATH``);
* the static JAX analysis of ``scripts/jax_coverage.py``;
* the transformability cases of ``tests/test_jax_transformability.py``;
* optionally a ``cProfile`` dump of a pipeline run and its ``JAX_LOG_COMPILES=1`` log,
  summarised with ``scripts/profile_by_library.py``.

Run::

    pytest --cov=sys_mapping --cov-branch --cov-report=json:coverage.json
    python scripts/coverage_report.py coverage.json [--profile run.prof --compile-log run.err]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pstats
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "coverage.rst"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _table(header, rows):
    out = [".. csv-table::", "   :header: " + ", ".join(f'"{h}"' for h in header), ""]
    out += ["   " + ", ".join(f'"{c}"' for c in r) for r in rows]
    return out + [""]


def test_coverage_section(report: dict) -> list[str]:
    t = report["totals"]
    lines = ["Test coverage", "-------------", "",
             f"The suite covers {t['covered_lines']} of {t['num_statements']} statements "
             f"({100 * t['covered_lines'] / t['num_statements']:.1f}%) and "
             f"{t['covered_branches']} of {t['num_branches']} branches "
             f"({100 * t['covered_branches'] / t['num_branches']:.1f}%), "
             f"{t['percent_covered']:.1f}% combined. Docstring examples run as part of the "
             "suite. CI measures the same with branch coverage on every push and fails below "
             "the ``fail_under`` value in ``pyproject.toml``; tests that need the LS10 and "
             "Gaia maps or the Uchuu mocks skip there.", ""]
    rows = []
    for f, v in sorted(report["files"].items()):
        s = v["summary"]
        rows.append([Path(f).name, s["num_statements"],
                     f"{100 * s['covered_lines'] / max(s['num_statements'], 1):.1f}",
                     s["num_branches"],
                     f"{100 * s['covered_branches'] / s['num_branches']:.1f}" if s["num_branches"] else "—",
                     f"{s['percent_covered']:.1f}"])
    lines += _table(["module", "statements", "lines %", "branches", "branches %", "combined %"], rows)
    return lines


def jax_static_section() -> list[str]:
    jc = _load(REPO / "scripts" / "jax_coverage.py", "_jax_coverage")
    results = [jc.analyse_module(p) for p in sorted((REPO / "sys_mapping").glob("*.py"))
               if p.name != "__init__.py"]
    tot_num = sum(r["numeric"] for r in results)
    tot_jax = sum(r["jax"] for r in results)
    pub = Counter(f["kind"] for r in results for f in r["functions"]
                  if f["public"] and f["kind"] != "none")
    lines = ["JAX coverage", "------------", "",
             "Static share", "~~~~~~~~~~~~", "",
             "Every call is attributed to the library providing it (``scripts/jax_coverage.py``). "
             f"Of {tot_num} numeric source lines, {tot_jax} ({100 * tot_jax / tot_num:.0f}%) call "
             f"JAX. Of the {sum(pub.values())} public functions that make numeric calls, "
             f"{pub['jax']} call JAX only, {pub['mixed']} mix JAX with other libraries and "
             f"{pub['other']} call none of it.", ""]
    rows = []
    for r in results:
        if not r["numeric"]:
            continue
        kinds = Counter(f["kind"] for f in r["functions"])
        others = ", ".join(f"{k} {v}" for k, v in r["by_family"].most_common() if k != "jax")
        rows.append([r["module"], r["numeric"], r["jax"], f"{100 * r['jax'] / r['numeric']:.0f}",
                     f"{kinds['jax']}/{kinds['mixed']}/{kinds['other']}", others or "—"])
    lines += _table(["module", "numeric lines", "JAX lines", "JAX %",
                     "functions jax/mixed/other", "other libraries (lines)"], rows)
    return lines


def transformability_section() -> list[str]:
    tj = _load(REPO / "tests" / "test_jax_transformability.py", "_jax_transformability")
    cases = [(name, tr) for name, (_, _, trs) in tj.CASES.items() for tr in trs]
    failing = set(tj.KNOWN_FAILURES)
    by_fn = defaultdict(list)
    for name, tr in cases:
        by_fn[name].append((tr, (name, tr) not in failing))
    full = sum(all(ok for _, ok in v) for v in by_fn.values())
    lines = ["Transformability", "~~~~~~~~~~~~~~~~", "",
             "``tests/test_jax_transformability.py`` applies ``jax.jit``, ``jax.vmap`` and, where "
             "a gradient is meaningful, ``jax.grad`` to the public numeric functions and checks "
             f"the result against the eager call. {len(cases) - len(failing)} of {len(cases)} "
             f"cases pass; {full} of {len(by_fn)} functions pass every transform that applies. "
             "The failures are strict expected failures, so a function that becomes "
             "transformable fails the suite until its entry is removed.", ""]
    rows = []
    for name, trs in by_fn.items():
        rows.append([name] + [("yes" if ok else "no") if any(t == tr for t, _ in trs) else "—"
                              for tr in ("jit", "vmap", "grad")
                              for ok in [dict(trs).get(tr, None)]]
                    + [tj._NUMPY_ON_TRACER.get(name, "")])
    lines += _table(["function", "jit", "vmap", "grad", "blocked by"], rows)
    return lines


def profile_section(profile: Path, compile_log: Path | None, label: str) -> list[str]:
    pl = _load(REPO / "scripts" / "profile_by_library.py", "_profile_by_library")
    stats = pstats.Stats(str(profile))
    by_lib: dict[str, float] = defaultdict(float)
    for (filename, _l, funcname), (_cc, _nc, tottime, _ct, _c) in stats.stats.items():
        by_lib[pl.classify(filename, funcname)] += tottime
    total = sum(by_lib.values())
    lines = ["Where a pipeline run spends its time", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "",
             f"A profiled run of ``scripts/run_ls10_analysis.py`` ({label}), own time per "
             f"library from ``scripts/profile_by_library.py``; {total:.0f} s in total. Time "
             "spent executing compiled XLA code is counted under JAX.", ""]
    rows = [[lib, f"{t:.0f}", f"{100 * t / total:.1f}"]
            for lib, t in sorted(by_lib.items(), key=lambda kv: -kv[1]) if t / total >= 0.001]
    lines += _table(["library", "seconds", "share %"], rows)
    stages = {}
    for (filename, _l, funcname), (_cc, nc, _tt, cumtime, _c) in stats.stats.items():
        if "/sys_mapping/sys_mapping/" in filename and funcname in STAGES:
            stages[funcname] = (cumtime, nc)
    if stages:
        lines += ["By stage, cumulative time of the package functions that dominate:", ""]
        lines += _table(["function", "calls", "seconds", "share %"],
                        [[fn, nc, f"{ct:.0f}", f"{100 * ct / total:.1f}"]
                         for fn, (ct, nc) in sorted(stages.items(), key=lambda kv: -kv[1][0])])
    if compile_log and compile_log.exists():
        text = compile_log.read_text(errors="replace")
        n = len(re.findall(r"Compiling |Finished XLA compilation", text))
        lines += [f"The run compiled {n} XLA programs.", ""]
    return lines


STAGES = ("template_correlation_matrix", "run_nuts", "run_additive_analytic",
          "measure_two_point_function", "generate_glass_fullsky_mock",
          "isd_template_significance", "calibrated_template_significance",
          "correct_two_point_function", "elasticnet_contamination_fit",
          "iterative_systematics_decontamination")

OUT_OF_SCOPE = [
    ("TreeCorr and Corrfunc pair counts", "tree-based C++ pair counting with no JAX equivalent"),
    ("healpy harmonic transforms (anafast, synfast, ud_grade)",
     "a JAX version needs jax-healpy or s2fft, which the package does not depend on"),
    ("GLASS field generation", "GLASS is a NumPy library"),
    ("scikit-learn ElasticNet", "cross-validated fits take 0.05–0.12 s"),
    ("FITS input and output", "I/O"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("coverage_json")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--compile-log", default=None)
    ap.add_argument("--profile-label", default="one LS10 sample")
    args = ap.parse_args()

    report = json.loads(Path(args.coverage_json).read_text())
    lines = ["Test and JAX coverage", "=====================", "",
             "How much of the package the test suite exercises, and how much of its numerical "
             "work runs through JAX. Generated by ``scripts/coverage_report.py``.", "",
             ".. contents:: On this page", "   :local:", "   :depth: 1", ""]
    lines += test_coverage_section(report)
    lines += jax_static_section()
    lines += transformability_section()
    if args.profile:
        lines += profile_section(Path(args.profile),
                                 Path(args.compile_log) if args.compile_log else None,
                                 args.profile_label)
    lines += ["Outside JAX", "~~~~~~~~~~~", ""]
    lines += [f"* **{what}**: {why}." for what, why in OUT_OF_SCOPE] + [""]
    lines += ["Measuring", "---------", "", ".. code-block:: bash", "",
              "   pytest --cov=sys_mapping --cov-branch --cov-report=json:coverage.json",
              "   python scripts/jax_coverage.py",
              "   JAX_LOG_COMPILES=1 python -m cProfile -o run.prof scripts/run_ls10_analysis.py ... 2> run.err",
              "   python scripts/profile_by_library.py run.prof --compile-log run.err",
              "   python scripts/coverage_report.py coverage.json --profile run.prof --compile-log run.err",
              ""]
    OUT.write_text("\n".join(lines))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
