#!/usr/bin/env python3
"""Generate docs/results_benchmark.rst from the committed benchmark snapshot.

The measurements themselves live in the sys_mapping_benchmark repository; this only
renders the snapshot under ``docs/_static/benchmark/`` so the documentation builds
with no external checkout.  Re-run after refreshing that snapshot.

    python docs/generate_benchmark_page.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

DOCS = Path(__file__).resolve().parent
SNAP = DOCS / "_static" / "benchmark"
OUT = DOCS / "results_benchmark.rst"

GROUPS = [
    ("micro", "Per-function costs",
     "Numerically hot public API, JIT warm-up excluded."),
    ("maps", "HEALPix map utilities",
     "``pixelize_catalog`` uses :math:`10^5` galaxies."),
    ("stage1", "Stage-1 pre-selection",
     "All four ranking statistics of :func:`~sys_mapping.diagnostics.snr_template_ranking`."),
    ("stage2", "Stage-2 decontamination",
     "End-to-end per call to :func:`~sys_mapping.regression.run_decontamination`."),
]


def fmt(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.3g} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3g} ms"
    return f"{seconds * 1e6:.3g} µs"


def main() -> None:
    rows = list(csv.DictReader((SNAP / "benchmarks.csv").open()))
    for r in rows:
        r["median_s"] = float(r["median_s"])
        r["nside"] = int(r["nside"])
        r["n_sys"] = int(r["n_sys"])
    m = json.loads((SNAP / "machine.json").read_text())
    v = m.get("versions", {})
    load = m.get("loadavg_at_start", [float("nan")])[0]
    cores = m.get("n_cores_logical", 1)

    L = []
    add = L.append
    add("Benchmarks: how long each stage takes")
    add("=" * 38)
    add("")
    add(".. note::")
    add("   Generated from ``docs/_static/benchmark/benchmarks.csv`` by")
    add("   ``docs/generate_benchmark_page.py``.  The measurements are produced by the")
    add("   `sys_mapping_benchmark <https://github.com/JohanComparat/sys_mapping_benchmark>`_")
    add("   repository, which is kept separate so this package's CI does not carry 259")
    add("   timing cases.")
    add("")
    add("Provenance")
    add("----------")
    add("")
    add(f"Measured on {m.get('cpu','?')} ({cores} logical cores, "
        f"{m.get('mem_total_gb','?')} GB RAM), Python {m.get('python','?')}, "
        f"JAX {v.get('jax','?')} on the ``{m.get('jax_backend','?')}`` backend with "
        f"64-bit precision {'enabled' if m.get('jax_x64') else 'disabled'}; "
        f"NumPy {v.get('numpy','?')}, SciPy {v.get('scipy','?')}, healpy "
        f"{v.get('healpy','?')}, BlackJAX {v.get('blackjax','?')}, emcee "
        f"{v.get('emcee','?')}.  Commit ``{m.get('git_commit','?')}``, "
        f"{m.get('timestamp_utc','?')}.")
    add("")
    add(f"One-minute load average at the start of the run was {load:.2f} on {cores} "
        f"cores — {'the machine was effectively idle' if load < 0.5 * cores else '**the machine was contended; these are upper bounds**'}.")
    add("")
    add("Each entry is the **median** over repeated calls with JIT warm-up excluded;")
    add("median rather than mean because JIT stragglers and scheduler noise are")
    add("one-sided and inflate the mean.")
    add("")

    configs = sorted({(r["nside"], r["n_sys"]) for r in rows})
    headers = ["Operation"] + [f"NSIDE {ns}, n_s={k}" for ns, k in configs]

    for key, title, blurb in GROUPS:
        sub = [r for r in rows if r["group"] == key]
        if not sub:
            continue
        add("")
        add(title)
        add("-" * len(title))
        add("")
        add(blurb)
        add("")
        by_op: dict[str, dict] = defaultdict(dict)
        order: list[str] = []
        for r in sub:
            if r["operation"] not in by_op:
                order.append(r["operation"])
            by_op[r["operation"]][(r["nside"], r["n_sys"])] = r["median_s"]
        add(".. csv-table::")
        add("   :header: " + ", ".join(f'"{h}"' for h in headers))
        add("   :widths: 34" + ", 11" * len(configs))
        add("")
        for op in order:
            cells = [fmt(by_op[op][c]) if c in by_op[op] else "--" for c in configs]
            add(f'   "``{op}``", ' + ", ".join(f'"{c}"' for c in cells))
        add("")

    add("")
    add("Reading these")
    add("-------------")
    add("")
    add("* The JAX kernels are **dispatch-dominated** at these sizes: the forward and")
    add("  inverse contamination models differ by one element-wise division yet cost")
    add("  almost the same, because both are microseconds of arithmetic behind a fixed")
    add("  dispatch overhead.")
    add("* ``likelihood_ratio_test`` is far more expensive than two likelihood")
    add("  evaluations because **each call compiles two new likelihood functions**.")
    add("  Build them once with :func:`~sys_mapping.likelihood.make_log_likelihood` and")
    add("  difference them directly if you are testing repeatedly.")
    add("* Stage 2 spans **five orders of magnitude**, from OLS to MCMC-comb.  This is")
    add("  why :download:`run_ls10_analysis.py <../scripts/run_ls10_analysis.py>` runs")
    add("  the methods fastest-first and can checkpoint after the fast phase.")
    add("* Stage 1 is sub-millisecond for every statistic, so pre-selection cost is")
    add("  entirely in the GLASS mock null, which is embarrassingly parallel")
    add("  (``preselect_n_jobs``).")
    add("")
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(rows)} measurements, {len(configs)} configurations)")


if __name__ == "__main__":
    main()
