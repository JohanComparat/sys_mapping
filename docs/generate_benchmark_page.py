#!/usr/bin/env python3
"""Generate docs/results_benchmark.rst from the committed benchmark snapshot.

The measurements come from ``benchmark/benchmark_pipeline.py`` in the
sys_mapping_benchmark repository (family D of its OAR campaign); this only renders
the snapshot under ``docs/_static/benchmark/`` so the documentation builds with no
external checkout.  A snapshot is ``benchmarks.csv`` and ``machine.json``, plus the
campaign's ``machine.txt`` and ``pkg_version.txt`` when it ran on the cluster.

    python docs/generate_benchmark_page.py [--note "one line shown under the provenance"]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path

DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent
SNAP = DOCS / "_static" / "benchmark"
OUT = DOCS / "results_benchmark.rst"

GROUPS = [
    ("micro", "Per-function costs", "Public functions on the hot path."),
    ("maps", "HEALPix map utilities", "``pixelize_catalog`` bins :math:`10^5` galaxies."),
    ("stage1", "Stage-1 pre-selection",
     "The ranking statistics of :func:`~sys_mapping.diagnostics.snr_template_ranking`."),
    ("stage2", "Stage-2 decontamination",
     "One call to :func:`~sys_mapping.regression.run_decontamination` per method."),
]


def fmt(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "--"
    if seconds >= 1.0:
        return f"{seconds:.3g} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3g} ms"
    return f"{seconds * 1e6:.3g} µs"


def _library(snap: Path, machine: dict) -> tuple[str, str]:
    """Version and short commit of the sys_mapping that was timed."""
    commit = None
    pv = snap / "pkg_version.txt"
    if pv.exists():
        text = pv.read_text()
        m = re.search(r"head:\s*([0-9a-f]{7,40})(?:\s*\((\d+\.\d+\.\d+))?", text)
        if m:
            commit = m.group(1)[:7]
            if m.group(2):          # a staged tree records the version it carries
                return m.group(2), commit
    # Snapshots taken inside this repository record its commit in machine.json.
    commit = commit or machine.get("sys_mapping_commit") or machine.get("git_commit")
    version = "?"
    if commit:
        for path, pattern in (("sys_mapping/__init__.py", r'__version__\s*=\s*"([^"]+)"'),
                              ("pyproject.toml", r'^version\s*=\s*"([^"]+)"')):
            try:
                text = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"],
                                      capture_output=True, text=True, check=True).stdout
            except (OSError, subprocess.CalledProcessError):
                continue
            m = re.search(pattern, text, re.M)
            if m:
                version = m.group(1)
                break
    return version, commit or "?"


def main() -> None:
    ap = argparse.ArgumentParser(description="Render docs/results_benchmark.rst.")
    ap.add_argument("--snapshot", type=Path, default=SNAP)
    ap.add_argument("--note", default=None, help="One line printed under the provenance.")
    args = ap.parse_args()
    snap = args.snapshot

    rows = list(csv.DictReader((snap / "benchmarks.csv").open()))

    # The harness records a failed cell as a row with empty timings rather than
    # dropping it, so these must parse to nan instead of raising.
    def _num(value, cast=float, default=float("nan")):
        try:
            return cast(value)
        except (TypeError, ValueError):
            return default

    for r in rows:
        r["median_s"] = _num(r["median_s"])
        r["nside"] = _num(r["nside"], int, 0)
        r["n_sys"] = _num(r["n_sys"], int, 0)
    m = json.loads((snap / "machine.json").read_text())
    v = m.get("versions", {})
    load = m.get("loadavg_at_start", [float("nan")])[0]
    cores = m.get("n_cores_logical", 1)
    host = ""
    if (snap / "machine.txt").exists():
        mt = dict(line.split("=", 1) for line in (snap / "machine.txt").read_text().splitlines()
                  if "=" in line)
        host = mt.get("host", "")
        cores = int(mt.get("ncore", cores))
    version, commit = _library(snap, m)
    where = f"dahu node {host}, " if host.startswith("dahu") else (f"{host}, " if host else "")
    date = str(m.get("timestamp_utc", "?"))[:10]
    nuts = sorted({int(n) for r in rows for n in re.findall(r"nuts=(\d+)", r.get("note") or "")})

    L = []
    add = L.append
    add("Benchmarks")
    add("==========")
    add("")
    add(f"sys_mapping {version} (commit ``{commit}``), {where}{m.get('cpu', '?')}, {date}.")
    if args.note:
        add(args.note)
    add("")
    add(f"We time {len(rows)} cases of the pipeline with ``benchmark/benchmark_pipeline.py`` of the")
    add("`sys_mapping_benchmark <https://github.com/JohanComparat/sys_mapping_benchmark>`_")
    add("repository; ``docs/generate_benchmark_page.py`` renders")
    add("``docs/_static/benchmark/benchmarks.csv`` into this page.")
    add("Each entry is the median over repeated calls, with the JIT compilation of the first call")
    add("excluded.")
    add(f"The run used {cores} cores, Python {m.get('python', '?')}, JAX {v.get('jax', '?')} "
        f"(``{m.get('jax_backend', '?')}``, 64-bit "
        f"{'on' if m.get('jax_x64') else 'off'}), NumPy {v.get('numpy', '?')}, "
        f"SciPy {v.get('scipy', '?')}, healpy {v.get('healpy', '?')} and "
        f"BlackJAX {v.get('blackjax', '?')}; the one-minute load average at the start was "
        f"{load:.2f}.")
    add("MCMC-add is the analytic posterior and MCMC-comb NUTS with two chains"
        + (f", {nuts[0]} warmup steps and {nuts[0]} draws per chain." if len(nuts) == 1 else "."))
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

    def _finite(group):
        return [r["median_s"] for r in rows
                if r["group"] == group and math.isfinite(r["median_s"]) and r["median_s"] > 0]

    s1, s2 = _finite("stage1"), _finite("stage2")
    if s1 or s2:
        add("")
        add("Ranges")
        add("------")
        add("")
        if s2:
            add(f"Stage-2 calls span {math.log10(max(s2) / min(s2)):.1f} decades, from "
                f"{fmt(min(s2))} to {fmt(max(s2))}.")
        if s1:
            add(f"A Stage-1 ranking call takes {fmt(min(s1))} to {fmt(max(s1))}; the cost of "
                "pre-selection is the GLASS null, which runs in parallel over "
                "``preselect_n_jobs``.")
        add("")
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(rows)} measurements, {len(configs)} configurations)")


if __name__ == "__main__":
    main()
