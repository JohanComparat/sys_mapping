#!/usr/bin/env python
"""Attribute the wall time of a profiled run to the libraries that spent it.

Reads a ``cProfile`` dump and sums each function's *own* time (``tottime``) by the library
its code lives in: JAX (``jax``, ``jaxlib``, ``blackjax``, ``optax``, including the time
compiled XLA code runs inside ``jaxlib`` built-ins), NumPy, SciPy, healpy, TreeCorr,
Corrfunc, scikit-learn, GLASS, emcee, astropy, ``sys_mapping`` Python, the calling script,
and the rest.  Optionally counts JAX compilations from a log captured with
``JAX_LOG_COMPILES=1``.

Run::

    JAX_LOG_COMPILES=1 python -m cProfile -o run.prof scripts/run_ls10_analysis.py ... 2> run.err
    python scripts/profile_by_library.py run.prof --compile-log run.err
"""
from __future__ import annotations

import argparse
import pstats
import re
from collections import defaultdict

# Ordered: the first pattern that matches a file path (or built-in name) wins.
LIBRARIES = [
    ("jax", re.compile(r"(/|^)(jax|jaxlib|blackjax|optax|jaxopt|equinox)(/|_)|jaxlib|xla_extension|pjit|_jax")),
    ("treecorr", re.compile(r"treecorr")),
    ("corrfunc", re.compile(r"Corrfunc")),
    ("healpy", re.compile(r"healpy")),
    ("glass", re.compile(r"/glass/|cosmology|camb")),
    ("sklearn", re.compile(r"sklearn")),
    ("scipy", re.compile(r"scipy")),
    ("astropy", re.compile(r"astropy|fitsio")),
    ("emcee", re.compile(r"emcee")),
    ("numpy", re.compile(r"numpy|method 'reduce' of 'numpy|numpy\.")),
    ("sys_mapping", re.compile(r"/sys_mapping/sys_mapping/")),
    ("script", re.compile(r"/sys_mapping/scripts/")),
]


def classify(filename: str, funcname: str) -> str:
    key = f"{filename} {funcname}"
    for name, pattern in LIBRARIES:
        if pattern.search(key):
            return name
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("profile")
    ap.add_argument("--compile-log", default=None)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    stats = pstats.Stats(args.profile)
    by_lib: dict[str, float] = defaultdict(float)
    rows = []
    for (filename, _line, funcname), (_cc, _nc, tottime, cumtime, _callers) in stats.stats.items():
        lib = classify(filename, funcname)
        by_lib[lib] += tottime
        rows.append((tottime, lib, f"{filename.split('site-packages/')[-1]}:{funcname}"))
    total = sum(by_lib.values())
    print(f"total own time {total:.1f} s")
    for lib, t in sorted(by_lib.items(), key=lambda kv: -kv[1]):
        print(f"  {lib:12s} {t:9.1f} s  {t / total:6.1%}")
    print(f"\ntop {args.top} functions by own time")
    for tottime, lib, where in sorted(rows, reverse=True)[:args.top]:
        print(f"  {tottime:8.1f} s  {lib:10s} {where}")

    if args.compile_log:
        text = open(args.compile_log, errors="replace").read()
        n = len(re.findall(r"Compiling |Finished XLA compilation", text))
        traced = re.findall(r"Finished tracing \+ transforming (\S+)", text)
        print(f"\nJAX compilations: {n}; traced functions: {len(traced)}")
        counts = defaultdict(int)
        for name in traced:
            counts[name] += 1
        for name, c in sorted(counts.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {c:5d}  {name}")


if __name__ == "__main__":
    main()
