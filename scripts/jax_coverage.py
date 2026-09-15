#!/usr/bin/env python
"""Static JAX coverage of the sys_mapping package.

Every call in the package is attributed to the library that provides it, from the import
aliases in scope (module level and inside functions): JAX (``jax``, ``jax.numpy``,
``jax.lax``, ``blackjax``, ``optax``), NumPy, SciPy, healpy, TreeCorr, Corrfunc,
scikit-learn, GLASS, emcee.  A *numeric line* is a source line holding at least one such
call; it is a JAX line when any of its calls is JAX.  Each function is classified by the
libraries its body calls: ``jax``, ``mixed``, ``other`` (numeric but no JAX) or
``none`` (no numeric call).

This measures where the code is written against JAX, not where time is spent; the dynamic
share comes from profiling a pipeline run.

Run::

    python scripts/jax_coverage.py [--csv docs/_static/coverage/jax_static.csv]
"""
from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "sys_mapping"

# Top-level module name -> library family.
FAMILY = {
    "jax": "jax", "blackjax": "jax", "optax": "jax", "jaxopt": "jax",
    "numpy": "numpy", "scipy": "scipy", "healpy": "healpy", "treecorr": "treecorr",
    "Corrfunc": "corrfunc", "sklearn": "sklearn", "glass": "glass", "emcee": "emcee",
}


def _aliases(tree: ast.AST) -> dict[str, str]:
    """Every name bound by an import anywhere in the module, mapped to its family."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                fam = FAMILY.get(a.name.split(".")[0])
                if fam:
                    out[(a.asname or a.name).split(".")[0]] = fam
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            fam = FAMILY.get(node.module.split(".")[0])
            if fam:
                for a in node.names:
                    out[a.asname or a.name] = fam
    return out


def _root(func: ast.expr) -> str | None:
    while isinstance(func, ast.Attribute):
        func = func.value
    if isinstance(func, ast.Call):
        return _root(func.func)
    return func.id if isinstance(func, ast.Name) else None


def _call_families(node: ast.AST, aliases: dict[str, str]):
    """Yield (line, family) for every attributed call under ``node``."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fam = aliases.get(_root(sub.func) or "")
            if fam:
                yield sub.lineno, fam


def _definitions(body, prefix=""):
    """Module- and class-level functions, including those defined inside ``if``/``try``
    blocks (the JAX kernels are defined under ``try: import jax``)."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield prefix, node
        elif isinstance(node, ast.ClassDef):
            yield from _definitions(node.body, prefix + node.name + ".")
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            for block in ("body", "orelse", "finalbody"):
                yield from _definitions(getattr(node, block, []), prefix)
            for handler in getattr(node, "handlers", []):
                yield from _definitions(handler.body, prefix)


def analyse_module(path: Path) -> dict:
    tree = ast.parse(path.read_text())
    aliases = _aliases(tree)
    line_fams: dict[int, set[str]] = defaultdict(set)
    for line, fam in _call_families(tree, aliases):
        line_fams[line].add(fam)
    numeric = len(line_fams)
    jax_lines = sum(1 for f in line_fams.values() if "jax" in f)
    by_family = Counter(f for fams in line_fams.values() for f in fams)

    functions = []
    for prefix, fn in _definitions(tree.body):
        fams = {fam for _, fam in _call_families(fn, aliases)}
        # A decorator such as ``@jax.jit`` is an attribute rather than a call.
        if any(aliases.get(_root(d) or "") == "jax" for d in fn.decorator_list):
            fams.add("jax")
        kind = ("none" if not fams else "jax" if fams == {"jax"}
                else "mixed" if "jax" in fams else "other")
        functions.append(dict(name=prefix + fn.name, kind=kind,
                              libraries=",".join(sorted(fams)),
                              lines=fn.end_lineno - fn.lineno + 1,
                              public=not fn.name.startswith("_")))
    return dict(module=path.stem, numeric=numeric, jax=jax_lines, by_family=by_family,
                functions=functions)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=None, help="write the per-function table here")
    args = ap.parse_args()

    results = [analyse_module(p) for p in sorted(PACKAGE.glob("*.py"))
               if p.name != "__init__.py"]
    tot_num = sum(r["numeric"] for r in results)
    tot_jax = sum(r["jax"] for r in results)
    print(f"{'module':16s} {'numeric':>8s} {'jax':>5s} {'share':>6s}  "
          f"functions jax/mixed/other/none  other libraries")
    for r in results:
        kinds = Counter(f["kind"] for f in r["functions"])
        share = r["jax"] / r["numeric"] if r["numeric"] else 0.0
        others = ", ".join(f"{k} {v}" for k, v in r["by_family"].most_common() if k != "jax")
        print(f"{r['module']:16s} {r['numeric']:8d} {r['jax']:5d} {share:6.0%}  "
              f"{kinds['jax']:3d}/{kinds['mixed']:3d}/{kinds['other']:3d}/{kinds['none']:3d}"
              f"          {others}")
    print(f"{'total':16s} {tot_num:8d} {tot_jax:5d} {tot_jax / tot_num:6.0%}")
    pub = [f for r in results for f in r["functions"] if f["public"] and f["kind"] != "none"]
    pk = Counter(f["kind"] for f in pub)
    print(f"public numeric functions: {len(pub)} — jax {pk['jax']}, mixed {pk['mixed']}, "
          f"other {pk['other']}")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["module", "function", "public", "lines", "kind", "libraries"])
            for r in results:
                for f in r["functions"]:
                    w.writerow([r["module"], f["name"], f["public"], f["lines"], f["kind"],
                                f["libraries"]])
        print(f"-> {out}")


if __name__ == "__main__":
    main()
