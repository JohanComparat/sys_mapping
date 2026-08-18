#!/usr/bin/env bash
# Build the documentation with a CLEAN, full pass.
#
# Sphinx renders the left navigation (globaltoc) into every page. An INCREMENTAL build
# (`make html`) only re-renders the pages whose source changed, so after a toctree change
# the unchanged pages keep a stale sidebar and the left column differs between pages.
# Always building clean (-E reads all sources, -a writes all outputs, and we delete the
# previous build) guarantees the sidebar is identical on every page.
#
# Usage:  bash bash/build_docs.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DOCS="$REPO/docs"
# Default to whatever `python` the caller has activated — a hardcoded laptop path
# does not exist on the compute host, and run_remote_full.sh has already activated
# the sys_map env by the time it calls this. Override with SYS_MAP_PYTHON.
PY="${SYS_MAP_PYTHON:-$(command -v python || true)}"
if [[ -z "$PY" || ! -x "$PY" ]]; then
    echo "ERROR: no python found. Activate the sys_map env, or set SYS_MAP_PYTHON=/path/to/python" >&2
    exit 1
fi

rm -rf "$DOCS/_build/html" "$DOCS/_build/doctrees"
JAX_PLATFORMS=cpu "$PY" -m sphinx -E -a -b html "$DOCS" "$DOCS/_build/html"
echo "Docs built: $DOCS/_build/html/index.html"
