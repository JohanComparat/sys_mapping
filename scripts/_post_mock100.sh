#!/bin/bash
# Post-processing for the 100-mock validation run.
# Copies summary PNGs to the Sphinx static tree and rebuilds docs.
#
# Usage: bash scripts/_post_mock100.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -d "$HOME/miniforge3/envs/sys_map" ]]; then
    _MAMBA_ROOT="$HOME/miniforge3"
elif [[ -d "$HOME/mamba/envs/sys_map" ]]; then
    _MAMBA_ROOT="$HOME/mamba"
else
    echo "ERROR: sys_map environment not found in $HOME/miniforge3 or $HOME/mamba" >&2; exit 1
fi

OUTDIR="results/mock_analysis_100"
STATICDIR="docs/_static/results_mock_analysis"

mkdir -p "$STATICDIR"

# Copy summary PNGs
cp "$OUTDIR/mock_parameter_recovery_all_methods.png" "$STATICDIR/"
cp "$OUTDIR/mock_sigma_recovery.png"                 "$STATICDIR/"
cp "$OUTDIR/mock_lrt_statistics.png"                 "$STATICDIR/"

# Rebuild docs
source "$_MAMBA_ROOT/etc/profile.d/mamba.sh" && mamba activate sys_map
make -C docs html

echo "=== Post-processing complete ==="
