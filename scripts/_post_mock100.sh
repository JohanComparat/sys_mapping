#!/bin/bash
# Copy the products of the 100-mock synthetic run into the Sphinx static tree.
#
#   bash scripts/_post_mock100.sh [OUTDIR]
#
# OUTDIR is the --output-dir of
#   python scripts/run_mock_analysis.py --synthetic --n-mocks 100 --n-sys 3 --nside 64
# (default results/mock_analysis_100).  The per-mock JSON files stay in OUTDIR;
# the page reads the figures, the CSV and the ISD calibration.

set -euo pipefail
cd "$(dirname "$0")/.."

OUTDIR="${1:-results/mock_analysis_100}"
STATICDIR="docs/_static/results_mock_analysis"

mkdir -p "$STATICDIR"
for f in mock_parameter_recovery_all_methods.png mock_sigma_recovery.png \
         mock_lrt_statistics.png mock_b_parameter_recovery.png \
         mock_results.csv isd_chi2_68.json; do
    cp "$OUTDIR/$f" "$STATICDIR/"
done
echo "copied $OUTDIR -> $STATICDIR; rebuild the docs with bash bash/build_docs.sh"
