#!/bin/bash
# Build HEALPix systematic template maps (GAIA DR2 or LS10 BGS randoms).
# Runs in the background via nohup using all 32 available cores.
#
# Usage:
#   bash/build_maps.sh --source ls10
#   bash/build_maps.sh --source gaia
#   bash/build_maps.sh --source ls10 --nside 64 128
#   OUTPUT_DIR=/path/to/out bash/build_maps.sh --source ls10
#
# Monitor:
#   tail -f logs/build_maps_<timestamp>.log

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -d "$HOME/miniforge3/envs/sys_map" ]]; then
    _MAMBA_ROOT="$HOME/miniforge3"
elif [[ -d "$HOME/mamba/envs/sys_map" ]]; then
    _MAMBA_ROOT="$HOME/mamba"
else
    echo "ERROR: sys_map environment not found in $HOME/miniforge3 or $HOME/mamba" >&2; exit 1
fi
source "$_MAMBA_ROOT/etc/profile.d/mamba.sh"
mamba activate sys_map

mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/build_maps_${TIMESTAMP}.log"

OUTPUT_DIR=${OUTPUT_DIR:-$HOME/data/legacysurvey/dr10/systematics}

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

echo "Starting build_maps at ${TIMESTAMP}" | tee "$LOGFILE"
echo "  OUTPUT_DIR=$OUTPUT_DIR" | tee -a "$LOGFILE"
echo "Log: $LOGFILE"

nohup python scripts/build_systematic_maps.py \
    --output-dir "$OUTPUT_DIR" \
    --nside 32 64 128 256 \
    "$@" \
    >> "$LOGFILE" 2>&1 &

PID=$!
echo "PID $PID — nohup running, log: $LOGFILE"
echo $PID > "logs/build_maps_${TIMESTAMP}.pid"
