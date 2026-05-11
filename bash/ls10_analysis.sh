#!/bin/bash
# Full sys_mapping pipeline on LS10 BGS galaxy catalogs.
# Runs in the background via nohup using all 32 available cores.
#
# Usage:
#   bash/ls10_analysis.sh                             # all samples
#   bash/ls10_analysis.sh --sample LS10_VLIM_ANY_...  # single sample
#   CATALOG_DIR=/path/to/BGS bash/ls10_analysis.sh
#
# Monitor:
#   tail -f logs/ls10_analysis_<timestamp>.log

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

mkdir -p logs data/sys_weights

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/ls10_analysis_${TIMESTAMP}.log"

CATALOG_DIR=${CATALOG_DIR:-$HOME/data/legacysurvey/dr10/sweep/BGS_VLIM_Mstar}
TEMPLATE_DIR=${TEMPLATE_DIR:-$HOME/data/legacysurvey/dr10/systematics}

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$(nproc)"

echo "Starting ls10_analysis at ${TIMESTAMP}" | tee "$LOGFILE"
echo "  CATALOG_DIR=$CATALOG_DIR" | tee -a "$LOGFILE"
echo "  TEMPLATE_DIR=$TEMPLATE_DIR" | tee -a "$LOGFILE"
echo "Log: $LOGFILE"

nohup python scripts/run_ls10_analysis.py \
    --catalog-dir "$CATALOG_DIR" \
    --template-dir "$TEMPLATE_DIR" \
    --nside 64 \
    --n-walkers 210 \
    --n-steps 2100 \
    --n-burn 210 \
    --output-dir data/sys_weights/ \
    "$@" \
    >> "$LOGFILE" 2>&1 &

PID=$!
echo "PID $PID — nohup running, log: $LOGFILE"
echo $PID > "logs/ls10_analysis_${TIMESTAMP}.pid"
