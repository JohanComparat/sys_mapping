#!/bin/bash
# Reproduce Berlfein 2024 paper results on synthetic mocks.
# Runs in the background via nohup using all 32 available cores.
#
# Usage:
#   bash/paper_validation.sh                           # default settings
#   bash/paper_validation.sh --nside 512 --n-real 119  # full paper settings
#
# Monitor:
#   tail -f logs/paper_validation_<timestamp>.log

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

if [[ -d "$HOME/miniforge3/envs/sys_map" ]]; then
    _MAMBA_ROOT="$HOME/miniforge3"
elif [[ -d "$HOME/mamba/envs/sys_map" ]]; then
    _MAMBA_ROOT="$HOME/mamba"
else
    echo "ERROR: sys_map environment not found in $HOME/miniforge3 or $HOME/mamba" >&2; exit 1
fi
source "$_MAMBA_ROOT/etc/profile.d/mamba.sh"
mamba activate sys_map

mkdir -p logs results/paper_validation

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/paper_validation_${TIMESTAMP}.log"

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$(nproc)"

echo "Starting paper_validation at ${TIMESTAMP}" | tee "$LOGFILE"
echo "Log: $LOGFILE"

nohup python scripts/run_paper_validation.py \
    --nside 64 \
    --n-sys 25 \
    --n-real 4 \
    --n-walkers 110 \
    --n-steps 200 \
    --n-burn 50 \
    --output-dir results/paper_validation/ \
    "$@" \
    >> "$LOGFILE" 2>&1 &

PID=$!
echo "PID $PID — nohup running, log: $LOGFILE"
echo $PID > "logs/paper_validation_${TIMESTAMP}.pid"
