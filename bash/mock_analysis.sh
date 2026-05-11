#!/bin/bash
# Validate sys_mapping on mock galaxy catalogs.
# Runs in the background via nohup using all 32 available cores.
#
# Usage:
#   bash/mock_analysis.sh --synthetic --n-mocks 20    # synthetic, no files needed
#   bash/mock_analysis.sh \
#       --mock-dir /path/to/mocks \
#       --rand-file /path/to/randoms.fits
#   TEMPLATE_DIR=/path/to/systematics bash/mock_analysis.sh --synthetic
#
# Monitor:
#   tail -f logs/mock_analysis_<timestamp>.log

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

mkdir -p logs results/mock_analysis

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/mock_analysis_${TIMESTAMP}.log"

MOCK_DIR=${MOCK_DIR:-$HOME/data/legacysurvey/dr10/sweep/BGS_VLIM_Mstar/mocks}
RAND_FILE=${RAND_FILE:-$HOME/data/legacysurvey/dr10/randoms/resolve/randoms-1-0-BGS.fits}
TEMPLATE_DIR=${TEMPLATE_DIR:-$HOME/data/legacysurvey/dr10/systematics}

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$(nproc)"

echo "Starting mock_analysis at ${TIMESTAMP}" | tee "$LOGFILE"
echo "Log: $LOGFILE"

nohup python scripts/run_mock_analysis.py \
    --mock-dir "$MOCK_DIR" \
    --rand-file "$RAND_FILE" \
    --template-dir "$TEMPLATE_DIR" \
    --nside 64 \
    --n-walkers 110 \
    --n-steps 1000 \
    --n-burn 200 \
    --output-dir results/mock_analysis/ \
    "$@" \
    >> "$LOGFILE" 2>&1 &

PID=$!
echo "PID $PID — nohup running, log: $LOGFILE"
echo $PID > "logs/mock_analysis_${TIMESTAMP}.pid"
