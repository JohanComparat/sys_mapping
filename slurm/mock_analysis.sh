#!/bin/bash
#SBATCH --job-name=sys_map_mock
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_mock%a.out
#SBATCH --error=logs/%j_mock%a.err

# ── sys_mapping validation on mock galaxy catalogs ─────────────────────────
# Run as an array job (one task per mock batch) or sequentially.
#
# Synthetic mocks (no data files needed):
#   sbatch slurm/mock_analysis.sh --synthetic --n-mocks 20
#
# Real mocks from disk:
#   sbatch --array=0-9 slurm/mock_analysis.sh \
#       --mock-dir /path/to/mocks \
#       --rand-file /path/to/randoms.fits

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -d "$HOME/miniforge3/envs/sys_map" ]]; then
    _MAMBA_ROOT="$HOME/miniforge3"
elif [[ -d "$HOME/mamba/envs/sys_map" ]]; then
    _MAMBA_ROOT="$HOME/mamba"
else
    echo "ERROR: sys_map environment not found in $HOME/miniforge3 or $HOME/mamba" >&2; exit 1
fi
source "$_MAMBA_ROOT/etc/profile.d/mamba.sh" && mamba activate sys_map

mkdir -p logs results/mock_analysis

MOCK_DIR=${MOCK_DIR:-$HOME/data/legacysurvey/dr10/sweep/BGS_VLIM_Mstar/mocks}
RAND_FILE=${RAND_FILE:-$HOME/data/legacysurvey/dr10/randoms/resolve/randoms-1-0-BGS.fits}
TEMPLATE_DIR=${TEMPLATE_DIR:-$HOME/data/legacysurvey/dr10/systematics}

python scripts/run_mock_analysis.py \
    --mock-dir "$MOCK_DIR" \
    --rand-file "$RAND_FILE" \
    --template-dir "$TEMPLATE_DIR" \
    --nside 64 \
    --n-walkers 110 \
    --n-steps 1000 \
    --n-burn 200 \
    --output-dir results/mock_analysis/ \
    "$@"
