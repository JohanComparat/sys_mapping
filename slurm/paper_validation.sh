#!/bin/bash
#SBATCH --job-name=sys_map_paper
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_paper_validation.out
#SBATCH --error=logs/%j_paper_validation.err

# ── Reproduce Berlfein 2024 paper results on synthetic mocks ──────────────
# Full paper settings: NSIDE=512, 119 realisations, 250 walkers, 1500 steps
# Reduced for quick test: NSIDE=64, 4 realisations (defaults below)
#
# Submit:  sbatch slurm/paper_validation.sh
# Override: sbatch slurm/paper_validation.sh --nside 512 --n-real 119

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

if [[ -d "$HOME/miniforge3/envs/sys_map" ]]; then
    _MAMBA_ROOT="$HOME/miniforge3"
elif [[ -d "$HOME/mamba/envs/sys_map" ]]; then
    _MAMBA_ROOT="$HOME/mamba"
else
    echo "ERROR: sys_map environment not found in $HOME/miniforge3 or $HOME/mamba" >&2; exit 1
fi
source "$_MAMBA_ROOT/etc/profile.d/mamba.sh" && mamba activate sys_map

mkdir -p logs

python scripts/run_paper_validation.py \
    --nside 64 \
    --n-sys 25 \
    --n-real 4 \
    --n-walkers 110 \
    --n-steps 200 \
    --n-burn 50 \
    --output-dir results/paper_validation/ \
    "$@"
