#!/bin/bash
#SBATCH --job-name=sys_map_ls10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_ls10_analysis.out
#SBATCH --error=logs/%j_ls10_analysis.err

# ── Full sys_mapping pipeline on LS10 BGS galaxy catalogs ─────────────────
# Processes all *_DATA.fits / *_RAND.fits pairs found in --catalog-dir.
# For GPU execution, add --device gpu and ensure jax[cuda12] is installed.
#
# Submit (all samples):
#   sbatch slurm/ls10_analysis.sh --catalog-dir /path/to/BGS_VLIM_Mstar
# Single sample:
#   sbatch slurm/ls10_analysis.sh \
#       --catalog-dir /path/to/BGS_VLIM_Mstar \
#       --sample LS10_VLIM_ANY_10.5_Mstar_12.0_0.05_z_0.26_N_3263228

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

mkdir -p logs data/sys_weights

CATALOG_DIR=${CATALOG_DIR:-$HOME/data/legacysurvey/dr10/sweep/BGS_VLIM_Mstar}
TEMPLATE_DIR=${TEMPLATE_DIR:-$HOME/data/legacysurvey/dr10/systematics}

python scripts/run_ls10_analysis.py \
    --catalog-dir "$CATALOG_DIR" \
    --template-dir "$TEMPLATE_DIR" \
    --nside 64 \
    --n-walkers 210 \
    --n-steps 2100 \
    --n-burn 210 \
    --output-dir data/sys_weights/ \
    "$@"
