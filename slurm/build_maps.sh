#!/bin/bash
#SBATCH --job-name=sys_map_build_maps
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_build_maps.out
#SBATCH --error=logs/%j_build_maps.err

# ── Build HEALPix systematic template maps ────────────────────────────────
# Run once to generate GAIA and LS10 maps at all NSIDEs.
# These outputs are then consumed by the analysis pipelines.
#
# Build LS10 maps:
#   sbatch slurm/build_maps.sh --source ls10
#
# Build GAIA maps:
#   sbatch slurm/build_maps.sh --source gaia
#
# Both sources, specific NSIDE:
#   sbatch slurm/build_maps.sh --source ls10 --nside 64 128
#   sbatch slurm/build_maps.sh --source gaia  --nside 64 128

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

mkdir -p logs

OUTPUT_DIR=${OUTPUT_DIR:-$HOME/data/legacysurvey/dr10/systematics}

python scripts/build_systematic_maps.py \
    --output-dir "$OUTPUT_DIR" \
    --nside 32 64 128 256 \
    "$@"
