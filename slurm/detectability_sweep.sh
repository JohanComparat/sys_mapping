#!/bin/bash
#SBATCH --job-name=sys_map_detsweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_detectability_sweep.out
#SBATCH --error=logs/%j_detectability_sweep.err

# ── Detectability-law sweep (Stage 2) on the compute cluster ──────────────
# Traces A_min over nside × density × f_sky × amplitude. PREPARED, NOT YET RUN.
# Validate first with --check (zero compute), then submit the fast stages.
#
# Submit:
#   sbatch slurm/detectability_sweep.sh              # check + both sweeps + anchors
# Regenerate the doc pages too:
#   STAGES="all" sbatch slurm/detectability_sweep.sh
# Continue after a wall-clock timeout (skips the cells already in the CSVs):
#   STAGES=sweep_euclid RESUME=1 sbatch slurm/detectability_sweep.sh
#
# --time/--mem: at the default --n-sims 30 the full grid is ~10 core-h, and the
# largest cell (nside 1024) peaks near 3 GB. Run `bash bash/run_remote_full.sh
# check` first — it estimates both from the grid you are actually about to run.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -d "$HOME/miniforge3/envs/sys_map" ]]; then
    _MAMBA_ROOT="$HOME/miniforge3"
elif [[ -d "$HOME/mamba/envs/sys_map" ]]; then
    _MAMBA_ROOT="$HOME/mamba"
else
    echo "ERROR: sys_map environment not found in $HOME/miniforge3 or $HOME/mamba" >&2; exit 1
fi
source "$_MAMBA_ROOT/etc/profile.d/mamba.sh"; mamba activate sys_map
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" JAX_ENABLE_X64=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# mcmc_anchors is included by default: MCMC-add on the additive/non-skewed model
# dispatches to the closed-form analytic posterior, not a sampler, so the anchor
# grid costs ~2 min — there is no reason to hold it back for a separate submit.
STAGES="${STAGES:-check sweep_ls10 sweep_euclid mcmc_anchors}"
bash bash/run_remote_full.sh $STAGES
