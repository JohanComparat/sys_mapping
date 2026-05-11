#!/usr/bin/env bash
# run_all_methods_sequential.sh
#
# Runs the LS10 BGS dataset for each fitting method in speed order:
#
#   Phase 1 — OLS        (seconds)
#   Phase 2 — ISD-1      (seconds)
#   Phase 3 — ElasticNet (minutes)
#   Phase 4 — ISD-3      (minutes)
#   Phase 5 — MCMC-add   (hours)
#   Phase 6 — MCMC-comb  (hours, full pipeline: 2PCF, weight maps, LRT)
#
# After each phase completes, Sphinx documentation is rebuilt so intermediate
# results are immediately browsable.
#
# Usage:
#   nohup bash scripts/run_all_methods_sequential.sh > logs/run_all.log 2>&1 &
#
# Tune via environment variables:
#   DEVICE=cpu                   JAX device: cpu | gpu | auto (default: cpu)
#   CATALOG_DIR=<path>           LS10 BGS catalog directory
#   TEMPLATE_DIR=<path>          LS10 systematic template directory (optional)
#   LS10_OUTPUT_DIR=data/sys_weights/
#   LS10_NSIDE=64                HEALPix NSIDE for LS10 (default: 64)
#   SKIP_LS10=1                  Skip LS10 dataset entirely
#   METHODS="OLS ISD-1 ElasticNet ISD-3 MCMC-add MCMC-comb"  Phases to run (default: all)
#   FORCE=1                      Re-run even if output already exists
#
# Examples:
#   # OLS-only quick preview (no MCMC)
#   METHODS="OLS" bash scripts/run_all_methods_sequential.sh
#
#   # Resume from MCMC-add phase (OLS/ElasticNet already done)
#   METHODS="MCMC-add MCMC-comb" bash scripts/run_all_methods_sequential.sh
#
#   # Run on GPU
#   DEVICE=gpu bash scripts/run_all_methods_sequential.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Python / Sphinx resolution ────────────────────────────────────────────────
if [[ -d "$HOME/miniforge3/envs/sys_map" ]]; then
    PYTHON="$HOME/miniforge3/envs/sys_map/bin/python"
    SPHINX="$HOME/miniforge3/envs/sys_map/bin/sphinx-build"
elif [[ -d "$HOME/mamba/envs/sys_map" ]]; then
    PYTHON="$HOME/mamba/envs/sys_map/bin/python"
    SPHINX="$HOME/mamba/envs/sys_map/bin/sphinx-build"
else
    echo "ERROR: sys_map conda environment not found in ~/miniforge3 or ~/mamba" >&2
    exit 1
fi

# ── Configuration (override via env) ─────────────────────────────────────────
DEVICE="${DEVICE:-cpu}"
CATALOG_DIR="${CATALOG_DIR:-$HOME/data/legacysurvey/dr10/sweep/BGS_VLIM_Mstar}"
TEMPLATE_DIR="${TEMPLATE_DIR:-}"
LS10_OUTPUT_DIR="${LS10_OUTPUT_DIR:-data/sys_weights/}"
LS10_NSIDE="${LS10_NSIDE:-64}"
SKIP_LS10="${SKIP_LS10:-0}"
METHODS="${METHODS:-OLS ISD-1 ElasticNet ISD-3 MCMC-add MCMC-comb}"
FORCE="${FORCE:-0}"
DOCS_DIR="$(pwd)/docs"

# ── JAX device selection ──────────────────────────────────────────────────────
# cpu: force CPU (avoids accidental GPU use on login nodes)
# gpu/cuda/auto: leave JAX_PLATFORMS unset so JAX auto-detects
if [[ "$DEVICE" == "cpu" ]]; then
    export JAX_PLATFORMS=cpu
fi

# ── Thread counts: all cores per sequential phase ─────────────────────────────
export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$(nproc)"

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Helpers ───────────────────────────────────────────────────────────────────
rebuild_docs() {
    local TAG="$1"
    echo ""
    echo "  [docs] Rebuilding Sphinx after ${TAG} phase ..."
    "$SPHINX" -b html -E -a "$DOCS_DIR" "$DOCS_DIR/_build/html" -q \
        && echo "  [docs] Done — $DOCS_DIR/_build/html" \
        || echo "  [docs] WARNING: sphinx-build returned non-zero; continuing."
}

elapsed_human() {
    local secs=$(( $(date +%s) - $1 ))
    printf "%dh%02dm%02ds" $(( secs/3600 )) $(( (secs%3600)/60 )) $(( secs%60 ))
}

# ── Header ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  run_all_methods_sequential.sh"
echo "  Start     : $(date)"
echo "  Device    : $DEVICE"
echo "  Methods   : $METHODS"
echo "  SKIP_LS10 : $SKIP_LS10"
echo "  FORCE     : $FORCE"
echo "  Docs      : $DOCS_DIR/_build/html"
echo "============================================================"

# ── Main loop: method is the outer loop ──────────────────────────────────────
for METHOD in $METHODS; do
    PHASE_T0=$(date +%s)
    echo ""
    echo "============================================================"
    echo "  PHASE: $METHOD   [$(date)]"
    echo "============================================================"

    # Build optional --force flag for Python scripts
    FORCE_FLAG=()
    [[ "$FORCE" -eq 1 ]] && FORCE_FLAG=(--force)

    # ── LS10 BGS ────────────────────────────────────────────────────────────
    if [[ "$SKIP_LS10" -eq 0 ]]; then
        echo ""
        echo "  LS10 BGS  — $METHOD"
        LS10_TMPL_ARGS=()
        [[ -n "$TEMPLATE_DIR" ]] && LS10_TMPL_ARGS=(--template-dir "$TEMPLATE_DIR")
        "$PYTHON" scripts/run_ls10_analysis.py \
            --catalog-dir "$CATALOG_DIR" \
            "${LS10_TMPL_ARGS[@]}" \
            --only-methods "$METHOD" \
            --nside "$LS10_NSIDE" \
            --output-dir "$LS10_OUTPUT_DIR" \
            "${FORCE_FLAG[@]}" \
            2>&1 | tee "logs/ls10_${METHOD}_${TIMESTAMP}.log"
        echo "  LS10 done."
    else
        echo "  [LS10] Skipped (SKIP_LS10=1)"
    fi

    # ── Rebuild docs after this method phase ─────────────────────────────────
    rebuild_docs "$METHOD"

    echo ""
    echo "  === $METHOD complete in $(elapsed_human $PHASE_T0) ==="
done

echo ""
echo "============================================================"
echo "  All phases complete.  [$(date)]"
echo "  Docs : $DOCS_DIR/_build/html"
echo "============================================================"
