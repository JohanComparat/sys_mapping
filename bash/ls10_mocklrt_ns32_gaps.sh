#!/bin/bash
# ==============================================================================
# ls10_mocklrt_ns32_gaps.sh — fill the 4 NSIDE-32 mock-calibrated LRT cells that
# still carry lrt.calibration == "failed", running them CONCURRENTLY.
#
# The 5 completed ns32 cells (logM* 10.0, 10.25, 10.5, 10.75, 11.0) are left
# alone; these 4 complete the 9/9 table at N=30, matching their seed/settings.
#
#   bash bash/ls10_mocklrt_ns32_gaps.sh          # N=30, 4 concurrent
#   NMOCK=50 JOBS=2 bash bash/ls10_mocklrt_ns32_gaps.sh
#
# Cost: ~2.6 h/cell measured from the completed runs -> ~3 h wall at JOBS=4.
# ==============================================================================
set -uo pipefail
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
# float64 is REQUIRED: float32 makes 1+sum(b.t) <= 0 on the near-degenerate LS10
# basis -> log(<=0) -> NaN at NUTS init ("comb: nan").
export JAX_ENABLE_X64=${JAX_ENABLE_X64:-1}
cd "$(dirname "$0")/.."

NMOCK=${NMOCK:-30}
JOBS=${JOBS:-4}
CATALOG_DIR=${CATALOG_DIR:-$HOME/data/legacysurvey/dr10/sweep/BGS_VLIM_Mstar}
SYST_BASE=${SYST_BASE:-$HOME/data/legacysurvey/dr10/systematics}
OUTBASE=${OUTBASE:-results/ls10_mocklrt}
NUTS=${NUTS:-1000}
NCHAINS=${NCHAINS:-4}
PY=${PY:-$HOME/mamba/envs/sys_map/bin/python}
NS=32
TDIR="$SYST_BASE/$(printf '%04d' "$NS")"
[ -d "$TDIR" ] || { echo "ERROR: no template dir $TDIR"; exit 1; }

# Share the 16 cores across $JOBS processes rather than letting each grab all.
THREADS=$(( $(nproc) / JOBS )); [ "$THREADS" -lt 1 ] && THREADS=1
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export XLA_FLAGS="--xla_force_host_platform_device_count=$NCHAINS"

SAMPLES=(
  LS10_VLIM_ANY_9.0_Mstar_12.0_0.05_z_0.08_N_0523486
  LS10_VLIM_ANY_9.5_Mstar_12.0_0.05_z_0.12_N_1432502
  LS10_VLIM_ANY_11.25_Mstar_12.0_0.05_z_0.35_N_0541855
  LS10_VLIM_ANY_11.5_Mstar_12.0_0.05_z_0.35_N_0120882
)

echo "ls10_mocklrt_ns32_gaps: ${#SAMPLES[@]} cells, N=$NMOCK, JOBS=$JOBS, ${THREADS} threads each"
echo "  catalog=$CATALOG_DIR  tdir=$TDIR  out=$OUTBASE  start=$(date '+%F %T')"

run_one() {
  local s="$1"
  echo "=== START $s  $(date '+%F %T') ==="
  "$PY" scripts/run_ls10_analysis.py \
    --catalog-dir "$CATALOG_DIR" --template-dir "$TDIR" \
    --sample "$s" --nside "$NS" \
    --only-methods MCMC-add MCMC-comb \
    --nuts-warmup "$NUTS" --nuts-samples "$NUTS" --n-chains "$NCHAINS" \
    --lrt-null-mocks "$NMOCK" --lrt-null-seed 90000 --force --no-rst \
    --output-dir "$OUTBASE/${s}_ns${NS}" \
    > "$OUTBASE/${s}_ns${NS}.log" 2>&1
  echo "=== DONE  $s rc=$? $(date '+%F %T') ==="
}
export -f run_one
export PY CATALOG_DIR TDIR NS NMOCK NUTS NCHAINS OUTBASE

pids=()
for s in "${SAMPLES[@]}"; do
  run_one "$s" &
  pids+=($!)
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
done
wait
echo "ALL DONE $(date '+%F %T') — collate $OUTBASE/*_ns32/*_params.json"
