#!/bin/bash
# ==============================================================================
# ls10_mocklrt.sh — mock-calibrated Likelihood-Ratio Test across the LS10 VLIM
# samples, SEQUENTIALLY (one fit at a time).
#
# The Wilks chi^2 p-value of the additive-vs-combined LRT is overconfident on a
# spatially-correlated field. This runs run_ls10_analysis.py with --lrt-null-mocks,
# which builds an empirical lambda_LR null from uncontaminated GLASS mocks and reports
# a mock-calibrated p-value (params.json: lrt.p_value = mock, lrt.p_chi2 = Wilks,
# lrt.null_lambda = the null array). See docs/results_ls10.rst (LRT section).
#
# Usage:
#   bash/ls10_mocklrt.sh                 # nside "32 64", N=50  (all 9 samples)
#   bash/ls10_mocklrt.sh "32"  20        # nside 32 only, N=20
#   bash/ls10_mocklrt.sh "32 64 128" 50
#   RESUME=1 bash/ls10_mocklrt.sh "32" 80   # top up an existing N=50 null to N=80 (adds only
#                                           # the 30 missing mocks; no data re-fit). Same NUTS/NCHAINS.
#
# Env overrides:
#   CATALOG_DIR  BGS VLIM catalogs  (default ~/…/sweep/BGS_VLIM_Mstar)
#   SYST_BASE    systematics base with per-nside subdirs 0032/ 0064/ …
#                (default ~/…/dr10/systematics)   ← templates live in the NSIDE
#                subdir; pointing at the parent makes the pipeline silently fall
#                back to SYNTHETIC templates.
#   OUTBASE      output root (default results/ls10_mocklrt)
#   NUTS         NUTS warmup=samples per fit (default 1000)
#   NCHAINS      NUTS chains (default 4)
#   ONLY_METHODS methods to fit (default "MCMC-add MCMC-comb"; the LRT needs both.
#                Set to "" to run the full method set — much slower.)
#
# Cost: MCMC-comb on the 11 near-degenerate LS10 templates is ~5–7 min/fit, so
# each sample's N-mock null is ~N×(that). Run under nohup/tmux; tail $OUTBASE/*.log.
# ==============================================================================
set -uo pipefail
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
# float64 is REQUIRED: the combined model NUTS returns NaN acceptance ("comb: nan")
# under float32 (log of a non-positive 1+Σb·t on near-degenerate templates). A stale
# site-packages sys_mapping can shadow the repo's x64 call, so force it at the env level.
export JAX_ENABLE_X64=${JAX_ENABLE_X64:-1}
cd "$(dirname "$0")/.."

NSIDES=${1:-"32 64"}
NMOCK=${2:-50}
CATALOG_DIR=${CATALOG_DIR:-$HOME/data/legacysurvey/dr10/sweep/BGS_VLIM_Mstar}
SYST_BASE=${SYST_BASE:-$HOME/data/legacysurvey/dr10/systematics}
OUTBASE=${OUTBASE:-results/ls10_mocklrt}
NUTS=${NUTS:-1000}
NCHAINS=${NCHAINS:-4}
ONLY_METHODS=${ONLY_METHODS:-"MCMC-add MCMC-comb"}
PY=${PY:-python}
# RESUME=1 → top-up mode: add mocks to an existing null up to $NMOCK (only runs the missing
# ones, no data re-fit). Re-run with a larger NMOCK to tighten the mock-p floor 1/(N+1).
# Use the SAME NUTS/NCHAINS as the original run.
RESUME=${RESUME:-0}
mkdir -p "$OUTBASE"
resume_args=(); logsfx=""; [ "$RESUME" = 1 ] && { resume_args=(--resume-null); logsfx=".resume"; }

# All *_DATA.fits samples in the catalog dir (base sample ids).
mapfile -t SAMPLES < <(ls "$CATALOG_DIR"/*_DATA.fits 2>/dev/null | sed 's#.*/##; s/_DATA.fits//' | sort)
if [ ${#SAMPLES[@]} -eq 0 ]; then echo "ERROR: no *_DATA.fits in $CATALOG_DIR"; exit 1; fi

echo "ls10_mocklrt: ${#SAMPLES[@]} samples × nsides [$NSIDES], N=$NMOCK, NUTS=$NUTS, chains=$NCHAINS"
echo "  catalog=$CATALOG_DIR  syst_base=$SYST_BASE  out=$OUTBASE"

only_args=(); [ -n "$ONLY_METHODS" ] && only_args=(--only-methods $ONLY_METHODS)

for ns in $NSIDES; do
  tdir="$SYST_BASE/$(printf '%04d' "$ns")"
  [ -d "$tdir" ] || { echo "WARNING: no template dir $tdir — skipping nside $ns"; continue; }
  for s in "${SAMPLES[@]}"; do
    echo "=== $s  nside=$ns  $(date '+%F %T') ==="
    "$PY" scripts/run_ls10_analysis.py \
      --catalog-dir "$CATALOG_DIR" --template-dir "$tdir" \
      --sample "$s" --nside "$ns" "${only_args[@]}" \
      --nuts-warmup "$NUTS" --nuts-samples "$NUTS" --n-chains "$NCHAINS" \
      --lrt-null-mocks "$NMOCK" --lrt-null-seed 90000 --force --no-rst "${resume_args[@]}" \
      --output-dir "$OUTBASE/${s}_ns${ns}" > "$OUTBASE/${s}_ns${ns}${logsfx}.log" 2>&1
    echo "    done (rc=$?)"
  done
done
echo "ALL DONE $(date '+%F %T')  — collate $OUTBASE/*/*_params.json (lrt.p_value vs lrt.p_chi2)"
