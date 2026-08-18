#!/bin/bash
# ==============================================================================
# run_remote_full.sh — FULL detectability-law sweep for a REMOTE server.
#
# STAGE 2 of the systematic-detectability investigation: trace the analytic
# curves on docs/detectability_law.rst empirically over the full
# `nside × density × f_sky × amplitude` grid and pin the clustering floor with
# MCMC anchors. PREPARED, NOT YET RUN — this is the saved-for-later recipe.
#
# Stages are independent; pass one or more names, or "all":
#   check         validate inputs + print the plan for EVERY stage grid (NO compute)
#   sweep_ls10    fast-method sweep over the grid, LS10-like geometries
#   sweep_euclid  fast-method sweep over the grid, Euclid-like geometries (fine pixels, small f_sky)
#   mcmc_anchors  MCMC-add at a few (nside, density) cells to pin the clustering floor
#   docs          regenerate the detectability_law + synthesis pages and clean-build
#
# Examples:
#   bash bash/run_remote_full.sh check
#   nohup bash bash/run_remote_full.sh sweep_ls10 sweep_euclid > logs/detsweep.log 2>&1 &
#   NSIMS=50 bash bash/run_remote_full.sh all
#   RESUME=1 bash bash/run_remote_full.sh sweep_euclid      # continue after a timeout
#
# Tunables (env): NSIMS (sims/config, 30), NMOCK (sandwich mocks, 100),
#   METHODS ("OLS ISD-1"), NSIDES, N_MEANS, FSKYS, AMPS, OUT
#   (results/detectability_sweep.csv), RESUME (0/1).
# Cost: run `check` — it estimates core-h and peak RSS from the actual grid.
#   Run under nohup/tmux; tail logs/.
#
# results/ is gitignored, so the CSVs do NOT come back with `git pull`. Fetch them
# before running the `docs` stage locally:
#   rsync -av <host>:<path>/sys_mapping/results/detectability_sweep_*.csv results/
# ==============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

if   [[ -d "$HOME/miniforge3/envs/sys_map" ]]; then _MR="$HOME/miniforge3"
elif [[ -d "$HOME/mamba/envs/sys_map"      ]]; then _MR="$HOME/mamba"
else echo "ERROR: sys_map conda env not found (expected \$HOME/mamba or \$HOME/miniforge3)"; exit 1; fi
source "$_MR/etc/profile.d/mamba.sh"; mamba activate sys_map; PY=python
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
mkdir -p logs results

NSIMS="${NSIMS:-30}"
NMOCK="${NMOCK:-100}"
METHODS="${METHODS:-OLS ISD-1}"
OUT="${OUT:-results/detectability_sweep.csv}"
RESUME="${RESUME:-0}"
STAGES="${*:-check}"

# Per-stage grids, defined once so `check` validates exactly what the compute
# stages will run. These MUST expand unquoted at the call sites below — each axis
# is a multi-value list that has to word-split into separate argv entries, and
# quoting it hands argparse one string ("invalid int value: '32 64 128 256'").
# `:-` (not `-`) is right here: an empty axis should fall back to the default, not
# mean "empty grid" (unlike the ONLY_METHODS="" case fixed in cc4e088).
LS10_GRID="--nsides ${NSIDES:-32 64 128 256} --n-means ${N_MEANS:-8 30 127 490} \
--fskys ${FSKYS:-0.1 0.25 0.44} --amps ${AMPS:-0.005 0.01 0.03 0.05 0.1}"
EUCLID_GRID="--nsides ${NSIDES:-256 512 1024} --n-means ${N_MEANS:-8 14 30} \
--fskys ${FSKYS:-0.007 0.02 0.05} --amps ${AMPS:-0.003 0.005 0.01 0.03 0.05}"
MCMC_GRID="--nsides ${NSIDES:-64 256} --n-means ${N_MEANS:-30 127} \
--fskys ${FSKYS:-0.25} --amps ${AMPS:-0.01 0.05}"

COMMON="--n-sims $NSIMS --n-mock $NMOCK"
[[ "$RESUME" == "1" ]] && COMMON="$COMMON --resume"

rc=0; failed=""
run(){ echo -e "\n\033[1;36m== $* ==\033[0m"; }
have(){ case " $STAGES " in *" $1 "*|*" all "*) return 0;; *) return 1;; esac; }
# Stages are independent, so a failure does not abort the rest — but it must be
# recorded, or a dead 10-hour stage still ends in a green DONE. $name is the
# user-facing stage name, so "$failed" is directly re-runnable; dedupe it because
# some stages invoke track more than once.
track(){
  local name="$1"; shift
  "$@" && return 0
  rc=$?
  case " $failed " in *" $name "*) :;; *) failed="$failed $name";; esac
}

# check_grid <name> <methods> <grid args...> — the grid must arrive already
# word-split (call it unquoted), so "$@" here is one argv entry per token.
check_grid(){
  local name="$1" meth="$2"; shift 2
  echo -e "\n--- grid: $name ---"
  track check $PY scripts/run_detectability_sweep.py "$@" \
      --check $COMMON --methods $meth --out "${OUT%.csv}_${name}.csv"
}

if have check; then
  run "STAGE check — inputs + plan for every stage grid (no compute)"
  check_grid ls10   "$METHODS" $LS10_GRID
  check_grid euclid "$METHODS" $EUCLID_GRID
  check_grid mcmc   MCMC-add   $MCMC_GRID   # the anchor stage overrides METHODS
fi

if have sweep_ls10; then
  run "STAGE sweep_ls10 — wide/dense/coarse geometries"
  track sweep_ls10 $PY scripts/run_detectability_sweep.py $LS10_GRID \
      $COMMON --methods $METHODS --out "${OUT%.csv}_ls10.csv"
fi

if have sweep_euclid; then
  run "STAGE sweep_euclid — deep/fine-pixel/small-f_sky geometries"
  track sweep_euclid $PY scripts/run_detectability_sweep.py $EUCLID_GRID \
      $COMMON --methods $METHODS --out "${OUT%.csv}_euclid.csv"
fi

if have mcmc_anchors; then
  run "STAGE mcmc_anchors — MCMC-add floor anchors"
  track mcmc_anchors $PY scripts/run_detectability_sweep.py $MCMC_GRID \
      $COMMON --methods MCMC-add --out "${OUT%.csv}_mcmc.csv"
fi

if have docs; then
  run "STAGE docs — regenerate + clean build"
  track docs $PY scripts/analyze_detectability_law.py
  track docs $PY scripts/make_survey_design_synthesis.py
  track docs bash bash/build_docs.sh
fi

if [[ $rc -eq 0 ]]; then
  echo -e "\n\033[1;32mDONE: $STAGES\033[0m"
else
  echo -e "\n\033[1;31mFAILED:$failed\033[0m (exit $rc; other stages in '$STAGES' may have succeeded)"
  echo "continue a partial sweep with:  RESUME=1 bash bash/run_remote_full.sh$failed"
fi
exit $rc
