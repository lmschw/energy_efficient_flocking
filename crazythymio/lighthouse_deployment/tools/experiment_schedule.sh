#!/bin/bash
# usage: tools/experiment_schedule.sh <repetitions per condition> <duration_s>
#   e.g. tools/experiment_schedule.sh 30 60
# Fixed, deterministic, RESUMABLE schedule: all repetitions of each condition in turn, in this order:
#   1. lj       -- the baseline (paper rules, scaled to LJ_R0 = 0.5 m, see below)
#   2. plain    -- evolved genome plain_seed123_best.npy (trained without the safety clamp)
#   3. clamped  -- evolved genome plain_seed123_clamped_best.npy (kept for last)
# Each entry: placement instructions -> Enter -> run all 7 -> logs into hardware_runs/<condition>_rep<k>_<stamp>/.
# Ctrl-C stops (emergency stop); run the same command again to resume -- finished repetitions are detected from the
# existing folders. Env overrides: CONDITIONS="lj plain clamped" (order), LJ_R0=0.7 LJ_SCALE_EPS= (paper spacing),
# CORRIDOR_X="-1.64 1.20".  ROBOTS="1 2 3 4 5 7" runs with a subset (e.g. without a robot with a hardware fault).
# LJ scaling (default ON): r0 = 0.5 m and, by alpha = 0.5/0.7, epsilon, cutoff radius and alignment radius too, so the
# force at every scaled distance equals the paper's and the gains stay valid.
set -u
REPS=$1; DUR=$2
HERE="$(cd "$(dirname "$0")/.." && pwd)"; RUNS="$HERE/hardware_runs"; mkdir -p "$RUNS"
CONDITIONS="${CONDITIONS:-lj plain clamped}"
export LJ_R0="${LJ_R0-0.5}" LJ_SCALE_EPS="${LJ_SCALE_EPS-1}"   # baseline scaled to 0.5 m by default (set LJ_R0= to disable)
export CORRIDOR_X="${CORRIDOR_X:--1.64 1.20}"
SCHED="$RUNS/schedule_${REPS}x_$(echo $CONDITIONS | tr ' ' '-').txt"
if [ ! -f "$SCHED" ]; then
  for c in $CONDITIONS; do for k in $(seq 1 $REPS); do echo "$c $k"; done; done > "$SCHED"
  echo "created schedule $SCHED"
fi
total=$(wc -l < "$SCHED"); n=0
while read -r COND K; do
  n=$((n+1))
  if ls -d "$RUNS/${COND}_rep${K}_"*/run_info.json >/dev/null 2>&1; then continue; fi    # already done
  case $COND in
    lj) CTRL=lj; GENOME=none ;;
    clamped) CTRL=hebbian; GENOME=plain_seed123_clamped_best.npy ;;
    plain) CTRL=hebbian; GENOME=plain_seed123_best.npy ;;
    *) echo "unknown condition $COND"; exit 1 ;;
  esac
  echo; echo "=== run $n/$total: condition '$COND', repetition $K ==="
  cat <<'MSG'
Place the robots by eye, deck up, lights off:
  - at the +x end of the arena, centred across the width, every robot facing -x (Thymio front toward the far end),
  - back row of 4 about a forearm apart (~40 cm gaps), front row of 3 staggered in between, ~40 cm ahead (toward -x),
  - back row ~20 cm in from the +x edge, everyone clear of the sides. Some randomness is fine.
Step out of the arena, then press Enter to start (Ctrl-C to stop; rerun the command to resume).
MSG
  read -r _ </dev/tty
  while true; do
    RUN_TAG="${COND}_rep${K}" "$HERE/tools/run_swarm.sh" "${ROBOTS:-1 2 3 4 5 6 7}" "$DUR" "$CTRL" "$GENOME"; rc=$?
    [ $rc -eq 0 ] && break
    if [ $rc -eq 2 ]; then
      echo; echo "Run aborted before anybody moved (a robot was not ready). Fix it (reseat/power-cycle/put it in view of the stations),"
      echo "then press Enter to retry the SAME repetition (Ctrl-C to stop; rerun the command later to resume)."; read -r _ </dev/tty
    else echo "run failed (exit $rc) -- fix and rerun the command to resume"; exit 1; fi
  done
done < "$SCHED"
echo; echo "schedule complete ($total runs)."
