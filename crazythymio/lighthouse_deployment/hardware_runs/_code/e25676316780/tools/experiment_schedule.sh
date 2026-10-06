#!/bin/bash
# usage: tools/experiment_schedule.sh <repetitions per condition> <duration_s>
#   e.g. tools/experiment_schedule.sh 30 60
# EXTRA REPETITIONS, ALTERNATING (added 2026-10-06, after the main 30x lj/plain/clamped block):
#   CONDITIONS="lj plain" INTERLEAVE=1 FIRST_REP=31 LAUNCHER=run_swarm_fast.sh tools/experiment_schedule.sh 40 60
#   -> lj 31, plain 31, lj 32, plain 32, ... lj 40, plain 40: ordinary lj_rep<k>_* / plain_rep<k>_* runs that simply
#   continue the numbering. Stopping at any point leaves (nearly) balanced pairs.
#   FIRST_REP (default 1): first repetition number; <repetitions> is then the LAST repetition number.
# Fixed, deterministic, RESUMABLE schedule: all repetitions of each condition in turn, in this order:
#   1. lj       -- the baseline (paper rules, scaled to LJ_R0 = 0.5 m, see below)
#   2. plain    -- evolved genome plain_seed123_best.npy (trained without the safety clamp)
#   3. clamped  -- evolved genome plain_seed123_clamped_best.npy (kept for last)
# Each entry: the robots start reconnecting at once while you place them -> Enter -> run all 7 -> logs into
# hardware_runs/<condition>_rep<k>_<stamp>/ -> the next entry immediately starts reconnecting again.
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
FIRST_REP="${FIRST_REP:-1}"
if [ "$FIRST_REP" = 1 ]; then RANGE="${REPS}x"; else RANGE="reps${FIRST_REP}-${REPS}"; fi
SCHED="$RUNS/schedule_${RANGE}_$(echo $CONDITIONS | tr ' ' '-')${INTERLEAVE:+_interleaved}.txt"
if [ ! -f "$SCHED" ]; then
  if [ -n "${INTERLEAVE:-}" ]; then
    for k in $(seq $FIRST_REP $REPS); do for c in $CONDITIONS; do echo "$c $k"; done; done > "$SCHED"
  else
    for c in $CONDITIONS; do for k in $(seq $FIRST_REP $REPS); do echo "$c $k"; done; done > "$SCHED"
  fi
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
  while true; do
    WAIT_FOR_ENTER=1 RUN_TAG="${COND}_rep${K}" "$HERE/tools/${LAUNCHER:-run_swarm.sh}" "${ROBOTS:-1 2 3 4 5 6 7}" "$DUR" "$CTRL" "$GENOME" </dev/null; rc=$?   # </dev/null: ssh must not eat the schedule lines
    [ $rc -eq 0 ] && break
    if [ $rc -eq 3 ]; then
      echo; echo "That run was INVALID (see the reason above: a robot lost its position or did not log the whole run) -- it was moved"
      echo "aside and the SAME repetition is repeated now. Check that robot (power-cycle its Crazyflie if it lost its position)."
      continue
    fi
    if [ $rc -eq 2 ]; then
      echo; echo "Run aborted before anybody moved (a robot was not ready -- see the reason above). Fix that robot; the SAME"
      echo "repetition is retried right away (robots reconnect while you fix/place them; Enter starts it). Ctrl-C to stop."
    else echo "run failed (exit $rc) -- fix and rerun the command to resume"; exit 1; fi
  done
done < "$SCHED"
echo; echo "schedule complete ($total runs) -- stopping the robots' controller processes."; "$HERE/tools/stop_all.sh"
