#!/bin/bash
# usage: tools/experiment_schedule.sh <repetitions per condition> <duration_s>
#   e.g. tools/experiment_schedule.sh 30 60
# EXTRA ALTERNATING BLOCK (added 2026-10-06, after the main 30x lj/plain/clamped data):
#   CONDITIONS="ljalt plainalt" INTERLEAVE=1 LAUNCHER=run_swarm_fast.sh tools/experiment_schedule.sh 10 60
#   -> ljalt 1, plainalt 1, ljalt 2, plainalt 2, ... (same controllers/settings as lj / plain); folders
#   ljalt_rep<k>_* / plainalt_rep<k>_* so they are never confused with the main block but analyse as their own conditions.
#   Stopping at any point leaves (nearly) balanced pairs.
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
SCHED="$RUNS/schedule_${REPS}x_$(echo $CONDITIONS | tr ' ' '-')${INTERLEAVE:+_interleaved}.txt"
if [ ! -f "$SCHED" ]; then
  if [ -n "${INTERLEAVE:-}" ]; then
    for k in $(seq 1 $REPS); do for c in $CONDITIONS; do echo "$c $k"; done; done > "$SCHED"
  else
    for c in $CONDITIONS; do for k in $(seq 1 $REPS); do echo "$c $k"; done; done > "$SCHED"
  fi
  echo "created schedule $SCHED"
fi
total=$(wc -l < "$SCHED"); n=0
while read -r COND K; do
  n=$((n+1))
  if ls -d "$RUNS/${COND}_rep${K}_"*/run_info.json >/dev/null 2>&1; then continue; fi    # already done
  case $COND in
    lj|ljalt) CTRL=lj; GENOME=none ;;
    clamped) CTRL=hebbian; GENOME=plain_seed123_clamped_best.npy ;;
    plain|plainalt) CTRL=hebbian; GENOME=plain_seed123_best.npy ;;
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
