#!/bin/bash
# usage: tools/experiment.sh <lj|hebbian> <repetitions> <duration_s> [genome]   (all 7 robots)
#   e.g. tools/experiment.sh lj 5 90
#        tools/experiment.sh hebbian 5 90 plain_seed123_clamped_best.npy
# For every repetition: you place the robots by eye (instructions are printed) -> runs all 7 with a common start time ->
# collects logs into hardware_runs/<condition>_rep<k>_<stamp>/ (+ run_info.json + config snapshot).
# Ctrl-C = emergency stop.
set -u
CTRL=$1; REPS=$2; DUR=$3; GENOME=${4:-plain_seed123_clamped_best.npy}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CORRIDOR_X="${CORRIDOR_X:--1.64 1.20}"; export CORRIDOR_X
if [ "$CTRL" = "lj" ]; then COND=lj; else COND=$(basename "$GENOME" _best.npy); fi
for rep in $(seq 1 $REPS); do
  echo; echo "=== $COND repetition $rep/$REPS ==="
  cat <<'MSG'
Place the 7 robots by eye, deck up, lights off:
  - at the +x end of the arena (where robots 3 and 4 started in the pair runs), centred across the width,
  - every robot facing -x, i.e. the Thymio front pointing toward the far end,
  - back row of 4 about a forearm apart (~40 cm gaps), front row of 3 staggered in between, ~40 cm ahead (toward -x),
  - the back row about 20 cm in from the +x edge, everyone clear of the sides.
Some randomness is fine. Step out of the arena, then press Enter to start the run.
MSG
  read -r _
  RUN_TAG="${COND}_rep${rep}" "$HERE/tools/run_swarm.sh" "1 2 3 4 5 6 7" "$DUR" "$CTRL" "$GENOME" || { echo "run failed"; exit 1; }
done
echo; echo "all repetitions done:"; ls -1d "$HERE"/hardware_runs/${COND}_rep*
