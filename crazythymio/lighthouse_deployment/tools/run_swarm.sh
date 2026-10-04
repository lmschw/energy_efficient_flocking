#!/bin/bash
# usage: tools/run_swarm.sh "<robot numbers>" <duration_s> [hebbian|lj] [genome]
#   e.g. tools/run_swarm.sh "3 4" 60 lj          (LJ rule-based baseline, no genome)
#        tools/run_swarm.sh "3 4" 60 hebbian      (evolved genome, default plain_seed123_clamped_best.npy)
# Starts run_hebbian.py on those robots' Pis with a COMMON absolute start time (15 s from now, Pi clocks
# re-synced from this laptop first), stops after <duration_s>, then copies every log back to
# hardware_runs/<timestamp>/. Robots must already stand in the arena, deck up, lights off.
# Emergency stop: tools/stop_all.sh (or Ctrl-C here, which calls it).
set -u
ROBOTS=$1; DUR=$2; CTRL=${3:-hebbian}; GENOME=${4:-plain_seed123_clamped_best.npy}
if [ "$CTRL" = "lj" ]; then CTRLARGS="--controller lj"; else CTRLARGS="--controller hebbian --genome $GENOME"; fi
declare -A IPS=( [1]=10.15.2.83 [2]=10.15.2.81 [3]=10.15.3.13 [4]=10.15.2.197 [5]=10.15.2.25 [6]=10.15.2.70 [7]=10.15.2.250 )
HERE="$(cd "$(dirname "$0")/.." && pwd)"; KEY=$HOME/.ssh/id_ed25519_pis
SSHO="-i $KEY -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=8"
declare -A RID=( [1]=1 [2]=233 [3]=235 [4]=232 [5]=236 [6]=226 [7]=234 )
# ONLY the participating robots are neighbours: idle robots keep broadcasting their position over the radio, and
# listing them here would make the swarm react to robots that are not part of the test.
HOSTS=""; IDS=""
for n in $ROBOTS; do HOSTS="${HOSTS:+$HOSTS,}robot-$n"; IDS="${IDS:+$IDS,}${RID[$n]}"; done
CORRIDOR="-2.04 1.50"          # arena y-limits (robot centre), measured 2026-10-04, 0.2 m inset from the walls
CORRIDOR_X="${CORRIDOR_X:-}"   # arena x-limits "MIN MAX" -- set via env, e.g. CORRIDOR_X="-1.9 1.6" tools/run_swarm.sh ...
XARGS=""; [ -n "$CORRIDOR_X" ] && XARGS="--corridor-x $CORRIDOR_X"
STAMP=$(date +%Y%m%d_%H%M%S); OUT="$HERE/hardware_runs/$STAMP"; mkdir -p "$OUT"
trap '"$HERE/tools/stop_all.sh"; exit 130' INT TERM
for n in $ROBOTS; do
  ip=${IPS[$n]}
  rsync -az -e "ssh $SSHO" --exclude __pycache__ --exclude cache --exclude sessions --exclude hardware_runs --exclude logs "$HERE/" tugay@$ip:~/Desktop/crazy_thymio/lighthouse_deployment/ || { echo "robot $n: copy failed, aborting"; exit 1; }
  ssh $SSHO tugay@$ip "sudo date -s @$(date +%s) >/dev/null; rm -rf ~/Desktop/crazy_thymio/lighthouse_deployment/logs/$STAMP"
done
START=$(( $(date +%s) + 15 ))
echo "run $STAMP: neighbours = $HOSTS (ids $IDS); robots [$ROBOTS], $DUR s, controller $CTRL, corridor y [$CORRIDOR] x [${CORRIDOR_X:-NONE}], start at $(date -d @$START +%T)"
if [ "$CTRL" = "lj" ] && [ -z "$CORRIDOR_X" ]; then echo "WARNING: LJ pulls toward -x with no x limit set -- robots can leave the arena (set CORRIDOR_X)"; fi
for n in $ROBOTS; do
  ip=${IPS[$n]}
  ssh $SSHO tugay@$ip "cd ~/Desktop/crazy_thymio/lighthouse_deployment && ../.venv/bin/python -u run_hebbian.py --self-hostname robot-$n --hostnames $HOSTS --ids $IDS $CTRLARGS --corridor-y $CORRIDOR $XARGS --start-at $START --duration $DUR --log-dir logs/$STAMP > /tmp/run_$STAMP.log 2>&1" &
done
wait
echo "run finished; collecting logs"
for n in $ROBOTS; do
  ip=${IPS[$n]}
  rsync -az -e "ssh $SSHO" tugay@$ip:~/Desktop/crazy_thymio/lighthouse_deployment/logs/$STAMP/ "$OUT/" 2>/dev/null
  scp $SSHO tugay@$ip:/tmp/run_$STAMP.log "$OUT/robot-$n.console.log" 2>/dev/null
done
echo "logs in $OUT"; ls -1 "$OUT"
