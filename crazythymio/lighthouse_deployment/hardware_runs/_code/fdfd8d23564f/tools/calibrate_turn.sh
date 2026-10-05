#!/bin/bash
# usage: tools/calibrate_turn.sh <robot 1..7>   -- spins that robot IN PLACE (about 1.6-2.4 rad each, 3 spins) and reports how
# much it really turns compared with the command. Keep 20 cm free around it. Appends to lighthouse_config/turn_calibration.csv.
set -e
N=$1
case $N in
  1) IP=10.15.2.83 ;;  2) IP=10.15.2.81 ;;  3) IP=10.15.3.13 ;;  4) IP=10.15.2.197 ;;
  5) IP=10.15.2.25 ;;  6) IP=10.15.2.70 ;;  7) IP=10.15.2.250 ;;
  *) echo "usage: $0 <1..7>"; exit 1 ;;
esac
HERE="$(cd "$(dirname "$0")/.." && pwd)"; KEY=$HOME/.ssh/id_ed25519_pis
HOSTS=robot-1,robot-2,robot-3,robot-4,robot-5,robot-6,robot-7; IDS=1,233,235,232,236,226,234
rsync -az -e "ssh -i $KEY -o BatchMode=yes -o IdentitiesOnly=yes" --exclude __pycache__ --exclude cache --exclude sessions --exclude hardware_runs --exclude logs "$HERE/" tugay@$IP:~/Desktop/crazy_thymio/lighthouse_deployment/
OUT=$(ssh -i $KEY -o BatchMode=yes -o IdentitiesOnly=yes tugay@$IP "cd ~/Desktop/crazy_thymio/lighthouse_deployment && ../.venv/bin/python -u run_hebbian.py --self-hostname robot-$N --hostnames $HOSTS --ids $IDS --calibrate-turn 2>&1 | grep -v 'no LogEntry'")
echo "$OUT"
LINE=$(echo "$OUT" | grep '^RESULT ' | tail -1)
if [ -n "$LINE" ]; then
  F="$HERE/lighthouse_config/turn_calibration.csv"; [ -f "$F" ] || echo "timestamp,robot,ratio_50,ratio_m50,ratio_90,mean" > "$F"
  echo "$(date -Is),robot-$N,$(echo $LINE | sed -E 's/.*turn_ratio_50=([-0-9.]+).*/\1/'),$(echo $LINE | sed -E 's/.*turn_ratio_m50=([-0-9.]+).*/\1/'),$(echo $LINE | sed -E 's/.*turn_ratio_90=([-0-9.]+).*/\1/'),$(echo $LINE | sed -E 's/.*mean=([-0-9.]+).*/\1/')" >> "$F"
  echo "saved to $F"
fi
