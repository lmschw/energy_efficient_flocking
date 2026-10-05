#!/bin/bash
# usage: tools/measure_offset.sh <robot 1..7> [<robot> ...]
# Turns each robot IN PLACE in 45-degree steps through a full circle (~45 s per robot) and fits where its Crazyflie sits
# relative to the Thymio's centre of rotation. Keep 20 cm free around it, stations in view, lights off, don't touch it.
# Appends to lighthouse_config/board_offset.csv.
HERE="$(cd "$(dirname "$0")/.." && pwd)"; KEY=$HOME/.ssh/id_ed25519_pis
F="$HERE/lighthouse_config/board_offset.csv"; [ -f "$F" ] || echo "timestamp,robot,board_offset_x,board_offset_y,error_cm,residual_cm,coverage_deg" > "$F"
for N in "$@"; do
  case $N in 1) IP=10.15.2.83;; 2) IP=10.15.2.81;; 3) IP=10.15.3.13;; 4) IP=10.15.2.197;; 5) IP=10.15.2.25;; 6) IP=10.15.2.70;; 7) IP=10.15.2.250;; *) echo "robot numbers 1..7"; exit 1;; esac
  echo "=== robot $N"
  rsync -az -e "ssh -i $KEY -o BatchMode=yes -o IdentitiesOnly=yes" --exclude __pycache__ --exclude cache --exclude sessions --exclude hardware_runs --exclude logs "$HERE/" tugay@$IP:~/Desktop/crazy_thymio/lighthouse_deployment/
  OUT=$(ssh -i $KEY -o BatchMode=yes -o IdentitiesOnly=yes tugay@$IP "cd ~/Desktop/crazy_thymio/lighthouse_deployment && ../.venv/bin/python -u run_hebbian.py --self-hostname robot-$N --hostnames robot-1,robot-2,robot-3,robot-4,robot-5,robot-6,robot-7 --ids 1,233,235,232,236,226,234 --measure-offset 2>&1 | grep -E 'stop |centre of|TRUE|RESULT|No steady|Exception|another run|firmware|refus|not|Error'")
  echo "$OUT"
  LINE=$(echo "$OUT" | grep '^RESULT ' | tail -1)
  [ -n "$LINE" ] && echo "$(date -Is),robot-$N,$(echo $LINE | sed -E 's/.*board_offset_x=([-0-9.]+) board_offset_y=([-0-9.]+) error_cm=([0-9.]+) residual_cm=([0-9.]+) coverage_deg=([0-9.]+).*/\1,\2,\3,\4,\5/')" >> "$F" && echo "saved to $F"
done
