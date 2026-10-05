#!/bin/bash
# usage: tools/latency_test.sh <robot 1..7> [log period ms]  -- spins that robot IN PLACE 4 times (~50 deg each, alternating direction) and measures the
# total delay from motor command to the first visible heading change. Keep 20 cm free around it.
N=$1; LP=${2:-}
case $N in 1) IP=10.15.2.83;; 2) IP=10.15.2.81;; 3) IP=10.15.3.13;; 4) IP=10.15.2.197;; 5) IP=10.15.2.25;; 6) IP=10.15.2.70;; 7) IP=10.15.2.250;; *) echo "usage: $0 <1..7> [ms]"; exit 1;; esac
HERE="$(cd "$(dirname "$0")/.." && pwd)"; KEY=$HOME/.ssh/id_ed25519_pis
rsync -az -e "ssh -i $KEY -o BatchMode=yes -o IdentitiesOnly=yes" --exclude __pycache__ --exclude cache --exclude sessions --exclude hardware_runs --exclude logs "$HERE/" tugay@$IP:~/Desktop/crazy_thymio/lighthouse_deployment/
ssh -i $KEY -o BatchMode=yes -o IdentitiesOnly=yes tugay@$IP "cd ~/Desktop/crazy_thymio/lighthouse_deployment && ../.venv/bin/python -u run_hebbian.py --self-hostname robot-$N --hostnames robot-1,robot-2,robot-3,robot-4,robot-5,robot-6,robot-7 --ids 1,233,235,232,236,226,234 ${LP:+--log-period-ms $LP} --latency-test 2>&1 | grep -E 'spin|RESULT|No steady|Exception'"
