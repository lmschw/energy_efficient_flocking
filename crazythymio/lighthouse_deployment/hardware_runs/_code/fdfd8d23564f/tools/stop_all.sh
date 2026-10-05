#!/bin/bash
# Emergency stop: ends run_hebbian.py on every Pi (SIGTERM -> experiment.stop() -> motors 0).
KEY=$HOME/.ssh/id_ed25519_pis
for ip in 10.15.2.83 10.15.2.81 10.15.3.13 10.15.2.197 10.15.2.25 10.15.2.70 10.15.2.250; do
  ( ssh -i $KEY -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=5 tugay@$ip 'pkill -TERM -f "[r]un_hebbian.py"; sleep 1.5; pgrep -f "[r]un_hebbian.py" >/dev/null && pkill -KILL -f "[r]un_hebbian.py"; sleep 0.5; cd ~/Desktop/crazy_thymio/lighthouse_deployment && timeout 15 ../.venv/bin/python tools/thymio_stop.py >/dev/null 2>&1; echo "$(hostname -I): stopped, motors 0"' 2>&1 | head -1 ) &
done
wait
