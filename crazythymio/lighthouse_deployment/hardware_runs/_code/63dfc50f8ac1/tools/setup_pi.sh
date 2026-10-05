#!/bin/bash
# usage: setup_pi.sh <ip> <robot-name e.g. robot-1> [user]
IP=$1; ROBOT=$2; USER=${3:-tugay}
HOSTS=robot-1,robot-2,robot-3,robot-4,robot-5,robot-6,robot-7; IDS=1,233,235,232,236,226,234
KEY=$HOME/.ssh/id_ed25519_pis
SSH="ssh -i $KEY -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=8 $USER@$IP"
echo "== connect"; $SSH 'echo "as $(whoami) on $(hostname) $(hostname -I) | $(uname -m) | sudo: $(sudo -n true 2>&1 && echo yes)"; lsusb | grep -i "0483\|0617"; ls ~/Desktop/crazy_thymio/.venv/bin/python; ~/Desktop/crazy_thymio/.venv/bin/python -c "import cflib,tdmclient,numpy;print(\"venv ok: cflib tdmclient numpy\", numpy.__version__)" 2>&1 | tail -1; vcgencmd get_throttled' || { echo "CANNOT CONNECT"; exit 1; }
echo "== copy code + clock"
rsync -az --delete -e "ssh -i $KEY -o BatchMode=yes -o IdentitiesOnly=yes" --exclude __pycache__ --exclude cache --exclude sessions --exclude '*.pyc' /home/lilly/dev/energy_efficient_flocking/energy_efficient_flocking/crazythymio/lighthouse_deployment/ $USER@$IP:~/Desktop/crazy_thymio/lighthouse_deployment/ && echo "code copied"
$SSH "sudo date -s @$(date +%s) >/dev/null; date"
echo "== thymio device manager"
$SSH 'if ! (cd ~/Desktop/crazy_thymio && timeout 10 .venv/bin/python -c "
import socket;s=socket.socket();s.settimeout(3);s.connect((\"127.0.0.1\",8596));print(\"tdm port open\")" 2>/dev/null); then echo "tdm not running -> starting"; setsid nohup flatpak run --command=thymio-device-manager org.mobsya.ThymioSuite > /tmp/tdm.log 2>&1 < /dev/null & sleep 8; fi'
echo "== offline test"; $SSH 'cd ~/Desktop/crazy_thymio/lighthouse_deployment && ../.venv/bin/python test_offline.py 2>&1 | tail -1'
echo "== real check ($ROBOT)"
$SSH "cd ~/Desktop/crazy_thymio/lighthouse_deployment && before=\$(sudo dmesg | grep -c 'USB disconnect'); timeout 60 ../.venv/bin/python -u run_hebbian.py --self-hostname $ROBOT --hostnames $HOSTS --ids $IDS --check > /tmp/check.log 2>&1; echo \"run exit \$?\"; after=\$(sudo dmesg | grep -c 'USB disconnect'); echo \"USB disconnects during run: \$((after-before))\"; grep -v 'no LogEntry' /tmp/check.log | sed -n '2p;12p;24p' | cut -c1-330"
