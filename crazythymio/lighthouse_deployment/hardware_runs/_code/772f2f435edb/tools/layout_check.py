#!/usr/bin/env python3
"""Reads all 7 positions through one robot's Pi (neighbour table) and compares them with tools/layout_7.txt.
usage: tools/layout_check.py [reference robot number, default 4] [tolerance m, default 0.25]
Exit code 0 = every robot within tolerance. Prints per-robot deviation."""
import math, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ref = int(sys.argv[1]) if len(sys.argv) > 1 else 4
tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
IPS = {1: "10.15.2.83", 2: "10.15.2.81", 3: "10.15.3.13", 4: "10.15.2.197", 5: "10.15.2.25", 6: "10.15.2.70", 7: "10.15.2.250"}
HOSTS = "robot-1,robot-2,robot-3,robot-4,robot-5,robot-6,robot-7"; IDS = "1,233,235,232,236,226,234"
key = os.path.expanduser("~/.ssh/id_ed25519_pis")
ssh = ["ssh", "-i", key, "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=8", f"tugay@{IPS[ref]}"]
subprocess.run(["rsync", "-az", "-e", f"ssh -i {key} -o BatchMode=yes -o IdentitiesOnly=yes", "--exclude", "__pycache__", "--exclude", "cache",
                "--exclude", "sessions", "--exclude", "hardware_runs", "--exclude", "logs", os.path.join(HERE, "..") + "/",
                f"tugay@{IPS[ref]}:~/Desktop/crazy_thymio/lighthouse_deployment/"], check=True)
out = subprocess.run(ssh + [f"cd ~/Desktop/crazy_thymio/lighthouse_deployment && ../.venv/bin/python -u run_hebbian.py --self-hostname robot-{ref} --hostnames {HOSTS} --ids {IDS} --layout 2>&1 | grep -E 'LAYOUT|steady|No steady'"],
                     capture_output=True, text=True).stdout
pos = {}
for line in out.splitlines():
    if line.startswith("LAYOUT"):
        _, name, *rest = line.split()
        pos[name] = None if rest[0] == "MISSING" else (float(rest[0]), float(rest[1]))
target = {}
for line in open(os.path.join(HERE, "layout_7.txt")):
    if line.strip() and not line.startswith("#"):
        n, x, y = line.split(); target[n] = (float(x), float(y))
worst = 0.0; bad = []
print(f"{'robot':8} {'target x,y':>14} {'measured x,y':>16} {'off [m]':>8}")
for n, (tx, ty) in target.items():
    m = pos.get(n)
    if m is None:
        print(f"{n:8} {tx:+7.2f},{ty:+6.2f}   NOT SEEN (off, no stations, or radio)"); bad.append(n); continue
    d = math.hypot(m[0] - tx, m[1] - ty); worst = max(worst, d)
    flag = "" if d <= tol else "  <-- move"
    if d > tol: bad.append(n)
    print(f"{n:8} {tx:+7.2f},{ty:+6.2f}   {m[0]:+7.2f},{m[1]:+6.2f} {d:8.2f}{flag}")
print(f"worst deviation {worst:.2f} m (tolerance {tol:.2f} m)")
sys.exit(0 if not bad else 1)
