#!/usr/bin/env python3
"""usage: diagnose_robot.py <robot 1..7> [run stamp] [--mark]
Works out why a robot could not get ready and prints a plain-language cause + fix. With --mark, also turns that robot's
Thymio top LED RED (done by its own Pi, no radio involved) so you can find it; run_hebbian.py clears it on the next start."""
import os, re, subprocess, sys
IPS = {1: "10.15.2.83", 2: "10.15.2.81", 3: "10.15.3.13", 4: "10.15.2.197", 5: "10.15.2.25", 6: "10.15.2.70", 7: "10.15.2.250"}
n = int(sys.argv[1]); stamp = next((a for a in sys.argv[2:] if not a.startswith("--")), None); mark = "--mark" in sys.argv
key = os.path.expanduser("~/.ssh/id_ed25519_pis")
def ssh(cmd, t=25):
    r = subprocess.run(["ssh", "-i", key, "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=6",
                        f"tugay@{IPS[n]}", cmd], capture_output=True, text=True, timeout=t)
    return r.returncode, r.stdout
try:
    rc, out = ssh(f"""echo '##LSUSB'; lsusb; echo '##DMESG'; sudo dmesg -T | grep -iE 'usb [0-9]-[0-9]' | tail -6;
echo '##TDM'; pgrep -f '[t]hymio-device-manager' >/dev/null && echo running || echo stopped;
echo '##CONSOLE'; {"tail -8 /tmp/hebb_serve/serve.log" if stamp in (None, "serve") else "tail -8 /tmp/run_" + stamp + ".log"}""")
except subprocess.TimeoutExpired:
    rc, out = 255, ""
if rc == 255 or not out:
    print(f"robot-{n}: Pi {IPS[n]} NOT REACHABLE over the network -> is the Pi on / on the Wi-Fi? (cannot mark it, no Pi to talk to)")
    sys.exit(0)
sec = dict(re.findall(r"##(\w+)\n(.*?)(?=\n##|\Z)", out, re.S))
lsusb, dmesg, tdm, con = sec.get("LSUSB", ""), sec.get("DMESG", ""), sec.get("TDM", "").strip(), sec.get("CONSOLE", "")
cf_on_bus, thymio_on_bus = "0483:5740" in lsusb, "0617:000a" in lsusb
usb_errors = re.findall(r"(error -\d+|not accepting address|descriptor read)", dmesg)
last = [l for l in con.strip().splitlines() if l.strip() and "no LogEntry" not in l]
last = last[-1][:150] if last else "(no console output)"
if not thymio_on_bus:
    cause, fix = "Thymio not connected to the Pi", "switch the Thymio on / reseat its USB cable to the Pi (the red LED needs the Thymio, so it cannot be marked)"
elif not cf_on_bus:
    cause = "Crazyflie not detected on the Pi's USB" + (f" (kernel USB errors: {', '.join(sorted(set(usb_errors)))})" if usb_errors else "")
    fix = "reseat/replace the Crazyflie's USB cable, then power-cycle the Crazyflie"
elif re.search(r"Could not open usb|Crazyflie disconnected|Too many packets lost|link error", con):
    cause, fix = "Crazyflie is on USB but does not answer (stuck link, e.g. after a radio session)", "power-cycle the Crazyflie"
elif re.search(r"no steady Lighthouse pose within|NOT READY at the start time", con):
    cause, fix = "no stable Lighthouse position", "put the robot in the arena with a clear view of the base stations (deck uncovered, lights off), don't move it"
elif re.search(r"No Thymio found|ConnectionRefused|Connection refused", con) or tdm == "stopped":
    cause, fix = "Thymio Device Manager not running / Thymio not reachable through it", "it is restarted automatically on the next attempt; if it repeats, power-cycle the Thymio"
else:
    cause, fix = "unrecognised failure", "see the last console line"
print(f"robot-{n}: {cause}\n          -> FIX: {fix}\n          last console line: {last}")
if mark and thymio_on_bus:
    subprocess.run(["ssh", "-i", key, "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", f"tugay@{IPS[n]}",
                    "pkill -f '[r]un_hebbian.py' ; sleep 0.5; cd ~/Desktop/crazy_thymio/lighthouse_deployment && ../.venv/bin/python tools/thymio_wiggle.py 4"],
                   capture_output=True, text=True, timeout=40)
    print(f"          robot-{n} just WIGGLED left-right 4 times -- that is the one (again: tools/identify.sh {n})")
