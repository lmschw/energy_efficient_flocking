#!/usr/bin/env python3
"""DO NOT USE BEFORE A RUN: any radio session to a Crazyflie -- even one closed cleanly (tested 2026-10-04 on robot 4) --
leaves its USB link to the Pi unresponsive until the Crazyflie is power-cycled. Failed robots are now marked with a RED
Thymio top LED instead (tools/diagnose_robot.py --mark).
usage: identify_robot_leds.py <radio uri> [max_seconds=900]
Holds ALL LEDs of that Crazyflie on, steady, so a person can pick out the robot. Stops on SIGTERM/SIGINT, when the board drops
off the radio (power-cycle), or after max_seconds -- and then ALWAYS switches the LEDs off and closes the radio link cleanly
(a hard kill of a radio session can leave the board ignoring its USB link until it is power-cycled)."""
import signal, sys, time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
stop = False
def _stop(*_):
    global stop; stop = True
signal.signal(signal.SIGTERM, _stop); signal.signal(signal.SIGINT, _stop)
uri = sys.argv[1]; limit = time.time() + (float(sys.argv[2]) if len(sys.argv) > 2 else 900.0)
cflib.crtp.init_drivers()
try:
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="/tmp/cfcache_idled")) as scf:
        p = scf.cf.param
        print("holding LEDs on", uri, flush=True)
        try:
            while not stop and time.time() < limit:
                p.set_value("led.bitmask", "255"); time.sleep(0.3)
        except Exception as e:          # board vanished (power-cycled) or param error
            print("stopped:", type(e).__name__, flush=True)
        finally:
            try: p.set_value("led.bitmask", "0")
            except Exception: pass
        print("LEDs off, closing link", flush=True)
except Exception as e:
    print("could not hold LEDs:", type(e).__name__, e, flush=True)
