import sys, time, cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
cflib.crtp.init_drivers()
uri=sys.argv[1]
with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='/tmp/cfcache_led')) as scf:
    p=scf.cf.param
    print("holding LEDs solid on", uri, flush=True)
    try:
        while True:
            p.set_value('led.bitmask','255'); time.sleep(0.3)
    finally:
        p.set_value('led.bitmask','0')
