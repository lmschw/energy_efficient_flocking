"""Runs the Hebbian ABCD controller on ONE Thymio + Raspberry Pi + Crazyflie-board (Lighthouse)
robot. Start the same command (only --self-hostname differs) on every Pi.

  python hebbian/run_hebbian.py --self-hostname thymio-01 \
      --hostnames thymio-01,thymio-02,thymio-03 --ids 1,2,3 \
      --genome ../ants26_replication/hardware_deployment/plain_seed123_clamped_best.npy \
      --origin 2.0 1.5 --corridor-y -1.5 1.5

Modes besides the experiment itself:
  --check              print own pose / neighbors / IR for 20 s, motors off
  --calibrate-heading  drive straight for 4 s, print the board's HEADING offset and a
                       suggested MOTOR_UNITS_PER_MPS (robot needs ~0.5 m free path ahead)
See crazythymio/hebbian/README.md for the full procedure.
"""
import argparse
import asyncio
import csv
import math
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HW = os.path.join(HERE, "..", "..", "ants26_replication", "hardware_deployment")
sys.path.insert(0, HERE)
sys.path.insert(0, HW)

import numpy as np  # noqa: E402

import controller_config as cfg  # noqa: E402
from lighthouse_robot import LighthouseRobot  # noqa: E402


class CsvLogger:
    """Same log(state, command) interface as thymio_swarm_platform's SessionLogger."""

    def __init__(self, path):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.header = None

    def log(self, state, command):
        row = {**state, **command}
        if self.header is None:
            self.header = list(row)
            self.writer.writerow(self.header)
        self.writer.writerow([row[k] for k in self.header])
        self.file.flush()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--self-hostname", required=True)
    p.add_argument("--hostnames", required=True, help="comma list, SAME order on every Pi")
    p.add_argument("--ids", required=True,
                   help="comma list of radio ids (1..10), parallel to --hostnames")
    p.add_argument("--genome", help="hebbian_*_best.npy (needed unless --check/--calibrate-heading)")
    p.add_argument("--uri", default="usb://0")
    p.add_argument("--origin", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"),
                   help="Lighthouse coordinates [m] of the arena center = simulation (0, 0)")
    p.add_argument("--corridor-y", type=float, nargs=2, metavar=("MIN", "MAX"),
                   help="arena y-limits in the shifted frame; enables the wall slow-down governor")
    p.add_argument("--heading-offsets", default="",
                   help="host=rad,host=rad  (from --calibrate-heading)")
    p.add_argument("--battery-mode", choices=["none", "simulated"], help="override controller_config")
    p.add_argument("--motor-units-per-mps", type=float, help="override controller_config")
    p.add_argument("--start-at", type=float, help="unix time at which to start driving "
                   "(Pis must be time-synced); default: wait for Enter")
    p.add_argument("--duration", type=float, help="stop after this many seconds")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--check", action="store_true")
    p.add_argument("--calibrate-heading", action="store_true")
    return p.parse_args()


def apply_config(args):
    cfg.POSE_SOURCE = "lighthouse"
    cfg.POSITION_AXES = [0, 1]
    if args.heading_offsets:
        for item in args.heading_offsets.split(","):
            host, val = item.split("=")
            cfg.LIGHTHOUSE_HEADING_OFFSET_RAD[host.strip()] = float(val)
    if args.corridor_y:
        cfg.CORRIDOR_Y_MIN, cfg.CORRIDOR_Y_MAX = args.corridor_y
    else:
        cfg.CORRIDOR_Y_MIN = cfg.CORRIDOR_Y_MAX = None
    if args.battery_mode:
        cfg.BATTERY_MODE = args.battery_mode
    if args.motor_units_per_mps:
        cfg.MOTOR_UNITS_PER_MPS = args.motor_units_per_mps


async def check(robot, seconds=20):
    t_end = time.time() + seconds
    while time.time() < t_end:
        own = robot.own_pose_record()
        ir = await robot.proximity_horizontal()
        print(f"own (x,y,z,yaw)={None if own is None else tuple(round(v, 3) for v in own)}  "
              f"neighbors={ {k: tuple(round(c, 3) for c in v) for k, v in robot.board.neighbors.items()} }  "
              f"IR={ir}")
        await asyncio.sleep(0.5)


async def calibrate(robot, args):
    units = 150
    print(f"Driving straight at motor target {units} for 4 s -- keep the path clear.")
    p0 = robot.own_pose_record()
    if p0 is None:
        raise SystemExit("No Lighthouse pose -- fix tracking first (--check).")
    await robot.drive(units, units)
    await asyncio.sleep(4.0)
    await robot.stop()
    await asyncio.sleep(0.5)
    p1 = robot.own_pose_record()
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    dist = math.hypot(dx, dy)
    travel = math.atan2(dy, dx)
    yaw_mid = (p0[3] + p1[3]) / 2.0
    offset = (travel - yaw_mid + math.pi) % (2 * math.pi) - math.pi
    print(f"travelled {dist:.3f} m, travel bearing {travel:.3f} rad, board yaw {yaw_mid:.3f} rad")
    print(f"  --heading-offsets {args.self_hostname}={offset:.4f}")
    print(f"  suggested MOTOR_UNITS_PER_MPS = {units / (dist / 4.0):.1f} "
          f"(current {cfg.MOTOR_UNITS_PER_MPS}) -- pass as --motor-units-per-mps")


async def main():
    args = parse_args()
    apply_config(args)
    hostnames = args.hostnames.split(",")
    ids = [int(i) for i in args.ids.split(",")]
    assert len(hostnames) == len(ids) and args.self_hostname in hostnames
    assert len(set(ids)) == len(ids) and all(1 <= i <= 10 for i in ids), "ids must be unique, 1..10"

    robot = LighthouseRobot(args.self_hostname, dict(zip(hostnames, ids)), args.uri, tuple(args.origin))
    await robot.connect()
    try:
        if args.check:
            await check(robot)
            return
        if args.calibrate_heading:
            await calibrate(robot, args)
            return
        if not args.genome:
            raise SystemExit("--genome is required for an experiment run")

        from hebbian_swarm_experiment import HebbianSwarmExperiment
        os.makedirs(args.log_dir, exist_ok=True)
        log_path = os.path.join(args.log_dir, f"{args.self_hostname}_{int(time.time())}.csv")
        experiment = HebbianSwarmExperiment(
            robot=robot, logger=CsvLogger(log_path),
            config={"genome_path": args.genome, "hostnames": hostnames,
                    "self_hostname": args.self_hostname})
        # Sensing/speed estimates need a warm-up: wait for own pose before starting.
        while robot.own_pose_record() is None:
            print("waiting for Lighthouse pose ...")
            await asyncio.sleep(1.0)
        if args.start_at:
            await asyncio.sleep(max(0.0, args.start_at - time.time()))
        else:
            await asyncio.get_running_loop().run_in_executor(None, input, "Press Enter to start ... ")

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(experiment.stop()))
        if args.duration:
            loop.call_later(args.duration, lambda: asyncio.ensure_future(experiment.stop()))
        print(f"running, logging to {log_path}  (Ctrl-C to stop)")
        await experiment.run()
    finally:
        await robot.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
