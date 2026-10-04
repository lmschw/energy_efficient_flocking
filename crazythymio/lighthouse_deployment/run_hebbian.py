"""Runs the Hebbian ABCD controller on ONE Thymio + Raspberry Pi + Crazyflie-board (Lighthouse)
robot. Start the same command (only --self-hostname differs) on every Pi.

  python run_hebbian.py --self-hostname thymio-01 \
      --hostnames thymio-01,thymio-02,thymio-03 --ids 1,2,3 \
      --genome plain_seed123_clamped_best.npy \
      --origin 2.0 1.5 --corridor-y -1.5 1.5

Modes besides the experiment itself:
  --check              print own pose / neighbors / IR for 20 s, motors off
  --calibrate-heading  drive straight for 4 s, print the board's HEADING offset and a
                       suggested MOTOR_UNITS_PER_MPS (robot needs ~0.5 m free path ahead)
See README.md in this folder for the full procedure.
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
sys.path.insert(0, HERE)   # LIGHTHOUSE deployment: imports only this folder's own modules

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
                   help="comma list of radio ids (low byte of each board's radio address, decimal "
                        "1..255, e.g. address ...E9 -> 233), parallel to --hostnames")
    p.add_argument("--genome", help="hebbian_*_best.npy (needed for --controller hebbian)")
    p.add_argument("--controller", choices=["hebbian", "lj"], default="hebbian",
                   help="hebbian = evolved genome, lj = the paper's LJ rule-based baseline (no genome)")
    p.add_argument("--uri", default="usb://0")
    p.add_argument("--origin", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"),
                   help="Lighthouse coordinates [m] of the arena center = simulation (0, 0)")
    p.add_argument("--corridor-y", type=float, nargs=2, metavar=("MIN", "MAX"),
                   help="arena y-limits in the shifted frame; enables the wall slow-down governor")
    p.add_argument("--calibrate-turn", action="store_true",
                   help="spin in place at a few motor levels and report realised vs nominal turn rate (robot stays put)")
    p.add_argument("--lj-r0", type=float, help="override the LJ preferred spacing r0 [m] (paper: 0.7); LJ controller only")
    p.add_argument("--lj-scale-epsilon", action="store_true",
                   help="with --lj-r0: also scale everything else that has units of length or force in the LJ law by "
                        "alpha = r0/0.7 (epsilon, cutoff radius, alignment radius), so the force at every scaled distance "
                        "equals the paper's and the gains K1, K2, k_goal stay valid")
    p.add_argument("--safety", choices=["on", "off"], help="deployment-side safety layers (agent clamp, wall governors, untracked "
                   "crawl); default: controller_config.SAFETY_LAYERS_ENABLED (currently off)")
    p.add_argument("--corridor-x", type=float, nargs=2, metavar=("MIN", "MAX"),
                   help="arena x-limits in the shifted frame; enables the x slow-down governor")
    p.add_argument("--heading-offsets", default="",
                   help="host=rad,host=rad  (from --calibrate-heading)")
    p.add_argument("--battery-mode", choices=["none", "simulated"], help="override controller_config")
    p.add_argument("--motor-units-per-mps", type=float, help="override controller_config")
    p.add_argument("--start-at", type=float, help="unix time at which to start driving "
                   "(Pis must be time-synced); default: wait for Enter")
    p.add_argument("--ready-file", help="touch this file once connected and steady (handshake with tools/run_swarm.sh)")
    p.add_argument("--go-file", help="then wait for this file; it holds \"<start_epoch> <stop_epoch>\" in THIS Pi's clock")
    p.add_argument("--stop-at", type=float, help="unix time (THIS Pi's clock) at which to stop; overrides --duration, so all robots\n                   "
                   "stop together even if one started late")
    p.add_argument("--duration", type=float, help="stop after this many seconds")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--check", action="store_true")
    p.add_argument("--calibrate-heading", action="store_true")
    p.add_argument("--layout", action="store_true",
                   help="print this robot's own steady position and every neighbour's (from the radio table) and exit")
    p.add_argument("--where", action="store_true",
                   help="print this robot's steady position (in the --origin-shifted frame) and exit")
    return p.parse_args()


def apply_config(args):
    if args.heading_offsets:
        for item in args.heading_offsets.split(","):
            host, val = item.split("=")
            cfg.LIGHTHOUSE_HEADING_OFFSET_RAD[host.strip()] = float(val)
    if args.corridor_y:
        cfg.CORRIDOR_Y_MIN, cfg.CORRIDOR_Y_MAX = args.corridor_y
    else:
        cfg.CORRIDOR_Y_MIN = cfg.CORRIDOR_Y_MAX = None
    if args.corridor_x:
        cfg.CORRIDOR_X_MIN, cfg.CORRIDOR_X_MAX = args.corridor_x
    if args.lj_r0:
        alpha = args.lj_r0 / cfg.LJ_R0
        if args.lj_scale_epsilon:
            cfg.LJ_EPSILON *= alpha
            cfg.LJ_R_CUT *= alpha
            cfg.LJ_R_ALIGN *= alpha
        cfg.LJ_R0 = args.lj_r0
    if args.safety:
        cfg.SAFETY_LAYERS_ENABLED = args.safety == "on"
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


async def steady_pose(robot, window_s=2.0, tol_m=0.01, timeout_s=25.0):
    """Waits until the own pose has been steady (x,y spread < tol_m over window_s) and returns
    (x, y, z, yaw) averaged over that window -- the estimator needs a few seconds after a reset
    (or after the robot stopped) before its output means anything."""
    t_end = time.time() + timeout_s
    window = []
    while time.time() < t_end:
        p = robot.own_pose_record()
        window = [w for w in window if time.time() - w[0] < window_s]
        if p is not None:
            window.append((time.time(), p))
        if len(window) >= int(window_s / 0.1) - 3 and time.time() - window[0][0] > window_s - 0.3:
            xs = [w[1][0] for w in window]
            ys = [w[1][1] for w in window]
            if max(xs) - min(xs) < tol_m and max(ys) - min(ys) < tol_m:
                yaws = [w[1][3] for w in window]
                return (sum(xs) / len(xs), sum(ys) / len(ys), sum(w[1][2] for w in window) / len(window),
                        math.atan2(sum(math.sin(y) for y in yaws), sum(math.cos(y) for y in yaws)))
        await asyncio.sleep(0.1)
    return None


async def calibrate(robot, args):
    units = 150
    seconds = 4.0
    print("Waiting for a steady Lighthouse pose ...")
    p0 = await steady_pose(robot)
    if p0 is None:
        raise SystemExit("No steady Lighthouse pose within 25 s -- robot in view of the stations? "
                         "(lights off, deck up, not being moved). Nothing was driven.")
    print(f"start pose x={p0[0]:+.3f} y={p0[1]:+.3f} yaw={math.degrees(p0[3]):+.0f} deg. "
          f"Driving straight ahead at motor target {units} for {seconds:.0f} s -- keep the path clear (~0.4 m).")
    try:
        await robot.drive(units, units)
        await asyncio.sleep(seconds)
    finally:
        await robot.stop()
    await asyncio.sleep(1.0)
    p1 = await steady_pose(robot)
    if p1 is None:
        raise SystemExit("Pose did not settle after the drive -- lost the stations while moving? "
                         "Rerun with the robot starting in the open.")
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    dist = math.hypot(dx, dy)
    if dist < 0.03:
        raise SystemExit(f"Robot moved only {dist * 100:.1f} cm -- motors not driving? (Device Manager / "
                         "battery / wheels blocked). No calibration written.")
    travel = math.atan2(dy, dx)
    offset = (travel - p0[3] + math.pi) % (2 * math.pi) - math.pi
    motor = units / (dist / seconds)
    print(f"travelled {dist:.3f} m (dx={dx:+.3f}, dy={dy:+.3f}) in {seconds:.0f} s")
    print(f"travel bearing {math.degrees(travel):+.1f} deg, board yaw at start {math.degrees(p0[3]):+.1f} deg")
    print(f"  --heading-offsets {args.self_hostname}={offset:.4f}")
    print(f"  suggested MOTOR_UNITS_PER_MPS = {motor:.1f} (current {cfg.MOTOR_UNITS_PER_MPS}) "
          f"-- pass as --motor-units-per-mps")
    print(f"RESULT {args.self_hostname} heading_offset={offset:.4f} motor_units_per_mps={motor:.1f} "
          f"distance_m={dist:.3f}")


async def calibrate_turn(robot, args):
    """Spins the robot IN PLACE (left = -u, right = +u: counter-clockwise for u > 0) and compares the realised yaw rate
    (Lighthouse yaw, steady before/after) with the nominal one from the kinematics in motor_utils.py. ratio < 1 means the
    robot turns less than commanded; WHEEL_DISTANCE_M / ratio is the effective track width that would fix it."""
    nominal = lambda u: 2.0 * u / (cfg.MOTOR_UNITS_PER_MPS * cfg.WHEEL_DISTANCE_M)
    print("Spinning in place at three motor levels (the robot stays where it is) ...")
    results = []
    for u, dur in ((50, 4.0), (-50, 4.0), (90, 3.0)):
        p0 = await steady_pose(robot)
        if p0 is None:
            raise SystemExit("No steady pose before the spin -- nothing was driven.")
        try:
            await robot.drive(-u, u)
            await asyncio.sleep(dur)
        finally:
            await robot.stop()
        await asyncio.sleep(1.0)
        p1 = await steady_pose(robot)
        if p1 is None:
            raise SystemExit("Pose did not settle after the spin.")
        dyaw = (p1[3] - p0[3] + math.pi) % (2 * math.pi) - math.pi
        rate = dyaw / dur
        ratio = rate / nominal(u)
        drift = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        results.append((u, rate, nominal(u), ratio))
        print(f"  motor +-{abs(u):3d}: realised {rate:+.3f} rad/s, nominal {nominal(u):+.3f} rad/s -> ratio {ratio:.2f} "
              f"(centre drift {drift * 100:.1f} cm)")
    ratios = [r[3] for r in results]
    mean_ratio = sum(ratios) / len(ratios)
    print(f"  mean ratio {mean_ratio:.2f}  ->  effective track width {cfg.WHEEL_DISTANCE_M / mean_ratio:.3f} m "
          f"(config has {cfg.WHEEL_DISTANCE_M})")
    print(f"RESULT {args.self_hostname} turn_ratio_50={results[0][3]:.3f} turn_ratio_m50={results[1][3]:.3f} "
          f"turn_ratio_90={results[2][3]:.3f} mean={mean_ratio:.3f}")


async def main():
    args = parse_args()
    apply_config(args)
    hostnames = args.hostnames.split(",")
    ids = [int(i) for i in args.ids.split(",")]
    assert len(hostnames) == len(ids) and args.self_hostname in hostnames
    assert len(set(ids)) == len(ids) and all(1 <= i <= 255 for i in ids), "ids must be unique, 1..255"

    robot = LighthouseRobot(args.self_hostname, dict(zip(hostnames, ids)), args.uri, tuple(args.origin))
    await robot.connect()
    try:
        if args.check:
            await check(robot)
            return
        if args.calibrate_heading:
            await calibrate(robot, args)
            return
        if args.layout:
            p = await steady_pose(robot, timeout_s=30.0)
            if p is None:
                raise SystemExit("No steady pose within 30 s.")
            await asyncio.sleep(2.0)      # let the neighbour table fill
            print(f"LAYOUT {args.self_hostname} {p[0]:+.3f} {p[1]:+.3f}")
            by_id = dict(robot.board.neighbors)
            for host, rid in zip(hostnames, ids):
                if host != args.self_hostname:
                    if rid in by_id:
                        print(f"LAYOUT {host} {by_id[rid][0] - 0:+.3f} {by_id[rid][1] - 0:+.3f}")
                    else:
                        print(f"LAYOUT {host} MISSING")
            return
        if args.calibrate_turn:
            await calibrate_turn(robot, args)
            return
        if args.where:
            p = await steady_pose(robot, timeout_s=30.0)
            if p is None:
                raise SystemExit("No steady pose within 30 s (not in view of the stations, or being moved).")
            print(f"WHERE {args.self_hostname} x={p[0]:+.3f} y={p[1]:+.3f} z={p[2]:+.3f} yaw_deg={math.degrees(p[3]):+.0f}")
            return
        if args.controller == "hebbian" and not args.genome:
            raise SystemExit("--genome is required for --controller hebbian")
        if args.controller == "lj":
            from lj_baseline_experiment import LJBaselineExperiment as Experiment
        else:
            from hebbian_swarm_experiment import HebbianSwarmExperiment as Experiment
        os.makedirs(args.log_dir, exist_ok=True)
        log_path = os.path.join(args.log_dir, f"{args.self_hostname}_{int(time.time())}.csv")
        conf = {"hostnames": hostnames, "self_hostname": args.self_hostname}
        if args.controller == "hebbian":
            conf["genome_path"] = args.genome
        experiment = Experiment(robot=robot, logger=CsvLogger(log_path), config=conf)
        # The estimator was reset at connect and needs a few seconds to converge on the Lighthouse fix (a robot that
        # starts with an unconverged pose broadcasts a wrong position to everyone -- seen: y = +7 m at the start of a
        # run). Require a STEADY pose before taking part; otherwise this robot sits the run out.
        print("waiting for a steady Lighthouse pose ...", flush=True)
        wait_s = 90.0 if args.go_file else 25.0     # under the handshake nobody has started yet, so slow robots can take their time
        if await steady_pose(robot, timeout_s=wait_s) is None:
            raise SystemExit(f"no steady Lighthouse pose within {wait_s:.0f} s -- this robot does NOT take part in the run")
        if args.go_file:
            if args.ready_file:
                open(args.ready_file, "w").write("ready\n")
            print("READY -- waiting for the go signal", flush=True)
            t_wait = time.time() + 180.0
            while not os.path.exists(args.go_file):
                if time.time() > t_wait:
                    raise SystemExit("no go signal within 180 s -- this robot does NOT take part in the run")
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.05)
            args.start_at, args.stop_at = [float(v) for v in open(args.go_file).read().split()[:2]]
        if args.start_at:
            late = time.time() - args.start_at
            if late > 2.0:
                # a robot that is only ready after the others have gone would shift its whole run in time and desynchronise
                # the swarm -- it sits this run out instead (the run is then n-1 robots; run_swarm.sh reports it)
                raise SystemExit(f"NOT READY at the start time ({late:.1f} s late) -- this robot does NOT take part in the run")
            await asyncio.sleep(max(0.0, args.start_at - time.time()))
        else:
            await asyncio.get_running_loop().run_in_executor(None, input, "Press Enter to start ... ")

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(experiment.stop()))
        if args.stop_at:
            loop.call_later(max(0.0, args.stop_at - time.time()), lambda: asyncio.ensure_future(experiment.stop()))
        elif args.duration:
            loop.call_later(args.duration, lambda: asyncio.ensure_future(experiment.stop()))
        print(f"running, logging to {log_path}  (Ctrl-C to stop)")
        await experiment.run()
    finally:
        await robot.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
