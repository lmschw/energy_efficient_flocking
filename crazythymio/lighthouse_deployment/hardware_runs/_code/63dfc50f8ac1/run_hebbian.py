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
    p.add_argument("--controller", choices=["hebbian", "lj", "idle"], default="hebbian",
                   help="hebbian = evolved genome, lj = the paper's LJ rule-based baseline (no genome)")
    p.add_argument("--uri", default="usb://0")
    p.add_argument("--origin", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"),
                   help="Lighthouse coordinates [m] of the arena center = simulation (0, 0)")
    p.add_argument("--corridor-y", type=float, nargs=2, metavar=("MIN", "MAX"),
                   help="arena y-limits in the shifted frame; enables the wall slow-down governor")
    p.add_argument("--measure-offset", action="store_true",
                   help="turn in place in 45 deg steps through a full circle and fit where the board sits relative to the "
                        "Thymio's centre of rotation (the firmware's BOARD_OFFSET)")
    p.add_argument("--latency-test", action="store_true",
                   help="measure the total delay from a motor command to the first visible change in the reported heading")
    p.add_argument("--calibrate-turn", action="store_true",
                   help="spin in place at a few motor levels and report realised vs nominal turn rate (robot stays put)")
    p.add_argument("--lj-r0", type=float, help="override the LJ preferred spacing r0 [m] (paper: 0.7); LJ controller only")
    p.add_argument("--lj-scale-epsilon", action="store_true",
                   help="with --lj-r0: also scale everything else that has units of length or force in the LJ law by "
                        "alpha = r0/0.7 (epsilon, cutoff radius, alignment radius), so the force at every scaled distance "
                        "equals the paper's and the gains K1, K2, k_goal stay valid")
    p.add_argument("--log-period-ms", type=int, help="override controller_config.LOG_PERIOD_MS")
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
    p.add_argument("--serve", metavar="DIR", help="stay connected and run one experiment per GO file in DIR (see serve())")
    p.add_argument("--code-hash", default="", help="identifier of the deployed code; reported in DIR/ready")
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
    if args.log_period_ms:
        cfg.LOG_PERIOD_MS = args.log_period_ms
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


async def measure_offset(robot, args):
    """The firmware reports c = board + R(yaw) * e_fw (e_fw = BOARD_OFFSET_X/Y, board frame). The robot turns IN PLACE about
    its true centre p, so with the true offset e_true:  c(yaw) = p + R(yaw) * (e_fw - e_true).  Turning in 45 deg steps
    through a full circle and fitting p and d = e_fw - e_true by least squares gives e_true = e_fw - d."""
    import numpy as np
    E_FW = getattr(robot.board, "board_offset", (-0.09, 0.04))   # what the firmware applies right now (hebb.offx/offy)
    pts = []
    for k in range(9):
        p = await steady_pose(robot, window_s=1.5)
        if p is None:
            raise SystemExit("No steady pose -- stopping (nothing else is driven).")
        pts.append((p[0], p[1], p[3]))
        print(f"  stop {k + 1}/9: reported centre ({p[0]:+.3f}, {p[1]:+.3f})  board yaw {math.degrees(p[3]):+5.0f} deg", flush=True)
        if k == 8:
            break
        try:
            await robot.drive(-50, 50)
            await asyncio.sleep(1.95)            # ~45 deg at the measured ~0.40 rad/s for +-50 units
        finally:
            await robot.stop()
        await asyncio.sleep(0.8)
    A = []; b = []
    for x, y, yw in pts:
        c, s_ = math.cos(yw), math.sin(yw)
        A.append([1, 0, c, -s_]); b.append(x)
        A.append([0, 1, s_, c]); b.append(y)
    sol, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    px, py, dx, dy = sol
    res = np.array(b) - np.array(A) @ sol
    ex, ey = E_FW[0] - dx, E_FW[1] - dy
    yaws = sorted(math.degrees(p[2]) % 360 for p in pts)
    cover = 360 - max((yaws[(i + 1) % len(yaws)] - yaws[i]) % 360 for i in range(len(yaws)))
    print(f"  centre of rotation ({px:+.3f}, {py:+.3f}); fit residual {np.sqrt((res ** 2).mean()) * 100:.1f} cm; "
          f"headings covered {cover:.0f} deg")
    print(f"  firmware offset now ({E_FW[0]:+.3f}, {E_FW[1]:+.3f}) m  ->  TRUE board offset ({ex:+.3f}, {ey:+.3f}) m "
          f"(board frame: x forward, y left); current error {math.hypot(dx, dy) * 100:.1f} cm")
    print(f"RESULT {args.self_hostname} board_offset_x={ex:.4f} board_offset_y={ey:.4f} "
          f"error_cm={math.hypot(dx, dy) * 100:.1f} residual_cm={np.sqrt((res ** 2).mean()) * 100:.1f} coverage_deg={cover:.0f}")


async def latency_test(robot, args):
    """Robot stationary -> command an in-place spin at a known time -> watch the REPORTED heading (the same data the
    controller uses) -> fit a line to the turning part and extrapolate back to the baseline: the time where that line
    starts is the total dead time (command -> TDM -> motor -> wheels -> Lighthouse/Kalman -> USB log -> Pi)."""
    import statistics
    u = 90
    w_nom = 2.0 * u / (cfg.MOTOR_UNITS_PER_MPS * cfg.WHEEL_DISTANCE_M)
    deads = []
    for rep, sign in enumerate((1, -1, 1, -1)):
        if await steady_pose(robot) is None:
            raise SystemExit("No steady pose -- nothing was driven.")
        samples = []
        last_t = None
        t_cmd = time.time()
        await robot.drive(-sign * u, sign * u)
        while time.time() - t_cmd < 1.2:
            if robot.board.own_time != last_t and robot.board.own:
                last_t = robot.board.own_time
                samples.append((robot.board.own_time - t_cmd, math.radians(robot.board.own["stateEstimate.yaw"])))
            await asyncio.sleep(0.002)
        await robot.stop()
        await asyncio.sleep(1.0)
        t = [s[0] for s in samples]
        y = [math.atan2(math.sin(s[1] - samples[0][1]), math.cos(s[1] - samples[0][1])) * sign for s in samples]
        base = [yy for tt, yy in zip(t, y) if tt < 0.03] or [0.0]
        b = statistics.mean(base)
        pts = [(tt, yy) for tt, yy in zip(t, y) if math.radians(4) < yy - b < math.radians(30)]
        if len(pts) < 3:
            print(f"  spin {rep + 1}: too few turning samples ({len(pts)})"); continue
        mt = statistics.mean(p[0] for p in pts); my = statistics.mean(p[1] for p in pts)
        slope = sum((p[0] - mt) * (p[1] - my) for p in pts) / sum((p[0] - mt) ** 2 for p in pts)
        dead = mt - (my - b) / slope
        deads.append(dead)
        print(f"  spin {rep + 1}: dead time {dead * 1000:5.0f} ms, turn rate once moving {slope:.2f} rad/s (nominal {w_nom:.2f}), "
              f"{len(samples)} heading samples in 1.2 s (log period {cfg.LOG_PERIOD_MS} ms)")
    if deads:
        print(f"RESULT {args.self_hostname} dead_time_ms={statistics.median(deads) * 1000:.0f} log_period_ms={cfg.LOG_PERIOD_MS}")


class _Tee:
    """Duplicates console output into the run's own console file (serve mode keeps one process over many runs)."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for st in self.streams:
            st.write(text)
            st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


def _make_idle_experiment_class():
    from lj_baseline_experiment import LJBaselineExperiment
    from pose_utils import poses_to_agents

    class IdleExperiment(LJBaselineExperiment):
        """Logs exactly like a real run but never moves: motors 0 on every tick. For testing the pipeline on hardware."""
        async def _tick(self):
            self._tick_count += 1
            poses = await self.robot.get_all_global_poses()
            agents, i = poses_to_agents(poses, self.hostnames, self.self_hostname)
            await self.robot.drive(0, 0)
            if self.logger:
                self.logger.log(state={"tick": self._tick_count, "timestamp": time.time(), "x": float(agents[i, 0]),
                                       "y": float(agents[i, 1]), "heading": float(agents[i, 2]),
                                       "tracked": int(abs(agents[i, 0]) < cfg.UNTRACKED_XY_THRESHOLD),
                                       "n_neighbors_seen": sum(1 for h in self.hostnames
                                                               if h != self.self_hostname and poses.get(h) is not None)},
                                command={"v": 0.0, "w": 0.0, "left": 0, "right": 0})
            return 0.0, 0.0, 0, 0
    return IdleExperiment


_RUN_KEYS = ("LJ_R0", "LJ_EPSILON", "LJ_R_CUT", "LJ_R_ALIGN", "SAFETY_LAYERS_ENABLED", "CORRIDOR_X_MIN", "CORRIDOR_X_MAX",
             "CORRIDOR_Y_MIN", "CORRIDOR_Y_MAX", "BATTERY_MODE")


async def serve(robot, args, hostnames, ids):
    """SERVE MODE: the robot stays connected (Thymio + Crazyflie) across runs, so there is no reconnect between runs.
    Protocol, all files in args.serve on this Pi (written atomically by tools/run_swarm.sh):
      ready            written by THIS process once the pose is steady -- {"time", "code"}; removed when a GO is taken
      go               JSON run spec: start, stop (THIS Pi's clock), stamp, log_dir, controller, genome, lj_r0, lj_scale,
                       safety, corridor_y, corridor_x
      done_<stamp>     written after the run: {"status": "ok"|"late"|"error", "log": path}
      quit             makes this process exit
    Every run gets a FRESH experiment object (fresh random Hebbian weights, full simulated battery) and its own config
    (restored from the values this process started with, then the run spec applied)."""
    import json
    d = args.serve
    os.makedirs(d, exist_ok=True)
    base = {k: getattr(cfg, k) for k in _RUN_KEYS}
    state = {"quit": False, "exp": None}
    loop = asyncio.get_running_loop()

    def _on_signal():
        state["quit"] = True
        if state["exp"] is not None:
            asyncio.ensure_future(state["exp"].stop())
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)
    print(f"SERVE mode, code {args.code_hash}", flush=True)
    while not state["quit"] and not os.path.exists(os.path.join(d, "quit")):
        if await steady_pose(robot, timeout_s=5.0) is None:
            continue                               # robot being moved / not in view yet: keep waiting
        json.dump({"time": time.time(), "code": args.code_hash}, open(os.path.join(d, "ready.tmp"), "w"))
        os.replace(os.path.join(d, "ready.tmp"), os.path.join(d, "ready"))
        go = os.path.join(d, "go")
        while not os.path.exists(go) and not state["quit"] and not os.path.exists(os.path.join(d, "quit")):
            await asyncio.sleep(0.05)
        if not os.path.exists(go):
            break
        spec = json.load(open(go))
        os.remove(go)
        try:
            os.remove(os.path.join(d, "ready"))
        except FileNotFoundError:
            pass
        status, log_path = "error", None
        os.makedirs(spec["log_dir"], exist_ok=True)
        console = open(os.path.join(spec["log_dir"], f"{args.self_hostname}.console.txt"), "w")
        real_stdout = sys.stdout
        sys.stdout = _Tee(real_stdout, console)
        try:
            for k, v in base.items():
                setattr(cfg, k, v)
            if spec.get("lj_r0"):
                alpha = spec["lj_r0"] / cfg.LJ_R0
                if spec.get("lj_scale"):
                    cfg.LJ_EPSILON *= alpha; cfg.LJ_R_CUT *= alpha; cfg.LJ_R_ALIGN *= alpha
                cfg.LJ_R0 = spec["lj_r0"]
            cfg.SAFETY_LAYERS_ENABLED = bool(spec.get("safety", False))
            cfg.CORRIDOR_Y_MIN, cfg.CORRIDOR_Y_MAX = spec.get("corridor_y") or (None, None)
            cfg.CORRIDOR_X_MIN, cfg.CORRIDOR_X_MAX = spec.get("corridor_x") or (None, None)
            conf = {"hostnames": hostnames, "self_hostname": args.self_hostname}
            if spec["controller"] == "lj":
                from lj_baseline_experiment import LJBaselineExperiment as Experiment
            elif spec["controller"] == "idle":
                Experiment = _make_idle_experiment_class()
            else:
                from hebbian_swarm_experiment import HebbianSwarmExperiment as Experiment
                conf["genome_path"] = spec["genome"]
            log_path = os.path.join(spec["log_dir"], f"{args.self_hostname}_{int(time.time())}.csv")
            exp = Experiment(robot=robot, logger=CsvLogger(log_path), config=conf)
            late = time.time() - spec["start"]
            if late > 2.0:
                print(f"NOT READY at the start time ({late:.1f} s late) -- sitting this run out", flush=True)
                status = "late"
            else:
                print(f"run {spec['stamp']}: {spec['controller']} {spec.get('genome') or ''}, logging to {log_path}", flush=True)
                await asyncio.sleep(max(0.0, spec["start"] - time.time()))
                state["exp"] = exp
                handle = loop.call_later(max(0.0, spec["stop"] - time.time()), lambda: asyncio.ensure_future(exp.stop()))
                await exp.run()
                handle.cancel()
                status = "ok"
        except Exception as e:
            print("run failed:", type(e).__name__, e, flush=True)
        finally:
            state["exp"] = None
            for _ in range(3):
                try:
                    await asyncio.wait_for(robot.stop(), timeout=2.0)
                    break
                except Exception as e:
                    print("motor stop failed, retrying:", type(e).__name__, flush=True)
            sys.stdout = real_stdout
            console.close()
            json.dump({"status": status, "log": log_path}, open(os.path.join(d, f"done_{spec['stamp']}.tmp"), "w"))
            os.replace(os.path.join(d, f"done_{spec['stamp']}.tmp"), os.path.join(d, f"done_{spec['stamp']}"))
    print("SERVE mode ended", flush=True)


async def main():
    args = parse_args()
    apply_config(args)
    hostnames = args.hostnames.split(",")
    ids = [int(i) for i in args.ids.split(",")]
    assert len(hostnames) == len(ids) and args.self_hostname in hostnames
    assert len(set(ids)) == len(ids) and all(1 <= i <= 255 for i in ids), "ids must be unique, 1..255"

    # ONE controller per Pi: a second instance would fight over the Thymio lock and break the first one's connection
    # (seen: the first then could not stop its motors at the end of a run and the robot kept driving at full speed).
    import fcntl
    global _instance_lock
    _instance_lock = open("/tmp/run_hebbian.lock", "w")
    try:
        fcntl.flock(_instance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("another run_hebbian.py is already running on this Pi -- refusing to start a second one")
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
        if args.measure_offset:
            await measure_offset(robot, args)
            return
        if args.latency_test:
            await latency_test(robot, args)
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
        if args.serve:
            await serve(robot, args, hostnames, ids)
            return
        if args.controller == "hebbian" and not args.genome:
            raise SystemExit("--genome is required for --controller hebbian")
        if args.controller == "lj":
            from lj_baseline_experiment import LJBaselineExperiment as Experiment
        elif args.controller == "idle":
            Experiment = _make_idle_experiment_class()
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
        wait_s = 1800.0 if args.go_file else 25.0   # under the handshake nobody has started yet; the operator may still be placing robots
        if await steady_pose(robot, timeout_s=wait_s) is None:
            raise SystemExit(f"no steady Lighthouse pose within {wait_s:.0f} s -- this robot does NOT take part in the run")
        if args.go_file:
            if args.ready_file:
                open(args.ready_file, "w").write("ready\n")
            print("READY -- waiting for the go signal", flush=True)
            t_wait = time.time() + 1800.0
            while not os.path.exists(args.go_file):
                if time.time() > t_wait:
                    raise SystemExit("no go signal within 30 min -- this robot does NOT take part in the run")
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
        # make sure the motors are told to stop, but never hang here (a hung exit once left a robot driving)
        for _ in range(3):
            try:
                await asyncio.wait_for(robot.stop(), timeout=2.0)
                break
            except Exception as e:
                print("motor stop failed, retrying:", type(e).__name__, flush=True)
        try:
            await asyncio.wait_for(robot.disconnect(), timeout=5.0)
        except Exception as e:
            print("disconnect did not finish cleanly:", type(e).__name__, flush=True)


if __name__ == "__main__":
    code = 0
    try:
        asyncio.run(main())
    except SystemExit as e:
        print(e, flush=True)
        code = 1
    finally:
        sys.stdout.flush()
        # cflib / tdmclient leave non-daemon threads behind that can keep the interpreter alive forever after main()
        # returned: exit hard so the process (and the launcher waiting for it) really ends.
        os._exit(code)
