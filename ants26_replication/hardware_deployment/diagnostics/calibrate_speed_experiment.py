"""Speed calibration experiment -- NOT the real controller. Drives straight at a sweep
of raw motor.target values and measures actual speed from OptiTrack position deltas, to
replace controller_config.MOTOR_UNITS_PER_MPS's unverified placeholder with a measured
value (see ../README.md "Calibration"). There is no wheel odometry on this platform (see
README's "Known open risks"), so OptiTrack position deltas are the only ground truth
available -- same reasoning wind_battery_model.py's speed estimate already relies on.

Deploy the same way as the real controller (see ../README.md), with
config = {"self_hostname": "...", "motor_targets": [...], "hold_seconds": ...,
"settle_seconds": ...} -- no genome_path or hostnames list needed, since this doesn't
sense neighbors, just its own tracked position.

Drives the robot straight for `hold_seconds` at each of `motor_targets` in turn (default
[100, 200, 300, 400, 500], i.e. fractions of MAX_MOTOR_TARGET=500), with a `settle_seconds`
full stop between legs. At LINEAR_VEL_MAX=0.2 m/s and the defaults below, the whole sweep
can cover a couple of meters in a straight line -- give the robot a clear, straight,
obstacle-free runway before starting, and don't run multiple robots' calibration sweeps
where their runways could cross (they don't sense each other in this experiment, so
nothing stops them colliding if you point two at each other).

Prints a per-leg measured speed and, at the end, a recommended MOTOR_UNITS_PER_MPS fit
via least-squares through the origin (target ~= k * measured_speed) over all legs, and
logs every leg plus the final recommendation via the platform's SessionLogger so
run_speed_calibration.py can aggregate it back on the controller machine.
"""
import asyncio
import os
import sys

# See hebbian_swarm_experiment.py's identical comment: the daemon only puts the project
# ROOT on sys.path, not this file's own directory -- needed once deployed in a subpackage.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time

import controller_config as cfg

DEFAULT_MOTOR_TARGETS = [100, 200, 300, 400, 500]
DEFAULT_HOLD_SECONDS = 3.0
DEFAULT_SETTLE_SECONDS = 2.0


class CalibrateSpeedExperiment:
    def __init__(self, robot, config=None, logger=None):
        self.robot = robot
        self.config = config or {}
        self.logger = logger
        self.running = True
        self.paused = False

        self.self_hostname = self.config.get("self_hostname")
        if not self.self_hostname:
            raise ValueError("config['self_hostname'] is required (used to look up this "
                              "robot's own tracked pose).")
        self.motor_targets = list(self.config.get("motor_targets", DEFAULT_MOTOR_TARGETS))
        self.hold_seconds = float(self.config.get("hold_seconds", DEFAULT_HOLD_SECONDS))
        self.settle_seconds = float(self.config.get("settle_seconds", DEFAULT_SETTLE_SECONDS))

    def _own_xy(self, poses):
        pose = poses.get(self.self_hostname)
        if pose is None:
            return None
        ax0, ax1 = cfg.POSITION_AXES
        return pose.position[ax0], pose.position[ax1]

    def _log(self, phase, motor_target=None, start_xy=None, end_xy=None,
             elapsed_s=None, measured_speed_mps=None, recommended_units_per_mps=None):
        if not self.logger:
            return
        sx, sy = start_xy if start_xy is not None else (None, None)
        ex, ey = end_xy if end_xy is not None else (None, None)
        self.logger.log(
            state={"phase": phase, "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey,
                   "elapsed_s": elapsed_s, "measured_speed_mps": measured_speed_mps,
                   "recommended_units_per_mps": recommended_units_per_mps},
            command={"motor_target": motor_target},
        )

    async def run(self):
        print(f"[{self.self_hostname}] speed calibration starting -- targets="
              f"{self.motor_targets}, hold={self.hold_seconds}s, settle={self.settle_seconds}s. "
              f"Make sure the robot has a clear straight runway.")

        results = []
        for target in self.motor_targets:
            if not self.running:
                break
            while self.paused and self.running:
                await self.robot.stop()
                await asyncio.sleep(0.1)
            if not self.running:
                break

            poses = await self.robot.get_all_global_poses()
            start_xy = self._own_xy(poses)
            if start_xy is None:
                print(f"[{self.self_hostname}] not currently tracked -- skipping target={target}")
                continue

            t0 = time.monotonic()
            await self.robot.drive(target, target)
            await asyncio.sleep(self.hold_seconds)
            await self.robot.stop()
            elapsed = time.monotonic() - t0

            poses = await self.robot.get_all_global_poses()
            end_xy = self._own_xy(poses)
            await asyncio.sleep(self.settle_seconds)  # let it fully stop before the next leg

            if end_xy is None:
                print(f"[{self.self_hostname}] lost tracking mid-leg -- skipping target={target}")
                continue

            dist = ((end_xy[0] - start_xy[0]) ** 2 + (end_xy[1] - start_xy[1]) ** 2) ** 0.5
            speed = dist / elapsed
            print(f"[{self.self_hostname}] target={target} -> distance={dist:.3f} m over "
                  f"{elapsed:.2f} s = {speed:.4f} m/s")
            results.append((target, speed))
            self._log("measure", motor_target=target, start_xy=start_xy, end_xy=end_xy,
                       elapsed_s=elapsed, measured_speed_mps=speed)

        if results:
            sum_v2 = sum(v * v for _, v in results)
            sum_tv = sum(t * v for t, v in results)
            recommended_k = sum_tv / sum_v2 if sum_v2 > 0 else float("nan")
            print(f"[{self.self_hostname}] recommended MOTOR_UNITS_PER_MPS = "
                  f"{recommended_k:.2f} (currently {cfg.MOTOR_UNITS_PER_MPS:.2f} in "
                  f"controller_config.py)")
            self._log("summary", recommended_units_per_mps=recommended_k)
        else:
            print(f"[{self.self_hostname}] no usable measurements -- check OptiTrack tracking "
                  f"and hostname_map before retrying.")

        await self.robot.stop()

    async def pause(self):
        self.paused = True

    async def resume(self):
        self.paused = False

    async def stop(self):
        self.running = False
