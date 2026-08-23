"""Position/heading calibration experiment -- NOT the real controller. Drives straight
forward for a fixed duration and DERIVES controller_config.POSITION_AXES and
HEADING_OFFSET_RAD from the resulting OptiTrack position/orientation stream, instead of
requiring you to manually point the robot in a specific direction and eyeball raw
values against print_poses_experiment.py's live feed. Same motivation as
calibrate_speed_experiment.py: there's no wheel odometry on this platform, so OptiTrack
is the only ground truth, and a straight drive gives it something unambiguous to
measure -- a regression over many samples is also inherently more robust to single-
sample OptiTrack noise/jitter than reading one or two live values by eye.

Deploy the same way as the real controller (see ../README.md), with
config = {"self_hostname": "...", "motor_target": <int>, "drive_seconds": <float>} --
no genome_path or hostnames list needed, since this doesn't sense neighbors, just its
own tracked pose.

What this CAN determine from a straight drive alone:
  - POSITION_AXES: whichever 2 of OptiTrack's raw (x, y, z) axes actually change during
    the drive are the ground-plane axes; the ~constant one is "up" and gets excluded.
  - HEADING_OFFSET_RAD: this codebase's convention is heading=0 means "facing +y" in the
    (POSITION_AXES-selected) sim frame (see move() in
    initial_implementation/experiment/simulation_free_global_mod_2_LJ.py and
    print_poses_experiment.py's calibration notes) -- i.e. moving straight forward at
    heading=0 produces a travel bearing of +pi/2 in that frame. Comparing the ACTUAL
    measured travel bearing against the robot's raw yaw reading at the same time gives
    HEADING_OFFSET_RAD directly, PROVIDED you already know ROTATION_SIGN.

What this CANNOT determine (fundamentally, not a limitation of this script specifically):
  - ROTATION_SIGN: a pure straight-line drive (w=0 the whole time) carries no information
    about which way positive angular velocity turns the robot. Two HEADING_OFFSET_RAD
    candidates are reported, one per ROTATION_SIGN hypothesis -- determine ROTATION_SIGN
    separately by commanding a turn and observing which way it goes (see ../README.md),
    then use whichever candidate matches.

Also reports a rough single-point MOTOR_UNITS_PER_MPS as a free bonus/sanity-check from
the same drive -- calibrate_speed_experiment.py's multi-target least-squares sweep
remains the authoritative source for that constant specifically.
"""
import asyncio
import os
import sys

# See hebbian_swarm_experiment.py's identical comment: the daemon only puts the project
# ROOT on sys.path, not this file's own directory -- needed once deployed in a subpackage.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import time

import numpy as np

import controller_config as cfg
from pose_utils import quaternion_to_yaw

DEFAULT_MOTOR_TARGET = 300
DEFAULT_DRIVE_SECONDS = 10.0
SAMPLE_INTERVAL_S = 0.5  # matches OptiTrack's own ~2 Hz push rate -- no point polling faster
PRE_DRIVE_SAMPLES = 4    # ~2s stationary, to confirm tracking before committing to the drive


def _wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class CalibratePositionHeadingExperiment:
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
        self.motor_target = int(self.config.get("motor_target", DEFAULT_MOTOR_TARGET))
        self.drive_seconds = float(self.config.get("drive_seconds", DEFAULT_DRIVE_SECONDS))

    def _own_pose(self, poses):
        return poses.get(self.self_hostname)

    async def _collect_samples(self, duration_s, driving):
        """Polls this robot's own pose every SAMPLE_INTERVAL_S for duration_s, returning
        a list of (t, x_raw, y_raw, z_raw, raw_yaw) tuples (t relative to this call's
        start). Samples where the robot isn't currently tracked are skipped."""
        samples = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration_s:
            if not self.running:
                break
            while self.paused and self.running:
                if driving:
                    await self.robot.stop()
                await asyncio.sleep(0.1)
            poses = await self.robot.get_all_global_poses()
            pose = self._own_pose(poses)
            if pose is not None:
                raw_yaw = quaternion_to_yaw(*pose.orientation)
                samples.append((time.monotonic() - t0, pose.position[0], pose.position[1],
                                 pose.position[2], raw_yaw))
            await asyncio.sleep(SAMPLE_INTERVAL_S)
        return samples

    async def run(self):
        print(f"[{self.self_hostname}] position/heading calibration starting -- "
              f"motor_target={self.motor_target}, drive_seconds={self.drive_seconds}. "
              f"Give the robot a clear, straight, obstacle-free runway.")

        print(f"[{self.self_hostname}] checking tracking before driving...")
        pre_samples = await self._collect_samples(PRE_DRIVE_SAMPLES * SAMPLE_INTERVAL_S, driving=False)
        if len(pre_samples) < 2:
            print(f"[{self.self_hostname}] not reliably tracked (got {len(pre_samples)} "
                  f"pose samples while stationary) -- aborting. Check OptiTrack/hostname_map "
                  f"before retrying.")
            await self.robot.stop()
            return
        print(f"[{self.self_hostname}] tracked OK ({len(pre_samples)} samples). Driving straight...")

        await self.robot.drive(self.motor_target, self.motor_target)
        drive_samples = await self._collect_samples(self.drive_seconds, driving=True)
        await self.robot.stop()

        print(f"[{self.self_hostname}] drive complete -- {len(drive_samples)} pose samples "
              f"collected during the drive.")
        if len(drive_samples) < 3:
            print(f"[{self.self_hostname}] too few samples to fit a reliable line "
                  f"({len(drive_samples)}) -- try a longer drive_seconds, or check tracking "
                  f"stability during motion.")
            return

        t = np.array([s[0] for s in drive_samples])
        raw = np.array([[s[1], s[2], s[3]] for s in drive_samples])  # (n, 3): raw x,y,z
        raw_yaws = np.array([s[4] for s in drive_samples])

        # Fit each raw axis against time; the axis with the smallest slope is "up" (a
        # wheeled ground robot shouldn't move vertically during a straight drive).
        slopes = np.zeros(3)
        r_squared = np.zeros(3)
        for axis in range(3):
            slope, intercept = np.polyfit(t, raw[:, axis], 1)
            pred = slope * t + intercept
            ss_res = np.sum((raw[:, axis] - pred) ** 2)
            ss_tot = np.sum((raw[:, axis] - raw[:, axis].mean()) ** 2)
            slopes[axis] = slope
            r_squared[axis] = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

        up_axis = int(np.argmin(np.abs(slopes)))
        position_axes = tuple(sorted(a for a in range(3) if a != up_axis))
        ax0, ax1 = position_axes

        print(f"[{self.self_hostname}] per-axis slope [m/s] and fit quality [R^2] "
              f"(the 'up' axis should have ~0 slope and a poor fit):")
        for axis in range(3):
            tag = " <- inferred UP (excluded)" if axis == up_axis else ""
            print(f"    raw axis {axis}: slope={slopes[axis]:+.4f} m/s  R^2={r_squared[axis]:.3f}{tag}")
        if r_squared[up_axis] > 0.5:
            print(f"[{self.self_hostname}] WARNING: the inferred 'up' axis still fits a "
                  f"line reasonably well (R^2={r_squared[up_axis]:.3f}) -- the robot may "
                  f"not have driven straight, or all 3 axes are noisy. Inspect the slopes "
                  f"above yourself before trusting POSITION_AXES={position_axes}.")

        dx = slopes[ax0] * self.drive_seconds
        dy = slopes[ax1] * self.drive_seconds
        travel_speed = math.hypot(slopes[ax0], slopes[ax1])
        travel_angle = math.atan2(dy, dx) if travel_speed > 1e-6 else None

        mean_raw_yaw = float(np.mean(np.unwrap(raw_yaws)))
        yaw_std = float(np.std(raw_yaws))

        print(f"\n[{self.self_hostname}] === Results ===")
        print(f"    Inferred POSITION_AXES = {position_axes}  (raw axis {up_axis} excluded as 'up')")
        print(f"    Measured travel speed = {travel_speed:.4f} m/s in the (ax{ax0}, ax{ax1}) plane")
        print(f"    Rough MOTOR_UNITS_PER_MPS ~= {self.motor_target / travel_speed:.1f}"
              if travel_speed > 1e-6 else "    Rough MOTOR_UNITS_PER_MPS: robot didn't move enough to estimate")
        print(f"    Raw yaw during drive: mean={mean_raw_yaw:+.4f} rad, std={yaw_std:.4f} rad"
              f"{'  (WARNING: high variance -- was the robot actually driving straight?)' if yaw_std > 0.1 else ''}")

        if travel_angle is not None:
            expected_heading_sim = _wrap_to_pi(travel_angle - math.pi / 2.0)
            offset_if_positive = _wrap_to_pi(expected_heading_sim - mean_raw_yaw)
            offset_if_negative = _wrap_to_pi(expected_heading_sim + mean_raw_yaw)
            print(f"    HEADING_OFFSET_RAD candidates (ROTATION_SIGN NOT determined by this "
                  f"test -- see module docstring):")
            print(f"      if ROTATION_SIGN =  1.0: HEADING_OFFSET_RAD = {offset_if_positive:+.4f}")
            print(f"      if ROTATION_SIGN = -1.0: HEADING_OFFSET_RAD = {offset_if_negative:+.4f}")
            print(f"    Determine ROTATION_SIGN with a separate turning test, then use the "
                  f"matching candidate above.")
        else:
            print(f"    Robot didn't move enough during the drive to measure a travel "
                  f"direction -- check motor_target and runway clearance, then retry.")

        if self.logger:
            self.logger.log(
                state={"position_axes": list(position_axes), "up_axis": up_axis,
                       "slopes_mps": slopes.tolist(), "r_squared": r_squared.tolist(),
                       "travel_speed_mps": travel_speed, "mean_raw_yaw": mean_raw_yaw,
                       "yaw_std": yaw_std,
                       "heading_offset_if_rotation_sign_positive":
                           offset_if_positive if travel_angle is not None else None,
                       "heading_offset_if_rotation_sign_negative":
                           offset_if_negative if travel_angle is not None else None},
                command={"motor_target": self.motor_target, "drive_seconds": self.drive_seconds},
            )

    async def pause(self):
        self.paused = True

    async def resume(self):
        self.paused = False

    async def stop(self):
        self.running = False
