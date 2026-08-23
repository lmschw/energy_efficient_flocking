"""Position/heading/speed calibration experiment -- NOT the real controller. Drives
straight (by default, ONE attempt -- see "Why a single attempt by default" below) and,
from the resulting OptiTrack data, DERIVES controller_config.POSITION_AXES,
HEADING_OFFSET_RAD, and MOTOR_UNITS_PER_MPS all at once -- one deployment instead of
separately running this alongside calibrate_speed_experiment.py. There's no wheel
odometry on this platform, so OptiTrack is the only ground truth, and a straight drive
gives it something unambiguous to measure -- a regression over many samples within that
one drive is also inherently more robust to single-sample OptiTrack noise/jitter than
reading one or two live values by eye (see print_poses_experiment.py, the original
manual approach this was built to replace).

Deploy the same way as the real controller (see ../README.md), with
config = {"self_hostname": "...", "motor_targets": [...], "hold_seconds": <float>,
"settle_seconds": <float>} -- no genome_path or hostnames list needed, since this
doesn't sense neighbors, just its own tracked pose.

Why a single attempt by default: real Thymios don't drive perfectly straight, and this
platform has no way to drive a robot back to an exact start position/heading (no wheel
odometry, no closed-loop control of any kind -- see "Known open risks" in ../README.md).
So a MULTI-leg sweep (motor_targets with more than one value) doesn't reset between
legs -- leg 2 starts from wherever leg 1's drift left the robot, not from leg 1's actual
start. One bad/curved leg then physically displaces every leg after it, which can run a
robot out of tracked volume or usable runway, and pulls the final (averaged-across-legs)
recommendation toward whatever that bad leg measured. A single ~10s drive is normally
enough for a good regression fit (see DEFAULT_HOLD_SECONDS) without that risk. If a
run looks bad (see the R^2/yaw-std warnings this prints), the fix is to manually put the
robot back at its start position and re-run with a single target again -- not to queue up
more targets and hope the sweep recovers.

What this CAN determine from a straight drive:
  - POSITION_AXES: whichever 2 of OptiTrack's raw (x, y, z) axes actually change during
    the drive are the ground-plane axes; the ~constant one is "up" and gets excluded.
    If you do pass multiple motor_targets, this is computed per leg as a consistency
    check -- they should all agree; a mismatch means something's wrong (bad tracking,
    non-straight motion, or drift between legs per the above).
  - HEADING_OFFSET_RAD: this codebase's convention is heading=0 means "facing +y" in the
    (POSITION_AXES-selected) sim frame (see move() in
    initial_implementation/experiment/simulation_free_global_mod_2_LJ.py and
    print_poses_experiment.py's calibration notes) -- i.e. moving straight forward at
    heading=0 produces a travel bearing of +pi/2 in that frame. Comparing the ACTUAL
    measured travel bearing against the robot's raw yaw reading at the same time gives
    HEADING_OFFSET_RAD directly, PROVIDED you already know ROTATION_SIGN.
  - MOTOR_UNITS_PER_MPS: target / measured_speed for a single leg -- exactly what a
    least-squares fit (calibrate_speed_experiment.py's approach) degenerates to with one
    data point, so no separate code path is needed for the single-attempt default.

What this CANNOT determine (fundamentally, not a limitation of this script specifically):
  - ROTATION_SIGN: a pure straight-line drive (w=0 the whole time) carries no information
    about which way positive angular velocity turns the robot. Two HEADING_OFFSET_RAD
    recommendations are reported, one per ROTATION_SIGN hypothesis -- determine
    ROTATION_SIGN separately by commanding a turn and observing which way it goes (see
    ../README.md), then use whichever candidate matches.

calibrate_speed_experiment.py is now largely redundant with this script (this one does
everything it does, plus position/heading, in one deployment) -- left in place rather
than deleted, in case you specifically want its narrower, multi-target speed-only sweep
(e.g. a one-time careful rig characterization where you have the space/patience to
reposition between legs yourself).
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

DEFAULT_MOTOR_TARGETS = [300]  # single attempt by default -- see module docstring
DEFAULT_HOLD_SECONDS = 10.0    # long enough for a good position/heading regression fit
DEFAULT_SETTLE_SECONDS = 2.0
SAMPLE_INTERVAL_S = 0.5  # matches OptiTrack's own ~2 Hz push rate -- no point polling faster
PRE_DRIVE_SAMPLES = 4    # ~2s stationary, to confirm tracking before committing to the sweep


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
        stale_keys = [k for k in ("motor_target", "drive_seconds") if k in self.config]
        if stale_keys:
            raise ValueError(
                f"config has {stale_keys}, which this experiment no longer accepts -- it "
                f"now takes 'motor_targets' (a list, default {DEFAULT_MOTOR_TARGETS} -- one "
                f"attempt by default, see module docstring for why) and 'hold_seconds' "
                f"(drive duration, "
                f"default {DEFAULT_HOLD_SECONDS}) instead -- passing the old keys would "
                f"otherwise silently fall back to these defaults and ignore whatever you "
                f"specified, since dict.get() doesn't know an old key from a typo.")
        self.motor_targets = list(self.config.get("motor_targets", DEFAULT_MOTOR_TARGETS))
        self.hold_seconds = float(self.config.get("hold_seconds", DEFAULT_HOLD_SECONDS))
        self.settle_seconds = float(self.config.get("settle_seconds", DEFAULT_SETTLE_SECONDS))

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

    @staticmethod
    def _analyze_leg(drive_samples):
        """Fits one leg's continuous pose samples: which raw axis is 'up', the swarm's
        measured travel speed/bearing in the other two, and the resulting
        HEADING_OFFSET_RAD candidates. Returns None if there's not enough data to trust
        a fit. Pure function (no self.*) so it's trivially unit-testable."""
        if len(drive_samples) < 3:
            return None

        t = np.array([s[0] for s in drive_samples])
        raw = np.array([[s[1], s[2], s[3]] for s in drive_samples])  # (n, 3): raw x,y,z
        raw_yaws = np.array([s[4] for s in drive_samples])

        slopes = np.zeros(3)
        r_squared = np.zeros(3)
        for axis in range(3):
            slope, intercept = np.polyfit(t, raw[:, axis], 1)
            pred = slope * t + intercept
            ss_res = np.sum((raw[:, axis] - pred) ** 2)
            ss_tot = np.sum((raw[:, axis] - raw[:, axis].mean()) ** 2)
            slopes[axis] = slope
            r_squared[axis] = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

        # Pick "up" by WORST fit quality (R^2), not smallest raw slope. Raw slope alone
        # is the wrong criterion: if a robot happens to drive mostly along one ground
        # axis with little component along the other, that other (real, ground-plane)
        # axis can have a smaller slope than noisy vertical jitter on the true up axis --
        # this is a confirmed real failure mode, not a hypothetical (a real run picked
        # the horizontal axis 0 as "up" over the true up axis 1 this way, corrupting
        # POSITION_AXES/HEADING_OFFSET_RAD/MOTOR_UNITS_PER_MPS all at once, since the
        # true up axis's jitter got included as part of the "measured travel speed").
        # R^2 doesn't have this problem: real motion (even slow, even shallow-angled)
        # fits a line far better than up/down jitter with no systematic trend does.
        up_axis = int(np.argmin(r_squared))
        position_axes = tuple(sorted(a for a in range(3) if a != up_axis))
        ax0, ax1 = position_axes

        travel_speed = math.hypot(slopes[ax0], slopes[ax1])
        travel_angle = math.atan2(slopes[ax1], slopes[ax0]) if travel_speed > 1e-6 else None

        mean_raw_yaw = float(np.mean(np.unwrap(raw_yaws)))
        yaw_std = float(np.std(raw_yaws))

        offset_pos = offset_neg = None
        if travel_angle is not None:
            expected_heading_sim = _wrap_to_pi(travel_angle - math.pi / 2.0)
            offset_pos = _wrap_to_pi(expected_heading_sim - mean_raw_yaw)
            offset_neg = _wrap_to_pi(expected_heading_sim + mean_raw_yaw)

        return {"position_axes": position_axes, "up_axis": up_axis, "slopes": slopes,
                "r_squared": r_squared, "travel_speed": travel_speed,
                "mean_raw_yaw": mean_raw_yaw, "yaw_std": yaw_std,
                "offset_pos": offset_pos, "offset_neg": offset_neg}

    async def run(self):
        print(f"[{self.self_hostname}] position/heading/speed calibration starting -- "
              f"targets={self.motor_targets}, hold={self.hold_seconds}s, "
              f"settle={self.settle_seconds}s. Give the robot a clear, straight, "
              f"obstacle-free runway -- total sweep time ~"
              f"{len(self.motor_targets) * (self.hold_seconds + self.settle_seconds):.0f}s.")

        print(f"[{self.self_hostname}] checking tracking before starting...")
        pre_samples = await self._collect_samples(PRE_DRIVE_SAMPLES * SAMPLE_INTERVAL_S, driving=False)
        if len(pre_samples) < 2:
            print(f"[{self.self_hostname}] not reliably tracked (got {len(pre_samples)} "
                  f"pose samples while stationary) -- aborting. Check OptiTrack/hostname_map "
                  f"before retrying.")
            await self.robot.stop()
            return
        print(f"[{self.self_hostname}] tracked OK ({len(pre_samples)} samples).")

        legs = []  # list of (target, analysis dict or None)
        for target in self.motor_targets:
            if not self.running:
                break
            while self.paused and self.running:
                await self.robot.stop()
                await asyncio.sleep(0.1)
            if not self.running:
                break

            print(f"[{self.self_hostname}] leg target={target}: driving for {self.hold_seconds}s...")
            await self.robot.drive(target, target)
            drive_samples = await self._collect_samples(self.hold_seconds, driving=True)
            await self.robot.stop()
            await asyncio.sleep(self.settle_seconds)  # let it fully stop before the next leg

            analysis = self._analyze_leg(drive_samples)
            if analysis is None:
                print(f"[{self.self_hostname}] leg target={target}: too few samples "
                      f"({len(drive_samples)}) -- skipping this leg.")
                legs.append((target, None))
                continue

            r2 = analysis["r_squared"]
            print(f"[{self.self_hostname}] leg target={target}: speed={analysis['travel_speed']:.4f} m/s, "
                  f"position_axes={analysis['position_axes']}, "
                  f"heading_offset(+sign)={analysis['offset_pos']:+.4f}" if analysis['offset_pos'] is not None
                  else f"[{self.self_hostname}] leg target={target}: didn't move enough to measure direction")
            print(f"    per-axis R^2 (fit quality): axis0={r2[0]:.3f} axis1={r2[1]:.3f} axis2={r2[2]:.3f}"
                  f"  (up_axis={analysis['up_axis']} should be the clear outlier -- low, "
                  f"well below the other two)")
            r2_sorted = np.sort(r2)
            confidence_margin = r2_sorted[1] - r2_sorted[0]  # worst vs. second-worst
            if confidence_margin < 0.2:
                print(f"    WARNING: up-axis choice is not confident (R^2 margin between "
                      f"worst and second-worst axis is only {confidence_margin:.3f}) -- the "
                      f"robot may not have driven straight/far enough, or two axes are "
                      f"similarly noisy. Don't trust POSITION_AXES from this leg alone; "
                      f"re-run with a longer/straighter runway.")
            legs.append((target, analysis))
            self._log("leg", target=target, position_axes=analysis["position_axes"],
                       up_axis=analysis["up_axis"], slopes_mps=analysis["slopes"],
                       r_squared=analysis["r_squared"], travel_speed_mps=analysis["travel_speed"],
                       mean_raw_yaw=analysis["mean_raw_yaw"], yaw_std=analysis["yaw_std"],
                       offset_pos=analysis["offset_pos"], offset_neg=analysis["offset_neg"])

        usable = [(t, a) for t, a in legs if a is not None and a["travel_speed"] > 1e-6]
        if not usable:
            print(f"[{self.self_hostname}] no usable legs -- check tracking and runway, then retry.")
            await self.robot.stop()
            return

        # MOTOR_UNITS_PER_MPS: least-squares fit through the origin, same formula
        # calibrate_speed_experiment.py uses, across every usable leg.
        sum_v2 = sum(a["travel_speed"] ** 2 for _, a in usable)
        sum_tv = sum(t * a["travel_speed"] for t, a in usable)
        recommended_units_per_mps = sum_tv / sum_v2 if sum_v2 > 0 else float("nan")

        # POSITION_AXES: should be identical every leg -- report the most common, flag disagreement.
        axes_votes = {}
        for _, a in usable:
            axes_votes[a["position_axes"]] = axes_votes.get(a["position_axes"], 0) + 1
        recommended_axes = max(axes_votes, key=axes_votes.get)
        axes_agree = len(axes_votes) == 1

        offsets_pos = [a["offset_pos"] for _, a in usable if a["offset_pos"] is not None]
        offsets_neg = [a["offset_neg"] for _, a in usable if a["offset_neg"] is not None]

        print(f"\n[{self.self_hostname}] === Final recommendations ({len(usable)}/{len(legs)} legs usable) ===")
        print(f"    POSITION_AXES = {recommended_axes}" +
              ("" if axes_agree else f"  -- WARNING: legs disagreed ({axes_votes}), inspect per-leg output above"))
        print(f"    MOTOR_UNITS_PER_MPS = {recommended_units_per_mps:.2f} "
              f"(currently {cfg.MOTOR_UNITS_PER_MPS:.2f} in controller_config.py)")
        if offsets_pos:
            mean_pos, std_pos = float(np.mean(offsets_pos)), float(np.std(offsets_pos))
            mean_neg, std_neg = float(np.mean(offsets_neg)), float(np.std(offsets_neg))
            print(f"    HEADING_OFFSET_RAD (ROTATION_SIGN NOT determined by this test -- "
                  f"see module docstring):")
            print(f"      if ROTATION_SIGN =  1.0: {mean_pos:+.4f} (std across legs: {std_pos:.4f})")
            print(f"      if ROTATION_SIGN = -1.0: {mean_neg:+.4f} (std across legs: {std_neg:.4f})")
            print(f"    A large std above means the legs disagreed -- trust the mean less, "
                  f"re-run with a longer runway/hold_seconds.")
        else:
            print(f"    HEADING_OFFSET_RAD: no leg moved enough to measure a travel direction.")

        self._log("summary", position_axes=recommended_axes, axes_agree=axes_agree,
                   recommended_units_per_mps=recommended_units_per_mps,
                   offset_pos=float(np.mean(offsets_pos)) if offsets_pos else None,
                   offset_neg=float(np.mean(offsets_neg)) if offsets_neg else None,
                   n_legs_usable=len(usable), n_legs_total=len(legs))

        await self.robot.stop()

    def _log(self, phase, target=None, position_axes=None, up_axis=None, slopes_mps=None,
              r_squared=None, travel_speed_mps=None, mean_raw_yaw=None, yaw_std=None,
              offset_pos=None, offset_neg=None, axes_agree=None, recommended_units_per_mps=None,
              n_legs_usable=None, n_legs_total=None):
        """Every call -- 'leg' (once per motor_targets entry) or 'summary' (once at the
        end) -- MUST pass the exact same set of keys to logger.log(), even as None for
        whatever a given phase doesn't have: SessionLogger.log() infers its CSV header
        from the FIRST call's keys and indexes every later row against that same fixed
        header, so a later call with a different key set raises a KeyError instead of
        just writing blank cells. Mirrors calibrate_speed_experiment.py's _log()."""
        if not self.logger:
            return
        self.logger.log(
            state={"phase": phase,
                   "position_axes": list(position_axes) if position_axes is not None else None,
                   "up_axis": up_axis,
                   "slopes_mps": slopes_mps.tolist() if slopes_mps is not None else None,
                   "r_squared": r_squared.tolist() if r_squared is not None else None,
                   "travel_speed_mps": travel_speed_mps, "mean_raw_yaw": mean_raw_yaw,
                   "yaw_std": yaw_std,
                   "heading_offset_if_rotation_sign_positive": offset_pos,
                   "heading_offset_if_rotation_sign_negative": offset_neg,
                   "axes_agree": axes_agree, "recommended_units_per_mps": recommended_units_per_mps,
                   "n_legs_usable": n_legs_usable, "n_legs_total": n_legs_total},
            command={"motor_target": target},
        )

    async def pause(self):
        self.paused = True

    async def resume(self):
        self.paused = False

    async def stop(self):
        self.running = False
