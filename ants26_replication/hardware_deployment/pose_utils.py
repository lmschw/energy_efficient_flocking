"""Converts OptiTrack poses (position xyz meters, orientation quaternion xyzw -- the
representation swarm_platform.tracking.pose.Pose and Robot.get_all_global_poses() use)
into the (n_agents, 4) [x, y, heading, battery] array sensor_model.get_sensor_data() and
hebbian_controller expect. This is the ONLY translation layer between real motion-capture
poses and the simulation's agent representation -- the sensing/control math itself
(sensor_model.py, hebbian_controller.py) is identical between sim and hardware.
"""
from collections import namedtuple

import numpy as np

import controller_config as cfg

# Duck-type compatible with swarm_platform.tracking.pose.Pose (same field names/shapes).
# Defined locally so this package has no hard dependency on swarm_platform being
# installed -- useful for local_test_harness.py and for development off-hardware. When
# actually deployed, real Pose objects from the platform work here unchanged.
Pose = namedtuple("Pose", ["position", "orientation"])


def quaternion_to_yaw(qx, qy, qz, qw):
    """Standard Z-up yaw extraction from a quaternion. If your Motive calibration is
    Y-up (a common default), you likely need to remap axes in POSITION_AXES *and*
    permute which quaternion components are passed in here as (qx, qy, qz) so that the
    "up" component lines up with this formula's z -- verify with
    diagnostics/print_poses.py rather than assuming this is correct for your rig."""
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


_warned_missing_offset_hosts = set()  # module-level: print the fallback warning once per
                                       # hostname, not once per tick (this runs every
                                       # CONTROL_TICK_SECONDS, so a per-call warning would
                                       # flood the log within seconds).

_UP_AXIS_INDEX = ({0, 1, 2} - set(cfg.POSITION_AXES)).pop()
# The one raw position component POSITION_AXES doesn't use for the ground plane -- see
# cfg.UP_AXIS_PLAUSIBLE_RANGE_M's comment for why this is worth checking at all: a rigid
# body can solve against the wrong markers and keep reporting a well-formed but
# physically implausible pose with no error from Motive/NatNet itself.


def _check_up_axis_plausible(host, pose):
    """Prints a warning (does not alter the pose or treat the robot as untracked -- see
    cfg.UP_AXIS_PLAUSIBLE_RANGE_M's comment) if this robot's raw up-axis reading falls
    outside the configured plausible floor-height band. Deliberately warns every
    occurrence rather than once-per-host (unlike _heading_offset_for()'s missing-config
    warning): this is a live, potentially transient tracking-quality signal (a robot can
    become briefly mistracked and then recover), not a static one-time config gap, so
    seeing it stop is as informative as seeing it start."""
    lo, hi = cfg.UP_AXIS_PLAUSIBLE_RANGE_M
    up_val = pose.position[_UP_AXIS_INDEX]
    if not (lo <= up_val <= hi):
        print(f"[pose_utils] WARNING: '{host}' up-axis (raw component {_UP_AXIS_INDEX}) "
              f"reading {up_val:.3f}m is outside the plausible floor-height band "
              f"[{lo}, {hi}]m -- this robot's rigid body may be solving against the "
              f"wrong markers (stray reflection, unstable marker set, ceiling fixture) "
              f"rather than tracking the real robot. Position/heading derived from this "
              f"pose is likely garbage this tick.")


def _heading_offset_for(host):
    """cfg.HEADING_OFFSET_RAD is per-robot (a dict), not one shared constant -- real
    robots have disagreed by more than measurement noise would explain (see that
    constant's comment in controller_config.py), most likely because their rigid bodies
    weren't defined with the same 'front' convention in Motive. Falls back to
    HEADING_OFFSET_RAD_DEFAULT (once-warned, not a crash) for any hostname not listed --
    e.g. a new robot added to the fleet before its own calibration."""
    if host in cfg.HEADING_OFFSET_RAD:
        return cfg.HEADING_OFFSET_RAD[host]
    if host not in _warned_missing_offset_hosts:
        _warned_missing_offset_hosts.add(host)
        print(f"[pose_utils] WARNING: no HEADING_OFFSET_RAD entry for '{host}' -- falling "
              f"back to HEADING_OFFSET_RAD_DEFAULT ({cfg.HEADING_OFFSET_RAD_DEFAULT}). "
              f"Calibrate this robot individually and add it to controller_config.py's "
              f"HEADING_OFFSET_RAD dict.")
    return cfg.HEADING_OFFSET_RAD_DEFAULT


def poses_to_agents(poses, hostnames, self_hostname):
    """poses: dict hostname -> Pose, e.g. from `await robot.get_all_global_poses()`.
    hostnames: the full ordered list of every robot participating in this run -- must
    be identical (same list, same order) across every Pi's config, so each robot's own
    quadrant-sensing math is well-defined the same way everywhere.

    Returns (agents, self_index): agents is (len(hostnames), 4) [x, y, heading, battery];
    self_index is hostnames.index(self_hostname), i.e. which row is "this robot" (needed
    by hebbian_swarm_experiment.py to pick out this robot's own sensor row afterward).

    A robot with no current pose (not yet tracked, or outside the mocap volume) is
    placed far away rather than at (0, 0) -- so it reads as "no neighbor there" to
    sensor_model's range cutoff instead of being mistaken for a real, very-close robot.
    """
    ax0, ax1 = cfg.POSITION_AXES
    agents = np.zeros((len(hostnames), 4))
    self_index = hostnames.index(self_hostname)
    for i, host in enumerate(hostnames):
        pose = poses.get(host)
        if pose is None:
            agents[i] = [1e4, 1e4, 0.0, cfg.BATTERY_SENSOR_PLACEHOLDER]
            continue
        _check_up_axis_plausible(host, pose)
        x, y = pose.position[ax0], pose.position[ax1]
        yaw = quaternion_to_yaw(*pose.orientation) * cfg.ROTATION_SIGN + _heading_offset_for(host)
        agents[i] = [x, y, _wrap_to_pi(yaw), cfg.BATTERY_SENSOR_PLACEHOLDER]
    return agents, self_index
