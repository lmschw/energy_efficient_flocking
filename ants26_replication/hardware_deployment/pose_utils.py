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
# cfg.UP_AXIS_OUTLIER_THRESHOLD_M's comment for why this is worth checking at all: a
# rigid body can solve against the wrong markers and keep reporting a well-formed but
# physically implausible pose with no error from Motive/NatNet itself.


def _find_up_axis_outliers(poses, hostnames):
    """Returns the set of hostnames whose raw up-axis reading is a clear outlier
    relative to the OTHER currently-tracked robots this tick -- NOT compared against
    any fixed absolute band. An earlier version of this check used a hardcoded
    plausible-height band (e.g. -0.5 to 0.5m); that broke on the very next session,
    because the ground-plane/origin calibration is NOT guaranteed to match between
    sessions (documented, then immediately violated by hardcoding numbers from one
    session as if they were universal) -- confirmed the hard way: an entire real trial
    had every robot's genuinely correct up-axis reading fall outside that band, so
    poses_to_agents() silently treated every robot as permanently self-blind for the
    whole run, reproducing the exact "no neighbors, spin in place" symptom this was
    meant to prevent -- a real bug in the earlier fix, not a hardware issue. Comparing
    against the CURRENT session's own peer robots instead needs no a-priori knowledge
    of what "correct" looks like this session; confirmed session-to-session that
    legitimate cross-robot spread stays under ~0.5m while a genuine bad track (a stray
    reflective object at ~2m up stealing 6 robots' identities in turn; a persistently
    mistracked robot reading ~1.2m+ off its peers) is comfortably larger --
    UP_AXIS_OUTLIER_THRESHOLD_M sits between those two regimes. Needs >=2 real poses
    this tick to have any peer to compare against; with 0 or 1, returns no outliers
    (nothing to detect an outlier against)."""
    up_vals = {host: poses[host].position[_UP_AXIS_INDEX]
               for host in hostnames if poses.get(host) is not None}
    if len(up_vals) < 2:
        return set()
    median = float(np.median(list(up_vals.values())))
    threshold = cfg.UP_AXIS_OUTLIER_THRESHOLD_M
    outliers = set()
    for host, up_val in up_vals.items():
        deviation = abs(up_val - median)
        if deviation > threshold:
            outliers.add(host)
            print(f"[pose_utils] WARNING: '{host}' up-axis (raw component "
                  f"{_UP_AXIS_INDEX}) reading {up_val:.3f}m deviates {deviation:.3f}m "
                  f"from this tick's peer median ({median:.3f}m, {len(up_vals)} robots) "
                  f"-- past UP_AXIS_OUTLIER_THRESHOLD_M ({threshold}m). This robot's "
                  f"rigid body may be solving against the wrong markers (stray "
                  f"reflection, unstable marker set, ceiling fixture) rather than "
                  f"tracking the real robot. Treating '{host}' as UNTRACKED this tick "
                  f"instead of acting on this pose.")
    return outliers


_last_raw_position = {}  # hostname -> last seen raw (x, y, z) tuple, across ticks
_stale_streak = {}       # hostname -> count of consecutive ticks with that same value


def _find_stale_poses(poses, hostnames):
    """Returns the set of hostnames whose raw position has repeated bit-for-bit for
    STALE_POSE_TICK_THRESHOLD or more consecutive ticks -- see that constant's comment
    in controller_config.py for the real trial data (aggregated_1905.csv) that motivated
    this: a robot's tracking can silently freeze (drop out of NatNet's frame stream
    without ever being reported as untracked) while the daemon keeps re-serving its last
    known pose as if it were current. Real marker noise essentially never reproduces the
    exact same float across many consecutive polls, so an exact repeat streak this long
    is treated as a frozen feed, not a genuinely stationary robot. Maintains per-hostname
    state across calls (one call per control tick), unlike _find_up_axis_outliers which
    only looks within a single tick."""
    stale = set()
    for host in hostnames:
        pose = poses.get(host)
        if pose is None:
            _last_raw_position.pop(host, None)
            _stale_streak.pop(host, None)
            continue
        raw = tuple(pose.position)
        if _last_raw_position.get(host) == raw:
            _stale_streak[host] = _stale_streak.get(host, 1) + 1
        else:
            _stale_streak[host] = 1
        _last_raw_position[host] = raw
        if _stale_streak[host] >= cfg.STALE_POSE_TICK_THRESHOLD:
            stale.add(host)
            print(f"[pose_utils] WARNING: '{host}' raw position {raw} has been "
                  f"bit-for-bit identical for {_stale_streak[host]} consecutive ticks "
                  f"(>= STALE_POSE_TICK_THRESHOLD={cfg.STALE_POSE_TICK_THRESHOLD}). "
                  f"Tracking for this robot has likely frozen (dropped out of NatNet's "
                  f"frame stream without being reported as untracked) rather than the "
                  f"robot genuinely holding still. Treating '{host}' as UNTRACKED this "
                  f"tick instead of acting on this stale pose.")
    return stale


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

    A robot with no current pose (not yet tracked, outside the mocap volume, whose
    reading was just flagged as an up-axis outlier against this tick's other tracked
    robots -- see _find_up_axis_outliers() -- or whose raw position has frozen for
    several consecutive ticks -- see _find_stale_poses()) is placed far away rather than
    at (0, 0) -- so it reads as "no neighbor there" to sensor_model's range cutoff
    instead of being mistaken for a real, very-close robot.
    """
    ax0, ax1 = cfg.POSITION_AXES
    agents = np.zeros((len(hostnames), 4))
    self_index = hostnames.index(self_hostname)
    outlier_hosts = _find_up_axis_outliers(poses, hostnames) | _find_stale_poses(poses, hostnames)
    for i, host in enumerate(hostnames):
        pose = poses.get(host)
        if pose is None or host in outlier_hosts:
            agents[i] = [1e4, 1e4, 0.0, cfg.BATTERY_SENSOR_PLACEHOLDER]
            continue
        x, y = pose.position[ax0], pose.position[ax1]
        yaw = quaternion_to_yaw(*pose.orientation) * cfg.ROTATION_SIGN + _heading_offset_for(host)
        agents[i] = [x, y, _wrap_to_pi(yaw), cfg.BATTERY_SENSOR_PLACEHOLDER]
    return agents, self_index
