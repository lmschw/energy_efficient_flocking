"""Lighthouse (CrazyThymio) pose -> (n_agents, 4) [x, y, heading, battery] array for
sensor_model.get_sensor_data() / hebbian_controller. Standalone counterpart of
ants26_replication/hardware_deployment/pose_utils.py (OptiTrack); the two share no code.

Differences to the OptiTrack version, on purpose:
  * ground plane is (x, y), z is up -- no POSITION_AXES / up-axis juggling;
  * no up-axis-outlier or stale-pose detectors: freshness is decided upstream in
    lighthouse_robot.py (own pose older than POSE_TIMEOUT_S -> None; neighbor not heard for 1 s
    -> firmware drops it). Neighbor positions are 1 mm-quantized radio packets that
    legitimately repeat bit-for-bit while a robot stands still, which would false-trigger a
    stale-feed check;
  * heading = Lighthouse yaw (CCW from +x) - pi/2 + per-robot board offset, because the
    simulation's heading 0 faces +y.
"""
from collections import namedtuple

import numpy as np

import controller_config as cfg

# Duck-type compatible with swarm_platform.tracking.pose.Pose (position xyz, orientation xyzw).
Pose = namedtuple("Pose", ["position", "orientation"])


def quaternion_to_yaw(qx, qy, qz, qw):
    """Z-up yaw from a quaternion."""
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def poses_to_agents(poses, hostnames, self_hostname):
    """poses: dict hostname -> Pose. hostnames: the full ordered list of robots, identical on
    every Pi. Returns (agents, self_index). A robot without a pose is placed at the (1e4, 1e4)
    sentinel so it reads as "no neighbor there" rather than as a very close robot."""
    agents = np.zeros((len(hostnames), 4))
    self_index = hostnames.index(self_hostname)
    for i, host in enumerate(hostnames):
        pose = poses.get(host)
        if pose is None:
            agents[i] = [1e4, 1e4, 0.0, cfg.BATTERY_SENSOR_PLACEHOLDER]
            continue
        x, y = pose.position[0], pose.position[1]
        offset = cfg.LIGHTHOUSE_HEADING_OFFSET_RAD.get(host, cfg.LIGHTHOUSE_HEADING_OFFSET_RAD_DEFAULT)
        yaw = quaternion_to_yaw(*pose.orientation) * cfg.ROTATION_SIGN - np.pi / 2.0 + offset
        agents[i] = [x, y, _wrap_to_pi(yaw), cfg.BATTERY_SENSOR_PLACEHOLDER]
    return agents, self_index
