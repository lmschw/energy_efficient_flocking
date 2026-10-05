"""4-quadrant range/bearing sensing -- ported verbatim (only the config import changed)
from the verified/tested experiment/sensor_model.py in the main simulation package, so
hardware and simulation compute sensing identically given the same agents array. See
that file for the derivation notes and the MATLAB reference (getsensordata() in
simulation_free_global_mod_2.m).

On real hardware, "agents" is built each control tick from OptiTrack poses (see
pose_utils.poses_to_agents()) rather than from simulated state -- everything downstream
of that array is unchanged.
"""
import numpy as np

import controller_config as cfg


def get_sensor_data(agents, sensing_radius=None):
    """agents: (n_agents, 4) array of [x, y, heading, battery].

    Returns a (10, n_agents) array per agent: [front_dist, front_bearing, back_dist,
    back_bearing, right_dist, right_bearing, left_dist, left_bearing, own_battery,
    own_heading], each rescaled to roughly [-1, 1] (front/back/right/left here name the
    robot's own body-relative quadrants, not compass directions).
    """
    if sensing_radius is None:
        sensing_radius = cfg.SENSING_RADIUS
    n_agents = agents.shape[0]

    x, y, heading = agents[:, 0], agents[:, 1], agents[:, 2]
    dx = x[None, :] - x[:, None]   # dx[i, j] = x_j - x_i
    dy = y[None, :] - y[:, None]
    angle = np.arctan2(dy, dx)
    rel_angle = angle - np.pi / 2.0 - heading[:, None]
    rel_angle = (rel_angle + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-pi, pi]
    distance = np.hypot(dx, dy)

    not_self = ~np.eye(n_agents, dtype=bool)
    in_range = (distance < sensing_radius) & not_self

    right_mask = in_range & (rel_angle >= -3 * np.pi / 4) & (rel_angle <= -np.pi / 4)
    front_mask = in_range & (rel_angle >= -np.pi / 4) & (rel_angle <= np.pi / 4)
    back_mask = in_range & ((rel_angle <= -3 * np.pi / 4) | (rel_angle >= 3 * np.pi / 4))
    left_mask = in_range & (rel_angle >= np.pi / 4) & (rel_angle <= 3 * np.pi / 4)

    right_bearing_all = rel_angle + 3 * np.pi / 4
    front_bearing_all = rel_angle + np.pi / 4
    back_bearing_all = np.where(rel_angle <= 0, rel_angle + 5 * np.pi / 4, rel_angle - 3 * np.pi / 4)
    left_bearing_all = rel_angle - np.pi / 4

    def _nearest(mask, bearing_all):
        masked_dist = np.where(mask, distance, np.inf)
        nearest_idx = np.argmin(masked_dist, axis=1)
        found = np.isfinite(masked_dist[np.arange(n_agents), nearest_idx])
        dist_out = np.where(found, masked_dist[np.arange(n_agents), nearest_idx], sensing_radius)
        bearing_out = np.where(found, bearing_all[np.arange(n_agents), nearest_idx], 0.0)
        return dist_out, bearing_out

    front_d, front_b = _nearest(front_mask, front_bearing_all)
    back_d, back_b = _nearest(back_mask, back_bearing_all)
    right_d, right_b = _nearest(right_mask, right_bearing_all)
    left_d, left_b = _nearest(left_mask, left_bearing_all)

    inputs = np.stack([
        front_d * 2.0 / sensing_radius - 1.0, front_b * 4.0 / np.pi - 1.0,
        back_d * 2.0 / sensing_radius - 1.0, back_b * 4.0 / np.pi - 1.0,
        right_d * 2.0 / sensing_radius - 1.0, right_b * 4.0 / np.pi - 1.0,
        left_d * 2.0 / sensing_radius - 1.0, left_b * 4.0 / np.pi - 1.0,
        agents[:, 3] / 50.0 - 1.0,
        agents[:, 2] / np.pi,
    ], axis=0)  # (10, n_agents)
    return inputs
