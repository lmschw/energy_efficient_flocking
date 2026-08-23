"""4-quadrant range/bearing sensing for the Hebbian ABCD controller, EXTENDED beyond
the paper with a 4-quadrant wall-proximity input (see config.HEBBIAN_N_INPUTS's
comment for why: the original 10-input architecture has no sensory signal for the
simulation boundary at all, which measurably kept a collapsed-curriculum variant of
this controller from learning to avoid it).

Neighbor-sensing part is a 1:1 port of getsensordata() in simulation_free_global_mod_2.m,
with one deliberate fix: the MATLAB source never actually masks out neighbors beyond the
sensing radius R (it only uses R as the "no neighbor found" default), so a distant
neighbor could still register as the "nearest in quadrant" and produce a normalized
distance outside [-1, 1]. The paper is explicit that "every robot can perceive
neighbours only when they are within a sensing radius R" (Section 2.1), so this port
enforces that cutoff.
"""
import numpy as np

try:
    import config
except ModuleNotFoundError:
    from . import config


def _wall_quadrant_distances(agents, heading, sensing_radius):
    """Egocentric distance to the nearest simulation-boundary wall in each of the same
    4 quadrants neighbor-sensing uses below -- distance-only (no bearing), since a wall
    has no identity to orient toward the way a neighbor does. Each wall is an
    axis-aligned line, so the nearest point on it from any agent is always exactly
    perpendicular, in a FIXED global direction (unlike a neighbor's bearing, which
    depends on the neighbor's specific position) -- that fixed direction is converted
    to the robot's egocentric frame with the identical rel_angle formula used below.

    Only the boundaries move() actually enforces: X_RANGE's upper bound (soft
    wind-tracking window edge) and Y_RANGE's upper/lower bounds (hard clamped). There's
    no lower X wall -- agents migrate indefinitely in -x by design.
    """
    n_agents = agents.shape[0]
    x, y = agents[:, 0], agents[:, 1]

    wall_x_upper = config.X_RANGE[1] - config.ROBOT_RAD
    wall_y_upper = config.Y_RANGE[1] - config.ROBOT_RAD
    wall_y_lower = config.Y_RANGE[0] + config.ROBOT_RAD

    # (perpendicular distance, fixed global bearing) per wall.
    walls = [
        (wall_x_upper - x, 0.0),           # nearest point is straight in +x
        (wall_y_upper - y, np.pi / 2.0),   # nearest point is straight in +y
        (y - wall_y_lower, -np.pi / 2.0),  # nearest point is straight in -y
    ]

    quadrant_dist = np.full((4, n_agents), sensing_radius)  # front, back, right, left
    for raw_dist, global_angle in walls:
        dist = np.maximum(raw_dist, 0.0)  # defensive -- shouldn't go negative given move()'s clamp
        rel_angle = global_angle - np.pi / 2.0 - heading
        rel_angle = (rel_angle + np.pi) % (2.0 * np.pi) - np.pi
        in_range = dist < sensing_radius

        front = in_range & (rel_angle >= -np.pi / 4) & (rel_angle <= np.pi / 4)
        back = in_range & ((rel_angle <= -3 * np.pi / 4) | (rel_angle >= 3 * np.pi / 4))
        right = in_range & (rel_angle >= -3 * np.pi / 4) & (rel_angle <= -np.pi / 4)
        left = in_range & (rel_angle >= np.pi / 4) & (rel_angle <= 3 * np.pi / 4)

        quadrant_dist[0] = np.where(front, np.minimum(quadrant_dist[0], dist), quadrant_dist[0])
        quadrant_dist[1] = np.where(back, np.minimum(quadrant_dist[1], dist), quadrant_dist[1])
        quadrant_dist[2] = np.where(right, np.minimum(quadrant_dist[2], dist), quadrant_dist[2])
        quadrant_dist[3] = np.where(left, np.minimum(quadrant_dist[3], dist), quadrant_dist[3])

    return quadrant_dist  # (4, n_agents) = [front, back, right, left]


def get_sensor_data(agents, sensing_radius=None):
    """agents: (n_agents, 4) array of [x, y, heading, battery].

    Returns a (14, n_agents) array per agent: [front_dist, front_bearing, back_dist,
    back_bearing, right_dist, right_bearing, left_dist, left_bearing] (neighbor
    sensing, indices 0-7), own_battery (index 8), own_heading (index 9) -- SAME indices
    as the original 10-input architecture -- then [wall_front_dist, wall_back_dist,
    wall_right_dist, wall_left_dist] (indices 10-13, the +4 extension over the paper's
    original 10 inputs -- see module docstring), each rescaled to roughly [-1, 1]
    (front/back/right/left here name the robot's own body-relative quadrants, not
    compass directions).
    """
    if sensing_radius is None:
        sensing_radius = config.HEBBIAN_SENSING_RADIUS
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

    wall_front, wall_back, wall_right, wall_left = _wall_quadrant_distances(agents, heading, sensing_radius)

    inputs = np.stack([
        front_d * 2.0 / sensing_radius - 1.0, front_b * 4.0 / np.pi - 1.0,
        back_d * 2.0 / sensing_radius - 1.0, back_b * 4.0 / np.pi - 1.0,
        right_d * 2.0 / sensing_radius - 1.0, right_b * 4.0 / np.pi - 1.0,
        left_d * 2.0 / sensing_radius - 1.0, left_b * 4.0 / np.pi - 1.0,
        agents[:, 3] / 50.0 - 1.0,   # index 8, own battery -- SAME index as the original
        agents[:, 2] / np.pi,        # index 9, own heading -- 10-input architecture, so
        wall_front * 2.0 / sensing_radius - 1.0,  # anything hardcoding "battery is at
        wall_back * 2.0 / sensing_radius - 1.0,   # index 8" (e.g. simulation_hebbian.py's
        wall_right * 2.0 / sensing_radius - 1.0,  # battery-sensor ablation) still works
        wall_left * 2.0 / sensing_radius - 1.0,   # unchanged; the +4 wall inputs are
    ], axis=0)  # (14, n_agents)                   # purely appended at the end (10-13).
    return inputs
