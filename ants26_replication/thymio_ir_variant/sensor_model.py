"""7-value Thymio II IR proximity sensing for the Hebbian ABCD controller.

REPLACES the paper's idealized 4-quadrant range/bearing sensor (see ../experiment/
sensor_model.py) with a simulation of the real Thymio II's onboard prox.horizontal
array: 7 short-range (~12cm) raw IR reflectance readings at fixed angles on the robot
body, that cannot distinguish a neighbor from a wall, cannot report bearing beyond
"which of 7 fixed-angle sensors fired," and cannot identify which neighbor (if any)
triggered a reading. See config.py's THYMIO_IR_* constants for exact sourcing of the 7
angles/range/aperture, and ../hardware_deployment/README.md's "Sensing: OptiTrack
substitutes for onboard range/bearing" section for why this gap existed in the first
place -- the idealized sensor was never something a real, non-motion-captured Thymio
swarm could compute (robot.proximity_horizontal() is never called anywhere in this
project's hardware-deployment code).

Each of the 7 sensors independently detects the NEAREST reflecting surface -- wall OR
another agent, whichever is closer -- within its narrow detection cone
(THYMIO_IR_HALF_APERTURE) and range (THYMIO_IR_RANGE), exactly as a real IR reflectance
sensor would: it has no way to know or care what it's reflecting off, so wall-avoidance
and neighbor-avoidance share the same 7 channels here rather than needing a separate
dedicated wall input (contrast wall_sensor_variant/, which adds 4 idealized wall-only
inputs on top of the idealized neighbor sensor). Output is a [0, 1] intensity per
sensor (0 = nothing within range, matching the real sensor's raw reading of 0; 1 =
touching), not a signed [-1, 1] value like the other sensor inputs -- there's no
natural "negative" reading for a reflectance intensity.
"""
import numpy as np

try:
    import config
except ModuleNotFoundError:
    from . import config


def _wall_ray_distances(x, y, global_dir):
    """Distance from (x, y) to the nearest of the 3 simulated-arena walls along the ray
    pointing in global_dir (a global angle in radians, one per agent), or +inf if the
    ray points away from every wall. Only 3 walls exist -- X_RANGE's upper bound and
    Y_RANGE's upper/lower bounds -- matching every other wall computation in this
    project (e.g. wall_sensor_variant's _wall_quadrant_distances); there's no lower X
    wall since agents migrate indefinitely in -x by design.

    Unlike wall_sensor_variant's quadrant-based approximation (a perpendicular-offset
    shortcut, valid only for a broad quadrant treated as facing the wall dead-on), this
    is genuine ray-to-line intersection -- appropriate here since each IR sensor is a
    narrow, specific ray, not a broad quadrant.
    """
    cos_d, sin_d = np.cos(global_dir), np.sin(global_dir)
    wall_x_upper = config.X_RANGE[1] - config.ROBOT_RAD
    wall_y_upper = config.Y_RANGE[1] - config.ROBOT_RAD
    wall_y_lower = config.Y_RANGE[0] + config.ROBOT_RAD

    dist = np.full(x.shape, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wall_x_upper - x) / cos_d
        valid = (np.abs(cos_d) > 1e-9) & (t > 0)
        dist = np.where(valid, np.minimum(dist, t), dist)

        t = (wall_y_upper - y) / sin_d
        valid = (np.abs(sin_d) > 1e-9) & (t > 0)
        dist = np.where(valid, np.minimum(dist, t), dist)

        t = (wall_y_lower - y) / sin_d
        valid = (np.abs(sin_d) > 1e-9) & (t > 0)
        dist = np.where(valid, np.minimum(dist, t), dist)

    return dist


def get_sensor_data(agents, ir_range=None):
    """agents: (n_agents, 4) array of [x, y, heading, battery].

    Returns a (9, n_agents) array: 7 IR intensities (config.THYMIO_IR_ANGLES order,
    each in [0, 1]) + own battery + own heading, matching this variant's
    config.HEBBIAN_N_INPUTS=9.
    """
    if ir_range is None:
        ir_range = config.THYMIO_IR_RANGE
    n_agents = agents.shape[0]
    x, y, heading = agents[:, 0], agents[:, 1], agents[:, 2]

    # Neighbor bearing/distance -- same rel_angle convention as ../experiment/
    # sensor_model.py: rel_angle=0 means "straight ahead" in the robot's own frame,
    # derived from this project's established heading convention (forward-facing
    # global direction = heading + pi/2; see simulation_hebbian.py's _move()).
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    global_bearing = np.arctan2(dy, dx)
    rel_angle = global_bearing - np.pi / 2.0 - heading[:, None]
    rel_angle = (rel_angle + np.pi) % (2.0 * np.pi) - np.pi
    center_distance = np.hypot(dx, dy)
    # THYMIO_IR_RANGE is the sensor's range from its own physical mounting position on
    # the robot's body -- i.e. gap between surfaces -- not center-to-center. Using
    # center_distance directly here would make IR max range numerically coincide with
    # the collision threshold (both center_distance ~= 2*ROBOT_RAD + a small slack,
    # see simulation_hebbian.py's min_dist), so a sensor would only ever fire at the
    # exact moment of collision, never before -- caught by direct comparison against
    # config.COLLISION_MIN_DIST_SLACK. Subtracting 2*ROBOT_RAD converts to the gap
    # between the two robots' surfaces, giving a real pre-collision detection window
    # (up to THYMIO_IR_RANGE of surface gap, vs. collision at COLLISION_MIN_DIST_SLACK).
    surface_gap = np.maximum(center_distance - 2.0 * config.ROBOT_RAD, 0.0)
    not_self = ~np.eye(n_agents, dtype=bool)

    half_ap = config.THYMIO_IR_HALF_APERTURE
    intensities = np.zeros((len(config.THYMIO_IR_ANGLES), n_agents))

    for s, sensor_angle in enumerate(config.THYMIO_IR_ANGLES):
        angle_diff = (rel_angle - sensor_angle + np.pi) % (2.0 * np.pi) - np.pi
        in_cone = not_self & (np.abs(angle_diff) <= half_ap) & (surface_gap <= ir_range)
        neighbor_dist = np.where(in_cone, surface_gap, np.inf).min(axis=1)

        global_dir = heading + np.pi / 2.0 + sensor_angle
        wall_dist = _wall_ray_distances(x, y, global_dir)
        wall_dist = np.where(wall_dist <= ir_range, wall_dist, np.inf)

        nearest = np.minimum(neighbor_dist, wall_dist)
        intensity = np.where(np.isfinite(nearest), 1.0 - nearest / ir_range, 0.0)
        intensities[s] = np.clip(intensity, 0.0, 1.0)

    inputs = np.concatenate([
        intensities,
        (agents[:, 3] / 50.0 - 1.0)[None, :],
        (agents[:, 2] / np.pi)[None, :],
    ], axis=0)  # (9, n_agents)
    return inputs
