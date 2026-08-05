"""Runs the paper's Table 3 "standard collective motion baseline" -- the cluster-4
comparison point in Fig. 5a -- for analyze_hebbian_results.py.

A trimmed, standalone copy of initial_implementation/experiment/
simulation_free_global_mod_2_LJ.py's LJ control law (_flocking_velocity_command/move)
and fitness, duplicated here rather than imported across the project boundary. Only
what analyze_hebbian_results.py's baseline comparison actually needs: the numpy
(kinematic) backend, no video rendering, no PyBullet, no CMA-ES training. See that file
for the full original (video output, PyBullet rigid-body backend, LJ genome training).
"""
import numpy as np

try:
    import config
    from wind_physics import wrap_to_pi, RayTraceCircularRobots, dragforce, batterydrainage, _spawn_agents
except ModuleNotFoundError:
    from . import config
    from .wind_physics import wrap_to_pi, RayTraceCircularRobots, dragforce, batterydrainage, _spawn_agents


def _collision_and_window_bookkeeping(agents, n_agents, min_dist, walls, collision_counter):
    """Counts near-collisions/wall-proximity events and computes the wind-tracking
    window. Does not mutate agent positions."""
    agents_xy = agents[:, 0:2]
    D = np.linalg.norm(agents_xy[:, None, :] - agents_xy[None, :, :], axis=-1)
    close_agents = (D < min_dist) & (~np.eye(n_agents, dtype=bool))
    collision_counter += np.count_nonzero(np.triu(close_agents, k=1))

    wall_margin = config.ROBOT_RAD * config.WALL_MARGIN_FACTOR
    wall_hits_step = np.sum((agents[:, 0] > walls[1] - wall_margin) |
                            (agents[:, 1] > walls[2] - wall_margin) |
                            (agents[:, 1] < walls[3] + wall_margin))
    collision_counter += config.WALL_COLLISION_WEIGHT * wall_hits_step

    min_x = np.min(agents[:, 0])
    max_x = min(np.max(agents[:, 0]), min_x + config.WIND_TRACKING_MAX_SPAN)
    window_width = config.WIND_TRACKING_WINDOW_WIDTH
    xRange = [min_x - (window_width - (max_x - min_x)) / 2.0, max_x + (window_width - (max_x - min_x)) / 2.0]
    return xRange, collision_counter, max_x


def move(agents, vel, dt, n_agents, min_dist, walls, collision_counter):
    vel_actual = np.zeros((n_agents, 3))
    vel_actual[:, 0:2] = vel
    vel_actual[:, 2] = agents[:, 2]

    theta = agents[:, 2]
    dx = -vel[:, 0] * dt * np.sin(theta)
    dy = vel[:, 0] * dt * np.cos(theta)

    agents_old = agents.copy()
    agents[:, 0] += dx
    agents[:, 1] += dy
    agents[:, 2] = wrap_to_pi(agents[:, 2] + vel[:, 1] * dt)

    xRange, collision_counter, max_x = _collision_and_window_bookkeeping(
        agents, n_agents, min_dist, walls, collision_counter)

    agents[:, 0] = np.minimum(agents[:, 0], max_x)
    agents[:, 1] = np.minimum(agents[:, 1], walls[2])
    agents[:, 1] = np.maximum(agents[:, 1], walls[3])

    x_old, y_old = agents_old[:, 0], agents_old[:, 1]
    x_new, y_new = agents[:, 0], agents[:, 1]
    dist = np.sqrt((x_old - x_new) ** 2 + (y_old - y_new) ** 2)

    vel_actual[:, 2] = np.arctan2((y_new - y_old), (x_new - x_old)) - np.pi / 2.0
    vel_actual[:, 0] = dist / dt
    vel_actual[np.isnan(vel_actual[:, 2]), 2] = 0.0

    return vel_actual, agents, xRange, collision_counter


def _resolve_rules(rules):
    defaults = config.DEFAULT_RULES
    return tuple(rules[key] if rules else defaults[key]
                 for key in ("r0", "epsilon", "k_align", "k_goal", "K1", "K2", "U"))


def _flocking_velocity_command(agents, n_agents, r0, epsilon, k_align, k_goal, K1, K2, U, r_cut, r_min, R_align):
    """LJ spacing + heading alignment + goal-pull force, mapped to a (linear speed,
    angular rate) command."""
    sigma = r0 / np.sqrt(2.0)
    X = agents[:, 0:2]
    th = agents[:, 2] + np.pi / 2.0

    Dx = np.subtract.outer(X[:, 0], X[:, 0])
    Dy = np.subtract.outer(X[:, 1], X[:, 1])
    R2 = Dx**2 + Dy**2 + np.eye(n_agents)
    R = np.sqrt(R2)

    ex, ey = Dx / R, Dy / R
    cos_th_mat = np.tile(np.cos(th)[:, None], (1, n_agents))
    sin_th_mat = np.tile(np.sin(th)[:, None], (1, n_agents))

    exr = ex * cos_th_mat + ey * sin_th_mat
    eyr = ey * cos_th_mat - ex * sin_th_mat

    mask = (R > r_min) & (R < r_cut)
    np.fill_diagonal(mask, False)

    sig_over_r6 = (sigma**2) / (R**2 + 1e-6)
    sig_over_r12 = sig_over_r6**2
    Fmag = 8.0 * epsilon * (2.0 * sig_over_r12 - sig_over_r6) / (R + 1e-6) * mask

    Fp_x = np.sum(Fmag * exr, axis=1)
    Fp_y = np.sum(Fmag * eyr, axis=1)

    align_mask = (R < R_align) & (~np.eye(n_agents, dtype=bool))
    Th1 = np.tile(th, (n_agents, 1))
    Th2 = np.tile(th[:, None], (1, n_agents))

    H_x = align_mask * np.cos(Th1)
    H_y = align_mask * np.sin(Th1)
    Hb_x = H_x * np.cos(Th2) + H_y * np.sin(Th2)
    Hb_y = H_y * np.cos(Th2) - H_x * np.sin(Th2)

    Fa_x = np.sum(Hb_x, axis=1)
    Fa_y = np.sum(Hb_y, axis=1)
    A = np.maximum(np.sqrt(Fa_x**2 + Fa_y**2), 1e-9)
    Fa_x = Fa_x / A
    Fa_y = Fa_y / A

    Fg_gx = -1.0
    Fg_x = Fg_gx * np.cos(th)
    Fg_y = Fg_gx * -np.sin(th)

    F_x = Fp_x + k_align * Fa_x + k_goal * Fg_x
    F_y = Fp_y + k_align * Fa_y + k_goal * Fg_y

    vel = np.zeros((n_agents, 2))
    vel[:, 0] = np.clip(K1 * F_x + U, -config.LINEAR_VEL_MAX, config.LINEAR_VEL_MAX)
    vel[:, 1] = np.clip(K2 * F_y, -config.ANGULAR_VEL_MAX, config.ANGULAR_VEL_MAX)
    return vel


def _fitness(dist_travelled, average_batt, collision_time, battery_aware):
    base = config.EFF_DISTANCE_WEIGHT * dist_travelled - collision_time / config.EFF_COLLISION_WEIGHT
    if battery_aware:
        return base + average_batt / config.EFF_BATTERY_WEIGHT
    return base


def simulate_lj_baseline(rules=None, seed=None, n_agents=None, record_trajectory=False,
                          record_battery=False, battery_aware=True):
    """Runs the LJ model (numpy/kinematic backend only) until any agent's battery
    depletes. rules: a 7-gain dict (r0/epsilon/k_align/k_goal/K1/K2/U); pass
    config.PAPER_BASELINE_RULES for the Table 3 baseline. Returns (eff, dist_travelled,
    average_batt, collision_counter[, telemetry])."""
    if seed is not None:
        np.random.seed(seed)

    dt = config.DT
    n_agents = n_agents if n_agents is not None else 20
    robot_rad = config.ROBOT_RAD
    wind_rad = config.WIND_RAD
    xRange = list(config.X_RANGE)
    yRange = list(config.Y_RANGE)
    v_wind = config.V_WIND
    t = 0.0
    collision_counter = 0

    Uinf, Nx, Ny, kappa = config.UINF, config.HEBBIAN_NX, config.HEBBIAN_NY, config.KAPPA
    spawn_square_size = config.SPAWN_SQUARE_SIZE
    midpoint = list(config.SPAWN_MIDPOINT)
    min_battery, max_battery = config.MIN_BATTERY, config.MAX_BATTERY

    walls = [xRange[0] + robot_rad, xRange[1] - robot_rad, yRange[1] - robot_rad, yRange[0] + robot_rad]
    min_dist = config.COLLISION_MIN_DIST_SLACK + 2.0 * robot_rad
    min_dist_initial = config.SPAWN_MIN_DIST_SLACK + 2.0 * robot_rad

    agents = _spawn_agents(n_agents, midpoint, spawn_square_size, min_dist_initial, max_battery, min_battery)

    vel = np.zeros((n_agents, 2))
    batteryEmpty = False
    positions_log = [agents[:, 0:2].copy()] if record_trajectory else None
    battery_log = [agents[:, 3].copy()] if record_battery else None

    r0, epsilon, k_align, k_goal, K1, K2, U = _resolve_rules(rules)
    r_cut, r_min, R_align = config.R_CUT, config.R_MIN, config.R_ALIGN

    while not batteryEmpty:
        vel[:, :] = _flocking_velocity_command(agents, n_agents, r0, epsilon, k_align, k_goal, K1, K2, U,
                                                r_cut, r_min, R_align)

        vel_actual, agents, xRange, collision_counter = move(agents, vel, dt, n_agents, min_dist, walls,
                                                               collision_counter)
        if record_trajectory:
            positions_log.append(agents[:, 0:2].copy())

        yVals, xVals, powerVals = RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny)
        F_drag = dragforce(agents, wind_rad, xVals, yVals, powerVals, n_agents, vel_actual, v_wind, kappa)
        agents, batt_drain = batterydrainage(agents, vel_actual, F_drag, robot_rad, dt)
        if record_battery:
            battery_log.append(agents[:, 3].copy())

        batteryEmpty = np.any(agents[:, 3] <= 0.0)
        t += dt

    average_batt = np.mean(agents[:, 3])
    dist_travelled = -np.mean(agents[:, 0])
    collision_time = collision_counter * dt
    eff = _fitness(dist_travelled, average_batt, collision_time, battery_aware)

    if record_trajectory or record_battery:
        telemetry = {
            "positions": np.array(positions_log) if record_trajectory else None,
            "battery": np.array(battery_log) if record_battery else None,
        }
        return eff, dist_travelled, average_batt, collision_counter, telemetry
    return eff, dist_travelled, average_batt, collision_counter
