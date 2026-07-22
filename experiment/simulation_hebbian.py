"""Full episode simulation for the Hebbian ABCD controller (paper Sections 2-3).

1:1 port of simulation_free_global_mod_2.m's main loop, reusing the wind/drag/battery
physics and spawn logic already validated in simulation_free_global_mod_2_LJ.py (that
file's RayTraceCircularRobots/dragforce/batterydrainage/_spawn_agents are identical
between the LJ and non-LJ MATLAB simulation files -- only the controller differs).

Unlike move() in the MATLAB source, _move() here tracks inter-robot and wall collisions
SEPARATELY rather than pre-summing them into one counter, because the paper's stage 2/3
fitness formulas (Table 2) weight and use collision_time and wall_collision_time
independently.
"""
import io

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    import config
except ModuleNotFoundError:
    from . import config

try:
    from sensor_model import get_sensor_data
    from hebbian_controller import init_weights, hebbian_step
    from simulation_free_global_mod_2_LJ import (
        wrap_to_pi, RayTraceCircularRobots, dragforce, batterydrainage, _spawn_agents,
        agent_wind_percentage, _open_video_writer,
    )
except ModuleNotFoundError:
    from .sensor_model import get_sensor_data
    from .hebbian_controller import init_weights, hebbian_step
    from .simulation_free_global_mod_2_LJ import (
        wrap_to_pi, RayTraceCircularRobots, dragforce, batterydrainage, _spawn_agents,
        agent_wind_percentage, _open_video_writer,
    )


def _move(agents, vel, dt, n_agents, min_dist, walls):
    """Kinematic integration + collision bookkeeping, mirroring move() in
    simulation_free_global_mod_2.m, but returning inter-robot and wall collision
    counts separately instead of pre-combining them with a hardcoded x3 weight."""
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

    agents_xy = agents[:, 0:2]
    D = np.linalg.norm(agents_xy[:, None, :] - agents_xy[None, :, :], axis=-1)
    close_agents = (D < min_dist) & (~np.eye(n_agents, dtype=bool))
    pair_collisions = int(np.count_nonzero(np.triu(close_agents, k=1)))

    wall_margin = config.ROBOT_RAD * config.WALL_MARGIN_FACTOR
    wall_hits = int(np.sum((agents[:, 0] > walls[1] - wall_margin) |
                           (agents[:, 1] > walls[2] - wall_margin) |
                           (agents[:, 1] < walls[3] + wall_margin)))

    min_x = np.min(agents[:, 0])
    max_x = min(np.max(agents[:, 0]), min_x + config.WIND_TRACKING_MAX_SPAN)
    window_width = config.WIND_TRACKING_WINDOW_WIDTH
    xRange = [min_x - (window_width - (max_x - min_x)) / 2.0, max_x + (window_width - (max_x - min_x)) / 2.0]

    agents[:, 0] = np.minimum(agents[:, 0], max_x)
    agents[:, 1] = np.minimum(agents[:, 1], walls[2])
    agents[:, 1] = np.maximum(agents[:, 1], walls[3])

    x_old, y_old = agents_old[:, 0], agents_old[:, 1]
    x_new, y_new = agents[:, 0], agents[:, 1]
    dist = np.sqrt((x_old - x_new) ** 2 + (y_old - y_new) ** 2)

    vel_actual[:, 2] = np.arctan2((y_new - y_old), (x_new - x_old)) - np.pi / 2.0
    vel_actual[:, 0] = dist / dt
    vel_actual[np.isnan(vel_actual[:, 2]), 2] = 0.0

    return vel_actual, agents, xRange, pair_collisions, wall_hits


def simulate_hebbian_episode(abcd_rules, seed=None, n_agents=None, wind_enabled=True,
                              max_battery=None, min_battery=None, nx=None, ny=None,
                              use_battery_sensor=True,
                              record_trajectory=False, record_battery=False,
                              record_wind_exposure=False):
    """Runs one full episode with the Hebbian ABCD controller, until any agent's
    battery depletes (same termination condition for every stage -- only the fitness
    formula computed from the results differs per stage; see stage_fitness()).

    abcd_rules: dict from hebbian_controller.unflatten_abcd(), shared by every agent.
    max_battery/min_battery: starting charge for all agents but one / for that one
        agent (default HEBBIAN_MAX_BATTERY/MIN_BATTERY, both 100 -- pass e.g.
        min_battery=50 to reproduce the paper's battery-awareness experiment).
    nx/ny: wind grid resolution override (default HEBBIAN_NX/NY); lower values cut
        simulation cost since the wake-marching loop is O(Nx) per step.
    use_battery_sensor: if False, the NN's battery input is always fed 0 (a fixed,
        uninformative value) instead of the real reading -- for evolving a baseline
        that genuinely cannot sense battery, per the paper's suggested follow-up
        ("evolve a swarm that does not incorporate battery monitoring").  This is a
        functional ablation, not a dimensional one: forward-pass-wise it is exactly
        equivalent to removing the input (0 * any weight contributes nothing to the
        hidden layer, every step, regardless of how that row's weights drift under
        the Hebbian update), while keeping the genome the same size/shape as the
        battery-aware controller so the two remain directly comparable.

    Returns (dist_travelled, average_batt, collision_time, wall_collision_time[, telemetry]).
    telemetry (only if record_trajectory/record_battery/record_wind_exposure) is a
    dict with "positions" ((n_steps, n_agents, 2) or None), "battery" ((n_steps,
    n_agents) or None), and "wind_pct" ((n_steps, n_agents) or None).
    """
    if seed is not None:
        np.random.seed(seed)

    dt = config.DT
    n_agents = n_agents if n_agents is not None else config.HEBBIAN_N_AGENTS
    robot_rad = config.ROBOT_RAD
    wind_rad = config.WIND_RAD
    xRange = list(config.X_RANGE)
    yRange = list(config.Y_RANGE)
    v_wind = config.V_WIND

    Uinf, kappa = config.UINF, config.KAPPA
    Nx = nx if nx is not None else config.HEBBIAN_NX
    Ny = ny if ny is not None else config.HEBBIAN_NY
    spawn_square_size = config.SPAWN_SQUARE_SIZE
    midpoint = list(config.SPAWN_MIDPOINT)
    max_battery = max_battery if max_battery is not None else config.HEBBIAN_MAX_BATTERY
    min_battery = min_battery if min_battery is not None else config.HEBBIAN_MIN_BATTERY

    walls = [xRange[0] + robot_rad, xRange[1] - robot_rad, yRange[1] - robot_rad, yRange[0] + robot_rad]
    min_dist = config.COLLISION_MIN_DIST_SLACK + 2.0 * robot_rad
    min_dist_initial = config.SPAWN_MIN_DIST_SLACK + 2.0 * robot_rad

    agents = _spawn_agents(n_agents, midpoint, spawn_square_size, min_dist_initial, max_battery, min_battery)
    weights = [init_weights() for _ in range(n_agents)]

    pair_collision_counter = 0
    wall_collision_counter = 0
    batteryEmpty = False
    positions_log = [agents[:, 0:2].copy()] if record_trajectory else None
    battery_log = [agents[:, 3].copy()] if record_battery else None
    wind_pct_log = [np.full(n_agents, 100.0)] if record_wind_exposure else None
    vel = np.zeros((n_agents, 2))

    while not batteryEmpty:
        sensor_inputs = get_sensor_data(agents)  # (10, n_agents)
        if not use_battery_sensor:
            sensor_inputs[8, :] = 0.0
        for i in range(n_agents):
            w1, w2, w3 = weights[i]
            v_i, w_i, w1n, w2n, w3n = hebbian_step(sensor_inputs[:, i], w1, w2, w3, abcd_rules)
            vel[i, 0] = v_i
            vel[i, 1] = w_i
            weights[i] = (w1n, w2n, w3n)

        vel_actual, agents, xRange, pair_hits, wall_hits = _move(agents, vel, dt, n_agents, min_dist, walls)
        pair_collision_counter += pair_hits
        wall_collision_counter += wall_hits
        if record_trajectory:
            positions_log.append(agents[:, 0:2].copy())

        if wind_enabled:
            yVals, xVals, powerVals = RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny)
            F_drag = dragforce(agents, wind_rad, xVals, yVals, powerVals, n_agents, vel_actual, v_wind, kappa)
            if record_wind_exposure:
                wind_pct_log.append(agent_wind_percentage(agents, wind_rad, xVals, yVals, powerVals, n_agents))
        else:
            F_drag = np.zeros((n_agents, 2))
            if record_wind_exposure:
                wind_pct_log.append(np.full(n_agents, 100.0))
        agents, batt_drain = batterydrainage(agents, vel_actual, F_drag, robot_rad, dt)
        if record_battery:
            battery_log.append(agents[:, 3].copy())

        batteryEmpty = np.any(agents[:, 3] <= 0.0)

    average_batt = np.mean(agents[:, 3])
    dist_travelled = -np.mean(agents[:, 0])
    collision_time = pair_collision_counter * dt
    wall_collision_time = wall_collision_counter * dt

    if record_trajectory or record_battery or record_wind_exposure:
        telemetry = {
            "positions": np.array(positions_log) if record_trajectory else None,
            "battery": np.array(battery_log) if record_battery else None,
            "wind_pct": np.array(wind_pct_log) if record_wind_exposure else None,
        }
        return dist_travelled, average_batt, collision_time, wall_collision_time, telemetry
    return dist_travelled, average_batt, collision_time, wall_collision_time


def _plot_hebbian_frame(ax, fig, agents, r, max_battery, t, video_writer, wind_field=None):
    """Video-frame renderer for the Hebbian controller. Same agent-circle/heading-arrow
    drawing as simulation_free_global_mod_2_LJ.py's plot_all(), but (a) takes max_battery
    as a parameter instead of hardcoding the LJ model's config.MAX_BATTERY (Hebbian
    battery is on a 0-100 scale, not the LJ model's 150), and (b) the wind-field
    background is optional, since stage 1 (walk_left) trains with wind disabled and has
    no wind field to show."""
    ax.clear()
    if wind_field is not None:
        yVals, xVals, powerVals = wind_field
        X, Y = np.meshgrid(xVals, yVals)
        ax.pcolormesh(X, Y, powerVals.T, shading='interp', cmap='viridis', vmin=0, vmax=np.max(powerVals))
    ax.set_aspect('equal')
    ax.set_title(f"Hebbian ABCD controller -- t = {t:.1f}")
    ax.set_xlabel("X - [m]")
    ax.set_ylabel("Y - [m]")

    com_x = np.mean(agents[:, 0])
    com_y = np.mean(agents[:, 1])
    half_width = config.VIDEO_VIEWPORT_HALF_WIDTH
    ax.set_xlim([com_x - half_width, com_x + half_width])
    ax.set_ylim([com_y - half_width, com_y + half_width])

    for i in range(agents.shape[0]):
        x, y, theta, battery = agents[i, 0], agents[i, 1], agents[i, 2], agents[i, 3]
        norm_b = np.clip(battery / max_battery, 0.0, 1.0)
        color = (1.0 - norm_b, norm_b, 0.0)

        circle = plt.Circle((x, y), r, fill=True, facecolor=color, edgecolor='k')
        ax.add_patch(circle)

        arrow_len = config.VIDEO_ARROW_LEN
        ax.quiver(x, y, -arrow_len * np.sin(theta), arrow_len * np.cos(theta),
                  angles='xy', scale_units='xy', scale=1, color='r', width=config.VIDEO_QUIVER_WIDTH)

    buf = io.BytesIO()
    fig.savefig(buf, format='raw')
    buf.seek(0)
    w, h = int(fig.bbox.bounds[2]), int(fig.bbox.bounds[3])
    frame = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape((h, w, 4))
    buf.close()

    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    if frame.shape[1] != config.VIDEO_SIZE[0] or frame.shape[0] != config.VIDEO_SIZE[1]:
        frame = cv2.resize(frame, config.VIDEO_SIZE)

    video_writer.write(frame)
    if plt.get_backend().lower() != 'agg':
        plt.pause(0.001)


def render_hebbian_episode_video(abcd_rules, seed=None, n_agents=None, wind_enabled=True,
                                  max_battery=None, min_battery=None, nx=None, ny=None,
                                  use_battery_sensor=True, video_path=None, record_battery=False):
    """Same episode loop/physics as simulate_hebbian_episode(), but renders a video frame
    every step (reusing simulation_free_global_mod_2_LJ.py's video-writing machinery)
    instead of just returning summary statistics. Kept as a separate function rather than
    a visualize=True branch on simulate_hebbian_episode(), since that function is CMA-ES's
    hot path (thousands of calls per training run) and shouldn't carry video-rendering
    overhead into every call.

    Returns (dist_travelled, average_batt, collision_time, wall_collision_time[, telemetry])
    -- telemetry (only if record_battery) is {"battery": (n_steps, n_agents) array}.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = config.DT
    n_agents = n_agents if n_agents is not None else config.HEBBIAN_N_AGENTS
    robot_rad = config.ROBOT_RAD
    wind_rad = config.WIND_RAD
    xRange = list(config.X_RANGE)
    yRange = list(config.Y_RANGE)
    v_wind = config.V_WIND

    Uinf, kappa = config.UINF, config.KAPPA
    Nx = nx if nx is not None else config.HEBBIAN_NX
    Ny = ny if ny is not None else config.HEBBIAN_NY
    spawn_square_size = config.SPAWN_SQUARE_SIZE
    midpoint = list(config.SPAWN_MIDPOINT)
    max_battery = max_battery if max_battery is not None else config.HEBBIAN_MAX_BATTERY
    min_battery = min_battery if min_battery is not None else config.HEBBIAN_MIN_BATTERY

    walls = [xRange[0] + robot_rad, xRange[1] - robot_rad, yRange[1] - robot_rad, yRange[0] + robot_rad]
    min_dist = config.COLLISION_MIN_DIST_SLACK + 2.0 * robot_rad
    min_dist_initial = config.SPAWN_MIN_DIST_SLACK + 2.0 * robot_rad

    agents = _spawn_agents(n_agents, midpoint, spawn_square_size, min_dist_initial, max_battery, min_battery)
    weights = [init_weights() for _ in range(n_agents)]

    v_out, fig, ax = _open_video_writer(True, video_path=video_path or config.HEBBIAN_VIDEO_PATH)

    pair_collision_counter = 0
    wall_collision_counter = 0
    batteryEmpty = False
    battery_log = [agents[:, 3].copy()] if record_battery else None
    vel = np.zeros((n_agents, 2))
    t = 0.0

    wind_field = None
    if wind_enabled:
        wind_field = RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny)
    _plot_hebbian_frame(ax, fig, agents, robot_rad, max_battery, t, v_out, wind_field)

    while not batteryEmpty:
        sensor_inputs = get_sensor_data(agents)
        if not use_battery_sensor:
            sensor_inputs[8, :] = 0.0
        for i in range(n_agents):
            w1, w2, w3 = weights[i]
            v_i, w_i, w1n, w2n, w3n = hebbian_step(sensor_inputs[:, i], w1, w2, w3, abcd_rules)
            vel[i, 0] = v_i
            vel[i, 1] = w_i
            weights[i] = (w1n, w2n, w3n)

        vel_actual, agents, xRange, pair_hits, wall_hits = _move(agents, vel, dt, n_agents, min_dist, walls)
        pair_collision_counter += pair_hits
        wall_collision_counter += wall_hits

        if wind_enabled:
            yVals, xVals, powerVals = RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny)
            F_drag = dragforce(agents, wind_rad, xVals, yVals, powerVals, n_agents, vel_actual, v_wind, kappa)
            wind_field = (yVals, xVals, powerVals)
        else:
            F_drag = np.zeros((n_agents, 2))
            wind_field = None
        agents, batt_drain = batterydrainage(agents, vel_actual, F_drag, robot_rad, dt)
        if record_battery:
            battery_log.append(agents[:, 3].copy())

        batteryEmpty = np.any(agents[:, 3] <= 0.0)
        t += dt
        _plot_hebbian_frame(ax, fig, agents, robot_rad, max_battery, t, v_out, wind_field)

    average_batt = np.mean(agents[:, 3])
    dist_travelled = -np.mean(agents[:, 0])
    collision_time = pair_collision_counter * dt
    wall_collision_time = wall_collision_counter * dt

    v_out.release()
    plt.close(fig)

    if record_battery:
        telemetry = {"battery": np.array(battery_log)}
        return dist_travelled, average_batt, collision_time, wall_collision_time, telemetry
    return dist_travelled, average_batt, collision_time, wall_collision_time


def stage_fitness(dist_travelled, average_batt, collision_time, wall_collision_time, stage):
    """Table 2's per-stage fitness formula:
    eff = dist + avg_batt/battery_w - (wall_col_mult*wall_col_time [+ collision_time]) / collision_w
    """
    battery_w, collision_w, wall_col_mult, include_inter_robot = config.HEBBIAN_STAGE_FITNESS_WEIGHTS[stage]
    eff = dist_travelled
    if battery_w is not None:
        eff += average_batt / battery_w
    if collision_w is not None:
        penalty = wall_col_mult * wall_collision_time
        if include_inter_robot:
            penalty += collision_time
        eff -= penalty / collision_w
    return eff
