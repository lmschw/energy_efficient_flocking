"""Full episode simulation for the Hebbian ABCD controller (paper Sections 2-3).

1:1 port of simulation_free_global_mod_2.m's main loop, reusing the wind/drag/battery
physics and spawn logic in this package's own wind_physics.py (a standalone copy of the
same functions originally validated in initial_implementation/experiment/
simulation_free_global_mod_2_LJ.py -- RayTraceCircularRobots/dragforce/batterydrainage/
_spawn_agents are identical between the LJ and non-LJ MATLAB simulation files, only the
controller differs -- duplicated here so this package has no import dependency on
initial_implementation; see wind_physics.py's docstring).

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
    from wind_physics import (
        wrap_to_pi, RayTraceCircularRobots, dragforce, batterydrainage, _spawn_agents,
        agent_wind_percentage, _open_video_writer,
    )
except ModuleNotFoundError:
    from .sensor_model import get_sensor_data
    from .hebbian_controller import init_weights, hebbian_step
    from .wind_physics import (
        wrap_to_pi, RayTraceCircularRobots, dragforce, batterydrainage, _spawn_agents,
        agent_wind_percentage, _open_video_writer,
    )


def _resolve_collisions(agents, min_dist, n_agents, max_iter=10, strength=1.0):
    """Iteratively pushes overlapping agents apart toward min_dist -- REVIVES a
    mechanism that exists in the MATLAB reference (simulation_free_global_mod_2.m's
    move()) but is commented out there, so neither the reference nor this port's
    prior behavior ever actually prevented agents from overlapping; both only counted
    and penalized it after the fact via the fitness formula.

    Faithful port of the disabled MATLAB loop: processes pairs in one of two priority
    orderings, alternating across iterations (first half of max_iter: agents ordered
    right-to-left by x; second half: ordered by distance to the nearest arena corner).
    For each ordered pair (i before j in that pass's order), if too close, agent j is
    pushed along their connecting bearing -- agent i (earlier/higher-priority in that
    pass) stays fixed. Runs for up to max_iter passes so a push that creates a new
    conflict with a third agent gets a chance to resolve too (matches the MATLAB
    constant of 10 exactly at strength=1.0).

    strength: how far to close the gap per push, in [0, 1]. 1.0 (MATLAB-faithful)
    snaps straight to min_dist in one push. Lower values only close part of the gap
    per push (target_dist = dist + strength*(min_dist-dist)), so a hard pileup takes
    several of the max_iter passes to fully separate rather than being erased
    instantly -- added after strength=1.0 (n10_seed42) showed the controller driving
    into MORE collision-course commands than any other variant tried (357.5s) and
    battery cratering: full-strength resolution may remove the physical consequence of
    crowding so completely that there's nothing left teaching the controller to avoid
    it in the first place, since the fitness-visible collision-count penalty alone
    apparently isn't enough. A softer push keeps some lingering cost (reduced net
    progress, more resolution passes needed) without reintroducing true interpenetration.
    """
    corners = np.array([[5.0, 5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, -5.0]])
    order1 = np.argsort(-agents[:, 0])  # right to left, matches MATLAB's order1
    dist_to_corners = np.linalg.norm(agents[:, None, 0:2] - corners[None, :, :], axis=2)
    order2 = np.argsort(dist_to_corners.min(axis=1))  # nearest-corner-first, matches order2

    for it in range(max_iter):
        order = order1 if it < max_iter // 2 + 1 else order2
        for idx in range(n_agents):
            i = order[idx]
            for jdx in range(idx + 1, n_agents):
                j = order[jdx]
                dx = agents[j, 0] - agents[i, 0]
                dy = agents[j, 1] - agents[i, 1]
                dist = np.hypot(dx, dy)
                if dist < min_dist:
                    angle = np.arctan2(dy, dx)
                    target_dist = dist + strength * (min_dist - dist)
                    agents[j, 0] = agents[i, 0] + target_dist * np.cos(angle)
                    agents[j, 1] = agents[i, 1] + target_dist * np.sin(angle)
    return agents


def _proximity_penalty(D, iu, min_dist):
    """Graduated warning penalty for inter-agent surface gap, evaluated on the SAME
    pre-resolution center-distance matrix D used for pair_collisions -- see
    HEBBIAN_PROXIMITY_* in config.py for the two-zone shape this implements and why.
    Outer zone (inner_gap <= gap < outer_gap) ramps LINEARLY 0->1 -- a gentle, constant-slope
    warning. Inner zone (contact <= gap < inner_gap) ramps QUADRATICALLY 1->(1+STEEP_MULT) --
    a steep, accelerating penalty as an agent approaches actual contact. contact is derived
    from the SAME min_dist the caller uses for pair_collisions/instant-death, so if
    HEBBIAN_MIN_DIST_INFLATION is active (training-time sim-to-real margin, see config.py),
    the warning zone's inner edge inflates consistently with everything else rather than
    still pointing at the true physical contact distance. Returns the per-step sum over all
    pairs (unscaled by dt; the caller time-weights it, matching collision_time's own dt
    scaling)."""
    gap = D[iu] - 2.0 * config.ROBOT_RAD
    outer = config.HEBBIAN_PROXIMITY_OUTER_GAP
    inner = config.HEBBIAN_PROXIMITY_INNER_GAP
    contact = min_dist - 2.0 * config.ROBOT_RAD
    mid_frac = np.clip((outer - gap) / (outer - inner), 0.0, 1.0)
    steep_frac = np.clip((inner - gap) / (inner - contact), 0.0, 1.0)
    pair_penalty = mid_frac + config.HEBBIAN_PROXIMITY_STEEP_MULT * steep_frac ** 2
    return float(np.sum(pair_penalty))


def _apply_safety_clamp(agents, vel, n_agents, walls):
    """Hard, non-learned forward-speed limiter -- see HEBBIAN_SAFETY_CLAMP_*/
    HEBBIAN_WALL_SAFETY_CLAMP_* in config.py. Caps each agent's forward speed based on its
    CURRENT gap to (a) the nearest neighbor and (b) the nearest wall (using the TRUE
    ROBOT_RAD, never HEBBIAN_MIN_DIST_INFLATION -- this models a real onboard reflex, not a
    training illusion), each linearly falling to zero at contact; the agent gets whichever
    scale is more restrictive. Turning (vel[:,1]) is left untouched so a braked agent can
    still steer clear. Modifies vel in place and returns it."""
    outer = config.HEBBIAN_SAFETY_CLAMP_OUTER_GAP
    inner = config.HEBBIAN_SAFETY_CLAMP_INNER_GAP
    if n_agents > 1:
        agents_xy = agents[:, 0:2]
        D = np.linalg.norm(agents_xy[:, None, :] - agents_xy[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        nearest_gap = D.min(axis=1) - 2.0 * config.ROBOT_RAD
        agent_scale = np.clip((nearest_gap - inner) / (outer - inner), 0.0, 1.0)
    else:
        agent_scale = np.ones(n_agents)

    # walls = [x_left, x_right, y_top, y_bottom], y_top/y_bottom already inset by ROBOT_RAD
    # (see _move()'s wall clamp below), so agents[:,1] vs walls[2]/walls[3] IS the surface gap.
    wall_outer = config.HEBBIAN_WALL_SAFETY_CLAMP_OUTER_GAP
    wall_inner = config.HEBBIAN_WALL_SAFETY_CLAMP_INNER_GAP
    gap_top = walls[2] - agents[:, 1]
    gap_bottom = agents[:, 1] - walls[3]
    nearest_wall_gap = np.minimum(gap_top, gap_bottom)
    wall_scale = np.clip((nearest_wall_gap - wall_inner) / (wall_outer - wall_inner), 0.0, 1.0)

    vel[:, 0] *= np.minimum(agent_scale, wall_scale)
    return vel


def _move(agents, vel, dt, n_agents, min_dist, walls, resolve_collisions=False,
          resolve_strength=1.0, resolve_max_iter=10):
    """Kinematic integration + collision bookkeeping, mirroring move() in
    simulation_free_global_mod_2.m, but returning inter-robot and wall collision
    counts separately instead of pre-combining them with a hardcoded x3 weight.

    resolve_collisions: if True, runs _resolve_collisions() after the collision
    COUNT/penalty bookkeeping below but before the wall clamp -- same order as the
    (disabled) MATLAB block, so the fitness penalty still sees/counts every step an
    agent was commanded onto a collision course (the learning signal is unchanged),
    while the actual simulated positions -- and therefore every downstream
    consumer: wind drag, next step's sensor readings, the video -- never contain
    truly overlapping agents.

    If config.HEBBIAN_SAFETY_CLAMP_ENABLED, a hard forward-speed limit (see
    _apply_safety_clamp()) is applied to vel FIRST, based on agents' pre-step positions --
    a non-learned reflex the fitness-visible collision bookkeeping below still sees the
    (clamped) outcome of, and which vel_actual/wind-drag physics also see, since it modifies
    vel in place before anything else reads it."""
    if config.HEBBIAN_SAFETY_CLAMP_ENABLED:
        vel = _apply_safety_clamp(agents, vel, n_agents, walls)

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
    iu = np.triu_indices(n_agents, k=1)
    mean_pairwise_dist = float(np.mean(D[iu])) if n_agents > 1 else 0.0
    proximity_penalty = _proximity_penalty(D, iu, min_dist) if n_agents > 1 else 0.0

    wall_margin = config.ROBOT_RAD * config.WALL_MARGIN_FACTOR
    wall_hits = int(np.sum((agents[:, 0] > walls[1] - wall_margin) |
                           (agents[:, 1] > walls[2] - wall_margin) |
                           (agents[:, 1] < walls[3] + wall_margin)))

    if resolve_collisions:
        agents = _resolve_collisions(agents, min_dist, n_agents,
                                      max_iter=resolve_max_iter, strength=resolve_strength)

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

    return vel_actual, agents, xRange, pair_collisions, wall_hits, mean_pairwise_dist, proximity_penalty


def simulate_hebbian_episode(abcd_rules, seed=None, n_agents=None, wind_enabled=True,
                              max_battery=None, min_battery=None, nx=None, ny=None,
                              use_battery_sensor=True,
                              record_trajectory=False, record_battery=False,
                              record_wind_exposure=False):
    """Runs one full episode with the Hebbian ABCD controller, until any agent's
    battery depletes (same termination condition for every stage -- only the fitness
    formula computed from the results differs per stage; see stage_fitness()), OR,
    if config.HEBBIAN_COLLISION_INSTANT_DEATH is True, the instant any pair of agents
    crosses the same min_dist threshold used for the collision-count fitness penalty
    -- see that flag's own comment in config.py.

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

    Returns (dist_travelled, average_batt, collision_time, wall_collision_time,
    cohesion_dist, proximity_penalty[, telemetry]). cohesion_dist is the episode-averaged
    mean pairwise inter-agent distance (mean over all agent pairs, mean over all steps) --
    a pure geometric closeness measure, not a velocity/heading term, so rewarding a LOW
    value only ever pulls agents toward each other; it carries no information about which
    direction to move or how to get there, so any resulting formation behavior (e.g.
    drafting) is left entirely to CMA-ES/Hebbian learning to discover, same as the
    existing terms. proximity_penalty is the episode-summed, dt-weighted graduated
    warning penalty from HEBBIAN_PROXIMITY_* (config.py) -- see that constant's comment
    for the two-zone shape; unlike collision_time (a binary in/out-of-collision count),
    this rewards a genome for staying OUTSIDE the warning zone entirely and punishes
    approach continuously, giving CMA-ES a gradient to climb before it ever reaches the
    collision threshold. telemetry (only if record_trajectory/record_battery/
    record_wind_exposure) is a dict with "positions" ((n_steps, n_agents, 2) or None),
    "battery" ((n_steps, n_agents) or None), and "wind_pct" ((n_steps, n_agents) or None).
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
    # HEBBIAN_MIN_DIST_INFLATION (>1.0 during training only, see config.py) makes CMA-ES treat
    # agents as bigger than they physically are, for a sim-to-real safety margin.
    min_dist = (config.COLLISION_MIN_DIST_SLACK + 2.0 * robot_rad) * config.HEBBIAN_MIN_DIST_INFLATION
    min_dist_initial = config.SPAWN_MIN_DIST_SLACK + 2.0 * robot_rad

    agents = _spawn_agents(n_agents, midpoint, spawn_square_size, min_dist_initial, max_battery, min_battery)
    weights = [init_weights() for _ in range(n_agents)]

    pair_collision_counter = 0
    wall_collision_counter = 0
    proximity_penalty_sum = 0.0
    cohesion_dist_sum = 0.0
    cohesion_dist_steps = 0
    batteryEmpty = False
    collision_death = False
    positions_log = [agents[:, 0:2].copy()] if record_trajectory else None
    battery_log = [agents[:, 3].copy()] if record_battery else None
    wind_pct_log = [np.full(n_agents, 100.0)] if record_wind_exposure else None
    vel = np.zeros((n_agents, 2))

    while not (batteryEmpty or collision_death):
        sensor_inputs = get_sensor_data(agents)  # (10, n_agents)
        if not use_battery_sensor:
            sensor_inputs[8, :] = 0.0
        for i in range(n_agents):
            w1, w2, w3 = weights[i]
            v_i, w_i, w1n, w2n, w3n = hebbian_step(sensor_inputs[:, i], w1, w2, w3, abcd_rules)
            vel[i, 0] = v_i
            vel[i, 1] = w_i
            weights[i] = (w1n, w2n, w3n)

        vel_actual, agents, xRange, pair_hits, wall_hits, mean_pairwise_dist, proximity_step = _move(
            agents, vel, dt, n_agents, min_dist, walls,
            resolve_collisions=config.HEBBIAN_RESOLVE_COLLISIONS,
            resolve_strength=config.HEBBIAN_RESOLVE_COLLISIONS_STRENGTH,
            resolve_max_iter=config.HEBBIAN_RESOLVE_COLLISIONS_MAX_ITER)
        pair_collision_counter += pair_hits
        wall_collision_counter += wall_hits
        proximity_penalty_sum += proximity_step
        cohesion_dist_sum += mean_pairwise_dist
        cohesion_dist_steps += 1
        if config.HEBBIAN_COLLISION_INSTANT_DEATH and pair_hits > 0:
            collision_death = True
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
    proximity_penalty = proximity_penalty_sum * dt
    cohesion_dist = cohesion_dist_sum / cohesion_dist_steps if cohesion_dist_steps else 0.0

    if record_trajectory or record_battery or record_wind_exposure:
        telemetry = {
            "positions": np.array(positions_log) if record_trajectory else None,
            "battery": np.array(battery_log) if record_battery else None,
            "wind_pct": np.array(wind_pct_log) if record_wind_exposure else None,
        }
        return (dist_travelled, average_batt, collision_time, wall_collision_time, cohesion_dist,
                proximity_penalty, telemetry)
    return dist_travelled, average_batt, collision_time, wall_collision_time, cohesion_dist, proximity_penalty


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

        vel_actual, agents, xRange, pair_hits, wall_hits, _, _ = _move(
            agents, vel, dt, n_agents, min_dist, walls,
            resolve_collisions=config.HEBBIAN_RESOLVE_COLLISIONS,
            resolve_strength=config.HEBBIAN_RESOLVE_COLLISIONS_STRENGTH,
            resolve_max_iter=config.HEBBIAN_RESOLVE_COLLISIONS_MAX_ITER)
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


def stage_fitness(dist_travelled, average_batt, collision_time, wall_collision_time, cohesion_dist,
                   proximity_penalty, stage):
    """Table 2's per-stage fitness formula, extended with an explicit distance weight
    (config.HEBBIAN_EFF_DISTANCE_WEIGHT) so distance-travelled dominates battery
    preservation -- mirroring the LJ model's own EFF_DISTANCE_WEIGHT fix for the
    identical failure mode. Table 2's literal formula has no such multiplier; see that
    constant's comment for why this is a deliberate, disclosed deviation.

    eff = HEBBIAN_EFF_DISTANCE_WEIGHT*dist + avg_batt/battery_w
          - (wall_col_mult*wall_col_time [+ collision_time]) / collision_w
          - cohesion_dist / cohesion_w
          - proximity_penalty / proximity_w

    cohesion_dist is a pure geometric closeness term (mean pairwise inter-agent
    distance) -- purely positional, carries no velocity/heading/direction information,
    so it can only ever pull agents toward each other; it can't reward "moving" in any
    sense. See HEBBIAN_STAGE_FITNESS_WEIGHTS's comment for why this is separate from,
    and much smaller than, the collision penalty above. proximity_penalty is the
    graduated pre-collision warning term -- see HEBBIAN_PROXIMITY_* in config.py.
    """
    battery_w, collision_w, wall_col_mult, include_inter_robot, cohesion_w, proximity_w = \
        config.HEBBIAN_STAGE_FITNESS_WEIGHTS[stage]
    eff = config.HEBBIAN_EFF_DISTANCE_WEIGHT * dist_travelled
    if battery_w is not None:
        eff += average_batt / battery_w
    if collision_w is not None:
        penalty = wall_col_mult * wall_collision_time
        if include_inter_robot:
            penalty += collision_time
        eff -= penalty / collision_w
    if cohesion_w is not None:
        eff -= cohesion_dist / cohesion_w
    if proximity_w is not None:
        eff -= proximity_penalty / proximity_w
    return eff
