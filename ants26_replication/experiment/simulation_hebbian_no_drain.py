"""Two functions, both copies of simulation_hebbian.simulate_hebbian_episode() with exactly
one change: any agent within min_dist of another agent AT THE END of a step has that step's
battery drain partially or fully refunded, while the fitness-visible collision_time
bookkeeping (and therefore the flat collision_w penalty in stage_fitness()) is completely
unaffected in either case.

  - simulate_hebbian_episode_no_drain_on_collision: FULL refund (drain_fraction=0.0 hardcoded)
    -- the original version, job 6 (2026-09-09/10).
  - simulate_hebbian_episode_partial_drain_on_collision: PARTIAL refund, `drain_fraction`
    parameter (e.g. 0.1 = colliding agents still pay 10% of normal drain, 90% refunded) --
    the follow-up requested after job 6, to see whether a small non-zero cost still nudges
    CMA-ES away from gratuitous collision while keeping most of the drain-holiday benefit.
    Kept as a SEPARATE function (not a parameterized version of the first) so job 6's already-
    validated behavior can never be accidentally altered by a later edit made for this one.

This isolates a question raised after `pre_clamp_best` (no clamp, collision_w=250 penalty,
202.5s of collision_time at n=10) came in well below the LJ baseline on battery despite
comparable distance: batterydrainage() has NO explicit dependency on collision state --
drain is purely a function of an agent's own velocity and the wind-drag force it
experiences -- so it was an open question whether the battery shortfall is coming from
something in the ordinary drain formula behaving differently near collisions (e.g. erratic
velocity/heading during close encounters), independent of the collision_w penalty itself.

Both include the same MAX_STEPS safety cap (see job 6's post-mortem in project memory): a
persistent near-collision can make an agent's battery last far longer than normal before
depleting, which is pure wasted simulation time whenever the fitness formula doesn't reward
that survival (e.g. walk_left, pure distance) -- capping episode length bounds the cost of
that regardless of drain_fraction.

Not merged into simulation_hebbian.py itself, to keep that module's return signature/behavior
byte-for-byte what hardware_transfer_test/'s reproduction recipe depends on.
"""
import numpy as np

import config
from hebbian_controller import init_weights, hebbian_step
from sensor_model import get_sensor_data
from simulation_hebbian import _move
from wind_physics import RayTraceCircularRobots, dragforce, batterydrainage, _spawn_agents, agent_wind_percentage


def simulate_hebbian_episode_no_drain_on_collision(
        abcd_rules, seed=None, n_agents=None, wind_enabled=True,
        max_battery=None, min_battery=None, nx=None, ny=None, use_battery_sensor=True,
        record_trajectory=False, record_battery=False, record_wind_exposure=False):
    """Same signature/return shape as simulate_hebbian_episode() -- see that function's
    docstring. The only behavioral difference is the drain-holiday for colliding agents."""
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
    refunded_drain_total = 0.0  # diagnostic: how much drain was refunded over the episode
    step_count = 0
    MAX_STEPS = 5000  # safety cap: if two agents settle into a persistent near-collision,
    # their drain gets refunded almost every step, so the episode may need vastly more steps
    # than normal to ever deplete battery -- especially costly during walk_left, whose fitness
    # (pure distance) gets ZERO benefit from that stall, so it's wasted simulation time with no
    # offsetting learning signal. 5000 is generous relative to a normal episode (a few hundred
    # steps with wind on, up to ~1000 for wind-off walk_left) -- this only bites pathological
    # cases. Hit here means the episode is treated as ended, same as batteryEmpty.

    while not (batteryEmpty or collision_death) and step_count < MAX_STEPS:
        step_count += 1
        sensor_inputs = get_sensor_data(agents)
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

        # --- drain holiday: refund this step's drain for any agent within min_dist of another ---
        agents_xy = agents[:, 0:2]
        D = np.linalg.norm(agents_xy[:, None, :] - agents_xy[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        colliding_agent = D.min(axis=1) < min_dist if n_agents > 1 else np.zeros(n_agents, dtype=bool)
        if np.any(colliding_agent):
            refund = config.BATTERY_DRAIN_SCALE * batt_drain[colliding_agent]
            agents[colliding_agent, 3] += refund
            refunded_drain_total += float(np.sum(refund))

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
            "refunded_drain_total": refunded_drain_total,
        }
        return (dist_travelled, average_batt, collision_time, wall_collision_time, cohesion_dist,
                proximity_penalty, telemetry)
    return dist_travelled, average_batt, collision_time, wall_collision_time, cohesion_dist, proximity_penalty
