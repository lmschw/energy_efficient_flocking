"""Two functions, both copies of simulation_hebbian.simulate_hebbian_episode() with exactly
one change: any agent within min_dist of another agent AT THE END of a step has the
DRAG/MOVEMENT portion of that step's battery drain partially or fully refunded -- but NEVER
the unconditional idle-power floor (BATTERY_MIN_DRAIN), which is always paid regardless of
collision state (see "IMPORTANT" below for why). The fitness-visible collision_time
bookkeeping (and therefore the flat collision_w penalty in stage_fitness()) is completely
unaffected by any of this in either function.

  - simulate_hebbian_episode_no_drain_on_collision: FULL refund of the above-floor portion
    (drain_fraction=0.0 hardcoded) -- job 6 (2026-09-09/10).
  - simulate_hebbian_episode_partial_drain_on_collision: PARTIAL refund of the above-floor
    portion, `drain_fraction` parameter (e.g. 0.1 = colliding agents still pay 10% of that
    portion, 90% refunded) -- the follow-up requested after job 6, to see whether a small
    non-zero cost still nudges CMA-ES away from gratuitous collision while keeping most of
    the drain-holiday benefit. Kept as a SEPARATE function (not a parameterized version of
    the first) so job 6's already-validated behavior can never be accidentally altered by a
    later edit made for this one.

This isolates a question raised after `pre_clamp_best` (no clamp, collision_w=250 penalty,
202.5s of collision_time at n=10) came in well below the LJ baseline on battery despite
comparable distance: batterydrainage() has NO explicit dependency on collision state --
drain is purely a function of an agent's own velocity and the wind-drag force it
experiences -- so it was an open question whether the battery shortfall is coming from
something in the ordinary drain formula behaving differently near collisions (e.g. erratic
velocity/heading during close encounters), independent of the collision_w penalty itself.

IMPORTANT -- why the idle floor is never refunded, even at drain_fraction=0.0: job 6's FIRST
attempt refunded 100% of drain including the floor, which meant a permanently-colliding agent
paid literally zero net drain per step. CMA-ES found and exploited this immediately: cluster
into permanent mass collision (all pairs colliding every step) and cruise at max speed for as
long as the simulation allows -- confirmed via the actual result, dist=264m (13x the LJ
baseline), collision_time=78306s, hitting the 5000-step MAX_STEPS cap exactly, having received
13640.8 units of refunded drain. That's not a flocking strategy, it's fitness-hacking the
refund mechanic itself. Always charging the floor bounds the worst case mathematically: even
at drain_fraction=0.0, max possible episode length is
HEBBIAN_MAX_BATTERY / (BATTERY_DRAIN_SCALE * BATTERY_MIN_DRAIN * DT) = 100/(2.0*0.10*0.5) =
1000 steps -- comparable to a legitimately long episode, not an order of magnitude beyond one.

Both also include a MAX_STEPS=5000 safety cap as a secondary backstop (see job 6's post-mortem
in project memory for the original, floor-unaware motivation) -- with the floor fix above, the
mathematical 1000-step bound makes this now rarely if ever binding; it's kept as cheap insurance
rather than the load-bearing safeguard it mistakenly was before.

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

        # --- drain holiday: refund this step's drain for any agent within min_dist of another,
        # EXCEPT the unconditional idle-power floor (BATTERY_MIN_DRAIN), which is always paid
        # regardless of collision state -- refunding the floor too would let a permanently-
        # colliding agent pay literally zero net drain, giving it unlimited "free" flight time
        # bounded only by MAX_STEPS (confirmed: job 6's first attempt did exactly this --
        # agents clustered into permanent mass collision and cruised to the 5000-step cap at
        # max speed, producing a fitness-hacked 264m/78306s-collision result that has nothing
        # to do with realistic flocking). Only the portion of drain ABOVE the floor -- i.e.
        # attributable to wheel/drag power, not baseline idle draw -- is refundable. ---
        agents_xy = agents[:, 0:2]
        D = np.linalg.norm(agents_xy[:, None, :] - agents_xy[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        colliding_agent = D.min(axis=1) < min_dist if n_agents > 1 else np.zeros(n_agents, dtype=bool)
        if np.any(colliding_agent):
            floor = config.BATTERY_MIN_DRAIN * dt
            refundable = np.maximum(batt_drain - floor, 0.0)
            refund = config.BATTERY_DRAIN_SCALE * refundable[colliding_agent]
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


def simulate_hebbian_episode_partial_drain_on_collision(
        abcd_rules, seed=None, n_agents=None, wind_enabled=True,
        max_battery=None, min_battery=None, nx=None, ny=None, use_battery_sensor=True,
        record_trajectory=False, record_battery=False, record_wind_exposure=False,
        drain_fraction=0.1):
    """Same as simulate_hebbian_episode_no_drain_on_collision, except colliding agents still
    pay `drain_fraction` of their normal drain (default 0.1 = 10%, i.e. 90% refunded) instead
    of the full refund. See module docstring."""
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
    refunded_drain_total = 0.0
    step_count = 0
    MAX_STEPS = 5000  # see simulate_hebbian_episode_no_drain_on_collision's comment -- same
    # rationale applies whenever drain_fraction < 1.0, just less severely as it approaches 1.0.

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

        # --- partial drain holiday: refund (1 - drain_fraction) of this step's drain for any
        # agent within min_dist of another (drain_fraction=0.0 reduces to a full refund), EXCEPT
        # the unconditional idle-power floor (BATTERY_MIN_DRAIN), which is always paid regardless
        # of collision state or drain_fraction -- see the sibling function's comment for why
        # (job 6's first attempt refunded the floor too and produced a fitness-hacked
        # permanent-mass-collision exploit that ran to the step cap at max speed). ---
        agents_xy = agents[:, 0:2]
        D = np.linalg.norm(agents_xy[:, None, :] - agents_xy[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        colliding_agent = D.min(axis=1) < min_dist if n_agents > 1 else np.zeros(n_agents, dtype=bool)
        if np.any(colliding_agent):
            floor = config.BATTERY_MIN_DRAIN * dt
            refundable = np.maximum(batt_drain - floor, 0.0)
            refund = (1.0 - drain_fraction) * config.BATTERY_DRAIN_SCALE * refundable[colliding_agent]
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
