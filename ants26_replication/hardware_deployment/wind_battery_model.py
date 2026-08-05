"""Computes a per-robot *simulated* battery level in software, using the exact same
wind-wake + drag-force + battery-drainage equations as the simulation (ported verbatim,
only the config import changed, from experiment/simulation_free_global_mod_2_LJ.py's
RayTraceCircularRobots / agent_wind_percentage / dragforce / batterydrainage), driven by
real OptiTrack positions instead of simulated ones.

Why this exists: there's no way to generate a real, uniform headwind on real hardware
without a wind tunnel, and no real battery/power telemetry exists on this platform either
(see README.md). Rather than feeding a battery-aware genome a meaningless fixed value,
this reproduces the exact wind exposure and battery drain the genome was actually evolved
against, computed from the swarm's real relative positions instead of simulated ones.
controller_config.BATTERY_MODE = "simulated" wires this in (see
hebbian_swarm_experiment.py). Disclose this choice explicitly in any writeup: the
reported battery is a physically-modeled software quantity, not a measurement of real
power draw.

Requires scipy -- only this module does, only imported when BATTERY_MODE == "simulated"
(see hebbian_swarm_experiment.py's lazy import), so it's not a dependency otherwise.
"""
from functools import lru_cache

import numpy as np
from scipy.signal import convolve2d

import controller_config as cfg


@lru_cache(maxsize=None)
def _exp_kernel(Nx, Ny, x_smoothing, y_smoothing, decay):
    """Mirrors the exp(-decay*(|dx|+|dy|)) kernel built in RayTraceCircularRobots.m."""
    Kx = 2 * (Nx // x_smoothing) + 1
    Ky = 2 * (Ny // y_smoothing) + 1
    xg, yg = np.meshgrid(np.arange(1, Kx + 1), np.arange(1, Ky + 1))
    cx = np.ceil(Kx / 2.0)
    cy = np.ceil(Ky / 2.0)
    kernel = np.exp(-decay * (np.abs(xg - cx) + np.abs(yg - cy)))
    return kernel / kernel.sum()


def _replicate_smooth(P, kernel):
    """Mirrors padarray(P, ..., 'replicate', 'both') + conv2(..., 'valid')."""
    py, px = kernel.shape[0] // 2, kernel.shape[1] // 2
    Ppad = np.pad(P, ((py, py), (px, px)), mode='edge')
    return convolve2d(Ppad, kernel, mode='valid')


def RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny):
    """1:1 port of experiment/simulation_free_global_mod_2_LJ.py's RayTraceCircularRobots
    (itself a 1:1 port of RayTraceCircularRobots.m) -- x-marching wake persistence/
    recovery, two exponential-kernel smoothing passes, and the run-length based wall
    effect. Only the config import differs."""
    recovery_rate = cfg.WAKE_RECOVERY_RATE
    percent_drop = cfg.WAKE_PERCENT_DROP
    max_wall_span = cfg.WAKE_MAX_WALL_SPAN
    min_power_x = cfg.WAKE_MIN_POWER_X
    alpha, beta = cfg.WAKE_ALPHA, cfg.WAKE_BETA
    x_smoothing1, y_smoothing1 = cfg.WAKE_X_SMOOTHING_1, cfg.WAKE_Y_SMOOTHING_1
    x_smoothing2, y_smoothing2 = cfg.WAKE_X_SMOOTHING_2, cfg.WAKE_Y_SMOOTHING_2
    min_power_y = cfg.WAKE_MIN_POWER_Y

    xVals = np.linspace(xRange[0], xRange[1], Nx)
    yVals = np.linspace(yRange[0], yRange[1], Ny)
    dx = xVals[1] - xVals[0]
    dy = yVals[1] - yVals[0]

    X, Y = np.meshgrid(xVals, yVals)  # Ny x Nx, matches MATLAB's meshgrid(xVals, yVals)
    distPages = np.hypot(X[:, :, None] - agents[:, 0][None, None, :],
                          Y[:, :, None] - agents[:, 1][None, None, :])  # Ny x Nx x M

    distMin = np.min(distPages, axis=2)
    idxMat = np.argmin(distPages, axis=2)
    inMask = distMin < wind_rad

    P = np.full((Ny, Nx), float(Uinf))

    for i in range(1, Nx):
        insideNow = inMask[:, i]
        insidePrev = inMask[:, i - 1]
        robotNow = idxMat[:, i]
        robotPrev = idxMat[:, i - 1]
        Pprev = P[:, i - 1]

        justEntered = insideNow & ~insidePrev
        P[justEntered, i] = np.maximum(min_power_x, (1 - percent_drop) * Pprev[justEntered])

        staySame = insideNow & insidePrev & (robotNow == robotPrev)
        P[staySame, i] = Pprev[staySame]

        switchRobot = insideNow & insidePrev & (robotNow != robotPrev)
        P[switchRobot, i] = np.maximum(min_power_x, (1 - percent_drop) * Pprev[switchRobot])

        justExit = ~insideNow & insidePrev
        P[justExit, i] = Pprev[justExit]

        stillOut = ~insideNow & ~insidePrev
        gap = Uinf - Pprev[stillOut]
        P[stillOut, i] = np.minimum(Uinf, Pprev[stillOut] + gap * recovery_rate * dx)

    Psm = _replicate_smooth(P, _exp_kernel(Nx, Ny, x_smoothing1, y_smoothing1, alpha))

    thr_ok = Uinf - cfg.WAKE_THR_OK_DELTA
    okMask = Psm >= thr_ok

    rowIdx = np.arange(1, Ny + 1, dtype=float)[:, None]
    prevZero = np.maximum.accumulate(rowIdx * okMask, axis=0)
    distUp = (~okMask) * (rowIdx - prevZero)

    okFlip = np.flipud(okMask)
    nextZero = np.maximum.accumulate(rowIdx * okFlip, axis=0)
    distDown = (~okMask) * np.flipud(rowIdx - nextZero)

    kernelHalfY = np.minimum(distUp, distDown)
    wallScale = 2.0 * (np.tanh(3.0 * (2.0 * kernelHalfY * dy / max_wall_span - 1.0)) + 1.0)
    powerDef = (Psm / 100.0) ** wallScale
    Psm = np.maximum(min_power_y, Psm * powerDef)

    powerValsSmoothed = _replicate_smooth(Psm, _exp_kernel(Nx, Ny, x_smoothing2, y_smoothing2, beta))

    powerVals = powerValsSmoothed.T  # Nx x Ny, matching the convention dragforce expects
    return yVals, xVals, powerVals


def agent_wind_percentage(agents, wind_rad, xVals, yVals, powerVals, n_agents):
    """Each agent's experienced wind-speed percentage (U% in the paper). 1:1 port."""
    powerVs = powerVals.T
    powerVals_agents = 100.0 * np.ones(n_agents)
    for i in range(n_agents):
        x_r = agents[i, 0]
        y_r = agents[i, 1]

        x_to_left = np.where(xVals <= x_r - cfg.DRAG_UPSTREAM_LOOKAHEAD_FACTOR * wind_rad)[0]
        if len(x_to_left) > 0:
            x = x_to_left[-1]
        else:
            x = 0

        if x < 2:
            powerVals_agents[i] = 100.0
        else:
            y = np.argmin(np.abs(yVals - y_r))
            powerVals_agents[i] = powerVs[y, x]
    return powerVals_agents


def dragforce(agents, wind_rad, xVals, yVals, powerVals, n_agents, vel_actual, v_wind, kappa):
    """1:1 port."""
    powerVals_agents = agent_wind_percentage(agents, wind_rad, xVals, yVals, powerVals, n_agents)
    F_drag = np.zeros((n_agents, 2))

    for i in range(n_agents):
        v_wind_agent = (powerVals_agents[i] / 100.0) * v_wind
        v_parallel = vel_actual[i, 0] * np.sin(vel_actual[i, 2])
        v_rel = v_wind_agent + v_parallel
        F_drag[i, 0] = 0.5 * cfg.DRAG_AIR_DENSITY * cfg.DRAG_COEFFICIENT_AREA * kappa * (v_rel ** 2)

    return F_drag


def batterydrainage(agents, vel_actual, F_drag, robot_rad, dt):
    """1:1 port."""
    n_agents = agents.shape[0]
    vel_vec = np.zeros((n_agents, 2))
    vel_vec[:, 0] = -vel_actual[:, 0] * np.sin(vel_actual[:, 2])
    vel_vec[:, 1] = vel_actual[:, 0] * np.cos(vel_actual[:, 2])

    dottprod = np.zeros(n_agents)
    for i in range(n_agents):
        dottprod[i] = np.dot(vel_vec[i, :], F_drag[i, :])

    dv = vel_actual[:, 1] * robot_rad
    wheels_vel = np.column_stack((vel_actual[:, 0] - dv, vel_actual[:, 0] + dv))

    batt_drain = np.maximum(np.sum(np.abs(wheels_vel), axis=1) / cfg.BATTERY_WHEEL_POWER_DIVISOR - dottprod,
                             cfg.BATTERY_MIN_DRAIN) * dt
    agents[:, 3] = agents[:, 3] - cfg.BATTERY_DRAIN_SCALE * batt_drain
    return agents, batt_drain


def _tracked_x(agents):
    tracked = np.abs(agents[:, 0]) < cfg.UNTRACKED_XY_THRESHOLD
    return agents[tracked, 0] if np.any(tracked) else agents[:, 0]


def compute_wind_tracking_xrange(agents):
    """Mirrors _collision_and_window_bookkeeping's xRange computation in
    experiment/simulation_free_global_mod_2_LJ.py (the collision-counting half of that
    function isn't needed here). Untracked robots (pose_utils.py's (1e4, 1e4) sentinel)
    are excluded first -- otherwise a single untracked neighbor would blow up min/max and
    wreck the window for everyone."""
    x = _tracked_x(agents)
    min_x = np.min(x)
    max_x = min(np.max(x), min_x + cfg.WIND_TRACKING_MAX_SPAN)
    window_width = cfg.WIND_TRACKING_WINDOW_WIDTH
    return [min_x - (window_width - (max_x - min_x)) / 2.0,
            max_x + (window_width - (max_x - min_x)) / 2.0]


def compute_virtual_battery_update(agents, self_index, current_battery, vel_actual_self, dt):
    """agents: (n_agents, 4) real [x, y, heading, battery] this tick, as built by
    pose_utils.poses_to_agents (untracked robots at its (1e4, 1e4) sentinel). self_index:
    which row is this robot. current_battery: this robot's own virtual battery level as of
    the end of the previously completed interval. vel_actual_self: (speed, angular_vel,
    travel_heading) describing the motion actually completed since the last call, in the
    same convention move()/dragforce()/batterydrainage() use in
    experiment/simulation_free_global_mod_2_LJ.py -- speed = |displacement| / dt,
    travel_heading = atan2(dy, dx) - pi/2, angular_vel = the commanded w that was active
    over that interval (real hardware has no wheel encoders, so commanded w is the only
    available proxy -- see hebbian_swarm_experiment.py). dt: seconds elapsed over that
    interval (CONTROL_TICK_SECONDS).

    Only this robot's own row is populated with real velocity data; the wind field itself
    (which depends on every tracked robot's position as a wake source) still uses the full
    agents array, so this robot's battery correctly reflects the whole swarm's real
    configuration, not just its own motion.

    Returns (new_battery, batt_drain, wind_percentage_self) -- new_battery is what to
    write back into agents[self_index, 3] before sensing next tick; wind_percentage_self
    (0-100, 100 = undisturbed freestream) is purely for logging/diagnostics.
    """
    n_agents = agents.shape[0]
    xRange = compute_wind_tracking_xrange(agents)
    yRange = list(cfg.WIND_Y_RANGE)

    yVals, xVals, powerVals = RayTraceCircularRobots(
        agents, cfg.WIND_RAD, cfg.UINF, xRange, yRange, cfg.NX, cfg.NY)
    wind_pct_all = agent_wind_percentage(agents, cfg.WIND_RAD, xVals, yVals, powerVals, n_agents)

    vel_actual = np.zeros((n_agents, 3))
    vel_actual[self_index] = vel_actual_self
    F_drag = dragforce(agents, cfg.WIND_RAD, xVals, yVals, powerVals, n_agents,
                        vel_actual, cfg.V_WIND, cfg.KAPPA)

    agents_batt = agents.copy()
    agents_batt[self_index, 3] = current_battery
    agents_batt, batt_drain = batterydrainage(agents_batt, vel_actual, F_drag, cfg.ROBOT_RAD, dt)

    return (float(agents_batt[self_index, 3]), float(batt_drain[self_index]),
            float(wind_pct_all[self_index]))
