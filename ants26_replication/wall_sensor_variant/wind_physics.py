"""Wind-wake ray-tracing, drag force, and battery drainage -- the physics shared by
simulation_hebbian.py's own episode loop and lj_baseline.py's Table-3 comparison run.

Standalone copy: ports the same functions as initial_implementation/experiment/
simulation_free_global_mod_2_LJ.py's RayTraceCircularRobots/agent_wind_percentage/
dragforce/batterydrainage/_spawn_agents/_open_video_writer (1:1 ports of the MATLAB
reference, RayTraceCircularRobots.m etc.), duplicated here rather than imported across
the project boundary so this replication has no dependency on initial_implementation.
If you change the physics in one copy, decide explicitly whether the same change
belongs in the other -- nothing keeps them in sync automatically.
"""
from functools import lru_cache

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

try:
    import config
except ModuleNotFoundError:
    from . import config


def wrap_to_pi(angle):
    """1:1 Mirror of MATLAB's wrapToPi function."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


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
    """1:1 port of RayTraceCircularRobots.m: x-marching wake persistence/recovery,
    two exponential-kernel smoothing passes, and the run-length based wall effect."""
    recovery_rate = config.WAKE_RECOVERY_RATE
    percent_drop = config.WAKE_PERCENT_DROP
    max_wall_span = config.WAKE_MAX_WALL_SPAN
    min_power_x = config.WAKE_MIN_POWER_X
    alpha, beta = config.WAKE_ALPHA, config.WAKE_BETA
    x_smoothing1, y_smoothing1 = config.WAKE_X_SMOOTHING_1, config.WAKE_Y_SMOOTHING_1
    x_smoothing2, y_smoothing2 = config.WAKE_X_SMOOTHING_2, config.WAKE_Y_SMOOTHING_2
    min_power_y = config.WAKE_MIN_POWER_Y

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

    thr_ok = Uinf - config.WAKE_THR_OK_DELTA
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

    powerVals = powerValsSmoothed.T  # Nx x Ny, matching the convention dragforce/plot_all expect
    return yVals, xVals, powerVals


def agent_wind_percentage(agents, wind_rad, xVals, yVals, powerVals, n_agents):
    """Each agent's experienced wind-speed percentage (U% in the paper) -- the same
    downstream-grid-cell lookup dragforce() uses internally, factored out so analysis
    code (wind-exposure telemetry, the battery-awareness experiment) can read it
    directly without duplicating the lookup."""
    powerVs = powerVals.T
    powerVals_agents = 100.0 * np.ones(n_agents)
    for i in range(n_agents):
        x_r = agents[i, 0]
        y_r = agents[i, 1]

        x_to_left = np.where(xVals <= x_r - config.DRAG_UPSTREAM_LOOKAHEAD_FACTOR * wind_rad)[0]
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
    powerVals_agents = agent_wind_percentage(agents, wind_rad, xVals, yVals, powerVals, n_agents)
    F_drag = np.zeros((n_agents, 2))

    for i in range(n_agents):
        v_wind_agent = (powerVals_agents[i] / 100.0) * v_wind
        v_parallel = vel_actual[i, 0] * np.sin(vel_actual[i, 2])
        v_rel = v_wind_agent + v_parallel
        F_drag[i, 0] = 0.5 * config.DRAG_AIR_DENSITY * config.DRAG_COEFFICIENT_AREA * kappa * (v_rel ** 2)

    return F_drag


def batterydrainage(agents, vel_actual, F_drag, robot_rad, dt):
    n_agents = agents.shape[0]
    vel_vec = np.zeros((n_agents, 2))
    vel_vec[:, 0] = -vel_actual[:, 0] * np.sin(vel_actual[:, 2])
    vel_vec[:, 1] = vel_actual[:, 0] * np.cos(vel_actual[:, 2])

    dottprod = np.zeros(n_agents)
    for i in range(n_agents):
        dottprod[i] = np.dot(vel_vec[i, :], F_drag[i, :])

    dv = vel_actual[:, 1] * robot_rad
    wheels_vel = np.column_stack((vel_actual[:, 0] - dv, vel_actual[:, 0] + dv))

    batt_drain = np.maximum(np.sum(np.abs(wheels_vel), axis=1) / config.BATTERY_WHEEL_POWER_DIVISOR - dottprod,
                             config.BATTERY_MIN_DRAIN) * dt
    agents[:, 3] = agents[:, 3] - config.BATTERY_DRAIN_SCALE * batt_drain
    return agents, batt_drain


def _spawn_agents(n_agents, midpoint, spawn_square_size, min_dist_initial, max_battery, min_battery):
    agents = np.random.rand(n_agents, 4)
    agents[:, 0] = midpoint[0] + (agents[:, 0] - 0.5) * spawn_square_size
    agents[:, 1] = midpoint[1] + (agents[:, 1] - 0.5) * spawn_square_size
    agents[:, 2] = wrap_to_pi(agents[:, 2] * 2.0 * np.pi)
    agents[0:n_agents - 1, 3] = max_battery
    agents[-1, 3] = min_battery

    collision_detected = True
    while collision_detected:
        collision_detected = False
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                distance = np.sqrt((agents[i, 0] - agents[j, 0]) ** 2 + (agents[i, 1] - agents[j, 1]) ** 2)
                if distance < min_dist_initial:
                    agents[j, 0] = midpoint[0] + (np.random.rand() - 0.5) * spawn_square_size
                    agents[j, 1] = midpoint[1] + (np.random.rand() - 0.5) * spawn_square_size
                    collision_detected = True
    return agents


def _open_video_writer(visualize, video_path=None):
    if not visualize:
        plt.switch_backend('Agg')
        return None, None, None
    try:
        plt.switch_backend('TkAgg')
    except ImportError:
        print("TkAgg backend unavailable (no tkinter/display) -- writing video without a live window.")
        plt.switch_backend('Agg')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    v_out = cv2.VideoWriter(video_path or config.HEBBIAN_VIDEO_PATH, fourcc, config.VIDEO_FPS, config.VIDEO_SIZE)
    fig, ax = plt.subplots(figsize=config.VIDEO_FIGSIZE)
    return v_out, fig, ax
