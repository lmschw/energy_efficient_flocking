import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
from functools import lru_cache
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

def RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny, useGPU=0):
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

def dragforce(agents, wind_rad, xVals, yVals, powerVals, n_agents, vel_actual, v_wind, kappa):
    powerVs = powerVals.T
    powerVals_agents = 100.0 * np.ones(n_agents)
    F_drag = np.zeros((n_agents, 2))
    
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

def _collision_and_window_bookkeeping(agents, n_agents, min_dist, walls, collision_counter):
    """Shared by move() (numpy backend) and _pybullet_move() (pybullet backend):
    counts near-collisions/wall-proximity events and computes the wind-tracking
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

def move(agents, vel, dt, v_avg, n_agents, min_dist, walls, collision_counter):
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

    vel_actual[:, 2] = np.atan2((y_new - y_old), (x_new - x_old)) - np.pi / 2.0
    vel_actual[:, 0] = dist / dt
    vel_actual[np.isnan(vel_actual[:, 2]), 2] = 0.0

    return vel_actual, agents, xRange, collision_counter

def plot_all(ax, fig, agents, r, yVals, xVals, powerVals, t, video_writer):
    ax.clear()
    X, Y = np.meshgrid(xVals, yVals)
    ax.pcolormesh(X, Y, powerVals.T, shading='interp', cmap='viridis', vmin=0, vmax=np.max(powerVals))
    ax.set_aspect('equal')
    ax.set_title(f"Swarm intelligence experiment -- t = {t:.1f}")
    ax.set_xlabel("X - [m]")
    ax.set_ylabel("Y - [m]")
    
    # Smoothly tracking Center of Mass Viewport
    com_x = np.mean(agents[:, 0])
    com_y = np.mean(agents[:, 1])
    half_width = config.VIDEO_VIEWPORT_HALF_WIDTH
    ax.set_xlim([com_x - half_width, com_x + half_width])
    ax.set_ylim([com_y - half_width, com_y + half_width])

    for i in range(agents.shape[0]):
        x, y, theta, battery = agents[i, 0], agents[i, 1], agents[i, 2], agents[i, 3]
        norm_b = np.clip(battery / config.MAX_BATTERY, 0.0, 1.0)
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

def _resolve_rules(rules):
    defaults = config.DEFAULT_RULES
    return tuple(rules[key] if rules else defaults[key]
                 for key in ("r0", "epsilon", "k_align", "k_goal", "K1", "K2", "U"))

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

def _flocking_velocity_command(agents, n_agents, r0, epsilon, k_align, k_goal, K1, K2, U, r_cut, r_min, R_align):
    """LJ spacing + heading alignment + goal-pull force, mapped to a (linear speed,
    angular rate) command. Identical control law for every backend -- only how that
    command gets turned into motion (kinematic teleport vs. real force/torque) differs."""
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

def _open_video_writer(visualize):
    if not visualize:
        plt.switch_backend('Agg')
        return None, None, None
    try:
        plt.switch_backend('TkAgg')
    except ImportError:
        print("TkAgg backend unavailable (no tkinter/display) -- writing video without a live window.")
        plt.switch_backend('Agg')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    v_out = cv2.VideoWriter(config.VIDEO_PATH, fourcc, config.VIDEO_FPS, config.VIDEO_SIZE)
    fig, ax = plt.subplots(figsize=config.VIDEO_FIGSIZE)
    return v_out, fig, ax

def _fitness(dist_travelled, average_batt, collision_time, battery_aware):
    if battery_aware:
        return dist_travelled + average_batt / config.EFF_BATTERY_WEIGHT - collision_time / config.EFF_COLLISION_WEIGHT
    return dist_travelled - collision_time / config.EFF_COLLISION_WEIGHT

def _simulate_numpy(rules, seed, visualize, n_agents, record_trajectory, record_battery, battery_aware):
    if seed is not None:
        np.random.seed(seed)

    v_out, fig, ax = _open_video_writer(visualize)

    dt = config.DT
    n_agents = n_agents if n_agents is not None else config.N_AGENTS
    robot_rad = config.ROBOT_RAD
    wind_rad = config.WIND_RAD
    xRange = list(config.X_RANGE)
    yRange = list(config.Y_RANGE)
    v_avg = config.V_AVG
    v_wind = config.V_WIND
    t = 0.0
    collision_counter = 0

    Uinf, Nx, Ny, kappa = config.UINF, config.NX, config.NY, config.KAPPA
    spawn_square_size = config.SPAWN_SQUARE_SIZE
    midpoint = list(config.SPAWN_MIDPOINT)
    min_battery, max_battery = config.MIN_BATTERY, config.MAX_BATTERY

    walls = [xRange[0] + robot_rad, xRange[1] - robot_rad, yRange[1] - robot_rad, yRange[0] + robot_rad]
    min_dist = config.COLLISION_MIN_DIST_SLACK + 2.0 * robot_rad
    min_dist_initial = config.SPAWN_MIN_DIST_SLACK + 2.0 * robot_rad

    agents = _spawn_agents(n_agents, midpoint, spawn_square_size, min_dist_initial, max_battery, min_battery)

    vel = np.zeros((n_agents, 2))
    batteryEmpty = False
    heading_sum = 0.0
    step_counter = 0
    positions_log = [agents[:, 0:2].copy()] if record_trajectory else None
    battery_log = [agents[:, 3].copy()] if record_battery else None

    yVals, xVals, powerVals = RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny)
    if visualize:
        plot_all(ax, fig, agents, robot_rad, yVals, xVals, powerVals, t, v_out)

    r0, epsilon, k_align, k_goal, K1, K2, U = _resolve_rules(rules)
    r_cut, r_min, R_align = config.R_CUT, config.R_MIN, config.R_ALIGN

    while not batteryEmpty:
        vel[:, :] = _flocking_velocity_command(agents, n_agents, r0, epsilon, k_align, k_goal, K1, K2, U,
                                                r_cut, r_min, R_align)

        vel_actual, agents, xRange, collision_counter = move(agents, vel, dt, v_avg, n_agents, min_dist, walls, collision_counter)
        heading_sum += np.mean(np.cos(agents[:, 2] - np.pi / 2.0))
        step_counter += 1
        if record_trajectory:
            positions_log.append(agents[:, 0:2].copy())

        yVals, xVals, powerVals = RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny)
        F_drag = dragforce(agents, wind_rad, xVals, yVals, powerVals, n_agents, vel_actual, v_wind, kappa)
        agents, batt_drain = batterydrainage(agents, vel_actual, F_drag, robot_rad, dt)
        if record_battery:
            battery_log.append(agents[:, 3].copy())

        batteryEmpty = np.any(agents[:, 3] <= 0.0)
        t += dt
        if visualize:
            plot_all(ax, fig, agents, robot_rad, yVals, xVals, powerVals, t, v_out)

    average_batt = np.mean(agents[:, 3])
    dist_travelled = -np.mean(agents[:, 0])
    collision_time = collision_counter * dt
    eff = _fitness(dist_travelled, average_batt, collision_time, battery_aware)

    if visualize:
        v_out.release()
        plt.close()

    if record_trajectory or record_battery:
        telemetry = {
            "positions": np.array(positions_log) if record_trajectory else None,
            "battery": np.array(battery_log) if record_battery else None,
        }
        return eff, dist_travelled, average_batt, collision_counter, telemetry
    return eff, dist_travelled, average_batt, collision_counter

def _pybullet_wall_specs(yRange, robot_rad):
    """Two very long static walls bounding y only -- x is left open since agents
    migrate indefinitely in -x, exactly like the numpy backend."""
    half_length = config.PYBULLET_WALL_HALF_LENGTH
    half_thickness = config.PYBULLET_WALL_THICKNESS
    half_height = config.PYBULLET_WALL_HEIGHT / 2.0
    top_y = yRange[1] - robot_rad + half_thickness
    bottom_y = yRange[0] + robot_rad - half_thickness
    half_extents = [half_length, half_thickness, half_height]
    return [
        (half_extents, [0.0, top_y, half_height]),
        (half_extents, [0.0, bottom_y, half_height]),
    ]

def _pybullet_move(p, body_ids, agents, vel_cmd, n_agents, min_dist, walls, collision_counter, substeps):
    """Drop-in replacement for move(): drives each PyBullet body toward the commanded
    (linear speed, yaw rate) with a simple force/torque P-controller, steps real
    rigid-body dynamics + collisions for one outer dt, and returns state in the same
    (vel_actual, agents, xRange, collision_counter) shape/semantics move() does."""
    th = agents[:, 2] + np.pi / 2.0

    cur_lin = np.zeros((n_agents, 2))
    cur_ang = np.zeros(n_agents)
    for i, body_id in enumerate(body_ids):
        lin, ang = p.getBaseVelocity(body_id)
        cur_lin[i] = lin[0], lin[1]
        cur_ang[i] = ang[2]
    cur_speed = cur_lin[:, 0] * np.cos(th) + cur_lin[:, 1] * np.sin(th)

    force_mag = np.clip(config.PYBULLET_FORCE_GAIN * (vel_cmd[:, 0] - cur_speed),
                         -config.PYBULLET_MAX_FORCE, config.PYBULLET_MAX_FORCE)
    torque = np.clip(config.PYBULLET_TORQUE_GAIN * (vel_cmd[:, 1] - cur_ang),
                      -config.PYBULLET_MAX_TORQUE, config.PYBULLET_MAX_TORQUE)
    fx = force_mag * np.cos(th)
    fy = force_mag * np.sin(th)

    for _ in range(substeps):
        for i, body_id in enumerate(body_ids):
            pos, _ = p.getBasePositionAndOrientation(body_id)
            p.applyExternalForce(body_id, -1, [fx[i], fy[i], 0.0], pos, p.WORLD_FRAME)
            p.applyExternalTorque(body_id, -1, [0.0, 0.0, torque[i]], p.WORLD_FRAME)
        p.stepSimulation()

    for i, body_id in enumerate(body_ids):
        pos, orn = p.getBasePositionAndOrientation(body_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        agents[i, 0], agents[i, 1] = pos[0], pos[1]
        agents[i, 2] = wrap_to_pi(yaw - np.pi / 2.0)

    vel_actual = np.zeros((n_agents, 3))
    for i, body_id in enumerate(body_ids):
        lin, ang = p.getBaseVelocity(body_id)
        speed = np.hypot(lin[0], lin[1])
        vel_actual[i, 0] = speed
        vel_actual[i, 1] = ang[2]
        vel_actual[i, 2] = 0.0 if speed < 1e-9 else np.arctan2(lin[1], lin[0]) - np.pi / 2.0

    xRange, collision_counter, _ = _collision_and_window_bookkeeping(
        agents, n_agents, min_dist, walls, collision_counter)

    return vel_actual, agents, xRange, collision_counter

def _open_pybullet_video_writer(visualize):
    """cv2-only counterpart to _open_video_writer(): PyBullet frames are rendered by
    PyBullet's own rasterizer, not matplotlib, so no figure/backend juggling is needed."""
    if not visualize:
        return None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(config.PYBULLET_VIDEO_PATH, fourcc, config.VIDEO_FPS, config.VIDEO_SIZE)

def _pybullet_camera_matrices(p, com_x, com_y, width, height):
    distance = config.VIDEO_VIEWPORT_HALF_WIDTH * config.PYBULLET_VIDEO_DISTANCE_FACTOR
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[com_x, com_y, 0.0], distance=distance,
        yaw=config.PYBULLET_VIDEO_YAW, pitch=config.PYBULLET_VIDEO_PITCH, roll=0.0, upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(
        fov=config.PYBULLET_VIDEO_FOV, aspect=width / height, nearVal=0.1, farVal=distance * 3.0)
    return view, proj

def _world_to_screen(view, proj, width, height, x, y, z=0.0):
    """Project a world-space point through PyBullet's (OpenGL-style, column-major) view
    and projection matrices to pixel coordinates, for drawing heading arrows on the frame."""
    view_m = np.array(view, dtype=np.float64).reshape(4, 4, order='F')
    proj_m = np.array(proj, dtype=np.float64).reshape(4, 4, order='F')
    clip = proj_m @ (view_m @ np.array([x, y, z, 1.0]))
    if clip[3] <= 1e-9:
        return None
    ndc = clip[:3] / clip[3]
    sx = (ndc[0] * 0.5 + 0.5) * width
    sy = (1.0 - (ndc[1] * 0.5 + 0.5)) * height
    return sx, sy

def _pybullet_render_frame(p, body_ids, agents, t, video_writer):
    width, height = config.VIDEO_SIZE
    com_x, com_y = np.mean(agents[:, 0]), np.mean(agents[:, 1])
    view, proj = _pybullet_camera_matrices(p, com_x, com_y, width, height)

    for i, body_id in enumerate(body_ids):
        norm_b = np.clip(agents[i, 3] / config.MAX_BATTERY, 0.0, 1.0)
        p.changeVisualShape(body_id, -1, rgbaColor=[1.0 - norm_b, norm_b, 0.0, 1.0])

    _, _, rgba, _, _ = p.getCameraImage(width, height, viewMatrix=view, projectionMatrix=proj,
                                         renderer=p.ER_TINY_RENDERER)
    frame = np.reshape(np.array(rgba, dtype=np.uint8), (height, width, 4))[:, :, :3]
    frame_bgr = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    arrow_len = config.VIDEO_ARROW_LEN
    for i in range(agents.shape[0]):
        theta = agents[i, 2]
        p0 = _world_to_screen(view, proj, width, height, agents[i, 0], agents[i, 1])
        p1 = _world_to_screen(view, proj, width, height,
                               agents[i, 0] - arrow_len * np.sin(theta),
                               agents[i, 1] + arrow_len * np.cos(theta))
        if p0 is not None and p1 is not None:
            cv2.arrowedLine(frame_bgr, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                             (0, 0, 255), 2, tipLength=0.3)

    cv2.putText(frame_bgr, f"t = {t:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    video_writer.write(frame_bgr)

def _simulate_pybullet(rules, seed, visualize, n_agents, record_trajectory, record_battery, battery_aware):
    try:
        import pybullet as p
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "backend='pybullet' requires the pybullet package: pip install pybullet"
        ) from exc

    if seed is not None:
        np.random.seed(seed)

    v_out = _open_pybullet_video_writer(visualize)

    dt = config.DT
    n_agents = n_agents if n_agents is not None else config.N_AGENTS
    robot_rad = config.ROBOT_RAD
    wind_rad = config.WIND_RAD
    xRange = list(config.X_RANGE)
    yRange = list(config.Y_RANGE)
    v_wind = config.V_WIND
    t = 0.0
    collision_counter = 0

    Uinf, Nx, Ny, kappa = config.UINF, config.NX, config.NY, config.KAPPA
    spawn_square_size = config.SPAWN_SQUARE_SIZE
    midpoint = list(config.SPAWN_MIDPOINT)
    min_battery, max_battery = config.MIN_BATTERY, config.MAX_BATTERY

    walls = [xRange[0] + robot_rad, xRange[1] - robot_rad, yRange[1] - robot_rad, yRange[0] + robot_rad]
    min_dist = config.COLLISION_MIN_DIST_SLACK + 2.0 * robot_rad
    min_dist_initial = config.SPAWN_MIN_DIST_SLACK + 2.0 * robot_rad

    agents = _spawn_agents(n_agents, midpoint, spawn_square_size, min_dist_initial, max_battery, min_battery)

    client = p.connect(p.GUI if (visualize and config.PYBULLET_USE_GUI) else p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        substeps = max(1, round(dt / config.PYBULLET_INTERNAL_TIMESTEP))
        p.setTimeStep(dt / substeps, physicsClientId=client)

        ground_col = p.createCollisionShape(p.GEOM_PLANE)
        ground_id = p.createMultiBody(0, ground_col)
        p.changeDynamics(ground_id, -1, lateralFriction=config.PYBULLET_GROUND_FRICTION)

        for half_extents, position in _pybullet_wall_specs(yRange, robot_rad):
            wall_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            p.createMultiBody(0, wall_col, basePosition=position)

        body_z = config.PYBULLET_BODY_HEIGHT / 2.0
        col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=robot_rad, height=config.PYBULLET_BODY_HEIGHT)
        body_ids = []
        for i in range(n_agents):
            body_id = p.createMultiBody(
                baseMass=config.PYBULLET_MASS,
                baseCollisionShapeIndex=col_shape,
                basePosition=[agents[i, 0], agents[i, 1], body_z],
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, agents[i, 2] + np.pi / 2.0]),
            )
            p.changeDynamics(body_id, -1,
                              lateralFriction=config.PYBULLET_BODY_FRICTION,
                              linearDamping=config.PYBULLET_LINEAR_DAMPING,
                              angularDamping=config.PYBULLET_ANGULAR_DAMPING,
                              restitution=config.PYBULLET_RESTITUTION)
            body_ids.append(body_id)

        batteryEmpty = False
        heading_sum = 0.0
        step_counter = 0
        positions_log = [agents[:, 0:2].copy()] if record_trajectory else None
        battery_log = [agents[:, 3].copy()] if record_battery else None

        yVals, xVals, powerVals = RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny)
        if visualize:
            _pybullet_render_frame(p, body_ids, agents, t, v_out)

        r0, epsilon, k_align, k_goal, K1, K2, U = _resolve_rules(rules)
        r_cut, r_min, R_align = config.R_CUT, config.R_MIN, config.R_ALIGN

        while not batteryEmpty:
            vel_cmd = _flocking_velocity_command(agents, n_agents, r0, epsilon, k_align, k_goal, K1, K2, U,
                                                  r_cut, r_min, R_align)

            vel_actual, agents, xRange, collision_counter = _pybullet_move(
                p, body_ids, agents, vel_cmd, n_agents, min_dist, walls, collision_counter, substeps)
            heading_sum += np.mean(np.cos(agents[:, 2] - np.pi / 2.0))
            step_counter += 1
            if record_trajectory:
                positions_log.append(agents[:, 0:2].copy())

            yVals, xVals, powerVals = RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny)
            F_drag = dragforce(agents, wind_rad, xVals, yVals, powerVals, n_agents, vel_actual, v_wind, kappa)
            agents, batt_drain = batterydrainage(agents, vel_actual, F_drag, robot_rad, dt)
            if record_battery:
                battery_log.append(agents[:, 3].copy())

            batteryEmpty = np.any(agents[:, 3] <= 0.0)
            t += dt
            if visualize:
                _pybullet_render_frame(p, body_ids, agents, t, v_out)

        average_batt = np.mean(agents[:, 3])
        dist_travelled = -np.mean(agents[:, 0])
        collision_time = collision_counter * dt
        eff = _fitness(dist_travelled, average_batt, collision_time, battery_aware)
    finally:
        p.disconnect(physicsClientId=client)

    if visualize:
        v_out.release()

    if record_trajectory or record_battery:
        telemetry = {
            "positions": np.array(positions_log) if record_trajectory else None,
            "battery": np.array(battery_log) if record_battery else None,
        }
        return eff, dist_travelled, average_batt, collision_counter, telemetry
    return eff, dist_travelled, average_batt, collision_counter

def simulation_free_global_mod_2_LJ(rules=None, seed=None, visualize=False, n_agents=None,
                                     record_trajectory=False, record_battery=False, backend="numpy",
                                     battery_aware=True):
    """backend: "numpy" (default) -- the original kinematic port, faithful to the MATLAB
    reference. "pybullet" -- agents are real rigid bodies (mass/friction/collision) driven
    by a force/torque controller chasing the same (u, w) control law; see the PYBULLET_*
    constants in config.py. The two backends are not expected to produce matching numbers --
    gains evolved against one will likely need re-tuning against the other.

    record_trajectory / record_battery: if either is True, a 5th return value is added --
    a dict {"positions": (n_steps, n_agents, 2) array or None, "battery": (n_steps, n_agents)
    array or None} -- instead of the usual 4-tuple.

    battery_aware: if False, the returned `eff` (and therefore whatever a CMA-ES run
    optimizes against) drops the average_batt/EFF_BATTERY_WEIGHT term entirely -- useful
    for evolving a "doesn't care about battery" baseline to compare against. dist_travelled/
    average_batt/collision_counter are always computed the same way regardless; only the
    scalar fitness formula changes."""
    if backend == "numpy":
        return _simulate_numpy(rules, seed, visualize, n_agents, record_trajectory, record_battery, battery_aware)
    elif backend == "pybullet":
        return _simulate_pybullet(rules, seed, visualize, n_agents, record_trajectory, record_battery, battery_aware)
    else:
        raise ValueError(f"Unknown backend '{backend}' -- expected 'numpy' or 'pybullet'.")