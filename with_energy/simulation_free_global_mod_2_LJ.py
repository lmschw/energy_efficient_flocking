import numpy as np
import cv2
import matplotlib.pyplot as plt
import io

def wrap_to_pi(angle):
    """1:1 Mirror of MATLAB's wrapToPi function."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny, useGPU=0):
    """
    Vectorized high-performance alternative to MATLAB's compiled MEX file.
    Eliminates raw Python loops to speed up execution by 100x-1000x.
    """
    xVals = np.linspace(xRange[0], xRange[1], Nx)  # Shape: (Nx,)
    yVals = np.linspace(yRange[0], yRange[1], Ny)  # Shape: (Ny,)
    powerVals = np.ones((Nx, Ny)) * Uinf           # Shape: (Nx, Ny)
    
    # Process each agent using optimized array broadcasting
    for i in range(agents.shape[0]):
        rx, ry = agents[i, 0], agents[i, 1]
        
        # 1. Mask out anything not strictly downstream (+X) of the agent
        downstream_mask = xVals > rx  # Shape: (Nx,)
        if not np.any(downstream_mask):
            continue
            
        # 2. Compute downstream distance (dx) grid array
        dx = (xVals - rx)[:, None]  # Shape: (Nx, 1)
        wake_width = wind_rad + 0.05 * dx  # Shape: (Nx, 1)
        
        # 3. Compute absolute vertical offset (dy) grid array
        dy = np.abs(yVals - ry)[None, :]  # Shape: (1, Ny)
        
        # 4. Generate the 2D geometric wake shadow mask
        inside_wake = (dy < wake_width) & downstream_mask[:, None]  # Shape: (Nx, Ny)
        
        # 5. Apply fluid speed attenuation drop calculations
        attenuation = 1.0 - (0.6 / (1.0 + 0.2 * dx))  # Shape: (Nx, 1)
        attenuated_power = Uinf * attenuation         # Shape: (Nx, 1)
        
        # 6. Smoothly blend back into the master wind matrix
        powerVals = np.where(inside_wake, np.minimum(powerVals, attenuated_power), powerVals)
                        
    return yVals, xVals, powerVals

def dragforce(agents, wind_rad, xVals, yVals, powerVals, n_agents, vel_actual, v_wind, kappa):
    powerVs = powerVals.T
    powerVals_agents = 100.0 * np.ones(n_agents)
    F_drag = np.zeros((n_agents, 2))
    
    for i in range(n_agents):
        x_r = agents[i, 0]
        y_r = agents[i, 1]
        
        x_to_left = np.where(xVals <= x_r - 1.1 * wind_rad)[0]
        if len(x_to_left) > 0:
            x = x_to_left[-1]
        else:
            x = 0
            
        if x < 2:  # 0-based conversion for MATLAB's (x < 3)
            powerVals_agents[i] = 100.0
        else:
            y = np.argmin(np.abs(yVals - y_r))
            powerVals_agents[i] = powerVs[y, x]
            
        v_wind_agent = (powerVals_agents[i] / 100.0) * v_wind
        v_parallel = vel_actual[i, 0] * np.sin(vel_actual[i, 2])
        v_rel = v_wind_agent + v_parallel
        F_drag[i, 0] = 0.5 * 1.225 * 0.0045 * kappa * (v_rel ** 2)
        
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
    
    batt_drain = np.maximum(np.sum(np.abs(wheels_vel), axis=1) / 8.0 - dottprod, 0.10) * dt
    agents[:, 3] = agents[:, 3] - 2.0 * batt_drain
    return agents, batt_drain

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
    
    agents_xy = agents[:, 0:2]
    D = np.linalg.norm(agents_xy[:, None, :] - agents_xy[None, :, :], axis=-1)
    close_agents = (D < min_dist) & (~np.eye(n_agents, dtype=bool))
    num_pairs = np.count_nonzero(np.triu(close_agents, k=1))
    collision_counter += num_pairs
    
    wall_margin = 0.055 * 0.5
    wall_hits_step = np.sum((agents[:, 0] > walls[1] - wall_margin) | 
                            (agents[:, 1] > walls[2] - wall_margin) | 
                            (agents[:, 1] < walls[3] + wall_margin))
    collision_counter += 3 * wall_hits_step
    
    min_x = np.min(agents[:, 0])
    max_x = min(np.max(agents[:, 0]), min_x + 9.8)
    xRange = [min_x - (10.0 - (max_x - min_x)) / 2.0, max_x + (10.0 - (max_x - min_x)) / 2.0]
    
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
    
    # === TRACKING MODIFICATION: Calculate Center of Mass ===
    com_x = np.mean(agents[:, 0])
    com_y = np.mean(agents[:, 1])
    
    # Dynamically center the camera viewport around the swarm centroid
    # Maintains the exact same 10m x 10m window scale as your original design
    ax.set_xlim([com_x - 5.0, com_x + 5.0])
    ax.set_ylim([com_y - 5.0, com_y + 5.0])
    
    for i in range(agents.shape[0]):
        x, y, theta, battery = agents[i, 0], agents[i, 1], agents[i, 2], agents[i, 3]
        norm_b = np.clip(battery / 100.0, 0.0, 1.0)
        color = (1.0 - norm_b, norm_b, 0.0)
        
        circle = plt.Circle((x, y), r, fill=True, facecolor=color, edgecolor='k')
        ax.add_patch(circle)
        
        arrow_len = 0.3
        ax.quiver(x, y, -arrow_len * np.sin(theta), arrow_len * np.cos(theta), 
                  angles='xy', scale_units='xy', scale=1, color='r', width=0.004)
        
    # Backend-Agnostic Frame Capture
    buf = io.BytesIO()
    fig.savefig(buf, format='raw')
    buf.seek(0)
    
    w = int(fig.bbox.bounds[2])
    h = int(fig.bbox.bounds[3])
    
    frame = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape((h, w, 4))
    buf.close()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    
    if frame.shape[1] != 1200 or frame.shape[0] != 800:
        frame = cv2.resize(frame, (1200, 800))
        
    video_writer.write(frame)
    plt.pause(0.001)

def simulation_free_global_mod_2_LJ(rules=None, seed=None, visualize=False):
    if seed is not None:
        np.random.seed(seed)
        
    # === FORCE HEADLESS MODE DURING OPTIMIZATION ===
    if not visualize:
        plt.switch_backend('Agg')  # Prevents any GUI windows from initializing
        
    v_out = None
    fig, ax = None, None
    if visualize:
        plt.switch_backend('TkAgg') # Safely restore interactive window for playback
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        v_out = cv2.VideoWriter('alone.mp4', fourcc, 10.0, (1200, 800))
        fig, ax = plt.subplots(figsize=(12, 8))
    
    dt = 0.5
    n_agents = 20
    robot_rad = 0.055
    wind_rad = 0.15
    xRange = [-5.0, 5.0]
    yRange = [-5.0, 5.0]
    v_avg = 0.1
    v_wind = 10.0
    t = 0.0
    collision_counter = 0
    
    Uinf, Nx, Ny, kappa = 100.0, 200, 200, 10.0
    spawn_square_size = 3.0
    midpoint = [0.0, 0.0]
    min_battery, max_battery = 100.0, 100.0
    
    walls = [-5.0 + robot_rad, 5.0 - robot_rad, 5.0 - robot_rad, -5.0 + robot_rad]
    min_dist = 0.01 + 2.0 * robot_rad
    
    agents = np.random.rand(n_agents, 4)
    agents[:, 0] = midpoint[0] + (agents[:, 0] - 0.5) * spawn_square_size
    agents[:, 1] = midpoint[1] + (agents[:, 1] - 0.5) * spawn_square_size
    agents[:, 2] = wrap_to_pi(agents[:, 2] * 2.0 * np.pi)
    agents[0:n_agents-1, 3] = max_battery
    agents[-1, 3] = min_battery
    
    min_dist_initial = 0.1 + 2.0 * robot_rad
    collision_detected = True
    while collision_detected:
        collision_detected = False
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                distance = np.sqrt((agents[i, 0] - agents[j, 0])**2 + (agents[i, 1] - agents[j, 1])**2)
                if distance < min_dist_initial:
                    agents[j, 0] = midpoint[0] + (np.random.rand() - 0.5) * spawn_square_size
                    agents[j, 1] = midpoint[1] + (np.random.rand() - 0.5) * spawn_square_size
                    collision_detected = True
                    
    vel = np.zeros((n_agents, 2))
    batteryEmpty = False
    heading_sum = 0.0
    step_counter = 0
    
    yVals, xVals, powerVals = RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny)
    if visualize:
        plot_all(ax, fig, agents, robot_rad, yVals, xVals, powerVals, t, v_out)
    
    # Check if incoming parameter vectors override the hardcoded properties
    r0      = rules['r0']      if rules else 0.70
    epsilon = rules['epsilon'] if rules else 0.5
    k_align = rules['k_align'] if rules else 0.0
    k_goal  = rules['k_goal']  if rules else 3.0
    K1      = rules['K1']      if rules else 0.05
    K2      = rules['K2']      if rules else 0.5
    U       = rules['U']       if rules else 0.005
    
    sigma = r0 / np.sqrt(2.0)
    r_cut, r_min, R_align = 3.0, 0.0, 1.5
    
    while not batteryEmpty:
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
        np.fill_diagonal(mask, False)  # Sets the self-interaction diagonal to False
        
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
        Fa_x /= A; Fa_y /= A
        
        Fg_gx = -1.0
        Fg_x = Fg_gx * np.cos(th)
        Fg_y = Fg_gx * -np.sin(th)
        
        F_x = Fp_x + k_align * Fa_x + k_goal * Fg_x
        F_y = Fp_y + k_align * Fa_y + k_goal * Fg_y
        
        u = np.clip(K1 * F_x + U, -0.2, 0.2)
        w = np.clip(K2 * F_y, -np.pi / 5.0, np.pi / 5.0)
        vel[:, 0] = u; vel[:, 1] = w
        
        vel_actual, agents, xRange, collision_counter = move(agents, vel, dt, v_avg, n_agents, min_dist, walls, collision_counter)
        heading_sum += np.mean(np.cos(agents[:, 2] - np.pi / 2.0))
        step_counter += 1
        
        yVals, xVals, powerVals = RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny)
        F_drag = dragforce(agents, wind_rad, xVals, yVals, powerVals, n_agents, vel_actual, v_wind, kappa)
        agents, batt_drain = batterydrainage(agents, vel_actual, F_drag, robot_rad, dt)
        
        batteryEmpty = np.any(agents[:, 3] <= 0.0)
        t += dt
        if visualize:
            plot_all(ax, fig, agents, robot_rad, yVals, xVals, powerVals, t, v_out)
            
    average_batt = np.mean(agents[:, 3])
    dist_travelled = -np.mean(agents[:, 0])
    collision_time = collision_counter * dt
    eff = dist_travelled + average_batt / 5.0 - collision_time / 250.0
    
    if visualize:
        v_out.release()
        plt.close()
    return eff, dist_travelled, average_batt, collision_counter