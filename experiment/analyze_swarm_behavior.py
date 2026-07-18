import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Import your simulation module to access its underlying physics steps directly
import simulation_free_global_mod_2_LJ as sim_mod

def harvest_telemetry(genome, n_agents, seed):
    """
    Executes a high-resolution playback run of your physics engine using the 
    optimized genome, harvesting time-series telemetry data at every step.
    """
    np.random.seed(seed)
    
    # Configure environmental and physical constraints matching your engine
    dt = 0.5
    robot_rad = 0.055
    wind_rad = 0.15
    xRange, yRange = [-5.0, 5.0], [-5.0, 5.0]
    v_avg, v_wind, kappa = 0.1, 10.0, 10.0
    Uinf, Nx, Ny = 100.0, 200, 200
    spawn_square_size = 3.0
    midpoint = [0.0, 0.0]
    walls = [-5.0 + robot_rad, 5.0 - robot_rad, 5.0 - robot_rad, -5.0 + robot_rad]
    min_dist = 0.01 + 2.0 * robot_rad
    
    # Assemble agent arrays
    agents = np.random.rand(n_agents, 4)
    agents[:, 0] = midpoint[0] + (agents[:, 0] - 0.5) * spawn_square_size
    agents[:, 1] = midpoint[1] + (agents[:, 1] - 0.5) * spawn_square_size
    agents[:, 2] = sim_mod.wrap_to_pi(agents[:, 2] * 2.0 * np.pi)
    agents[0:n_agents-1, 3] = 100.0  # Max battery
    agents[-1, 3] = 100.0            # Min battery configuration
    
    # Unpack rules from the optimized genome array
    rules = {
        'r0': genome[0], 'epsilon': genome[1], 'k_align': genome[2],
        'k_goal': genome[3], 'K1': genome[4], 'K2': genome[5], 'U': genome[6]
    }
    sigma = rules['r0'] / np.sqrt(2.0)
    
    # Storage arrays for time-series tracking
    history = {
        "time": [], "order": [], "cohesion": [], "battery": [], "collisions": []
    }
    
    t = 0.0
    collision_counter = 0
    vel = np.zeros((n_agents, 2))
    
    # Run the simulation loop until an agent's battery completely empties
    while not np.any(agents[:, 3] <= 0.0):
        X = agents[:, 0:2]
        th = agents[:, 2] + np.pi / 2.0
        
        # --- Metric Calculation: Global Order Parameter (Psi) ---
        unit_vectors_x = -np.sin(agents[:, 2])
        unit_vectors_y = np.cos(agents[:, 2])
        mean_v_x = np.mean(unit_vectors_x)
        mean_v_y = np.mean(unit_vectors_y)
        psi = np.sqrt(mean_v_x**2 + mean_v_y**2)
        
        # --- Metric Calculation: Swarm Cohesion ---
        D_matrix = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        pairwise_dists = D_matrix[np.triu_indices(n_agents, k=1)]
        avg_cohesion = np.mean(pairwise_dists)
        
        # Log telemetry step
        history["time"].append(t)
        history["order"].append(psi)
        history["cohesion"].append(avg_cohesion)
        history["battery"].append(np.mean(agents[:, 3]))
        history["collisions"].append(collision_counter)
        
        # --- Core Physics Engine Progression Steps ---
        Dx = np.subtract.outer(X[:, 0], X[:, 0])
        Dy = np.subtract.outer(X[:, 1], X[:, 1])
        R2 = Dx**2 + Dy**2 + np.eye(n_agents)
        R = np.sqrt(R2)
        
        ex, ey = Dx / R, Dy / R
        cos_th_mat = np.tile(np.cos(th)[:, None], (1, n_agents))
        sin_th_mat = np.tile(np.sin(th)[:, None], (1, n_agents))
        exr = ex * cos_th_mat + ey * sin_th_mat
        eyr = ey * cos_th_mat - ex * sin_th_mat
        
        mask = (R > 0.0) & (R < 3.0)
        np.fill_diagonal(mask, False)
        
        sig_over_r6 = (sigma**2) / (R**2 + 1e-6)
        sig_over_r12 = sig_over_r6**2
        Fmag = 8.0 * rules['epsilon'] * (2.0 * sig_over_r12 - sig_over_r6) / (R + 1e-6) * mask
        
        Fp_x = np.sum(Fmag * exr, axis=1)
        Fp_y = np.sum(Fmag * eyr, axis=1)
        
        align_mask = (R < 1.5) & (~np.eye(n_agents, dtype=bool))
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
        
        Fg_x = -1.0 * np.cos(agents[:, 2])
        Fg_y = -1.0 * -np.sin(agents[:, 2])
        
        F_x = Fp_x + rules['k_align'] * Fa_x + rules['k_goal'] * Fg_x
        F_y = Fp_y + rules['k_align'] * Fa_y + rules['k_goal'] * Fg_y
        
        vel[:, 0] = np.clip(rules['K1'] * F_x + rules['U'], -0.2, 0.2)
        vel[:, 1] = np.clip(rules['K2'] * F_y, -np.pi / 5.0, np.pi / 5.0)
        
        vel_actual, agents, xRange, collision_counter = sim_mod.move(
            agents, vel, dt, v_avg, n_agents, min_dist, walls, collision_counter
        )
        
        yVals, xVals, powerVals = sim_mod.RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny)
        F_drag = sim_mod.dragforce(agents, wind_rad, xVals, yVals, powerVals, n_agents, vel_actual, v_wind, kappa)
        agents, _ = sim_mod.batterydrainage(agents, vel_actual, F_drag, robot_rad, dt)
        
        t += dt
        if t > 500:  # Timeout boundary safety cap
            break
            
    return history

if __name__ == "__main__":
    summary_json_path = "optimization_results/master_optimization_summary.json"
    if not os.path.exists(summary_json_path):
        raise FileNotFoundError("Optimization logs missing. Wait for 'run_batch_optimization.py' to finish.")
        
    with open(summary_json_path, "r") as f:
        trials_list = json.load(f)
        
    print("📊 Ingesting optimized genomes and harvesting swarm behavior metrics...")
    
    # Structural dictionary setup for plotting
    config_groups = {"small_scale": [], "baseline": [], "large_scale": []}
    colors = {"small_scale": "#D95319", "baseline": "#0072BD", "large_scale": "#7E2F8E"}
    
    for run in trials_list:
        cfg = run["config"]
        trial = run["trial"]
        seed = run["seed"]
        genome = run["genome"]
        
        print(f"   ↳ Processing Timeline: Configuration [{cfg.upper()}] | Trial {trial:02d}")
        telemetry = harvest_telemetry(genome, run["agents"], seed)
        config_groups[cfg].append(telemetry)
        
    # Set up the visualization subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Evolved Swarm Behavior Analysis & Scalability Profile", fontsize=14, fontweight="bold")
    
    for cfg, runs in config_groups.items():
        c = colors[cfg]
        
        # Plot individual trials as thin lines to visualize variance
        for run in runs:
            axes[0, 0].plot(run["time"], run["order"], color=c, alpha=0.15)
            axes[0, 1].plot(run["time"], run["cohesion"], color=c, alpha=0.15)
            axes[1, 0].plot(run["time"], run["battery"], color=c, alpha=0.15)
            axes[1, 1].plot(run["time"], run["collisions"], color=c, alpha=0.15)
            
        # Compute interpolations to plot clean, unified average trajectory lines
        max_time_cap = max([max(r["time"]) for r in runs])
        common_time = np.linspace(0, max_time_cap, 200)
        
        interp_order, interp_cohesion, interp_batt, interp_coll = [], [], [], []
        for run in runs:
            interp_order.append(np.interp(common_time, run["time"], run["order"]))
            interp_cohesion.append(np.interp(common_time, run["time"], run["cohesion"]))
            interp_batt.append(np.interp(common_time, run["time"], run["battery"]))
            interp_coll.append(np.interp(common_time, run["time"], run["collisions"]))
            
        # Draw bold mean lines
        axes[0, 0].plot(common_time, np.mean(interp_order, axis=0), color=c, linewidth=2.5, label=cfg)
        axes[0, 1].plot(common_time, np.mean(interp_cohesion, axis=0), color=c, linewidth=2.5)
        axes[1, 0].plot(common_time, np.mean(interp_batt, axis=0), color=c, linewidth=2.5)
        axes[1, 1].plot(common_time, np.mean(interp_coll, axis=0), color=c, linewidth=2.5)

    # Polish and clear up the figure grids
    axes[0, 0].set_title("Flock Alignment Vector Evolution")
    axes[0, 0].set_ylabel("Order Parameter (\u03c8)")
    axes[0, 0].legend(loc="lower right")
    
    axes[0, 1].set_title("Inter-Agent Spatial Cohesion")
    axes[0, 1].set_ylabel("Average Separation Distance (m)")
    
    axes[1, 0].set_title("Swarm Battery Consumption Slope")
    axes[1, 0].set_ylabel("Mean Battery State (%)")
    axes[1, 0].set_xlabel("Simulation Runtime (s)")
    
    axes[1, 1].set_title("Cumulative Collision Timeline")
    axes[1, 1].set_ylabel("Total Unsafe Contact Impacts")
    axes[1, 1].set_xlabel("Simulation Runtime (s)")
    
    for ax in axes.flat:
        ax.grid(True, linestyle=":", alpha=0.6)
        
    plt.tight_layout()
    output_fig = "swarm_behavior_analysis.png"
    plt.savefig(output_fig, dpi=300)
    print(f"\n🎉 Analysis complete! Combined behavior plots saved as '{output_fig}'.")
    plt.show()