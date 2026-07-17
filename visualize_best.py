import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from without_energy.simulation_without_energy import AdvancedThymioSwarmEnv as env_without_battery
from with_energy.simulation_with_energy import AdvancedThymioSwarmEnv as env_with_battery

def run_live_visualization(genome_path="winning_abcd_rules.npy", uses_energy=False):
    if uses_energy:
        NUM_SENSORS = 7
        NUM_INPUTS = NUM_SENSORS + 1 + 1  # Updated to 9 to match the energy expansion!
        NUM_OUTPUTS = 2
        MATRIX_SIZE = NUM_OUTPUTS * NUM_INPUTS  # Now 18 parameters per rule matrix (72 total)
    else:
        NUM_SENSORS = 7
        NUM_INPUTS = NUM_SENSORS + 1  # 8
        NUM_OUTPUTS = 2
        MATRIX_SIZE = NUM_OUTPUTS * NUM_INPUTS  # 16
    
    try:
        flat_genome = np.load(genome_path)
        print(f"Loaded optimized genome rules from '{genome_path}'")
    except FileNotFoundError:
        print("⚠️ No trained genome found. Running with random initialization.")
        flat_genome = np.random.randn(MATRIX_SIZE * 4)

    A = flat_genome[0:MATRIX_SIZE].reshape(NUM_OUTPUTS, NUM_INPUTS)
    B = flat_genome[MATRIX_SIZE:2*MATRIX_SIZE].reshape(NUM_OUTPUTS, NUM_INPUTS)
    C = flat_genome[2*MATRIX_SIZE:3*MATRIX_SIZE].reshape(NUM_OUTPUTS, NUM_INPUTS)
    D = flat_genome[3*MATRIX_SIZE:4*MATRIX_SIZE].reshape(NUM_OUTPUTS, NUM_INPUTS)

    # Spawn the swarm in an open, unbounded universe
    if uses_energy:
        env = env_with_battery(num_robots=15, arena_size=10.0, num_sensors=NUM_SENSORS)
    else:
        env = env_without_battery(num_robots=15, arena_size=10.0, num_sensors=NUM_SENSORS)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Polimi Plastic Swarm - Infinite Plane Tracking", fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)

    # UI Elements
    robot_bodies = ax.scatter(env.positions[:, 0], env.positions[:, 1], 
                              s=250, color='crimson', edgecolors='black', zorder=3)
    quiver = ax.quiver(env.positions[:, 0], env.positions[:, 1], 
                       np.cos(env.headings), np.sin(env.headings), 
                       color='darkblue', scale=22, zorder=4)

    def update_frame(frame_idx):
        env.step(A, B, C, D)
        
        # Update physical drawing locations
        robot_bodies.set_offsets(env.positions)
        quiver.set_offsets(env.positions)
        quiver.set_UVC(np.cos(env.headings), np.sin(env.headings))
        
        # 1:1 MATLAB TRACKING CAMERA: Center the camera viewport around the moving swarm
        center_x = np.mean(env.positions[:, 0])
        center_y = np.mean(env.positions[:, 1])
        
        # Maintain a dynamic 6x6 meter viewport centered on the flock
        ax.set_xlim(center_x - 3.0, center_x + 3.0)
        ax.set_ylim(center_y - 3.0, center_y + 3.0)
        
        return robot_bodies, quiver

    ani = animation.FuncAnimation(fig, update_frame, frames=2000, interval=40, blit=False)
    plt.show()
    return ani

if __name__ == "__main__":
    main_animation = run_live_visualization()