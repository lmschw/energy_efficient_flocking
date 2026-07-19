import argparse
import numpy as np
from experiment import config
from experiment.battery_plot import plot_battery_levels
from experiment.simulation_free_global_mod_2_LJ import simulation_free_global_mod_2_LJ


def load_rules(genome_path):
    try:
        flat_genome = np.load(genome_path)
        print(f"Loaded optimized genome rules from '{genome_path}'")
        return config.genome_to_rules(flat_genome)
    except FileNotFoundError:
        print(f"No genome found at '{genome_path}'. Falling back to MATLAB baseline gains.")
        return dict(config.DEFAULT_RULES)


def run_live_visualization(genome_path=config.OPTIMIZE_GENOME_OUT_PATH, seed=config.OPTIMIZE_SEED,
                            backend="numpy", n_agents=None, battery_plot_path=None):
    rules = load_rules(genome_path)
    print("Rules:", {k: round(float(v), 4) for k, v in rules.items()})
    print(f"Backend: {backend}")

    result = simulation_free_global_mod_2_LJ(
        rules=rules, seed=seed, visualize=True, backend=backend, n_agents=n_agents,
        record_battery=battery_plot_path is not None
    )
    if battery_plot_path is not None:
        eff, dist_travelled, average_batt, collisions, telemetry = result
    else:
        eff, dist_travelled, average_batt, collisions = result

    print(f"Playback complete -- efficiency={eff:.4f}, "
          f"distance={dist_travelled:.4f}, avg_battery={average_batt:.4f}, "
          f"collisions={collisions}")
    video_path = config.PYBULLET_VIDEO_PATH if backend == "pybullet" else config.VIDEO_PATH
    print(f"Video saved to '{video_path}'.")

    if battery_plot_path is not None:
        plot_battery_levels(telemetry["battery"], config.DT, battery_plot_path,
                             title=f"Battery Level per Agent ({backend} backend)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play back a swarm run for a saved genome.")
    parser.add_argument("genome_path", nargs="?", default="optimization_results_pybullet/opt_small_scale_trial_02_seed_123_gains.npy",
                         help="Path to a .npy file with the 7 LJ rule gains "
                              "(r0, epsilon, k_align, k_goal, K1, K2, U).")
    parser.add_argument("--seed", type=int, default=config.OPTIMIZE_SEED, help="Random seed for the run.")
    parser.add_argument("--backend", choices=["numpy", "pybullet"], default="numpy",
                         help="Physics backend: 'numpy' (kinematic, matches MATLAB) or "
                              "'pybullet' (real rigid-body dynamics).")
    parser.add_argument("--n-agents", type=int, default=None, help="Override the number of agents.")
    parser.add_argument("--plot-battery", nargs="?", const="battery_levels.png", default=None,
                         metavar="PATH",
                         help="Also save a per-agent battery-vs-time plot (default: battery_levels.png).")
    args = parser.parse_args()

    run_live_visualization(args.genome_path, args.seed, backend=args.backend, n_agents=args.n_agents,
                            battery_plot_path=args.plot_battery)
