import argparse
import numpy as np
from experiment.simulation_free_global_mod_2_LJ import simulation_free_global_mod_2_LJ

# Baseline gains from the original MATLAB script (experiment/optimize.py's initial_guess),
# used as a fallback when no optimized genome file is available.
BASELINE_GAINS = [0.70, 0.5, 0.0, 3.0, 0.05, 0.5, 0.005]
GAIN_NAMES = ["r0", "epsilon", "k_align", "k_goal", "K1", "K2", "U"]


def load_rules(genome_path):
    try:
        flat_genome = np.load(genome_path)
        print(f"Loaded optimized genome rules from '{genome_path}'")
    except FileNotFoundError:
        print(f"No genome found at '{genome_path}'. Falling back to MATLAB baseline gains.")
        flat_genome = np.array(BASELINE_GAINS)

    return dict(zip(GAIN_NAMES, flat_genome))


def run_live_visualization(genome_path="optimized_gains.npy", seed=42):
    rules = load_rules(genome_path)
    print("Rules:", {k: round(float(v), 4) for k, v in rules.items()})

    eff, dist_travelled, average_batt, collisions = simulation_free_global_mod_2_LJ(
        rules=rules, seed=seed, visualize=True
    )

    print(f"Playback complete -- efficiency={eff:.4f}, "
          f"distance={dist_travelled:.4f}, avg_battery={average_batt:.4f}, "
          f"collisions={collisions}")
    print("Video saved to 'alone.mp4'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play back a swarm run for a saved genome.")
    parser.add_argument("genome_path", nargs="?", default="optimization_results/opt_small_scale_trial_03_seed_777_gains.npy",
                         help="Path to a .npy file with the 7 LJ rule gains "
                              "(r0, epsilon, k_align, k_goal, K1, K2, U).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the run.")
    args = parser.parse_args()

    run_live_visualization(args.genome_path, args.seed)
