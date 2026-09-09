"""Driver for the "upwind, no safety-clamp" ablation partner: same walk_upwind curriculum
and same current-generation physics (KAPPA=10, HEBBIAN_EFF_DISTANCE_WEIGHT=16) as this
variant's normal run, but with the three safety-clamp-era mechanisms turned off -- isolating
the safety-clamp on/off variable while holding the stage-1 substitution and all other physics
fixed. Mirrors the pre_clamp_best/safety_clamp_best split in hardware_transfer_test/ but for
the walk_upwind curriculum instead of walk_left.
"""
import argparse

import config
config.HEBBIAN_SAFETY_CLAMP_ENABLED = False
config.HEBBIAN_MIN_DIST_INFLATION = 1.0
config.HEBBIAN_RESOLVE_COLLISIONS = False

from optimize_hebbian import train_one_seed

parser = argparse.ArgumentParser()
parser.add_argument("--n-agents", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wind-grid", type=int, default=50)
parser.add_argument("--output-dir", default="../../results/hebbian_results_v2_upwind_no_clamp_pilot/n10_seed42")
args = parser.parse_args()

print(f"Safety clamp DISABLED for this run: HEBBIAN_SAFETY_CLAMP_ENABLED={config.HEBBIAN_SAFETY_CLAMP_ENABLED}, "
      f"HEBBIAN_MIN_DIST_INFLATION={config.HEBBIAN_MIN_DIST_INFLATION}, "
      f"HEBBIAN_RESOLVE_COLLISIONS={config.HEBBIAN_RESOLVE_COLLISIONS}")

train_one_seed(args.seed, args.output_dir, list(config.HEBBIAN_STAGES),
               config.HEBBIAN_CMAES_POPSIZE, config.HEBBIAN_CMAES_GEN_MAX,
               args.n_agents, config.HEBBIAN_N_REPEATS,
               battery=None, wind_grid=args.wind_grid, no_battery_sensor=False)
