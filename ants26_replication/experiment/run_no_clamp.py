"""Driver for the "walk_left, no safety-clamp" ablation partner: same walk_left curriculum
and current-generation physics (KAPPA=10, HEBBIAN_EFF_DISTANCE_WEIGHT=16) as a normal
../experiment/ run, but with the three safety-clamp-era mechanisms turned off -- completing
the 2x2 {walk_left, walk_upwind} x {clamp, no-clamp} comparison alongside
../upwind_safety_variant/'s own run_no_clamp.py. Mirrors the pre_clamp_best/safety_clamp_best
split in hardware_transfer_test/, at n=10 instead of n=20, for exact-physics comparability
against the walk_upwind runs.
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
parser.add_argument("--output-dir", default="../../results/hebbian_results_v2_walkleft_no_clamp_pilot/n10_seed42")
args = parser.parse_args()

print(f"Safety clamp DISABLED for this run: HEBBIAN_SAFETY_CLAMP_ENABLED={config.HEBBIAN_SAFETY_CLAMP_ENABLED}, "
      f"HEBBIAN_MIN_DIST_INFLATION={config.HEBBIAN_MIN_DIST_INFLATION}, "
      f"HEBBIAN_RESOLVE_COLLISIONS={config.HEBBIAN_RESOLVE_COLLISIONS}")

train_one_seed(args.seed, args.output_dir, list(config.HEBBIAN_STAGES),
               config.HEBBIAN_CMAES_POPSIZE, config.HEBBIAN_CMAES_GEN_MAX,
               args.n_agents, config.HEBBIAN_N_REPEATS,
               battery=None, wind_grid=args.wind_grid, no_battery_sensor=False)
