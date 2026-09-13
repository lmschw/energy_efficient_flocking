"""Combines the two mechanisms that each showed a partial, non-overlapping benefit on their
own: KAPPA=20 (job 11 -- confirmed real exposure-equalization learning, but left collision
waste untouched, so it still lost to LJ baseline overall) and the full drain-holiday-while-
colliding mechanic (job 6, drain_fraction=0.0, floor-protected fix -- the only point in the
jobs 5-10 sweep that beat baseline on distance, albeit volatile). Neither alone cleared LJ;
this tests whether addressing exposure-inequality and collision-waste at the same time does.

Same pre_clamp-style physics otherwise (no clamp/inflation/resolve/instant-death), original
walk_left curriculum, default fitness weights (the collision_w=250 penalty is still fully
active as the training signal, same as every drain-holiday job).
"""
import argparse

import config
config.HEBBIAN_SAFETY_CLAMP_ENABLED = False
config.HEBBIAN_MIN_DIST_INFLATION = 1.0
config.HEBBIAN_RESOLVE_COLLISIONS = False
config.HEBBIAN_COLLISION_INSTANT_DEATH = False

parser = argparse.ArgumentParser()
parser.add_argument("--n-agents", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wind-grid", type=int, default=50)
parser.add_argument("--kappa-mult", type=float, default=2.0)
parser.add_argument("--output-dir", default=None)
args = parser.parse_args()
config.KAPPA = 10.0 * args.kappa_mult
if args.output_dir is None:
    args.output_dir = f"../../results/hebbian_results_v2_kappa{int(config.KAPPA)}_drain_holiday/n10_seed42"

import optimize_hebbian
from simulation_hebbian_no_drain import simulate_hebbian_episode_no_drain_on_collision
optimize_hebbian.simulate_hebbian_episode = simulate_hebbian_episode_no_drain_on_collision

from optimize_hebbian import train_one_seed

print(f"KAPPA={config.KAPPA} (x{args.kappa_mult}); drain holiday ON (full refund above idle "
      f"floor, floor-protected fix); clamp/inflation/resolve/instant-death all OFF; "
      f"output: {args.output_dir}")

train_one_seed(args.seed, args.output_dir, list(config.HEBBIAN_STAGES),
               config.HEBBIAN_CMAES_POPSIZE, config.HEBBIAN_CMAES_GEN_MAX,
               args.n_agents, config.HEBBIAN_N_REPEATS,
               battery=None, wind_grid=args.wind_grid, no_battery_sensor=False)
