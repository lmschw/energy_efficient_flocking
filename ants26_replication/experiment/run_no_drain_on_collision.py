"""Trains with the ORIGINAL (pre-clamp) flat collision penalty fully active
(collision_w=250, include_inter_robot=True -- config.py defaults, unmodified) and no hard
clamp/min-dist-inflation/resolve-collisions/instant-death (same physics as pre_clamp_best),
but agents get a battery-drain holiday for any step they're within min_dist of another agent
-- see simulation_hebbian_no_drain.py's docstring for the full rationale. Isolates whether
pre_clamp_best's battery shortfall vs. the LJ baseline traces to the ordinary drain formula
behaving badly near collisions, independent of the collision_w penalty that's supposed to
discourage colliding in the first place.

Monkeypatches optimize_hebbian's imported `simulate_hebbian_episode` name to point at the
drain-holiday version -- fitness_wrapper() references that module-global name at call time,
so this redirects every candidate evaluation without touching optimize_hebbian.py itself.
"""
import argparse

import config
config.HEBBIAN_SAFETY_CLAMP_ENABLED = False
config.HEBBIAN_MIN_DIST_INFLATION = 1.0
config.HEBBIAN_RESOLVE_COLLISIONS = False
config.HEBBIAN_COLLISION_INSTANT_DEATH = False

import optimize_hebbian
from simulation_hebbian_no_drain import simulate_hebbian_episode_no_drain_on_collision
optimize_hebbian.simulate_hebbian_episode = simulate_hebbian_episode_no_drain_on_collision

from optimize_hebbian import train_one_seed

parser = argparse.ArgumentParser()
parser.add_argument("--n-agents", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wind-grid", type=int, default=50)
parser.add_argument("--output-dir", default="../../results/hebbian_results_v2_no_drain_on_collision/n10_seed42")
args = parser.parse_args()

print(f"save_battery_avoid_all fitness weights (unmodified, original penalty): "
      f"{config.HEBBIAN_STAGE_FITNESS_WEIGHTS['save_battery_avoid_all']}")
print("Clamp/inflation/resolve/instant-death all OFF (pre_clamp-style physics); "
      "battery drain holiday active for colliding agents.")

train_one_seed(args.seed, args.output_dir, list(config.HEBBIAN_STAGES),
               config.HEBBIAN_CMAES_POPSIZE, config.HEBBIAN_CMAES_GEN_MAX,
               args.n_agents, config.HEBBIAN_N_REPEATS,
               battery=None, wind_grid=args.wind_grid, no_battery_sensor=False)
