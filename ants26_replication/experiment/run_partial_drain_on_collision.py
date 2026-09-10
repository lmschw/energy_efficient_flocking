"""Follow-up to job 6 (full drain holiday while colliding): colliding agents still pay
`--drain-fraction` (default 0.1 = 10%) of their normal battery drain instead of a full
refund -- "a small drain for collisions, but significantly smaller than we had before" (the
original, unrefunded 100% drain). Same physics otherwise as job 6: original unmodified
save_battery_avoid_all collision_w=250 penalty (the training signal is untouched), clamp/
inflation/resolve/instant-death all OFF (pre_clamp-style).

Monkeypatches optimize_hebbian's imported `simulate_hebbian_episode` name, same mechanism as
run_no_drain_on_collision.py.
"""
import argparse
from functools import partial

import config
config.HEBBIAN_SAFETY_CLAMP_ENABLED = False
config.HEBBIAN_MIN_DIST_INFLATION = 1.0
config.HEBBIAN_RESOLVE_COLLISIONS = False
config.HEBBIAN_COLLISION_INSTANT_DEATH = False

parser = argparse.ArgumentParser()
parser.add_argument("--n-agents", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wind-grid", type=int, default=50)
parser.add_argument("--drain-fraction", type=float, default=0.1,
                     help="Fraction of normal drain colliding agents still pay (default 0.1 = 10%%).")
parser.add_argument("--output-dir", default=None)
args = parser.parse_args()
if args.output_dir is None:
    pct = int(round(args.drain_fraction * 100))
    args.output_dir = f"../../results/hebbian_results_v2_partial_drain_on_collision_{pct}pct/n10_seed42"

import optimize_hebbian
from simulation_hebbian_no_drain import simulate_hebbian_episode_partial_drain_on_collision
optimize_hebbian.simulate_hebbian_episode = partial(
    simulate_hebbian_episode_partial_drain_on_collision, drain_fraction=args.drain_fraction)

from optimize_hebbian import train_one_seed

print(f"save_battery_avoid_all fitness weights (unmodified, original penalty): "
      f"{config.HEBBIAN_STAGE_FITNESS_WEIGHTS['save_battery_avoid_all']}")
print(f"Clamp/inflation/resolve/instant-death all OFF (pre_clamp-style physics); "
      f"colliding agents pay {args.drain_fraction*100:.0f}% of normal drain "
      f"({(1-args.drain_fraction)*100:.0f}% refunded).")
print(f"Output: {args.output_dir}")

train_one_seed(args.seed, args.output_dir, list(config.HEBBIAN_STAGES),
               config.HEBBIAN_CMAES_POPSIZE, config.HEBBIAN_CMAES_GEN_MAX,
               args.n_agents, config.HEBBIAN_N_REPEATS,
               battery=None, wind_grid=args.wind_grid, no_battery_sensor=False)
