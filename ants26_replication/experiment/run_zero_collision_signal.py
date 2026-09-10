"""Trains save_battery_avoid_all with ZERO inter-robot collision signal of any kind -- no
flat collision_w penalty, no graduated proximity penalty, no instant-death -- only wall
avoidance remains (walls aren't "the agents"). Tests whether the very presence of a
collision-avoidance training signal (not just actual collisions, and not just the hard
clamp) is itself costing distance/battery, by removing it entirely and letting CMA-ES
freely form arbitrarily tight/overlapping formations during training.

The resulting genome is then meant to be evaluated TWICE (see evaluate_zero_collision_signal.py):
once as-is (no clamp -- expect chronic collision, since nothing ever discouraged it), and
once with the hard safety clamp bolted on at evaluation time only (not retrained with it) --
this second condition is the direct test of "train free, clamp after" as an alternative to
safety_clamp_best's "retrain from scratch with the clamp active" approach. Prior evidence
(see config.py's HEBBIAN_SAFETY_CLAMP_ENABLED history) already showed bolting the clamp onto
a genome that HAD some collision awareness caused gridlock; this tests the more extreme case
of bolting it onto a genome with NONE.
"""
import argparse

import config
config.HEBBIAN_SAFETY_CLAMP_ENABLED = False
config.HEBBIAN_MIN_DIST_INFLATION = 1.0
config.HEBBIAN_RESOLVE_COLLISIONS = False
config.HEBBIAN_COLLISION_INSTANT_DEATH = False
# save_battery_avoid_all normally: (5.0, 250.0, 3.0, True, None, None) -- flip
# include_inter_robot_collision False so collision_w only ever sees wall_collision_time.
_bw, _cw, _wm, _incl, _coh, _prox = config.HEBBIAN_STAGE_FITNESS_WEIGHTS["save_battery_avoid_all"]
config.HEBBIAN_STAGE_FITNESS_WEIGHTS["save_battery_avoid_all"] = (_bw, _cw, _wm, False, _coh, _prox)

from optimize_hebbian import train_one_seed

parser = argparse.ArgumentParser()
parser.add_argument("--n-agents", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wind-grid", type=int, default=50)
parser.add_argument("--output-dir", default="../../results/hebbian_results_v2_zero_collision_signal/n10_seed42")
args = parser.parse_args()

print(f"save_battery_avoid_all fitness weights (battery_w, collision_w, wall_mult, "
      f"include_inter_robot, cohesion_w, proximity_w) = "
      f"{config.HEBBIAN_STAGE_FITNESS_WEIGHTS['save_battery_avoid_all']}")
print("HEBBIAN_COLLISION_INSTANT_DEATH =", config.HEBBIAN_COLLISION_INSTANT_DEATH)

train_one_seed(args.seed, args.output_dir, list(config.HEBBIAN_STAGES),
               config.HEBBIAN_CMAES_POPSIZE, config.HEBBIAN_CMAES_GEN_MAX,
               args.n_agents, config.HEBBIAN_N_REPEATS,
               battery=None, wind_grid=args.wind_grid, no_battery_sensor=False)
