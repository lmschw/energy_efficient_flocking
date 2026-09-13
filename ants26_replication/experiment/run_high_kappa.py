"""Trains under KAPPA=20 (2x the paper's calibrated value of 10) instead of re-evaluating an
existing genome -- see project memory / conversation for the reasoning: F_drag never touches
movement kinematics, only the battery-drain formula's dottprod term, so KAPPA is the exact
"wind-exposure cost, decoupled from physical force" lever the user asked for. Doubling it
widens the survival gap between well- and badly-exposed agents (confirmed: at n=10 seed=42,
LJ baseline's final battery spread widens from [~-0.4, 79.3] to [~0.0, 96.6] and its distance
drops 20.07m->12.50m going from KAPPA=10->20), which a FIXED baseline can't do anything about,
but an EVOLVED controller might learn to counter by equalizing exposure (rotating front/back)
-- something the existing pre_clamp_best genome only does incidentally (never trained for a
harsher regime specifically). This retrains from scratch to see if CMA-ES actually discovers
that under real selection pressure, rather than just evaluating an unrelated genome.

Everything else matches pre_clamp_best's own recipe exactly (no clamp, no min-dist inflation,
no resolve-collisions, no instant-death, original walk_left curriculum, default fitness
weights) -- KAPPA=20 is the ONLY changed variable, for a clean single-variable comparison.
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
    args.output_dir = f"../../results/hebbian_results_v2_kappa{int(config.KAPPA)}/n10_seed42"

from optimize_hebbian import train_one_seed

print(f"KAPPA={config.KAPPA} (x{args.kappa_mult}); clamp/inflation/resolve/instant-death all "
      f"OFF (pre_clamp-style physics, single-variable change); output: {args.output_dir}")

train_one_seed(args.seed, args.output_dir, list(config.HEBBIAN_STAGES),
               config.HEBBIAN_CMAES_POPSIZE, config.HEBBIAN_CMAES_GEN_MAX,
               args.n_agents, config.HEBBIAN_N_REPEATS,
               battery=None, wind_grid=args.wind_grid, no_battery_sensor=False)
