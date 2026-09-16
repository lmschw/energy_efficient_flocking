"""Same as run_2stage_upwind_drain_holiday.py (2-stage curriculum + drain-holiday-while-
colliding mechanic via --drain-fraction), but ALSO trained WITH the hard safety clamp active
(config.py's own defaults: HEBBIAN_SAFETY_CLAMP_ENABLED=True, HEBBIAN_MIN_DIST_INFLATION=1.3,
HEBBIAN_RESOLVE_COLLISIONS=True strength=0.5 max_iter=2) -- i.e. this script simply does NOT
override those to False the way run_2stage_upwind_drain_holiday.py does.

Why this exists: upwind_2stage_drain0_seed123 (drain-holiday recipe, seed 123) was the
strongest hardware-deployment candidate found so far (97% both-beat-baseline under standard
physics, see project_upwind_safety_variant.md), but like every other winning genome this
session it was trained with the clamp OFF. Companion to run_2stage_upwind_clamped.py -- see
that file's docstring for the full "why retrain instead of bolt on" rationale (bolting the
clamp onto an already-trained non-clamp genome caused gridlock/distance-collapse in an earlier
pilot test). This lets the drain-holiday recipe's clamped result be compared directly against
the plain-drain recipe's clamped result (run_2stage_upwind_clamped.py) and both non-clamped
originals.

Usage: python run_2stage_upwind_drain_holiday_clamped.py --seed 123 --n-agents 20 --drain-fraction 0.0
"""
import argparse
import os

import numpy as np

import config
# Deliberately NOT overriding the clamp/inflation/resolve/instant-death constants -- see
# run_2stage_upwind_clamped.py's docstring; config.py's own current defaults already match
# safety_clamp_best's training config.

parser = argparse.ArgumentParser()
parser.add_argument("--n-agents", type=int, default=20)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--wind-grid", type=int, default=50)
parser.add_argument("--popsize", type=int, default=config.HEBBIAN_CMAES_POPSIZE)
parser.add_argument("--maxiter-stage1", type=int, default=config.HEBBIAN_CMAES_GEN_MAX)
parser.add_argument("--maxiter-stage2", type=int, default=2 * config.HEBBIAN_CMAES_GEN_MAX)
parser.add_argument("--n-repeats", type=int, default=config.HEBBIAN_N_REPEATS)
parser.add_argument("--drain-fraction", type=float, default=0.0,
                     help="Fraction of normal (above-idle-floor) drain paid while colliding "
                          "(0.0 = full holiday, matching upwind_2stage_drain0_seed123's recipe)")
parser.add_argument("--output-dir", default=None)
args = parser.parse_args()
if args.output_dir is None:
    pct = int(round(args.drain_fraction * 100))
    args.output_dir = f"../../results/hebbian_results_v2_2stage_upwind_drain_{pct}pct_clamped/n{args.n_agents}_seed{args.seed}"

from functools import partial
import optimize_hebbian
from simulation_hebbian_no_drain import simulate_hebbian_episode_partial_drain_on_collision
optimize_hebbian.simulate_hebbian_episode = partial(
    simulate_hebbian_episode_partial_drain_on_collision, drain_fraction=args.drain_fraction)

from optimize_hebbian import run_stage
from fitness_plot import FitnessPlotter

os.makedirs(args.output_dir, exist_ok=True)
print(f"2-stage curriculum + drain holiday ({args.drain_fraction*100:.0f}% of normal drain "
      f"paid while colliding) + SAFETY CLAMP active during training: walk_upwind "
      f"({args.maxiter_stage1} gen) -> save_battery_avoid_all ({args.maxiter_stage2} gen, "
      f"doubled). clamp={config.HEBBIAN_SAFETY_CLAMP_ENABLED} "
      f"inflation={config.HEBBIAN_MIN_DIST_INFLATION} "
      f"resolve={config.HEBBIAN_RESOLVE_COLLISIONS}. n_agents={args.n_agents} seed={args.seed} "
      f"wind_grid={args.wind_grid}. Output: {args.output_dir}")

np.random.seed(args.seed)
plotter = FitnessPlotter(path=os.path.join(args.output_dir, "hebbian_fitness_curve.png"))

genome = np.random.uniform(config.HEBBIAN_ABCD_BOUNDS[0], config.HEBBIAN_ABCD_BOUNDS[1], config.HEBBIAN_N_ABCD)
genome = run_stage("walk_upwind", genome, plotter, args.popsize, args.maxiter_stage1,
                    args.n_agents, args.n_repeats, args.seed, args.output_dir,
                    max_battery=None, min_battery=None, nx=args.wind_grid, ny=args.wind_grid,
                    use_battery_sensor=True, name_suffix="")
genome = run_stage("save_battery_avoid_all", genome, plotter, args.popsize, args.maxiter_stage2,
                    args.n_agents, args.n_repeats, args.seed, args.output_dir,
                    max_battery=None, min_battery=None, nx=args.wind_grid, ny=args.wind_grid,
                    use_battery_sensor=True, name_suffix="")
plotter.close()
print("2-stage + drain-holiday + safety-clamp training complete.")
