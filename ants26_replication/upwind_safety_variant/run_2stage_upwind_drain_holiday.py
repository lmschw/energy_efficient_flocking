"""Same 2-stage curriculum as run_2stage_upwind.py (walk_upwind 100 gen -> save_battery_avoid_all
directly, 200 gen, doubled), but with the drain-holiday-while-colliding mechanic layered on
top via --drain-fraction (0.0 = full holiday, 0.5 = pay half of normal drain above the idle
floor while colliding, etc. -- see simulation_hebbian_no_drain.py). Requested as a follow-up
to run_2stage_upwind.py's plain run, to test the curriculum-structure change AND the
collision-battery-cost change together rather than in isolation.

Monkeypatches optimize_hebbian's imported `simulate_hebbian_episode` name (same mechanism as
run_partial_drain_on_collision.py/run_kappa20_drain_holiday.py) before calling run_stage()
directly twice with different maxiter per stage (same reason as run_2stage_upwind.py --
train_one_seed() only supports one uniform maxiter across all stages).
"""
import argparse
import os

import numpy as np

import config
config.HEBBIAN_SAFETY_CLAMP_ENABLED = False
config.HEBBIAN_MIN_DIST_INFLATION = 1.0
config.HEBBIAN_RESOLVE_COLLISIONS = False
config.HEBBIAN_COLLISION_INSTANT_DEATH = False

parser = argparse.ArgumentParser()
parser.add_argument("--n-agents", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wind-grid", type=int, default=50)
parser.add_argument("--popsize", type=int, default=config.HEBBIAN_CMAES_POPSIZE)
parser.add_argument("--maxiter-stage1", type=int, default=config.HEBBIAN_CMAES_GEN_MAX)
parser.add_argument("--maxiter-stage2", type=int, default=2 * config.HEBBIAN_CMAES_GEN_MAX)
parser.add_argument("--n-repeats", type=int, default=config.HEBBIAN_N_REPEATS)
parser.add_argument("--drain-fraction", type=float, required=True,
                     help="Fraction of normal (above-idle-floor) drain paid while colliding "
                          "(0.0 = full holiday, 0.5 = half, etc.)")
parser.add_argument("--output-dir", default=None)
args = parser.parse_args()
if args.output_dir is None:
    pct = int(round(args.drain_fraction * 100))
    args.output_dir = f"../../results/hebbian_results_v2_2stage_upwind_drain_{pct}pct/n10_seed42"

from functools import partial
import optimize_hebbian
from simulation_hebbian_no_drain import simulate_hebbian_episode_partial_drain_on_collision
optimize_hebbian.simulate_hebbian_episode = partial(
    simulate_hebbian_episode_partial_drain_on_collision, drain_fraction=args.drain_fraction)

from optimize_hebbian import run_stage
from fitness_plot import FitnessPlotter

os.makedirs(args.output_dir, exist_ok=True)
print(f"2-stage curriculum + drain holiday ({args.drain_fraction*100:.0f}% of normal drain "
      f"paid while colliding): walk_upwind ({args.maxiter_stage1} gen) -> "
      f"save_battery_avoid_all ({args.maxiter_stage2} gen, doubled). "
      f"Clamp/inflation/resolve/instant-death OFF. n_agents={args.n_agents} seed={args.seed} "
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
print("2-stage + drain-holiday training complete.")
