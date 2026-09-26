"""Reverts to the 2-stage curriculum (walk_upwind -> save_battery_avoid_all directly, skipping
the intermediate save_battery_avoid_wall stage), doubling the second stage's generation budget
to keep the total CMA-ES budget constant (100 + 200 = 300, same as the normal 3-stage
100+100+100) -- matching the project's earlier 2-stage precedent (see
`hebbian_results_v2_original_2stage`/`_original_n20_2stage` and the historical note in
config.py: "2-stage curriculum skipping save_battery_avoid_wall entirely, doubling
save_battery_avoid_all to 200 generations" -- that attempt used the original walk_left
curriculum; this repeats the same STRUCTURE with walk_upwind for stage 1 instead), now that
save_battery_avoid_all's own weights already include full battery-awareness AND both wall and
inter-robot collision terms from the start (Table: battery_w=5.0, collision_w=250.0,
wall_col_mult=3.0, include_inter_robot=True) -- there's no separate "wall-only" stage to skip
through, the agent has to learn all three objectives simultaneously in one (longer) stage.

Uses upwind_safety_variant's own package (WIND_DIRECTION-based walk_upwind already set up
correctly here) rather than experiment/, but with clamp/inflation/resolve/instant-death all
OFF (pre_clamp-style physics) -- no drain-holiday mechanic in this run, standard battery
drain throughout, matching the historical 2-stage precedent's physics as closely as possible.
This is a clean test of the CURRICULUM STRUCTURE change alone, not stacked with any of the
collision/drain mechanics tried in jobs 5-12.

Calls optimize_hebbian.run_stage() directly (bypassing train_one_seed(), which only supports
one uniform maxiter across all stages) so stage 1 and stage 2 can have different generation
budgets.
"""
import argparse
import os

import numpy as np

import config
config.HEBBIAN_SAFETY_CLAMP_ENABLED = False
config.HEBBIAN_MIN_DIST_INFLATION = 1.0
config.HEBBIAN_RESOLVE_COLLISIONS = False
config.HEBBIAN_COLLISION_INSTANT_DEATH = False

from optimize_hebbian import run_stage
from fitness_plot import FitnessPlotter

parser = argparse.ArgumentParser()
parser.add_argument("--n-agents", type=int, nargs="+", default=[10],
                    help="Swarm size. Several values (e.g. 1 5 10 15 20) = mixed-n training: each "
                         "candidate is evaluated at every size, fitness = mean over sizes.")
parser.add_argument("--empty-quadrant-zero", action="store_true",
                    help="Encode an empty sensor quadrant as (0, 0) instead of (+1, -1) -- see "
                         "config.HEBBIAN_EMPTY_QUADRANT_ZERO.")
parser.add_argument("--resume", action="store_true",
                    help="Checkpoint CMA-ES every generation and resume from an existing checkpoint "
                         "in --output-dir. If stage 1's best genome is already saved there, stage 1 "
                         "is skipped and stage 2 starts (or resumes) from it.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wind-grid", type=int, default=50)
parser.add_argument("--popsize", type=int, default=config.HEBBIAN_CMAES_POPSIZE)
parser.add_argument("--maxiter-stage1", type=int, default=config.HEBBIAN_CMAES_GEN_MAX)
parser.add_argument("--maxiter-stage2", type=int, default=2 * config.HEBBIAN_CMAES_GEN_MAX)
parser.add_argument("--n-repeats", type=int, default=config.HEBBIAN_N_REPEATS)
parser.add_argument("--output-dir", default="../../results/hebbian_results_v2_2stage_upwind/n10_seed42")
args = parser.parse_args()
if len(args.n_agents) == 1:
    args.n_agents = args.n_agents[0]
config.HEBBIAN_EMPTY_QUADRANT_ZERO = args.empty_quadrant_zero

os.makedirs(args.output_dir, exist_ok=True)
print(f"2-stage curriculum: walk_upwind ({args.maxiter_stage1} gen) -> "
      f"save_battery_avoid_all ({args.maxiter_stage2} gen, doubled). "
      f"Clamp/inflation/resolve/instant-death OFF. n_agents={args.n_agents} seed={args.seed} "
      f"empty_quadrant_zero={config.HEBBIAN_EMPTY_QUADRANT_ZERO} "
      f"wind_grid={args.wind_grid}. Output: {args.output_dir}")

np.random.seed(args.seed)
plotter = FitnessPlotter(path=os.path.join(args.output_dir, "hebbian_fitness_curve.png"))

genome = np.random.uniform(config.HEBBIAN_ABCD_BOUNDS[0], config.HEBBIAN_ABCD_BOUNDS[1], config.HEBBIAN_N_ABCD)
stage1_path = os.path.join(args.output_dir, "hebbian_walk_upwind_best.npy")
if args.resume and os.path.exists(stage1_path):
    print(f"↳ Stage 1 already done -- loading {stage1_path} and skipping walk_upwind.")
    genome = np.load(stage1_path)
else:
    genome = run_stage("walk_upwind", genome, plotter, args.popsize, args.maxiter_stage1,
                        args.n_agents, args.n_repeats, args.seed, args.output_dir,
                        max_battery=None, min_battery=None, nx=args.wind_grid, ny=args.wind_grid,
                        use_battery_sensor=True, name_suffix="", checkpoint=args.resume)
genome = run_stage("save_battery_avoid_all", genome, plotter, args.popsize, args.maxiter_stage2,
                    args.n_agents, args.n_repeats, args.seed, args.output_dir,
                    max_battery=None, min_battery=None, nx=args.wind_grid, ny=args.wind_grid,
                    use_battery_sensor=True, name_suffix="", checkpoint=args.resume)
plotter.close()
print("2-stage training complete.")
