"""Same 2-stage curriculum as run_2stage_upwind.py (walk_upwind 100 gen -> save_battery_avoid_all
directly, 200 gen doubled), but trained WITH the hard safety layer active throughout --
HEBBIAN_SAFETY_CLAMP_ENABLED, HEBBIAN_MIN_DIST_INFLATION=1.3, HEBBIAN_RESOLVE_COLLISIONS (soft
push, strength=0.5, max_iter=2) -- i.e. config.py's own current defaults, matching exactly how
safety_clamp_best (the original 3-stage clamped genome) was trained. run_2stage_upwind.py
explicitly turns all of these OFF; this script just doesn't, so no config.* assignments are
needed here beyond leaving the defaults alone.

Why this exists: the two "beats LJ baseline" winning genomes from run_2stage_upwind.py
(seeds 123/888, see hardware_transfer_test/upwind_2stage_plain_seed{123,888}/) were trained
with the clamp OFF. Bolting the clamp onto an ALREADY-trained non-clamp genome post-hoc is a
confirmed bad idea (see summary.tex sec:why-two-genomes / project_upwind_safety_variant.md --
a pilot test of exactly that caused distance to collapse to ~0-1.5m via gridlock, since the
controller kept commanding full-speed moves into neighbors that then got hard-braked). The
established fix is to retrain from scratch with the clamp already active during evolution, so
CMA-ES can learn to route around it -- this script does that for the winning recipe, so its
result can be compared against the existing non-clamped winners for hardware-deployment
purposes (real Thymios have no physical compliance; an inter-agent hard clamp trained in is a
much stronger safety guarantee than a deployment-only bolt-on governor).

Usage: python run_2stage_upwind_clamped.py --seed 123 --n-agents 20
"""
import argparse
import os

import numpy as np

import config
# Deliberately NOT overriding HEBBIAN_SAFETY_CLAMP_ENABLED/HEBBIAN_MIN_DIST_INFLATION/
# HEBBIAN_RESOLVE_COLLISIONS/HEBBIAN_COLLISION_INSTANT_DEATH -- config.py's own current
# defaults (True, 1.3, True w/ strength=0.5 max_iter=2, False) already match safety_clamp_best's
# training config exactly (see config.py's own "ACTIVE for n20_seed42, hebbian_results_v2_
# original_n20_..." comments beside each constant).

from optimize_hebbian import run_stage
from fitness_plot import FitnessPlotter

parser = argparse.ArgumentParser()
parser.add_argument("--n-agents", type=int, default=20)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--wind-grid", type=int, default=50)
parser.add_argument("--popsize", type=int, default=config.HEBBIAN_CMAES_POPSIZE)
parser.add_argument("--maxiter-stage1", type=int, default=config.HEBBIAN_CMAES_GEN_MAX)
parser.add_argument("--maxiter-stage2", type=int, default=2 * config.HEBBIAN_CMAES_GEN_MAX)
parser.add_argument("--n-repeats", type=int, default=config.HEBBIAN_N_REPEATS)
parser.add_argument("--output-dir", default=None)
args = parser.parse_args()
if args.output_dir is None:
    args.output_dir = f"../../results/hebbian_results_v2_2stage_upwind_clamped/n{args.n_agents}_seed{args.seed}"

os.makedirs(args.output_dir, exist_ok=True)
print(f"2-stage curriculum + SAFETY CLAMP active during training: walk_upwind "
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
print("2-stage + safety-clamp training complete.")
