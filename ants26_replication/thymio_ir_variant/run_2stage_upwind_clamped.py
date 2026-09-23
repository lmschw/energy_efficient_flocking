"""2-stage curriculum (walk_upwind 100 gen -> save_battery_avoid_all directly, 200 gen
doubled) trained WITH the hard safety layer active throughout -- HEBBIAN_SAFETY_CLAMP_ENABLED,
HEBBIAN_MIN_DIST_INFLATION=1.3, HEBBIAN_RESOLVE_COLLISIONS (soft push, strength=0.5, max_iter=2)
-- i.e. config.py's own current defaults, so no config.* overrides are needed here beyond
leaving them alone.

Ported from ../upwind_safety_variant/run_2stage_upwind_clamped.py -- the exact recipe that
produced plain_seed123_clamped (the genome actually deployed to real hardware), reused here
against THIS package's 9-input Thymio-IR sensor model (sensor_model.py) instead of the
idealized 4-quadrant range/bearing sensor. See config.py's "Neural controller architecture"
comment and sensor_model.py's module docstring for why: the idealized sensor requires
reconstructing global range/bearing at deployment time, which is what has made real-hardware
calibration (POSITION_AXES, HEADING_OFFSET_RAD, corridor bounds, OptiTrack coordinate-frame
consistency across sessions) so fragile in ants26_replication/hardware_deployment/. A genome
trained against the Thymio's own onboard IR array instead would need OptiTrack only to RECORD
trajectories for analysis, not to compute anything the running controller depends on.

Usage: python run_2stage_upwind_clamped.py --seed 123 --n-agents 20
"""
import argparse
import os

import numpy as np

import config
# Deliberately NOT overriding HEBBIAN_SAFETY_CLAMP_ENABLED/HEBBIAN_MIN_DIST_INFLATION/
# HEBBIAN_RESOLVE_COLLISIONS/HEBBIAN_COLLISION_INSTANT_DEATH -- config.py's own current
# defaults already match plain_seed123_clamped's training config exactly (ported from
# ../upwind_safety_variant/config.py -- see this file's own "Inter-agent/wall safety clamp..."
# section).

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
    args.output_dir = f"../../results/hebbian_results_v2_thymio_ir_2stage_upwind_clamped/n{args.n_agents}_seed{args.seed}"

os.makedirs(args.output_dir, exist_ok=True)
print(f"2-stage curriculum + SAFETY CLAMP active during training (Thymio IR sensor): walk_upwind "
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
print("2-stage + safety-clamp (Thymio IR sensor) training complete.")
