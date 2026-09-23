"""Re-runs ONLY the save_battery_avoid_all stage, seeded from the already-trained
hebbian_walk_upwind_best.npy in n20_seed123's original results dir, against the newly
widened graduated proximity penalty (config.py's HEBBIAN_PROXIMITY_OUTER_GAP/INNER_GAP,
now matching the safety clamp's own thresholds instead of the much-tighter ported
default) -- see config.py's comment on that constant and HEBBIAN_STAGE_FITNESS_WEIGHTS
for the full rationale.

Stage 1 (walk_upwind) has no crowding-related fitness term at all (proximity_w=None,
collision_w=None for that stage), so its result is completely unaffected by this
config change -- no need to re-run 100 generations of identical work.

Usage: python rerun_stage2_proximity_penalty.py
"""
import os

import numpy as np

import config
from optimize_hebbian import run_stage
from fitness_plot import FitnessPlotter

SEED = 123
N_AGENTS = 20
WIND_GRID = 50
POPSIZE = config.HEBBIAN_CMAES_POPSIZE
MAXITER_STAGE2 = 2 * config.HEBBIAN_CMAES_GEN_MAX

ORIGINAL_DIR = "../../results/hebbian_results_v2_thymio_ir_2stage_upwind_clamped/n20_seed123"
OUTPUT_DIR = "../../results/hebbian_results_v2_thymio_ir_2stage_upwind_clamped_proxpenalty/n20_seed123"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Re-running save_battery_avoid_all only, seeded from "
      f"'{ORIGINAL_DIR}/hebbian_walk_upwind_best.npy', with proximity_w={config.HEBBIAN_STAGE_FITNESS_WEIGHTS['save_battery_avoid_all'][5]} "
      f"outer_gap={config.HEBBIAN_PROXIMITY_OUTER_GAP} inner_gap={config.HEBBIAN_PROXIMITY_INNER_GAP}. "
      f"n_agents={N_AGENTS} seed={SEED} wind_grid={WIND_GRID}. Output: {OUTPUT_DIR}")

np.random.seed(SEED)
plotter = FitnessPlotter(path=os.path.join(OUTPUT_DIR, "hebbian_fitness_curve.png"))

genome = np.load(os.path.join(ORIGINAL_DIR, "hebbian_walk_upwind_best.npy"))
genome = run_stage("save_battery_avoid_all", genome, plotter, POPSIZE, MAXITER_STAGE2,
                    N_AGENTS, config.HEBBIAN_N_REPEATS, SEED, OUTPUT_DIR,
                    max_battery=None, min_battery=None, nx=WIND_GRID, ny=WIND_GRID,
                    use_battery_sensor=True, name_suffix="")
plotter.close()
print("save_battery_avoid_all (proximity-penalty variant) training complete.")
