"""Staged CMA-ES training for the Hebbian ABCD controller (paper Sections 2.2-2.3).

Three sequential stages of increasing task complexity (Table 2 / Fig. 1):
  1. walk_left                  -- wind disabled, fitness = distance only
  2. save_battery_avoid_wall    -- wind enabled, + battery term, + wall-collision penalty
  3. save_battery_avoid_all     -- wind enabled, + battery term, + wall AND inter-robot collision penalty

Each stage runs CMA-ES (population 30, 100 generations, sigma0=0.3) with every
candidate evaluated over 3 random seeds, taking the MEDIAN efficiency as its fitness
(paper Section 2.2; matches evaluateABCD.m's sort-and-take-median-of-3 technique).
Stage 1 starts from a fresh ABCD_init sampled uniformly from [-5, 5]; stages 2 and 3
each start from the previous stage's best genome ("Next stage: best x is initial x",
Fig. 1) -- this is the actual curriculum mechanism, distinct from the small/baseline/
large_scale agent-count sweep in run_batch_experiments.py (which has no equivalent in
the paper).

MATLAB reference for the underlying architecture: hebbianStep.m, the getsensordata()/
W(i) init loop in simulation_free_global_mod_2.m. NOT evaluateABCD.m/optimizeABCD.m,
which implement a different, stale 1680-parameter architecture (see hebbian_controller.py).
"""
import argparse
import json
import os
import sys

import cma
import numpy as np

import config
from hebbian_controller import unflatten_abcd
from simulation_hebbian import simulate_hebbian_episode, stage_fitness
from fitness_plot import FitnessPlotter

current_candidate = 0
total_candidates = 0
active_stage = None
active_n_agents = config.HEBBIAN_N_AGENTS
active_n_repeats = config.HEBBIAN_N_REPEATS
active_seed_base = 0
active_max_battery = None
active_min_battery = None
active_nx = None
active_ny = None
active_use_battery_sensor = True


def fitness_wrapper(genome):
    global current_candidate
    current_candidate += 1
    sys.stdout.write(f"\r   ↳ Evaluating Swarm Candidate: {current_candidate}/{total_candidates} ...")
    sys.stdout.flush()

    rules = unflatten_abcd(genome)
    wind_enabled = config.HEBBIAN_STAGE_WIND_ENABLED[active_stage]

    effs = []
    for r in range(active_n_repeats):
        seed = active_seed_base + current_candidate * 1000 + r  # distinct seed per repeat, per candidate
        try:
            dist, batt, ct, wct = simulate_hebbian_episode(
                rules, seed=seed, n_agents=active_n_agents, wind_enabled=wind_enabled,
                max_battery=active_max_battery, min_battery=active_min_battery,
                nx=active_nx, ny=active_ny, use_battery_sensor=active_use_battery_sensor)
            effs.append(stage_fitness(dist, batt, ct, wct, active_stage))
        except Exception as e:
            print(f"\n⚠️  Candidate {current_candidate} repeat {r} failed "
                  f"({type(e).__name__}: {e}) -- treating as worst-case for this repeat")
            effs.append(-99999.0)

    return -float(np.median(effs))  # CMA-ES minimizes


def run_stage(stage, x0, plotter, popsize, maxiter, n_agents, n_repeats, seed_base, output_dir,
              max_battery, min_battery, nx, ny, use_battery_sensor, name_suffix):
    global active_stage, total_candidates, active_n_agents, active_n_repeats, active_seed_base
    global active_max_battery, active_min_battery, active_nx, active_ny, active_use_battery_sensor

    active_stage = stage
    active_n_agents = n_agents
    active_n_repeats = n_repeats
    active_seed_base = seed_base
    active_max_battery = max_battery
    active_min_battery = min_battery
    active_nx = nx
    active_ny = ny
    active_use_battery_sensor = use_battery_sensor
    total_candidates = popsize

    print(f"\n{'=' * 70}\n🧬 STAGE: {stage}  (wind_enabled={config.HEBBIAN_STAGE_WIND_ENABLED[stage]}, "
          f"battery_sensor={use_battery_sensor})\n{'=' * 70}")

    es = cma.CMAEvolutionStrategy(x0, config.HEBBIAN_CMAES_SIGMA0, {
        'popsize': popsize,
        'maxiter': maxiter,
        'bounds': list(config.HEBBIAN_ABCD_BOUNDS),
    })
    plotter.reset_run(title=f"stage: {stage}{name_suffix}")

    gen = 0
    fitness_history = []
    while not es.stop():
        gen += 1
        current_candidate_reset()
        solutions = es.ask()
        fitness_values = [fitness_wrapper(sol) for sol in solutions]
        es.tell(solutions, fitness_values)
        plotter.update(gen, fitness_values)
        fitness_history.append(float(min(fitness_values)))
        sys.stdout.write("\r")
        print(f"✅ Gen {gen:03d}/{maxiter} | Best Loss (neg eff): {min(fitness_values):.4f}")

    best_genome = es.result[0]
    best_loss = float(es.result[1])

    genome_name = f"hebbian_{stage}{name_suffix}_best.npy"
    history_name = f"hebbian_{stage}{name_suffix}_history.json"
    np.save(os.path.join(output_dir, genome_name), best_genome)
    with open(os.path.join(output_dir, history_name), "w") as f:
        json.dump({"stage": stage, "battery_sensor": use_battery_sensor, "best_loss": best_loss,
                   "best_efficiency": -best_loss, "loss_curve": fitness_history,
                   "genome": best_genome.tolist(),
                   # Recorded so playback/analysis tools (visualize_hebbian.py) can detect
                   # and default to the actual training condition instead of silently
                   # falling back to config.HEBBIAN_N_AGENTS/HEBBIAN_MAX_BATTERY, which
                   # would misrepresent this genome's behavior if it differs (n_agents
                   # especially -- flocking dynamics are visibly agent-count-sensitive).
                   "n_agents": n_agents, "max_battery": max_battery, "min_battery": min_battery,
                   "wind_grid_nx": nx, "wind_grid_ny": ny}, f, indent=2)

    print(f"💾 Stage '{stage}' complete. Best efficiency: {-best_loss:.4f}. Saved {genome_name}")
    return best_genome


def current_candidate_reset():
    global current_candidate
    current_candidate = 0


def train_one_seed(seed, output_dir, stages, popsize, maxiter, n_agents, n_repeats,
                    battery, wind_grid, no_battery_sensor):
    """The full staged curriculum for a single seed -- stage 1 starts from a fresh
    uniform-random genome sampled under this seed, stages 2/3 hand off the previous
    stage's best genome, exactly as a single (non-swept) run always has. Factored out
    so --seeds can call this once per seed into its own output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    name_suffix = "_nosensor" if no_battery_sensor else ""
    plotter = FitnessPlotter(path=os.path.join(output_dir, f"hebbian_fitness_curve{name_suffix}.png"))

    genome = np.random.uniform(config.HEBBIAN_ABCD_BOUNDS[0], config.HEBBIAN_ABCD_BOUNDS[1],
                                config.HEBBIAN_N_ABCD)
    for stage in stages:
        genome = run_stage(stage, genome, plotter, popsize, maxiter, n_agents,
                            n_repeats, seed, output_dir,
                            max_battery=battery, min_battery=battery,
                            nx=wind_grid, ny=wind_grid,
                            use_battery_sensor=not no_battery_sensor,
                            name_suffix=name_suffix)
    plotter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Staged CMA-ES training of the Hebbian ABCD controller.")
    parser.add_argument("--popsize", type=int, default=config.HEBBIAN_CMAES_POPSIZE,
                         help=f"CMA-ES population size (paper: {config.HEBBIAN_CMAES_POPSIZE}).")
    parser.add_argument("--maxiter", type=int, default=config.HEBBIAN_CMAES_GEN_MAX,
                         help=f"Generations per stage (paper: {config.HEBBIAN_CMAES_GEN_MAX}).")
    parser.add_argument("--n-agents", type=int, default=config.HEBBIAN_N_AGENTS,
                         help=f"Swarm size (paper: {config.HEBBIAN_N_AGENTS}).")
    parser.add_argument("--n-repeats", type=int, default=config.HEBBIAN_N_REPEATS,
                         help=f"Random-seed repeats per candidate, fitness=median (paper: {config.HEBBIAN_N_REPEATS}).")
    parser.add_argument("--seed", type=int, default=0,
                         help="Base seed for reproducibility (single-run mode; ignored if --seeds is given).")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                         help="Run the ENTIRE staged curriculum independently for each of these seeds "
                              "(e.g. --seeds 42 123 777), each into its own '<output-dir>/seed_<seed>/' "
                              "subdirectory -- for reporting a distribution (mean/std/best-of-N) across "
                              "independent CMA-ES runs instead of trusting a single stochastic one. Pass "
                              "the flag with no values to use this project's 10 canonical seeds "
                              f"({config.HEBBIAN_BATCH_SEEDS}). Overrides --seed. Each seed's training is "
                              "fully independent (separate output subdirectory, no shared state), so "
                              "running several in parallel yourself -- separate terminals/machines, one "
                              "invocation per seed with matching --output-dir -- is safe if you have the "
                              "cores for it; this flag itself still runs them one after another. NOTE: a "
                              "single full 3-stage run has taken on the order of 10 hours in this "
                              "project's own timing, so a sequential 10-seed sweep is on the order of "
                              "several days of continuous compute -- use --wind-grid/--battery to cut "
                              "cost, or pass a subset of seeds first.")
    parser.add_argument("--output-dir", default="hebbian_results", help="Where to save genomes/histories.")
    parser.add_argument("--stages", nargs="+", default=list(config.HEBBIAN_STAGES),
                         choices=list(config.HEBBIAN_STAGES),
                         help="Which stages to run, in order (default: all three).")
    parser.add_argument("--battery", type=float, default=None,
                         help=f"Starting battery for all agents (paper: {config.HEBBIAN_MAX_BATTERY}). "
                              "Lower this to cut simulation cost -- fewer steps until termination.")
    parser.add_argument("--wind-grid", type=int, default=None,
                         help=f"Wind grid resolution, both axes (paper/default: {config.HEBBIAN_NX}). "
                              "This is the single biggest cost lever: the wake-marching loop is O(Nx) "
                              "per simulation step, so e.g. --wind-grid 50 cuts wind-enabled stages by "
                              "roughly 10x at the cost of a coarser wake model.")
    parser.add_argument("--no-battery-sensor", action="store_true",
                         help="Evolve a baseline that cannot sense its own battery level at all (the "
                              "input is always fed a fixed 0 instead of the real reading), per the "
                              "paper's suggested follow-up experiment. Outputs get a '_nosensor' suffix "
                              "so they never overwrite the normal battery-aware runs.")
    args = parser.parse_args()

    if args.seeds is not None:
        seeds = args.seeds if len(args.seeds) > 0 else list(config.HEBBIAN_BATCH_SEEDS)
        print(f"🚀 Launching {len(seeds)}-seed Hebbian ABCD training sweep: seeds={seeds}, "
              f"stages={args.stages}, popsize={args.popsize}, maxiter={args.maxiter}, "
              f"n_agents={args.n_agents}, n_repeats={args.n_repeats}, "
              f"battery_sensor={not args.no_battery_sensor}")
        for i, seed in enumerate(seeds):
            seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
            print(f"\n{'#' * 70}\n### SEED {seed} ({i + 1}/{len(seeds)}) -> {seed_dir}/\n{'#' * 70}")
            train_one_seed(seed, seed_dir, args.stages, args.popsize, args.maxiter, args.n_agents,
                           args.n_repeats, args.battery, args.wind_grid, args.no_battery_sensor)
        print(f"\n🎉 {len(seeds)}-seed sweep complete. Results in '{args.output_dir}/seed_<seed>/'.")
    else:
        print(f"🚀 Launching staged Hebbian ABCD training: stages={args.stages}, "
              f"popsize={args.popsize}, maxiter={args.maxiter}, n_agents={args.n_agents}, "
              f"n_repeats={args.n_repeats}, battery_sensor={not args.no_battery_sensor}")
        train_one_seed(args.seed, args.output_dir, args.stages, args.popsize, args.maxiter,
                       args.n_agents, args.n_repeats, args.battery, args.wind_grid,
                       args.no_battery_sensor)
        print(f"\n🎉 Staged training complete. Results in '{args.output_dir}/'.")
