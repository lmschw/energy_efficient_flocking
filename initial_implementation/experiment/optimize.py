import argparse
import sys
import numpy as np
import cma
import config
from simulation_free_global_mod_2_LJ import simulation_free_global_mod_2_LJ
from fitness_plot import FitnessPlotter

current_candidate = 0
total_candidates = 0
active_backend = "numpy"
active_battery_aware = True

def fitness_wrapper(genome):
    global current_candidate
    current_candidate += 1

    sys.stdout.write(f"\r   ↳ Evaluating Swarm Candidate: {current_candidate}/{total_candidates} ...")
    sys.stdout.flush()

    rules = config.genome_to_rules(genome)

    try:
        eff, _, _, _ = simulation_free_global_mod_2_LJ(rules=rules, seed=config.OPTIMIZE_SEED,
                                                        visualize=False, backend=active_backend,
                                                        battery_aware=active_battery_aware)
        return -eff
    except ModuleNotFoundError:
        raise  # environment/setup problem (e.g. missing pybullet), not a bad genome -- don't hide it
    except Exception as e:
        print(f"\n⚠️  Candidate {current_candidate} failed ({type(e).__name__}: {e}) -- penalizing with 99999.0")
        return 99999.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-run CMA-ES optimizer.")
    parser.add_argument("--backend", choices=["numpy", "pybullet"], default="numpy",
                         help="Physics backend: 'numpy' (kinematic, matches MATLAB) or "
                              "'pybullet' (real rigid-body dynamics).")
    parser.add_argument("--no-battery-objective", action="store_true",
                         help="Evolve against distance/collisions only, dropping the battery term "
                              "from the fitness function entirely. Produces a baseline genome to "
                              "compare against the normal battery-aware one (e.g. with "
                              "visualize_best.py --plot-battery), to check the battery term is "
                              "actually doing something.")
    args = parser.parse_args()
    active_backend = args.backend
    active_battery_aware = not args.no_battery_objective

    print(f"🚀 Launching Pure 1:1 CMA-ES Optimizer... [backend={active_backend}, "
          f"battery_aware={active_battery_aware}]")

    initial_guess = config.CMAES_INITIAL_GUESS

    options = {
        'popsize': config.OPTIMIZE_POPSIZE,
        'maxiter': config.OPTIMIZE_MAXITER,
        'bounds': config.CMAES_BOUNDS,
    }

    es = cma.CMAEvolutionStrategy(initial_guess, config.CMAES_SIGMA0, options)
    total_candidates = options['popsize']

    # Namespace outputs by backend/objective so different runs never overwrite each other
    suffix = ("" if active_backend == "numpy" else f"_{active_backend}") + \
             ("" if active_battery_aware else "_nobattery")
    genome_out_path = config.OPTIMIZE_GENOME_OUT_PATH.replace(".npy", f"{suffix}.npy")
    fitness_plot_path = f"fitness_curve{suffix}.png"
    plotter = FitnessPlotter(path=fitness_plot_path)

    gen = 0
    while not es.stop():
        gen += 1
        current_candidate = 0
        print(f"\n📊 Starting Generation {gen:02d}:")

        solutions = es.ask()
        fitness_values = [fitness_wrapper(sol) for sol in solutions]
        es.tell(solutions, fitness_values)
        plotter.update(gen, fitness_values)

        sys.stdout.write("\r")
        print(f"✅ Generation {gen:02d} Complete | Best Negative Efficiency (Loss): {min(fitness_values):.4f} | {fitness_plot_path} updated")

    plotter.close()
    best_genome = es.result[0]
    np.save(genome_out_path, best_genome)
    print(f"\n💾 Optimization Complete! Parameters exported to '{genome_out_path}'.")

    print(f"🎬 Rendering optimized playback video ('{config.VIDEO_PATH}')...")
    best_rules = config.genome_to_rules(best_genome)
    simulation_free_global_mod_2_LJ(rules=best_rules, seed=config.OPTIMIZE_SEED, visualize=True, backend=active_backend)