import sys
import numpy as np
import cma
import config
from simulation_free_global_mod_2_LJ import simulation_free_global_mod_2_LJ
from fitness_plot import FitnessPlotter

current_candidate = 0
total_candidates = 0

def fitness_wrapper(genome):
    global current_candidate
    current_candidate += 1

    sys.stdout.write(f"\r   ↳ Evaluating Swarm Candidate: {current_candidate}/{total_candidates} ...")
    sys.stdout.flush()

    rules = config.genome_to_rules(genome)

    try:
        eff, _, _, _ = simulation_free_global_mod_2_LJ(rules=rules, seed=config.OPTIMIZE_SEED, visualize=False)
        return -eff
    except Exception:
        return 99999.0

if __name__ == "__main__":
    print("🚀 Launching Pure 1:1 CMA-ES Optimizer...")

    initial_guess = config.CMAES_INITIAL_GUESS

    options = {
        'popsize': config.OPTIMIZE_POPSIZE,
        'maxiter': config.OPTIMIZE_MAXITER,
        'bounds': config.CMAES_BOUNDS,
    }

    es = cma.CMAEvolutionStrategy(initial_guess, config.CMAES_SIGMA0, options)
    total_candidates = options['popsize']
    plotter = FitnessPlotter()

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
        print(f"✅ Generation {gen:02d} Complete | Best Negative Efficiency (Loss): {min(fitness_values):.4f} | fitness_curve.png updated")

    plotter.close()
    best_genome = es.result[0]
    np.save(config.OPTIMIZE_GENOME_OUT_PATH, best_genome)
    print(f"\n💾 Optimization Complete! Parameters exported to '{config.OPTIMIZE_GENOME_OUT_PATH}'.")

    print(f"🎬 Rendering optimized playback video ('{config.VIDEO_PATH}')...")
    best_rules = config.genome_to_rules(best_genome)
    simulation_free_global_mod_2_LJ(rules=best_rules, seed=config.OPTIMIZE_SEED, visualize=True)