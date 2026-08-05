import argparse
import sys
import os
import json
import numpy as np
import cma

import config
# Import the module directly so we can dynamically manipulate its global variables
import simulation_free_global_mod_2_LJ
from fitness_plot import FitnessPlotter

# Global tracking variables for the active CMA-ES iteration
current_candidate = 0
total_candidates = 0
active_n_agents = config.N_AGENTS
active_trial_seed = config.OPTIMIZE_SEED
active_backend = "numpy"
active_battery_aware = True

def fitness_wrapper(genome):
    global current_candidate
    current_candidate += 1

    sys.stdout.write(f"\r   ↳ Evaluating Swarm Candidate: {current_candidate}/{total_candidates} ...")
    sys.stdout.flush()

    rules = config.genome_to_rules(genome)

    try:
        # Execute the simulation using the active trial seed and agent count
        eff, _, _, _ = simulation_free_global_mod_2_LJ.simulation_free_global_mod_2_LJ(
            rules=rules,
            seed=active_trial_seed,
            visualize=False,
            backend=active_backend,
            n_agents=active_n_agents,
            battery_aware=active_battery_aware
        )
        return -eff  # Negating because CMA-ES minimizes objectives
    except ModuleNotFoundError:
        raise  # environment/setup problem (e.g. missing pybullet), not a bad genome -- don't hide it
    except Exception as e:
        # Penalize a genome that causes physics/NaN instabilities, but say so instead of hiding it
        print(f"\n⚠️  Candidate {current_candidate} failed ({type(e).__name__}: {e}) -- penalizing with 99999.0")
        return 99999.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch CMA-ES optimization across scale configs and seeds.")
    parser.add_argument("--backend", choices=["numpy", "pybullet"], default="numpy",
                         help="Physics backend: 'numpy' (kinematic, matches MATLAB) or "
                              "'pybullet' (real rigid-body dynamics).")
    parser.add_argument("--no-battery-objective", action="store_true",
                         help="Evolve against distance/collisions only, dropping the battery term "
                              "from the fitness function entirely. Produces a baseline batch to "
                              "compare against the normal battery-aware results, to check the "
                              "battery term is actually doing something.")
    args = parser.parse_args()
    active_backend = args.backend
    active_battery_aware = not args.no_battery_objective

    print("=====================================================================")
    print(f"🚀 LAUNCHING PRODUCTION MULTI-SCALE & MULTI-SEED BATCH CMA-ES "
          f"[backend={active_backend}, battery_aware={active_battery_aware}]")
    print("=====================================================================")

    # 1. Scale configurations based on agent counts
    CONFIGS = config.BATCH_CONFIGS

    # 2. Explicit master seeds for the statistical replication trials
    MASTER_SEEDS = config.BATCH_MASTER_SEEDS
    NUM_TRIALS = len(MASTER_SEEDS)

    # Namespace outputs by backend/objective so different runs never overwrite/mix results
    suffix = ("" if active_backend == "numpy" else f"_{active_backend}") + \
             ("" if active_battery_aware else "_nobattery")
    output_dir = config.BATCH_OUTPUT_DIR + suffix
    os.makedirs(output_dir, exist_ok=True)
    fitness_plot_base, fitness_plot_ext = os.path.splitext(config.BATCH_FITNESS_PLOT_PATH)
    fitness_plot_path = f"{fitness_plot_base}{suffix}{fitness_plot_ext}"

    # Publication-grade CMA-ES Settings
    initial_guess = config.CMAES_INITIAL_GUESS
    production_options = {
        'popsize': config.BATCH_POPSIZE,
        'maxiter': config.BATCH_MAXITER,
        'bounds': config.CMAES_BOUNDS,
    }
    total_candidates = production_options['popsize']
    master_summary = []
    plotter = FitnessPlotter(path=fitness_plot_path)

    # Outer Loop: Scale Configurations
    for config_name, agent_count in CONFIGS.items():
        print(f"\n📂 Processing Configuration Category: [{config_name.upper()}] ({agent_count} Agents)")
        active_n_agents = agent_count
        
        # Inner Loop: 10 Statistical Replication Trials
        for trial_idx, trial_seed in enumerate(MASTER_SEEDS, start=1):
            print(f"\n🔬 Starting Trial {trial_idx:02d}/{NUM_TRIALS} using Seed: [{trial_seed}]")
            
            # Sync our environment and optimizer random number generator tracks
            active_trial_seed = trial_seed
            trial_options = production_options.copy()
            trial_options['seed'] = trial_seed 
            
            # Initialize a pristine strategy instance for this unique matrix environment sequence
            es = cma.CMAEvolutionStrategy(initial_guess, config.CMAES_SIGMA0, trial_options)
            plotter.reset_run(title=f"{config_name} | trial {trial_idx:02d}/{NUM_TRIALS} (seed {trial_seed})")

            gen = 0
            best_loss_history = []

            while not es.stop():
                gen += 1
                current_candidate = 0

                solutions = es.ask()
                fitness_values = [fitness_wrapper(sol) for sol in solutions]
                es.tell(solutions, fitness_values)
                plotter.update(gen, fitness_values)

                min_loss = min(fitness_values)
                best_loss_history.append(float(min_loss))

                sys.stdout.write("\r")
                print(f"   ↳ Gen {gen:02d}/{trial_options['maxiter']} Complete | Best Loss (Neg Eff): {min_loss:.4f} | {fitness_plot_path} updated")
            
            # Harvest best genome variables
            best_genome = es.result[0]
            final_best_loss = es.result[1]
            
            # Export optimized array data
            filename_base = f"opt_{config_name}_trial_{trial_idx:02d}_seed_{trial_seed}"
            np.save(os.path.join(output_dir, f"{filename_base}_gains.npy"), best_genome)
            
            trial_summary = {
                "config": config_name,
                "agents": agent_count,
                "trial": trial_idx,
                "seed": trial_seed,
                "backend": active_backend,
                "battery_aware": active_battery_aware,
                "best_loss": final_best_loss,
                "best_efficiency": -final_best_loss,
                "genome": best_genome.tolist(),
                "loss_curve": best_loss_history
            }
            master_summary.append(trial_summary)
            
            print(f"💾 Trial {trial_idx} Complete. Evolved Swarm Efficiency Score: {-final_best_loss:.4f}")

    plotter.close()

    # 3. Save out performance ledger logs
    with open(os.path.join(output_dir, "master_optimization_summary.json"), "w") as f:
        json.dump(master_summary, f, indent=4)
        
    csv_path = os.path.join(output_dir, "summary_table.csv")
    with open(csv_path, "w") as f:
        f.write("Configuration,Agents,Trial,Seed,Best_Loss,Best_Efficiency\n")
        for entry in master_summary:
            f.write(f"{entry['config']},{entry['agents']},{entry['trial']},{entry['seed']},{entry['best_loss']:.4f},{entry['best_efficiency']:.4f}\n")

    print("\n" + "="*70)
    print("🎉 HIGH-FIDELITY SEED-VARIED BATCH OPTIMIZATION COMPLETE!")
    print(f"📁 Check your local '{output_dir}/' folder for comprehensive summary files.")
    print("="*70)