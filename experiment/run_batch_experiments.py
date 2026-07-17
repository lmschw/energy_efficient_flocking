import sys
import os
import json
import numpy as np
import cma

# Import the module directly so we can dynamically manipulate its global variables
import simulation_free_global_mod_2_LJ

# Global tracking variables for the active CMA-ES iteration
current_candidate = 0
total_candidates = 0
active_n_agents = 20
active_trial_seed = 42

def fitness_wrapper(genome):
    global current_candidate
    current_candidate += 1
    
    sys.stdout.write(f"\r   ↳ Evaluating Swarm Candidate: {current_candidate}/{total_candidates} ...")
    sys.stdout.flush()
    
    rules = {
        'r0':      genome[0],
        'epsilon': genome[1],
        'k_align': genome[2],
        'k_goal':  genome[3],
        'K1':      genome[4],
        'K2':      genome[5],
        'U':       genome[6]
    }
    
    try:
        # Override the global n_agents variable inside your simulation module dynamically
        simulation_free_global_mod_2_LJ.n_agents = active_n_agents
        
        # Execute the simulation using the active trial seed
        eff, _, _, _ = simulation_free_global_mod_2_LJ.simulation_free_global_mod_2_LJ(
            rules=rules, 
            seed=active_trial_seed, 
            visualize=False
        )
        return -eff  # Negating because CMA-ES minimizes objectives
    except Exception as e:
        # Return a safe fallback penalty if an extreme genome causes physics/NaN instabilities
        return 99999.0  

if __name__ == "__main__":
    print("=====================================================================")
    print("🚀 LAUNCHING PRODUCTION MULTI-SCALE & MULTI-SEED BATCH CMA-ES")
    print("=====================================================================")
    
    # 1. Define 3 Scale Configurations based on agent counts
    CONFIGS = {
        "small_scale":  10,
        "baseline":     20,
        "large_scale":  40
    }
    
    # 2. 10 Explicit master seeds for the 10 statistical replication trials
    MASTER_SEEDS = [42, 123, 777, 2026, 888, 99, 412, 555, 1010, 8432]
    NUM_TRIALS = len(MASTER_SEEDS)
    
    output_dir = "optimization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Publication-grade CMA-ES Settings
    initial_guess = [0.70, 0.5, 0.0, 3.0, 0.05, 0.5, 0.005]
    production_options = {
        'popsize': 20,     # Broadened population size for high-fidelity gradient search
        'maxiter': 100,    # 100 generations provides ample runway for true behavioral convergence
        'bounds': [
            [0.2, 0.01, -1.0, 0.0, 0.001, 0.01, -0.05], # Lower Bounds
            [2.0, 2.0,   1.0, 10.0, 0.5,   2.0,   0.1]   # Upper Bounds
        ]
    }
    total_candidates = production_options['popsize']
    master_summary = []

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
            es = cma.CMAEvolutionStrategy(initial_guess, 0.15, trial_options)
            
            gen = 0
            best_loss_history = []
            
            while not es.stop():
                gen += 1
                current_candidate = 0 
                
                solutions = es.ask()
                fitness_values = [fitness_wrapper(sol) for sol in solutions]
                es.tell(solutions, fitness_values)
                
                min_loss = min(fitness_values)
                best_loss_history.append(float(min_loss))
                
                sys.stdout.write("\r")
                print(f"   ↳ Gen {gen:02d}/{trial_options['maxiter']} Complete | Best Loss (Neg Eff): {min_loss:.4f}")
            
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
                "best_loss": final_best_loss,
                "best_efficiency": -final_best_loss,
                "genome": best_genome.tolist(),
                "loss_curve": best_loss_history
            }
            master_summary.append(trial_summary)
            
            print(f"💾 Trial {trial_idx} Complete. Evolved Swarm Efficiency Score: {-final_best_loss:.4f}")

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