import sys
import numpy as np
import cma
from simulation_free_global_mod_2_LJ import simulation_free_global_mod_2_LJ

current_candidate = 0
total_candidates = 0

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
        eff, _, _, _ = simulation_free_global_mod_2_LJ(rules=rules, seed=42, visualize=False)
        return -eff  
    except Exception:
        return 99999.0  

if __name__ == "__main__":
    print("🚀 Launching Pure 1:1 CMA-ES Optimizer...")
    
    # Exact original MATLAB baseline values
    initial_guess = [0.70, 0.5, 0.0, 3.0, 0.05, 0.5, 0.005]
    
    options = {
        'popsize': 12,
        'maxiter': 40,
        'bounds': [
            [0.2, 0.01, -1.0, 0.0, 0.001, 0.01, -0.05], 
            [2.0, 2.0,   1.0, 10.0, 0.5,   2.0,   0.1]   
        ]
    }
    
    es = cma.CMAEvolutionStrategy(initial_guess, 0.15, options)
    total_candidates = options['popsize']
    
    gen = 0
    while not es.stop():
        gen += 1
        current_candidate = 0 
        print(f"\n📊 Starting Generation {gen:02d}:")
        
        solutions = es.ask()
        fitness_values = [fitness_wrapper(sol) for sol in solutions]
        es.tell(solutions, fitness_values)
        
        sys.stdout.write("\r")
        print(f"✅ Generation {gen:02d} Complete | Best Negative Efficiency (Loss): {min(fitness_values):.4f}")
        
    best_genome = es.result[0]
    np.save("optimized_gains.npy", best_genome)
    print("\n💾 Optimization Complete! Parameters exported to 'optimized_gains.npy'.")
    
    print("🎬 Rendering optimized playback video ('alone.mp4')...")
    best_rules = {
        'r0': best_genome[0], 'epsilon': best_genome[1], 'k_align': best_genome[2],
        'k_goal': best_genome[3], 'K1': best_genome[4], 'K2': best_genome[5], 'U': best_genome[6]
    }
    simulation_free_global_mod_2_LJ(rules=best_rules, seed=42, visualize=True)