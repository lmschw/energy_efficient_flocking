import numpy as np
import cma
from without_energy.simulation_without_energy import AdvancedThymioSwarmEnv

NUM_SENSORS = 7
NUM_INPUTS = NUM_SENSORS + 1  
NUM_OUTPUTS = 2
MATRIX_SIZE = NUM_OUTPUTS * NUM_INPUTS  

def unpack_genome(flat_genome):
    idx = MATRIX_SIZE
    A = flat_genome[0:idx].reshape(NUM_OUTPUTS, NUM_INPUTS)
    B = flat_genome[idx:2*idx].reshape(NUM_OUTPUTS, NUM_INPUTS)
    C = flat_genome[2*idx:3*idx].reshape(NUM_OUTPUTS, NUM_INPUTS)
    D = flat_genome[3*idx:4*idx].reshape(NUM_OUTPUTS, NUM_INPUTS)
    return A, B, C, D

def evaluate_gated_flocking(flat_genome, evaluation_steps=220):
    A, B, C, D = unpack_genome(flat_genome)
    env = AdvancedThymioSwarmEnv(num_robots=12, arena_size=12.0, num_sensors=NUM_SENSORS)
    
    step_rewards = []
    
    for _ in range(evaluation_steps):
        old_cm = np.mean(env.positions, axis=0)
        env.step(A, B, C, D)
        current_cm = np.mean(env.positions, axis=0)
        
        # Center of mass translation velocity
        v_cm = np.linalg.norm(current_cm - old_cm) / env.dt
        
        # Global swarm alignment metric (Polarization)
        mean_cos = np.mean(np.cos(env.headings))
        mean_sin = np.mean(np.sin(env.headings))
        pol_score = np.sqrt(mean_cos**2 + mean_sin**2)
        
        # Calculate localized dispersion spread
        distances = np.linalg.norm(env.positions - current_cm, axis=1)
        avg_cohesion = np.mean(distances)
        
        # === MULTIPLICATIVE COHESION GATE ===
        # If they remain inside a tight 0.5m radius pack, gate = 1.0.
        # If they begin drifting apart, the multiplier drops off exponentially.
        cohesion_gate = np.exp(-3.0 * max(0.0, avg_cohesion - 0.5))
        
        # Mathematical product rules out splitting behavior entirely
        step_rewards.append(v_cm * pol_score * cohesion_gate)
        
    return -np.mean(step_rewards)

if __name__ == "__main__":
    print(f"🚀 Launching Gated Kinematics Optimizer...")
    
    initial_guess = np.zeros(MATRIX_SIZE * 4)
    options = {'popsize': 16, 'maxiter': 65}
    es = cma.CMAEvolutionStrategy(initial_guess, 0.25, options)
    
    generation_idx = 0
    
    while not es.stop():
        generation_idx += 1
        solutions = es.ask()
        
        fitness_values = [evaluate_gated_flocking(sol) for sol in solutions]
        es.tell(solutions, fitness_values)
        
        print(f"Gen {generation_idx:03d} | Gated Flocking Cost: {min(fitness_values):.4f}")
        
    np.save("winning_abcd_rules.npy", es.result[0])
    print("\n💾 Optimal parameters exported to 'winning_abcd_rules.npy'.")