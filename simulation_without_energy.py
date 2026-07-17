import numpy as np

class AdvancedThymioSwarmEnv:
    def __init__(self, num_robots=15, arena_size=12.0, num_sensors=7):
        self.num_robots = num_robots
        self.arena_size = arena_size
        self.num_sensors = num_sensors
        self.num_inputs = num_sensors + 1  
        self.num_outputs = 2  
        
        spawn_center = arena_size / 2.0
        self.positions = np.random.uniform(spawn_center - 0.15, spawn_center + 0.15, (num_robots, 2))
        self.headings = np.random.uniform(-np.pi/4, np.pi/4, num_robots) # Symmetric forward bias
        
        self.W = np.zeros((num_robots, self.num_outputs, self.num_inputs))
        
        self.robot_radius = 0.08  
        self.wheel_axis = 0.09    
        self.dt = 0.1             
        self.sigma = 0.4          
        self.epsilon = 0.15       # Realigned interaction coefficient
        
        self.ray_angles = np.array([
            -np.pi/4, -np.pi/8, 0, np.pi/8, np.pi/4,  
            -5*np.pi/6, 5*np.pi/6                     
        ])
        self.max_sensor_range = 0.6  

    def get_sensor_readings(self):
        X_sensors = np.zeros((self.num_robots, self.num_sensors))
        for i in range(self.num_robots):
            p_k = self.positions[i]
            theta_k = self.headings[i]
            for r_idx, alpha in enumerate(self.ray_angles):
                global_angle = theta_k + alpha
                u = np.array([np.cos(global_angle), np.sin(global_angle)])
                min_t = self.max_sensor_range
                for j in range(self.num_robots):
                    if i == j: continue
                    v = self.positions[j] - p_k
                    t_proj = np.dot(v, u)
                    if t_proj < 0: continue
                    d_perp_sq = np.dot(v, v) - t_proj**2
                    if d_perp_sq <= self.robot_radius**2:
                        t_intersect = t_proj - np.sqrt(self.robot_radius**2 - d_perp_sq)
                        if 0 <= t_intersect < min_t:
                            min_t = t_intersect
                X_sensors[i, r_idx] = 1.0 - (min_t / self.max_sensor_range)
        bias_nodes = np.ones((self.num_robots, 1))
        return np.hstack((X_sensors, bias_nodes))

    def compute_lennard_jones_drift(self):
        diff = self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        dist = np.clip(dist, 0.05, None)
        
        force_mag = self.epsilon * ((self.sigma / dist)**4 - (self.sigma / dist)**2)
        force_mag = np.clip(force_mag, -0.2, 0.2)
        
        direction_vectors = diff / dist[:, :, np.newaxis]
        return np.sum(-force_mag[:, :, np.newaxis] * direction_vectors, axis=1)

    def step(self, A, B, C, D, eta=0.01, w_max=2.0):
        X = self.get_sensor_readings()
        X_col = X[:, :, np.newaxis]
        Y = np.tanh(np.matmul(self.W, X_col).squeeze(2))
        
        X_hebb = X[:, np.newaxis, :]   
        Y_hebb = Y[:, :, np.newaxis]   
        
        dW = eta * (A * (Y_hebb @ X_hebb) + B * X_hebb + C * Y_hebb + D)
        self.W = np.clip(self.W + dW, -w_max, w_max)
        
        # Base neural outputs mapping to motor speed targets
        v_linear_neural = ((Y[:, 1] * 0.5) + (Y[:, 0] * 0.5)) / 2.0
        v_angular_neural = ((Y[:, 1] * 0.5) - (Y[:, 0] * 0.5)) / self.wheel_axis
        
        # 2D Lennard Jones vectors acting on the chassis
        lj_forces = self.compute_lennard_jones_drift()
        
        # === 1:1 NON-HOLONOMIC PROJECTION ===
        # Project 2D forces into the robot's local frame of reference
        cos_t = np.cos(self.headings)
        sin_t = np.sin(self.headings)
        
        # Forward pushing force components
        f_forward = lj_forces[:, 0] * cos_t + lj_forces[:, 1] * sin_t
        # Cross product components (turning torque)
        f_rot = -lj_forces[:, 0] * sin_t + lj_forces[:, 1] * cos_t
        
        # Merge physical interaction vectors with wheel velocities
        v_linear = v_linear_neural + f_forward
        v_angular = v_angular_neural + (f_rot / self.wheel_axis)
        
        # Update kinematics strictly along wheels (No sideways slipping!)
        self.headings += v_angular * self.dt
        self.positions[:, 0] += v_linear * np.cos(self.headings) * self.dt
        self.positions[:, 1] += v_linear * np.sin(self.headings) * self.dt