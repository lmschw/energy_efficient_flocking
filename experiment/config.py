"""Central configuration for the swarm simulation and CMA-ES optimization scripts.

Every previously-hardcoded constant from simulation_free_global_mod_2_LJ.py,
optimize.py, and run_batch_experiments.py lives here so they can be tuned in one
place instead of being scattered across files.
"""

import math

# --- Core simulation parameters ---
DT = 0.5                    # time-step [s]
N_AGENTS = 20                # number of agents (single-run scripts; batch overrides per config below)
ROBOT_RAD = 0.055            # robot radius [m]
WIND_RAD = 0.15              # robot's wind-occlusion radius [m]
X_RANGE = [-5.0, 5.0]        # simulation X bounds [m]
Y_RANGE = [-5.0, 5.0]        # simulation Y bounds [m]
V_AVG = 0.1                  # nominal forward speed used by move()
V_WIND = 10.0                # freestream wind speed

# --- Agent spawning ---
SPAWN_SQUARE_SIZE = 3.0      # side length of the square agents are randomly spawned in [m]
SPAWN_MIDPOINT = [0.0, 0.0]  # center of the spawn square
SPAWN_MIN_DIST_SLACK = 0.1   # extra slack (on top of 2*ROBOT_RAD) enforced between spawned agents

# Battery capacity -- this is what controls how long a simulation runs, since the
# main loop terminates as soon as any agent's battery reaches 0. Raise these to
# make simulations (and therefore videos/optimization runs) last longer.
MAX_BATTERY = 100.0          # starting battery for all agents but one
MIN_BATTERY = 100.0          # starting battery for the single "weakest" agent

# --- Collision / walls ---
COLLISION_MIN_DIST_SLACK = 0.01      # min_dist = COLLISION_MIN_DIST_SLACK + 2*ROBOT_RAD
WALL_MARGIN_FACTOR = 0.5             # wall_margin = ROBOT_RAD * WALL_MARGIN_FACTOR
WALL_COLLISION_WEIGHT = 3            # each wall hit counts as this many collisions

# --- Wind-tracking camera window (the x-range RayTraceCircularRobots is evaluated over) ---
WIND_TRACKING_WINDOW_WIDTH = 10.0    # total width of the tracking window [m] (nominally X_RANGE's span)
WIND_TRACKING_MAX_SPAN = 9.8         # cap on the swarm's own x-extent within that window [m]

# --- Wind / wake ray-tracing (RayTraceCircularRobots) ---
UINF = 100.0                  # freestream ("full power") wind value
NX = 200                      # grid steps in x
NY = 200                      # grid steps in y
KAPPA = 10.0                  # drag force scale factor
WAKE_RECOVERY_RATE = 1.0      # fraction of wake gap recovered per grid step outside a robot's radius
WAKE_PERCENT_DROP = 0.25      # wind intensity drop on entering/switching a robot's wake
WAKE_MAX_WALL_SPAN = 0.7      # controls how sharply the wall effect kicks in (lower = more wall effect)
WAKE_MIN_POWER_X = 30.0       # floor applied to power immediately behind a robot
WAKE_MIN_POWER_Y = 10.0       # floor applied to power after the wall-effect pass
WAKE_ALPHA = 0.5              # decay rate of the first smoothing kernel
WAKE_BETA = 0.5               # decay rate of the second smoothing kernel
WAKE_X_SMOOTHING_1 = 100      # first-pass smoothing kernel size divisor (x)
WAKE_Y_SMOOTHING_1 = 50       # first-pass smoothing kernel size divisor (y)
WAKE_X_SMOOTHING_2 = 50       # second-pass smoothing kernel size divisor (x)
WAKE_Y_SMOOTHING_2 = 50       # second-pass smoothing kernel size divisor (y)
WAKE_THR_OK_DELTA = 1.0       # a cell counts as "free-stream" once within this much of UINF

# --- Drag force (dragforce) ---
DRAG_UPSTREAM_LOOKAHEAD_FACTOR = 1.1   # how far upstream (in wind_rad) to sample the wind grid
DRAG_AIR_DENSITY = 1.225                # kg/m^3
DRAG_COEFFICIENT_AREA = 0.0045          # effective drag coefficient * frontal area

# --- Battery drainage (batterydrainage) ---
BATTERY_WHEEL_POWER_DIVISOR = 8.0    # divisor applied to summed absolute wheel speeds
BATTERY_MIN_DRAIN = 0.10             # floor on per-step drain (idle power draw)
BATTERY_DRAIN_SCALE = 2.0            # overall drain multiplier

# --- Flocking rule defaults (LJ spacing + alignment + goal), used when no genome is supplied ---
DEFAULT_RULES = {
    "r0": 0.70,        # desired inter-robot spacing [m]
    "epsilon": 0.5,    # LJ force depth/scale
    "k_align": 0.0,    # heading alignment gain
    "k_goal": 3.0,     # pull-to-goal gain
    "K1": 0.05,        # force -> linear speed scaling
    "K2": 0.5,         # force -> angular speed scaling
    "U": 0.005,        # constant forward speed bias
}
R_CUT = 3.0        # LJ interaction cutoff radius [m]
R_MIN = 0.0         # LJ singularity guard radius [m]
R_ALIGN = 1.5       # neighbor radius used for heading alignment [m]
LINEAR_VEL_MAX = 0.20            # robot's max linear speed [m/s]
ANGULAR_VEL_MAX = math.pi / 5    # robot's max angular speed [rad/s]

# Order genomes (flat 7-value arrays saved/loaded as .npy) are packed in.
GAIN_NAMES = list(DEFAULT_RULES.keys())


def genome_to_rules(genome):
    """Map a flat genome array (in GAIN_NAMES order) to the named rules dict."""
    return dict(zip(GAIN_NAMES, genome))

# --- Video output (visualize=True) ---
VIDEO_PATH = "alone.mp4"
VIDEO_FPS = 10.0
VIDEO_SIZE = (1200, 800)
VIDEO_FIGSIZE = (12, 8)
VIDEO_VIEWPORT_HALF_WIDTH = 5.0   # camera half-width/height around the swarm's center of mass [m]
VIDEO_ARROW_LEN = 0.3             # heading-arrow length in the rendered frame [m]
VIDEO_QUIVER_WIDTH = 0.004        # heading-arrow line width

# --- Fitness function weighting (eff = dist_travelled + avg_batt/W1 - collision_time/W2) ---
EFF_BATTERY_WEIGHT = 5.0
EFF_COLLISION_WEIGHT = 250.0

# --- CMA-ES defaults shared by optimize.py and run_batch_experiments.py ---
CMAES_INITIAL_GUESS = [
    DEFAULT_RULES["r0"], DEFAULT_RULES["epsilon"], DEFAULT_RULES["k_align"],
    DEFAULT_RULES["k_goal"], DEFAULT_RULES["K1"], DEFAULT_RULES["K2"], DEFAULT_RULES["U"],
]
CMAES_SIGMA0 = 0.15
CMAES_BOUNDS = [
    [0.2, 0.01, -1.0, 0.0, 0.001, 0.01, -0.05],   # lower bounds
    [2.0, 2.0, 1.0, 10.0, 0.5, 2.0, 0.1],          # upper bounds
]

# --- optimize.py (single run) ---
OPTIMIZE_SEED = 42
OPTIMIZE_POPSIZE = 12
OPTIMIZE_MAXITER = 40
OPTIMIZE_GENOME_OUT_PATH = "optimized_gains.npy"

# --- run_batch_experiments.py (multi-scale, multi-seed batch) ---
BATCH_CONFIGS = {
    "small_scale": 4,
    "baseline": 10,
    "large_scale": 40,
}
BATCH_MASTER_SEEDS = [42, 123, 777, 2026, 888, 99, 412, 555, 1010, 8432]
BATCH_POPSIZE = 20
BATCH_MAXITER = 60
BATCH_OUTPUT_DIR = "optimization_results"
BATCH_FITNESS_PLOT_PATH = "batch_fitness_curve.png"

# --- PyBullet rigid-body backend (simulation_free_global_mod_2_LJ(..., backend="pybullet")) ---
# Real dynamics: agents are cylinders with mass/inertia/friction, driven by a simple
# force/torque controller that chases the same (u, w) commands the numpy backend uses
# kinematically, then integrated + collision-resolved by PyBullet. These physical
# constants are first-pass estimates (roughly Thymio-II scale) -- they were not
# produced by any fitting process and will likely need their own tuning pass (and the
# evolved LJ gains likely need re-optimizing against this backend specifically, since
# they were tuned against the kinematic model's very different dynamics).
PYBULLET_INTERNAL_TIMESTEP = 1.0 / 240.0   # PyBullet substep size [s]; DT is divided into this many substeps
PYBULLET_MASS = 0.27                        # robot mass [kg]
PYBULLET_BODY_HEIGHT = 0.05                 # cylinder height [m]
PYBULLET_BODY_FRICTION = 0.05               # lateral friction, robot body -- kept low since the drive
PYBULLET_GROUND_FRICTION = 0.05             # is a direct external force/torque (no wheel/ground
                                             # traction model), so high friction here just resists the
                                             # thrust rather than providing propulsion
PYBULLET_LINEAR_DAMPING = 0.04              # velocity damping (rolling/air resistance stand-in);
                                             # small on purpose -- PyBullet re-applies damping every
                                             # internal substep (PYBULLET_INTERNAL_TIMESTEP), and DT is
                                             # divided into ~120 of them, so it compounds fast
PYBULLET_ANGULAR_DAMPING = 0.1              # yaw-rate damping (same per-substep caveat as above)
PYBULLET_RESTITUTION = 0.1                  # collision bounciness (0=inelastic, 1=elastic)
PYBULLET_FORCE_GAIN = 0.5                   # N per (m/s) of forward-speed tracking error
PYBULLET_MAX_FORCE = 0.1                    # N, drive force cap -- sized so a full DT at max force
                                             # accelerates the tiny (0.27kg) body by roughly one
                                             # LINEAR_VEL_MAX, not many multiples of it
PYBULLET_TORQUE_GAIN = 0.002                # N*m per (rad/s) of yaw-rate tracking error
PYBULLET_MAX_TORQUE = 0.001                 # N*m, drive torque cap -- sized against this body's tiny
                                             # moment of inertia (~4e-4 kg*m^2) so DT at max torque
                                             # accelerates yaw rate by roughly one ANGULAR_VEL_MAX,
                                             # not the ~100x overshoot an unscaled torque cap gives
PYBULLET_WALL_HALF_LENGTH = 1000.0          # m; the arena is only bounded in y (agents migrate
                                             # indefinitely in -x, same as the numpy backend), so the
                                             # top/bottom walls are made very long rather than boxed in x
PYBULLET_WALL_THICKNESS = 0.05              # m, half-extent
PYBULLET_WALL_HEIGHT = 0.2                  # m
PYBULLET_USE_GUI = False                    # open a live PyBullet GUI window when visualize=True
                                             # (in addition to the usual matplotlib/cv2 video recording)
