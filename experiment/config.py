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
MAX_BATTERY = 150.0          # starting battery for all agents but one (1.5x the original 100.0,
                              # for ~1.5x longer runs -- drain rate is roughly constant per step)
MIN_BATTERY = 150.0          # starting battery for the single "weakest" agent

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
KAPPA = 20.0                  # drag force scale factor (2x'd from 10.0 -- battery drain from wind
                               # exposure enters as v_rel^2, so this is the strongest lever on how
                               # fast batteries actually deplete; see BATTERY_WHEEL_POWER_DIVISOR too)
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
BATTERY_WHEEL_POWER_DIVISOR = 4.0    # divisor applied to summed absolute wheel speeds (2x'd from 8.0
                                      # -- at typical cruise speeds this term was well under
                                      # BATTERY_MIN_DRAIN, so movement barely affected drain at all)
BATTERY_MIN_DRAIN = 0.10             # floor on per-step drain (idle power draw) -- left as-is; the
                                      # two constants above raise the movement/wind terms so they
                                      # actually exceed this floor under normal operation, rather than
                                      # lowering the floor itself
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

# Paper's Table 3 "standard collective motion baseline" -- the cluster-4 comparison point
# in Fig. 5a. NOT the same as DEFAULT_RULES above: that dict is simulation_free_global_
# mod_2_LJ.m's own hardcoded example starting point (epsilon=0.5, U=0.005), which differs
# from the paper's literal baseline (epsilon=1, no U term). Used by analyze_hebbian_results.py.
PAPER_BASELINE_RULES = {
    "r0": 0.70,        # d_des
    "epsilon": 1.0,    # paper Table 3 (vs. DEFAULT_RULES' 0.5)
    "k_align": 0.0,    # baseline uses "only proximal control and goal-seeking terms"
    "k_goal": 3.0,     # K_goal
    "K1": 0.05,
    "K2": 0.5,
    "U": 0.0,          # Table 3 has no constant-bias term; DEFAULT_RULES' 0.005 is LJ-file-specific
}
# Table 3 also lists alpha=2 ("steepness of potential function") and Dp=3 ("max interaction
# range"); Dp matches R_CUT above, but alpha has no equivalent here -- simulation_free_global_
# mod_2_LJ.py implements a fixed-exponent (12/6) Lennard-Jones force with no tunable steepness.

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

# --- PyBullet-rendered video (backend="pybullet", visualize=True) ---
# Rendered from PyBullet's own camera/rasterizer instead of the matplotlib wind-field
# view, so it actually shows the rigid-body dynamics (sliding, bouncing, real collisions).
PYBULLET_VIDEO_PATH = "alone_pybullet.mp4"
PYBULLET_VIDEO_FOV = 45.0                # camera field of view [deg]
PYBULLET_VIDEO_PITCH = -89.9             # camera pitch [deg]; ~straight down, comparable to the
                                          # numpy backend's top-down matplotlib view
PYBULLET_VIDEO_YAW = 0.0                 # camera yaw [deg]
PYBULLET_VIDEO_DISTANCE_FACTOR = 2.2     # camera distance = VIDEO_VIEWPORT_HALF_WIDTH * this

# --- Fitness function weighting ---
# eff = EFF_DISTANCE_WEIGHT*dist_travelled + avg_batt/EFF_BATTERY_WEIGHT - collision_time/EFF_COLLISION_WEIGHT
#
# dist_travelled must be the dominant, load-bearing term -- "go toward the goal" should win for
# every genome, battery-aware or not. average_batt is a secondary shaping signal on top of that:
# strong enough that battery-aware evolution can still discover energy-saving coordination (e.g.
# agents rotating through the exposed front position), but too weak to make "don't move" a
# competitive alternative to "travel efficiently". Raised from a 1:5 to a 3:10 ratio between these
# after seeing evolved genomes collapse into a stationary cluster instead of migrating -- at the old
# ratio, average_batt/5 (up to MAX_BATTERY/5 = 30) was comparable in magnitude to dist_travelled
# (typically 10-30), so preserving battery was competing with progress rather than refining it.
#
# Separately: the LJ separation force sums over every neighbor while the goal force is a fixed
# per-agent constant, so genomes that travel straight at small scale (e.g. n_agents=4) can degrade
# into near-stationary wandering at larger scale (e.g. n_agents=20, straightness ratio -- net
# displacement / path length -- dropping from 0.98 to 0.27 for the same genome). Per-request, we're
# not changing that force law; instead pushing EFF_DISTANCE_WEIGHT higher so evolution is pressured
# to find K1/k_goal values (within CMAES_BOUNDS) that overcome it AT WHATEVER SCALE IT'S EVOLVED AT.
# This means genomes must be (re-)evolved at the agent count they'll actually run at.
EFF_DISTANCE_WEIGHT = 8.0
EFF_BATTERY_WEIGHT = 10.0
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

# =====================================================================================
# --- Hebbian ABCD neural-network controller (paper replication) ---
# Reproduces "Energy-Efficient Flocking in Self-Organized Robot Swarms" (Mahdavi et al.,
# ANTS 2026): each robot runs a 10-10-10-2 MLP (ReLU, ReLU, tanh) updated online by a
# Hebbian rule; the rule's coefficients (not the weights themselves) are what CMA-ES
# evolves, shared by every agent in a swarm. See hebbian_controller.py, sensor_model.py,
# simulation_hebbian.py, optimize_hebbian.py. MATLAB reference: hebbianStep.m (forward
# pass + update), simulation_free_global_mod_2.m's getsensordata()/W(i) init loop.
# =====================================================================================

# --- Robot & sensing (Section 2.1) ---
HEBBIAN_N_AGENTS = 20             # swarm size used throughout the paper's experiments
HEBBIAN_SENSING_RADIUS = 2.01     # R: neighbor detection radius [m]; also the "no neighbor" default distance
HEBBIAN_LINEAR_VEL_MAX = 0.2      # m/s, tanh output #1 rescaled to [-this, this]
HEBBIAN_ANGULAR_VEL_MAX = math.pi / 5  # rad/s, tanh output #2 rescaled to [-this, this]

# --- Battery & wind grid (Eq. 6 / Section 3) ---
# Deliberately separate from MAX_BATTERY/MIN_BATTERY/NX/NY above: those were later
# bumped (100->150, and drain-rate constants) for the unrelated LJ-model experiments,
# but the paper's battery model -- and this module's B/50-1 sensor normalization -- is
# defined over B in [0, 100] (Eq. 6). Sharing the LJ config here would have silently
# fed out-of-range values into the NN's battery input.
HEBBIAN_MAX_BATTERY = 100.0        # starting battery for all agents but one
HEBBIAN_MIN_BATTERY = 100.0        # starting battery for the single "weakest" agent
                                    # (also reused directly by the battery-awareness
                                    # experiment in analyze_hebbian_results.py, set to 50)
HEBBIAN_NX = 200                   # wind grid resolution; lower to cut simulation cost
HEBBIAN_NY = 200                   # (the O(Nx) wake-marching loop dominates per-step cost)

# --- Neural controller architecture (Section 2.1) ---
HEBBIAN_N_INPUTS = 10             # 4 quadrants x (distance, bearing) + battery + compass heading
HEBBIAN_N_HIDDEN = 10             # both hidden layers
HEBBIAN_N_OUTPUTS = 2             # (v, w)
HEBBIAN_LEARNING_RATE = 0.1       # mu in delta_w = mu*(A*ni*nj + B*ni + C*nj + D)  (Eq. 1)
# Weight-matrix shapes, in flatten/unflatten order (matches evaluateABCD.m's unflattenABCD):
# W1: N_INPUTS x N_HIDDEN, W2: N_HIDDEN x N_HIDDEN, W3: N_HIDDEN x N_OUTPUTS.
# Paper: "randomly initialized ... using a uniform distribution in [-1, 1]" for all three;
# the MATLAB source (simulation_free_global_mod_2.m) actually samples W1 from randn() (a
# normal, unbounded distribution) instead of rand() -- we follow the paper's stated spec
# (uniform for all three) since that's the actual written methodology.
HEBBIAN_WEIGHT_INIT_RANGE = 1.0

# --- ABCD genotype (Section 2.1-2.2) ---
# 4 coefficients (A, B, C, D) per NN weight, shared across all agents in a swarm:
# 4 * (10*10 + 10*10 + 10*2) = 880 total parameters.
HEBBIAN_N_ABCD = 4 * (HEBBIAN_N_INPUTS * HEBBIAN_N_HIDDEN + HEBBIAN_N_HIDDEN * HEBBIAN_N_HIDDEN
                      + HEBBIAN_N_HIDDEN * HEBBIAN_N_OUTPUTS)
HEBBIAN_ABCD_INIT_RANGE = 5.0     # ABCD-rules initial mean sampled uniformly from [-this, this]
HEBBIAN_ABCD_BOUNDS = [-5.0, 5.0]  # CMA-ES hard bounds (opts.LBounds/UBounds in optimizeABCD.m)

# --- CMA-ES hyperparameters (Table 1) ---
HEBBIAN_CMAES_POPSIZE = 30        # lambda
HEBBIAN_CMAES_GEN_MAX = 100       # Ngen, termination condition, PER STAGE
HEBBIAN_CMAES_SIGMA0 = 0.3        # initial covariance/step-size
HEBBIAN_N_REPEATS = 3             # simulations per candidate (different seeds); fitness = median

# --- Staged curriculum (Section 2.3 / Table 2 / Fig. 1) ---
# Stage 1 has no wind and rewards distance only, to avoid evolving the trivial strategy of
# just riding the tailwind. Stage 2 turns on wind and adds battery + wall-collision terms.
# Stage 3 adds a general inter-robot collision penalty on top of stage 2, hypothesized to
# be what pushes evolution toward formation-reconfiguration strategies. Each stage's CMA-ES
# run is seeded from the previous stage's best genome ("Next stage: best x is initial x" in
# Fig. 1); stage 1 alone starts from a fresh uniform-random ABCD_init.
HEBBIAN_STAGES = ("walk_left", "save_battery_avoid_wall", "save_battery_avoid_all")
HEBBIAN_STAGE_WIND_ENABLED = {
    "walk_left": False,
    "save_battery_avoid_wall": True,
    "save_battery_avoid_all": True,
}
# Fitness weights per stage: eff = dist + batt/BATTERY_W - (collision_time + WALL_COL_MULT *
# wall_collision_time) / COLLISION_W. A weight of None means that term is entirely absent
# (matching Table 2's stage 1 having no battery or collision terms, and stages 2/3 excluding
# inter-robot/wall collisions respectively from view of that specific denominator).
HEBBIAN_STAGE_FITNESS_WEIGHTS = {
    #                             battery_w   collision_w   wall_col_mult   include_inter_robot_collision
    "walk_left":                 (None,        None,         3.0,            False),
    "save_battery_avoid_wall":   (5.0,         500.0,        3.0,            False),
    "save_battery_avoid_all":    (5.0,         250.0,        3.0,            True),
}
