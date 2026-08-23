"""Central configuration for the ANTS 2026 Hebbian ABCD replication (Mahdavi et al.,
"Energy-Efficient Flocking in Self-Organized Robot Swarms").

This is a standalone copy, split off from the original combined LJ+Hebbian config.py
(see ../../initial_implementation/experiment/config.py for the original LJ-model-only
project this replication grew out of). Physics constants that both models happen to
share (wind/wake ray-tracing, drag, battery drainage, spawning) are duplicated here
deliberately -- this file has zero import-time dependency on initial_implementation, by
design, so the two projects can evolve independently without one silently breaking the
other. If you tune one of the shared physics constants, decide explicitly whether the
same tuning belongs in both places; nothing here keeps them in sync automatically.
"""

import math

# --- Core simulation parameters (shared by the wind/drag/battery physics and the
# LJ Table-3 baseline comparison in analyze_hebbian_results.py) ---
DT = 0.5                    # time-step [s]
ROBOT_RAD = 0.055            # robot radius [m]
WIND_RAD = 0.15              # robot's wind-occlusion radius [m]
X_RANGE = [-5.0, 5.0]        # simulation X bounds [m]
Y_RANGE = [-5.0, 5.0]        # simulation Y bounds [m]
V_WIND = 10.0                # freestream wind speed

# --- Agent spawning ---
SPAWN_SQUARE_SIZE = 3.0      # side length of the square agents are randomly spawned in [m]
SPAWN_MIDPOINT = [0.0, 0.0]  # center of the spawn square
SPAWN_MIN_DIST_SLACK = 0.1   # extra slack (on top of 2*ROBOT_RAD) enforced between spawned agents

# --- Collision / walls ---
COLLISION_MIN_DIST_SLACK = 0.01      # min_dist = COLLISION_MIN_DIST_SLACK + 2*ROBOT_RAD
WALL_MARGIN_FACTOR = 0.5             # wall_margin = ROBOT_RAD * WALL_MARGIN_FACTOR
WALL_COLLISION_WEIGHT = 3            # each wall hit counts as this many collisions (LJ baseline only)

# --- Wind-tracking camera window (the x-range RayTraceCircularRobots is evaluated over) ---
WIND_TRACKING_WINDOW_WIDTH = 10.0    # total width of the tracking window [m] (nominally X_RANGE's span)
WIND_TRACKING_MAX_SPAN = 9.8         # cap on the swarm's own x-extent within that window [m]

# --- Wind / wake ray-tracing (RayTraceCircularRobots) ---
UINF = 100.0                  # freestream ("full power") wind value
KAPPA = 20.0                  # drag force scale factor -- battery drain from wind exposure enters
                               # as v_rel^2, so this is the strongest lever on drain rate
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
BATTERY_WHEEL_POWER_DIVISOR = 4.0    # divisor applied to summed absolute wheel speeds
BATTERY_MIN_DRAIN = 0.10             # floor on per-step drain (idle power draw)
BATTERY_DRAIN_SCALE = 2.0            # overall drain multiplier

# --- Video output (visualize_hebbian.py) ---
HEBBIAN_VIDEO_PATH = "hebbian_alone.mp4"
VIDEO_FPS = 10.0
VIDEO_SIZE = (1200, 800)
VIDEO_FIGSIZE = (12, 8)
VIDEO_VIEWPORT_HALF_WIDTH = 5.0   # camera half-width/height around the swarm's center of mass [m]
VIDEO_ARROW_LEN = 0.3             # heading-arrow length in the rendered frame [m]
VIDEO_QUIVER_WIDTH = 0.004        # heading-arrow line width

# --- Default seed for one-off playback (visualize_hebbian.py) ---
HEBBIAN_DEFAULT_SEED = 42

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
# The paper's battery model -- and this module's B/50-1 sensor normalization -- is
# defined over B in [0, 100] (Eq. 6).
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

# The 10 canonical seeds this project uses whenever a script wants several independent
# trials rather than trusting a single stochastic run. A literal duplicate of
# initial_implementation's BATCH_MASTER_SEEDS -- kept as its own list here (rather than
# importing across the project boundary) so this config has no dependency on the other
# project; used by optimize_hebbian.py's --seeds flag (with no explicit values) to run
# the entire staged curriculum once per seed.
HEBBIAN_BATCH_SEEDS = [42, 123, 777, 2026, 888, 99, 412, 555, 1010, 8432]

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
# Fitness weights per stage: eff = HEBBIAN_EFF_DISTANCE_WEIGHT*dist
# - HEBBIAN_STRAIGHTNESS_WEIGHT*y_drift + batt/BATTERY_W - collision_time/COLLISION_W.
# A weight of None means that term is entirely absent (matching Table 2's stage 1 having
# no battery or collision terms). NOTE: unlike the original 3-stage curriculum, there is
# no wall-specific term here at all anymore -- see HEBBIAN_STRAIGHTNESS_WEIGHT below for
# why, and note "save_battery_avoid_wall" is a bit of a misnomer in this variant (its old
# meaning -- a dedicated wall-collision penalty -- no longer exists; what's left is just
# "battery+wind, no collision term of any kind"). collision_w now covers ONLY inter-robot
# collisions (when include_inter_robot_collision=True), never walls.
HEBBIAN_STAGE_FITNESS_WEIGHTS = {
    #                             battery_w   collision_w   include_inter_robot_collision
    "walk_left":                 (None,        None,         False),
    "save_battery_avoid_wall":   (5.0,         None,         False),
    "save_battery_avoid_all":    (5.0,         250.0,        True),
}

# Explicit distance weight, mirroring the LJ model's own EFF_DISTANCE_WEIGHT --
# Table 2's literal formula has no such multiplier (dist_travelled has an implicit
# weight of 1.0), which measurably let CMA-ES discover that barely moving is a cheap way
# to preserve battery: real training data showed dist_travelled collapsing from 84.0
# (stage 1, no battery term) to ~1.5-1.6 (stages 2/3), with avg_batt/battery_w (up to
# 100/5=20) making up 90%+ of the reported "efficiency" -- the identical failure mode the
# LJ model's own EFF_DISTANCE_WEIGHT was introduced to fix. Started at the LJ model's own
# calibrated value (8.0) as a reasoned starting point, not independently re-derived; this
# is a deliberate, disclosed deviation from Table 2's literal formula, not a literal
# replication -- document it in the paper the same way the LJ model's weighting choice
# is documented in initial_implementation/experiment/config.py.
HEBBIAN_EFF_DISTANCE_WEIGHT = 8.0

# Replaces BOTH the wall-collision fitness penalty (original 3-stage curriculum) AND the
# wall-proximity sensor input (wall_sensor_variant/) with a single, much simpler idea:
# directly reward staying on a straight line toward the goal, so wall-approach never
# becomes an issue in the first place, instead of trying to learn to sense or react to
# the boundary at all. Real data motivated this: wall_sensor_variant's added sensory/
# parameter complexity (880->1040 params) came with a measurable COHESION regression
# (mean pairwise inter-agent distance growing 2-3x over an episode, vs. shrinking over
# the original 3-stage genome) -- speculatively because cohesion was only ever an
# indirect, emergent side effect of battery-saving (drafting), and CMA-ES traded it off
# against the newly-easy-to-satisfy direct wall-avoidance signal.
#
# y_drift is the WORST agent's |y_final - y_initial|, not a mean -- a first attempt
# using the mean failed: CMA-ES satisfied "low average drift" by keeping some agents
# near the spawn line while letting others still drift straight into a wall (real data:
# 3/7 agents hard-pinned at the exact wall-clamp value, 2 others drifted the opposite
# way, yet mean |y_drift| looked merely "moderate" -- the swarm had literally split into
# subgroups, which is also what was tanking cohesion). Penalizing the worst agent means
# a single wall-hugger tanks the whole candidate's fitness, not just nudges an average.
#
# Started at the same value as HEBBIAN_EFF_DISTANCE_WEIGHT as a reasoned default
# (comparable scale to the distance reward it's meant to balance against, not
# independently tuned) -- validate before trusting, like every other weight here.
HEBBIAN_STRAIGHTNESS_WEIGHT = 8.0

# =====================================================================================
# --- LJ Table-3 baseline (lj_baseline.py) -- used only by analyze_hebbian_results.py's
# Fig. 5a "cluster-4" comparison point. A trimmed, standalone copy of the original LJ
# model's control law + fitness; see initial_implementation/experiment/
# simulation_free_global_mod_2_LJ.py for the full version (video rendering, PyBullet
# backend, CMA-ES training -- none of which this replication needs).
# =====================================================================================
MAX_BATTERY = 150.0          # LJ model's own battery scale (NOT HEBBIAN_MAX_BATTERY's 0-100
MIN_BATTERY = 150.0          # scale) -- the baseline literally IS the LJ model, run as-is.

DEFAULT_RULES = {
    "r0": 0.70, "epsilon": 0.5, "k_align": 0.0, "k_goal": 3.0, "K1": 0.05, "K2": 0.5, "U": 0.005,
}
# Paper's Table 3 "standard collective motion baseline" -- the cluster-4 comparison point
# in Fig. 5a. NOT the same as DEFAULT_RULES above: that dict is simulation_free_global_
# mod_2_LJ.m's own hardcoded example starting point (epsilon=0.5, U=0.005), which differs
# from the paper's literal baseline (epsilon=1, no U term).
PAPER_BASELINE_RULES = {
    "r0": 0.70, "epsilon": 1.0, "k_align": 0.0, "k_goal": 3.0, "K1": 0.05, "K2": 0.5, "U": 0.0,
}
R_CUT = 3.0        # LJ interaction cutoff radius [m]
R_MIN = 0.0         # LJ singularity guard radius [m]
R_ALIGN = 1.5       # neighbor radius used for heading alignment [m]
LINEAR_VEL_MAX = 0.20            # robot's max linear speed [m/s]
ANGULAR_VEL_MAX = math.pi / 5    # robot's max angular speed [rad/s]

# eff = EFF_DISTANCE_WEIGHT*dist_travelled + avg_batt/EFF_BATTERY_WEIGHT - collision_time/EFF_COLLISION_WEIGHT
# (unused by analyze_hebbian_results.py's baseline comparison, which only reads dist/batt,
# but kept for parity with the original _fitness() this was copied from.)
EFF_DISTANCE_WEIGHT = 8.0
EFF_BATTERY_WEIGHT = 10.0
EFF_COLLISION_WEIGHT = 250.0
