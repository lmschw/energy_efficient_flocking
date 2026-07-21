"""Minimal, dependency-light config for running a trained Hebbian ABCD controller on
real Thymio+Raspberry Pi hardware via thymio_swarm_platform / thymio_raspberry_swarm_control.

Deliberately NOT importing energy_efficient_flocking/experiment/config.py: that module
pulls in constants for the LJ model, PyBullet, etc. that have no business being deployed
to a Raspberry Pi. Keep this file and its siblings (sensor_model.py, hebbian_controller.py)
pure-numpy and self-contained.

Architecture constants below MUST match whatever genome you actually trained with
optimize_hebbian.py -- they are not independently tunable at deployment time.
"""
import math

# --- Neural controller architecture (must match training) ---
N_INPUTS = 10
N_HIDDEN = 10
N_OUTPUTS = 2
N_ABCD = 4 * (N_INPUTS * N_HIDDEN + N_HIDDEN * N_HIDDEN + N_HIDDEN * N_OUTPUTS)  # 880
LEARNING_RATE = 0.1
WEIGHT_INIT_RANGE = 1.0

# --- Sensing (must match training) ---
SENSING_RADIUS = 2.01          # meters
LINEAR_VEL_MAX = 0.2           # m/s
ANGULAR_VEL_MAX = math.pi / 5  # rad/s

# --- Battery ---
# Neither the Thymio nor the Pi expose a battery/power reading anywhere in
# thymio_swarm_platform (checked robot.py, state.py, system_sounds.py -- nothing), and
# there's no wind tunnel available to reproduce the simulation's headwind on real
# hardware either. Two modes control what's fed into the NN's battery input (x_in[8]):
#
# "none" -- always BATTERY_SENSOR_PLACEHOLDER, exactly matching how a genome trained with
#   `optimize_hebbian.py --no-battery-sensor` was trained. Deploy a `_nosensor` genome
#   with this mode.
#
# "simulated" -- wind_battery_model.py computes a virtual battery level in software each
#   tick, using the *exact same* wind-wake + drag-force + drainage equations the genome
#   was evolved against (ported verbatim from experiment/simulation_free_global_mod_2_LJ.py),
#   driven by real OptiTrack positions instead of simulated ones. This exists specifically
#   because you can't generate a real, uniform headwind without a wind tunnel -- the wind
#   exposure each robot experiences is instead *modeled*, from the swarm's real relative
#   positions, exactly as during training. Deploy a battery-AWARE (non-`_nosensor`) genome
#   with this mode. Requires scipy (see wind_battery_model.py) -- lazily imported only
#   when this mode is selected, so it's not a dependency otherwise.
#
# Whichever mode you use, if it's "simulated": disclose this in any writeup. The reported
# battery is a physically-modeled software quantity computed from real robot positions,
# not a measurement of real power draw.
BATTERY_MODE = "none"   # "none" or "simulated"
BATTERY_SENSOR_PLACEHOLDER = 0.0   # used only when BATTERY_MODE == "none"

# --- Control tick rate ---
# MUST match the simulation's dt (experiment/config.py's DT = 0.5s), NOT a faster
# "smooth robotics" tick rate like the platform's other example experiments use (they
# poll at 0.05s/20Hz). The Hebbian update (eta=0.1) is applied once per tick during
# training; running ticks faster in deployment means far more weight updates per
# second of real time than the genome was ever evolved under, changing its effective
# learning dynamics. This also happens to match OptiTrack's ~2Hz (0.5s) pose-push rate
# (see README.md), so it avoids wasting ticks re-reading a stale, unchanged pose.
CONTROL_TICK_SECONDS = 0.5

# =====================================================================================
# --- Hardware calibration -- UNVERIFIED PLACEHOLDERS. Do not trust these numbers until
# you have measured them on your actual robots/Motive setup. See README.md's
# "Calibration" section and diagnostics/print_poses.py.
# =====================================================================================

WHEEL_RADIUS_M = 0.021       # thymio_swarm_platform RobotConfig.wheel_radius
WHEEL_DISTANCE_M = 0.085     # thymio_swarm_platform RobotConfig.wheel_distance
MAX_MOTOR_TARGET = 500       # thymio_swarm_platform RobotConfig.max_motor (raw units;
                             # NOT enforced by the platform's Robot.drive() itself)

MOTOR_UNITS_PER_MPS = 500.0 / 0.2
# UNVERIFIED: assumes raw motor target 500 (the commonly-cited firmware ceiling) maps to
# roughly this controller's own LINEAR_VEL_MAX = 0.2 m/s top speed. Measure your actual
# Thymio's real-world speed at a given motor.target value (drive at a fixed target for a
# timed run over a measured distance) and recompute this before trusting any distance-
# or speed-based comparison against the simulation's numbers.

HEADING_OFFSET_RAD = 0.0
# UNVERIFIED: the yaw angle (after quaternion_to_yaw(), see pose_utils.py) OptiTrack
# reports when a robot is physically oriented at this codebase's heading=0 (facing "+y"
# in the simulation's convention -- see wrap_to_pi()/move() in
# experiment/simulation_free_global_mod_2_LJ.py). Depends on your Motive ground-plane
# calibration and how each rigid body's "front" was defined when you created it. Use
# diagnostics/print_poses.py: point a robot at the sim's heading=0 direction, read its
# raw yaw, and set this to minus that value.

POSITION_AXES = (0, 1)
# UNVERIFIED: which two of OptiTrack's (x, y, z) position components map to this
# codebase's 2D ground-plane (x, y). Motive is commonly Y-up by default (ground plane =
# X/Z, i.e. axes (0, 2)), but this is fully dependent on your specific Motive
# calibration -- verify with diagnostics/print_poses.py, don't assume.

ROTATION_SIGN = 1.0
# UNVERIFIED: +1.0 or -1.0. If the robot turns the wrong way in practice (spins away
# from where it should be heading), flip this -- it multiplies the angular-rate output
# before conversion to left/right wheel targets in motor_utils.py.

# =====================================================================================
# --- Simulated battery drainage (BATTERY_MODE == "simulated") ------------------------
# Every constant below is copied verbatim from experiment/config.py's HEBBIAN_*/WAKE_*/
# DRAG_*/BATTERY_* sections -- "the same battery drainage as the simulation" means using
# the identical formulas AND the identical constants, not hardware-recalibrated ones.
# Change these only if you deliberately want to deviate from what the deployed genome was
# actually evolved against.
# =====================================================================================

INITIAL_BATTERY = 100.0
# experiment/config.py's HEBBIAN_MAX_BATTERY/HEBBIAN_MIN_BATTERY -- matches the [0, 100]
# range sensor_model.py's battery normalization (agents[:, 3] / 50.0 - 1.0) assumes.

ROBOT_RAD = 0.055
# experiment/config.py's ROBOT_RAD. Used by batterydrainage()'s wheel-speed-differential
# term -- NOT the same thing as WHEEL_DISTANCE_M above (that's for motor_utils.py's
# actual differential-drive kinematics); both are needed, kept separate on purpose.

WIND_RAD = 0.15              # a robot's own wind-occlusion radius [m]
WIND_Y_RANGE = (-5.0, 5.0)   # experiment/config.py's Y_RANGE. The wake field is computed
                              # over this fixed y-span regardless of the swarm's actual y
                              # position; tune to your real tracked volume's y-extent if
                              # it differs meaningfully from the simulation's 10m arena.
WIND_TRACKING_WINDOW_WIDTH = 10.0
WIND_TRACKING_MAX_SPAN = 9.8
UNTRACKED_XY_THRESHOLD = 1e3
# pose_utils.py places untracked robots at (1e4, 1e4). Any agent beyond this threshold is
# excluded from the wind-field x-window computation (it would otherwise blow up the
# min/max) rather than treated as a real occluder -- it's already effectively excluded
# from the wake field itself, since 1e4 is far outside any realistic grid.

UINF = 100.0
NX = 200
NY = 200
# Wind grid resolution. The O(Nx) wake-marching loop plus two 2D convolutions run once
# per control tick when this mode is on. UNMEASURED on real Pi hardware -- a Pi is much
# slower than a dev laptop, and CONTROL_TICK_SECONDS = 0.5s is a hard real-time budget
# this computation must fit inside. Profile this on your actual Pi before trusting the
# default; drop to e.g. 50 (matches experiment/optimize_hebbian.py's --wind-grid 50) if a
# tick can't keep up.
KAPPA = 20.0
V_WIND = 10.0

WAKE_RECOVERY_RATE = 1.0
WAKE_PERCENT_DROP = 0.25
WAKE_MAX_WALL_SPAN = 0.7
WAKE_MIN_POWER_X = 30.0
WAKE_MIN_POWER_Y = 10.0
WAKE_ALPHA = 0.5
WAKE_BETA = 0.5
WAKE_X_SMOOTHING_1 = 100
WAKE_Y_SMOOTHING_1 = 50
WAKE_X_SMOOTHING_2 = 50
WAKE_Y_SMOOTHING_2 = 50
WAKE_THR_OK_DELTA = 1.0

DRAG_UPSTREAM_LOOKAHEAD_FACTOR = 1.1
DRAG_AIR_DENSITY = 1.225
DRAG_COEFFICIENT_AREA = 0.0045

BATTERY_WHEEL_POWER_DIVISOR = 4.0
BATTERY_MIN_DRAIN = 0.10
BATTERY_DRAIN_SCALE = 2.0
