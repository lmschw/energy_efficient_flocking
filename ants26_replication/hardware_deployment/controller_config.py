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
#   with this mode. No extra dependencies beyond numpy (already required regardless of
#   BATTERY_MODE) -- wind_battery_model.py's smoothing step used to need scipy, replaced
#   with an exact pure-numpy equivalent (the kernel is separable; see that file).
#
# Whichever mode you use, if it's "simulated": disclose this in any writeup. The reported
# battery is a physically-modeled software quantity computed from real robot positions,
# not a measurement of real power draw.
BATTERY_MODE = "simulated"   # "none" or "simulated"
# Set to "simulated" for this deployment: the chosen genome (hebbian_results_v2/
# hebbian_save_battery_avoid_all_best.npy, the paper-default n_agents=20 run through all
# 3 curriculum stages) was trained WITH the battery sensor (battery_sensor=True in its
# history JSON) -- there is no "_nosensor" variant of it. "none" mode would silently feed
# it a constant placeholder it was never trained to expect.
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

MOTOR_UNITS_PER_MPS = 3609.86
# STILL STALE / DO NOT TRUST -- same flawed calibration run as HEADING_OFFSET_RAD above:
# travel_speed is measured as hypot() over the (wrong) selected ground-plane axes, so
# this number is corrupted too, not just the heading/axes values. Re-run calibration with
# known_up_axis=1 set (see HEADING_OFFSET_RAD comment above) and replace this before
# trusting any distance- or speed-based comparison against the simulation's numbers.

HEADING_OFFSET_RAD_DEFAULT = -2.8531
# UNVERIFIED: the yaw angle (after quaternion_to_yaw(), see pose_utils.py) OptiTrack
# reports when a robot is physically oriented at this codebase's heading=0 (facing "+y"
# in the simulation's convention -- see wrap_to_pi()/move() in
# experiment/simulation_free_global_mod_2_LJ.py). Depends on your Motive ground-plane
# calibration and how each rigid body's "front" was defined when you created it.

HEADING_OFFSET_RAD = {
    "thymio-17": +0.0797,
    "thymio-20": +2.9517,
    "thymio-18": -3.0151,
}
# PER-ROBOT, not one shared constant -- pose_utils.poses_to_agents() looks up each pose's
# own hostname here, falling back to HEADING_OFFSET_RAD_DEFAULT (with a one-time warning,
# not a crash) for any hostname not listed -- e.g. a new robot added to the fleet before
# it's been individually calibrated.
#
# STILL STALE / DO NOT TRUST -- these numbers are from a calibration run where up-axis
# detection was WRONG for 2 of 3 robots (thymio-17 and thymio-20 both picked axis 2 as
# "up" instead of the actual up axis, 1; only thymio-18 picked axis 0, also wrong -- see
# calibration.txt). Since HEADING_OFFSET_RAD is derived from the travel bearing measured
# in the (wrong) selected ground-plane axes, these 3 values are all corrupted, not just
# POSITION_AXES below. calibrate_position_heading_experiment.py now supports a
# known_up_axis override (see its docstring) specifically to stop this from recurring --
# the controller-side launcher (examples/hebbian_position_heading_calibration.py in the
# thymio_swarm_platform repo) now sets known_up_axis=1 (this rig's confirmed Y-up axis).
# Re-run calibration with that fix in place and replace these three values before
# trusting them.

POSITION_AXES = (0,2)
# This is a deterministic consequence of the rig being confirmed Y-up (axis 1 is up),
# not something that needs re-measuring: excluding axis 1 always leaves (0, 2), Motive's
# usual X/Z ground plane. Fixed directly rather than left at the previous calibration
# run's (wrong, disagreeing-between-robots) (0, 1) value -- see calibration.txt and the
# HEADING_OFFSET_RAD comment above for why that run's up-axis detection was unreliable.
# Shared across all robots (a genuine platform-wide constant, unlike HEADING_OFFSET_RAD),
# so it does NOT need to be a per-robot dict.

ROTATION_SIGN = 1.0
# UNVERIFIED: +1.0 or -1.0. If the robot turns the wrong way in practice (spins away
# from where it should be heading), flip this -- it multiplies the angular-rate output
# before conversion to left/right wheel targets in motor_utils.py.

# =====================================================================================
# --- Corridor wall safety (deployment-only -- NOT a trained genome behavior) ---------
# =====================================================================================
# sensor_model.get_sensor_data()'s 10 inputs are 4 neighbor quadrants (dist+bearing),
# own battery, own heading -- there is NO wall-distance/absolute-position input anywhere
# in the architecture. Whatever wall-avoidance the paper-default genome learned in
# save_battery_avoid_wall/save_battery_avoid_all came from reward shaping alone during
# training, against a fixed 20-agent, Y_RANGE=[-5,5] simulated arena -- an emergent
# artifact of correlations specific to that arena/neighbor density, not a real sense of
# "wall nearby". Confirmed on real hardware: with only 3 robots in a much smaller room,
# those correlations don't hold, and the genome genuinely does not react to real walls
# at all -- it drove straight at the corridor edge without slowing.
#
# Fixing this properly means retraining with an actual wall-sensor input (see
# ants26_replication/wall_sensor_variant/ -- a separate, still-experimental
# architecture, not used here). Rather than block this deployment on that, this governor
# is a deployment-side safety layer bolted on AFTER the genome's own v, w computation
# (see hebbian_swarm_experiment.py._tick()): it scales the commanded forward speed v
# down toward 0 as the robot's real, tracked sim-frame y (agents[:, 1], i.e. whichever
# raw axis POSITION_AXES's second entry selects) approaches CORRIDOR_Y_MIN/MAX, and
# leaves the genome's own turning decision (w) completely untouched -- deliberately the
# simplest option (slow down only, not also steer back toward center), chosen over a
# more assertive heading override because it changes the genome's own behavior less and
# is easier to reason about as a pure safety cap. It is NOT direction-aware: v gets
# scaled down near a wall even if the robot happens to already be heading away from it --
# simpler and strictly safer than trying to also read intent from heading, at the cost
# of some unnecessary slowdown in that case.
CORRIDOR_Y_MIN = None   # meters, sim-frame y (agents[:, 1]) -- measure by deploying
CORRIDOR_Y_MAX = None   # diagnostics/print_poses_experiment.py (with hostnames/
                         # self_hostname set) and walking a robot to each wall; it prints
                         # a running min/max of y for exactly this purpose. The governor
                         # is disabled (v passes through unmodified) while either is
                         # None -- set BOTH before relying on it to prevent wall strikes.
CORRIDOR_SLOWDOWN_MARGIN_M = 0.5
# v scales linearly from 1.0 (at this distance or farther from either wall) to 0.0 (at
# the wall) over this margin. Tune to your corridor's real width and the robot's real
# top speed -- too small a margin gives the robot little time to actually slow down
# before reaching the wall; too large eats into usable corridor width unnecessarily.

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
