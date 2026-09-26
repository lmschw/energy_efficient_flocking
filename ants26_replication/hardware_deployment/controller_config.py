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
# Set to "simulated" for this deployment: the chosen genome (plain_seed123_clamped_best.npy,
# see "Current trial config" below) was trained WITH the battery sensor
# (use_battery_sensor=True) -- there is no "_nosensor" variant of it. "none" mode would
# silently feed it a constant placeholder it was never trained to expect.
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

MOTOR_UNITS_PER_MPS = 3553.09
#MOTOR_UNITS_PER_MPS = 3609.86
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

#if ROTATION_SIGN = 1.0:
HEADING_OFFSET_RAD = {
    "thymio-08": -1.3214,
    "thymio-12": -1.4675,
    "thymio-15": -0.0046,
    "thymio-17": +0.2781,
    "thymio-09": -1.3747,
    "thymio-25": -2.6076,
    "thymio-11": -1.2086,
    "thymio-04": -1.1698,
    "thymio-01": -0.4044,
    "thymio-07": -0.4476,
    "thymio-19": -0.2165,
    "thymio-18": -0.3709,
    "thymio-14": -1.1748,
}

# if ROTATION_SIGN = -1.0:
# HEADING_OFFSET_RAD = {
#     "thymio-08": -2.3280,
#     "thymio-12": -1.7524,
#     "thymio-15": +0.0893,
#     "thymio-17": -0.4416,
#     "thymio-09": -1.8180,
#     "thymio-25": -0.8526,
#     "thymio-11": -1.8365,
#     "thymio-04": +0.0737,
#     "thymio-01": -1.7569,
#     "thymio-07": -2.5926,
#     "thymio-19": +0.0595,
#     "thymio-18": +0.1304,
#     "thymio-14": -2.1559,
# }


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

POSITION_AXES = [0, 2]
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

UP_AXIS_OUTLIER_THRESHOLD_M = 0.8
# REPLACED (2026-09-23) an earlier hardcoded ABSOLUTE plausible-height band (e.g.
# -0.5 to 0.5m) after it broke on the very next real session: that band was calibrated
# from one session's example numbers (good robots reading ~0.143/-0.341) and trusted as
# if it were universal, despite this very file's own prior comment admitting the
# coordinate origin/ground-plane calibration isn't guaranteed to match between sessions.
# Confirmed the hard way: a later real trial had EVERY robot's genuinely correct up-axis
# reading (around -0.57 to -1.03) fall outside that old band, so poses_to_agents()
# silently treated every robot as permanently self-blind for the entire run --
# reproducing the exact "no neighbors detected, spin in place, no forward progress"
# symptom this check exists to prevent. Worse, that session's legitimate ~-1.0m reading
# overlaps almost exactly with a DIFFERENT session's confirmed-bad ghost-object reading
# -- an absolute band genuinely cannot tell them apart across sessions.
#
# Now RELATIVE instead: pose_utils._find_up_axis_outliers() compares each tracked
# robot's up-axis reading against the MEDIAN of the other robots tracked THIS SAME TICK,
# flagging (and rejecting, same as "no pose at all") one that deviates by more than this
# threshold -- no a-priori knowledge of "what near-zero means this session" required.
# Value chosen from the two confirmed real cases so far: legitimate cross-robot spread
# within a single session has stayed under ~0.5m (both the 0.143/-0.341 pair and the
# -0.57/-1.03 pair from two different sessions); genuine bad tracks have been either a
# stray object ~1.8-2.0m off real robots' shared height, or a persistently mistracked
# robot ~1.2m+ off its peers. 0.8m sits comfortably between those two regimes given the
# data seen so far -- but this is still a heuristic from a small number of observed
# cases, not a physically derived constant; retune if either regime turns out to
# overlap it in practice. Needs >=2 currently-tracked robots to have a peer to compare
# against -- with 0 or 1 tracked this tick, no outliers can be detected (nothing to
# compare against), so this is defense-in-depth on top of, not a replacement for,
# actually locating and removing/covering any stray reflective object you can find.

STALE_POSE_TICK_THRESHOLD = 3
# ADDED (2026-09-23): the up-axis fix above and the corridor fix before it were both real
# improvements (real trial data showed genuine changing x/y positions for much longer than
# before), but a *different* failure mode showed up in the very next real trial
# (aggregated_1905.csv): thymio-11's x, y, raw_x, raw_y, raw_z AND its four sensed
# quadrant distances were bit-for-bit IDENTICAL across 11 straight ticks (~5.5s at
# CONTROL_TICK_SECONDS=0.5) before the operator noticed it had stopped moving and pushed
# it back into view to reacquire tracking. Real OptiTrack marker noise essentially never
# reproduces the exact same float across many consecutive polls, so this wasn't "robot
# genuinely decided v=0" -- it was the tracking feed itself silently freezing (the robot
# dropping out of NatNet's frame stream without ever being reported as untracked; nothing
# in optitrack_client.py/session daemon has a staleness/timestamp check -- see that TODO)
# while the daemon kept re-serving its last known pose as if it were live. Since v is
# fed FROM the (frozen) sensed neighbor/self positions, the Hebbian weights converge on
# that constant input and v settles near 0 -- i.e. this reproduces the exact same
# "spin in place, no forward progress" symptom as the other two bugs, for a third,
# unrelated reason.
#
# pose_utils._find_stale_poses() treats a robot's raw position repeating bit-for-bit for
# this many CONSECUTIVE ticks (including the current one) as a frozen/stale feed rather
# than a genuinely still robot, and rejects it the same way as "no pose at all" (same
# sentinel fallback as the up-axis check). 3 tolerates a single coincidental exact repeat
# (possible if a robot is truly stationary and OptiTrack's solver happens to reproduce
# the same float twice) while still catching a genuine freeze within ~1 second of it
# starting, well before the 11-tick freeze actually observed. This is a heuristic, not a
# real timestamp check -- a proper fix would thread NatNet's own per-frame `timing`
# argument (currently received and discarded in optitrack_client.py's _callback) through
# the daemon relay so staleness could be measured directly; not done here since that
# spans both repos and this local check already catches the failure mode observed.

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

CORRIDOR_Y_MIN = -1.804  # meters, sim-frame y (agents[:, 1]) -- measure with
CORRIDOR_Y_MAX = 1.572   # hebbian_pose_calibration.py <hostname> (point-and-sample at
                         # each wall) and set BOTH before relying on this to prevent wall
                         # strikes. RESET TO None (2026-09-23): the previous 0.327/1.229
                         # values were stale from an earlier calibration session and no
                         # longer bracket the room's real y-range at all -- confirmed via
                         # real trial data showing every robot's actual sim_y around
                         # -1.2, nowhere near that window, so the corridor governor was
                         # silently zeroing every robot's forward speed unconditionally,
                         # everywhere in the room, regardless of tracking quality or
                         # neighbor visibility -- this was the actual cause of robots
                         # only ever spinning in place, not a tracking-quality issue.
                         # The governor is disabled (v passes through unmodified) while
                         # either is None, which is deliberate here: do NOT re-enable
                         # with guessed numbers -- recalibrate for real against the
                         # CURRENT session's actual corridor before setting these again,
                         # since walking into a real wall is what this governor exists to
                         # prevent.
CORRIDOR_SLOWDOWN_MARGIN_M = 0.5
# v scales linearly from 1.0 (at this distance or farther from either wall) to 0.0 (at
# the wall) over this margin. Tune to your corridor's real width and the robot's real
# top speed -- too small a margin gives the robot little time to actually slow down
# before reaching the wall; too large eats into usable corridor width unnecessarily.

# =====================================================================================
# --- Inter-agent safety clamp (deployment-ENFORCED, not just trained-in) -------------
# =====================================================================================
# CRITICAL: plain_seed123_clamped.npy was retrained WITH a hard forward-speed clamp active
# during evolution (see ants26_replication/experiment/simulation_hebbian.py's
# _apply_safety_clamp() and the training script
# ants26_replication/upwind_safety_variant/run_2stage_upwind_clamped.py) -- but that clamp is
# a SIMULATION-TIME mechanism, not something baked into the genome's weights. Training under
# it only shapes what the network LEARNED to do; it does not make the network intrinsically
# collision-avoidant on its own. If this clamp is not ALSO enforced here, at deployment time,
# deploying "the clamped genome" provides ZERO actual collision-safety benefit over the
# unclamped one -- confirmed necessary, not optional (this was nearly missed: nothing in this
# package enforced it before 2026-09-16, only the corridor/wall case below existed). Applied
# in hebbian_swarm_experiment.py._tick() the same way as the corridor governor: scales v only
# (never w), using the TRUE (uninflated) ROBOT_RAD the training run used but -- see the
# 2026-09-23 note below -- DELIBERATELY LOOSENED thresholds rather than the exact training
# values, a real off-distribution tradeoff made after a genuine deadlock, not an oversight.
AGENT_SAFETY_CLAMP_OUTER_GAP = 0.12   # gap [m] at which braking begins (full speed above this)
AGENT_SAFETY_CLAMP_INNER_GAP = -0.03  # gap [m] at which forward speed reaches zero (contact)
# LOOSENED (2026-09-23) from the original 0.30/0.05 -- which matched experiment/config.py's
# HEBBIAN_SAFETY_CLAMP_OUTER_GAP/INNER_GAP exactly -- after real trial data showed thymio-09
# and thymio-11 fully deadlocked: nearest_gap computed as -0.043m (center distance 0.067m,
# 2*ROBOT_RAD=0.11m), clipped to a scale of exactly 0.0, forcing v to zero for BOTH robots
# with no way to recover since the clamp suppresses v regardless of sign -- they couldn't
# even back away from each other, only rotate (w is never clamped). This may well be the
# same underlying mechanism behind the "spins but never translates" symptom seen throughout
# this investigation, not just this one trial.
#
# A measured gap of -0.043m is itself informative: two solid ~11cm-diameter Thymios cannot
# physically overlap by 4.3cm, so that reading has to include several cm of real OptiTrack
# tracking noise/marker-offset, not genuine contact. The original 0.30/0.05 thresholds were
# also just wider than the formation spacing actually used in testing (0.30 surface gap is
# 0.41m center-to-center, well above the ~20-50cm this project has been placing/observing
# robots at), so braking was likely at least partially active for most of every trial run.
#
# IMPORTANT CAVEAT (explicit user decision to accept this tradeoff, 2026-09-23): unlike
# CORRIDOR_SLOWDOWN_MARGIN_M, this constant IS coupled to what plain_seed123_clamped's
# weights were actually shaped by during training (see
# ants26_replication/upwind_safety_variant/run_2stage_upwind_clamped.py's use of these same
# values) -- loosening it does not just reduce collision margin, it also deploys the genome
# under braking dynamics measurably different from what it was evolved against. This was a
# deliberate, explicit tradeoff (mobility over exactly-matching the trained distribution),
# not a bug fix. If collisions/deadlocks recur, prefer addressing WHY robots converge this
# close in the first place (placement, or an emergent behavior worth its own investigation)
# over loosening further -- INNER_GAP is already negative, i.e. already tolerating apparent
# overlap, not just reduced margin above contact.

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
NX = 50
NY = 50
# Wind grid resolution -- MUST match whatever the deployed genome was actually trained at
# (see plain_seed123_clamped's training command, --wind-grid 50, same as every other genome
# in this session's investigation) or the wake field the battery model sees at deployment
# time is quantitatively different from what the genome was evolved against. Previously set
# to 200 with no explanation -- almost certainly a stale, never-corrected default rather than
# a deliberate choice (this exact mismatch was independently documented and fixed for the
# simulation-side analysis in hardware_transfer_test/leadership_metrics.py's docstring: "both
# genomes' sibling _history.json record they were trained at wind_grid_nx/ny=50, not
# config.py's current default of 200"). The O(Nx) wake-marching loop plus two 2D convolutions
# run once per control tick when this mode is on; at NX=NY=50 this is comfortably fast
# (config.py's KAPPA/NX values matched to the training run's own config), but still
# UNMEASURED on real Pi hardware -- profile before trusting.
KAPPA = 10.0
# MUST match the deployed genome's training-time value -- was 20.0 with no explanation
# (every genome trained in ants26_replication/experiment/ and .../upwind_safety_variant/
# uses config.py's own default of 10.0; nothing in this investigation ever trained under
# KAPPA=20 for a genome that ended up deployed). Fixed alongside the NX/NY correction above
# when plain_seed123_clamped became the deployed genome (2026-09-16) -- if you swap genomes
# again, re-check both constants against that genome's own training config.
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
