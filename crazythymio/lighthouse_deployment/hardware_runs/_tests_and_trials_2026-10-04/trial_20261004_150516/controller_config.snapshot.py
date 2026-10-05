"""Config for the LIGHTHOUSE deployment (Thymio + Pi + Crazyflie board, CrazyThymio stack).

This is a standalone copy, independent of ants26_replication/hardware_deployment/
controller_config.py (the OptiTrack deployment): nothing here is imported by, or shared
with, that folder. Changing a constant here does not affect OptiTrack runs and vice versa.

Architecture/sensing/battery/safety constants are copied verbatim from the OptiTrack config
(they must match the genome's training, not the tracking system); comments inside those
copied blocks still mention OptiTrack where they record that deployment's history. The
Lighthouse-specific hardware section (frames, heading offsets, corridor, motor calibration)
is new and UNCALIBRATED -- see README.md.
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
MAX_PLAUSIBLE_SPEED_MPS = 2.0 * LINEAR_VEL_MAX
# Tracked position deltas faster than this can't have been driven -- they're the robot being
# picked up and moved by hand (or a tracking jump). hebbian_swarm_experiment.py skips the
# simulated battery drain for such a tick instead of charging it as high-speed driving,
# which previously depleted the virtual battery and permanently stopped the robot.
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
# --- Hardware calibration (Lighthouse) -- UNVERIFIED, measure on your robots ----------
# =====================================================================================

WHEEL_RADIUS_M = 0.021
WHEEL_DISTANCE_M = 0.085
MAX_MOTOR_TARGET = 500       # raw Thymio motor units; clamped in motor_utils.py

MOTOR_UNITS_PER_MPS = 2949.0
# MEASURED 2026-10-04 with `run_hebbian.py --calibrate-heading` on all 7 robots (150 motor units for
# 4 s -> 0.202..0.214 m): 2798..2976, median 2949 (robot 5 slowest). Replaced the 3553.09 carried over
# from the OptiTrack deployment. Raw results: lighthouse_config/heading_calibration.csv. Note
# 0.2 m/s needs 590 units but MAX_MOTOR_TARGET caps at 500, i.e. top speed ~0.17 m/s.

LIGHTHOUSE_HEADING_OFFSET_RAD_DEFAULT = 0.0
LIGHTHOUSE_HEADING_OFFSET_RAD = {
    # MEASURED 2026-10-04 (lighthouse_config/heading_calibration.csv). robot-2's Crazyflie was first
    # mounted ~180 deg rotated (offset 3.0554); re-mounted and re-calibrated the same day.
    "robot-1": -0.0138,
    "robot-2": -0.0922,
    "robot-3": 0.0431,
    "robot-4": -0.0482,
    "robot-5": -0.0418,
    "robot-6": -0.0570,
    "robot-7": 0.0431,
}
# Rotation [rad] of the Crazyflie board's +x axis relative to the Thymio's front, added to the
# heading in pose_utils.poses_to_agents(). 0.0 assumes the board's x axis points along the
# Thymio's front. Per robot: measure with `run_hebbian.py --calibrate-heading`.

ROTATION_SIGN = 1.0
# +1.0: Lighthouse yaw (CCW from +x) increases in the same sense as the simulation's heading.
# Flip only if robots demonstrably turn the wrong way.

POSE_TIMEOUT_S = 1.0         # own pose older than this counts as untracked (lighthouse_robot.py)

CORRIDOR_Y_MIN = None
CORRIDOR_Y_MAX = None
CORRIDOR_X_MIN = None
CORRIDOR_X_MAX = None
# Same governor for x (added 2026-10-04: the LJ baseline pulls the swarm toward -x at ~0.15 m/s and the arena has
# no physical walls). Disabled while either is None; run_hebbian.py sets them from --corridor-x.
CORRIDOR_SLOWDOWN_MARGIN_M = 0.5
# Wall slow-down governor in the SHIFTED frame (arena center = 0,0): v scales linearly from 1
# (>= margin from both walls) to 0 at the wall. Disabled while either is None; run_hebbian.py
# sets them from --corridor-y. The genome has no wall sense of its own, so without this robots
# drive into walls.

SAFETY_LAYERS_ENABLED = False
# MASTER SWITCH for the deployment-side safety layers in the experiment classes: the agent-safety speed clamp
# (AGENT_SAFETY_CLAMP_*), the x/y wall governors (CORRIDOR_*), and the slow crawl when the own pose is lost
# (UNTRACKED_SAFE_V_CAP). OFF since 2026-10-04 on request: robots run the raw controller output, nothing else. The
# obstacle-backoff and IR-backoff reflexes have their own switches (both off). Turn back on with --safety on
# (run_hebbian.py) or SAFETY=1 (tools/run_swarm.sh / experiment_schedule.sh). With it OFF the robots can leave the
# arena (LJ pulls toward -x at ~0.15 m/s) and collide -- supervise, keep tools/stop_all.sh ready.

# =====================================================================================
# --- Inter-agent safety clamp (deployment-ENFORCED, same as the OptiTrack deployment) -
# =====================================================================================
AGENT_SAFETY_CLAMP_OUTER_GAP = 0.30   # gap [m] at which braking begins (full speed above this)
AGENT_SAFETY_CLAMP_INNER_GAP = 0.05   # gap [m] at which forward speed reaches zero
# Training values of plain_seed123_clamped (see the OptiTrack config for the history of
# loosening/restoring them); the clamp is a simulation-time mechanism that must ALSO be
# enforced at deployment, or the "clamped" genome gives no collision safety.

# =====================================================================================
# --- Deployment-only obstacle-backoff reflex (NOT a trained genome behavior) ---------
# =====================================================================================
# Direct answer to the AGENT_SAFETY_CLAMP_INNER_GAP history above: that clamp only ever
# scales v toward 0 as agents converge -- it has no way to make a deadlocked pair back
# away from each other (the 2026-09-23 thymio-09/thymio-11 incident above). Tight
# clustering is also a tracking problem on its own: OptiTrack rigid bodies solve worse
# (or drop out entirely -- see pose_utils.py's stale-freeze/up-axis-outlier detectors,
# both added after real dropout incidents) when robots' marker constellations are close
# together, independent of any actual collision risk. This reflex reduces how long
# robots spend clustered at all, rather than only reacting once they're already stuck:
# if something has been sensed dead ahead for OBSTACLE_TRIGGER_TICKS straight, back
# straight up for OBSTACLE_BACKOFF_TICKS; symmetrically, ease straight forward if
# something is persistently dead behind. Gated on PERSISTENCE, not a single noisy tick,
# same reasoning as STALE_POSE_TICK_THRESHOLD. Heading (w) is left completely untouched
# -- same simplest-option tradeoff as the corridor governor: this only ever overrides v
# while active, and composes with AGENT_SAFETY_CLAMP/corridor scaling applied afterward
# in hebbian_swarm_experiment.py (a [0,1] scale can weaken the backoff but never flips
# its sign, so a robot backing away never gets turned back around by those clamps).
OBSTACLE_BACKOFF_ENABLED = False
# DISABLED 2026-10-04 (Lighthouse deployment only). In the first 7-robot LJ baseline run (r0 = 0.5 m) the swarm sat inside this
# reflex's trigger zone (neighbour within OBSTACLE_TRIGGER_DIST ~0.5 m) for 52-86% of all ticks; it kept overriding v with
# +/-OBSTACLE_BACKOFF_SPEED, so the LJ law's ~+0.12 m/s was commanded as ~+0.03 m/s and the swarm stalled at x~0.
# Collision protection that remains: the agent-safety speed clamp (gap 0.30 -> 0.05 m) and the direction-aware wall governor.
OBSTACLE_TRIGGER_DIST = -0.5    # front_d/back_d threshold, in sensor_model.py's
                                # normalized units (-1.0=contact, +1.0=nothing sensed) --
                                # more negative is more conservative (reacts only when
                                # very close). Same quadrant reading hebbian_step already
                                # consumes -- see hebbian_swarm_experiment.py's
                                # _debug_front_d/_debug_back_d.
OBSTACLE_TRIGGER_TICKS = 3      # consecutive ticks something must be persistently sensed
                                # dead ahead/behind before the reflex engages (~1.5s at
                                # CONTROL_TICK_SECONDS=0.5) -- matches
                                # STALE_POSE_TICK_THRESHOLD's magnitude, for the same
                                # "don't react to one noisy tick" reasoning.
OBSTACLE_BACKOFF_SPEED = 0.05   # m/s, magnitude of the straight-line backoff/forward-ease
                                # commanded while the reflex is active. Deliberately
                                # small/slow -- this is a separation nudge, not an escape
                                # maneuver.
OBSTACLE_BACKOFF_TICKS = 4      # how many ticks the reflex holds once triggered (~2s)
                                # before re-evaluating from scratch.

IR_BACKOFF_ENABLED = False
# DISABLED 2026-10-04 on request (Lighthouse deployment only): the Thymio-IR emergency backoff in
# hebbian_swarm_experiment._apply_ir_backoff() / the LJ experiment is switched off; IR readings are still
# logged (ir_front_max / ir_rear_max). The position-based OBSTACLE_BACKOFF_* reflex below is unchanged.

IR_OBSTACLE_THRESHOLD = 2000     # raw prox.horizontal units. UNVERIFIED PLACEHOLDER --
                                  # has NOT been measured on this rig's actual robots
                                  # (raw scale depends on surface reflectivity/lighting).
                                  # Before trusting this: print robot.proximity_horizontal()
                                  # at a few known real distances (e.g. 15cm/10cm/5cm/contact
                                  # against another Thymio's actual body, not a hand) and set
                                  # this to comfortably below the contact-range reading.
IR_BACKOFF_SPEED = 0.08          # m/s, straight-line backoff commanded by
                                  # _apply_ir_backoff() in hebbian_swarm_experiment.py the
                                  # instant a front/rear IR sensor crosses
                                  # IR_OBSTACLE_THRESHOLD -- see that function's docstring.
                                  # Faster than OBSTACLE_BACKOFF_SPEED/UNTRACKED_SAFE_V_CAP
                                  # above on purpose: this fires on a direct, low-noise local
                                  # measurement of an imminent physical collision, not an
                                  # inferred/uncertain OptiTrack state, so a firm reaction is
                                  # appropriate here in a way it wasn't for those.
                                  # ir_front_max/ir_rear_max are logged every tick
                                  # specifically so this threshold can be recalibrated from
                                  # real trial data rather than guessed twice.

UNTRACKED_SAFE_V_CAP = 0.03     # m/s. hebbian_swarm_experiment.py._tick() applies this cap
                                # to |v| whenever self_tracked is False, in place of the
                                # normal corridor/agent-safety clamp (neither can be computed
                                # without a real position). Confirmed real mechanism (2026-09):
                                # robots colliding at full, unclamped speed -- tracking drops
                                # out most often from exactly the tight-clustering situations
                                # the agent-safety clamp exists to prevent, and losing our own
                                # position also makes every neighbor read as "far away" to our
                                # own sensing, so hebbian_step tends to command an even LESS
                                # cautious v right when we can verify safety least. Set below
                                # OBSTACLE_BACKOFF_SPEED (0.05) so if that reflex is also
                                # active this cap is the one that binds -- moot in practice,
                                # since _apply_obstacle_backoff() can't even fire while
                                # untracked (front_d/back_d are sentinel-blind too), so this
                                # cap is currently the only thing braking an untracked robot
                                # at all.

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

# =====================================================================================
# --- LJ baseline (lj_baseline_experiment.py only -- unused by the Hebbian controller) -
# =====================================================================================
# The paper's Table 3 "standard collective motion baseline" (Fig. 5a's cluster-4 point):
# a fixed, hand-designed LJ-spacing + heading-alignment + goal-pull control law, no
# learning, no genome. Rule gains copied verbatim from hardware_transfer_test/final/
# lj_baseline/paper_baseline_rules.json -- the exact baseline the paper's simulation
# results were evaluated against -- not re-tuned for hardware. Geometry constants copied
# verbatim from ants26_replication/experiment/config.py's R_CUT/R_MIN/R_ALIGN, for the
# same reason MOTOR_UNITS_PER_MPS etc. above must match the Hebbian genome's own
# training config. The velocity caps are NOT duplicated here -- LINEAR_VEL_MAX/
# ANGULAR_VEL_MAX above are the same 0.2 m/s / pi/5 rad/s in experiment/config.py too,
# so lj_baseline_experiment.py uses those directly.
LJ_R0 = 0.7
LJ_EPSILON = 1.0
LJ_K_ALIGN = 0.0
LJ_K_GOAL = 3.0
LJ_K1 = 0.05
LJ_K2 = 0.5
LJ_U = 0.0

LJ_R_CUT = 3.0
LJ_R_MIN = 0.0
LJ_R_ALIGN = 1.5
