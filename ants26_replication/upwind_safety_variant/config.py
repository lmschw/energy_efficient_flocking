"""Central configuration for the ANTS 2026 Hebbian ABCD replication (Mahdavi et al.,
"Energy-Efficient Flocking in Self-Organized Robot Swarms").

VARIANT of ../experiment/: stage 1 is "walk_upwind" instead of "walk_left" -- same net-
progress-along-a-fixed-axis reward shape, but now expressed via WIND_DIRECTION below (so
it reads as "progress against the wind" rather than an unexplained sign flip on x) and
trained WITH wind enabled instead of disabled. This repeats the experiment already run in
../upwind_variant/ (see that package -- n10_seed42 there reached final-stage efficiency
99.33 vs. the plain walk_left curriculum's 109.14, i.e. ~91%, a modest but real gap), but
this time forked from ../experiment/ AFTER the safety-clamp/min-dist-inflation/soft-
collision-resolution mechanisms and the KAPPA=10/EFF_DISTANCE_WEIGHT=16 recalibrations
were added there -- ../upwind_variant/ predates all of those and has none of them. The
question this variant answers: does "walk into the wind" still produce comparable
final-stage results once combined with the current (safety-clamp-equipped) pipeline, or
did the earlier ~91%-comparable result depend on specifics of the pre-clamp physics?

Original stage 1 (walk_left) deliberately trained wind-free specifically "to avoid
evolving the trivial strategy of just riding the tailwind" (see ../experiment/config.py's
own comment on this) -- this variant accepts that risk on purpose, to see whether a
genuinely wind-exposed agent can learn to push upwind using ONLY its existing inputs (own
battery level, own heading, neighbor quadrants -- sensor_model.py is unchanged, no
explicit wind/compass sensor added). The premise: battery drains faster the more an agent
fights the wind, so battery level is the only (indirect, interoceptive) cue available for
"am I currently heading upwind" -- the same mechanism save_battery_avoid_all already
leans on, just introduced one stage earlier and made the ONLY objective.

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

HEBBIAN_MIN_DIST_INFLATION = 1.3     # ACTIVE for n20_seed42, hebbian_results_v2_original_n20_
# safety_clamp/ (idea 4, sim-to-real margin) -- reverted to 1.0 for evaluation scripts, per
# this constant's own comment below.
# sim-to-real training margin: multiplies min_dist for
# collision counting/instant-death/proximity-penalty purposes (NOT sensor or wind physics).
# Set >1.0 during TRAINING to make CMA-ES treat agents as if they were bigger than they really
# are, so the learned avoidance keeps a buffer that becomes genuine physical clearance once
# evaluated/deployed at the TRUE (smaller) size -- a standard sim-to-real margin trick, meant
# to absorb hardware imprecision (sensor noise, actuation lag, localization error) that pure
# simulation optimism won't otherwise account for. Leave at 1.0 for evaluation/video/deployment
# so what's reported is the real, un-inflated physical behavior.

HEBBIAN_SAFETY_CLAMP_ENABLED = True  # ACTIVE for n20_seed42, hebbian_results_v2_original_n20_
# safety_clamp/ (idea 1) -- deliberately turned on DURING TRAINING too (not just eval/
# deployment), after a quick check showed bolting this onto an already-trained genome (n20_
# 3stage's, never trained with it) backfired badly: distance collapsed to ~0-1.5m and
# collision_time exploded to 5000s+ (agents kept commanding full speed into neighbors, got
# braked, and gridlocked in a standoff for the whole episode instead of routing around). CMA-ES
# needs to experience the clamp during training to learn to route around it.
# a hard, NON-LEARNED velocity override: if enabled,
# _move() caps each agent's FORWARD speed as a function of its CURRENT gap to the nearest
# neighbor (HEBBIAN_SAFETY_CLAMP_OUTER_GAP -> HEBBIAN_SAFETY_CLAMP_INNER_GAP, linear falloff to
# zero at contact), regardless of what the NN commands. Turning (angular velocity) is left
# untouched, so a braked agent can still steer away. This is deliberately independent of
# training/CMA-ES -- the whole point is a deployment-time safety guarantee that doesn't depend
# on evolution having learned good avoidance, since nothing in this investigation has produced
# a genome that reliably avoids contact on its own. ALWAYS uses the TRUE ROBOT_RAD (unaffected
# by HEBBIAN_MIN_DIST_INFLATION above), since it's meant to mirror a real onboard reflex (e.g.
# a Thymio's IR-triggered emergency brake), not a training-time illusion.
HEBBIAN_SAFETY_CLAMP_OUTER_GAP = 0.30   # gap at which braking begins (full speed above this)
HEBBIAN_SAFETY_CLAMP_INNER_GAP = 0.05   # gap at which forward speed reaches zero
# RECALIBRATED (n20_seed42, hebbian_results_v2_original_n20_safety_clamp/): the original
# values (1x/COLLISION_MIN_DIST_SLACK -> 0.055m/0.01m) were sized off ROBOT_RAD, not off how
# far an agent can actually travel in one simulation step -- at v_max=0.2 m/s and DT=0.5s, a
# single agent can move up to 0.1m per step, and two agents closing head-on can cover up to
# 2*0.2*0.5=0.2m combined -- nearly the ENTIRE old zone width (0.045m), meaning a pair
# approaching at full speed could tunnel straight through the whole braking zone and into
# contact within one timestep, before the clamp ever got a chance to meaningfully reduce
# speed. Confirmed empirically: eval showed min_pairwise still collapsing to ~0.0015 for one
# seed even with the clamp ON, and toggling the clamp off at eval barely changed distance/
# battery but cut collision_time from 504.5s to 7.2s -- the clamp was engaging too late to
# matter, only holding agents in a prolonged near-contact stall rather than preventing the
# approach. New zone width (0.30-0.05=0.25m) is comfortably larger than the 0.2m worst-case
# per-step closing distance, giving genuine deceleration room before contact is possible.

HEBBIAN_WALL_SAFETY_CLAMP_OUTER_GAP = 0.15   # gap to the wall at which braking begins
HEBBIAN_WALL_SAFETY_CLAMP_INNER_GAP = 0.02   # gap at which forward speed reaches zero (~contact)
# Same idea as HEBBIAN_SAFETY_CLAMP_* above, applied to wall proximity instead of nearest-
# neighbor proximity -- added (n20_seed42, hebbian_results_v2_original_n20_safety_clamp_wall/)
# after the recalibrated agent-agent clamp achieved genuine zero inter-robot contact
# (min_pairwise ~0.16, coll_time=0.0 across all 3 seeds) but pushed wall_collision UP sharply
# (172s avg vs n20_3stage's 27.5s) -- the policy apparently routes toward open wall-adjacent
# space to escape the braking penalty near other agents. Governed by the SAME
# HEBBIAN_SAFETY_CLAMP_ENABLED master switch (not a separate flag) since both are the same
# "idea 1" hard safety layer, just against a different hazard. Zone narrower than the
# agent-agent one (0.13m vs 0.25m) because a wall doesn't move -- worst-case single-step
# closing distance toward a fixed wall is v_max*dt=0.1m (vs 0.2m for two closing agents), so
# less margin is needed to still be comfortably wider than that.

HEBBIAN_RESOLVE_COLLISIONS = True
HEBBIAN_RESOLVE_COLLISIONS_STRENGTH = 0.5
HEBBIAN_RESOLVE_COLLISIONS_MAX_ITER = 2
# Turned back ON for n20_seed42, hebbian_results_v2_original_n20_safety_full3stage2/: the full
# 3-stage retrain with BOTH safety clamps active (hebbian_results_v2_original_n20_safety_
# full3stage/) converged to a "parking" pathology -- forward speed decelerating smoothly to
# zero as an agent nears a wall (HEBBIAN_WALL_SAFETY_CLAMP_INNER_GAP) is a genuine stable fixed
# point of the dynamics: once two agents both brake to a full stop at the SAME wall offset with
# no lateral force pushing them apart (the clamp only ever scales forward speed, which sets BOTH
# dx and dy via heading -- there is no separate sideways term), they can settle at literally
# identical (x, y) coordinates and stay there (confirmed via exact float comparison, not
# rounding -- e.g. seed 123 landed both agents at exactly (-5.9230619, -4.925)). Worse, eval
# showed the clamp ACTIVE produced MORE collision time than clamp OFF (24.2s vs 3.2s mean) --
# backwards from the entire point of a safety layer. Re-enabling this ALREADY-TUNED mechanism
# (see its own history above) gives agents an explicit physical push apart whenever they
# actually overlap, independent of whatever the braking dynamics converged to -- breaks the
# fixed point directly rather than trying to prevent the clamp from ever reaching it.
# If HEBBIAN_RESOLVE_COLLISIONS is True, _move() (simulation_hebbian.py) physically
# pushes overlapping agents apart toward min_dist every step, AFTER the collision-count/
# fitness-penalty bookkeeping but BEFORE the wall clamp -- reviving a mechanism that
# exists in the MATLAB reference (simulation_free_global_mod_2.m's move()) but is left
# commented out there, so neither the reference nor this port's own prior behavior ever
# actually prevented agents from overlapping -- both only counted and penalized it
# after the fact via the fitness formula. The fitness penalty is UNCHANGED (still
# counts every step an agent was commanded onto a collision course, same as before).
#
# Round 1 (strength=1.0, max_iter=10 -- MATLAB-faithful full snap-to-min_dist,
# n10_seed42 only): WORSE than plain weight=16 on every axis -- battery 22.80%->7.06%,
# and the controller drove into MORE collision-course commands than any other variant
# tried this session (357.5s). Full-strength resolution appears to remove the physical
# consequence of crowding so completely that nothing is left teaching the controller to
# avoid it -- the fitness-visible collision-count penalty alone (collision_w=250) isn't
# enough on its own.
#
# Round 2, first attempt (strength=0.5, max_iter still 10): each pass halves the
# REMAINING gap, so 10 passes at 0.5 still converges to within ~0.001 of min_dist --
# essentially indistinguishable from strength=1.0's final settled position, just via a
# different within-step path (verified numerically before launching -- caught before
# wasting a training run on a change with no real behavioral difference). Lowering
# strength alone does nothing unless max_iter is ALSO small enough that resolution
# doesn't fully converge within a single step.
#
# Round 2, corrected (strength=0.5, max_iter=2): converges to ~0.095m for a hard
# pileup, genuinely short of the 0.12m floor -- real, lingering residual crowding that
# persists into the next step's sensing/wind-drag physics rather than being erased,
# while still capping the worst-case (an uncapped free-for-all could in principle let
# agents drift arbitrarily close). This is the version actually being trained.

HEBBIAN_COLLISION_INSTANT_DEATH = False
# Turned back OFF for n20_seed42, hebbian_results_v2_original_n20_safety_clamp/: that curriculum
# test (below) confirmed instant-death flatlines at n=20 regardless of how the population is
# prepared beforehand -- it's a dead end at this swarm size. Testing the hard safety CLAMP
# (HEBBIAN_SAFETY_CLAMP_ENABLED, idea 1) + inflated training radius (HEBBIAN_MIN_DIST_INFLATION,
# idea 4) instead, as a clean single-variable-change test against the n20_3stage baseline,
# with instant-death and the graduated proximity penalty both OFF so this doesn't compound with
# either of those separate (and separately inconclusive) mechanisms.
# Previously turned back ON for n20_seed42, hebbian_results_v2_original_n20_curriculum/ STAGE 3: this is
# the curriculum-design test -- unlike hebbian_results_v2_original_n20_proximity_instant/
# (which flatlined, seeded from a genome with NO collision-awareness), this run seeds from
# hebbian_results_v2_original_n20_curriculum/n20_seed42/hebbian_save_battery_avoid_wall_best.npy,
# a stage-2 genome that already trained WITH the graduated proximity penalty (no instant-death)
# during save_battery_avoid_wall. The idea: a population that already avoids proximity
# somewhat shouldn't hit the near-universal-early-death wall instant-death caused before.
# Previously turned back OFF for n20_seed42, hebbian_results_v2_original_n20_proximity_noinstant/: the
# proximity+instant-death combo at n=20 (hebbian_results_v2_original_n20_proximity_instant/)
# collapsed completely -- episodes 3-15 steps, fitness flatlined at ~-24 for all 100
# generations. Root cause: going 10->20 agents doesn't just double collision risk, the number
# of POSSIBLE colliding pairs goes from C(10,2)=45 to C(20,2)=190 (>4x), so "any single pair
# touches" became nearly unavoidable within the first few steps for every candidate --
# CMA-ES had zero gradient from generation 1. Testing the graduated proximity penalty ALONE
# (no hard cliff) here, matching the pattern that worked better at n=10
# (hebbian_results_v2_original_proximity_penalty_close_noinstantdeath/, below).
# Earlier (n10) history:
# both proximity-penalty runs (wide zone AND halved zone) converged on near-zero
# min_pairwise... no, converged on LARGE min_pairwise (0.17-0.24m, well clear of even the
# halved 0.055m warning boundary) and exactly 0.0 proximity_penalty -- evolution wasn't
# threading the warning gradient at all, it was avoiding the whole risky region outright,
# and halving the zone width barely changed that (min_pairwise went UP, if anything).
# Working theory: instant-death's all-or-nothing risk (ANY contact ends the WHOLE episode)
# dominates the graduated penalty's actual shape -- no matter how close the warning zone
# lets agents get, a genome that ever misjudges and touches loses everything, so CMA-ES
# converges on "never approach" regardless of geometry. This run isolates the graduated
# penalty as the ONLY collision-avoidance signal (no hard cliff backstop) to test whether
# that's really what's suppressing close formation, or whether the graduated penalty alone
# would have done the same thing anyway.
# If True, simulate_hebbian_episode() ends the WHOLE episode the instant any pair of
# agents crosses min_dist (the same threshold the collision-count fitness penalty
# already uses) -- same pattern as the existing battery-empty termination ("any one
# agent ends it for everyone"), just for collision instead of battery. The collision-
# count bookkeeping is unaffected (still increments normally up to and including the
# fatal step), but since the episode is now cut off at first contact, collision_time
# will typically end up small/near-zero regardless of genome quality -- the real
# consequence lands on dist_travelled and average_batt instead, both of which get
# starved by a short episode. Mutually exclusive in practice with
# HEBBIAN_RESOLVE_COLLISIONS=True: resolution keeps every pair AT OR ABOVE min_dist by
# construction, so it would make this flag permanently vacuous if both were on at once
# -- disable resolution when testing this. Untested as of writing; the risk is that an
# early, mostly-random population may die almost instantly in nearly every candidate,
# giving CMA-ES too flat/uninformative a fitness landscape to climb -- worth watching
# the first several generations' fitness curve for exactly that failure mode.
#
# UPDATE: that risk was confirmed -- a solo run (n10_seed42, proximity penalty OFF) found
# 2 of 3 held-out seeds learned to never trigger instant-death at all (0 collisions), but
# with WORSE battery than the plain weight=16 baseline (10.9%/14.6% vs ~27.5% avg), and the
# 3rd seed collapsed into a 27-step near-zero-distance episode -- instant-death alone gives
# CMA-ES no gradient to climb before the cliff, so evolution either avoids the whole
# neighborhood (seeds 123/777, sacrificing whatever benefit close formation was providing)
# or falls in it (seed 42). See HEBBIAN_PROXIMITY_* below for the graduated warning penalty
# added to give it that gradient, pairing with this flag rather than replacing it.

# --- Graduated proximity penalty (soft warning zone before instant-death's hard cliff) ---
# Continuous, distance-based penalty on inter-agent surface gap (center distance minus
# 2*ROBOT_RAD), building in TWO zones so CMA-ES has an anticipatory gradient to climb away
# from collisions, instead of the all-or-nothing signal HEBBIAN_COLLISION_INSTANT_DEATH
# gives alone (see that flag's UPDATE note -- this is a direct response to what went wrong
# there). "Agent size" here is ROBOT_RAD (the unit already used everywhere else in this
# file), so OUTER_GAP=2*ROBOT_RAD reads as "twice the agent size" and INNER_GAP=1*ROBOT_RAD
# as "one agent size," both measured from each agent's own surface (not center).
#
#   gap >= OUTER_GAP                       : no penalty (safe zone)
#   INNER_GAP <= gap < OUTER_GAP           : penalty ramps 0 -> 1.0, quadratically, as gap
#                                             shrinks toward INNER_GAP (gentle warning)
#   contact_gap <= gap < INNER_GAP         : penalty ramps 1.0 -> (1.0+STEEP_MULT),
#                                             quadratically, as gap shrinks toward
#                                             contact_gap (STEEP_MULT sets how much harsher
#                                             this zone is than the mid zone -- e.g. 10.0
#                                             means the steep zone's penalty climbs 10x
#                                             farther over a comparable width, so a genome
#                                             approaching actual contact feels a sharply
#                                             rising gradient, not a constant-slope ramp)
#   gap < contact_gap (i.e. touching, D < min_dist) : this is the existing collision-count
#                                             threshold; HEBBIAN_COLLISION_INSTANT_DEATH
#                                             (if on) ends the episode here, same as always
#
# Per-stage strength is set via HEBBIAN_STAGE_FITNESS_WEIGHTS' new proximity_w divisor
# (eff -= proximity_penalty / proximity_w), analogous to collision_w. When proximity_w is
# active for a stage, that stage's collision_w should normally be None -- the graduated
# penalty is meant to REPLACE the flat collision-time penalty as the anticipatory shaping
# signal for that stage, not stack with it (both penalizing the same underlying event would
# double-count and make the two terms' relative weight harder to reason about).
HEBBIAN_PROXIMITY_OUTER_GAP = 1.0 * ROBOT_RAD   # 0.055m -- HALVED from 2*ROBOT_RAD (n10_seed42,
HEBBIAN_PROXIMITY_INNER_GAP = 0.5 * ROBOT_RAD   # 0.0275m -- HALVED from 1*ROBOT_RAD) after the
HEBBIAN_PROXIMITY_STEEP_MULT = 10.0             # first pass (2x/1x agent-size zones) came in worse
# on battery than weight=16 alone (~12.3% vs ~27.5% avg) despite working exactly as intended
# (clean zero-collision avoidance, no degenerate collapse) -- the working theory (see
# HEBBIAN_COLLISION_INSTANT_DEATH's UPDATE note and this section's own history) is that
# whatever battery benefit the tight-formation "rotation strategy" gets comes WITH close
# spacing (likely wind-shielding/drafting), so a warning zone that starts pushing agents
# apart a full agent-radius-plus out from contact may be triggering well before it needs to,
# discarding that benefit for clearance the genome didn't actually need. Halving both
# thresholds lets agents approach twice as close before feeling anything, tightening the
# warning zone around the actual contact threshold rather than the wider safety margin
# tried first -- proximity_w unchanged (250.0) pending a fresh numeric check against these
# narrower zones before the next training run.

# --- Wind-tracking camera window (the x-range RayTraceCircularRobots is evaluated over) ---
WIND_TRACKING_WINDOW_WIDTH = 10.0    # total width of the tracking window [m] (nominally X_RANGE's span)
WIND_TRACKING_MAX_SPAN = 9.8         # cap on the swarm's own x-extent within that window [m]

# --- Wind / wake ray-tracing (RayTraceCircularRobots) ---
UINF = 100.0                  # freestream ("full power") wind value
KAPPA = 10.0                   # drag force scale factor -- battery drain from wind exposure enters
                               # F_drag = 0.5*DRAG_AIR_DENSITY*DRAG_COEFFICIENT_AREA*KAPPA*v_rel^2.
                               # MATLAB reference (simulation_free_global_mod_2.m:39 and the LJ
                               # baseline's :39) uses kappa=10; the paper's own Eq. 3 states kappa=10
                               # explicitly too. Earlier tested kappa=10 alone (hebbian_results_v2_
                               # original_kappa10/n10_seed42) and reverted to 20 since it "fixed
                               # nothing" on its own (collision time as a fraction of episode length
                               # went UP). Reverted to 10.0 here (n10_seed42, hebbian_results_v2_
                               # original_paperbattery/) as part of a combined "match the paper's
                               # battery model exactly" test, alongside BATTERY_DRAIN_SCALE=1.0 (see
                               # that constant's own comment) -- per the user's explicit request to
                               # try both corrections together after testing the drain-scale fix
                               # alone. As v_rel^2, this is the strongest lever on drain rate.
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
BATTERY_DRAIN_SCALE = 2.0            # REVERTED back to 2.0 -- briefly tested 1.0
# (hebbian_results_v2_original_drainscale1/, hebbian_results_v2_original_paperbattery/,
# hebbian_results_v2_original_paperbattery_3stage/, hebbian_results_v2_thymio_newest/) on the
# theory that the paper's own Eq. 6 -- B(t+Delta_t) = B(t) - P_use*Delta_t, no separate scale
# factor -- meant this constant should be 1.0, since DT=0.5 and the old value of 2.0 multiply
# to exactly 1.0 (i.e. drain applied at full weight per step, zero net Delta_t scaling). THIS
# WAS A MISTAKE: the paper's prose equation doesn't match its own released code. The actual
# MATLAB reference (simulation_free_global_mod_2.m:496) reads literally
# "agents(:,4) = agents(:,4) - 2*batt_drain(:);" -- the x2 is genuinely in the code that
# presumably produced the paper's results, not an artifact of this port. This whole
# investigation otherwise treats the MATLAB reference as ground truth over the paper's prose
# (see e.g. KAPPA's history below); this constant should have followed the same rule and
# didn't. Net effect of the mistaken 1.0 value: episode length roughly tripled (battery lasts
# ~2x longer per step, so ~2x more steps to drain it -- compounding with other effects to
# ~2.3-3x), which cascaded into collision-avoidance training landing in much worse basins
# across every variant tried under it, and a trained-vs-baseline comparison where the trained
# controller trailed the baseline on BOTH distance and battery -- contradicting the paper's
# own headline result (90% MORE distance than baseline) rather than reproducing it. That
# contradiction is what surfaced the error. KAPPA=10 below is unaffected by this and remains
# correct -- confirmed in both the paper's Eq. 3 AND the MATLAB code's literal `kappa = 10;`.

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
HEBBIAN_STAGES = ("walk_upwind", "save_battery_avoid_wall", "save_battery_avoid_all")
HEBBIAN_STAGE_WIND_ENABLED = {
    "walk_upwind": True,   # DIFFERENT from ../experiment/'s walk_left (False) -- see module docstring
    "save_battery_avoid_wall": True,
    "save_battery_avoid_all": True,
}
# WIND_DIRECTION: unit-ish vector dist_travelled is projected onto (simulation_hebbian.py,
# lj_baseline.py) instead of the bare "-x" ../experiment/ uses. (-1, 0) points the same way
# -x already did -- wind blows toward +x (see wind_physics.py's wake propagation), so this
# is "into the wind", not a behavior change on its own; what actually differs is that
# walk_upwind (unlike walk_left) trains with wind ENABLED, so stage 1 must learn to make
# genuine upwind progress rather than just picking any fixed heading in still air.
WIND_DIRECTION = (-1.0, 0.0)
# Fitness weights per stage: eff = HEBBIAN_EFF_DISTANCE_WEIGHT*dist + batt/BATTERY_W -
# (collision_time + WALL_COL_MULT * wall_collision_time) / COLLISION_W - cohesion_dist /
# COHESION_W - proximity_penalty / PROXIMITY_W. A weight of None means that term is
# entirely absent (matching Table 2's stage 1 having no battery or collision terms, and
# stages 2/3 excluding inter-robot/wall collisions respectively from view of that specific
# denominator). See HEBBIAN_PROXIMITY_* above for what proximity_penalty measures --
# normally mutually exclusive with collision_w's inter-robot component (proximity_w
# replaces it as the anticipatory shaping signal rather than stacking with it).
HEBBIAN_STAGE_FITNESS_WEIGHTS = {
    #                             battery_w   collision_w   wall_col_mult   include_inter_robot_collision   cohesion_w   proximity_w
    "walk_upwind":                (None,        None,         3.0,            False,                          None,        None),
    "save_battery_avoid_wall":   (5.0,         500.0,        3.0,            False,                          None,        None),
    "save_battery_avoid_all":    (5.0,         250.0,        3.0,            True,                           None,        None),
}
# Reverted proximity_w/include_inter_robot_collision back to the established n20_3stage
# baseline (flat collision_w, no graduated penalty) for n20_seed42, hebbian_results_v2_
# original_n20_safety_clamp/: testing the hard safety clamp + inflated training radius
# (ideas 1 and 4 -- see HEBBIAN_SAFETY_CLAMP_* and HEBBIAN_MIN_DIST_INFLATION above) as a
# clean, single-variable-change test against the proven n20_3stage recipe, rather than
# compounding with the separate (mixed-results) graduated-proximity-penalty line above.
# save_battery_avoid_wall's proximity_w was previously added (None->250.0) for n20_seed42,
# hebbian_results_v2_original_n20_curriculum/: a new 3-stage curriculum design where stage 2
# gets inter-robot collision-awareness (via the graduated proximity penalty, same as stage 3)
# WITHOUT instant-death, so the population entering stage 3 already has some collision-avoiding
# behavior baked in before the harsh instant-death cliff is introduced there -- the idea being
# that stage 3 candidates seeded from a collision-aware stage-2 genome are much less likely to
# ALL die immediately (the failure mode that flatlined hebbian_results_v2_original_n20_
# proximity_instant/ at fitness ~-24 for all 100 generations). wall_col_mult/collision_w still
# apply to WALL collisions only for this stage (include_inter_robot_collision was already False
# here, so no double-counting with proximity_w).
# n20_seed42, hebbian_results_v2_original_n20_proximity_instant/: include_inter_robot_collision
# flipped True->False and proximity_w set to 250.0 (replacing the flat inter-robot penalty with
# the graduated one, wall_col_mult/collision_w still apply to WALL collisions unaffected).
# proximity_w=250 chosen to match collision_w's convention, then numerically checked against
# the n_agents=20 winning genome before committing: that genome's proximity_penalty=8408.20
# over one episode (chronic collider) / 250 = 33.6 points, ~8% of its distance term
# (16*25.52=408.25) -- a meaningful but not dominant fraction, similar order of magnitude to
# earlier n10 calibration checks.
# collision_w reverted 105->250 -- the rescale-to-105 experiment
# (hebbian_results_v2_original_paperbattery_rescaled/) was built on BATTERY_DRAIN_SCALE=1.0,
# which turned out to be a mistake (see that constant's own comment) and was killed mid-run.
# The numeric finding that motivated killing the theory in the first place still stands: at
# collision_w=250, the collision penalty is NOT diluted as a fraction of the distance term
# under longer episodes (it was actually a larger fraction, 0.335% vs the old regime's 0.12%),
# so collision_w's magnitude was never the thing distinguishing clean from chaotic runs.
# proximity_w reverted to None / include_inter_robot_collision back to True -- the graduated
# proximity-penalty line of experiments (wide zone, halved zone, halved zone without
# instant-death) is concluded; none beat weight=16 alone. See HEBBIAN_PROXIMITY_* above and
# HEBBIAN_COLLISION_INSTANT_DEATH's history for the full record. Restored to the established
# baseline (flat collision_w=250 penalty) before starting the battery-drain-scale experiments
# below.
# save_battery_avoid_all's include_inter_robot_collision flipped True->False and
# proximity_w set to 250.0 (n10_seed42, hebbian_results_v2_original_proximity_penalty/) --
# replacing the flat inter-robot collision_time penalty with the graduated
# HEBBIAN_PROXIMITY_* warning term for THIS experiment (wall_col_mult/collision_w still
# apply to wall collisions, unaffected). proximity_w=250.0 chosen to match collision_w's
# established divisor for continuity, then checked numerically (not just guessed) before
# launching: on the weight=16 baseline genome (n10_seed42, moderate chronic crowding),
# proximity_penalty/250 ~= 3.2 fitness points; on a fresh random/unevolved genome (worse
# crowding), ~= 6.0 points -- both a small-but-real fraction of typical fitness magnitude
# (~150-160 from distance alone), noticeably stronger signal than the old flat
# collision_time term ever produced (collision_w=250 on 50.5s of collision_time was only
# ~0.2 points) without being so strong it risks the collision_w=15 experiment's failure
# mode (agents spreading out with no compensating benefit -- see that experiment's
# history above). REVERT include_inter_robot_collision to True and proximity_w to None
# to restore the pre-experiment baseline.
# NOTE: collision_w=50 (down from 250) was used for the weight=16+collision_w=50
# combined test (hebbian_results_v2_original_w16c50/), launched and already running
# under its own imported config -- reverted back to 250 here so the collision-resolution
# test (HEBBIAN_RESOLVE_COLLISIONS above) launches against the established-best
# weight=16/collision_w=250 baseline instead, isolating physical resolution as the one
# new variable relative to that baseline.
# REVERTED to the true original weights (collision_w=250, no cohesion) -- the
# collision_w=15/cohesion_w=1.0 tuning below is preserved as a record of what was tried
# and why, but is currently INACTIVE. Reverted specifically to test the KAPPA=10 fix (see
# that constant's own comment) in isolation against the unmodified original, rather than
# compounding two unvalidated changes at once.
#
# --- history below: the collision_w=15/cohesion_w tuning investigation (inactive) ---
# save_battery_avoid_all's collision_w dropped 250 -> 15 (~16.7x stronger penalty): the
# n10_seed42 original-sweep genome was found (by direct simulation, not just the fitness
# formula) to spend ~195s of its ~222s episode with agents in active pairwise collision
# (mean inter-agent spacing collapsing from 1.69 at spawn to 0.55 by episode end) while
# covering essentially the same distance as the zero-collision LJ baseline (12.40m vs
# 12.51m) -- at collision_w=250 that entire 195s of chronic collision cost only ~0.78
# fitness points against distance's 8.0-per-meter weight, nowhere near enough to compete.
# At 15, the same 195s would cost ~13 points, roughly comparable to 1.6m of distance --
# enough to actually matter without reopening the original "just don't move" exploit
# (HEBBIAN_EFF_DISTANCE_WEIGHT below is unchanged, so distance is still dominant; the
# floor on how much collision-avoidance can cost is bounded by episode length, not
# unbounded like the pre-distance-weight battery term was).
#
# cohesion_w=2.0 added after that fix backfired: collision_time dropped 195s->30s as
# intended, but mean battery went DOWN (17.4%->14.8%), not up -- collision penalty alone
# just spreads agents out (end-of-episode spacing 0.55->0.92) without giving them any
# reason to stay near each other, so whatever battery benefit the old (colliding) tight
# formation incidentally provided (most plausibly wind-shielding/drafting) was lost with
# nothing replacing it. cohesion_dist (stage_fitness()'s docstring) is a PURELY
# GEOMETRIC term -- mean pairwise inter-agent distance, no velocity/heading/direction
# component at all -- so it can only ever reward being closer together; it cannot reward
# moving, turning, or any specific maneuver, leaving whatever behavior achieves that
# closeness (e.g. drafting formations) entirely up to CMA-ES/Hebbian learning to
# discover, same as every other term here. Deliberately small relative to the collision
# penalty: at cohesion_w=2.0, going from spawn spacing (~1.69) to the old tight-cluster
# spacing (~0.55) is worth only ~0.57 fitness points -- a nudge, not a dominant term --
# so it shouldn't be able to out-compete collision_w=15's much larger penalty and pull
# agents back into collision. Untested at other n_agents; the weight may need
# recalibrating if mean pairwise distance scales with swarm size.
#
# UPDATE after 3-seed check at cohesion_w=2.0 (n10, seeds 42/123/777): the effect is NOT
# consistent across seeds. seed42: battery 17.4%->22.2% (better), dist 12.40->10.38m,
# collision_time 195.5->96.0s. seed123: battery 10.3%->9.2% (WORSE), dist ~flat,
# collision_time 131.5->163.0s (WORSE -- cohesion out-pulled the collision penalty here).
# seed777: battery 33.4%->17.2% (much WORSE, nearly halved), dist 7.24->9.69m (better),
# collision_time 326.0->92.5s (much better). Only seed42 got the intended outcome; the
# other two show cohesion interacting unpredictably with whatever local optimum each
# seed's avoid_wall genome started from -- in seed123's case actively re-opening
# collisions cohesion_w=2.0 was supposed to stay clear of. cohesion_w=1.0 here is a
# follow-up tuning attempt: half the pull, testing whether a gentler nudge keeps
# collision_time closer to the collision-only fix's ~30s baseline (across seeds) while
# still buying back some battery, rather than cohesion sometimes overpowering the
# collision penalty as seen at w=2.0. Single-seed test (n10_seed42) pending re-validation
# across seeds if promising -- the 2.0 result above shows single-seed results here are
# not reliable evidence of a weight actually working.

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
HEBBIAN_EFF_DISTANCE_WEIGHT = 16.0
# Sweep result (3 seeds each, n10): 4.0 worse than 8.0 on every axis; 16.0 clearly best
# -- lowest average collision_time (100.2 vs 8.0's 217.7), tied-best average battery
# (21.8%); 32.0 regressed most of the way back to 8.0's level (collision 191.7, battery
# 18.6) -- NOT a monotonic "more weight = less collision" relationship, more like a
# local sweet spot near 16 that 32 overshoots. Left set at 16.0 (the best validated
# value found so far) as the resting state for this file. Still well short of the LJ
# baseline on every seed (baseline battery ~39-52%, zero collision always) -- see the
# full investigation writeup for what's still untested (fine-tuning near 16, e.g. 12 or
# 20; more seeds at 16 for confidence; the never-tried unstaged/literal-MATLAB-formula
# retrain).
# History: dropped 8.0 -> 4.0 first, as an isolated test of whether the 8x weight itself
# was making collision-avoidance's gradient too weak to compete with distance. Result
# (n10_seed42, single seed): WORSE on every axis except raw distance -- battery
# 17.42%->13.48%, collision_time 195.5->208.5s, episode length 222.0->134.5s. Not
# promising enough to justify 2.0 next as originally planned.
#
# Now testing the OTHER direction: 16.0 (doubled from the original 8.0), on the theory
# that if less weight didn't help, perhaps MORE weight (pushing further into the "just
# don't move" failure mode's opposite extreme) reveals something about which direction
# the landscape actually slopes, rather than assuming lower is always closer to
# "correct." 3 seeds this time (42/123/777), not 1 -- the collision_w=15/cohesion_w=2.0
# investigation earlier showed single-seed results here are not reliable evidence either
# way. Only stages 2/3 need retraining per seed (stage 1/walk_left has no competing
# terms, so rescaling this weight doesn't change what it optimizes for -- each seed's
# existing walk_left genome is reused unchanged).

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
