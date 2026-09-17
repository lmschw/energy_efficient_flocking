# Hardware deployment: Hebbian ABCD controller on real Thymio + Raspberry Pi swarms

Deploys a genome trained by `experiment/optimize_hebbian.py` (in this same repo) onto
real hardware via `thymio_swarm_platform` (found at `/home/lilly/dev/thymio_swarm/`).

**The whole repo IS the deployable project.** `/swarm_project.yaml` at the **repo root**
(NOT in this directory -- see that file's header comment for exactly why it can't live
here despite this directory being the only code it needs) registers this directory's
experiments, so `thymio_swarm_platform`'s `client.project()` can clone this repo's GitHub
remote directly -- exactly the same "point the platform at an external repo" pattern
`thymio_swarm_platform/examples/decision_external_repo.py` already uses for a different,
unrelated project. Nothing needs to be copied into any other repo. The **execution**
(controller-side scripts that connect to the coordinator and drive install/start/stop)
lives in `thymio_swarm_platform/examples/hebbian_swarm_trial.py` and
`hebbian_position_heading_calibration.py` instead -- that's the repo with `swarm_platform`
actually installed, and where its other example launchers already live.

## Files

| File | Purpose |
|---|---|
| `/swarm_project.yaml` (repo root, one level up from here) | Registers this directory's experiments with the platform, using dotted `class:` paths like `ants26_replication.hardware_deployment.hebbian_swarm_experiment.HebbianSwarmExperiment`. Read its header comment -- it explains both why it has to be at repo root and why the genome `.npy` still lives flat in here instead of in `../hebbian_results_v2/`. |
| `controller_config.py` | All tunable/calibration constants. **Read this first.** |
| `sensor_model.py` | 4-quadrant range/bearing sensing -- ported verbatim from `experiment/sensor_model.py` (same tested math, only the config import changed). |
| `hebbian_controller.py` | The MLP forward pass + Hebbian update + genome loading -- ported verbatim from `experiment/hebbian_controller.py`. |
| `pose_utils.py` | Converts OptiTrack poses into the `[x, y, heading, battery]` array format the two files above expect. **This is the only real translation layer** between simulation and hardware. |
| `motor_utils.py` | Converts the controller's `(v, w)` output into raw `Robot.drive(left, right)` motor targets. |
| `hebbian_swarm_experiment.py` | The actual experiment class matching the platform's contract (see below). |
| `wind_battery_model.py` | Optional: computes a *simulated* battery level from real robot positions -- ported verbatim from the simulation's wind/drag/battery equations, except its smoothing step (originally `scipy.signal.convolve2d`) is now an exact pure-numpy equivalent (the kernel is separable), so it has no extra dependencies beyond numpy. Only imported when `BATTERY_MODE = "simulated"` (see Battery below). |
| `hebbian_save_battery_avoid_all_best.npy` | The deployed genome -- a copy of `../hebbian_results_v2/hebbian_save_battery_avoid_all_best.npy` (see "Current trial config"). Kept flat here, not referenced from its original location, for the same reason as `swarm_project.yaml`'s note above. |
| `local_test_harness.py` | Validates the whole pipeline with fake robot/pose objects -- **run this before touching real hardware**, since the platform itself has no dry-run mode at all. |
| `diagnostics/calibrate_position_heading_experiment.py` | **Calibration helper (current).** Sweeps a series of straight-line drives and derives `POSITION_AXES`, `HEADING_OFFSET_RAD`, AND `MOTOR_UNITS_PER_MPS` all from the same OptiTrack data -- see Calibration below. Supersedes the two files below (kept, not deleted). |
| `diagnostics/print_poses_experiment.py` | Superseded by `calibrate_position_heading_experiment.py` for position/heading calibration, but still useful as a live raw-pose viewer -- also now the way to measure `CORRIDOR_Y_MIN`/`CORRIDOR_Y_MAX` (see "Corridor wall safety" below): it tracks and prints a running min/max of sim-frame `y` as you walk a robot around. |
| `diagnostics/calibrate_speed_experiment.py` | Superseded by `calibrate_position_heading_experiment.py`. Speed-only calibration helper (`MOTOR_UNITS_PER_MPS`) -- kept for a quick narrower recheck if you don't need the other two constants re-verified. |

Controller-side launchers (in `thymio_swarm_platform/examples/`, not here), run in this
order: `hebbian_position_heading_calibration.py`, `hebbian_swarm_trial.py`. Both point
`client.project()` at this repo's GitHub remote. (`hebbian_pose_calibration.py` and
`hebbian_speed_calibration.py` still exist alongside it for the superseded PI-side
experiments above, but you don't need either one for a normal calibration pass anymore.)

## Why this structure

`sensor_model.py` and `hebbian_controller.py` are pure-numpy and were already verified in
the main simulation package (`experiment/`) — cardinal-direction test cases, shape/bounds
checks, and a real (if small-scale) CMA-ES run that showed genuine learning. They're
copied here unchanged (only the config import differs) rather than depended-on directly,
because the full `experiment/` package pulls in scipy, opencv, pybullet, matplotlib —
none of which have any business being installed on a Raspberry Pi, and none of which this
deployment needs. If you change the sensing or controller math in `experiment/`, port the
same change here by hand; there's no shared import to keep them in sync automatically.

## The experiment contract (from investigating the platform)

There's no formal base class. Every `SwarmDaemon` (one per Pi) instantiates your class as
`experiment_cls(robot=<Robot>, config=<dict from your launch script>, logger=<SessionLogger|None>)`
and calls `await experiment.run()` as a background task; `pause()`/`resume()`/`stop()` are
called in response to session control messages. `HebbianSwarmExperiment.__init__`'s
`config` parameter name matters — it's called with the keyword `config=...`, not
positionally, so don't rename it.

`config` must contain:
- `genome_path` — path (on that Pi) to a `hebbian_<stage>_best.npy` from `optimize_hebbian.py`.
- `hostnames` — the full ordered list of every robot in this run, **identical on every Pi**.
- `self_hostname` — which entry in that list is this robot.

## Sensing: OptiTrack substitutes for onboard range/bearing

`sensor_model.get_sensor_data()` computes a 4-quadrant (front/back/left/right), continuous
distance-and-bearing reading to the nearest neighbor in each quadrant, out to
`SENSING_RADIUS = 2.01` m — reconstructed entirely from `robot.get_all_global_poses()`
(OptiTrack), via `pose_utils.poses_to_agents()`. **This package never calls
`robot.proximity_horizontal()`.** Confirmed directly from `thymio_swarm_platform`'s
`robot.py`: the Thymio's actual onboard sensor (`prox.horizontal`) is 5 front-facing + 2
rear-facing raw IR reflectance readings, effective range roughly 0-12 cm, no meters, no
bearing beyond "which of 7 fixed-angle sensors fired," no dedicated left/right coverage,
and no way to identify *which* neighbor triggered a reading — nothing like the idealized
sensor the controller (and the paper's own model, Section 2.1) assumes.

This isn't a shortcut specific to this deployment — the paper's sensing model was never
meant to correspond to Thymio's onboard IR array, and using motion capture as a stand-in
for an idealized local range/bearing sensor is standard practice in swarm robotics
validation for exactly this reason. But it is a real substitution worth stating plainly:
every method this package calls on `Robot` (`drive()`, `get_all_global_poses()`,
`stop()`) and on the logger (`SessionLogger.log(state, command)`) is confirmed to exist
in `thymio_swarm_platform` as-is — nothing here is invented or assumed — but the
*sensing modality* itself is a mocap-based reconstruction of an idealized sensor, not a
port of the Thymio's real onboard one. Disclose this alongside the battery/wind
substitution below if you're documenting this deployment's fidelity to the paper.

## Battery: there isn't one, and there's no wind tunnel either

Confirmed by directly reading `thymio_swarm_platform`'s `robot.py`/`state.py`/
`system_sounds.py`: **no battery or power reading exists anywhere in this platform**,
for either the Thymio or the Pi. Separately, the simulation's battery drain is driven by
each robot's exposure to a simulated uniform headwind (Section 3 / Eq. 6 of the paper) —
reproducing that physically on real hardware would need an actual wind tunnel.
`controller_config.BATTERY_MODE` picks between two ways of handling this:

**`BATTERY_MODE = "none"`.** The battery input is always fed a fixed
placeholder (`BATTERY_SENSOR_PLACEHOLDER = 0.0`) — exactly matching how a genome trained
with `optimize_hebbian.py --no-battery-sensor` was trained. **Deploy a `_nosensor`
genome** with this mode. If you deploy a battery-aware genome instead under `"none"`, its
battery input will only ever see this same fixed value in reality, so expect it to
behave like the "doesn't know its own battery" ablation regardless of which one you
picked.

**`BATTERY_MODE = "simulated"` (current default -- see "Current trial config" below).**
Instead of a real measurement, `wind_battery_model.py`
computes a *virtual* battery level in software each tick, using the exact same
wind-wake + drag-force + drainage equations (`RayTraceCircularRobots` / `dragforce` /
`batterydrainage`, ported verbatim from
`experiment/simulation_free_global_mod_2_LJ.py`) that the genome was actually evolved
against — driven by each robot's real OptiTrack position (and every other tracked
robot's real position, since the wake field depends on the whole swarm's relative
configuration) instead of a simulated one. Speed and direction-of-travel are derived
from the position delta between ticks (there's no wheel odometry to use instead — see
"Known open risks" below); angular velocity uses the commanded `w` that was actually
active over that interval, matching the simulation's own `vel_actual` convention (see
`move()` in `simulation_free_global_mod_2_LJ.py`). The experiment stops itself
(mirroring the simulation's own termination condition) if the simulated battery reaches
zero. **Deploy a battery-aware (non-`_nosensor`) genome** with this mode — it now has a
real, physically-modeled signal to respond to, rather than a constant.

This mode has no extra dependencies beyond numpy — `wind_battery_model.py`'s smoothing
step originally used `scipy.signal.convolve2d`, replaced with an exact pure-numpy
equivalent (the exponential kernel is separable into two 1D kernels, verified numerically
against the original scipy-based version to floating-point precision before swapping it
in). It has NOT been profiled on real Pi hardware, though — the wake computation is an
O(`NX`) loop plus two smoothing passes run once per control tick, and a Pi is much slower
than a dev laptop; a dev-machine timing came out to ~9ms per call at the default
`NX=NY=200` (500ms `CONTROL_TICK_SECONDS` budget), which leaves real headroom even for a
Pi several times slower, but that is not the same as an actual Pi measurement — check it
actually finishes within budget before trusting a real run, and drop
`controller_config.NX`/`NY` (e.g. to 50, matching `optimize_hebbian.py --wind-grid 50` if
that's the resolution you trained against) if it can't keep up.

**If you use `"simulated"`, disclose it explicitly in your writeup.** The reported
battery level is a physically-modeled software quantity computed from real robot
positions, not a measurement of real power draw — it substitutes for hardware you don't
have (a wind tunnel and battery telemetry), not for the physics itself, but that's a
methodological choice a reader needs to know about.

If you'd rather pursue real telemetry instead: see the negative-finding note in
`controller_config.py` for where you'd start (Thymio's own Aseba variables, or the Pi's
`vcgencmd`/`psutil`) — nothing in this package builds on that path.

## Fidelity caveat: control tick rate

`CONTROL_TICK_SECONDS = 0.5`, matching `experiment/config.py`'s `DT`, **not** the faster
20 Hz tick rate the platform's other example experiments use. The Hebbian weight update
(`eta = 0.1`) is applied once per tick; ticking faster than training used would apply many
more updates per second of real time than the genome ever experienced, changing its
effective learning dynamics. This also happens to match OptiTrack's own ~2 Hz push rate
(see below), so it avoids wasting ticks re-reading a stale, unchanged pose.

## Current trial config (3 real Thymios) -- updated 2026-09-16, final decision

- **Genome: `plain_seed123_clamped_best.npy`** -- replaces the earlier paper-default genome
  (`hebbian_results_v2/hebbian_save_battery_avoid_all_best.npy`, still present in this
  directory as `hebbian_save_battery_avoid_all_best.npy` for reference/rollback). Chosen at
  the end of a full investigation documented in
  `hardware_transfer_test/final/BEAT_BASELINE_INVESTIGATION_LOG.md` -- read that first for the
  complete story; short version: it's a 2-stage (`walk_upwind` -> `save_battery_avoid_all`)
  curriculum genome, `n_agents=20`, seed 123, retrained from scratch with the hard
  inter-agent safety clamp active throughout training (see "Inter-agent safety clamp" below
  -- this is NOT the same genome as the higher-both-beat-rate `plain_seed888`/`drain0_seed123`
  candidates also found by that investigation; `plain_seed123` was chosen specifically because
  it's the only one of the four that doesn't chronically hug/stick to a wall, judged more
  important for a physical trial than its lower simulated both-beat rate).
  `GENOME_PATH_ON_PI` in `thymio_swarm_platform/examples/hebbian_swarm_trial.py` was updated
  to match.
- **Same swarm-size caveat as before, now with real numbers for this exact genome**: it
  reliably beats the LJ baseline at `n_agents=20` (83% both-beat, clamped: 17%) but **not**
  at this trial's `n=3` -- the investigation log's n_agents-sweep finding found zero wins
  below `n=20` for every genome tested, including this one. Treat this trial as a **sim-to-
  real controller-transfer validation**, not a demonstration of "beats baseline" -- that
  claim belongs in simulation results at `n=20`, not this hardware trial.
- **`BATTERY_MODE = "simulated"`** -- this genome was trained WITH the battery sensor
  (`use_battery_sensor=True`), so `"none"` mode would silently feed it a placeholder input it
  never learned to use.
- **`KAPPA`/`NX`/`NY` were corrected** (2026-09-16) to `10.0`/`50`/`50` to match this genome's
  actual training-time physics -- see `controller_config.py`'s comments; the previous
  `20.0`/`200`/`200` values had no justifying comment and did not match any genome ever
  trained in this repo.
- **Hosts:** `thymio-17`, `thymio-18`, `thymio-20` -- their OptiTrack rigid-body mappings
  are in `/swarm_project.yaml` (repo root). If you change `HOSTS`, update both that file's
  `hostname_map` and the `HOSTS` list in all three
  `thymio_swarm_platform/examples/hebbian_*.py` launchers to match.

## Inter-agent safety clamp (deployment-ENFORCED, not just trained-in)

**This is the reason `plain_seed123_clamped` is safe to deploy near other robots at all --
read this before running it.** The clamp that `plain_seed123_clamped` was retrained under is
a *simulation-time* mechanism (`simulation_hebbian.py`'s `_apply_safety_clamp()`); training
under it only shapes what the network *learned*, it does not make the network intrinsically
collision-avoidant by itself. Nothing in this package enforced any inter-agent speed limit at
deployment time until 2026-09-16 (only the corridor/wall governor below existed) -- **deploying
the clamped genome without also enforcing the clamp here would give zero actual collision-
safety benefit over the unclamped genome.** This was nearly missed.

Fixed by porting the same mechanism into `hebbian_swarm_experiment.py`:
`_agent_safety_speed_scale()` computes the real, OptiTrack-measured gap to the nearest other
tracked robot and scales `v` (never `w`, same philosophy as the corridor governor) from 1.0
at `AGENT_SAFETY_CLAMP_OUTER_GAP` (0.30m) down to 0.0 at `AGENT_SAFETY_CLAMP_INNER_GAP` (0.05m,
contact) -- identical formula and thresholds to what the genome actually trained under. In
`_tick()`, this is combined with the corridor governor via `min()`, matching
`simulation_hebbian.py`'s own `np.minimum(agent_scale, wall_scale)` combination exactly:
whichever constraint is more restrictive wins. Verified directly (not just "doesn't crash"):
`_agent_safety_speed_scale` returns 1.0 at a 0.50m gap, 0.6 at 0.20m, 0.0 at and below 0.05m,
matching the formula's own math bit for bit; `local_test_harness.py` (3 robots 0.5-0.6m apart)
runs clean with visibly reduced `v` on ticks where the clamp engages.

**Note this does NOT cover physical de-overlap (`HEBBIAN_RESOLVE_COLLISIONS` in simulation) --
that mechanism teleports/pushes overlapping agents in software, which has no real-world analog
and was correctly never ported.** Only the speed-clamp half is portable, and it's the half
that matters for not colliding at speed.

If you ever deploy a *different* clamped genome, re-verify `AGENT_SAFETY_CLAMP_OUTER_GAP`/
`INNER_GAP` against that genome's own training config before trusting this mechanism for it.

## Other candidates from the same investigation, not chosen

The same investigation (see `hardware_transfer_test/final/BEAT_BASELINE_INVESTIGATION_LOG.md` for
the full comparison) also produced `plain_seed888` and `drain0_seed123` (both clamped and
unclamped forms) with higher both-beat rates than `plain_seed123` in simulation (83-97% vs.
83%, collapsing further under the clamp for all three). They were not chosen because both
chronically hug/stick to a wall at hardware-relevant swarm sizes (`drain0_seed123`: 47-72% of
its total agent-time-budget spent at a wall, even running completely alone; `plain_seed888`:
fine at low n but 40-49% at n=7-20) -- a failure mode invisible in the distance/battery/both-
beat numbers alone, only caught by reviewing generated videos. Their genome files are kept in
`hardware_transfer_test/archive/upwind_2stage_{plain_seed888,drain0_seed123}{,_clamped}/` for
reference if this trial's config ever needs revisiting.

## Corridor wall safety

The deployed genome has **no wall-distance sensory input at all** (`sensor_model.py`'s 10
inputs are 4 neighbor quadrants, own battery, own heading -- nothing about absolute
position or walls). Whatever wall-avoidance it learned in `save_battery_avoid_wall`/
`save_battery_avoid_all` is pure reward-shaping against the training simulation's 20-agent,
`Y_RANGE=[-5, 5]` arena -- confirmed on real hardware to just not generalize to a 3-robot
real room: it drives straight at the corridor edge without slowing at all.

Fixing this for real would mean retraining with an actual wall sensor (see
`../wall_sensor_variant/` -- a separate, still-experimental architecture, not used here).
Instead, `hebbian_swarm_experiment.py` applies a deployment-only safety governor: it
scales the genome's own commanded forward speed `v` down toward 0 as the robot's real
tracked position nears `CORRIDOR_Y_MIN`/`CORRIDOR_Y_MAX` (`controller_config.py`), leaving
the genome's own turning (`w`) untouched. It's a pure speed cap, not a steering override,
and it is **disabled by default** (`v` passes through unmodified) until both bounds are set.

To measure your real corridor bounds before this trial:
1. Deploy `diagnostics/print_poses_experiment.py` with `config = {"hostnames": [...],
   "self_hostname": "..."}` (same as position/heading calibration).
2. Walk (or drive) the robot to each wall of your actual usable runway -- it prints a
   running `corridor y range seen so far: [min, max]` as you go.
3. Set `CORRIDOR_Y_MIN`/`CORRIDOR_Y_MAX` in `controller_config.py` to those values (with a
   little headroom inward, not the exact wall-touching extremes), commit+push, and tune
   `CORRIDOR_SLOWDOWN_MARGIN_M` to your corridor's real width and the robot's real speed.

## Logged data, and comparing a real trial to the simulation results

`hebbian_swarm_experiment.py`'s `_tick()` writes one CSV row per robot per control tick via
the platform's `SessionLogger`, all added 2026-09-16 unless noted:
- `tick` (integer counter), `timestamp` (`time.time()`)
- `x`, `y` (already in the simulation's own coordinate frame/units via `pose_utils.py` --
  directly comparable, not raw OptiTrack), `heading`, `battery` (pre-existing)
- `raw_x`, `raw_y`, `raw_z` and `qx`, `qy`, `qz`, `qw` -- this robot's **complete, untranslated
  OptiTrack reading** (full 3D position + orientation quaternion), logged in addition to the
  derived `x`/`y`/`heading` above so nothing is discarded that a later analysis might want
  (tilt/roll from the quaternion, height/`raw_y` drift on this Y-up rig, an independent
  recomputation of heading). All four/three fields are empty together on any tick this robot
  isn't currently tracked -- same condition `poses_to_agents()` already handles for the
  derived fields, just also applied here.
- `v`/`w` (pre-existing; the **post-clamp** commanded values -- after both the corridor and
  inter-agent safety governors scale them, i.e. what was actually sent to the motors), `left`,
  `right` (pre-existing; raw motor targets, no sim equivalent)

After a trial, each robot's CSV is zipped, pulled to the controller, and merged into one
DataFrame (tagged by a `hostname` column) via
`swarm_platform.utils.unpack_results.aggregate_csvs()`.

Before the `tick`/`timestamp` columns were added, there was no way to align different robots'
rows to the same real instant except assuming a perfectly clean, gap-free
`CONTROL_TICK_SECONDS` cadence per robot -- fragile, since real ticks drift slightly over that
due to per-tick compute/network latency this doesn't otherwise measure. With `timestamp`
present, alignment across robots (needed for anything comparing multiple agents at once) can
be done properly by nearest-timestamp matching instead.

What's directly comparable to `hardware_transfer_test/final/`'s simulation plots:
- **Fig.~6 trajectory analog**: fully direct, `(x,y)` overlays straight onto the same axes.
- **Battery-vs-time** (`battery_plot.png` analog): direct if `BATTERY_MODE="simulated"` --
  but it's a physically-modeled quantity from real positions, not a measurement; say so in
  any caption.
- **Fig.~5a distance-vs-battery scatter**: direct in format, but with 3 robots and a handful
  of trial runs (not 30 seeds) it'll be a few points, not a statistical cloud.
- **Leadership/turn-taking metrics and any inter-agent-distance metric** (collision time,
  wall-sticking-over-time, whether the safety clamp actually engaged): reconstructible from
  `x,y` across robots, aligned via `timestamp`, but still approximate compared to
  simulation's perfectly-synchronized steps -- the clamp's own gap/scale values aren't
  logged directly (only their effect on `v`), so "did the clamp engage on this tick" has to
  be inferred from positions after the fact, not read off a column.

## A subpackage import gotcha (already fixed in this package, worth knowing about)

`thymio_swarm_platform`'s `ProjectLoader` only adds the project's **root** directory to
`sys.path` (confirmed in `swarm_platform/projects/loader.py`) -- which, since
`/swarm_project.yaml` sits at the actual repo root, is the WHOLE repo, not this directory.
So `hebbian_swarm_experiment.py` (registered as
`ants26_replication.hardware_deployment.hebbian_swarm_experiment...`) and everything under
`diagnostics/` alike need this directory added to `sys.path` themselves before their own
bare sibling imports (`import controller_config as cfg`) will resolve -- without it,
they'd raise `ModuleNotFoundError` at daemon-load time. `hebbian_swarm_experiment.py` and
every file under `diagnostics/` (including `calibrate_position_heading_experiment.py`)
each insert their own directory into `sys.path` at the top of the file (before their
sibling imports) to handle this -- if you add another experiment file anywhere in this
package, copy the same three-line shim.

Relatedly, `ProjectManager` never `chdir()`s into the active project directory before
running an experiment, so a bare filename like `config["genome_path"]` is not guaranteed
to resolve relative to the daemon's actual working directory at runtime.
`hebbian_swarm_experiment.py`'s `_resolve_genome_path()` falls back to looking next to its
own file if the given path doesn't already resolve -- so `config["genome_path"]` only
needs to be the `.npy`'s basename, which is what
`thymio_swarm_platform/examples/hebbian_swarm_trial.py` passes.

## Calibration — do this before trusting any real run

Four constants in `controller_config.py` are **unverified placeholders** and will
probably be wrong for your specific rig until you check them:

1. **`POSITION_AXES`** — which two of OptiTrack's `(x, y, z)` map to this codebase's 2D
   ground plane. Motive is commonly Y-up by default (ground plane = X/Z, i.e. `(0, 2)`),
   but this depends entirely on your calibration.
2. **`HEADING_OFFSET_RAD`** — the raw yaw OptiTrack reports when a robot is physically
   oriented at this codebase's `heading=0`. **This one is a dict keyed by hostname, not
   one shared constant** — real robots have disagreed by more than measurement noise
   would explain, most likely because their rigid bodies weren't defined with the same
   "front" convention in Motive. There's no universal right answer for any one robot's
   value; what matters is that each robot's own offset is correct, and that together they
   all agree on whichever physical direction the swarm should treat as "the goal
   direction" (recall: the trained controller always migrates toward -x in its own
   frame). A robot with no entry falls back to `HEADING_OFFSET_RAD_DEFAULT` with a
   one-time warning printed by `pose_utils.py`, not a crash — but don't rely on that for
   a robot you're actually deploying, only for one you haven't calibrated yet.
3. **`ROTATION_SIGN`** — flip this (`1.0` ↔ `-1.0`) if a deployed robot turns the wrong
   way.
4. **`MOTOR_UNITS_PER_MPS`** — the raw `motor.target` value per m/s of real speed.

### The command to run

From an environment with `swarm_platform` installed (its own venv, not this repo's):

```
cd /home/lilly/dev/thymio_swarm/thymio_swarm_platform/examples
python hebbian_position_heading_calibration.py
```

This deploys `calibrate_position_heading` (registered in `/swarm_project.yaml`) to every
host in that launcher's `HOSTS` list at once, drives each one straight for one attempt
(`MOTOR_TARGETS = [300]`, `HOLD_SECONDS = 10.0` — edit those constants at the top of the
launcher if you want a different target/duration), collects logs, and prints (1)-(4) all
together: a per-robot table, POSITION_AXES/MOTOR_UNITS_PER_MPS recommendations aggregated
across all robots (shared constants — one `controller_config.py` for every Pi), and a
**ready-to-paste `HEADING_OFFSET_RAD = {...}` dict literal** (one per `ROTATION_SIGN`
hypothesis — pick whichever matches your separately-observed `ROTATION_SIGN`), since that
constant is per-robot rather than aggregated to one shared value (unlike
POSITION_AXES/MOTOR_UNITS_PER_MPS, which are true platform-wide constants).

**`MOTOR_TARGETS` defaults to ONE attempt, not a multi-point sweep, on purpose.** Real
Thymios don't drive perfectly straight, and there's no way to drive a robot back to an
exact start position/heading between legs on this platform (no wheel odometry, no
closed-loop control at all). A multi-leg sweep doesn't reset between legs, so one
bad/curved leg physically displaces every leg after it — it derails the calibration
rather than improving it. **If a run looks bad** (check the R²/yaw-std warnings it
prints), the fix is to manually put the robot back at its start position and re-run the
same single-attempt launcher again — not to add more targets and hope a sweep averages
it out.

**Give every robot a clear, straight, obstacle-free runway before starting** — a couple
of meters, with no two robots' runways crossing (they calibrate independently and don't
sense each other during this test).

Update `controller_config.py` with what it recommends, commit + push (the Pis pull via
`git`, not your local checkout — see Deployment steps below), then determine (3),
`ROTATION_SIGN`, separately: a straight-line drive carries no information about turn
direction by construction, no calibration procedure can extract it from this data — just
observe which way a deployed robot spins during `hebbian_swarm_trial.py` and flip the
sign if it's backwards.

### Fallback: the old manual/single-purpose scripts still work

`diagnostics/print_poses_experiment.py` (for (1)/(2), by eye) and
`diagnostics/calibrate_speed_experiment.py` (for (4) only) are still in this package,
launched via `thymio_swarm_platform/examples/hebbian_pose_calibration.py` and
`hebbian_speed_calibration.py` respectively, in case you want a narrower recheck of just
one constant, or to watch a robot's raw pose feed live over SSH
(`journalctl -u swarm-daemon.service -f`) for some other reason. For a normal first
calibration pass, `hebbian_position_heading_calibration.py` above does the job of both at
once. One data point worth knowing either way:
`swarm_platform.robot.Robot.get_relative_poses()` itself unpacks a pose's position as
`ox, _, oz = own_pose.position` — i.e. the platform's own code already assumes a Y-up
Motive calibration (ground plane = X/Z, axes `(0, 2)`), which matches this file's own
placeholder guess. Still verify with a real measurement rather than trusting either guess.

## Deployment steps

**Steps 1-2 below are already done in this checkout** for the current trial config (see
"Current trial config" above) -- `/swarm_project.yaml` (repo root) already registers
`hebbian_swarm`, `calibrate_position_heading` (plus the superseded `calibrate_speed` and
`print_poses`), and the genome `.npy` already sits flat in this directory. What's still
open: calibration (step 3, unverified placeholders still in `controller_config.py`), and
pushing this repo's commits so the Pis can actually pull them (git clone/pull runs on the
Pi side -- your local checkout being up to date changes nothing until it's pushed).

0. **`/swarm_project.yaml` must stay at the repo root.** It was originally placed in this
   directory, which worked for the Pi-side loader (`rglob`-based, tolerates nesting) but
   raised `FileNotFoundError` from the CONTROLLER-side `Project.load_config()`
   (`thymio_swarm_platform/swarm_platform/controller/project.py`), which does a plain
   flat `git clone` and hardcodes `<local_clone>/swarm_project.yaml` with no recursive
   search at all. Don't move it back into a subdirectory.
1. Pick (or finish training) a genome — `--no-battery-sensor` for `BATTERY_MODE =
   "none"`, or a battery-aware genome (no `--no-battery-sensor`) for `BATTERY_MODE =
   "simulated"` — ideally via `python ../experiment/optimize_hebbian.py [--no-battery-sensor]
   [--wind-grid 50 ...]`. Copy the resulting `.npy` flat into this directory (see
   `/swarm_project.yaml`'s header comment for why) and point `GENOME_PATH_ON_PI` in
   `thymio_swarm_platform/examples/hebbian_swarm_trial.py` at its basename.
2. Run `python local_test_harness.py [genome_path]` locally first — no hardware needed,
   validates the whole pipeline (shapes, bounds, missing-pose handling, and both battery
   modes).
3. Calibrate: `thymio_swarm_platform/examples/hebbian_position_heading_calibration.py`
   (position axes, heading offset, AND speed, all from one straight-line-drive sweep,
   printed back to the controller machine directly -- see Calibration above). Update
   `controller_config.py`'s placeholders with what it recommends, then determine
   `ROTATION_SIGN` separately by observing a deployed robot's turn direction (that
   script's docstring covers why it can't be derived from the same sweep).
4. Commit and push this repo — the Pis pull it via `git`, they never see your local
   checkout — then run `thymio_swarm_platform/examples/hebbian_swarm_trial.py` (from an
   environment where `swarm_platform` is importable, e.g. its own venv). It uses
   `session.start(experiment, config=shared_config, host_configs=per_host_config)` --
   `host_configs` merges a per-hostname dict on top of the shared `config` for that one
   host (see `swarm_platform.controller.session.SwarmSession.start()`), which is a
   cleaner way to give each robot a different `self_hostname` in a single call than
   calling `session.start()` once per host.

## Known open risks (not resolved by anything in this repo)

- **OptiTrack update cadence vs. reliability**: poses refresh at ~2 Hz; if a robot briefly
  leaves the tracked volume or a marker is occluded, `pose_utils.poses_to_agents` places
  it far away (reads as "no neighbor") rather than crashing, but a genome trained purely
  in simulation has never experienced that specific failure mode and may not respond
  gracefully to it.
- **No wheel odometry/motor feedback exists on this platform** — `drive()` is fire-and-forget;
  there's no way to confirm the robot actually achieved the commanded speed, so any
  mismatch between `MOTOR_UNITS_PER_MPS` and reality directly and silently distorts the
  controller's effective `v`/`w` without any error signal to notice it by. This is also
  why `BATTERY_MODE = "simulated"`'s speed/heading estimate comes from the OptiTrack
  position delta between ticks rather than the commanded velocity — it's the only
  available ground truth, but it's noisier (OptiTrack jitter, ~2 Hz update rate) than a
  real encoder would give.
- **The Hebbian NN was trained entirely in simulation** — sim-to-real gap (wheel slip,
  latency, IMU noise, actual Thymio dynamics vs. the kinematic model) is unvalidated by
  anything in this package; `local_test_harness.py` only proves the code runs correctly,
  not that trained behavior transfers.
- **Swarm-size mismatch**: the current trial's genome was trained at `n_agents=20`; the
  real trial runs 3 robots. Sensing (quadrant-nearest-neighbor) and the controller
  architecture don't depend on swarm size, so nothing *crashes*, but a genome that
  learned its Hebbian update dynamics around 19 simulated neighbors has never experienced
  the much sparser 2-neighbor case a 3-robot swarm actually presents -- expect behavior
  that may look meaningfully different from the paper's own reported results at `n=20`.
