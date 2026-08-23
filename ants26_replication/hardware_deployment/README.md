# Hardware deployment: Hebbian ABCD controller on real Thymio + Raspberry Pi swarms

Deploys a genome trained by `experiment/optimize_hebbian.py` (in this same repo) onto
real hardware via the existing `thymio_swarm_platform` / `thymio_raspberry_swarm_control`
platform (found at `/home/lilly/dev/thymio_swarm/`). This directory is itself a
self-contained, source-of-truth package (edit/test/calibrate here); its files get copied
out into `thymio_raspberry_swarm_control` as subpackages when deploying (already done for
the "Current trial config" below -- see `experiments/hebbian_swarm/` and
`experiments/calibrate_speed/` in that checkout) since that's the repo the Pis actually
`git pull` their code from.

## Files

| File | Purpose |
|---|---|
| `controller_config.py` | All tunable/calibration constants. **Read this first.** |
| `sensor_model.py` | 4-quadrant range/bearing sensing -- ported verbatim from `experiment/sensor_model.py` (same tested math, only the config import changed). |
| `hebbian_controller.py` | The MLP forward pass + Hebbian update + genome loading -- ported verbatim from `experiment/hebbian_controller.py`. |
| `pose_utils.py` | Converts OptiTrack poses into the `[x, y, heading, battery]` array format the two files above expect. **This is the only real translation layer** between simulation and hardware. |
| `motor_utils.py` | Converts the controller's `(v, w)` output into raw `Robot.drive(left, right)` motor targets. |
| `hebbian_swarm_experiment.py` | The actual experiment class matching the platform's contract (see below). |
| `wind_battery_model.py` | Optional: computes a *simulated* battery level from real robot positions -- ported verbatim from the simulation's wind/drag/battery equations. Only imported when `BATTERY_MODE = "simulated"` (see Battery below). Requires `scipy`. |
| `local_test_harness.py` | Validates the whole pipeline with fake robot/pose objects -- **run this before touching real hardware**, since the platform itself has no dry-run mode at all. |
| `diagnostics/print_poses_experiment.py` | Calibration helper #1 (position axes / heading offset) -- deployed separately from the real controller (see Calibration below). |
| `diagnostics/calibrate_speed_experiment.py` | Calibration helper #2 (`MOTOR_UNITS_PER_MPS`) -- drives a sweep of raw motor targets and measures real speed from OptiTrack position deltas (see Calibration below). |
| `run_speed_calibration.py` | Controller-side launcher for `calibrate_speed_experiment.py` -- deploys it to all of `HOSTS`, collects the per-robot recommended `MOTOR_UNITS_PER_MPS`, prints a summary. Run this from wherever `swarm_platform` is importable, not from this repo. |
| `run_hardware_trial.py` | Controller-side launcher for the real 3-agent trial -- same pattern as `thymio_swarm_platform/examples/decision_external_repo.py`. Run this only after both calibration steps are done. |

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

This mode requires `scipy` (only `wind_battery_model.py` does; it's lazily imported only
when this mode is selected, so it's not a dependency of the rest of the package). It has
not been profiled on real Pi hardware — the wake computation is an O(`NX`) loop plus two
2D convolutions run once per control tick, and a Pi is much slower than a dev laptop, so
check it actually finishes within `CONTROL_TICK_SECONDS` before trusting a real run; drop
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

## Current trial config (3 real Thymios)

- **Genome:** `hebbian_results_v2/hebbian_save_battery_avoid_all_best.npy` -- the
  paper-default run (`n_agents=20`), trained through all 3 curriculum stages
  (`walk_left` -> `save_battery_avoid_wall` -> `save_battery_avoid_all`). Nothing in this
  repo has a genome actually trained at `n_agents=3` (the closest available is `n=4`, in
  `hebbian_results_v2_n4/`) -- running the `n=20` genome with only 2 real neighbors is a
  real, disclosed sim-to-real gap (much sparser than what it was evolved against), a
  deliberate choice over the untested-at-scale `n=4`/`n=7` variants.
- **`BATTERY_MODE = "simulated"`** -- this genome was trained WITH the battery sensor
  (`battery_sensor=True` in its history JSON), so `"none"` mode would silently feed it a
  placeholder input it never learned to use. Requires `scipy` on each Pi.
- **Hosts:** `thymio-15`, `thymio-16`, `thymio-17` -- already have OptiTrack rigid-body
  mappings in `thymio_raspberry_swarm_control/swarm_project.yaml`.

## A subpackage import gotcha (already fixed in this package, worth knowing about)

`thymio_swarm_platform`'s `ProjectLoader` only adds the project's **root** directory to
`sys.path` (confirmed in `swarm_platform/projects/loader.py`), not the directory an
individual experiment file lives in. Once these files are copied into a subpackage like
`experiments/hebbian_swarm/`, their bare sibling imports (`import controller_config as
cfg`, used throughout so `local_test_harness.py` can `python local_test_harness.py`
directly with no package machinery) would raise `ModuleNotFoundError` at daemon-load time
without a fix. `hebbian_swarm_experiment.py`, `diagnostics/print_poses_experiment.py`,
and `diagnostics/calibrate_speed_experiment.py` each insert their own directory into
`sys.path` at the top of the file (before their sibling imports) to fix this -- if you add
another entry-point experiment file to this package, copy the same three-line shim.

Relatedly, `ProjectManager` never `chdir()`s into the active project directory before
running an experiment, so a relative `genome_path` like
`"experiments/hebbian_swarm/hebbian_..._best.npy"` is correct relative to the project
root but NOT reliably correct relative to the daemon's actual working directory at
runtime. `hebbian_swarm_experiment.py`'s `_resolve_genome_path()` falls back to looking
next to its own file if the path doesn't already resolve -- so `config["genome_path"]`
only needs to be the `.npy`'s basename as long as it's deployed alongside
`hebbian_swarm_experiment.py` (see `run_hardware_trial.py`).

## Calibration — do this before trusting any real run

Three constants in `controller_config.py` are **unverified placeholders** and will
probably be wrong for your specific rig until you check them:

1. **`POSITION_AXES`** — which two of OptiTrack's `(x, y, z)` map to this codebase's 2D
   ground plane. Motive is commonly Y-up by default (ground plane = X/Z, i.e. `(0, 2)`),
   but this depends entirely on your calibration.
2. **`HEADING_OFFSET_RAD`** — the raw yaw OptiTrack reports when a robot is physically
   oriented at this codebase's `heading=0`. There's no universal right answer here; what
   matters is that it's consistent across every robot and matches whichever physical
   direction you want the swarm to treat as "the goal direction" (recall: the trained
   controller always migrates toward -x in its own frame).
3. **`ROTATION_SIGN`** — flip this (`1.0` ↔ `-1.0`) if a deployed robot turns the wrong
   way.

To calibrate (1) and (2): deploy `diagnostics/print_poses_experiment.py` instead of the
real controller first. It needs `pose_utils.py` and `controller_config.py` alongside it
(copy it *out of* the `diagnostics/` folder into the same flat directory as the other
files when deploying — it's kept visually separate here only so it isn't mistaken for
part of the real controller). Run it, physically move/rotate a tracked robot, and watch
the printed raw position/yaw values against what you'd expect; adjust `POSITION_AXES`
and `HEADING_OFFSET_RAD` until they line up, then redeploy with the real controller. One
data point worth starting from: `swarm_platform.robot.Robot.get_relative_poses()` itself
unpacks a pose's position as `ox, _, oz = own_pose.position` -- i.e. the platform's own
code already assumes a Y-up Motive calibration (ground plane = X/Z, axes `(0, 2)`), which
matches this file's own guess above. Still verify with the printed values rather than
trusting either guess.

To calibrate (4), `MOTOR_UNITS_PER_MPS`: deploy `diagnostics/calibrate_speed_experiment.py`
via `run_speed_calibration.py` (see below) and use the value it recommends.

## Deployment steps

**Steps 1-3 below are already done in this checkout** for the current trial config (see
"Current trial config" above) -- `thymio_raspberry_swarm_control/experiments/hebbian_swarm/`
and `experiments/calibrate_speed/` already exist there with these files copied in and
registered in that repo's `swarm_project.yaml`. What's still open: calibration (step 4,
unverified placeholders still in `controller_config.py`), and pushing that repo's commits
so the Pis can actually pull them (git clone/pull runs on the Pi side -- your local
checkout being up to date changes nothing until it's pushed).

1. Pick (or finish training) a genome — `--no-battery-sensor` for `BATTERY_MODE =
   "none"`, or a battery-aware genome (no `--no-battery-sensor`) for `BATTERY_MODE =
   "simulated"` — ideally via `python ../experiment/optimize_hebbian.py [--no-battery-sensor]
   [--wind-grid 50 ...]`.
2. Run `python local_test_harness.py [genome_path]` locally first — no hardware needed,
   validates the whole pipeline (shapes, bounds, missing-pose handling, and both battery
   modes if `scipy` is installed locally).
3. Copy `controller_config.py`, `sensor_model.py`, `hebbian_controller.py`,
   `pose_utils.py`, `motor_utils.py`, `hebbian_swarm_experiment.py`,
   `wind_battery_model.py` (needed whenever `BATTERY_MODE = "simulated"`; requires `scipy`
   on that Pi), and the trained `.npy` genome file into a subpackage in your checkout of
   `thymio_raspberry_swarm_control`, e.g. `experiments/hebbian_swarm/` -- and separately,
   `controller_config.py` + `diagnostics/calibrate_speed_experiment.py` into
   `experiments/calibrate_speed/`. Register both in that repo's `swarm_project.yaml`:
   ```yaml
   experiments:
     hebbian_swarm:
       class: experiments.hebbian_swarm.hebbian_swarm_experiment.HebbianSwarmExperiment
       tracking: true   # required -- this is what makes the platform push OptiTrack poses at all
     calibrate_speed:
       class: experiments.calibrate_speed.calibrate_speed_experiment.CalibrateSpeedExperiment
       tracking: true
   ```
   Commit and push -- the Pis pull this repo via `git`, they never see your local checkout.
4. Calibrate, in order: `diagnostics/print_poses_experiment.py` (position axes / heading,
   see above), then `run_speed_calibration.py` (speed). Update `controller_config.py`'s
   placeholders with what you measure, **in both** this repo's `hardware_deployment/` and
   the deployed copy in `thymio_raspberry_swarm_control`, then commit+push the latter again.
5. Run `run_hardware_trial.py` (from an environment where `swarm_platform` is importable,
   e.g. `thymio_swarm_platform`'s own venv -- not this repo's). It uses
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
