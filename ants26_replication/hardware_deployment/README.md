# Hardware deployment: Hebbian ABCD controller on real Thymio + Raspberry Pi swarms

Deploys a genome trained by `experiment/optimize_hebbian.py` (in this same repo) onto
real hardware via the existing `thymio_swarm_platform` / `thymio_raspberry_swarm_control`
platform (found at `/home/lilly/dev/thymio_swarm/`). This directory does **not** modify
either of those repos -- it's a self-contained package you copy files out of when you're
ready for a hardware trial.

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
| `diagnostics/print_poses_experiment.py` | A calibration helper you deploy first, separately from the real controller (see Calibration below). |

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

**`BATTERY_MODE = "none"` (default).** The battery input is always fed a fixed
placeholder (`BATTERY_SENSOR_PLACEHOLDER = 0.0`) — exactly matching how a genome trained
with `optimize_hebbian.py --no-battery-sensor` was trained. **Deploy a `_nosensor`
genome** with this mode. If you deploy a battery-aware genome instead under `"none"`, its
battery input will only ever see this same fixed value in reality, so expect it to
behave like the "doesn't know its own battery" ablation regardless of which one you
picked.

**`BATTERY_MODE = "simulated"`.** Instead of a real measurement, `wind_battery_model.py`
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
and `HEADING_OFFSET_RAD` until they line up, then redeploy with the real controller.

## Deployment steps

1. Pick (or finish training) a genome — `--no-battery-sensor` for `BATTERY_MODE =
   "none"`, or a battery-aware genome (no `--no-battery-sensor`) for `BATTERY_MODE =
   "simulated"` — ideally via `python ../experiment/optimize_hebbian.py [--no-battery-sensor]
   [--wind-grid 50 ...]`.
2. Run `python local_test_harness.py [genome_path]` locally first — no hardware needed,
   validates the whole pipeline (shapes, bounds, missing-pose handling, and both battery
   modes if `scipy` is installed locally).
3. Calibrate (see above) using `diagnostics/print_poses_experiment.py`.
4. Copy `controller_config.py`, `sensor_model.py`, `hebbian_controller.py`,
   `pose_utils.py`, `motor_utils.py`, `hebbian_swarm_experiment.py` (and, if using
   `BATTERY_MODE = "simulated"`, `wind_battery_model.py` plus `scipy` on that Pi) and the
   trained `.npy` genome file into a new subpackage in your checkout of
   `thymio_raspberry_swarm_control`, e.g. `experiments/hebbian_swarm/`.
5. Register it in that repo's `swarm_project.yaml`:
   ```yaml
   experiments:
     hebbian_swarm:
       class: experiments.hebbian_swarm.hebbian_swarm_experiment.HebbianSwarmExperiment
       tracking: true   # required -- this is what makes the platform push OptiTrack poses at all
   ```
6. From a controller script (see `thymio_swarm_platform/examples/decision_external_repo.py`
   for the general pattern):
   ```python
   client = SwarmClient("<coordinator ip>")
   project = client.project(repository="<your thymio_raspberry_swarm_control fork/remote>",
                             hosts=["thymio-18", "thymio-19", ...])
   await project.install(); await project.update(); await project.activate()
   session = project.session("hebbian-trial-1")
   await session.start("hebbian_swarm", config={
       "genome_path": "experiments/hebbian_swarm/hebbian_save_battery_avoid_all_nosensor_best.npy",
       "hostnames": ["thymio-18", "thymio-19", ...],   # same list, same order, every Pi
       "self_hostname": "thymio-18",  # <-- different per Pi; either template this per
                                      #     robot's config or read it from that Pi's own
                                      #     hostname at daemon-launch time
   })
   # ... await session.stop(); await session.collect_logs()
   ```
   Note `self_hostname` must differ per robot while everything else in `config` stays
   identical — you'll need to either generate one `session.start(...)` config dict per
   host (calling it multiple times with different `hosts=[...]` subsets) or otherwise
   inject each Pi's own hostname before its daemon instantiates the experiment.

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
