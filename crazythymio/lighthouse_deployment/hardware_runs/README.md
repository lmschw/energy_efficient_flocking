# Lighthouse hardware runs (CrazyThymio, 7 robots)

**Tracking system: Lighthouse V2** (base stations + a Crazyflie 2.1 with Lighthouse deck on every Thymio).
This is **not** the OptiTrack data -- that lives in `/hardware_results/` and `/hardware_transfer_test/`, recorded with a
different pose pipeline (`ants26_replication/hardware_deployment/`). Every `run_info.json` here says
`"tracking_system": "Lighthouse V2 ..."` and `"data_format": "lighthouse_v2"`.

## Layout

```
hardware_runs/
  <condition>_rep<k>_<YYYYmmdd_HHMMSS>/   one experiment run, condition = lj | plain | clamped, k = 1..30
  schedule_30x_lj-plain-clamped.txt       the fixed run order (30 x lj, then 30 x plain, then 30 x clamped)
  schedule_reps31-<n>_lj-plain_interleaved.txt   extra repetitions lj/plain 31.. recorded AFTER the main block, alternating
                                          (lj 31, plain 31, lj 32, ...); same settings, ordinary lj_rep<k>/plain_rep<k> folders
  _code/<hash>/                           exact code + genomes + calibration used (one copy per code version)
  _tests_and_trials_2026-10-04/           bring-up tests and trials -- NOT experiment data
```
Only folders matching `<condition>_rep<k>_*` **that contain a `run_info.json`** are complete runs. A folder without
`run_info.json` was interrupted (the schedule repeats that repetition).

## Files in one run folder

| file | content |
|---|---|
| `robot-N_<epoch>.csv` | one row per control tick (0.5 s) of robot N -- columns below |
| `robot-N.console.log` / `robot-N.console.txt` | that robot's console output (`.log` from `run_swarm.sh`, `.txt` from `run_swarm_fast.sh`) |
| `run_info.json` | controller, genome (+ md5), LJ spacing, robots, start time, duration, per-Pi clock offsets, safety settings, arena limits, code version + `code_snapshot` |
| `controller_config.snapshot.py` | all controller / battery / avoidance parameters used |
| `lighthouse_system.snapshot.yaml`, `heading_calibration.snapshot.csv`, `board_offset.snapshot.csv` | calibration used |

## CSV columns

| column | meaning |
|---|---|
| `tick`, `timestamp` | tick counter; unix time **in that robot's Pi clock** (see time base) |
| `x`, `y` | robot centre [m], arena frame (origin = calibration origin, about the arena centre); +x / -x ends of the arena at x ~ +1.40 / -1.84 |
| `heading` | simulation convention: 0 = facing +y, counter-clockwise positive [rad]; facing direction = heading + pi/2 |
| `battery` | simulated battery [%] (wind/drag model from the simulation, driven by the real positions) |
| `wind_pct`, `batt_drain` | simulated wind exposure this tick (100 = full freestream, lower = sheltered by others) and battery drain this tick. Empty on the first tick. |
| `raw_x`, `raw_y`, `raw_z`, `qx..qw` | pose as received (z up, yaw-only quaternion) |
| `tracked` | 1 = own pose valid this tick (0: position is the 1e4 sentinel -- drop such rows) |
| `n_neighbors_seen` | other robots with a valid position this tick |
| `front_d`, `back_d` (`right_d`, `left_d` for Hebbian) | controller's normalised quadrant distances (-1 contact .. +1 nothing within 2.01 m) |
| `ir_front_max`, `ir_rear_max` | Thymio IR proximity (raw); a physical-contact indicator |
| `v_policy` | forward speed the controller asked for [m/s] |
| `scale_avoid` | direction-aware collision avoidance factor (1 = no braking) |
| `scale_corridor`, `scale_agent` | legacy safety layers (1.0 -- switched off in these experiments) |
| `v`, `w` | command after all adaptations [m/s, rad/s]; `left`, `right` = Thymio motor targets |

## Time base
Each Pi logs in its own clock. To put all robots on one time axis: `t_laptop = timestamp - pi_clock_offset_s["robot-N"]`
(from `run_info.json`). The run starts at `start_epoch` and ends at `start_epoch + duration_s` (laptop clock). Measured
start/stop spread across robots: < 0.1 s.

## Metrics as in the simulation
* `dist` (distance travelled upwind): use the **displacement** `mean(x_start) - mean(x_end)` over robots. The simulation's
  `-mean(final x)` only equals that because simulated swarms start near x = 0; these runs start near x = +1.
* `collision_time`: time with a pair of robot centres closer than 2 x ROBOT_RAD = 0.11 m (interpolate all robots onto a
  common time grid first; `ir_*` gives an independent contact check).
* battery: `battery` column (final / mean), `wind_pct` for the sheltering mechanism.

## Known data caveats
* Lighthouse occasionally reports one sample ~10 cm off (seen twice in calibration spins) -- use a median filter or drop
  isolated jumps > 5 cm in one tick.
* **Estimate jumps on collisions:** when robots touch, a robot's Lighthouse/Kalman estimate can jump by > 0.3 m per tick
  and drift for seconds (81% of jump onsets in plain runs coincide with IR contact vs 32% of ticks overall). LJ (no
  contact) 0.1% of ticks affected, plain 2.0%, clamped (reps 1-7, 11) ~1%. Runs are kept (the collisions are the
  controller's behaviour); mask steps > 0.3 m/tick and the following excursion in trajectory analyses, or report with/without.
* **Robot-6 failure 2026-10-05 ~23:09-23:16:** estimate wrong from the first tick (x = +4 .. +90 m) in clamped reps 9 and 10,
  diverged mid-arena at 7 s in rep 8, dropped ticks in rep 11 -> all four moved to `_tests_and_trials_2026-10-04/INVALID_*`
  and repeated. `run_swarm_fast.sh` now also rejects a run when a robot's start position is outside the arena.
* Fixed condition order (no interleaving): slow drifts (Thymio battery level, room) line up with condition.
* Deployment adaptations to state in the paper: see `../FLEET.md` and the comments in `controller_config.snapshot.py`
  (LJ scaled to r0 = 0.5 m, direction-aware avoidance, turn priority at wheel saturation, simulated battery,
  per-robot board/heading/motor calibration, all legacy safety layers off).
