# Hardware validation findings — plain_seed123 (clamped and unclamped), n=5

Investigation log covering the 2026-09-26 hardware validation session: five physical
Thymios (thymio-01/03/07/08/09/11 across sessions — see per-dataset host lists below),
running the `plain_seed123_clamped` and `plain_seed123` (unclamped) genomes in a
corridor with `CORRIDOR_Y_MIN=-1.70`/`CORRIDOR_Y_MAX=1.40` (~3.1m usable width).

Datasets: `eef_clamped_n=5/` (30 runs, clamped genome) and `eef_plain_n=5/` (30 runs,
unclamped genome, collected after the fixes below were deployed).

## 1. Root causes found, and fixes applied

Found and fixed, in the order encountered:

1. **`ROTATION_SIGN` was wrong.** Flipping it (`controller_config.py`) resolved most of
   the reported bad behavior on its own — confirmed by the user directly on hardware.
2. **The agent-agent/corridor safety clamps only ran when `self_tracked` was `True`.**
   The instant a robot's own OptiTrack pose was flagged untracked (which happens often —
   see §2), *all* speed clamping was skipped and the network's raw `v` went straight to
   the motors — at the exact moment (tight clustering / marker occlusion) the clamp
   exists to protect against. **Fix:** `hebbian_swarm_experiment.py`'s `_tick()` now caps
   `|v|` to `UNTRACKED_SAFE_V_CAP` (0.03 m/s) whenever untracked, instead of applying no
   limit at all.
3. **No robot-local, OptiTrack-independent safety signal existed at all.** Added
   `_apply_ir_backoff()`, using the Thymio's own onboard IR (`robot.proximity_horizontal()`
   — 5 front + 2 rear raw values), which backs the robot away/eases it forward the instant
   something is physically close, regardless of OptiTrack/tracking state. Also logged
   `ir_front_max`/`ir_rear_max` every tick for threshold calibration.
   **Known limitation:** front/rear-only coverage — no side sensors exist on stock
   Thymio II, so side-by-side proximity (plausibly the dominant geometry in a narrow
   corridor) isn't caught by this reflex.
4. Added `_apply_obstacle_backoff()` — a persistence-gated (3-tick), OptiTrack-derived
   version of the same idea (front_d/back_d-based), motivated directly by a previously
   documented deadlock (`AGENT_SAFETY_CLAMP_INNER_GAP`'s history: two robots stuck
   together, unable to back away, because the existing clamp only ever suppresses `v`
   toward zero and never reverses it).
5. **"Robots colliding" — retracted.** An initial pass flagged very small OptiTrack
   inter-robot distances (median ~8mm, most runs briefly under the 11cm robot diameter)
   as likely collisions. **The user directly observed no collisions in these trials.**
   Checked whether this was duplicated/conflated rigid-body IDs (it isn't — 99.2% of
   close pairs have genuinely distinct, non-identical coordinates) and concluded it's
   consistent with this project's own already-documented finding
   (`AGENT_SAFETY_CLAMP_INNER_GAP`'s comment: a -0.043m "gap" was previously measured
   between two robots that physically cannot overlap that much) — **OptiTrack's solved
   position gets biased by several cm when two rigid bodies' marker constellations are
   close**, not real contact. This pattern recurs in *both* the clamped and unclamped
   datasets (29-30 of 30 runs each), independent of genome — further evidence it's a
   measurement artifact, not behavior-specific.
6. **Wall-hugging-and-stopping, root cause.** Two independent, unrelated mechanisms,
   confirmed by reproducing the fitness function's own `wall_collision_time` in pure
   simulation (no hardware involved):
   - The training reward (`dist_travelled = -mean(agent_x)`) pushes the swarm to walk
     forever in **-x**, but the **wall-hit check only tests the +x wall**
     (`_move()` in `simulation_hebbian.py`) and the **position clamp only bounds the
     upper x** — the -x direction the genome is actually rewarded to pursue was never
     penalized *or* physically bounded during training. Confirmed directly: the
     `plain_seed123_clamped` n=5 simulation trajectory runs straight through the nominal
     x=-5 domain edge and keeps going to x=-6 to -7 for the rest of the episode
     (`wall_collision_time` = 162.5 of 297.5s, 55%).
   - On real hardware, the actual "hugs a wall" symptom is on the **y-axis** (the
     corridor's narrow, physically bounded dimension) via `CORRIDOR_Y_MIN/MAX`'s
     deployment-only speed governor — a *different*, unrelated mechanism from the sim
     bug above. That governor only slows forward speed near a wall and deliberately
     never corrects heading (documented tradeoff), so a robot whose heading points
     wall-ward just sits there. **This is not fixed** — it's characterized, not solved,
     and still shows up in both datasets (~19-29% of tracked time in the 0.5m brake
     zone).

## 2. Tracking reliability

| | clamped (`eef_clamped_n=5`) | unclamped (`eef_plain_n=5`) |
|---|---|---|
| fraction of ticks tracked | 0.777 mean / 0.883 median | **0.868 / 0.946** |
| permanent end-of-run freezes | 38 / 150 robot-runs | **17 / 150** |
| usable runs for reconfiguration analysis (≥10s continuous, all 5 tracked) | 19 / 30 | **28 / 30** |

Tracking reliability is noticeably *better* in the unclamped dataset — most likely a
session/environment effect (later session, after the ROTATION_SIGN and other fixes were
already in place) rather than a genome effect, but it directly answers the "would going
unclamped make tracking worse" question: no.

## 3. Movement statistics

| | clamped | unclamped |
|---|---|---|
| path length per robot-run (m) | 5.03 mean / 4.93 median | **6.26 / 6.14** |
| net displacement, start→end (m) | 2.48 / 2.28 | 2.20 / 2.06 |
| \|v\| (m/s) | 0.052 mean / 0.030 median | 0.053 / 0.030 |
| \|w\| (rad/s) | 0.406 / 0.462 | 0.486 / **0.613** |
| wall-brake-zone occupancy (tracked ticks) | 0.261 / 0.188 | 0.286 / 0.210 |
| min inter-robot distance per run | 0.023 mean / 0.0079 median | 0.014 / 0.0084 |
| runs with min distance < 11cm (robot diameter) | 27/30 (measurement artifact, §1.5) | 29/30 (same) |
| IR backoff trigger fraction | 3.0% mean | 2.2% mean |

Unclamped covers more ground (6.26m vs 5.03m path length) and turns faster (median
\|w\| 0.61 vs 0.46 rad/s) — consistent with the paper's own claim that the unclamped
genome achieves better raw travel distance. Everything else (speed magnitude, wall-zone
occupancy, the close-distance artifact, IR trigger rate) is essentially unchanged
between the two — the differences that do exist track the genome, not measurement noise.

**Sim-vs-hardware trajectory/speed comparison:**
- Clamped: [`../../collation/diagnostics/sim_vs_hardware_n5_clamped_v2.png`](../../collation/diagnostics/sim_vs_hardware_n5_clamped_v2.png)
  — sim mean speed 0.030 m/s vs hardware 0.071 m/s; sim gets stuck at the wall for over
  half its (longer, 298s) episode, hardware trials are shorter and don't show the same
  dead stretch.
- Unclamped: [`../../collation/diagnostics/sim_vs_hardware_n5_unclamped_v2.png`](../../collation/diagnostics/sim_vs_hardware_n5_unclamped_v2.png)
  — sim mean speed 0.151 m/s vs hardware 0.077 m/s (sim is now the *faster* one, opposite
  direction from the clamped comparison); sim's speed distribution is bimodal (near-0 and
  near-max), hardware's less so.

## 4. Reconfiguration / leadership metrics (Voelkl/Nagy/exchange-rate/Butail-Porfiri-proxy)

Same pipeline as `leadership_metrics.py` (metrics 1-4), applied to real trajectories
instead of re-simulated ones, using the longest continuous all-5-tracked window per run.

| metric | clamped hw (19 runs) | clamped sim ref | unclamped hw (28 runs) | unclamped sim ref |
|---|---|---|---|---|
| reciprocity r | 0.076 / 0.054 | -0.320 | **+0.357 / +0.405** | **-0.265** |
| occupancy entropy | 0.687 / 0.770 | 0.728 | 0.725 / 0.762 | 0.988 |
| exchange rate (/s) | 0.515 / 0.500 | 0.185 | 0.582 / 0.603 | 0.203 |
| exchange rate (/m) | 13.5 / 12.8 | 7.34 | 16.4 / 12.9 | 2.00 |
| persistence rate (/s) | 0.202 / 0.214 | 0.114 | 0.218 / 0.232 | 0.119 |
| hierarchy entropy | 0.836 / 0.851 | 0.776 | 0.871 / 0.855 | 0.986 |
| hierarchy steepness | 0.865 / 0.907 | 1.049 | 0.825 / 0.842 | **0.467** |

**Bottom line: the core paper claim — front position rotates among agents, not
dominated by one robot — replicates on real hardware for both genomes** (occupancy and
hierarchy entropy are both high and in the same range as their sim references).

Two things that don't cleanly replicate, worth a sentence each in the paper rather than
smoothing over:
- **Front-position exchange rate is consistently higher on hardware** (roughly 2-3x, even
  after filtering out sub-3-step noise) for both genomes. Plausibly genuine (a physical
  5-robot group in a narrow corridor has smaller front-back gaps than sim's open field),
  plausibly partly OptiTrack jitter flipping the front rank more easily — can't fully
  separate the two with this data.
- **Reciprocity flips sign, and — with 28 usable unclamped runs instead of 19 — this now
  looks like a real qualitative difference, not sampling noise.** Sim says agents that
  lead a lot do *not* also follow a lot (negative r, both genomes); hardware says the
  opposite (positive r, both genomes, and consistently so for the unclamped set).
- **Hierarchy steepness is much higher (less egalitarian) on hardware for the unclamped
  genome specifically** (0.83 vs sim's 0.47) — a bigger gap than the clamped comparison
  showed (0.87 vs 1.05).

Plots:
- Clamped: [`leadership_analysis/metric_hardware_vs_sim_n5.png`](leadership_analysis/metric_hardware_vs_sim_n5.png)
- Unclamped: [`leadership_analysis/metric_hardware_vs_sim_n5_unclamped.png`](leadership_analysis/metric_hardware_vs_sim_n5_unclamped.png)

### Metric-4-specific hardware plots (persistence-filtered hand-offs)

Both datasets, matching the existing sim pipeline's metric-4 figure types:

| | clamped | unclamped |
|---|---|---|
| per-run handoff timelines (one file per usable run) | `leadership_analysis/metric4_hw_handoff_timeline_eef_clamped_n=5_run##.png` (19 files) | `leadership_analysis/metric4_hw_unclamped_handoff_timeline_eef_unclamped_n=5_run##.png` (28 files) |
| switch-event raster across runs | [`metric4_hw_switch_event_raster_n5.png`](leadership_analysis/metric4_hw_switch_event_raster_n5.png) | [`metric4_hw_unclamped_switch_event_raster_n5.png`](leadership_analysis/metric4_hw_unclamped_switch_event_raster_n5.png) |
| hand-off count box+jitter | [`metric4_hw_run_sweep_n5.png`](leadership_analysis/metric4_hw_run_sweep_n5.png) | [`metric4_hw_unclamped_run_sweep_n5.png`](leadership_analysis/metric4_hw_unclamped_run_sweep_n5.png) |
| exchange rate per run | [`metric4_hw_exchange_rate_filtered_n5.png`](leadership_analysis/metric4_hw_exchange_rate_filtered_n5.png) | [`metric4_hw_unclamped_exchange_rate_filtered_n5.png`](leadership_analysis/metric4_hw_unclamped_exchange_rate_filtered_n5.png) |

## 5. Recommended runs for the paper

Scored on tracked fraction, absence of permanent freezes, low wall-zone occupancy, and
path length (all normalized and averaged):

- **Clamped: `eef_clamped_n=5_run15`** — 95% tracked, zero permanent freezes, 11% wall-zone
  (vs ~26% average), 5.91m path, and representative (not outlier) reconfiguration numbers
  (r=0.30, occupancy entropy=0.92). Trajectory:
  [`../../collation/diagnostics/per_run_trajectories/eef_clamped_n=5_run15_trajectory.png`](../../collation/diagnostics/per_run_trajectories/eef_clamped_n=5_run15_trajectory.png).
  Handoff timeline: [`metric4_hw_handoff_timeline_eef_clamped_n=5_run15.png`](leadership_analysis/metric4_hw_handoff_timeline_eef_clamped_n=5_run15.png)
  — clean rotation through all 5 robots, 15 real hand-offs.
- **Unclamped: `eef_unclamped_n=5_run13`** — 96% tracked, zero permanent freezes, 9%
  wall-zone, 8.60m path (longest clean run in the set). Trajectory:
  [`../../collation/diagnostics/per_run_trajectories_unclamped/eef_unclamped_n=5_run13_trajectory.png`](../../collation/diagnostics/per_run_trajectories_unclamped/eef_unclamped_n=5_run13_trajectory.png).

All 30+30 individual per-run trajectory plots (for picking an alternate, or for
supplementary material) are in:
- `../../collation/diagnostics/per_run_trajectories/` (clamped)
- `../../collation/diagnostics/per_run_trajectories_unclamped/` (unclamped)

## 6. Code changes (this session)

- `ants26_replication/hardware_deployment/controller_config.py` — `ROTATION_SIGN`
  flipped; added `OBSTACLE_BACKOFF_*`, `IR_OBSTACLE_THRESHOLD`/`IR_BACKOFF_SPEED`,
  `UNTRACKED_SAFE_V_CAP`.
- `ants26_replication/hardware_deployment/hebbian_swarm_experiment.py` — added
  `_apply_obstacle_backoff()`, `_apply_ir_backoff()`; untracked branch now caps `v`
  instead of leaving it unclamped; logs `ir_front_max`/`ir_rear_max` every tick.
- `ants26_replication/hardware_deployment/local_test_harness.py` — `FakeRobot` gained a
  `proximity_horizontal()` stub so the harness still runs.

## 7. Still open

- Wall-hugging on the y-axis is characterized, not fixed — no code change addresses it;
  it's a documented, deliberate deployment-side tradeoff (speed-only clamp, no heading
  correction).
- `IR_OBSTACLE_THRESHOLD=2000` is an unverified placeholder — recalibrate from the
  logged `ir_front_max`/`ir_rear_max` columns against known real distances before
  trusting it further.
- The IR reflex cannot detect side-by-side proximity (no side sensors on stock Thymio
  II) — the likely dominant close-approach geometry in a narrow corridor.
- Reciprocity and hierarchy-steepness discrepancies (§4) are real enough now (28-run
  unclamped set) to warrant a sentence in the paper's discussion rather than being
  waved off as noise.
