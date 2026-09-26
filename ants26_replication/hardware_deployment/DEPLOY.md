# Deploying `plain_seed123_clamped` to the real Thymio swarm

A copy-paste, step-by-step guide to running the currently-chosen genome
(`plain_seed123_clamped_best.npy`) on the 3 real Thymios. Follow the steps **in order** —
each one exists because skipping it caused a real problem in an earlier attempt (noted inline
where relevant). Every command block says which machine/directory/environment it runs in.

**You do NOT need to redo any training or simulation experiments for this.** Everything below
is about running the *already-trained* genome on real hardware. The one thing you might still
need to do is **calibration** (Step 3) — that's a one-time physical-rig measurement, not an
experiment, and only needed if it hasn't already been done since the fix noted in that step.

---

## Step 0 — Check what's already done (30 seconds)

```bash
cd /home/lilly/dev/energy_efficient_flocking/energy_efficient_flocking
git status              # should say "working tree clean" and "up-to-date with origin/main"
git log --oneline -3    # should show recent hardware_deployment-related commits
```

If `git status` shows anything other than a clean tree, **stop and ask before proceeding** —
uncommitted local changes here would not be what the Pis pull in Step 5.

Then open `ants26_replication/hardware_deployment/controller_config.py` and check lines
~79-127. If you see comments starting with **`STILL STALE / DO NOT TRUST`** or
**`UNVERIFIED`** next to `MOTOR_UNITS_PER_MPS`, `HEADING_OFFSET_RAD`, or `ROTATION_SIGN` —
calibration (Step 3) has not been redone yet and you must do it. If those comments are gone
(replaced with real measured values and a note saying when they were calibrated), **skip
straight to Step 5.**

---

## Step 1 — Local dry run (no robots, no coordinator needed)

Runs entirely on your own machine, using this repo's normal Python environment (just numpy —
no `swarm_platform` needed for this step).

```bash
cd /home/lilly/dev/energy_efficient_flocking/energy_efficient_flocking/ants26_replication/hardware_deployment
python3 local_test_harness.py plain_seed123_clamped_best.npy
```

**Expected output**: ends with `All checks passed.` and no tracebacks. If this fails, **stop
here** — nothing below will work either, and it's much faster to debug on your own machine
than on the real robots.

---

## Step 2 — Confirm the code the Pis will actually pull

The Pis clone/pull `https://github.com/lmschw/energy_efficient_flocking.git` themselves — your
local checkout being up to date means nothing until it's pushed. Step 0 already checked this;
if you had to make any changes since, push them now:

```bash
cd /home/lilly/dev/energy_efficient_flocking/energy_efficient_flocking
git add -A
git commit -m "your message"
git push
```

---

## Step 3 — Calibration (skip entirely if Step 0 said you could)

This is a **one-time physical measurement**, not an experiment to "redo" — it tells the
controller which OptiTrack axes are the ground plane, which way "heading = 0" points for each
robot, and how fast the motors actually go. Without it, the robots will drive in the wrong
direction or at the wrong speed regardless of how good the genome is.

**Physical setup first**: give each of the 3 robots (`thymio-17`, `thymio-18`, `thymio-20`) a
clear, straight, obstacle-free runway a couple of meters long, with no two runways crossing.

Run from the **`thymio_swarm_platform` venv**, not this repo's environment:

```bash
cd /home/lilly/dev/thymio_swarm/thymio_swarm_platform
source .venv/bin/activate
cd examples
python3 hebbian_position_heading_calibration.py
```

This drives each robot forward once (~10s), then prints a per-robot table plus two
ready-to-paste blocks: one `HEADING_OFFSET_RAD = {...}` for `ROTATION_SIGN = 1.0` and one for
`ROTATION_SIGN = -1.0` — **you don't know which one is right yet**, so just copy both
mentally for now and continue to Step 3b.

**If a robot's printed R²/yaw-std looks bad** (the script warns about this): manually put
that robot back at its exact start position and heading, and re-run the same command — do
**not** add more targets to try to average it out (real Thymios don't drive straight, and a
multi-leg sweep can't reset between legs on this platform, so one bad leg would corrupt every
leg after it).

### Step 3a — Save the calibration results

Edit `ants26_replication/hardware_deployment/controller_config.py` on your own machine:
- Set `POSITION_AXES` to the aggregated recommendation printed above.
- Set `MOTOR_UNITS_PER_MPS` to the aggregated recommendation.
- Leave `HEADING_OFFSET_RAD` for a moment — you'll pick the right block in Step 3b.
- Delete or update the `STILL STALE / DO NOT TRUST` comments above these three constants so
  Step 0's check passes cleanly next time.

### Step 3b — Determine `ROTATION_SIGN` and finalize `HEADING_OFFSET_RAD`

A straight-line drive carries no information about turn direction, so this has to be observed
live. Easiest way: start the real trial once with a guess (`ROTATION_SIGN = 1.0`, and the
matching `HEADING_OFFSET_RAD` block from Step 3's output), watch 10 seconds of behavior, and
flip the sign if wrong:

- **Correct**: robots turn *toward* their intended heading.
- **Wrong**: robots spin *away* from it — stop immediately (type `s` in the running
  `hebbian_swarm_trial.py` prompt, see Step 5), flip `ROTATION_SIGN` to the other value,
  paste in the *other* `HEADING_OFFSET_RAD` block from Step 3's printed output, and retry.

Once correct, commit + push (Step 2's commands) before continuing.

### Step 3c — Measure the real corridor bounds

Needed so the built-in wall-safety governor actually knows where your walls are — **without
this, robots will drive straight into a wall without slowing at all** (the genome has no wall
sense of its own; see `README.md`'s "Corridor wall safety" section).

This is a **point-and-sample** tool, run **one robot at a time**, not a continuous tracker —
you place the robot, step clear of the tracked volume (so your own body isn't occluding its
markers — bending over the robot to move it can block line-of-sight to some cameras and not
others, which looks exactly like a frozen/broken reading even though nothing is actually
wrong), then trigger a reading from the controller terminal. No SSH/journalctl needed — the
script collects every sample itself and prints a clean table when you stop it.

```bash
cd /home/lilly/dev/thymio_swarm/thymio_swarm_platform
source .venv/bin/activate
cd examples
python3 hebbian_pose_calibration.py thymio-17
```

Physically place the robot at one wall of the actual usable runway, **step away from the
tracked volume entirely**, then press `p`. Move it to the opposite wall, step clear again,
press `r` (does the same thing as `p` — either works, see the script's own docstring for
why) for sample #2. Type `s` to stop — this prints a table of both samples plus, if it found
two `sim_y` values, a ready-to-use suggestion:

```
=== thymio-17: 2 sample(s), in order ===
 sample trigger  ...  sim_x  sim_y  sim_heading
      1       p  ...  0.021  1.850       -0.041
      2       r  ...  0.034 -1.760       -0.058

If these were your two corridor-wall samples: CORRIDOR_Y_MIN=-1.760  CORRIDOR_Y_MAX=1.850
(add a little headroom inward before using these).
```

Repeat the whole command (`python3 hebbian_pose_calibration.py <hostname>`) for each other
robot, at the same two physical points. Robots touching the same wall report different
`sim_y` (each rigid body's tracked origin is offset differently from the robot's center),
so the wall bounds are **per robot**: a shared pair either lets some robots hit the wall or
stops the others mid-corridor.

Each robot's (min, max) = the smaller/larger of its own sampled `sim_y` values — each with a
little headroom inward, not the exact wall-touching value.

Edit `controller_config.py`:
```python
CORRIDOR_Y_BOUNDS = {
    "thymio-17": (-1.740, 1.830),   # its measured min/max, with a little headroom inward
    "thymio-18": (-1.810, 1.770),
}
```
`CORRIDOR_Y_MIN`/`MAX` stay as the fallback for any robot not listed in `CORRIDOR_Y_BOUNDS`.
Then tune `CORRIDOR_SLOWDOWN_MARGIN_M` if needed (default 0.5m — larger gives more braking
distance at the cost of usable corridor width). Commit + push (Step 2).

---

## Step 4 — Final pre-flight check

```bash
cd /home/lilly/dev/energy_efficient_flocking/energy_efficient_flocking
git status   # clean, up-to-date with origin/main — same check as Step 0
```

Confirm all three of `POSITION_AXES`, `MOTOR_UNITS_PER_MPS`, `HEADING_OFFSET_RAD`,
`ROTATION_SIGN`, `CORRIDOR_Y_MIN`, `CORRIDOR_Y_MAX` in `controller_config.py` now hold real
measured values (not `None` or a stale-flagged placeholder).

---

## Step 5 — Deploy: run the real trial

This is the actual deployment step. Run from the **`thymio_swarm_platform` venv**:

```bash
cd /home/lilly/dev/thymio_swarm/thymio_swarm_platform
source .venv/bin/activate
cd examples
python3 hebbian_swarm_trial.py
```

This installs/updates/activates the project on all 3 Pis (pulling your just-pushed code),
starts all 3 robots simultaneously, and runs for `EXPERIMENT_DURATION_SECONDS` (120s by
default — the script's own comment recommends keeping this short for the first live run).

While it's running, you get an interactive prompt:
```
[p]ause  [r]esume  [s]top >
```
Use `p`/`r` to pause/resume if something looks unsafe (e.g. heading toward a person or
obstacle outside what the corridor/clamp governors were set up for), `s` to stop early. It
also stops automatically when the duration elapses.

**On stop (either way), the script automatically**: stops all robots, collects each robot's
log, deletes the remote copies, and aggregates everything into one file (Step 6).

**Watch for immediately after robots start moving**: OptiTrack takes a moment to lock onto
every rigid body after a session starts. A robot briefly not-tracked is normal for the first
tick or two; if a robot never starts moving at all, or all three stop simultaneously within
the first ~2 seconds with no error, that's the known "battery zeroed by an untracked-robot
position jump" failure mode described in the README's "Known open risks" — check that
robot's OptiTrack rigid body is actually defined and visible in Motive before retrying.

---

## Step 6 — After the trial: find and sanity-check your data

The aggregated log lands at:
```
/home/lilly/dev/thymio_swarm/thymio_swarm_platform/examples/results/hebbian-swarm-3agent-run/processed/aggregated.csv
```

Quick sanity check before trusting it for anything:

```bash
cd /home/lilly/dev/thymio_swarm/thymio_swarm_platform/examples
python3 -c "
import pandas as pd
df = pd.read_csv('results/hebbian-swarm-3agent-run/processed/aggregated.csv')
print(df['hostname'].value_counts())          # should show ~3 robots, similar row counts each
print(df[['tick','timestamp','x','y','heading','battery','v','w']].describe())
print('any untracked rows (x/y == 1e4):', ((df['x'] > 1e3) | (df['y'] > 1e3)).sum())
"
```

Things to check:
- All 3 hostnames present, with a similar number of rows each (a much shorter log for one
  robot suggests it stopped early — check its own log for why).
- `battery` decreasing over time within each `hostname` (if `BATTERY_MODE = "simulated"`,
  which it is for this genome).
- `v` never exceeding `LINEAR_VEL_MAX` (0.2) and `w` never exceeding `ANGULAR_VEL_MAX`
  (`math.pi/5 ≈ 0.628`) by more than floating-point noise.
- Few or no untracked rows — a large count suggests an OptiTrack/Motive tracking problem
  during the run, not a controller issue.

For comparing against the simulation plots (trajectories, battery-vs-time, etc.), see
`README.md`'s "Logged data, and comparing a real trial to the simulation results" section —
it explains exactly which columns map to which simulation quantities and what's directly
comparable vs. approximate.

---

## Rollback: reverting to the previous genome

If something about `plain_seed123_clamped` looks wrong on real hardware and you need to fall
back to the genome that was deployed before it:

```python
# In thymio_swarm_platform/examples/hebbian_swarm_trial.py:
GENOME_PATH_ON_PI = "hebbian_save_battery_avoid_all_best.npy"  # the old genome, still
                                                                 # present in hardware_deployment/
```
No other changes needed — the old genome doesn't use `BATTERY_MODE`/clamp settings any
differently. Re-run Step 5.

---

## Troubleshooting quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| Robot drives in a completely wrong direction | `HEADING_OFFSET_RAD`/`POSITION_AXES` wrong or missing for that hostname | Re-check Step 3a/3b; look for a one-time `[pose_utils] WARNING: no HEADING_OFFSET_RAD entry` printed on that Pi |
| Robot spins the wrong way when turning | `ROTATION_SIGN` wrong | Flip it (Step 3b), commit+push, retry |
| Robot drives straight at a wall without slowing | `CORRIDOR_Y_MIN`/`MAX` still `None`, or wrong axis | Redo Step 3c |
| Robots stop almost immediately, battery reads ~0 | Untracked-robot position-jump bug (see Step 5's note) | Verify OptiTrack tracking before retrying |
| Two robots collide despite the safety clamp | Check `AGENT_SAFETY_CLAMP_OUTER_GAP`/`INNER_GAP` weren't edited — they must match training exactly, see the comment above them in `controller_config.py` | Don't tune these without retraining |
| `ModuleNotFoundError` running any `hebbian_*.py` launcher | Ran it from the wrong environment | Must be the `thymio_swarm_platform/.venv`, not this repo's own Python |
| Pis seem to run old code after you pushed | Forgot Step 2, or pushed to the wrong branch | `git log` on `main` should show your commit; `project.update()` inside each launcher pulls it automatically on next run |
