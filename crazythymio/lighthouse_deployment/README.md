# Hebbian controller on Thymio + Crazyflie board + Lighthouse positioning

> **This folder is fully separate from the OptiTrack deployment**
> (`ants26_replication/hardware_deployment/`). It has its own `controller_config.py`,
> `pose_utils.py`, `hebbian_swarm_experiment.py`, sensing/controller/battery modules and genome
> copies, and imports nothing from that folder (and vice versa; the OptiTrack folder is
> unchanged from before the Lighthouse work). Never put both folders on the same `sys.path` --
> they both define a module called `controller_config`. `test_offline.py` asserts the
> isolation.

Replaces the OptiTrack/`thymio_swarm_platform` stack with the CrazyThymio one (code from
https://github.com/fudavd/CrazyThymio): every robot is a **Thymio + Raspberry Pi + Crazyflie
board with a Lighthouse deck**. There is no central tracker or coordinator any more:

```
Lighthouse base stations --> Crazyflie board (own x, y, yaw)
                                  |  radio P2P broadcast: every board sends its position,
                                  |  every board keeps a table of the others (custom firmware)
                                  v  USB
                           Raspberry Pi  --run_hebbian.py--> HebbianSwarmExperiment (unchanged)
                                  |  USB (tdmclient)
                                  v
                               Thymio motors / IR
```

The controller math is a copy of the OptiTrack deployment's (same genome format, same sensing, same safety layers); only the tracking layer differs. Files in this folder:

| File | Purpose |
|---|---|
| `../firmware/app_share_pos_hebbian/` | Crazyflie app: Lighthouse, broadcasts own *center* position, exposes neighbor positions to the Pi. **Replaces** upstream `app_share_pos` (that one only exposes pre-computed quadrant distances, not the bearings the Hebbian net needs). |
| `controller_config.py` | **This deployment's own config**: architecture/battery/safety constants copied from the OptiTrack config, plus the Lighthouse hardware section (heading offsets, corridor, motor calibration, pose timeout). No OptiTrack axes/offsets. |
| `pose_utils.py` | Lighthouse pose -> agents array (z-up, yaw-based heading; no OptiTrack outlier/stale detectors). |
| `hebbian_swarm_experiment.py`, `sensor_model.py`, `hebbian_controller.py`, `motor_utils.py`, `wind_battery_model.py` | Copies of the OptiTrack deployment's modules; edit here without touching the OptiTrack folder. |
| `plain_seed123_clamped_best.npy`, `plain_seed123_best.npy` | Genome copies. |
| `lighthouse_robot.py` | `LighthouseRobot`: same interface as the platform's `Robot` (`get_all_global_poses`, `drive`, `stop`, `proximity_horizontal`), backed by cflib + tdmclient. |
| `run_hebbian.py` | Per-robot launcher (+ `--check`, `--calibrate-heading`). |
| `test_offline.py` | Hardware-free wiring + isolation test. |

**Status: written and checked offline only** (`venv/bin/python crazythymio/lighthouse_deployment/test_offline.py`
passes; the firmware passes a syntax check against stub headers but has *not* been compiled
or flashed; nothing has run on real hardware). Expect to debug on first contact.

## Frames and conventions (important)

* Lighthouse: z up, yaw in degrees CCW from +x. Simulation: heading 0 faces **+y**, CCW positive.
  `pose_utils` converts: `heading = yaw - pi/2 + LIGHTHOUSE_HEADING_OFFSET_RAD[host]`.
  Positions are used as they are, so **+x/+y of the Lighthouse frame are the simulation's +x/+y**.
* `--origin X Y` shifts the Lighthouse frame so the arena center is the simulation's (0, 0).
  The battery/wind model and the wall governor are defined in the simulation frame, so the
  arena's +y must correspond to the same physical direction as in your earlier runs. I did
  not check which direction the simulated wind blows in `wind_battery_model.py` -- do that
  before deciding how to orient the geometry.
* The board is mounted off-center on the Thymio. `BOARD_OFFSET_X/Y` at the top of the
  firmware file (default: upstream's -0.09/+0.04 m, board frame x forward / y left) moves
  every broadcast position to the robot's center. **Measure it on your robots.**

## One-time setup

### 1. Base stations + deck
1. Mount 2+ Lighthouse base stations (V2 recommended) high up, covering the whole arena with
   line of sight to the decks. Give each a unique channel (V2: 1, 2, ...).
2. Lighthouse deck on each Crazyflie board, facing up, with no occluders above it.
3. Flash the stock Crazyflie firmware first if the board is on an old version; update with
   `cfclient` (Bootloader). Lighthouse needs a reasonably recent release.

### 2. Geometry (once per arena, same file on all boards)
In `cfclient` (`pip install cfclient`), connect to one board -> *Lighthouse Positioning* tab:
1. *Manage Geometry* -> *Start* -> follow the wizard to place origin, +x direction, +y and the
   scale reference (put the robot at the floor positions it asks for). Put the origin at the
   arena center if you like -- then `--origin 0 0` and you can skip the shift.
2. Save the geometry/calibration to a file, then *Write to Crazyflie* on **every** board, so
   every robot shares the same frame. Check the tab shows both base stations "OK".

### 3. Radio ids (once per board)
Every board needs a **unique radio id 1..10** (lowest byte of its radio address) and **all
boards the same channel and datarate** (P2P only works on a shared channel). In `cfclient`:
*Connect -> Configure 2.x*: e.g. address `0xE7E7E7E701` for robot 1, `...02` for robot 2.
Keep a table `hostname -> id`; `--hostnames`/`--ids` below are exactly that table.

### 4. Firmware
```bash
git clone --branch CrazyThymio https://github.com/tugayalperen/CrazyThymio-firmware.git
cd CrazyThymio-firmware && git submodule update --init --recursive
cp -r <repo>/crazythymio/firmware/app_share_pos_hebbian examples/
cd examples/app_share_pos_hebbian
# edit BOARD_OFFSET_X/Y in src/share_pos_hebbian.c first (see above)
make clean && make -j        # needs the Crazyflie build toolchain: arm-none-eabi-gcc, make
# board in bootloader mode (hold power button 3 s, blue LED blinks), Crazyradio plugged in:
make cload
```
`app-config` selects `CONFIG_DECK_LIGHTHOUSE=y` (upstream used the UWB Loco deck). If the
build fails on the app stack, raise `CONFIG_APP_STACKSIZE`.

### 5. Raspberry Pi (each robot)
Follow `crazythymio/real_exp/initial_config.sh` (Thymio Device Manager via flatpak, udev
rules for Thymio + Bitcraze, venv). It also starts `thymio-device-manager` at boot -- the
launcher needs it running. Then:
```bash
cd ~/Desktop/crazy_thymio && git clone <this repo> energy_efficient_flocking
source .venv/bin/activate
pip install numpy cflib tdmclient
```
(the Pi only needs `crazythymio/lighthouse_deployment/` + `crazythymio/firmware/`; no scipy,
matplotlib etc.). Plug the Crazyflie board into the Pi by USB (the `usb://0` uri).

## Verification, in this order

Run from `energy_efficient_flocking/crazythymio/lighthouse_deployment/`; replace hostnames/ids with yours. Use the
same `--hostnames/--ids` on every Pi.

```bash
H="--hostnames thymio-01,thymio-02,thymio-03 --ids 1,2,3"
```

**A. Offline plumbing (any machine):** `python test_offline.py`

**B. Tracking + radio table.** Robots standing still, all powered, motors off:
```bash
python run_hebbian.py --self-hostname thymio-01 $H --check
```
Prints own `(x, y, z, yaw)`, the neighbor table (radio id -> position) and IR values for 20 s.
Check: (i) own position changes when you carry the robot, (ii) every *other* robot shows up in
the neighbors, and their positions agree with what each of them prints as its own,
(iii) if the neighbor list stays empty: channel/datarate/ids, or firmware not flashed.
(iv) Place a robot at a known spot (e.g. arena center) and read off its position -> `--origin`.

**C. Heading offset + speed calibration, one robot at a time** (0.5 m clear path ahead):
```bash
python run_hebbian.py --self-hostname thymio-01 $H --calibrate-heading
```
Drives straight 4 s and prints `--heading-offsets thymio-01=<rad>` (board x-axis vs. Thymio
front; ~0 or ~±pi/2 depending on mounting) and a suggested `MOTOR_UNITS_PER_MPS`. Collect
one offset per robot, `--heading-offsets thymio-01=0.03,thymio-02=-0.10,...`. If the
suggested motor constant differs a lot from `controller_config.MOTOR_UNITS_PER_MPS` (3553),
pass `--motor-units-per-mps`. Re-run with the offsets once: it should now print ~0.

**D. Arena walls.** Carry a robot to the arena edges in the `--origin`-shifted frame and note
y at each wall (`--check`), then pass `--corridor-y MIN MAX` with ~0.2 m inside margin.
Without it the wall governor is off and the genome has no wall sense -- robots will drive
into walls.

**E. Short dry-run with 2 robots**, held in hand/lifted first (wheels free): `--duration 30`,
watch the printed motor commands respond to moving the other robot.

## Experiment run

On every Pi (same arguments except `--self-hostname`):
```bash
python run_hebbian.py --self-hostname thymio-01 $H \
   --genome plain_seed123_clamped_best.npy \
   --origin 2.0 1.5 --corridor-y -1.5 1.5 \
   --heading-offsets thymio-01=0.03,thymio-02=-0.10,thymio-03=0.00 \
   --duration 600
```
* Start together: either press Enter on all Pis at once (e.g. `tmux` with synchronized
  panes), or give all the same `--start-at <unix time>` (Pis' clocks must agree --
  `initial_config.sh` sets them, or use NTP).
* `Ctrl-C` stops motors cleanly. `--duration` stops after N seconds.
* Log: `logs/<hostname>_<timestamp>.csv` per robot, same columns as before (`x, y, heading,
  battery, front_d, ..., v, w, left, right`; `raw_x/y/z` and `q*` are now the Lighthouse
  position/yaw quaternion). The existing `hardware_transfer_test` analysis expects the
  platform's aggregated format -- merge the per-robot CSVs by `timestamp` if you reuse it.
* `BATTERY_MODE` is set in this folder's `controller_config.py` (`simulated` by default; override with
  `--battery-mode none` together with a `_nosensor` genome).
* Safety layers that carry over unchanged: agent-safety speed clamp, wall governor (if
  `--corridor-y`), IR backoff (`IR_OBSTACLE_THRESHOLD` is still an uncalibrated
  placeholder), obstacle backoff, and the slow crawl when the own pose is lost.
  If a *neighbor* drops out (no radio for 1 s) it simply disappears from sensing.

## Known risks / things I could not verify
* Firmware not built or flashed; no run on hardware; the cflib/tdmclient calls follow the
  upstream scripts and the platform's `ThymioConnection`, not tested against live devices.
* Lighthouse on a ground robot: the stock Kalman estimator is tuned for flight. If z or
  position jumps, check the Lighthouse tab's status and base-station line of sight (the
  Thymio's body/neighbors can occlude the deck).
* `MAX_PLAUSIBLE_SPEED_MPS`, stale/outlier thresholds etc. were tuned for OptiTrack noise;
  the OptiTrack-only stale/up-axis detectors are skipped in Lighthouse mode.
* Position latency: board (50 ms) + radio + 100 ms log period + 0.5 s control tick;
  neighbors are therefore up to ~0.2 s old.
* `MOTOR_UNITS_PER_MPS` was carried over from the OptiTrack config (stale there too);
  recalibrate as in step C. Heading offsets and corridor start empty/disabled.
