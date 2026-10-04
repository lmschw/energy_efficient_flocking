# Lighthouse fleet (state of 2026-10-04)

One Crazyflie board + Thymio + Raspberry Pi per robot. `--hostnames` / `--ids` for `run_hebbian.py`
must be identical on every Pi (order matters):

```
--hostnames robot-1,robot-2,robot-3,robot-4,robot-5,robot-6,robot-7
--ids       1,233,235,232,236,226,234
```

| Robot | Crazyflie radio URI | id (= address low byte) | Pi (user `tugay`) | Verified |
|---|---|---|---|---|
| robot-1 | `radio://0/100/2M/E7E7E7E701` | 1   | 10.15.2.83  | `--check` OK |
| robot-2 | `radio://0/100/2M/E7E7E7E7E9` | 233 | 10.15.2.81  | `--check` OK |
| robot-3 | `radio://0/100/2M/E7E7E7E7EB` | 235 | 10.15.3.13  | `--check` OK |
| robot-4 | `radio://0/100/2M/E7E7E7E7E8` | 232 | 10.15.2.197 | `--check` OK |
| robot-5 | `radio://0/100/2M/E7E7E7E7EC` | 236 | 10.15.2.25  | `--check` OK |
| robot-6 | `radio://0/100/2M/E7E7E7E7E2` | 226 | 10.15.2.70  | `--check` OK |
| robot-7 | `radio://0/100/2M/E7E7E7E7EA` | 234 | 10.15.2.250 | `--check` OK |

IPs are DHCP leases and can change. All boards: channel 100, 2M, Hebbian firmware (autodetect build),
Lighthouse bitstream V6, geometry from `lighthouse_config/lighthouse_system.yaml` (origin and +x as
placed during calibration; robot 3 on the minus-x side of the origin read x = -0.994).

Tools: `tools/setup_pi.sh <ip> <robot-name>` (copy code, set clock, start Thymio Device Manager if
needed, offline test, `--check`; needs the key `~/.ssh/id_ed25519_pis` installed on the Pi);
`tools/identify_robot_leds.py <radio uri>` holds a board's LEDs on (stop it with SIGTERM, then
power-cycle that Crazyflie before using its USB link from the Pi). See README.md "Fleet bring-up notes".

## Arena (measured 2026-10-04)
Robot-centre y at the walls: right wall +1.699, left wall -2.239 (about 4.05 m wall to wall; the middle is
y = -0.27, not 0). x at the ends (robot centre): +1.399 and -1.838 (about 3.35 m end to end), no physical walls; x governor added. Corridor used for
runs: `--corridor-y -2.04 1.50` (0.2 m inset). Heading offsets / motor constant: lighthouse_config/
heading_calibration.csv, applied in controller_config.py.

Run: `CORRIDOR_X="<xmin> <xmax>" tools/run_swarm.sh "3 4" 60 lj` (robots, seconds, controller lj|hebbian).
Stop: `tools/stop_all.sh`. The LJ baseline pulls the swarm toward -x at ~0.15 m/s (constant goal force), so set
CORRIDOR_X before using it. IR backoff is disabled (IR_BACKOFF_ENABLED = False in controller_config.py);
the x governor is new (CORRIDOR_X_MIN/MAX, --corridor-x). Robot 2's board was re-mounted and re-calibrated (offset now in controller_config.py).

## Experiments (all 7 robots)
`tools/experiment.sh <lj|hebbian> <reps> <duration_s> [genome]` -- per repetition: you place the robots (by eye, see the printed instructions; `tools/layout_check.py` + `layout_7.txt` still exist as an optional check),
`tools/run_swarm.sh` runs all 7 with a common start
time and collects everything into `hardware_runs/<condition>_rep<k>_<stamp>/` together with `run_info.json` and
snapshots of controller_config.py, the Lighthouse geometry and the heading calibration. Emergency stop: Ctrl-C or
`tools/stop_all.sh`. Conditions of the paper: `lj`, `hebbian plain_seed123_best.npy` (unclamped),
`hebbian plain_seed123_clamped_best.npy`. Baseline LJ pair test (robots 3+4, 2026-10-04): both travelled
x +1.27 -> -1.64 (path ~3.4 m, mean spacing 0.72 m, closest 0.47 m).

Timing from the 2-robot tests (2026-10-04): LJ reaches the -x limit in ~25 s, clamped genome ~35 s, unclamped genome
had not arrived after 60 s -> common duration for all conditions: 60 s (compare time-to-end and distance at fixed times).
