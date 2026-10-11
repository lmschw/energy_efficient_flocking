"""Records full simulation episodes (until the first robot's battery is empty) for the promo
videos: the Stage 2 controller (plain_seed123, unclamped, as in the paper's no-clamp
version) and the baseline, each with 7 and 20 robots.

The simulator itself is not modified: its module-level _spawn_agents, RayTraceCircularRobots
and batterydrainage are wrapped for the duration of one run to copy out the full robot state
(x, y, heading, battery) and the wind field every step. Settings match
paper_results_scatter_no_clamp.py / n_agents_sweep_analysis_no_clamp.py.

Usage: python record_runs.py [--seed 42]
Writes runs/<condition>_n<N>_seed<S>.npz
"""
import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(FINAL_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "ants26_replication", "experiment"))
sys.path.insert(0, FINAL_DIR)

import config  # noqa: E402
import lj_baseline  # noqa: E402
import simulation_hebbian  # noqa: E402
from hebbian_controller import unflatten_abcd  # noqa: E402
from leadership_metrics import _ConfigOverride, SEED, WIND_GRID  # noqa: E402

NO_CLAMP = dict(HEBBIAN_SAFETY_CLAMP_ENABLED=False, HEBBIAN_MIN_DIST_INFLATION=1.0,
                HEBBIAN_RESOLVE_COLLISIONS=False)
GENOME_PATH = os.path.join(FINAL_DIR, "plain_seed123", "genome_trained_n20.npy")
BASELINE_RULES_PATH = os.path.join(FINAL_DIR, "lj_baseline", "paper_baseline_rules.json")
OUT_DIR = os.path.join(SCRIPT_DIR, "runs")


class _Recorder:
    """Wraps a simulator module's spawn/wind/battery functions to log every step."""

    def __init__(self, module):
        self.module = module
        self.states = []
        self.winds = []

    def __enter__(self):
        m = self.module
        self._orig = (m._spawn_agents, m.RayTraceCircularRobots, m.batterydrainage)
        spawn, raytrace, drain = self._orig

        def _spawn(*a, **k):
            agents = spawn(*a, **k)
            self.states.append(agents.copy())
            return agents

        def _raytrace(*a, **k):
            out = raytrace(*a, **k)
            self.winds.append(out)
            return out

        def _drain(*a, **k):
            agents, d = drain(*a, **k)
            self.states.append(agents.copy())
            return agents, d

        m._spawn_agents, m.RayTraceCircularRobots, m.batterydrainage = _spawn, _raytrace, _drain
        return self

    def __exit__(self, *exc):
        m = self.module
        m._spawn_agents, m.RayTraceCircularRobots, m.batterydrainage = self._orig


def record(condition, n_agents, seed):
    if condition == "baseline":
        rules = json.load(open(BASELINE_RULES_PATH))
        with _ConfigOverride(HEBBIAN_NX=WIND_GRID, HEBBIAN_NY=WIND_GRID), _Recorder(lj_baseline) as rec:
            _, dist, batt, _ = lj_baseline.simulate_lj_baseline(rules=rules, seed=seed, n_agents=n_agents)
        battery_scale = config.MAX_BATTERY
    else:
        rules = unflatten_abcd(np.load(GENOME_PATH))
        with _ConfigOverride(**NO_CLAMP), _Recorder(simulation_hebbian) as rec:
            dist, batt, *_ = simulation_hebbian.simulate_hebbian_episode(
                rules, seed=seed, n_agents=n_agents, wind_enabled=True, nx=WIND_GRID, ny=WIND_GRID)
        battery_scale = config.HEBBIAN_MAX_BATTERY

    # Wind field at t=0 (the simulators only compute it after the first move).
    rec.winds.insert(0, simulation_hebbian.RayTraceCircularRobots(
        rec.states[0], config.WIND_RAD, config.UINF, list(config.X_RANGE), list(config.Y_RANGE),
        WIND_GRID, WIND_GRID))

    states = np.array(rec.states)                       # (T+1, n, 4): x, y, theta, battery
    assert len(rec.winds) == len(states)
    y_vals = np.array(rec.winds[0][0])
    x_vals = np.array([w[1] for w in rec.winds])        # (T+1, Nx) -- moving window
    wind = np.array([w[2] for w in rec.winds])          # (T+1, Nx, Ny), percent of free stream
    path = os.path.join(OUT_DIR, f"{condition}_n{n_agents}_seed{seed}.npz")
    np.savez_compressed(path, states=states, x_vals=x_vals, y_vals=y_vals, wind=wind,
                        dt=config.DT, battery_scale=battery_scale, dist=dist,
                        batt_pct=batt / battery_scale * 100.0, walls_y=np.array(config.Y_RANGE))
    print(f"{condition:9s} n={n_agents:2d} seed={seed}: {len(states) - 1} steps "
          f"({(len(states) - 1) * config.DT:.0f} s), dist={dist:.2f} m, "
          f"batt={batt / battery_scale * 100:.1f}%  -> {os.path.relpath(path, SCRIPT_DIR)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, nargs="+", default=[7, 20])
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for n in args.n:
        for condition in ("baseline", "evolved"):
            record(condition, n, args.seed)


if __name__ == "__main__":
    main()
