"""Simulates plain_seed123 (unclamped), plain_seed123_clamped, and the LJ baseline in an
arena narrowed to the SAME width as the real hardware corridor -- 3.5 m along y (the axis the
safety clamp and wall-collision counter act on), vs. the standard 10 m x 10 m training arena
used for every other simulation result in this directory. X stays at the default (effectively
unbounded) range, matching how the wall clamp/collision counter only ever acted along y in
training (see swarm_intelligence_revised.tex, sec:safety) -- only the width the user asked
about is narrowed here.

Same evaluation recipe as generate_plain_seed123_clamped.py / leadership_metrics.py: seed=42
(config.HEBBIAN_DEFAULT_SEED, the standard single-episode evaluation seed throughout this
project -- distinct from the seed=123 used to train this genome), wind grid 50x50, one episode
per (condition, n_agents) cell, single variable (arena width) changed vs. the existing
n5/n10/n20 metrics.json files in plain_seed123/, plain_seed123_clamped/, lj_baseline/.

Usage: python arena_3p5m_sweep.py
"""
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # .../hardware_transfer_test/final -> repo root
EXPERIMENT_DIR = os.path.join(REPO_ROOT, "ants26_replication", "experiment")
sys.path.insert(0, EXPERIMENT_DIR)

import config  # noqa: E402
from hebbian_controller import unflatten_abcd  # noqa: E402
from simulation_hebbian import simulate_hebbian_episode  # noqa: E402
from lj_baseline import simulate_lj_baseline  # noqa: E402
from leadership_metrics import _ConfigOverride, SEED, WIND_GRID, _CONDITION_OVERRIDES  # noqa: E402

N_AGENTS_SWEEP = (5, 10, 15)
ARENA_Y_WIDTH = 3.5
Y_RANGE_3P5M = [-ARENA_Y_WIDTH / 2.0, ARENA_Y_WIDTH / 2.0]  # [-1.75, 1.75]

OUT_DIR = os.path.join(SCRIPT_DIR, "arena_3p5m_sweep")
LJ_RULES_PATH = os.path.join(SCRIPT_DIR, "lj_baseline", "paper_baseline_rules.json")
GENOME_PATH = {
    "plain_seed123": os.path.join(SCRIPT_DIR, "plain_seed123", "genome_trained_n20.npy"),
    "plain_seed123_clamped": os.path.join(SCRIPT_DIR, "plain_seed123_clamped", "genome_trained_n20.npy"),
}


def run_hebbian(condition, n_agents):
    overrides = _CONDITION_OVERRIDES[condition]
    genome = np.load(GENOME_PATH[condition])
    rules = unflatten_abcd(genome)
    cfg_patch = dict(
        HEBBIAN_SAFETY_CLAMP_ENABLED=overrides["safety_clamp"],
        HEBBIAN_MIN_DIST_INFLATION=overrides["min_dist_inflation"],
        HEBBIAN_RESOLVE_COLLISIONS=overrides["resolve_collisions"],
        Y_RANGE=Y_RANGE_3P5M,
    )
    if "resolve_strength" in overrides:
        cfg_patch["HEBBIAN_RESOLVE_COLLISIONS_STRENGTH"] = overrides["resolve_strength"]
        cfg_patch["HEBBIAN_RESOLVE_COLLISIONS_MAX_ITER"] = overrides["resolve_max_iter"]
    with _ConfigOverride(**cfg_patch):
        dist, batt, ct, wct, _, _, tele = simulate_hebbian_episode(
            rules, seed=SEED, n_agents=n_agents, wind_enabled=True,
            nx=WIND_GRID, ny=WIND_GRID, record_battery=True)
    n_steps = tele["battery"].shape[0] - 1
    return {"n_agents": n_agents, "dist": float(dist), "batt_pct": float(batt),
            "collision_time": float(ct), "wall_collision_time": float(wct), "n_steps": int(n_steps)}


def run_lj(n_agents):
    rules = json.load(open(LJ_RULES_PATH))
    with _ConfigOverride(HEBBIAN_NX=WIND_GRID, HEBBIAN_NY=WIND_GRID, Y_RANGE=Y_RANGE_3P5M):
        _, dist, batt, ct = simulate_lj_baseline(rules=rules, seed=SEED, n_agents=n_agents)
    return {"n_agents": n_agents, "dist": float(dist),
            "batt_pct": float(batt / config.MAX_BATTERY * 100.0),
            "collision_time": float(ct), "wall_collision_time": None, "n_steps": None}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for condition in ("plain_seed123", "plain_seed123_clamped", "lj_baseline"):
        results[condition] = {}
        for n_agents in N_AGENTS_SWEEP:
            if condition == "lj_baseline":
                m = run_lj(n_agents)
            else:
                m = run_hebbian(condition, n_agents)
            results[condition][str(n_agents)] = m
            cond_dir = os.path.join(OUT_DIR, condition, f"n{n_agents}")
            os.makedirs(cond_dir, exist_ok=True)
            with open(os.path.join(cond_dir, "metrics.json"), "w") as f:
                json.dump(m, f, indent=2)
            wct_str = f"{m['wall_collision_time']:.1f}s" if m["wall_collision_time"] is not None else "n/a"
            print(f"{condition:24s} n={n_agents:2d}: dist={m['dist']:6.2f}m batt={m['batt_pct']:6.2f}% "
                  f"ct={m['collision_time']:6.1f}s wct={wct_str}")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"arena_y_range": Y_RANGE_3P5M, "seed": SEED, "wind_grid": WIND_GRID,
                    "results": results}, f, indent=2)
    print(f"\nSaved {os.path.join(OUT_DIR, 'summary.json')}")


if __name__ == "__main__":
    main()
