"""Generates hardware_transfer_test/upwind_2stage_plain_seed888_clamped/n<N>/{video.mp4,
battery_plot.png,metrics.json} for the one clamped-retrain survivor worth deploying as-is
(73% both-beat-baseline at n=20, see project_upwind_safety_variant.md's 2026-09-16 update) --
unlike generate_upwind_2stage_conditions.py's three genomes, this one is evaluated WITH the
safety clamp active (HEBBIAN_SAFETY_CLAMP_ENABLED=True, MIN_DIST_INFLATION=1.3,
RESOLVE_COLLISIONS=True strength=0.5 max_iter=2), matching how it was both trained and
30-seed-evaluated -- it was retrained from scratch specifically to work correctly with the
clamp on, unlike the other three genomes in this directory.

Usage: python generate_plain_seed888_clamped.py
"""
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENT_DIR = os.path.join(REPO_ROOT, "ants26_replication", "experiment")
sys.path.insert(0, EXPERIMENT_DIR)

import config  # noqa: E402
from hebbian_controller import unflatten_abcd  # noqa: E402
from simulation_hebbian import render_hebbian_episode_video  # noqa: E402
from battery_plot import plot_battery_levels  # noqa: E402
from leadership_metrics import _ConfigOverride, SEED, WIND_GRID  # noqa: E402

N_AGENTS_SWEEP = (1, 2, 3, 4, 5, 7, 10, 20)
LABEL = "upwind_2stage_plain_seed888_clamped"

WITH_CLAMP = dict(
    HEBBIAN_SAFETY_CLAMP_ENABLED=True,
    HEBBIAN_MIN_DIST_INFLATION=1.3,
    HEBBIAN_RESOLVE_COLLISIONS=True,
    HEBBIAN_RESOLVE_COLLISIONS_STRENGTH=0.5,
    HEBBIAN_RESOLVE_COLLISIONS_MAX_ITER=2,
)


def main():
    genome_path = os.path.join(SCRIPT_DIR, LABEL, "genome_trained_n20.npy")
    genome = np.load(genome_path)
    rules = unflatten_abcd(genome)

    summary_path = os.path.join(SCRIPT_DIR, "summary.json")
    full_summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    summary = {}

    for n_agents in N_AGENTS_SWEEP:
        out_dir = os.path.join(SCRIPT_DIR, LABEL, f"n{n_agents}")
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, "video.mp4")

        with _ConfigOverride(**WITH_CLAMP):
            dist, batt, ct, wct, tele = render_hebbian_episode_video(
                rules, seed=SEED, n_agents=n_agents, wind_enabled=True,
                nx=WIND_GRID, ny=WIND_GRID, video_path=video_path, record_battery=True)

        n_steps = tele["battery"].shape[0] - 1
        plot_battery_levels(tele["battery"], config.DT, os.path.join(out_dir, "battery_plot.png"),
                             title=f"Battery Level per Agent (n_agents={n_agents})")

        metrics = {"n_agents": n_agents, "dist": dist, "batt_pct": batt,
                   "collision_time": ct, "wall_collision_time": wct, "n_steps": n_steps}
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        summary[str(n_agents)] = metrics
        print(f"  {LABEL} n={n_agents}: dist={dist:.2f} batt={batt:.2f}% "
              f"ct={ct:.1f}s wct={wct:.1f}s n_steps={n_steps}")

    full_summary[LABEL] = summary
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"\nUpdated {summary_path}")


if __name__ == "__main__":
    main()
