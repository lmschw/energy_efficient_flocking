"""Generates hardware_transfer_test/<condition>/n<N>/{video.mp4,battery_plot.png,metrics.json}
for the four new walk_upwind/walk_left x clamp/no-clamp conditions, matching the exact layout
and conventions of the original three (lj_baseline/pre_clamp_best/safety_clamp_best).

Uses ants26_replication/experiment/'s package uniformly for ALL FOUR genomes, even the two
trained under upwind_safety_variant/ -- the genome format (880 ABCD floats), network forward
pass, Hebbian update rule, and sensor model are identical across variants, and
upwind_safety_variant/'s WIND_DIRECTION=(-1,0) dot-product distance formula is numerically
identical to experiment/'s bare "-x" (see that variant's config.py docstring) -- so evaluating
with experiment/'s render_hebbian_episode_video is equivalent and avoids duplicating video-
rendering machinery. Evaluated with wind ENABLED regardless of origin (both variants train
their final stage with wind on, matching hardware_transfer_test's own convention).

Usage: python generate_new_conditions.py
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

N_AGENTS_SWEEP = (2, 3, 4, 5, 10, 20)

# label -> (safety_clamp, min_dist_inflation, resolve_collisions[, resolve_strength, resolve_max_iter])
CONDITIONS = {
    "upwind_clamp": dict(safety_clamp=True, min_dist_inflation=1.3, resolve_collisions=True,
                          resolve_strength=0.5, resolve_max_iter=2),
    "upwind_no_clamp": dict(safety_clamp=False, min_dist_inflation=1.0, resolve_collisions=False),
    "walk_left_n10_clamp": dict(safety_clamp=True, min_dist_inflation=1.3, resolve_collisions=True,
                                 resolve_strength=0.5, resolve_max_iter=2),
    "walk_left_n10_no_clamp": dict(safety_clamp=False, min_dist_inflation=1.0, resolve_collisions=False),
}


def run_condition(label, overrides):
    genome_path = os.path.join(SCRIPT_DIR, label, "genome_trained_n10.npy")
    genome = np.load(genome_path)
    rules = unflatten_abcd(genome)

    cfg_patch = dict(
        HEBBIAN_SAFETY_CLAMP_ENABLED=overrides["safety_clamp"],
        HEBBIAN_MIN_DIST_INFLATION=overrides["min_dist_inflation"],
        HEBBIAN_RESOLVE_COLLISIONS=overrides["resolve_collisions"],
    )
    if "resolve_strength" in overrides:
        cfg_patch["HEBBIAN_RESOLVE_COLLISIONS_STRENGTH"] = overrides["resolve_strength"]
        cfg_patch["HEBBIAN_RESOLVE_COLLISIONS_MAX_ITER"] = overrides["resolve_max_iter"]

    summary = {}
    for n_agents in N_AGENTS_SWEEP:
        out_dir = os.path.join(SCRIPT_DIR, label, f"n{n_agents}")
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, "video.mp4")

        with _ConfigOverride(**cfg_patch):
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
        print(f"  {label} n={n_agents}: dist={dist:.2f} batt={batt:.2f}% "
              f"ct={ct:.1f}s wct={wct:.1f}s n_steps={n_steps}")

    return summary


def main():
    summary_path = os.path.join(SCRIPT_DIR, "summary.json")
    full_summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    for label, overrides in CONDITIONS.items():
        genome_path = os.path.join(SCRIPT_DIR, label, "genome_trained_n10.npy")
        if not os.path.exists(genome_path):
            print(f"\n=== {label} === SKIPPED (no genome yet at {genome_path})")
            continue
        print(f"\n=== {label} ===")
        full_summary[label] = run_condition(label, overrides)

    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"\nUpdated {summary_path}")


if __name__ == "__main__":
    main()
