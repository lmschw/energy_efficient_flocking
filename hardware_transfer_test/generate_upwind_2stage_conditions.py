"""Generates hardware_transfer_test/<condition>/n<N>/{video.mp4,battery_plot.png,metrics.json}
for the three n=20-trained 2-stage-upwind winners identified during the "must beat LJ baseline"
investigation (see memory: project_upwind_safety_variant.md, LATEST STATUS section):

  - upwind_2stage_plain_seed123 / upwind_2stage_plain_seed888: trained with the plain (no
    drain-holiday) 2-stage walk_upwind -> save_battery_avoid_all curriculum at n_agents=20.
  - upwind_2stage_drain0_seed123: same curriculum, but trained under the 0%-drain-on-collision
    ("drain holiday") mechanic. Evaluated here under STANDARD (real, non-refunded) drain
    physics regardless -- real hardware does not get free energy for colliding, so this is the
    number that matters for deployment, and it was already confirmed (30-seed check) that this
    genome holds up under standard physics (29/30 both-beat-baseline).

All three use genome_trained_n20.npy (not n10) and are evaluated with clamp OFF / no collision
resolution (NO_CLAMP-equivalent), matching how they were trained (HEBBIAN_SAFETY_CLAMP_ENABLED=
False, HEBBIAN_MIN_DIST_INFLATION=1.0, HEBBIAN_RESOLVE_COLLISIONS=False in upwind_safety_variant/).
Uses experiment/'s render_hebbian_episode_video uniformly, same rationale as
generate_new_conditions.py (genome format / forward pass / sensor model are identical across
variants; upwind_safety_variant's WIND_DIRECTION=(-1,0) dot-product is numerically identical to
experiment/'s bare "-x").

n_agents sweep is {1,2,3,4,5,7,10,20} -- extended beyond this project's usual {2,3,4,5,10,20}
at the user's request to see how these genomes generalize down to a single agent and n=7.
n=1 and n=7 are new territory (no prior condition in this project used them); both were
sanity-checked beforehand for crashes in the n_agents>1-guarded pairwise-distance/collision
code paths (simulation_hebbian.py's _move/_apply_safety_clamp) and run cleanly.

TIGHTER SPAWN FOR VISUALIZATION: config.SPAWN_SQUARE_SIZE defaults to 3.0m, while the sensing
radius (config.HEBBIAN_SENSING_RADIUS) is only 2.01m -- at n=2 especially, this let agents
spawn ~29% of the time (measured empirically) entirely outside each other's sensing range,
producing videos where they never react to each other at all (seed 42, n=2 was exactly such a
draw: dist collapsed to 1.67m vs. 8-10m+ once spawn is tightened). VIDEO_SPAWN_SQUARE_SIZE=1.0
below overrides this ONLY for video/plot generation (max possible pairwise distance in a 1.0m
square is 1.41m, always under the 2.01m sensing radius, guaranteed regardless of n up to at
least 20 -- verified n=20 spawn still converges in <0.4s). This does NOT change
config.SPAWN_SQUARE_SIZE globally and does NOT affect the already-reported 30-seed statistical
comparisons in overleaf_summary/ (those keep the default 3.0m spawn, matching the paper's own
protocol) -- it only makes these illustrative single-seed videos actually show interaction.

Usage: python generate_upwind_2stage_conditions.py
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
VIDEO_SPAWN_SQUARE_SIZE = 1.0

NO_CLAMP = dict(safety_clamp=False, min_dist_inflation=1.0, resolve_collisions=False)

CONDITIONS = {
    "upwind_2stage_plain_seed123": NO_CLAMP,
    "upwind_2stage_plain_seed888": NO_CLAMP,
    "upwind_2stage_drain0_seed123": NO_CLAMP,
}


def run_condition(label, overrides):
    genome_path = os.path.join(SCRIPT_DIR, label, "genome_trained_n20.npy")
    genome = np.load(genome_path)
    rules = unflatten_abcd(genome)

    cfg_patch = dict(
        HEBBIAN_SAFETY_CLAMP_ENABLED=overrides["safety_clamp"],
        HEBBIAN_MIN_DIST_INFLATION=overrides["min_dist_inflation"],
        HEBBIAN_RESOLVE_COLLISIONS=overrides["resolve_collisions"],
        SPAWN_SQUARE_SIZE=VIDEO_SPAWN_SQUARE_SIZE,
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
        genome_path = os.path.join(SCRIPT_DIR, label, "genome_trained_n20.npy")
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
