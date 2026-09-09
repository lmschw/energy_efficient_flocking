"""Standalone Fig. 5a-style distance-vs-battery comparison across ALL SEVEN conditions
(the original three from hardware_transfer_test/ plus the four new walk_left/walk_upwind x
clamp/no-clamp ablation genomes) -- kept separate from leadership_metrics.py/leadership_plots.py
deliberately, so extending the condition set here can't disturb that already-verified
turn-taking-metrics pipeline (whose CONDITIONS tuple, color palette, and plot layouts are all
sized for exactly three conditions).

All seven are evaluated at n_agents=10 (a valid setting for every genome regardless of what
n_agents it was trained at -- the network doesn't encode swarm size) across the same 30 seeds
(1000-1029) used throughout this project's other seed-sweep figures, for direct comparability.

Usage: python compare_all_conditions.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENT_DIR = os.path.join(REPO_ROOT, "ants26_replication", "experiment")
sys.path.insert(0, EXPERIMENT_DIR)

import config  # noqa: E402
from hebbian_controller import unflatten_abcd  # noqa: E402
from simulation_hebbian import simulate_hebbian_episode  # noqa: E402
from lj_baseline import simulate_lj_baseline  # noqa: E402
from leadership_metrics import _ConfigOverride, WIND_GRID  # noqa: E402

N_AGENTS = 10
SEEDS = list(range(1000, 1030))
OUT_DIR = os.path.join(SCRIPT_DIR, "overleaf_summary", "figures")

_BASELINE_RULES_PATH = os.path.join(SCRIPT_DIR, "lj_baseline", "paper_baseline_rules.json")

# label -> (genome_path or None for lj_baseline, clamp overrides, display color)
CONDITIONS = {
    "lj_baseline": (None, None, "#77AC30"),
    "pre_clamp_best": (os.path.join(SCRIPT_DIR, "pre_clamp_best", "genome_trained_n20.npy"),
                        dict(safety_clamp=False, min_dist_inflation=1.0, resolve_collisions=False),
                        "#D95319"),
    "safety_clamp_best": (os.path.join(SCRIPT_DIR, "safety_clamp_best", "genome_trained_n20.npy"),
                           dict(safety_clamp=True, min_dist_inflation=1.3, resolve_collisions=True,
                                resolve_strength=0.5, resolve_max_iter=2),
                           "#0072BD"),
    "walk_left_n10_no_clamp": (os.path.join(SCRIPT_DIR, "walk_left_n10_no_clamp", "genome_trained_n10.npy"),
                                dict(safety_clamp=False, min_dist_inflation=1.0, resolve_collisions=False),
                                "#7E2F8E"),
    "walk_left_n10_clamp": (os.path.join(SCRIPT_DIR, "walk_left_n10_clamp", "genome_trained_n10.npy"),
                             dict(safety_clamp=True, min_dist_inflation=1.3, resolve_collisions=True,
                                  resolve_strength=0.5, resolve_max_iter=2),
                             "#4DBEEE"),
    "upwind_no_clamp": (os.path.join(SCRIPT_DIR, "upwind_no_clamp", "genome_trained_n10.npy"),
                         dict(safety_clamp=False, min_dist_inflation=1.0, resolve_collisions=False),
                         "#EDB120"),
    "upwind_clamp": (os.path.join(SCRIPT_DIR, "upwind_clamp", "genome_trained_n10.npy"),
                      dict(safety_clamp=True, min_dist_inflation=1.3, resolve_collisions=True,
                           resolve_strength=0.5, resolve_max_iter=2),
                      "#A2142F"),
}


def run_one(label, genome_path, overrides, seed):
    if label == "lj_baseline":
        rules = json.load(open(_BASELINE_RULES_PATH))
        with _ConfigOverride(HEBBIAN_NX=WIND_GRID, HEBBIAN_NY=WIND_GRID):
            _, dist, batt, _ = simulate_lj_baseline(rules=rules, seed=seed, n_agents=N_AGENTS)
        return dist, batt / config.MAX_BATTERY * 100.0

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
    with _ConfigOverride(**cfg_patch):
        dist, batt, _, _, _, _ = simulate_hebbian_episode(
            rules, seed=seed, n_agents=N_AGENTS, wind_enabled=True, nx=WIND_GRID, ny=WIND_GRID)
    return dist, batt  # HEBBIAN_MAX_BATTERY=100, so batt is already a percentage


def _confidence_ellipse(ax, x, y, color, n_std=1.0):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse((np.mean(x), np.mean(y)), width, height, angle=angle,
                          facecolor=color, alpha=0.12, edgecolor=color, linewidth=1.5, zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(9, 7))
    results = {}
    for label, (genome_path, overrides, color) in CONDITIONS.items():
        if genome_path is not None and not os.path.exists(genome_path):
            print(f"SKIPPED {label} (no genome at {genome_path})")
            continue
        dists, batts = [], []
        for seed in SEEDS:
            d, b = run_one(label, genome_path, overrides, seed)
            dists.append(d)
            batts.append(b)
        dists, batts = np.array(dists), np.array(batts)
        results[label] = {"dist_mean": float(dists.mean()), "dist_std": float(dists.std()),
                           "batt_mean": float(batts.mean()), "batt_std": float(batts.std())}
        ax.scatter(dists, batts, s=20, alpha=0.7, color=color, label=label,
                   edgecolors="k", linewidths=0.3, zorder=2)
        _confidence_ellipse(ax, dists, batts, color)
        print(f"{label}: dist={dists.mean():.2f}+-{dists.std():.2f}  batt={batts.mean():.2f}+-{batts.std():.2f}%")

    ax.set_xlabel("Distance travelled [m]")
    ax.set_ylabel("Remaining battery [% of full charge]")
    ax.set_title(f"All conditions -- distance vs. remaining battery ({len(SEEDS)} runs each, n={N_AGENTS})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, "all_conditions_distance_vs_battery")
    fig.savefig(stem + ".png", dpi=150)
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".svg")
    plt.close(fig)
    print(f"\nSaved {stem}.{{png,pdf,svg}}")

    with open(os.path.join(SCRIPT_DIR, "overleaf_summary", "all_conditions_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
