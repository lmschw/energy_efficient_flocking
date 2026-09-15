"""Headline figure for the "must beat LJ baseline" investigation (see memory:
project_upwind_safety_variant.md, LATEST STATUS section): distance-vs-battery scatter, n=20,
30 seeds, comparing the n=20 LJ baseline against the three winning 2-stage-upwind genomes
(plain-drain seeds 123/888, 0%-drain-holiday seed 123) -- all evaluated under STANDARD (real,
non-refunded) drain physics, since real hardware never gets free energy for colliding.

n=20 (not n=10) is used throughout because these genomes were trained at n=20, and because the
LJ baseline's own behavior is swarm-size-dependent (n=10: dist=21.00/batt=24.57%; n=20:
dist~8.81/batt~18.68% -- baseline collapses at the larger swarm size, so this is NOT the same
comparison as the earlier all_conditions_distance_vs_battery.py figure, which used n=10).

Usage: python beat_baseline_final_comparison.py
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

N_AGENTS = 20
SEEDS = list(range(1000, 1030))
OUT_DIR = os.path.join(SCRIPT_DIR, "overleaf_summary", "figures")

_BASELINE_RULES_PATH = os.path.join(SCRIPT_DIR, "lj_baseline", "paper_baseline_rules.json")

NO_CLAMP = dict(safety_clamp=False, min_dist_inflation=1.0, resolve_collisions=False)

CONDITIONS = {
    "lj_baseline_n20": (None, None, "#77AC30"),
    "upwind_2stage_plain_seed123": (
        os.path.join(SCRIPT_DIR, "upwind_2stage_plain_seed123", "genome_trained_n20.npy"),
        NO_CLAMP, "#0072BD"),
    "upwind_2stage_plain_seed888": (
        os.path.join(SCRIPT_DIR, "upwind_2stage_plain_seed888", "genome_trained_n20.npy"),
        NO_CLAMP, "#7E2F8E"),
    "upwind_2stage_drain0_seed123": (
        os.path.join(SCRIPT_DIR, "upwind_2stage_drain0_seed123", "genome_trained_n20.npy"),
        NO_CLAMP, "#D95319"),
}


def run_one(label, genome_path, overrides, seed):
    if genome_path is None:
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
    with _ConfigOverride(**cfg_patch):
        dist, batt, _, _, _, _ = simulate_hebbian_episode(
            rules, seed=seed, n_agents=N_AGENTS, wind_enabled=True, nx=WIND_GRID, ny=WIND_GRID)
    return dist, batt


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
        ax.scatter(dists, batts, s=24, alpha=0.7, color=color, label=label,
                   edgecolors="k", linewidths=0.3, zorder=2)
        _confidence_ellipse(ax, dists, batts, color)
        print(f"{label}: dist={dists.mean():.2f}+-{dists.std():.2f}  batt={batts.mean():.2f}+-{batts.std():.2f}%")

    ax.set_xlabel("Distance travelled [m]")
    ax.set_ylabel("Remaining battery [% of full charge]")
    ax.set_title(f"Winning genomes vs. LJ baseline -- distance vs. remaining battery "
                 f"({len(SEEDS)} runs each, n={N_AGENTS}, standard/real drain physics)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, "beat_baseline_n20_distance_vs_battery")
    fig.savefig(stem + ".png", dpi=150)
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".svg")
    plt.close(fig)
    print(f"\nSaved {stem}.{{png,pdf,svg}}")

    with open(os.path.join(SCRIPT_DIR, "overleaf_summary", "beat_baseline_n20_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
