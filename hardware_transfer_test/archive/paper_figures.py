"""Regenerates the two evaluation figures from Mahdavi et al.'s ANTS 2026 paper ("Energy-
Efficient Flocking in Self-Organized Robot Swarms") that ants26_replication/experiment/
analyze_hebbian_results.py already reproduces for a single controller/stage sweep -- but here
for all three hardware_transfer_test conditions side by side (lj_baseline / pre_clamp_best /
safety_clamp_best), using the exact reproduction recipe documented in leadership_metrics.py.

  - Fig. 5a: distance travelled vs. remaining battery, N seeded runs per condition, plotted as
    one scatter cluster + 1-std confidence ellipse per condition.
  - Fig. 6: trajectories at a small and a large swarm size, to show that formation
    reconfiguration is a collective phenomenon (visible in a multi-agent run) rather than an
    individual one -- the paper's original comparison used 1 vs. 5 agents; hardware_transfer_test
    has no n=1 condition, so this uses n=2 vs. n=20 instead (noted in the caption).

Usage:
    python paper_figures.py                     # both figures, defaults below
    python paper_figures.py --n-agents 10 --n-seeds 30
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from leadership_metrics import (CONDITIONS, CONDITION_LABELS, CONDITION_COLORS, SEED,
                                 analyze, run_trajectory)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "overleaf_summary", "figures")


def _savefig(fig, out_dir, name, rect=None):
    fig.tight_layout(rect=rect)
    stem = os.path.join(out_dir, os.path.splitext(name)[0])
    fig.savefig(stem + ".png", dpi=150)
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".svg")
    plt.close(fig)
    print(f"Saved {stem}.{{png,pdf,svg}}")


def _confidence_ellipse(ax, x, y, color, n_std=1.0):
    """1-std covariance ellipse -- same construction as analyze_hebbian_results.py's Fig. 5a."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse((np.mean(x), np.mean(y)), width, height, angle=angle,
                          facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.5, zorder=1))


def fig5a_distance_vs_battery(n_agents, seeds, out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for condition in CONDITIONS:
        dists, batts = [], []
        for seed in seeds:
            r = analyze(condition, n_agents, seed=seed)
            dists.append(r["dist_travelled"])
            batts.append(r["battery_pct"])
        color = CONDITION_COLORS[condition]
        dists, batts = np.array(dists), np.array(batts)
        ax.scatter(dists, batts, s=22, alpha=0.7, color=color, label=CONDITION_LABELS[condition],
                   edgecolors="k", linewidths=0.3, zorder=2)
        _confidence_ellipse(ax, dists, batts, color)

    ax.set_xlabel("Distance travelled [m]")
    ax.set_ylabel("Remaining battery (% of full charge, mean across agents)")
    ax.set_title(f"Fig. 5a analog -- distance vs. remaining battery ({len(seeds)} runs/condition, n={n_agents})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    _savefig(fig, out_dir, f"paper_fig5a_distance_vs_battery_n{n_agents}.png")


def fig6_trajectory_comparison(agent_counts, seed, out_dir):
    n_rows, n_cols = len(CONDITIONS), len(agent_counts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows), squeeze=False)

    for row, condition in enumerate(CONDITIONS):
        for col, n_agents in enumerate(agent_counts):
            ax = axes[row][col]
            positions, _, _, _ = run_trajectory(condition, n_agents, seed=seed)
            for a in range(n_agents):
                ax.plot(positions[:, a, 0], positions[:, a, 1], linewidth=1)
                ax.scatter(positions[0, a, 0], positions[0, a, 1], color="green", s=15, zorder=3)
                ax.scatter(positions[-1, a, 0], positions[-1, a, 1], color="red", s=15, zorder=3)
            ax.set_title(f"{CONDITION_LABELS[condition]} | {n_agents} agents", fontsize=10)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(f"Fig. 6 analog -- trajectories at {agent_counts[0]} vs. {agent_counts[1]} agents "
                 "(green=start, red=end); paper's original used 1 vs. 5 agents",
                 fontweight="bold")
    _savefig(fig, out_dir, "paper_fig6_trajectory_comparison.png", rect=[0, 0, 1, 0.95])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-agents", type=int, default=10, help="n_agents for Fig. 5a.")
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1000,
                         help="Matches leadership_metrics.py's seed-sweep default so Fig. 5a "
                              "uses the exact same 30 runs as the metric-3/4 seed-sweep figures.")
    parser.add_argument("--traj-agent-counts", type=int, nargs=2, default=[2, 20])
    parser.add_argument("--traj-seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    print(f"Fig. 5a: n_agents={args.n_agents}, {len(seeds)} seeds x {len(CONDITIONS)} conditions")
    fig5a_distance_vs_battery(args.n_agents, seeds, args.output_dir)

    print(f"Fig. 6: agent_counts={args.traj_agent_counts}, seed={args.traj_seed}")
    fig6_trajectory_comparison(args.traj_agent_counts, args.traj_seed, args.output_dir)
