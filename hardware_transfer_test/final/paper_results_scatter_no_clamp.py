"""No-clamp fork of paper_results.py, trimmed to just the Fig. 5a scatter (distance vs.
remaining battery, n=20, 100 seeds), dropping the "Stage 2, clamped" cluster for the paper
version that removes the clamped controller entirely.

The battery-awareness box plot and the stage-1-vs-stage-2 trajectory figure from the original
script are unaffected by the clamped controller (they never use it), so they are not
regenerated here -- reuse the existing paper_battery_awareness_boxplot.* and
paper_trajectories_stage1_vs_stage2.* files as-is.

Usage: python paper_results_scatter_no_clamp.py   (runs in parallel; ~1-2 minutes)
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "ants26_replication", "experiment"))

import config  # noqa: E402
from hebbian_controller import unflatten_abcd  # noqa: E402
from simulation_hebbian import simulate_hebbian_episode  # noqa: E402
from lj_baseline import simulate_lj_baseline  # noqa: E402
from leadership_metrics import _ConfigOverride, WIND_GRID  # noqa: E402

OUT_DIR_FIG = os.path.join(SCRIPT_DIR, "overleaf_summary", "figures")
N_AGENTS = 20
SEEDS = list(range(1000, 1100))
LJ_RULES_PATH = os.path.join(SCRIPT_DIR, "lj_baseline", "paper_baseline_rules.json")

NO_CLAMP = dict(HEBBIAN_SAFETY_CLAMP_ENABLED=False, HEBBIAN_MIN_DIST_INFLATION=1.0,
                HEBBIAN_RESOLVE_COLLISIONS=False)
UNCLAMPED_RUN = os.path.join(REPO_ROOT, "results", "hebbian_results_v2_2stage_upwind", "n20_seed123")

# label -> (genome path or None for LJ, config overrides, colour, cluster number in the figure)
# renumbered 1, 2, 3 (was 1, 2, 4 in the original with-clamp figure) since cluster 3
# (the clamped controller) no longer exists in this version.
CONTROLLERS = {
    "Stage 1": (os.path.join(UNCLAMPED_RUN, "hebbian_walk_upwind_best.npy"), NO_CLAMP, "#EDB120", 1),
    "Stage 2": (os.path.join(SCRIPT_DIR, "plain_seed123", "genome_trained_n20.npy"), NO_CLAMP, "#D95319", 2),
    "LJ baseline": (None, {}, "#77AC30", 3),
}


def _hebbian(path, cfg, seed, n_agents, **kw):
    rules = unflatten_abcd(np.load(path))
    with _ConfigOverride(**cfg):
        return simulate_hebbian_episode(rules, seed=seed, n_agents=n_agents, wind_enabled=True,
                                        nx=WIND_GRID, ny=WIND_GRID, **kw)


def _lj(seed, n_agents, battery):
    import json
    rules = json.load(open(LJ_RULES_PATH))
    with _ConfigOverride(HEBBIAN_NX=WIND_GRID, HEBBIAN_NY=WIND_GRID, MAX_BATTERY=battery, MIN_BATTERY=battery):
        _, dist, batt, _ = simulate_lj_baseline(rules=rules, seed=seed, n_agents=n_agents)
    return float(dist), float(batt / battery * 100.0)


def eval_job(args):
    label, seed = args
    if label == "LJ baseline":
        return (label, seed, *_lj(seed, N_AGENTS, config.MAX_BATTERY))
    path, cfg, *_ = CONTROLLERS[label]
    dist, batt, *_ = _hebbian(path, cfg, seed, N_AGENTS)
    return label, seed, float(dist), float(batt)


def _ellipse(ax, x, y, color):
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse((np.mean(x), np.mean(y)), w, h, angle=angle, fc=color, alpha=0.15,
                         ec=color, lw=1.5, zorder=1))


def _save(fig, name):
    stem = os.path.join(OUT_DIR_FIG, name)
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.{{png,pdf,svg}}")


def main():
    os.makedirs(OUT_DIR_FIG, exist_ok=True)
    labels = list(CONTROLLERS)
    with ProcessPoolExecutor() as pool:
        evals = list(pool.map(eval_job, [(l, s) for l in labels for s in SEEDS]))

    per = {l: np.array([(d, b) for (ll, _, d, b) in evals if ll == l]) for l in labels}
    for l in labels:
        x = per[l]
        print(f"{l:30s} dist={x[:, 0].mean():6.2f}+-{x[:, 0].std():5.2f}  "
              f"batt={x[:, 1].mean():6.2f}+-{x[:, 1].std():5.2f}")

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for l, (_, _, color, num) in CONTROLLERS.items():
        x = per[l]
        ax.scatter(x[:, 0], x[:, 1], s=12, color=color, alpha=0.6, lw=0, zorder=2, label=f"({num}) {l}")
        _ellipse(ax, x[:, 0], x[:, 1], color)
        ax.text(x[:, 0].mean(), x[:, 1].mean(), str(num), ha="center", va="center", fontsize=11,
                fontweight="bold", color="k", zorder=3)
    ax.set_xlabel("Distance travelled against the wind [m]")
    ax.set_ylabel("Mean remaining battery [% of starting charge]")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, "paper_dist_vs_battery_n20_no_clamp")


if __name__ == "__main__":
    main()
