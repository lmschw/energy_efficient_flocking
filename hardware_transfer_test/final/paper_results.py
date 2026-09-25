"""Simulation results for the paper draft (swarm_intelligence.tex), following its protocol:
100 evaluation runs at n=20 per controller (the paper's Fig. 5a), the battery-awareness test
(one agent starting at 50% vs. all at 100%, t-test on that agent's mean wind exposure), and
1- vs. 5-agent trajectories for the stage-1 vs. stage-2 controller (the paper's Fig. 6).

Controllers: the stage-1 (walk_upwind) and stage-2 (final) genomes of the unclamped
plain_seed123 run, the clamped stage-2 genome, and the LJ baseline. The LJ baseline is also
run with a 100-unit battery (instead of the LJ model's native 150) so the effect of that
difference can be judged; it is reported in the JSON but not plotted.

Usage: python paper_results.py   (runs in parallel; a few minutes on a multi-core machine)
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "ants26_replication", "experiment"))

import config  # noqa: E402
from hebbian_controller import unflatten_abcd  # noqa: E402
from simulation_hebbian import simulate_hebbian_episode  # noqa: E402
from lj_baseline import simulate_lj_baseline  # noqa: E402
from leadership_metrics import _ConfigOverride, WIND_GRID  # noqa: E402

OUT_DIR_FIG = os.path.join(SCRIPT_DIR, "overleaf_summary", "figures")
OUT_JSON = os.path.join(SCRIPT_DIR, "overleaf_summary", "paper_results.json")
N_AGENTS = 20
SEEDS = list(range(1000, 1100))
TRAJ_SEED = config.HEBBIAN_DEFAULT_SEED  # 42
LJ_RULES_PATH = os.path.join(SCRIPT_DIR, "lj_baseline", "paper_baseline_rules.json")

NO_CLAMP = dict(HEBBIAN_SAFETY_CLAMP_ENABLED=False, HEBBIAN_MIN_DIST_INFLATION=1.0,
                HEBBIAN_RESOLVE_COLLISIONS=False)
WITH_CLAMP = dict(HEBBIAN_SAFETY_CLAMP_ENABLED=True, HEBBIAN_MIN_DIST_INFLATION=1.3,
                  HEBBIAN_RESOLVE_COLLISIONS=True, HEBBIAN_RESOLVE_COLLISIONS_STRENGTH=0.5,
                  HEBBIAN_RESOLVE_COLLISIONS_MAX_ITER=2)
UNCLAMPED_RUN = os.path.join(REPO_ROOT, "results", "hebbian_results_v2_2stage_upwind", "n20_seed123")

# label -> (genome path or None for LJ, config overrides, colour, cluster number in the figure)
CONTROLLERS = {
    "Stage 1": (os.path.join(UNCLAMPED_RUN, "hebbian_walk_upwind_best.npy"), NO_CLAMP, "#EDB120", 1),
    "Stage 2": (os.path.join(SCRIPT_DIR, "plain_seed123", "genome_trained_n20.npy"), NO_CLAMP, "#D95319", 2),
    "Stage 2, clamped": (os.path.join(SCRIPT_DIR, "plain_seed123_clamped", "genome_trained_n20.npy"),
                         WITH_CLAMP, "#0072BD", 3),
    "LJ baseline": (None, {}, "#77AC30", 4),
}


def _hebbian(path, cfg, seed, n_agents, **kw):
    rules = unflatten_abcd(np.load(path))
    with _ConfigOverride(**cfg):
        return simulate_hebbian_episode(rules, seed=seed, n_agents=n_agents, wind_enabled=True,
                                        nx=WIND_GRID, ny=WIND_GRID, **kw)


def _lj(seed, n_agents, battery):
    rules = json.load(open(LJ_RULES_PATH))
    with _ConfigOverride(HEBBIAN_NX=WIND_GRID, HEBBIAN_NY=WIND_GRID, MAX_BATTERY=battery, MIN_BATTERY=battery):
        _, dist, batt, _ = simulate_lj_baseline(rules=rules, seed=seed, n_agents=n_agents)
    return float(dist), float(batt / battery * 100.0)


def eval_job(args):
    """(label, seed) -> (label, seed, distance [m], mean remaining battery [% of start])."""
    label, seed = args
    if label == "LJ baseline":
        return (label, seed, *_lj(seed, N_AGENTS, config.MAX_BATTERY))
    if label == "LJ baseline, 100-unit battery":
        return (label, seed, *_lj(seed, N_AGENTS, config.HEBBIAN_MAX_BATTERY))
    path, cfg, *_ = CONTROLLERS[label]
    dist, batt, *_ = _hebbian(path, cfg, seed, N_AGENTS)
    return label, seed, float(dist), float(batt)


def awareness_job(args):
    """(min_battery, seed) -> mean wind exposure [%] of the last agent (the one that starts at
    min_battery when min_battery < 100) over the episode."""
    min_batt, seed = args
    path, cfg, *_ = CONTROLLERS["Stage 2"]
    *_, tele = _hebbian(path, cfg, seed, N_AGENTS, max_battery=100.0, min_battery=min_batt,
                        record_wind_exposure=True)
    return min_batt, seed, float(np.mean(tele["wind_pct"][1:, -1]))


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
    labels = list(CONTROLLERS) + ["LJ baseline, 100-unit battery"]
    with ProcessPoolExecutor() as pool:
        evals = list(pool.map(eval_job, [(l, s) for l in labels for s in SEEDS]))
        aware = list(pool.map(awareness_job, [(b, s) for b in (100.0, 50.0) for s in SEEDS]))

    out = {"n_agents": N_AGENTS, "seeds": [SEEDS[0], SEEDS[-1]], "controllers": {}}
    per = {l: np.array([(d, b) for (ll, _, d, b) in evals if ll == l]) for l in labels}
    lj = np.array([(d, b) for (ll, _, d, b) in sorted(evals, key=lambda e: e[1]) if ll == "LJ baseline"])
    for l in labels:
        x = np.array([(d, b) for (ll, _, d, b) in sorted(evals, key=lambda e: e[1]) if ll == l])
        entry = {"dist_mean": x[:, 0].mean(), "dist_std": x[:, 0].std(),
                 "batt_mean": x[:, 1].mean(), "batt_std": x[:, 1].std()}
        if not l.startswith("LJ"):
            entry["dist_gain_vs_lj_pct"] = 100 * (x[:, 0].mean() / lj[:, 0].mean() - 1)
            entry["both_beat_paired"] = int(np.sum((x[:, 0] >= lj[:, 0]) & (x[:, 1] >= lj[:, 1])))
        out["controllers"][l] = {k: float(v) if not isinstance(v, int) else v for k, v in entry.items()}
        print(f"{l:30s} dist={entry['dist_mean']:6.2f}±{entry['dist_std']:5.2f}  "
              f"batt={entry['batt_mean']:6.2f}±{entry['batt_std']:5.2f}"
              + (f"  dist vs LJ {entry['dist_gain_vs_lj_pct']:+.0f}%  both-beat {entry['both_beat_paired']}/100"
                 if "both_beat_paired" in entry else ""))

    full = np.array([w for b, _, w in aware if b == 100.0])
    half = np.array([w for b, _, w in aware if b == 50.0])
    t = stats.ttest_ind(full, half)
    out["battery_awareness"] = {
        "full_mean": float(full.mean()), "full_std": float(full.std(ddof=1)),
        "half_mean": float(half.mean()), "half_std": float(half.std(ddof=1)),
        "t": float(t.statistic), "p": float(t.pvalue), "df": int(len(full) + len(half) - 2)}
    print("battery awareness:", out["battery_awareness"])
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"Saved {OUT_JSON}")

    # --- Fig. 5a analogue: distance vs. remaining battery ---
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
    _save(fig, "paper_dist_vs_battery_n20")

    # --- battery-awareness box plot ---
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.boxplot([half, full], widths=0.5, showfliers=False)
    rng = np.random.default_rng(0)
    for i, d in enumerate((half, full), start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=9, color="#D95319", alpha=0.6, lw=0)
    ax.set_xticks([1, 2], ["starts at 50%", "starts at 100%"])
    ax.set_ylabel("Mean wind experienced [% of free stream]")
    ax.grid(True, axis="y", ls=":", alpha=0.6)
    _save(fig, "paper_battery_awareness_boxplot")

    # --- Fig. 6 analogue: trajectories, 1 and 5 agents, stage 1 vs stage 2 ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    for col, l in enumerate(("Stage 1", "Stage 2")):
        path, cfg, *_ = CONTROLLERS[l]
        for row, n in enumerate((1, 5)):
            *_, tele = _hebbian(path, cfg, TRAJ_SEED, n, record_trajectory=True)
            pos = tele["positions"]
            ax = axes[row, col]
            for a in range(n):
                ax.plot(pos[:, a, 0], pos[:, a, 1], lw=0.9)
                ax.plot(pos[0, a, 0], pos[0, a, 1], "o", color="k", ms=3)
            ax.set_title(f"{l}, {n} agent{'s' if n > 1 else ''}", fontsize=10)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, ls=":", alpha=0.6)
            ax.set_xlabel("x [m] (wind blows towards +x)")
            ax.set_ylabel("y [m]")
    fig.tight_layout()
    _save(fig, "paper_trajectories_stage1_vs_stage2")


if __name__ == "__main__":
    main()
