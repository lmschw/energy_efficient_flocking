"""Analysis of a run_batch_experiments.py batch: fitness convergence per curriculum
stage, distance-vs-battery for repeated runs of each stage's best controller (and
the un-evolved baseline), and trajectories of each stage's best controller across
different agent counts.

Reads optimization_results/master_optimization_summary.json (produced by
run_batch_experiments.py) and writes PNGs to analysis_results/.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import config
from simulation_free_global_mod_2_LJ import simulation_free_global_mod_2_LJ

SUMMARY_PATH = os.path.join(config.BATCH_OUTPUT_DIR, "master_optimization_summary.json")
OUTPUT_DIR = "analysis_results"

N_BATTERY_SIMULATIONS = 100          # part 2: repeats per controller
BATTERY_SIM_SEED_START = 10_000      # keep well clear of the batch's own training seeds
TRAJECTORY_AGENT_COUNTS = range(1, 11)   # part 3: agent counts to sweep
TRAJECTORY_SEED = 0

STAGE_COLORS = {"small_scale": "#D95319", "baseline": "#0072BD", "large_scale": "#7E2F8E"}
BASELINE_COLOR = "#77AC30"


def load_trials():
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(
            f"'{SUMMARY_PATH}' not found -- run experiment/run_batch_experiments.py to completion first."
        )
    with open(SUMMARY_PATH) as f:
        return json.load(f)


def best_trial_per_stage(trials):
    """The highest-best_efficiency trial for each curriculum stage."""
    best = {}
    for trial in trials:
        stage = trial["config"]
        if stage not in best or trial["best_efficiency"] > best[stage]["best_efficiency"]:
            best[stage] = trial
    return best


def stage_color(stage, fallback_index):
    if stage in STAGE_COLORS:
        return STAGE_COLORS[stage]
    palette = list(STAGE_COLORS.values())
    return palette[fallback_index % len(palette)]


# --- 1) Fitness per curriculum stage across the evolution process (min/median/max) ---
def plot_fitness_per_stage(trials, out_path):
    stages = sorted(set(t["config"] for t in trials))
    fig, axes = plt.subplots(1, len(stages), figsize=(6 * len(stages), 5), sharey=True)
    if len(stages) == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        stage_trials = [t for t in trials if t["config"] == stage]
        curves = [[-loss for loss in t["loss_curve"]] for t in stage_trials]  # loss -> efficiency
        max_len = max(len(c) for c in curves)
        padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])
        gens = np.arange(1, max_len + 1)

        color = stage_color(stage, list(stages).index(stage))
        ax.fill_between(gens, padded.min(axis=0), padded.max(axis=0), color=color, alpha=0.15)
        ax.plot(gens, padded.min(axis=0), color=color, linestyle=":", label="min")
        ax.plot(gens, np.median(padded, axis=0), color=color, linewidth=2.5, label="median")
        ax.plot(gens, padded.max(axis=0), color=color, linestyle="--", label="max")

        ax.set_title(f"{stage} ({stage_trials[0]['agents']} agents, n={len(stage_trials)} trials)")
        ax.set_xlabel("Generation")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right")

    axes[0].set_ylabel("Best efficiency in population")
    fig.suptitle("Fitness per Curriculum Stage Across the Evolution Process", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# --- 2) Distance vs remaining battery for repeated runs of each controller ---
def collect_distance_battery(rules, n_sims, seed_start, label):
    distances, batteries = [], []
    for i in range(n_sims):
        seed = seed_start + i
        _, dist_travelled, average_batt, _ = simulation_free_global_mod_2_LJ(
            rules=rules, seed=seed, visualize=False
        )
        distances.append(dist_travelled)
        batteries.append(average_batt)
        sys.stdout.write(f"\r   ↳ {label}: {i + 1}/{n_sims} simulations")
        sys.stdout.flush()
    print()
    return np.array(distances), np.array(batteries)


def plot_distance_vs_battery(stage_bests, out_path, n_sims=None):
    n_sims = n_sims if n_sims is not None else N_BATTERY_SIMULATIONS
    controllers = [("default_baseline", config.DEFAULT_RULES, BASELINE_COLOR)]
    for i, (stage, trial) in enumerate(sorted(stage_bests.items())):
        rules = config.genome_to_rules(trial["genome"])
        controllers.append((f"{stage}_best", rules, stage_color(stage, i)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, rules, color in controllers:
        distances, batteries = collect_distance_battery(rules, n_sims, BATTERY_SIM_SEED_START, label)
        ax.scatter(distances, batteries, s=25, alpha=0.6, color=color, label=label,
                   edgecolors="k", linewidths=0.3)

    ax.set_xlabel("Distance travelled [m]")
    ax.set_ylabel("Remaining battery (mean across agents)")
    ax.set_title(f"Distance vs. Remaining Battery -- {n_sims} runs per controller")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# --- 3) Trajectories of each stage's best controller across agent counts ---
def plot_trajectories_for_stage(stage, rules, agent_counts, out_path, seed=TRAJECTORY_SEED):
    n_cols = min(5, len(agent_counts))
    n_rows = int(np.ceil(len(agent_counts) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for idx, n_agents in enumerate(agent_counts):
        ax = axes[idx // n_cols][idx % n_cols]
        _, _, _, _, positions = simulation_free_global_mod_2_LJ(
            rules=rules, seed=seed, visualize=False, n_agents=n_agents, record_trajectory=True
        )
        for a in range(n_agents):
            ax.plot(positions[:, a, 0], positions[:, a, 1], linewidth=1)
            ax.scatter(positions[0, a, 0], positions[0, a, 1], color="green", s=15, zorder=3)
            ax.scatter(positions[-1, a, 0], positions[-1, a, 1], color="red", s=15, zorder=3)
        ax.set_title(f"{n_agents} agent{'s' if n_agents != 1 else ''}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(True, linestyle=":", alpha=0.5)

    for idx in range(len(agent_counts), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(f"Trajectories of the {stage} Best Controller Across Agent Counts "
                 "(green=start, red=end)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trials = load_trials()
    stage_bests = best_trial_per_stage(trials)

    plot_fitness_per_stage(trials, os.path.join(OUTPUT_DIR, "fitness_per_stage.png"))

    plot_distance_vs_battery(stage_bests, os.path.join(OUTPUT_DIR, "distance_vs_battery.png"))

    for stage, trial in sorted(stage_bests.items()):
        rules = config.genome_to_rules(trial["genome"])
        out_path = os.path.join(OUTPUT_DIR, f"trajectories_{stage}.png")
        plot_trajectories_for_stage(stage, rules, TRAJECTORY_AGENT_COUNTS, out_path)


if __name__ == "__main__":
    main()
