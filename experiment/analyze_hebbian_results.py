"""Reproduces the paper's validation experiments (Section 4):

  1. Fig. 5a -- distance travelled vs. remaining battery, 100 simulations of each
     stage's best controller plus the Table 3 baseline, plotted as one cluster per
     controller with a 1-std confidence ellipse.
  2. Fig. 5b / Table 4 -- the battery-awareness experiment: does the "weakest" agent
     travel through calmer (lower average wind-speed) regions when it starts at 50%
     battery instead of 100%? Reports the same independent-samples t-test as Table 4.
  3. Fig. 6 -- trajectories of the stage-1 and stage-3 (or any two chosen) controllers
     at 1 and 5 agents, to see whether formation-reconfiguration manoeuvres appear
     only in a group (paper's argument that these are collective, not individual,
     behaviours).
  4. Trajectory repeats -- not from the paper: the same controller/n_agents run several
     times (different seeds), plotted side by side, as a sanity check for whether the
     learned behavior is consistent run to run or fragile/seed-dependent.

Usage:
    python analyze_hebbian_results.py --results-dir hebbian_results
    python analyze_hebbian_results.py --results-dir hebbian_results --n-sims 20   # faster look
"""
import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats

import config
from hebbian_controller import unflatten_abcd
from simulation_hebbian import simulate_hebbian_episode
from simulation_free_global_mod_2_LJ import simulation_free_global_mod_2_LJ

OUTPUT_DIR = "hebbian_analysis"
N_SIMS = 100                 # paper: "100 simulation runs" for both Fig. 5a and 5b
TRAJECTORY_AGENT_COUNTS = (1, 5)  # paper's Fig. 6 setup
COLOR_CYCLE = ["#D95319", "#0072BD", "#7E2F8E", "#77AC30", "#A2142F"]


def load_stage_genome(results_dir, stage, suffix=""):
    path = os.path.join(results_dir, f"hebbian_{stage}{suffix}_best.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' not found -- run optimize_hebbian.py first "
                                 f"(add --no-battery-sensor if suffix='_nosensor').")
    return unflatten_abcd(np.load(path))


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
                          facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.5, zorder=1))


# --- 1) Fig. 5a: distance vs remaining battery ---
def run_distance_vs_battery(controllers, n_sims, n_agents, seed_start, out_path):
    """controllers: ordered dict label -> callable(seed) -> (dist_travelled, average_batt)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (label, run_one) in enumerate(controllers.items()):
        dists, batts = [], []
        for s in range(n_sims):
            dist, batt = run_one(seed_start + s)
            dists.append(dist)
            batts.append(batt)
            sys.stdout.write(f"\r   ↳ {label}: {s + 1}/{n_sims} simulations")
            sys.stdout.flush()
        print()
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        dists, batts = np.array(dists), np.array(batts)
        ax.scatter(dists, batts, s=18, alpha=0.6, color=color, label=label,
                   edgecolors="k", linewidths=0.2, zorder=2)
        _confidence_ellipse(ax, dists, batts, color)

    ax.set_xlabel("Distance travelled [m]")
    ax.set_ylabel("Remaining battery (mean across agents)")
    ax.set_title(f"Distance vs. Remaining Battery -- {n_sims} runs per controller")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# --- 2) Fig. 5b / Table 4: battery-awareness wind-exposure experiment ---
def run_battery_awareness_experiment(rules, n_sims, n_agents, seed_start, out_path):
    """Tracks the wind exposure of the swarm's "weakest" agent (spawned last, per
    _spawn_agents' convention) across n_sims runs, once with it starting at 100%
    battery (matching everyone else) and once at 50%, all else identical. Reproduces
    Table 4's independent-samples t-test on mean wind-speed percentage experienced."""

    def _condition(min_battery, label):
        exposures = []
        for s in range(n_sims):
            _, _, _, _, tele = simulate_hebbian_episode(
                rules, seed=seed_start + s, n_agents=n_agents, wind_enabled=True,
                max_battery=100.0, min_battery=min_battery, record_wind_exposure=True)
            exposures.append(float(np.mean(tele["wind_pct"][:, -1])))
            sys.stdout.write(f"\r   ↳ {label}: {s + 1}/{n_sims} simulations")
            sys.stdout.flush()
        print()
        return np.array(exposures)

    exp_100 = _condition(100.0, "agent at 100% battery")
    exp_50 = _condition(50.0, "agent at 50% battery")

    t_stat, p_value = stats.ttest_ind(exp_100, exp_50)
    df = len(exp_100) + len(exp_50) - 2

    print(f"\n{'Experiment':<24}{'No. of runs':<14}{'Mean':<10}{'Std':<10}")
    print(f"{'Agent w/ 100% battery':<24}{len(exp_100):<14}{exp_100.mean():<10.2f}{exp_100.std(ddof=1):<10.2f}")
    print(f"{'Agent w/ 50% battery':<24}{len(exp_50):<14}{exp_50.mean():<10.2f}{exp_50.std(ddof=1):<10.2f}")
    print(f"t-test: t={t_stat:.4f}, p={p_value:.4e}, df={df}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.boxplot([exp_100, exp_50], tick_labels=["100%", "50%"])
    ax.set_ylabel("Average wind speed experienced [%]")
    ax.set_xlabel("Initial battery of the tracked agent")
    ax.set_title(f"Battery-Awareness Wind Exposure (p={p_value:.2e})")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    return {"exp_100": exp_100.tolist(), "exp_50": exp_50.tolist(),
            "t_stat": float(t_stat), "p_value": float(p_value), "df": df}


# --- 3) Fig. 6: trajectories at different agent counts ---
def run_trajectory_comparison(controllers, agent_counts, seed, out_path):
    """controllers: ordered dict label -> rules dict (Hebbian ABCD rules)."""
    n_rows, n_cols = len(controllers), len(agent_counts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    for row, (label, rules) in enumerate(controllers.items()):
        for col, n_agents in enumerate(agent_counts):
            ax = axes[row][col]
            _, _, _, _, tele = simulate_hebbian_episode(
                rules, seed=seed, n_agents=n_agents, wind_enabled=True, record_trajectory=True)
            positions = tele["positions"]
            for a in range(n_agents):
                ax.plot(positions[:, a, 0], positions[:, a, 1], linewidth=1)
                ax.scatter(positions[0, a, 0], positions[0, a, 1], color="green", s=15, zorder=3)
                ax.scatter(positions[-1, a, 0], positions[-1, a, 1], color="red", s=15, zorder=3)
            ax.set_title(f"{label} | {n_agents} agent{'s' if n_agents != 1 else ''}")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Trajectory Comparison Across Agent Counts (green=start, red=end)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# --- 4) Trajectory repeats: same controller/n_agents, several seeds, sanity check ---
def run_trajectory_repeats(rules, n_agents, wind_enabled, n_repeats, seed_start, out_path):
    """Runs the SAME controller and n_agents n_repeats times, each with a different
    seed (different spawn positions), plotting every run side by side. Unlike
    run_trajectory_comparison() (which varies controller/agent-count to compare
    conditions), this holds everything fixed except the random seed -- a sanity check
    for whether the learned behavior is consistent run to run or fragile/seed-dependent,
    which a single trajectory plot can't tell you."""
    n_cols = min(5, n_repeats)
    n_rows = math.ceil(n_repeats / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    for i in range(n_repeats):
        seed = seed_start + i
        ax = axes[i // n_cols][i % n_cols]
        _, _, _, _, tele = simulate_hebbian_episode(
            rules, seed=seed, n_agents=n_agents, wind_enabled=wind_enabled, record_trajectory=True)
        positions = tele["positions"]
        for a in range(n_agents):
            ax.plot(positions[:, a, 0], positions[:, a, 1], linewidth=1)
            ax.scatter(positions[0, a, 0], positions[0, a, 1], color="green", s=15, zorder=3)
            ax.scatter(positions[-1, a, 0], positions[-1, a, 1], color="red", s=15, zorder=3)
        ax.set_title(f"seed={seed} ({positions.shape[0]} steps)")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle=":", alpha=0.5)

    for j in range(n_repeats, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(f"Trajectory repeats -- {n_repeats} runs, {n_agents} agents, "
                 f"wind_enabled={wind_enabled} (green=start, red=end)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Reproduce the paper's Fig. 5a/5b/6 validation experiments.")
    parser.add_argument("--results-dir", default="hebbian_results",
                         help="Directory with hebbian_<stage>[_nosensor]_best.npy files.")
    parser.add_argument("--suffix", default="", choices=["", "_nosensor"],
                         help="Load the sensor-equipped ('') or battery-sensor-ablated genomes.")
    parser.add_argument("--n-sims", type=int, default=N_SIMS,
                         help=f"Simulations per controller/condition (paper: {N_SIMS}).")
    parser.add_argument("--n-agents", type=int, default=config.HEBBIAN_N_AGENTS,
                         help=f"Swarm size (paper: {config.HEBBIAN_N_AGENTS}).")
    parser.add_argument("--seed", type=int, default=10_000, help="Base seed for all experiments.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Where to save plots.")
    parser.add_argument("--skip", nargs="*", default=[],
                         choices=["distance_battery", "awareness", "trajectories", "trajectory_repeats"],
                         help="Skip one or more of the experiments.")
    parser.add_argument("--trajectory-repeat-stage", default=None, choices=list(config.HEBBIAN_STAGES),
                         help="Which stage's controller to use for the trajectory-repeats sanity "
                              "check (default: the last/most complete stage).")
    parser.add_argument("--n-trajectory-repeats", type=int, default=10,
                         help="How many seeds to run for the trajectory-repeats sanity check.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    stages = list(config.HEBBIAN_STAGES)
    stage_rules = {stage: load_stage_genome(args.results_dir, stage, args.suffix) for stage in stages}

    if "distance_battery" not in args.skip:
        print("\n=== Fig. 5a: distance vs. remaining battery ===")

        def _make_hebbian_runner(rules):
            def _run(seed):
                dist, batt, _, _ = simulate_hebbian_episode(
                    rules, seed=seed, n_agents=args.n_agents, wind_enabled=True)
                return dist, batt
            return _run

        def _run_baseline(seed):
            _, dist, batt, _ = simulation_free_global_mod_2_LJ(
                rules=config.PAPER_BASELINE_RULES, seed=seed, visualize=False, n_agents=args.n_agents)
            return dist, batt

        controllers = {stage: _make_hebbian_runner(stage_rules[stage]) for stage in stages}
        controllers["baseline"] = _run_baseline
        run_distance_vs_battery(controllers, args.n_sims, args.n_agents, args.seed,
                                 os.path.join(args.output_dir, f"distance_vs_battery{args.suffix}.png"))

    if "awareness" not in args.skip:
        print("\n=== Fig. 5b / Table 4: battery-awareness experiment (stage 3 controller) ===")
        run_battery_awareness_experiment(
            stage_rules[stages[-1]], args.n_sims, args.n_agents, args.seed + 50_000,
            os.path.join(args.output_dir, f"battery_awareness{args.suffix}.png"))

    if "trajectories" not in args.skip:
        print("\n=== Fig. 6: trajectory comparison (stage 1 vs. stage 3) ===")
        controllers = {stages[0]: stage_rules[stages[0]], stages[-1]: stage_rules[stages[-1]]}
        run_trajectory_comparison(controllers, TRAJECTORY_AGENT_COUNTS, args.seed + 100_000,
                                   os.path.join(args.output_dir, f"trajectories{args.suffix}.png"))

    if "trajectory_repeats" not in args.skip:
        repeat_stage = args.trajectory_repeat_stage or stages[-1]
        print(f"\n=== Trajectory repeats: '{repeat_stage}' controller, "
              f"{args.n_trajectory_repeats} seeds, {args.n_agents} agents ===")
        run_trajectory_repeats(
            stage_rules[repeat_stage], args.n_agents, config.HEBBIAN_STAGE_WIND_ENABLED[repeat_stage],
            args.n_trajectory_repeats, args.seed + 200_000,
            os.path.join(args.output_dir, f"trajectory_repeats_{repeat_stage}{args.suffix}.png"))

    print(f"\nDone. Plots saved in '{args.output_dir}/'.")


if __name__ == "__main__":
    main()
