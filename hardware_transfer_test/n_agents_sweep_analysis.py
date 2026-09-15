"""Extends the "beat the LJ baseline" investigation (see memory: project_upwind_safety_variant.md)
down to the small swarm sizes relevant for real hardware trials, and adds the requested
across-n_agents comparison plots for the three winning 2-stage-upwind genomes
(upwind_2stage_plain_seed123, upwind_2stage_plain_seed888, upwind_2stage_drain0_seed123).

Everything here uses the TIGHTER spawn square (SPAWN_SQUARE_SIZE=1.0, see
generate_upwind_2stage_conditions.py's docstring) rather than config.py's default 3.0m --
at the default spawn, agents can spawn entirely outside each other's 2.01m sensing range
(~29% of the time for n=2, empirically), which is especially punishing for the small-n
sizes this script targets. All eval is under STANDARD (real, non-refunded) drain physics
regardless of each genome's training-time drain regime, matching the hardware-realism
check already done at n=20.

Produces, for n_agents in {1,2,3,4,5,7,10,20} (metric 3 / battery-spread skipped at n=1,
which has no "other agent" for either to be meaningful about):

1. `overleaf_summary/n_agents_sweep_comparison.json` -- per-condition, per-n: dist/batt
   mean+std (30 seeds, 1000-1029) and both-beat-rate against the LJ baseline AT THAT SAME n
   (not the n=20 baseline used in the earlier headline comparison -- the LJ baseline itself
   is swarm-size-dependent, so "beating baseline" at n=3 means beating the n=3 baseline).
2. `figures/n_agents_sweep_distance_battery.{pdf,svg,png}` -- distance and battery vs.
   n_agents, one line per condition with a +-1 std band, two panels. Designed so a handful of
   real-robot data points (no need for 30 runs) can be overlaid on top later.
3. `figures/n_agents_sweep_position_change.{pdf,svg,png}` -- the front-rank position-
   exchange rate ("position changing... seen in the videos", Mirzaeinia et al. 2019-style,
   same underlying signal as the project's existing Metric 3) vs. n_agents, single seed=42
   (matching how this metric's original vs.-swarm-size figure was built), per-metre
   normalized so episode-length differences between conditions don't confound it.
4. `figures/n_agents_sweep_battery_equity.{pdf,svg,png}` -- suggested addition: final
   per-agent battery std (i.e. how *unevenly* charge is distributed across the swarm) vs.
   n_agents, single seed=42. Tests whether the wind-shielding/position-swapping energy-
   sharing strategy visible in the battery_plot.png files actually needs a large swarm to
   work, or holds up even at hardware-relevant sizes -- directly relevant to "how many real
   robots do I need to see this benefit."

Usage: python n_agents_sweep_analysis.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENT_DIR = os.path.join(REPO_ROOT, "ants26_replication", "experiment")
sys.path.insert(0, EXPERIMENT_DIR)

import config  # noqa: E402
from hebbian_controller import unflatten_abcd  # noqa: E402
from simulation_hebbian import simulate_hebbian_episode  # noqa: E402
from lj_baseline import simulate_lj_baseline  # noqa: E402
from leadership_metrics import _ConfigOverride, WIND_GRID  # noqa: E402
from leadership_metrics import _front_rank_series, occupancy_and_exchange  # noqa: E402

N_AGENTS_SWEEP = (1, 2, 3, 4, 5, 7, 10, 20)
SEEDS = list(range(1000, 1030))
SEED_SINGLE = config.HEBBIAN_DEFAULT_SEED  # 42
TIGHT_SPAWN = 1.0
OUT_DIR_FIG = os.path.join(SCRIPT_DIR, "overleaf_summary", "figures")
OUT_JSON = os.path.join(SCRIPT_DIR, "overleaf_summary", "n_agents_sweep_comparison.json")

_BASELINE_RULES_PATH = os.path.join(SCRIPT_DIR, "lj_baseline", "paper_baseline_rules.json")
NO_CLAMP = dict(HEBBIAN_SAFETY_CLAMP_ENABLED=False, HEBBIAN_MIN_DIST_INFLATION=1.0,
                HEBBIAN_RESOLVE_COLLISIONS=False, SPAWN_SQUARE_SIZE=TIGHT_SPAWN)

CONDITIONS = {
    "lj_baseline": (None, "#77AC30"),
    "upwind_2stage_plain_seed123": (
        os.path.join(SCRIPT_DIR, "upwind_2stage_plain_seed123", "genome_trained_n20.npy"), "#0072BD"),
    "upwind_2stage_plain_seed888": (
        os.path.join(SCRIPT_DIR, "upwind_2stage_plain_seed888", "genome_trained_n20.npy"), "#7E2F8E"),
    "upwind_2stage_drain0_seed123": (
        os.path.join(SCRIPT_DIR, "upwind_2stage_drain0_seed123", "genome_trained_n20.npy"), "#D95319"),
}


def _run_scalar(label, genome_path, n_agents, seed):
    if genome_path is None:
        rules = json.load(open(_BASELINE_RULES_PATH))
        with _ConfigOverride(HEBBIAN_NX=WIND_GRID, HEBBIAN_NY=WIND_GRID, SPAWN_SQUARE_SIZE=TIGHT_SPAWN):
            _, dist, batt, _ = simulate_lj_baseline(rules=rules, seed=seed, n_agents=n_agents)
        return dist, batt / config.MAX_BATTERY * 100.0

    genome = np.load(genome_path)
    rules = unflatten_abcd(genome)
    with _ConfigOverride(**NO_CLAMP):
        dist, batt, _, _, _, _ = simulate_hebbian_episode(
            rules, seed=seed, n_agents=n_agents, wind_enabled=True, nx=WIND_GRID, ny=WIND_GRID)
    return dist, batt


def _run_trajectory(label, genome_path, n_agents, seed):
    if genome_path is None:
        rules = json.load(open(_BASELINE_RULES_PATH))
        with _ConfigOverride(HEBBIAN_NX=WIND_GRID, HEBBIAN_NY=WIND_GRID, SPAWN_SQUARE_SIZE=TIGHT_SPAWN):
            _, _, _, _, tele = simulate_lj_baseline(rules=rules, seed=seed, n_agents=n_agents,
                                                     record_trajectory=True)
        return tele["positions"]

    genome = np.load(genome_path)
    rules = unflatten_abcd(genome)
    with _ConfigOverride(**NO_CLAMP):
        _, _, _, _, _, _, tele = simulate_hebbian_episode(
            rules, seed=seed, n_agents=n_agents, wind_enabled=True, nx=WIND_GRID, ny=WIND_GRID,
            record_trajectory=True)
    return tele["positions"]


def _run_battery_trace(label, genome_path, n_agents, seed):
    """Final per-agent battery array (raw scale for LJ, 0-100 for Hebbian -- caller normalizes)."""
    if genome_path is None:
        rules = json.load(open(_BASELINE_RULES_PATH))
        with _ConfigOverride(HEBBIAN_NX=WIND_GRID, HEBBIAN_NY=WIND_GRID, SPAWN_SQUARE_SIZE=TIGHT_SPAWN):
            _, _, _, _, tele = simulate_lj_baseline(rules=rules, seed=seed, n_agents=n_agents,
                                                     record_battery=True)
        return tele["battery"][-1] / config.MAX_BATTERY * 100.0

    genome = np.load(genome_path)
    rules = unflatten_abcd(genome)
    with _ConfigOverride(**NO_CLAMP):
        _, _, _, _, _, _, tele = simulate_hebbian_episode(
            rules, seed=seed, n_agents=n_agents, wind_enabled=True, nx=WIND_GRID, ny=WIND_GRID,
            record_battery=True)
    return tele["battery"][-1]


def main():
    results = {label: {} for label in CONDITIONS}

    print("=== 30-seed statistics + both-beat rate, tight spawn, standard physics ===")
    for n_agents in N_AGENTS_SWEEP:
        per_label_db = {}
        for label, (genome_path, _color) in CONDITIONS.items():
            dists, batts = [], []
            for seed in SEEDS:
                d, b = _run_scalar(label, genome_path, n_agents, seed)
                dists.append(d)
                batts.append(b)
            per_label_db[label] = (np.array(dists), np.array(batts))

        lj_d, lj_b = per_label_db["lj_baseline"]
        for label, (dists, batts) in per_label_db.items():
            both_beat = None
            if label != "lj_baseline":
                both_beat = int(np.sum((dists >= lj_d) & (batts >= lj_b)))
            results[label][n_agents] = {
                "dist_mean": float(dists.mean()), "dist_std": float(dists.std()),
                "batt_mean": float(batts.mean()), "batt_std": float(batts.std()),
                "both_beat": both_beat,
            }
            beat_str = f"  both-beat={both_beat}/{len(SEEDS)}" if both_beat is not None else ""
            print(f"  n={n_agents:2d} {label:32s} dist={dists.mean():6.2f}+-{dists.std():5.2f} "
                  f"batt={batts.mean():6.2f}+-{batts.std():5.2f}%{beat_str}")

    print("\n=== single-seed (42) position-change + battery-equity metrics ===")
    for n_agents in N_AGENTS_SWEEP:
        for label, (genome_path, _color) in CONDITIONS.items():
            batt_final = _run_battery_trace(label, genome_path, n_agents, SEED_SINGLE)
            results[label][n_agents]["battery_std_final"] = float(np.std(batt_final))

            if n_agents < 2:
                results[label][n_agents]["exchange_rate_per_m"] = None
                continue
            positions = _run_trajectory(label, genome_path, n_agents, SEED_SINGLE)
            ranks, _bearing = _front_rank_series(positions)
            occ = occupancy_and_exchange(ranks, config.DT, positions)
            results[label][n_agents]["exchange_rate_per_m"] = float(occ["exchange_rate_per_m"])
            print(f"  n={n_agents:2d} {label:32s} exch/m={occ['exchange_rate_per_m']:.4f} "
                  f"batt_std={np.std(batt_final):.2f}")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT_JSON}")

    os.makedirs(OUT_DIR_FIG, exist_ok=True)

    # --- Figure 1: distance & battery vs. n_agents ---
    fig, (ax_d, ax_b) = plt.subplots(1, 2, figsize=(13, 5.5))
    ns = np.array(N_AGENTS_SWEEP)
    for label, (_gp, color) in CONDITIONS.items():
        dmean = np.array([results[label][n]["dist_mean"] for n in ns])
        dstd = np.array([results[label][n]["dist_std"] for n in ns])
        bmean = np.array([results[label][n]["batt_mean"] for n in ns])
        bstd = np.array([results[label][n]["batt_std"] for n in ns])
        ax_d.plot(ns, dmean, "o-", color=color, label=label)
        ax_d.fill_between(ns, dmean - dstd, dmean + dstd, color=color, alpha=0.15)
        ax_b.plot(ns, bmean, "o-", color=color, label=label)
        ax_b.fill_between(ns, bmean - bstd, bmean + bstd, color=color, alpha=0.15)
    ax_d.set_xlabel("n_agents"); ax_d.set_ylabel("Distance travelled [m]")
    ax_d.set_title("Distance vs. swarm size (mean ± 1 std, 30 seeds)")
    ax_d.set_xticks(ns); ax_d.grid(True, linestyle=":", alpha=0.6); ax_d.legend(fontsize=8)
    ax_b.set_xlabel("n_agents"); ax_b.set_ylabel("Remaining battery [% of full charge]")
    ax_b.set_title("Battery vs. swarm size (mean ± 1 std, 30 seeds)")
    ax_b.set_xticks(ns); ax_b.grid(True, linestyle=":", alpha=0.6); ax_b.legend(fontsize=8)
    fig.tight_layout()
    stem = os.path.join(OUT_DIR_FIG, "n_agents_sweep_distance_battery")
    fig.savefig(stem + ".png", dpi=150); fig.savefig(stem + ".pdf"); fig.savefig(stem + ".svg")
    plt.close(fig)
    print(f"Saved {stem}.{{png,pdf,svg}}")

    # --- Figure 2: position-change (exchange rate per metre) vs. n_agents ---
    ns2 = np.array([n for n in N_AGENTS_SWEEP if n >= 2])
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, (_gp, color) in CONDITIONS.items():
        y = np.array([results[label][n]["exchange_rate_per_m"] for n in ns2])
        ax.plot(ns2, y, "o-", color=color, label=label)
    ax.set_xlabel("n_agents"); ax.set_ylabel("Front-position exchange rate [switches / m]")
    ax.set_title("Position-change (front-rank hand-off) rate vs. swarm size\n(seed 42, single run per condition)")
    ax.set_xticks(ns2); ax.grid(True, linestyle=":", alpha=0.6); ax.legend(fontsize=8)
    fig.tight_layout()
    stem = os.path.join(OUT_DIR_FIG, "n_agents_sweep_position_change")
    fig.savefig(stem + ".png", dpi=150); fig.savefig(stem + ".pdf"); fig.savefig(stem + ".svg")
    plt.close(fig)
    print(f"Saved {stem}.{{png,pdf,svg}}")

    # --- Figure 3 (suggested addition): battery equity (final per-agent std) vs. n_agents ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, (_gp, color) in CONDITIONS.items():
        y = np.array([results[label][n]["battery_std_final"] for n in ns])
        ax.plot(ns, y, "o-", color=color, label=label)
    ax.set_xlabel("n_agents"); ax.set_ylabel("Final battery std across agents [pp]")
    ax.set_title("Battery-sharing equity vs. swarm size\n(lower = more even charge distribution; seed 42)")
    ax.set_xticks(ns); ax.grid(True, linestyle=":", alpha=0.6); ax.legend(fontsize=8)
    fig.tight_layout()
    stem = os.path.join(OUT_DIR_FIG, "n_agents_sweep_battery_equity")
    fig.savefig(stem + ".png", dpi=150); fig.savefig(stem + ".pdf"); fig.savefig(stem + ".svg")
    plt.close(fig)
    print(f"Saved {stem}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
