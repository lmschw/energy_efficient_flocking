"""Isolated fork of n_agents_sweep_analysis.py for the FINAL chosen genome: plain_seed123
(unclamped) and plain_seed123_clamped (retrained from scratch with the hard safety clamp),
vs. the LJ baseline, across n_agents in {1,2,3,4,5,7,10,20}. Kept separate from the original
script (which covered all three original winning genomes) per the user's request to isolate
this genome's analysis.

Unlike the original script, each condition now carries its OWN clamp config (plain_seed123
unclamped; plain_seed123_clamped WITH the clamp/inflation/resolve-collisions settings it was
actually trained and 30-seed-evaluated with) rather than a single shared NO_CLAMP dict.

Default (3.0m) spawn square throughout, standard (real, non-refunded) drain physics --
matching every other statistical comparison already reported for these two genomes.

Usage: python n_agents_sweep_analysis_plain_seed123.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # .../hardware_transfer_test/final -> repo root
EXPERIMENT_DIR = os.path.join(REPO_ROOT, "ants26_replication", "experiment")
sys.path.insert(0, EXPERIMENT_DIR)

import config  # noqa: E402
from hebbian_controller import unflatten_abcd  # noqa: E402
from simulation_hebbian import simulate_hebbian_episode  # noqa: E402
from lj_baseline import simulate_lj_baseline  # noqa: E402
from leadership_metrics import _ConfigOverride, WIND_GRID  # noqa: E402
from leadership_metrics import _front_rank_series, occupancy_and_exchange  # noqa: E402

N_AGENTS_SWEEP = tuple(range(1, 21))
SEEDS = list(range(1000, 1030))
SEED_SINGLE = config.HEBBIAN_DEFAULT_SEED  # 42
OUT_DIR_FIG = os.path.join(SCRIPT_DIR, "overleaf_summary", "figures")
OUT_JSON = os.path.join(SCRIPT_DIR, "overleaf_summary", "n_agents_sweep_comparison_plain_seed123.json")

_BASELINE_RULES_PATH = os.path.join(SCRIPT_DIR, "lj_baseline", "paper_baseline_rules.json")

NO_CLAMP = dict(HEBBIAN_SAFETY_CLAMP_ENABLED=False, HEBBIAN_MIN_DIST_INFLATION=1.0,
                HEBBIAN_RESOLVE_COLLISIONS=False)
WITH_CLAMP = dict(HEBBIAN_SAFETY_CLAMP_ENABLED=True, HEBBIAN_MIN_DIST_INFLATION=1.3,
                   HEBBIAN_RESOLVE_COLLISIONS=True, HEBBIAN_RESOLVE_COLLISIONS_STRENGTH=0.5,
                   HEBBIAN_RESOLVE_COLLISIONS_MAX_ITER=2)

CONDITIONS = {
    "lj_baseline": (None, None, "#77AC30"),
    "plain_seed123": (
        os.path.join(SCRIPT_DIR, "plain_seed123", "genome_trained_n20.npy"),
        NO_CLAMP, "#D95319"),
    "plain_seed123_clamped": (
        os.path.join(SCRIPT_DIR, "plain_seed123_clamped", "genome_trained_n20.npy"),
        WITH_CLAMP, "#0072BD"),
}


def _run_scalar(genome_path, cfg, n_agents, seed):
    if genome_path is None:
        rules = json.load(open(_BASELINE_RULES_PATH))
        with _ConfigOverride(HEBBIAN_NX=WIND_GRID, HEBBIAN_NY=WIND_GRID):
            _, dist, batt, _ = simulate_lj_baseline(rules=rules, seed=seed, n_agents=n_agents)
        return dist, batt / config.MAX_BATTERY * 100.0

    genome = np.load(genome_path)
    rules = unflatten_abcd(genome)
    with _ConfigOverride(**cfg):
        dist, batt, _, _, _, _ = simulate_hebbian_episode(
            rules, seed=seed, n_agents=n_agents, wind_enabled=True, nx=WIND_GRID, ny=WIND_GRID)
    return dist, batt


def _run_trajectory(genome_path, cfg, n_agents, seed):
    if genome_path is None:
        rules = json.load(open(_BASELINE_RULES_PATH))
        with _ConfigOverride(HEBBIAN_NX=WIND_GRID, HEBBIAN_NY=WIND_GRID):
            _, _, _, _, tele = simulate_lj_baseline(rules=rules, seed=seed, n_agents=n_agents,
                                                     record_trajectory=True)
        return tele["positions"]

    genome = np.load(genome_path)
    rules = unflatten_abcd(genome)
    with _ConfigOverride(**cfg):
        _, _, _, _, _, _, tele = simulate_hebbian_episode(
            rules, seed=seed, n_agents=n_agents, wind_enabled=True, nx=WIND_GRID, ny=WIND_GRID,
            record_trajectory=True)
    return tele["positions"]


def _run_battery_trace(genome_path, cfg, n_agents, seed):
    if genome_path is None:
        rules = json.load(open(_BASELINE_RULES_PATH))
        with _ConfigOverride(HEBBIAN_NX=WIND_GRID, HEBBIAN_NY=WIND_GRID):
            _, _, _, _, tele = simulate_lj_baseline(rules=rules, seed=seed, n_agents=n_agents,
                                                     record_battery=True)
        return tele["battery"][-1] / config.MAX_BATTERY * 100.0

    genome = np.load(genome_path)
    rules = unflatten_abcd(genome)
    with _ConfigOverride(**cfg):
        _, _, _, _, _, _, tele = simulate_hebbian_episode(
            rules, seed=seed, n_agents=n_agents, wind_enabled=True, nx=WIND_GRID, ny=WIND_GRID,
            record_battery=True)
    return tele["battery"][-1]


def _load_existing():
    """Reuse already-computed n_agents entries from OUT_JSON (runs are deterministic per seed),
    so extending N_AGENTS_SWEEP only simulates the new swarm sizes."""
    results = {label: {} for label in CONDITIONS}
    if os.path.exists(OUT_JSON):
        for label, per_n in json.load(open(OUT_JSON)).items():
            if label in results:
                results[label] = {int(n): v for n, v in per_n.items()}
    return results


def main():
    results = _load_existing()
    todo = [n for n in N_AGENTS_SWEEP
            if not all(n in results[label] and "battery_std_final" in results[label][n]
                       for label in CONDITIONS)]
    print(f"Reusing n_agents={sorted(set(N_AGENTS_SWEEP) - set(todo))}; simulating {todo}")

    print("=== 30-seed statistics + both-beat rate, default spawn, standard physics ===")
    for n_agents in todo:
        per_label_db = {}
        for label, (genome_path, cfg, _color) in CONDITIONS.items():
            dists, batts = [], []
            for seed in SEEDS:
                d, b = _run_scalar(genome_path, cfg, n_agents, seed)
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
            print(f"  n={n_agents:2d} {label:24s} dist={dists.mean():6.2f}+-{dists.std():5.2f} "
                  f"batt={batts.mean():6.2f}+-{batts.std():5.2f}%{beat_str}")

    print("\n=== single-seed (42) position-change + battery-equity metrics ===")
    for n_agents in todo:
        for label, (genome_path, cfg, _color) in CONDITIONS.items():
            batt_final = _run_battery_trace(genome_path, cfg, n_agents, SEED_SINGLE)
            results[label][n_agents]["battery_std_final"] = float(np.std(batt_final))

            if n_agents < 2:
                results[label][n_agents]["exchange_rate_per_m"] = None
                continue
            positions = _run_trajectory(genome_path, cfg, n_agents, SEED_SINGLE)
            ranks, _bearing = _front_rank_series(positions)
            occ = occupancy_and_exchange(ranks, config.DT, positions)
            results[label][n_agents]["exchange_rate_per_m"] = float(occ["exchange_rate_per_m"])
            print(f"  n={n_agents:2d} {label:24s} exch/m={occ['exchange_rate_per_m']:.4f} "
                  f"batt_std={np.std(batt_final):.2f}")

    with open(OUT_JSON, "w") as f:
        json.dump({label: {n: per_n[n] for n in sorted(per_n)} for label, per_n in results.items()},
                  f, indent=2)
    print(f"\nSaved {OUT_JSON}")

    os.makedirs(OUT_DIR_FIG, exist_ok=True)
    ns = np.array(N_AGENTS_SWEEP)

    # --- Figure 1: distance & battery vs. n_agents ---
    fig, (ax_d, ax_b) = plt.subplots(1, 2, figsize=(13, 5.5))
    for label, (_gp, _cfg, color) in CONDITIONS.items():
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
    ax_d.set_xticks(ns); ax_d.grid(True, linestyle=":", alpha=0.6); ax_d.legend(fontsize=9)
    ax_b.set_xlabel("n_agents"); ax_b.set_ylabel("Remaining battery [% of full charge]")
    ax_b.set_title("Battery vs. swarm size (mean ± 1 std, 30 seeds)")
    ax_b.set_xticks(ns); ax_b.grid(True, linestyle=":", alpha=0.6); ax_b.legend(fontsize=9)
    fig.tight_layout()
    stem = os.path.join(OUT_DIR_FIG, "n_agents_sweep_distance_battery_plain_seed123")
    fig.savefig(stem + ".png", dpi=150); fig.savefig(stem + ".pdf"); fig.savefig(stem + ".svg")
    plt.close(fig)
    print(f"Saved {stem}.{{png,pdf,svg}}")

    # --- Figure 2: position-change (exchange rate per metre) vs. n_agents ---
    ns2 = np.array([n for n in N_AGENTS_SWEEP if n >= 2])
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, (_gp, _cfg, color) in CONDITIONS.items():
        y = np.array([results[label][n]["exchange_rate_per_m"] for n in ns2])
        ax.plot(ns2, y, "o-", color=color, label=label)
    ax.set_xlabel("n_agents"); ax.set_ylabel("Front-position exchange rate [switches / m]")
    ax.set_title("Position-change (front-rank hand-off) rate vs. swarm size\n(seed 42, single run per condition)")
    ax.set_xticks(ns2); ax.grid(True, linestyle=":", alpha=0.6); ax.legend(fontsize=9)
    fig.tight_layout()
    stem = os.path.join(OUT_DIR_FIG, "n_agents_sweep_position_change_plain_seed123")
    fig.savefig(stem + ".png", dpi=150); fig.savefig(stem + ".pdf"); fig.savefig(stem + ".svg")
    plt.close(fig)
    print(f"Saved {stem}.{{png,pdf,svg}}")

    # --- Figure 3: battery equity (final per-agent std) vs. n_agents ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, (_gp, _cfg, color) in CONDITIONS.items():
        y = np.array([results[label][n]["battery_std_final"] for n in ns])
        ax.plot(ns, y, "o-", color=color, label=label)
    ax.set_xlabel("n_agents"); ax.set_ylabel("Final battery std across agents [pp]")
    ax.set_title("Battery-sharing equity vs. swarm size\n(lower = more even charge distribution; seed 42)")
    ax.set_xticks(ns); ax.grid(True, linestyle=":", alpha=0.6); ax.legend(fontsize=9)
    fig.tight_layout()
    stem = os.path.join(OUT_DIR_FIG, "n_agents_sweep_battery_equity_plain_seed123")
    fig.savefig(stem + ".png", dpi=150); fig.savefig(stem + ".pdf"); fig.savefig(stem + ".svg")
    plt.close(fig)
    print(f"Saved {stem}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
