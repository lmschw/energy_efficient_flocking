"""Aggregates the raw per-tick hardware CSVs in hardware_results/n={5,10,15}/eef_*/ into
per-condition summary statistics (path length, net displacement, mean |v|, wall-zone
occupancy, tracked fraction, final battery), the same descriptive quantities the n=5 write-up
(hardware_transfer_test/final/HARDWARE_VALIDATION_FINDINGS.md,
overleaf_summary/swarm_intelligence_revised.tex sec. on hardware validation) reports for the
clamped/unclamped n=5 runs, computed fresh here so n=10 and n=15 (added in later commits, never
aggregated before -- only raw CSVs exist for those) can be compared on the same basis.

Each run CSV is long-format: one row per (robot, tick), hostname identifies the robot.
Untracked ticks are marked with the sentinel x=y=10000.0 (confirmed by inspection -- no other
values anywhere near that range). "wall zone" = within 0.5 m of the tracked corridor's y bounds
(CORRIDOR_Y_MIN=-1.70, CORRIDOR_Y_MAX=1.40, ~3.1 m usable width within the 3.5 m physical
walls -- see HARDWARE_VALIDATION_FINDINGS.md), matching the brake-zone definition already used
for the n=5 numbers.

Usage: python hardware_aggregate.py
"""
import csv
import glob
import json
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
HW_ROOT = os.path.join(REPO_ROOT, "hardware_results")

CORRIDOR_Y_MIN, CORRIDOR_Y_MAX = -1.70, 1.40
BRAKE_ZONE = 0.5
SENTINEL = 10000.0

DATASETS = {
    5: {"lj_baseline": "eef_lj_baseline_n=5", "plain_seed123_clamped": "eef_clamped_n=5",
        "plain_seed123": "eef_plain_n=5"},
    10: {"lj_baseline": "eef_lj_baseline_n=10", "plain_seed123_clamped": "eef_clamped_n=10",
         "plain_seed123": "eef_unclamped_n=10"},
    15: {"plain_seed123": "eef_unclamped_n=15"},
}


def _find_csvs(n_agents, dirname):
    base = os.path.join(HW_ROOT, f"n={n_agents}", dirname)
    for sub in (base, os.path.join(base, "compilation")):
        files = sorted(glob.glob(os.path.join(sub, "*.csv")))
        if files:
            return files
    return []


def _load_run(path):
    """Returns dict hostname -> list of (tick, x, y, battery, v) rows, tracked rows only
    filtered out for path/displacement math but kept (as NaN position) for tracked-fraction."""
    by_host = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            host = row["hostname"]
            by_host.setdefault(host, []).append(row)
    return by_host


def _robot_stats(rows):
    """rows: list of csv row dicts for one robot in one run, in tick order."""
    xs, ys, tracked = [], [], []
    battery = []
    speeds = []
    for row in rows:
        x, y = float(row["x"]), float(row["y"])
        is_tracked = not (x == SENTINEL and y == SENTINEL)
        tracked.append(is_tracked)
        xs.append(x if is_tracked else np.nan)
        ys.append(y if is_tracked else np.nan)
        battery.append(float(row["battery"]))
        speeds.append(abs(float(row["v"])))

    xs, ys = np.array(xs), np.array(ys)
    tracked = np.array(tracked)
    tracked_frac = tracked.mean() if len(tracked) else 0.0

    valid_idx = np.where(tracked)[0]
    # Path length only sums CONSECUTIVE tracked ticks (i, i+1 both tracked) -- a dropout
    # followed by reacquisition elsewhere is a tracking gap, not real robot motion, so it must
    # not contribute a straight-line "jump" distance.
    path_len = 0.0
    for i in range(len(tracked) - 1):
        if tracked[i] and tracked[i + 1]:
            path_len += float(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]))
    net_disp = float(np.hypot(xs[valid_idx[-1]] - xs[valid_idx[0]], ys[valid_idx[-1]] - ys[valid_idx[0]])) \
        if len(valid_idx) >= 2 else 0.0

    wall_zone_hits = 0
    for i in valid_idx:
        if ys[i] < CORRIDOR_Y_MIN + BRAKE_ZONE or ys[i] > CORRIDOR_Y_MAX - BRAKE_ZONE:
            wall_zone_hits += 1
    wall_zone_frac = wall_zone_hits / len(valid_idx) if len(valid_idx) else 0.0

    final_battery = battery[-1] if battery else np.nan
    mean_speed = float(np.mean([s for s, t in zip(speeds, tracked) if t])) if tracked.any() else 0.0

    return dict(tracked_frac=tracked_frac, path_len=path_len, net_disp=net_disp,
                wall_zone_frac=wall_zone_frac, final_battery=final_battery, mean_speed=mean_speed,
                n_ticks=len(rows))


def aggregate(n_agents, dirname):
    csvs = _find_csvs(n_agents, dirname)
    if not csvs:
        return None
    per_robot_runs = []
    for path in csvs:
        by_host = _load_run(path)
        for host, rows in by_host.items():
            per_robot_runs.append(_robot_stats(rows))

    def col(key):
        return np.array([r[key] for r in per_robot_runs])

    return {
        "n_runs": len(csvs), "n_robot_runs": len(per_robot_runs),
        "tracked_frac_mean": float(col("tracked_frac").mean()),
        "path_len_mean": float(col("path_len").mean()), "path_len_std": float(col("path_len").std()),
        "net_disp_mean": float(col("net_disp").mean()), "net_disp_std": float(col("net_disp").std()),
        "wall_zone_frac_mean": float(col("wall_zone_frac").mean()),
        "final_battery_mean": float(np.nanmean(col("final_battery"))),
        "final_battery_std": float(np.nanstd(col("final_battery"))),
        "mean_speed_mean": float(col("mean_speed").mean()),
    }


def main():
    out = {}
    for n_agents, conditions in DATASETS.items():
        out[str(n_agents)] = {}
        for condition, dirname in conditions.items():
            stats = aggregate(n_agents, dirname)
            out[str(n_agents)][condition] = stats
            if stats is None:
                print(f"n={n_agents:2d} {condition:24s}: NO DATA FOUND ({dirname})")
                continue
            print(f"n={n_agents:2d} {condition:24s}: runs={stats['n_runs']:2d} "
                  f"tracked={stats['tracked_frac_mean']:.3f} "
                  f"path_len={stats['path_len_mean']:5.2f}+-{stats['path_len_std']:.2f}m "
                  f"net_disp={stats['net_disp_mean']:5.2f}+-{stats['net_disp_std']:.2f}m "
                  f"batt_final={stats['final_battery_mean']:5.1f}+-{stats['final_battery_std']:.1f}% "
                  f"wall_zone={stats['wall_zone_frac_mean']:.3f}")

    out_path = os.path.join(SCRIPT_DIR, "hardware_aggregate.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
