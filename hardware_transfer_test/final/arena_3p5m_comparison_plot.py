"""Builds the comparison figure for the 3.5 m-wide-arena simulation sweep
(arena_3p5m_sweep.py) against the existing 10 m x 10 m arena sim results
(plain_seed123/, plain_seed123_clamped/, lj_baseline/) and the hardware aggregates
(hardware_aggregate.py), for n_agents in {5, 10}. Two panels: net displacement/distance,
and wall-zone occupancy as a fraction of agent-time (the two quantities that are measured
comparably, if not identically, across sim and hardware -- see arena_3p5m_sweep.py's and
hardware_aggregate.py's docstrings for the exact definitions/caveats).

Usage: python arena_3p5m_comparison_plot.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = ("plain_seed123", "plain_seed123_clamped", "lj_baseline")
LABELS = {"plain_seed123": "unclamped", "plain_seed123_clamped": "clamped", "lj_baseline": "LJ baseline"}
COLORS = {"plain_seed123": "#D95319", "plain_seed123_clamped": "#0072BD", "lj_baseline": "#77AC30"}
DT = 0.5


def load_existing_sim_metrics(condition, n_agents):
    path = os.path.join(SCRIPT_DIR, condition, f"n{n_agents}", "metrics.json")
    return json.load(open(path))


def main():
    sweep = json.load(open(os.path.join(SCRIPT_DIR, "arena_3p5m_sweep", "summary.json")))["results"]
    hw = json.load(open(os.path.join(SCRIPT_DIR, "hardware_aggregate.json")))

    ns = (5, 10)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax_dist, ax_wall = axes
    width = 0.22
    x = np.arange(len(ns))

    for i, condition in enumerate(CONDITIONS):
        offset = (i - 1) * width
        dist_10m = [load_existing_sim_metrics(condition, n)["dist"] for n in ns]
        dist_35m = [sweep[condition][str(n)]["dist"] for n in ns]
        hw_disp = [hw[str(n)][condition]["net_disp_mean"] if hw[str(n)].get(condition) else np.nan for n in ns]
        hw_disp_err = [hw[str(n)][condition]["net_disp_std"] if hw[str(n)].get(condition) else 0 for n in ns]

        ax_dist.bar(x + offset - width / 3, dist_10m, width / 3, color=COLORS[condition], alpha=0.35,
                    edgecolor=COLORS[condition])
        ax_dist.bar(x + offset, dist_35m, width / 3, color=COLORS[condition], alpha=0.75,
                    edgecolor=COLORS[condition])
        ax2 = ax_dist.twinx() if i == 0 and False else None  # placeholder, hw plotted separately below

        wall_10m = []
        wall_35m = []
        for n in ns:
            m10 = load_existing_sim_metrics(condition, n)
            m35 = sweep[condition][str(n)]
            n_steps10, wct10 = m10.get("n_steps"), m10.get("wall_collision_time")
            n_steps35, wct35 = m35.get("n_steps"), m35.get("wall_collision_time")
            wall_10m.append(wct10 / (n_steps10 * DT * n) if (wct10 is not None and n_steps10) else 0.0)
            wall_35m.append(wct35 / (n_steps35 * DT * n) if (wct35 is not None and n_steps35) else 0.0)
        hw_wall = [hw[str(n)][condition]["wall_zone_frac_mean"] if hw[str(n)].get(condition) else np.nan for n in ns]

        ax_wall.bar(x + offset - width / 3, wall_10m, width / 3, color=COLORS[condition], alpha=0.35,
                    edgecolor=COLORS[condition])
        ax_wall.bar(x + offset, wall_35m, width / 3, color=COLORS[condition], alpha=0.75,
                    edgecolor=COLORS[condition])
        ax_wall.bar(x + offset + width / 3, hw_wall, width / 3, color=COLORS[condition], alpha=1.0,
                    edgecolor="k", hatch="//")

        ax_dist.bar(x + offset + width / 3, hw_disp, width / 3, yerr=hw_disp_err, color=COLORS[condition],
                    alpha=1.0, edgecolor="k", hatch="//", capsize=2)

    ax_dist.set_xticks(x, [f"n={n}" for n in ns])
    ax_dist.set_ylabel("Distance / net displacement [m]")
    ax_dist.set_title("Distance: sim (10m arena, light) vs sim (3.5m arena, solid)\nvs hardware net displacement (hatched)")
    ax_dist.grid(True, axis="y", ls=":", alpha=0.5)

    ax_wall.set_xticks(x, [f"n={n}" for n in ns])
    ax_wall.set_ylabel("Wall-zone occupancy [fraction of agent-time]")
    ax_wall.set_title("Wall proximity: sim (10m, light) vs sim (3.5m, solid)\nvs hardware brake-zone (hatched)")
    ax_wall.grid(True, axis="y", ls=":", alpha=0.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c], alpha=0.75) for c in CONDITIONS]
    fig.legend(handles, [LABELS[c] for c in CONDITIONS], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()

    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(SCRIPT_DIR, "arena_3p5m_sweep", f"comparison.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.join(SCRIPT_DIR, 'arena_3p5m_sweep', 'comparison.{png,pdf,svg}')}")


if __name__ == "__main__":
    main()
