"""Standalone illustrative figure for the safety-clamp scale function C(g) used by
_apply_safety_clamp() in ants26_replication/experiment/simulation_hebbian.py -- no simulation
involved, just plots the piecewise-linear formula itself for the two gap types (agent-agent,
agent-wall) using their actual config.py threshold values."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "overleaf_summary", "figures")

AGENT_INNER, AGENT_OUTER = 0.05, 0.30
WALL_INNER, WALL_OUTER = 0.02, 0.15


def clamp_scale(gap, inner, outer):
    return np.clip((gap - inner) / (outer - inner), 0.0, 1.0)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    g = np.linspace(0, 0.35, 500)
    ax.plot(g, clamp_scale(g, AGENT_INNER, AGENT_OUTER), color="#0072BD", linewidth=2,
             label=f"agent-agent gap ($g_\\mathrm{{in}}$={AGENT_INNER}m, $g_\\mathrm{{out}}$={AGENT_OUTER}m)")
    ax.plot(g, clamp_scale(g, WALL_INNER, WALL_OUTER), color="#D95319", linewidth=2, linestyle="--",
             label=f"agent-wall gap ($g_\\mathrm{{in}}$={WALL_INNER}m, $g_\\mathrm{{out}}$={WALL_OUTER}m)")

    for x, c in [(AGENT_INNER, "#0072BD"), (AGENT_OUTER, "#0072BD"),
                 (WALL_INNER, "#D95319"), (WALL_OUTER, "#D95319")]:
        ax.axvline(x, color=c, linewidth=0.7, linestyle=":", alpha=0.6)

    ax.set_xlabel("surface gap $g$ [m]")
    ax.set_ylabel(r"speed scale $C(g)$")
    ax.set_title(r"Safety-clamp forward-speed scale function $C(g)$")
    ax.set_xlim(0, 0.35)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    stem = os.path.join(OUTPUT_DIR, "safety_clamp_scale_function")
    fig.savefig(stem + ".png", dpi=150)
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".svg")
    plt.close(fig)
    print(f"Saved {stem}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
