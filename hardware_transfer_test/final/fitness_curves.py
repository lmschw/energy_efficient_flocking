"""CMA-ES training curves (best efficiency per generation) for the two final genomes,
plain_seed123 (unclamped) and plain_seed123_clamped, one panel per curriculum stage.

Built from the per-stage *_history.json files the training run saved, which record only the
best candidate's fitness per generation (`loss_curve`, negated efficiency) -- per-generation
mean fitness was drawn live into hebbian_fitness_curve.png during training but never saved as
data, so it can't be reproduced here.

Efficiency is each stage's own reward, so the two panels' y-axes are NOT comparable to each
other. The marked point is the best-ever candidate CMA-ES returned (es.result), which is the
genome that was saved and deployed -- not the last generation.

Usage: python fitness_curves.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS = os.path.join(REPO_ROOT, "results")
OUT_DIR_FIG = os.path.join(SCRIPT_DIR, "overleaf_summary", "figures")

CONDITIONS = {
    "plain_seed123 (unclamped)": (
        os.path.join(RESULTS, "hebbian_results_v2_2stage_upwind", "n20_seed123"), "#D95319"),
    "plain_seed123 (clamped)": (
        os.path.join(RESULTS, "hebbian_results_v2_2stage_upwind_clamped", "n20_seed123"), "#0072BD"),
}
STAGES = (  # (stage, panel title, legend location -- wherever the curves leave room)
    ("walk_upwind", "Stage 1: walk upwind (wind on)", "lower right"),
    ("save_battery_avoid_all", "Stage 2: save battery, avoid all", "lower left"),
)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (stage, stage_title, legend_loc) in zip(axes, STAGES):
        for label, (run_dir, color) in CONDITIONS.items():
            hist = json.load(open(os.path.join(run_dir, f"hebbian_{stage}_history.json")))
            eff = -np.asarray(hist["loss_curve"])
            gens = np.arange(1, len(eff) + 1)
            best_gen = int(np.argmax(eff)) + 1
            ax.plot(gens, eff, "-", color=color, lw=1.4, label=label)
            ax.plot(best_gen, eff[best_gen - 1], "*", color=color, ms=14, mec="k", mew=0.6,
                    label=f"  saved genome: gen {best_gen}, eff {eff[best_gen - 1]:.1f}")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best efficiency in generation (higher = better)")
        ax.set_title(stage_title)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8, loc=legend_loc)
    fig.suptitle("CMA-ES training progress, n=20, training seed 123 "
                 "(per-stage reward; panels not comparable)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(OUT_DIR_FIG, exist_ok=True)
    stem = os.path.join(OUT_DIR_FIG, "fitness_curves_plain_seed123")
    fig.savefig(stem + ".png", dpi=150); fig.savefig(stem + ".pdf"); fig.savefig(stem + ".svg")
    plt.close(fig)
    print(f"Saved {stem}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
