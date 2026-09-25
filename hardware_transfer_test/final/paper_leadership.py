"""Leadership figure + summary table for the paper draft (swarm_intelligence.tex), n=20.

Reads the 30-seed tables that leadership_metrics.py already produced
(leadership_analysis/seed_sweep_table_n{10,20}.json) -- no re-simulation for the statistics --
and re-runs seed 42 only for the example front-position timelines.

Outputs overleaf_summary/figures/paper_leadership_n20.{png,pdf,svg} and
overleaf_summary/paper_leadership_summary.json (median, IQR and Mann-Whitney U p-value
against the baseline for every metric, at n=10 and n=20).

Usage: python paper_leadership.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from leadership_metrics import SEED, analyze

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLE_DIR = os.path.join(SCRIPT_DIR, "leadership_analysis")
OUT_DIR_FIG = os.path.join(SCRIPT_DIR, "overleaf_summary", "figures")
OUT_JSON = os.path.join(SCRIPT_DIR, "overleaf_summary", "paper_leadership_summary.json")

CONDITIONS = {  # condition id -> (paper label, colour)
    "plain_seed123": ("Stage 2", "#D95319"),
    "plain_seed123_clamped": ("Stage 2, clamped", "#0072BD"),
    "lj_baseline": ("Baseline", "#77AC30"),
}
METRICS = ("reciprocity_r", "front_occupancy_normalized_entropy", "exchange_rate_per_sec",
           "persistence_filtered_rate_per_sec", "real_switches", "raw_switches",
           "hierarchy_leadership_entropy", "hierarchy_steepness")


def _table(n):
    rows = json.load(open(os.path.join(TABLE_DIR, f"seed_sweep_table_n{n}.json")))
    return {c: {m: np.array([r[m] for r in rows if r["condition"] == c], float) for m in METRICS}
            for c in CONDITIONS}


def _summary(tab):
    out = {}
    for c in CONDITIONS:
        out[c] = {}
        vals = dict(tab[c])
        vals["persist_share"] = tab[c]["real_switches"] / tab[c]["raw_switches"]
        base = dict(tab["lj_baseline"])
        base["persist_share"] = tab["lj_baseline"]["real_switches"] / tab["lj_baseline"]["raw_switches"]
        for m, v in vals.items():
            v = v[np.isfinite(v)]
            entry = {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
                     "q75": float(np.percentile(v, 75))}
            if c != "lj_baseline":
                b = base[m][np.isfinite(base[m])]
                entry["p_vs_baseline"] = float(stats.mannwhitneyu(v, b).pvalue)
            out[c][m] = entry
    return out


def _box(ax, data, ylabel):
    labels = list(CONDITIONS)
    bp = ax.boxplot([data[c] for c in labels], widths=0.55, showfliers=False, patch_artist=True)
    rng = np.random.default_rng(0)
    for i, c in enumerate(labels, start=1):
        color = CONDITIONS[c][1]
        bp["boxes"][i - 1].set(facecolor=color, alpha=0.3)
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(data[c])), data[c], s=8, color=color, lw=0, zorder=3)
    for med in bp["medians"]:
        med.set(color="k")
    ax.set_xticks(range(1, len(labels) + 1), [CONDITIONS[c][0].replace(", ", ",\n") for c in labels], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", ls=":", alpha=0.6)


def main():
    summary = {f"n{n}": _summary(_table(n)) for n in (10, 20)}
    json.dump(summary, open(OUT_JSON, "w"), indent=2)
    print(f"Saved {OUT_JSON}")

    tab = _table(20)
    fig = plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 1], hspace=0.55, wspace=0.35)

    for row, c in enumerate(("plain_seed123", "lj_baseline")):
        r = analyze(c, 20, seed=SEED)
        ax = fig.add_subplot(gs[row, 0])
        label, color = CONDITIONS[c]
        for seg in r["persistence"]["segments"]:
            ax.plot([seg["start_s"], seg["end_s"]], [seg["agent"], seg["agent"]], lw=4, color=color,
                    solid_capstyle="butt")
        ax.set_ylim(-1, 20); ax.set_yticks([0, 5, 10, 15, 19])
        ax.set_ylabel("front robot", fontsize=9)
        ax.set_title(f"({'a' if row == 0 else 'b'}) {label}: {r['persistence']['real_switches']} "
                     f"position changes in {len(r['occupancy']['front_id']) * r['dt']:.0f} s", fontsize=9.5, loc="left")
        ax.grid(True, ls=":", alpha=0.5)
        if row == 1:
            ax.set_xlabel("time [s]", fontsize=9)

    ax = fig.add_subplot(gs[:, 1])
    _box(ax, {c: 60 * tab[c]["persistence_filtered_rate_per_sec"] for c in CONDITIONS},
         "front-position changes per minute")
    ax.set_title("(c)", fontsize=9.5, loc="left")
    ax = fig.add_subplot(gs[:, 2])
    _box(ax, {c: 100 * tab[c]["real_switches"] / tab[c]["raw_switches"] for c in CONDITIONS},
         "raw front changes that persist ≥ 1.5 s [%]")
    ax.set_title("(d)", fontsize=9.5, loc="left")

    stem = os.path.join(OUT_DIR_FIG, "paper_leadership_n20")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.{{png,pdf,svg}}")

    for c in CONDITIONS:
        s = summary["n20"][c]
        print(c, {m: round(s[m]["median"], 3) for m in s})


if __name__ == "__main__":
    main()
