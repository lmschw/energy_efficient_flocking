"""Plotting for leadership_metrics.py -- see that module's docstring for what each metric
means. Kept in a separate file so `python leadership_metrics.py` can be re-run for just the
numbers (fast) without re-importing matplotlib's heavier machinery every time during
iteration; imported lazily by leadership_metrics.py's __main__ block.

Every figure is scoped to exactly ONE of the four metrics -- filenames are prefixed
metric1_/metric2_/metric3_/metric4_ accordingly, so each can be cited independently. Metric 3
(front occupancy/exchange-rate, computed from the raw per-step rank series) and metric 4
(persistence-filtered hand-off count, a simplified proxy for Butail & Porfiri's transfer-
entropy switch detector) are DELIBERATELY never drawn on the same axes, even though they're
derived from the same underlying front_id series -- they're different statistics with
different citations and belong in separate figures."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle

from leadership_metrics import CONDITIONS, CONDITION_LABELS, CONDITION_COLORS

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5})


def _savefig(fig, out_dir, name, rect=None):
    path = os.path.join(out_dir, name)
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def _strip_box(ax, data_by_condition, ylabel, title):
    positions = list(range(len(CONDITIONS)))
    data = [np.asarray(data_by_condition[c], dtype=float) for c in CONDITIONS]
    data_clean = [d[np.isfinite(d)] for d in data]
    bp = ax.boxplot(data_clean, positions=positions, widths=0.5, showfliers=False, patch_artist=True)
    for patch, condition in zip(bp["boxes"], CONDITIONS):
        patch.set_facecolor(CONDITION_COLORS[condition])
        patch.set_alpha(0.35)

    rng = np.random.default_rng(0)
    for pos, condition, d in zip(positions, CONDITIONS, data_clean):
        jitter = rng.uniform(-0.12, 0.12, size=len(d))
        ax.scatter(pos + jitter, d, s=14, color=CONDITION_COLORS[condition],
                   edgecolors="k", linewidths=0.3, zorder=3, alpha=0.85)

    ax.set_xticks(positions)
    ax.set_xticklabels([CONDITION_LABELS[c].split(" (")[0] for c in CONDITIONS], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)


# ============================== METRIC 1: Voelkl et al. (2015) reciprocity ==============================

def plot_reciprocity_scatter(results, n_agents_list, out_dir):
    n_cols = len(n_agents_list)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.2 * n_cols, 9), squeeze=False)

    for row, condition in enumerate(CONDITIONS):
        for col, n_agents in enumerate(n_agents_list):
            ax = axes[row][col]
            recip = results[(condition, n_agents)]["reciprocity"]
            lead, follow = recip["lead_time"], recip["follow_time"]
            color = CONDITION_COLORS[condition]
            ax.scatter(lead, follow, s=40, color=color, edgecolors="k", linewidths=0.4, zorder=3)
            lim = max(lead.max(), follow.max(), 1e-6) * 1.15
            ax.plot([0, lim], [0, lim], linestyle="--", color="gray", linewidth=1, zorder=1)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            r = recip["r"]
            r_label = f"r={r:.2f}" if np.isfinite(r) else "r=n/a"
            ax.set_title(f"n={n_agents}  {r_label}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{CONDITION_LABELS[condition]}\nfollow time [s]", fontsize=8)
            if row == 2:
                ax.set_xlabel("lead time [s]", fontsize=9)

    fig.suptitle("Metric 1 -- Voelkl et al. (2015) reciprocity: time leading vs. time following\n"
                 "(near the dashed diagonal = individuals that lead and follow equally)",
                 fontsize=11, fontweight="bold")
    _savefig(fig, out_dir, "metric1_reciprocity_scatter.png", rect=[0, 0, 1, 0.90])


def plot_reciprocity_summary(results, n_agents_list, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for condition in CONDITIONS:
        rs = [results[(condition, n)]["reciprocity"]["r"] for n in n_agents_list]
        ax.plot(n_agents_list, rs, marker="o", label=CONDITION_LABELS[condition],
                 color=CONDITION_COLORS[condition])
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("n_agents")
    ax.set_ylabel("Reciprocity index r\n(corr. of lead-time vs. follow-time across agents)")
    ax.set_title("Metric 1 -- leading/following time-matching vs. swarm size")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=9)
    _savefig(fig, out_dir, "metric1_reciprocity_summary.png")


def plot_seed_sweep_reciprocity(seed_results, n_agents, seeds, out_dir):
    fig, ax = plt.subplots(figsize=(5, 5.2))
    data_by_condition = {c: [seed_results[(c, s)]["reciprocity"]["r"] for s in seeds] for c in CONDITIONS}
    _strip_box(ax, data_by_condition, "reciprocity r", f"Metric 1 -- Voelkl reciprocity (n={n_agents})")
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    fig.suptitle(f"Reciprocity across {len(seeds)} seeds", fontweight="bold")
    _savefig(fig, out_dir, f"metric1_seed_sweep_n{n_agents}.png", rect=[0, 0, 1, 0.92])


# ===================== METRIC 2: Nagy/Akos/Biro/Vicsek (2010) hierarchy network =====================

def _draw_network(ax, leader_strength, title):
    n_agents = leader_strength.shape[0]
    angles = 2 * np.pi * np.arange(n_agents) / n_agents
    node_xy = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    max_w = leader_strength.max()
    for i in range(n_agents):
        for j in range(n_agents):
            w = leader_strength[i, j]
            if w <= 0:
                continue
            arrow = FancyArrowPatch(node_xy[i], node_xy[j],
                                     connectionstyle="arc3,rad=0.15",
                                     arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1 + 3 * (w / max_w if max_w > 0 else 0),
                                     color="black", alpha=0.35 + 0.55 * (w / max_w if max_w > 0 else 0),
                                     zorder=2)
            ax.add_patch(arrow)

    for i in range(n_agents):
        ax.add_patch(Circle(node_xy[i], 0.12, color="#0072BD", zorder=3))
        ax.text(*node_xy[i], str(i), color="white", ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=4)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_hierarchy_network(results, n_agents_list, out_dir, rep_n=None):
    candidates = [n for n in n_agents_list if 3 <= n <= 6] or n_agents_list
    rep_n = rep_n or candidates[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, condition in zip(axes, CONDITIONS):
        leader_strength = results[(condition, rep_n)]["leader_strength"]
        _draw_network(ax, leader_strength, f"{CONDITION_LABELS[condition]}\n(n={rep_n})")
    fig.suptitle("Metric 2 -- Nagy et al. (2010) directional-correlation-delay leadership network\n"
                 "(arrow i->j: i's turns precede j's with a lag; width/opacity = correlation strength)",
                 fontweight="bold")
    _savefig(fig, out_dir, f"metric2_hierarchy_network_n{rep_n}.png", rect=[0, 0, 1, 0.88])


def plot_hierarchy_summary(results, n_agents_list, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for condition in CONDITIONS:
        color = CONDITION_COLORS[condition]
        ents = [results[(condition, n)]["hierarchy"]["entropy"] for n in n_agents_list]
        steep = [results[(condition, n)]["hierarchy"]["steepness"] for n in n_agents_list]
        axes[0].plot(n_agents_list, ents, marker="o", color=color, label=CONDITION_LABELS[condition])
        axes[1].plot(n_agents_list, steep, marker="o", color=color, label=CONDITION_LABELS[condition])

    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("n_agents")
    axes[0].set_ylabel("normalized leadership entropy")
    axes[0].set_title("Leadership entropy (1 = everyone initiates turns equally)")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("n_agents")
    axes[1].set_ylabel("hierarchy steepness")
    axes[1].set_title("Steepness (0 = flat/egalitarian, 1 = one dominant initiator)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Metric 2 -- directional-correlation-delay hierarchy summary", fontweight="bold")
    _savefig(fig, out_dir, "metric2_hierarchy_summary.png", rect=[0, 0, 1, 0.93])


def plot_seed_sweep_hierarchy(seed_results, n_agents, seeds, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2))
    ent_by_cond = {c: [seed_results[(c, s)]["hierarchy"]["entropy"] for s in seeds] for c in CONDITIONS}
    steep_by_cond = {c: [seed_results[(c, s)]["hierarchy"]["steepness"] for s in seeds] for c in CONDITIONS}
    _strip_box(axes[0], ent_by_cond, "normalized entropy", "Leadership entropy")
    _strip_box(axes[1], steep_by_cond, "steepness", "Hierarchy steepness")
    fig.suptitle(f"Metric 2 -- hierarchy summary across {len(seeds)} seeds (n={n_agents})",
                 fontweight="bold")
    _savefig(fig, out_dir, f"metric2_seed_sweep_n{n_agents}.png", rect=[0, 0, 1, 0.90])


# ========================= METRIC 3: front-rank occupancy / raw exchange rate =========================

def plot_occupancy(results, n_agents_list, out_dir):
    rep_n = max(n_agents_list)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    width = 0.25
    for k, condition in enumerate(CONDITIONS):
        occ = results[(condition, rep_n)]["occupancy"]["occupancy_fraction"]
        x = np.arange(rep_n) + (k - 1) * width
        ax.bar(x, occ, width=width, label=CONDITION_LABELS[condition],
               color=CONDITION_COLORS[condition])
    ax.axhline(1.0 / rep_n, color="gray", linestyle="--", linewidth=1,
               label=f"uniform (1/{rep_n})")
    ax.set_xlabel("agent index")
    ax.set_ylabel("fraction of episode spent as rank-1 (front-most)")
    ax.set_title(f"Front-rank occupancy per agent (n={rep_n})")
    ax.legend(fontsize=8)

    ax = axes[1]
    for condition in CONDITIONS:
        ents = [results[(condition, n)]["occupancy"]["normalized_entropy"] for n in n_agents_list]
        ax.plot(n_agents_list, ents, marker="o", label=CONDITION_LABELS[condition],
                 color=CONDITION_COLORS[condition])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="perfectly flat (=1)")
    ax.set_xlabel("n_agents")
    ax.set_ylabel("normalized entropy of front-occupancy distribution")
    ax.set_title("Occupancy flatness -- 1.0 = everyone takes an equal turn leading")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    fig.suptitle("Metric 3 -- front-position occupancy (Mirzaeinia et al. 2019-style leader tracking)",
                 fontweight="bold")
    _savefig(fig, out_dir, "metric3_front_occupancy.png", rect=[0, 0, 1, 0.93])


def plot_exchange_rate_raw(results, n_agents_list, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for condition in CONDITIONS:
        color = CONDITION_COLORS[condition]
        raw_sec = [results[(condition, n)]["occupancy"]["exchange_rate_per_sec"] for n in n_agents_list]
        raw_m = [results[(condition, n)]["occupancy"]["exchange_rate_per_m"] for n in n_agents_list]
        axes[0].plot(n_agents_list, raw_sec, marker="o", color=color, label=CONDITION_LABELS[condition])
        axes[1].plot(n_agents_list, raw_m, marker="o", color=color, label=CONDITION_LABELS[condition])

    axes[0].set_xlabel("n_agents")
    axes[0].set_ylabel("raw leader hand-offs per second")
    axes[0].set_title("Exchange rate (time-normalized)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("n_agents")
    axes[1].set_ylabel("raw leader hand-offs per metre travelled")
    axes[1].set_title("Exchange rate (distance-normalized)")

    fig.suptitle("Metric 3 -- raw front-position exchange rate (every rank-1 identity change, "
                 "including single-step noise)", fontweight="bold")
    _savefig(fig, out_dir, "metric3_exchange_rate.png", rect=[0, 0, 1, 0.90])


def plot_front_id_timeline_raw(results, n_agents, out_dir):
    """Metric 3 only: the raw rank-1 identity every step, no filtering. Directly comparable
    against hardware_transfer_test/<condition>/n<N>/video.mp4 -- see metric4_handoff_timeline
    for the persistence-filtered "real hand-offs" version of this same underlying signal."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    for ax, condition in zip(axes, CONDITIONS):
        r = results[(condition, n_agents)]
        front_id = r["occupancy"]["front_id"]
        dt = r["dt"]
        t = np.arange(len(front_id)) * dt
        color = CONDITION_COLORS[condition]

        ax.step(t, front_id, where="post", linewidth=0.9, color=color)
        ax.set_ylabel("agent id")
        ax.set_yticks(range(n_agents))
        ax.set_title(f"{CONDITION_LABELS[condition]} (n={n_agents}) -- "
                     f"{r['occupancy']['raw_switches']} raw identity changes", fontsize=10)

    axes[-1].set_xlabel("time [s] -- compare directly against "
                         f"hardware_transfer_test/<condition>/n{n_agents}/video.mp4")
    fig.suptitle(f"Metric 3 -- raw rank-1 (front-most) identity over time (n={n_agents})",
                 fontweight="bold")
    _savefig(fig, out_dir, f"metric3_front_id_timeline_n{n_agents}.png", rect=[0, 0, 1, 0.95])


def plot_seed_sweep_occupancy(seed_results, n_agents, seeds, out_dir):
    fig, ax = plt.subplots(figsize=(5, 5.2))
    data_by_condition = {c: [seed_results[(c, s)]["occupancy"]["normalized_entropy"] for s in seeds]
                          for c in CONDITIONS}
    _strip_box(ax, data_by_condition, "normalized entropy",
               f"Metric 3 -- front-occupancy flatness (n={n_agents})")
    ax.axhline(1.0, color="gray", linewidth=1, linestyle="--")
    fig.suptitle(f"Occupancy flatness across {len(seeds)} seeds", fontweight="bold")
    _savefig(fig, out_dir, f"metric3_seed_sweep_n{n_agents}.png", rect=[0, 0, 1, 0.92])


# ============ METRIC 4: persistence-filtered switches (simplified Butail & Porfiri proxy) ============

def plot_exchange_rate_filtered(results, n_agents_list, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for condition in CONDITIONS:
        color = CONDITION_COLORS[condition]
        filt_sec = [results[(condition, n)]["persistence"]["rate_per_sec"] for n in n_agents_list]
        filt_m = [results[(condition, n)]["persistence"]["rate_per_m"] for n in n_agents_list]
        axes[0].plot(n_agents_list, filt_sec, marker="s", color=color, label=CONDITION_LABELS[condition])
        axes[1].plot(n_agents_list, filt_m, marker="s", color=color, label=CONDITION_LABELS[condition])

    axes[0].set_xlabel("n_agents")
    axes[0].set_ylabel("real leader hand-offs per second")
    axes[0].set_title("Exchange rate (time-normalized)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("n_agents")
    axes[1].set_ylabel("real leader hand-offs per metre travelled")
    axes[1].set_title("Exchange rate (distance-normalized)")

    fig.suptitle("Metric 4 -- persistence-filtered exchange rate (>=3-step dwell; simplified "
                 "proxy for Butail & Porfiri's transfer-entropy switch detector)", fontweight="bold")
    _savefig(fig, out_dir, "metric4_exchange_rate_filtered.png", rect=[0, 0, 1, 0.90])


def plot_handoff_timeline_filtered(results, n_agents, out_dir):
    """Metric 4 only: persistence-filtered "real" leader segments, no raw trace. Directly
    comparable against hardware_transfer_test/<condition>/n<N>/video.mp4 -- see
    metric3_front_id_timeline for the raw (unfiltered) version of this same underlying signal."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    for ax, condition in zip(axes, CONDITIONS):
        r = results[(condition, n_agents)]
        color = CONDITION_COLORS[condition]
        for seg in r["persistence"]["segments"]:
            ax.plot([seg["start_s"], seg["end_s"]], [seg["agent"], seg["agent"]],
                     linewidth=5, color=color, solid_capstyle="butt", zorder=3)

        ax.set_ylabel("agent id")
        ax.set_yticks(range(n_agents))
        ax.set_title(f"{CONDITION_LABELS[condition]} (n={n_agents}) -- "
                     f"{r['persistence']['real_switches']} real hand-offs (>=3-step dwell)", fontsize=10)

    axes[-1].set_xlabel("time [s] -- compare directly against "
                         f"hardware_transfer_test/<condition>/n{n_agents}/video.mp4")
    fig.suptitle(f"Metric 4 -- persistence-filtered front-most identity over time (n={n_agents})",
                 fontweight="bold")
    _savefig(fig, out_dir, f"metric4_handoff_timeline_n{n_agents}.png", rect=[0, 0, 1, 0.95])


def plot_switch_event_raster(seed_results, n_agents, seeds, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    for ax, condition in zip(axes, CONDITIONS):
        color = CONDITION_COLORS[condition]
        for row, seed in enumerate(seeds):
            segments = seed_results[(condition, seed)]["persistence"]["segments"]
            if len(segments) < 2:
                continue
            duration = segments[-1]["end_s"]
            # skip segments[0]'s start (t=0 is the initial leader, not a hand-off)
            xs = [seg["start_s"] / duration for seg in segments[1:]]
            ax.scatter(xs, [row] * len(xs), s=12, color=color, alpha=0.7, edgecolors="none")
        ax.set_ylabel("seed index", fontsize=9)
        n_handoffs = [max(0, len(seed_results[(condition, s)]["persistence"]["segments"]) - 1)
                      for s in seeds]
        ax.set_title(f"{CONDITION_LABELS[condition]} -- real hand-offs per run: "
                     f"median={np.median(n_handoffs):.0f}, range [{min(n_handoffs)}, {max(n_handoffs)}]",
                     fontsize=10)
        ax.set_ylim(-1, len(seeds))

    axes[-1].set_xlabel("real (persistence-filtered) hand-off time, as a fraction of that run's "
                         "episode length")
    axes[-1].set_xlim(0, 1)
    fig.suptitle(f"Metric 4 -- leader hand-off timing across {len(seeds)} seeds (n={n_agents})\n"
                 "each dot = one real hand-off event in one run; each row = one seed",
                 fontweight="bold")
    _savefig(fig, out_dir, f"metric4_switch_event_raster_n{n_agents}.png", rect=[0, 0, 1, 0.90])


def plot_seed_sweep_handoffs(seed_results, n_agents, seeds, out_dir):
    fig, ax = plt.subplots(figsize=(5, 5.2))
    data_by_condition = {c: [seed_results[(c, s)]["persistence"]["real_switches"] for s in seeds]
                          for c in CONDITIONS}
    _strip_box(ax, data_by_condition, "# real hand-offs",
               f"Metric 4 -- persistence-filtered hand-offs (n={n_agents})")
    fig.suptitle(f"Real hand-off count across {len(seeds)} seeds", fontweight="bold")
    _savefig(fig, out_dir, f"metric4_seed_sweep_n{n_agents}.png", rect=[0, 0, 1, 0.92])


# ================================================ orchestration ================================================

def make_all_plots(results, n_agents_list, out_dir):
    plot_reciprocity_scatter(results, n_agents_list, out_dir)
    plot_reciprocity_summary(results, n_agents_list, out_dir)
    plot_hierarchy_network(results, n_agents_list, out_dir)
    plot_hierarchy_summary(results, n_agents_list, out_dir)
    plot_occupancy(results, n_agents_list, out_dir)
    plot_exchange_rate_raw(results, n_agents_list, out_dir)
    plot_exchange_rate_filtered(results, n_agents_list, out_dir)
    for n_agents in n_agents_list:
        if n_agents >= 3:  # a timeline for n=2 is just two mirrored step functions
            plot_front_id_timeline_raw(results, n_agents, out_dir)
            plot_handoff_timeline_filtered(results, n_agents, out_dir)


def make_all_seed_sweep_plots(seed_results, n_agents, seeds, out_dir):
    plot_seed_sweep_reciprocity(seed_results, n_agents, seeds, out_dir)
    plot_seed_sweep_hierarchy(seed_results, n_agents, seeds, out_dir)
    plot_seed_sweep_occupancy(seed_results, n_agents, seeds, out_dir)
    plot_switch_event_raster(seed_results, n_agents, seeds, out_dir)
    plot_seed_sweep_handoffs(seed_results, n_agents, seeds, out_dir)
