"""Methodology illustrations for summary_plain_seed123.tex: the end-to-end pipeline, the
Hebbian controller, the sensing model + wind field, and the per-step simulation loop.

Every number drawn here is read from the simulation's config (or, for the wind-field panel,
computed by the simulation's own wind code from a real plain_seed123 episode) rather than
typed in, so the figures can't drift from the code that produced the results.

Usage: python methodology_figures.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Wedge

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
EXPERIMENT_DIR = os.path.join(REPO_ROOT, "ants26_replication", "experiment")
sys.path.insert(0, EXPERIMENT_DIR)

import config  # noqa: E402
from hebbian_controller import unflatten_abcd  # noqa: E402
from simulation_hebbian import simulate_hebbian_episode  # noqa: E402
from wind_physics import RayTraceCircularRobots  # noqa: E402
from leadership_metrics import _ConfigOverride, WIND_GRID  # noqa: E402

OUT_DIR_FIG = os.path.join(SCRIPT_DIR, "overleaf_summary", "figures")
GENOME_PATH = os.path.join(SCRIPT_DIR, "plain_seed123", "genome_trained_n20.npy")

C_UNCLAMPED = "#D95319"
C_CLAMPED = "#0072BD"
C_LJ = "#77AC30"
C_WIND = "#4C72B0"
INK = "#222222"
MUTED = "#666666"
FILL = "#F3F3F3"
EDGE = "#9A9A9A"

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})


def _savefig(fig, name):
    os.makedirs(OUT_DIR_FIG, exist_ok=True)
    stem = os.path.join(OUT_DIR_FIG, name)
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.{{png,pdf,svg}}")


def _box(ax, x, y, w, h, title, body="", edge=EDGE, fill=FILL, lw=1.0, title_size=9.5,
         body_size=8, title_color=INK):
    """Rounded box centred on (x, y); bold title on top, smaller body text beneath."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fill, ec=edge, lw=lw, zorder=2))
    if body:
        ax.text(x, y + h / 2 - 0.1, title, ha="center", va="top", fontsize=title_size,
                fontweight="bold", color=title_color, zorder=3)
        ax.text(x, y + h / 2 - 0.1 - 0.2 * title_size / 9.5, body, ha="center", va="top",
                fontsize=body_size, color=INK, zorder=3, linespacing=1.35)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=title_color, zorder=3)


def _arrow(ax, p0, p1, color=INK, lw=1.2, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11, color=color,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=1,
                                 shrinkA=2, shrinkB=2))


# ============================================ Figure 1: pipeline ============================================

def fig_pipeline():
    gen1 = config.HEBBIAN_CMAES_GEN_MAX
    gen2 = 2 * config.HEBBIAN_CMAES_GEN_MAX
    lam = config.HEBBIAN_CMAES_POPSIZE
    reps = config.HEBBIAN_N_REPEATS
    wd = config.HEBBIAN_EFF_DISTANCE_WEIGHT
    bw, cw, wmult = config.HEBBIAN_STAGE_FITNESS_WEIGHTS["save_battery_avoid_all"][:3]
    lo, hi = config.HEBBIAN_ABCD_BOUNDS

    fig, ax = plt.subplots(figsize=(14, 7.0))
    ax.set_xlim(0, 14); ax.set_ylim(0.1, 7.4); ax.axis("off")

    # --- section labels ---
    for x0, x1, label in ((0.15, 8.55, "1  Training (simulation, n = 20)"),
                          (8.75, 11.35, "2  Evaluation & selection"),
                          (11.55, 13.95, "3  Use")):
        ax.plot([x0, x1], [7.15, 7.15], color=MUTED, lw=0.8)
        ax.text(x0, 7.22, label, fontsize=10.5, fontweight="bold", color=INK, va="bottom")

    # --- two training tracks (same curriculum, run separately from scratch) ---
    tracks = (
        (5.55, C_UNCLAMPED, "Unclamped variant", "no safety mechanisms"),
        (3.75, C_CLAMPED, "Clamped variant",
         "speed clamp near robots/walls, 1.3× collision radius, soft collision resolution"),
    )
    for y, color, name, note in tracks:
        _box(ax, 1.15, y, 1.9, 1.15, "Random genome",
             f"ABCD ~ U[{lo:g}, {hi:g}]$^{{{config.HEBBIAN_N_ABCD}}}$\ntraining seed 123",
             edge=color, lw=1.4)
        _box(ax, 3.55, y, 2.3, 1.15, f"Stage 1: walk upwind",
             f"{gen1} generations, wind on\nfitness = {wd:g}·d", edge=color, lw=1.4)
        _box(ax, 6.7, y, 3.3, 1.15, "Stage 2: save battery, avoid all",
             f"{gen2} generations, wind on\nfitness = {wd:g}·d + $\\bar b$/{bw:g}"
             f" $-$ ({wmult:g}·$t_{{wall}}$ + $t_{{coll}}$)/{cw:g}", edge=color, lw=1.4)
        _arrow(ax, (2.1, y), (2.4, y))
        _arrow(ax, (4.7, y), (5.05, y))
        ax.text(0.2, y + 0.68, name, color=color, fontsize=9.5, fontweight="bold", va="bottom")
        ax.text(0.3 + 0.105 * len(name), y + 0.68, "— " + note, color=MUTED, fontsize=7.8,
                va="bottom")
        _arrow(ax, (8.35, y), (8.85, 4.9 if y > 4.6 else 4.3), color=color, lw=1.4)

    # --- the CMA-ES loop every stage box runs ---
    ax.add_patch(FancyBboxPatch((0.2, 0.25), 8.3, 2.45, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=EDGE, lw=0.9, ls="--", zorder=0))
    ax.text(0.35, 2.58, "Inside every stage: one CMA-ES generation", fontsize=9.5,
            fontweight="bold", color=INK, va="top")
    steps = (
        (1.2, "Sample", f"λ = {lam} candidate\nABCD genomes"),
        (3.25, "Simulate", f"each candidate:\n{reps} episodes\n({reps} spawn seeds)"),
        (5.3, "Score", "fitness = median\nover the\n{reps} episodes".format(reps=reps)),
        (7.35, "Update", "CMA-ES mean +\ncovariance; keep\nbest-ever genome"),
    )
    for x, t, b in steps:
        _box(ax, x, 1.3, 1.7, 1.05, t, b, body_size=7.8)
    for (x0, *_), (x1, *_) in zip(steps[:-1], steps[1:]):
        _arrow(ax, (x0 + 0.85, 1.3), (x1 - 0.85, 1.3))
    _arrow(ax, (7.35, 0.77), (1.2, 0.77), rad=-0.08, color=MUTED, lw=1.0)
    ax.text(4.25, 0.35, "next generation", fontsize=7.5, color=MUTED, ha="center")
    ax.text(0.35, 2.3, "Stage 2 starts from Stage 1's best genome; the saved genome is the best "
            "ever evaluated, not the last one.", fontsize=7.8, color=MUTED, va="top")

    # --- evaluation & selection ---
    _box(ax, 10.05, 4.6, 2.4, 2.2, "Evaluate",
         "30 spawn seeds × every\nswarm size n = 1…20\nstandard physics\n\nvs. rule-based LJ\nbaseline "
         "(same seeds)\n→ distance, battery,\nboth-beat rate", lw=1.2)
    _arrow(ax, (10.05, 3.5), (10.05, 2.85))
    _box(ax, 10.05, 1.95, 2.4, 1.75, "Select",
         "5 training seeds per recipe;\ncandidates that reliably\nbeat LJ at n = 20, then\nrejected any that hug\n"
         "walls → plain_seed123", lw=1.2)

    # --- use ---
    _box(ax, 12.75, 5.2, 2.3, 1.1, "Simulation results",
         "unclamped genome:\nbest simulated\nperformance", edge=C_UNCLAMPED, lw=1.4)
    _box(ax, 12.75, 2.75, 2.3, 2.35, "Hardware trial",
         "clamped genome on\nThymio + Raspberry Pi\n\nOptiTrack poses →\nsame virtual 4-quadrant\nsensor as simulation\n\n"
         "clamp re-enforced\non-robot", edge=C_CLAMPED, lw=1.4)
    _arrow(ax, (11.25, 2.3), (11.6, 5.0), color=C_UNCLAMPED, lw=1.4)
    _arrow(ax, (11.25, 1.7), (11.6, 2.4), color=C_CLAMPED, lw=1.4)

    _savefig(fig, "method_pipeline")


# ============================================ Figure 2: controller ============================================

def fig_controller():
    n_in, n_h, n_out = config.HEBBIAN_N_INPUTS, config.HEBBIAN_N_HIDDEN, config.HEBBIAN_N_OUTPUTS
    input_labels = ["front distance", "front bearing", "back distance", "back bearing",
                    "right distance", "right bearing", "left distance", "left bearing",
                    "own battery", "own heading"]
    assert len(input_labels) == n_in
    vmax, wmax = config.HEBBIAN_LINEAR_VEL_MAX, config.HEBBIAN_ANGULAR_VEL_MAX
    output_labels = [f"v  (±{vmax:g} m/s)", f"ω  (±π/{round(np.pi / wmax):d} rad/s)"]

    fig, ax = plt.subplots(figsize=(13.4, 6.2))
    ax.set_xlim(0, 13.4); ax.set_ylim(-0.3, 6.2); ax.axis("off")

    xs = (2.2, 3.9, 5.6, 7.3)
    def ys(n):
        return np.linspace(5.3, 0.5, n) if n > 2 else np.array([3.5, 2.3])
    layers = [ys(n_in), ys(n_h), ys(n_h), ys(n_out)]

    for li in range(3):
        for y0 in layers[li]:
            for y1 in layers[li + 1]:
                ax.plot([xs[li], xs[li + 1]], [y0, y1], color="#BBBBBB", lw=0.35, zorder=1)
    node_colors = ["#DDE8F3", "#EDEDED", "#EDEDED", "#FBE3D6"]
    for li, (x, yy) in enumerate(zip(xs, layers)):
        for y in yy:
            ax.add_patch(Circle((x, y), 0.14, fc=node_colors[li], ec=INK, lw=0.8, zorder=3))
    for y, lab in zip(layers[0], input_labels):
        ax.text(xs[0] - 0.25, y, lab, ha="right", va="center", fontsize=8.5)
    for y, lab in zip(layers[3], output_labels):
        ax.text(xs[3] + 0.25, y, lab, ha="left", va="center", fontsize=9)

    heads = ((xs[0], f"input ({n_in})"), (xs[1], f"hidden 1 ({n_h})\nReLU"),
             (xs[2], f"hidden 2 ({n_h})\nReLU"), (xs[3], f"output ({n_out})\ntanh"))
    for x, t in heads:
        ax.text(x, 5.65, t, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for (x0, x1), w in zip(((xs[0], xs[1]), (xs[1], xs[2]), (xs[2], xs[3])),
                           (f"$W_1$ ({n_in}×{n_h})", f"$W_2$ ({n_h}×{n_h})", f"$W_3$ ({n_h}×{n_out})")):
        ax.text((x0 + x1) / 2, 0.05, w, ha="center", va="top", fontsize=8.5, color=MUTED)
    ax.text(xs[0] - 1.0, 0.05, "each input rescaled to ≈ [−1, 1]", ha="center", va="top",
            fontsize=8, color=MUTED)

    # --- Hebbian rule panel ---
    px = 9.25
    ax.add_patch(FancyBboxPatch((px - 0.1, 0.9), 4.0, 4.95, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=FILL, ec=EDGE, lw=1.0))
    ax.text(px + 0.05, 5.65, "Every time step, every synapse", fontsize=9.5, fontweight="bold", va="top")
    ax.text(px + 1.83, 4.85, r"$\Delta w_{ij} = \eta\,(A_{ij}\,n_i n_j + B_{ij}\,n_i + C_{ij}\,n_j + D_{ij})$",
            fontsize=10, ha="center", va="center")
    ax.text(px + 0.05, 4.35,
            f"$n_i$, $n_j$: pre-/post-synaptic activity\n"
            f"η = {config.HEBBIAN_LEARNING_RATE:g}; a layer is rescaled to\nmax |w| = 1 whenever it exceeds 1",
            fontsize=8, va="top", linespacing=1.4)
    ax.text(px + 0.05, 3.05, "What is evolved vs. learned", fontsize=9.5, fontweight="bold", va="top")
    n_syn = n_in * n_h + n_h * n_h + n_h * n_out
    ax.text(px + 0.05, 2.7,
            f"Evolved (CMA-ES): the genome — one\n(A, B, C, D) per synapse = 4 × {n_syn}\n"
            f"= {config.HEBBIAN_N_ABCD} numbers in [{config.HEBBIAN_ABCD_BOUNDS[0]:g}, "
            f"{config.HEBBIAN_ABCD_BOUNDS[1]:g}], shared by all robots.\n\n"
            f"Learned online (per robot): the weights\n$W_1, W_2, W_3$ — redrawn from "
            f"U[−{config.HEBBIAN_WEIGHT_INIT_RANGE:g}, {config.HEBBIAN_WEIGHT_INIT_RANGE:g}]\n"
            "at the start of every episode.",
            fontsize=8, va="top", linespacing=1.4)

    _savefig(fig, "method_controller")


# ============================================ Figure 3: sensing + wind ============================================

def _wind_snapshot(n_agents=10, seed=42, frac=0.5):
    """Positions from a real plain_seed123 episode at `frac` of its length, plus the wind
    field the simulator computes for exactly those positions."""
    rules = unflatten_abcd(np.load(GENOME_PATH))
    with _ConfigOverride(HEBBIAN_SAFETY_CLAMP_ENABLED=False, HEBBIAN_MIN_DIST_INFLATION=1.0,
                         HEBBIAN_RESOLVE_COLLISIONS=False):
        *_, tele = simulate_hebbian_episode(rules, seed=seed, n_agents=n_agents, wind_enabled=True,
                                            nx=WIND_GRID, ny=WIND_GRID, record_trajectory=True)
    pos = tele["positions"][int(frac * (len(tele["positions"]) - 1))]
    agents = np.zeros((n_agents, 4)); agents[:, 0:2] = pos
    min_x = pos[:, 0].min()
    max_x = min(pos[:, 0].max(), min_x + config.WIND_TRACKING_MAX_SPAN)
    w = config.WIND_TRACKING_WINDOW_WIDTH
    x_range = [min_x - (w - (max_x - min_x)) / 2.0, max_x + (w - (max_x - min_x)) / 2.0]
    yv, xv, pv = RayTraceCircularRobots(agents, config.WIND_RAD, config.UINF, x_range,
                                        list(config.Y_RANGE), WIND_GRID, WIND_GRID)
    return pos, xv, yv, pv, x_range


def fig_sensing_wind():
    R = config.HEBBIAN_SENSING_RADIUS
    fig, (ax_s, ax_w) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1, 1.2]})

    # --- (a) 4-quadrant sensing, in the robot's own frame (heading = up) ---
    ax = ax_s
    quads = (("front", 45, 135, "#DDE8F3"), ("left", 135, 225, "#E9F2DF"),
             ("back", 225, 315, "#F4E4F0"), ("right", -45, 45, "#FBEBD8"))
    for name, a0, a1, col in quads:
        ax.add_patch(Wedge((0, 0), R, a0, a1, fc=col, ec="white", lw=1.5, zorder=0))
        am = np.deg2rad((a0 + a1) / 2)
        ax.text(0.8 * R * np.cos(am), 0.8 * R * np.sin(am), name, ha="center", va="center",
                fontsize=10, fontweight="bold", color=MUTED)
    ax.add_patch(Circle((0, 0), R, fill=False, ec=INK, lw=0.8, ls="--"))
    ax.text(R * np.cos(np.deg2rad(20)) + 0.05, R * np.sin(np.deg2rad(20)), f"R = {R:g} m",
            fontsize=8.5, color=INK, va="bottom")

    neighbours = np.array([(0.35, 1.05), (-0.45, 1.6), (-1.2, 0.2), (0.1, -0.8), (0.9, -1.5),
                           (1.25, 0.45), (1.9, 1.5), (-1.7, -1.6)])
    ang = np.rad2deg(np.arctan2(neighbours[:, 1], neighbours[:, 0])) % 360
    dist = np.hypot(*neighbours.T)
    nearest = set()
    for name, a0, a1, _ in quads:
        a0m, a1m = a0 % 360, a1 % 360
        inq = ((ang >= a0m) & (ang < a1m)) if a0m < a1m else ((ang >= a0m) | (ang < a1m))
        cand = np.where(inq & (dist < R))[0]
        if len(cand):
            nearest.add(int(cand[np.argmin(dist[cand])]))
    for i, (x, y) in enumerate(neighbours):
        if dist[i] >= R:
            ax.add_patch(Circle((x, y), 0.09, fc="white", ec=EDGE, lw=0.9, ls=":", zorder=3))
        elif i in nearest:
            ax.plot([0, x], [0, y], color=INK, lw=1.1, zorder=2)
            ax.add_patch(Circle((x, y), 0.09, fc=INK, ec=INK, zorder=3))
        else:
            ax.add_patch(Circle((x, y), 0.09, fc="#AAAAAA", ec="#AAAAAA", zorder=3))
    ax.add_patch(Circle((0, 0), 0.13, fc=C_UNCLAMPED, ec=INK, lw=0.8, zorder=4))
    ax.annotate("", xy=(0, 0.5), xytext=(0, 0.13),
                arrowprops=dict(arrowstyle="-|>", color=C_UNCLAMPED, lw=1.6), zorder=4)
    ax.set_xlim(-R - 0.3, R + 0.3); ax.set_ylim(-R - 0.75, R + 0.3); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) Sensing: nearest neighbour per body-relative quadrant", fontsize=10,
                 fontweight="bold", loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="-", color=INK, ms=6, label="sensed: distance + bearing"),
               plt.Line2D([], [], marker="o", ls="", color="#AAAAAA", ms=6, label="in range, not nearest: ignored"),
               plt.Line2D([], [], marker="o", ls="", mfc="white", mec=EDGE, ms=6, label="beyond R: ignored"),
               plt.Line2D([], [], marker="o", ls="", color=C_UNCLAMPED, ms=7, label="focal robot (arrow = heading)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=7.8,
              frameon=False)

    # --- (b) wind field from a real episode ---
    ax = ax_w
    pos, xv, yv, pv, x_range = _wind_snapshot()
    im = ax.pcolormesh(xv, yv, pv.T, cmap="Blues", vmin=0, vmax=config.UINF, shading="auto")
    for x, y in pos:
        ax.add_patch(Circle((x, y), config.ROBOT_RAD * 2.2, fc=C_UNCLAMPED, ec=INK, lw=0.6, zorder=3))
    for yw in config.Y_RANGE:
        ax.axhline(yw, color=INK, lw=2.5)
    ax.text(x_range[0] + 0.15, config.Y_RANGE[1] - 0.2, "wall", fontsize=8, va="top")
    ax.text(x_range[0] + 0.15, config.Y_RANGE[0] + 0.2, "wall", fontsize=8, va="bottom")
    ax.annotate("", xy=(x_range[0] + 2.6, 3.9), xytext=(x_range[0] + 0.4, 3.9),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8))
    ax.text(x_range[0] + 1.5, 4.1, "wind", ha="center", fontsize=9, fontweight="bold")
    ax.annotate("", xy=(x_range[1] - 3.4, -3.9), xytext=(x_range[1] - 1.2, -3.9),
                arrowprops=dict(arrowstyle="-|>", color=C_UNCLAMPED, lw=1.8))
    ax.text(x_range[1] - 2.3, -3.7, "rewarded direction\n(swarm centroid, upwind)", ha="center",
            va="bottom", fontsize=8, color=C_UNCLAMPED, fontweight="bold")
    ax.set_xlim(x_range); ax.set_ylim(config.Y_RANGE[0] - 0.2, config.Y_RANGE[1] + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m] (window follows the swarm)"); ax.set_ylabel("y [m]")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("wind power [% of free stream]")
    ax.set_title("(b) Wind: robots shelter those downwind of them\n"
                 "plain_seed123, n = 10, seed 42, half-way through the episode\n(robots drawn enlarged)",
                 fontsize=10, fontweight="bold", loc="left")

    fig.tight_layout()
    _savefig(fig, "method_sensing_wind")


# ============================================ Figure 4: per-step simulation loop ============================================

def fig_step_loop():
    fig, ax = plt.subplots(figsize=(13, 4.0))
    ax.set_xlim(0, 13); ax.set_ylim(0.05, 4.0); ax.axis("off")

    steps = (
        ("1  Sense", "nearest neighbour per\nquadrant: distance,\nbearing; battery, heading"),
        ("2  Controller", "forward pass → (v, ω);\nHebbian update of\nthis robot's weights"),
        ("3  Safety clamp", "scale v down near\nrobots / walls\n(clamped variant only)"),
        ("4  Move", f"kinematics, dt = {config.DT:g} s;\ncount collisions; soft\nresolution (clamped only)"),
        ("5  Wind", "ray-trace wakes\non 50 × 50 grid;\ndrag from local wind"),
        ("6  Battery", "drain = wheel effort\n+ work against drag\n(sheltered → cheaper)"),
    )
    w, h, y = 1.95, 1.2, 2.9
    x_positions = np.linspace(1.1, 11.9, len(steps))
    for x, (t, b) in zip(x_positions, steps):
        edge = C_CLAMPED if "Safety" in t else EDGE
        _box(ax, x, y, w, h, t, b, edge=edge, lw=1.3 if "Safety" in t else 1.0, body_size=7.8)
    for x0, x1 in zip(x_positions[:-1], x_positions[1:]):
        _arrow(ax, (x0 + w / 2, y), (x1 - w / 2, y))

    _box(ax, 9.6, 0.65, 3.0, 0.95, "Any battery empty?", "yes → episode ends:\ncompute distance d, mean battery $\\bar b$",
         body_size=7.8)
    _arrow(ax, (x_positions[-1], y - h / 2), (10.6, 1.15))
    _arrow(ax, (8.1, 0.65), (x_positions[0], y - h / 2), rad=-0.18, color=MUTED)
    ax.text(4.3, 1.25, "no → next time step", fontsize=8, color=MUTED, ha="center")
    ax.text(0.1, 3.95, "Every simulated time step, for all robots at once", fontsize=10.5,
            fontweight="bold", va="top")

    _savefig(fig, "method_step_loop")


if __name__ == "__main__":
    fig_pipeline()
    fig_controller()
    fig_sensing_wind()
    fig_step_loop()
