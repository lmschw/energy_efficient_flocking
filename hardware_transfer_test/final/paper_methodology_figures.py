"""Methodology figures for swarm_intelligence_revised_no_clamp.tex, replacing the images taken
from the previous paper: the method overview (Fig. 1), the 4-quadrant sensing model (Fig. 2)
and the wind-field model (Fig. 3).

Paper-sized versions of methodology_figures.py's summary figures: drawn at the LLNCS text width
so the fonts print at 6.5-8 pt, with the paper's notation, colours (Stage 2 orange, baseline
green) and only the unclamped pipeline. As there, numbers are read from the simulation config
and every wind field is computed by the simulation's own wind code.

Usage: python paper_methodology_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Wedge

from methodology_figures import EDGE, FILL, INK, MUTED, _arrow, _savefig, config
from wind_physics import RayTraceCircularRobots, agent_wind_percentage

TEXT_WIDTH = 4.8  # LLNCS \textwidth [in]
C_STAGE1 = "#EDB120"
C_STAGE2 = "#D95319"
C_BASELINE = "#77AC30"
C_HW = "#0072BD"
# Wind speed: free stream pale, shelter dark, so the wakes stand out on a light page.
WIND_CMAP = LinearSegmentedColormap.from_list("wind", ["#08306B", "#4A90C4", "#DCE9F5"])

plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.labelsize": 7})


def _pbox(ax, x0, y0, w, h, title, body="", edge=EDGE, fill=FILL, lw=0.8, title_color=INK,
          body_size=6.3):
    """Rounded box with lower-left corner (x0, y0) in inches; bold title, smaller body below."""
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=fill, ec=edge, lw=lw, zorder=2))
    ax.text(x0 + w / 2, y0 + h - 0.05, title, ha="center", va="top", fontsize=7,
            fontweight="bold", color=title_color, zorder=3)
    if body:
        ax.text(x0 + w / 2, y0 + h - 0.19, body, ha="center", va="top", fontsize=body_size,
                color=INK, zorder=3, linespacing=1.3)


# ============================================ Fig. 1: method overview ============================================

def _mini_mlp(ax, x0, y0, w, h):
    """Small 10-10-10-2 network glyph (4-4-4-2 nodes drawn) with input/output labels."""
    xs = np.linspace(x0 + 0.3, x0 + w - 0.2, 4)
    layers = [np.linspace(y0 + h - 0.05, y0 + 0.05, n) for n in (4, 4, 4, 2)]
    layers[3] = np.array([y0 + 0.65 * h, y0 + 0.35 * h])
    for li in range(3):
        for ya in layers[li]:
            for yb in layers[li + 1]:
                ax.plot([xs[li], xs[li + 1]], [ya, yb], color="#C4C4C4", lw=0.3, zorder=2)
    for li, (x, yy) in enumerate(zip(xs, layers)):
        fc = ("#DDE8F3", "#EDEDED", "#EDEDED", "#FBE3D6")[li]
        for y in yy:
            ax.add_patch(Circle((x, y), 0.028, fc=fc, ec=INK, lw=0.4, zorder=3))
    ax.text(xs[0] - 0.06, y0 + h / 2, "$d_q, \\psi_q$\n$B, \\theta$", ha="right", va="center",
            fontsize=6.3, linespacing=1.4)
    for y, lab in zip(layers[3], ("$v$", "$\\omega$")):
        ax.text(xs[3] + 0.06, y, lab, ha="left", va="center", fontsize=6.5)


def fig_pipeline():
    gen1 = config.HEBBIAN_CMAES_GEN_MAX
    gen2 = 2 * config.HEBBIAN_CMAES_GEN_MAX
    lam = config.HEBBIAN_CMAES_POPSIZE
    reps = config.HEBBIAN_N_REPEATS
    wd = config.HEBBIAN_EFF_DISTANCE_WEIGHT
    bw, cw, wmult = config.HEBBIAN_STAGE_FITNESS_WEIGHTS["save_battery_avoid_all"][:3]
    lo, hi = config.HEBBIAN_ABCD_BOUNDS
    n_h = config.HEBBIAN_N_HIDDEN
    mlp = f"{config.HEBBIAN_N_INPUTS}-{n_h}-{n_h}-{config.HEBBIAN_N_OUTPUTS}"

    W, H = TEXT_WIDTH, 3.15
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # --- section headers ---
    for x0, x1, label in ((0.05, 3.28, "Evolution in simulation (20 robots, windy corridor)"),
                          (3.43, 4.75, "Evaluation")):
        ax.plot([x0, x1], [H - 0.2, H - 0.2], color=MUTED, lw=0.6)
        ax.text(x0, H - 0.17, label, fontsize=7.5, fontweight="bold", va="bottom")

    # --- curriculum ---
    y, h = 2.12, 0.72
    _pbox(ax, 0.05, y, 0.88, h, "Random rules",
          f"ABCD ~ U[{lo:g}, {hi:g}]\n{config.HEBBIAN_N_ABCD} values,\nshared by all robots")
    _pbox(ax, 1.06, y, 0.86, h, "Stage 1", f"walk upwind\n{gen1} generations\n$f = {wd:g}\\,x_{{avg}}$",
          edge=C_STAGE1, lw=1.1)
    _pbox(ax, 2.05, y, 1.23, h, "Stage 2",
          f"+ battery and collisions\n{gen2} generations\n"
          f"$f = {wd:g}\\,x_{{avg}} + B_{{avg,end}}/{bw:g}$\n"
          f"$-\\,(t_{{col}} + {wmult:g}\\,t_{{wcol}})/{cw:g}$",
          edge=C_STAGE2, lw=1.1, body_size=6.0)
    _arrow(ax, (0.93, y + h / 2), (1.06, y + h / 2), lw=0.8)
    _arrow(ax, (1.92, y + h / 2), (2.05, y + h / 2), lw=0.8)

    # --- evaluation ---
    _pbox(ax, 3.43, y, 1.32, h, "Simulation",
          "vs. hand-designed\nbaseline (Ferrante et al.),\nsame start positions,\n1 to 20 robots")
    _pbox(ax, 3.43, 0.62, 1.32, 1.32, "Real robots",
          "7 Thymio-II robots\n\nLighthouse positions,\nshared by radio,\ngive the same 10 inputs\n\n"
          "wind and battery\ncomputed on board",
          edge=C_HW, lw=1.1)
    # Stage 2's best rules go to both evaluations: split in the gap between the columns.
    xs_split, y_hw = 3.355, 1.28
    ax.plot([3.28, xs_split, xs_split], [y + h / 2, y + h / 2, y_hw], color=INK, lw=0.8, zorder=1)
    _arrow(ax, (xs_split - 0.02, y + h / 2), (3.43, y + h / 2), lw=0.8)
    _arrow(ax, (xs_split - 0.02, y_hw), (3.43, y_hw), lw=0.8)

    # --- one CMA-ES generation, run inside every stage ---
    gx0, gy0, gw, gh = 0.05, 0.05, 3.23, 1.86
    ax.add_patch(FancyBboxPatch((gx0, gy0), gw, gh, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="white", ec=EDGE, lw=0.7, ls="--", zorder=0))
    ax.text(gx0 + 0.07, gy0 + gh - 0.06, "Each stage, every generation (CMA-ES)", fontsize=7,
            fontweight="bold", va="top")
    # Stages 1 and 2 both run this loop.
    for xs in (1.49, 2.67):
        ax.plot([xs, xs], [y, gy0 + gh], color=EDGE, lw=0.7, ls="--", zorder=0)

    steps = (("Sample", f"$\\lambda = {lam}$\ncandidate\nrule sets"),
             ("Simulate", f"{reps} runs each,\n20 robots,\nrandom starts"),
             ("Score", f"median\nfitness of\nthe {reps} runs"),
             ("Update", "move search\ndistribution;\nkeep best-ever"))
    bw_, bh_, by = 0.68, 0.62, 1.0
    bxs = np.linspace(gx0 + 0.08, gx0 + gw - 0.08 - bw_, len(steps))
    for x, (t, b) in zip(bxs, steps):
        _pbox(ax, x, by, bw_, bh_, t, b, body_size=6.0)
    for xa, xb in zip(bxs[:-1], bxs[1:]):
        _arrow(ax, (xa + bw_, by + bh_ / 2), (xb, by + bh_ / 2), lw=0.8)
    _arrow(ax, (bxs[-1] + bw_ / 2, by), (bxs[0] + bw_ / 2, by), rad=-0.13, color=MUTED, lw=0.7)
    ax.text(gx0 + gw / 2, by - 0.2, "next generation", fontsize=5.8, color=MUTED, ha="center",
            va="center", bbox=dict(fc="white", ec="none", pad=0.5), zorder=3)

    # --- what runs inside every simulated robot ---
    cx0, cy0, cw_, ch_ = gx0 + 0.08, gy0 + 0.07, gw - 0.16, 0.62
    ax.add_patch(FancyBboxPatch((cx0, cy0), cw_, ch_, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="#FBFBFB", ec=EDGE, lw=0.6, zorder=1))
    ax.plot([bxs[1] + bw_ / 2, bxs[1] + bw_ / 2], [by, cy0 + ch_], color=EDGE, lw=0.6, ls=":",
            zorder=1)
    _mini_mlp(ax, cx0 + 0.32, cy0 + 0.1, 0.95, ch_ - 0.2)
    tx = cx0 + 1.33
    ax.text(tx, cy0 + ch_ - 0.06, f"Each robot: {mlp} network,\nrandom start weights; "
            "every step:", fontsize=6.3, va="top", linespacing=1.3)
    ax.text(tx, cy0 + 0.09,
            r"$\delta w_{ij} = \mu(A_{ij} n_i n_j + B_{ij} n_i + C_{ij} n_j + D_{ij})$",
            fontsize=6.5, va="bottom")

    _savefig(fig, "paper_method_overview")


# ============================================ Fig. 2: sensing ============================================

def fig_sensing():
    """The 4-quadrant sensor of sensor_model.py, in the focal robot's frame (heading = up).

    Plot angle phi = 90 deg + the sensor's rel_angle. Quadrant q's bearing psi_q is measured
    counter-clockwise from the quadrant's clockwise edge (front: rel_angle + pi/4, etc.)."""
    R = config.HEBBIAN_SENSING_RADIUS
    fig, ax = plt.subplots(figsize=(2.55, 2.85))
    fig.subplots_adjust(0, 0.13, 1, 1)
    # (name, q, start phi, wedge colour); quadrant numbering follows the paper's text.
    quads = (("front", 1, 45, "#DDE8F3"), ("left", 2, 135, "#E9F2DF"),
             ("back", 3, 225, "#F4E4F0"), ("right", 4, -45, "#FBEBD8"))
    for name, q, a0, col in quads:
        ax.add_patch(Wedge((0, 0), R, a0, a0 + 90, fc=col, ec="white", lw=1.2, zorder=0))
        am = np.deg2rad(a0 + 45)
        ax.text(0.8 * R * np.cos(am), 0.8 * R * np.sin(am), f"{name}\n$q={q}$", ha="center",
                va="center", fontsize=6.5, color=MUTED, linespacing=1.2)
    ax.add_patch(Circle((0, 0), R, fill=False, ec=INK, lw=0.6, ls="--"))
    ax.text(R * np.cos(np.deg2rad(-62)) + 0.03, R * np.sin(np.deg2rad(-62)) - 0.03, "$R$",
            fontsize=7.5, ha="left", va="top")

    def at(phi_deg, d):
        return d * np.cos(np.deg2rad(phi_deg)), d * np.sin(np.deg2rad(phi_deg))

    # (quadrant q, phi, distance, label-side): nearest in quadrant -> sensed.
    sensed = ((1, 108, 1.15, -1), (2, 160, 1.3, 1), (3, 256, 1.25, 1))
    ignored = ((1, 66, 1.72), (3, 295, 1.55))
    beyond = ((4, 12, 2.3), (2, 205, 2.3))
    r_bot = 0.085
    for q, phi, d, side in sensed:
        x, y = at(phi, d)
        ax.plot([0, x], [0, y], color=INK, lw=0.9, zorder=2)
        ax.add_patch(Circle((x, y), r_bot, fc=INK, ec=INK, zorder=3))
        xm, ym = at(phi + side * 6, 0.72 * d)
        ax.text(xm, ym, f"$d_{q}$", fontsize=7.5, ha="center", va="center", zorder=4)
        a_start = (45, 135, 225)[q - 1]
        r_arc = (0.6, 0.5, 0.45)[q - 1]
        ax.add_patch(Arc((0, 0), 2 * r_arc, 2 * r_arc, theta1=a_start, theta2=phi, color=C_STAGE2,
                         lw=0.9, zorder=3))
        xa, ya = at((a_start + phi) / 2, r_arc + 0.17)
        ax.text(xa, ya, f"$\\psi_{q}$", fontsize=7.5, color=C_STAGE2, ha="center", va="center",
                zorder=4)
    for q, phi, d in ignored:
        ax.add_patch(Circle(at(phi, d), r_bot, fc="#AAAAAA", ec="#AAAAAA", zorder=3))
    for q, phi, d in beyond:
        ax.add_patch(Circle(at(phi, d), r_bot, fc="white", ec=EDGE, lw=0.7, ls=":", zorder=3))
    # Right quadrant is empty: its reading defaults to d_4 = R, psi_4 = 0.
    x4, y4 = at(16, R)
    ax.annotate("", xy=(x4, y4), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, ls="--", mutation_scale=7))
    ax.text(*at(31, 0.56 * R), "$d_4 = R$\n$\\psi_4 = 0$", fontsize=6.5, ha="center",
            va="center", linespacing=1.2)

    ax.add_patch(Circle((0, 0), 0.11, fc=C_STAGE2, ec=INK, lw=0.6, zorder=5))
    ax.annotate("", xy=(0, 0.42), xytext=(0, 0.11),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, mutation_scale=8), zorder=5)
    ax.set_xlim(-R - 0.35, R + 0.35); ax.set_ylim(-R - 0.2, R + 0.35)
    ax.set_aspect("equal"); ax.axis("off")

    handles = [plt.Line2D([], [], marker="o", ls="-", lw=0.9, color=INK, ms=4, label="sensed (nearest in quadrant)"),
               plt.Line2D([], [], marker="o", ls="", color="#AAAAAA", ms=4, label="not nearest: ignored"),
               plt.Line2D([], [], marker="o", ls="", mfc="white", mec=EDGE, ms=4, label="beyond $R$: ignored"),
               plt.Line2D([], [], marker="o", ls="", mfc=C_STAGE2, mec=INK, ms=5, label="focal robot, arrow = heading")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=5.8, frameon=False,
               handlelength=1.4, columnspacing=0.8, borderaxespad=0.2)
    _savefig(fig, "paper_method_sensing")


# ============================================ Fig. 3: wind model ============================================

def fig_wind():
    """A robot downwind of one robot vs. of a column of three, on the simulation's grid; each
    robot is labelled with the wind the simulator applies to it (agent_wind_percentage)."""
    W, H = TEXT_WIDTH, 1.75
    fig = plt.figure(figsize=(W, H))

    def axes_in(x0, y0, w, h):  # axes placed in inches, so the equal-aspect panels line up
        return fig.add_axes([x0 / W, y0 / H, w / W, h / H])

    x_lim, y_lim = (-0.8, 3.2), (-1.2, 1.2)  # 4 m x 2.4 m -> 1.9 in x 1.14 in
    x_range = [-2.0, -2.0 + config.WIND_TRACKING_WINDOW_WIDTH]
    behind = (0.6, 0.0)
    cases = (("(a) Behind one robot", [(0.0, 0.0)]),
             ("(b) Behind a column of three", [(0.0, -0.3), (0.0, 0.0), (0.0, 0.3)]))
    for k, (title, front) in enumerate(cases):
        ax = axes_in(0.42 + 2.05 * k, 0.32, 1.9, 1.14)
        pos = np.array(front + [behind])
        agents = np.zeros((len(pos), 4)); agents[:, :2] = pos
        yv, xv, pv = RayTraceCircularRobots(agents, config.WIND_RAD, config.UINF, x_range,
                                            list(config.Y_RANGE), 50, 50)
        felt = agent_wind_percentage(agents, config.WIND_RAD, xv, yv, pv, len(pos))
        im = ax.pcolormesh(xv, yv, pv.T, cmap=WIND_CMAP, vmin=0, vmax=config.UINF,
                           shading="auto", rasterized=True)
        for x, y in pos:
            ax.add_patch(Circle((x, y), config.ROBOT_RAD * 2, fc=C_STAGE2, ec=INK, lw=0.4, zorder=3))
        label = dict(fontsize=6.5, va="center", zorder=4,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        top = max(y for _, y in front)
        ax.text(front[0][0], top + 0.27, f"{felt[0]:.0f}%" + (" each" if len(front) > 1 else ""),
                ha="center", **label)
        ax.annotate(f"{felt[-1]:.0f}%", xy=(behind[0], behind[1] - 0.12),
                    xytext=(behind[0] + 0.25, behind[1] - 0.6), fontweight="bold",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=INK), **label)
        ax.annotate("", xy=(-0.35, 0.95), xytext=(-0.75, 0.95),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=7))
        ax.text(-0.3, 0.95, "wind", fontsize=6.3, va="center")
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)
        ax.set_xlabel("x [m]", labelpad=1)
        if k == 0:
            ax.set_ylabel("y [m]", labelpad=1)
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(title, fontsize=7.5, fontweight="bold", loc="left", pad=2)

    cb = fig.colorbar(im, cax=axes_in(4.42, 0.32, 0.07, 1.14))
    cb.set_label("$U^\\%$ [%]", fontsize=6.5, labelpad=1)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)
    _savefig(fig, "paper_method_wind")


if __name__ == "__main__":
    fig_pipeline()
    fig_sensing()
    fig_wind()
