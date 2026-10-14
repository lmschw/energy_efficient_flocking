"""Builds the silent promo video (promo_video.mp4, 1920x1080, 30 fps, no audio track).

Each segment is rendered to segments/NN_name.mp4 (in parallel), then they are concatenated.
Simulation footage is re-rendered from runs/*.npz with render_videos.py's Panel/Header
classes; plots are redrawn from the paper's result JSONs; the hardware clips
(baseline_13.mp4, evolved_24.mp4) are cropped, sped up and composited without their audio.

Usage: python make_promo.py [--only 03_wind 09_hardware ...] [--preview]
  --preview writes one PNG per segment (at its middle) to segments/previews/ instead of video.
"""
import argparse
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.collections import EllipseCollection, LineCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, FancyArrow, Wedge  # noqa: E402

import render_videos as rv  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.dirname(SCRIPT_DIR)
SUMMARY_DIR = os.path.join(FINAL_DIR, "overleaf_summary")
SEG_DIR = os.path.join(SCRIPT_DIR, "segments")
OUT_PATH = os.path.join(SCRIPT_DIR, "promo_video.mp4")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(FINAL_DIR)), "ants26_replication", "experiment"))
import config  # noqa: E402
from wind_physics import RayTraceCircularRobots, agent_wind_percentage  # noqa: E402

FPS, W, H = rv.FPS, rv.W_PX, rv.H_PX
FADE_S = 0.4
INK, MUTED = rv.INK, rv.MUTED
C_EVO, C_BASE = "#D95319", "#77AC30"          # same colours as the paper's figures
TITLE = "Energy-Efficient Flocking in Self-Organized Robot Swarms"
AUTHORS = ("Lilly Schwarzenbach, Sina Mahdavi, Georges Jetti, Michael Khayyat, "
           "Francesco Braghin, Eliseo Ferrante")
HW_SPEED = 2.5
HW_T = 57.5                                   # seconds of each hardware trial shown


# ---------------------------------------------------------------- helpers

def new_fig():
    fig = Figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
    return fig, FigureCanvasAgg(fig)


def grab(canvas):
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()


def caption(fig, text, y=0.062, size=30):
    return fig.text(0.5, y, text, ha="center", va="center", fontsize=size, color=INK)


def _trimmed(path):
    img = mpimg.imread(path)
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0
    rgb = img[:, :, :3]
    ink = rgb.min(axis=2) < 0.92
    if img.shape[2] == 4:
        ink &= img[:, :, 3] > 0.1
    ys, xs = np.where(ink)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


LOGOS = ("vu_logo.png", "polimi.jpeg", "nyuad.png")


def logos(fig, y0, h):
    """Three logos centred in a row, each h (figure fraction) tall."""
    imgs = [_trimmed(os.path.join(SCRIPT_DIR, p)) for p in LOGOS]
    widths = [h * H * im.shape[1] / im.shape[0] / W for im in imgs]
    gap = 0.06
    x = 0.5 - (sum(widths) + gap * (len(imgs) - 1)) / 2
    for im, w in zip(imgs, widths):
        ax = fig.add_axes((x, y0, w, h))
        ax.imshow(im, interpolation="lanczos")
        ax.axis("off")
        x += w + gap


def style_axes(ax, size=20):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9aa4b1")
    ax.tick_params(colors=MUTED, labelsize=size)
    ax.grid(axis="y", color="#e3e7ec", lw=1)
    ax.set_axisbelow(True)


class Static:
    """A segment made of a few static frames, crossfading between them at given times."""

    def __init__(self, states, duration):
        self.states, self.n_frames = states, int(round(duration * FPS))   # states: [(t_start, frame)]

    def frame(self, i):
        t = i / FPS
        k = max(j for j, (ts, _) in enumerate(self.states) if ts <= t)
        cur = self.states[k][1]
        if k > 0 and t - self.states[k][0] < 0.6:
            a = (t - self.states[k][0]) / 0.6
            return (self.states[k - 1][1] * (1 - a) + cur * a).astype(np.uint8)
        return cur


# ---------------------------------------------------------------- 01 title

def seg_title():
    fig, cv = new_fig()
    fig.text(0.5, 0.66, "Energy-Efficient Flocking in\nSelf-Organized Robot Swarms", ha="center", va="center",
             fontsize=54, fontweight="bold", color=INK, linespacing=1.25)
    fig.text(0.5, 0.53, AUTHORS, ha="center", fontsize=23, color=MUTED)
    fig.text(0.5, 0.43, "Robots that take turns at the windy front, like birds in a flock",
             ha="center", fontsize=27, color=INK, style="italic")
    logos(fig, 0.12, 0.12)
    return Static([(0, grab(cv))], 6.0)


# ---------------------------------------------------------------- simulation clip segments

class SimSegment:
    """Simulation footage with timed captions.
    layout 'single' (one run) or 'pair' (baseline | evolved); captions: [(t_video, text)]."""

    def __init__(self, n, conds, mode, speed, t0, duration, captions, show_final=False, hold=0.0):
        self.runs = [rv.Run(c, n) for c in conds]
        self.speed, self.t0, self.hold = speed, t0, hold
        self.sim_end = t0 + duration * speed
        self.n_frames = int(round((duration + hold) * FPS))
        self.captions = captions
        fig, self.cv = new_fig()
        self.panels, self.headers = [], []
        if len(conds) == 1:
            self.panels.append(rv.Panel(fig, self.runs[0], (0.025, 0.135, 0.95, 0.70), rv.VIEW_W[n][0], mode))
            self.headers.append(rv.Header(fig, self.runs[0], (0.03, 0.865, 0.94, 0.11), mode, speed))
            leg_x, leg_y, scale = 0.44, 0.885, 1.0
        else:
            scale = 0.85
            for i, r in enumerate(self.runs):
                left = 0.02 + i * 0.49
                self.panels.append(rv.Panel(fig, r, (left, 0.125, 0.47, 0.67), rv.VIEW_W[n][1], mode,
                                            header_scale=scale))
                self.headers.append(rv.Header(fig, r, (left + 0.005, 0.815, 0.46, 0.1), mode, speed,
                                              scale=scale))
            leg_x, leg_y = 0.33, 0.945
        for p in self.panels:
            p.show_final = show_final
        if mode == "battery":
            rv._colorbar(fig, (leg_x, leg_y, 0.11, 0.022), scale)
            rv._wind_legend(fig, (leg_x + 0.17, leg_y, 0.11, 0.022), scale)
        else:
            rv._wind_legend(fig, (leg_x + 0.07, leg_y, 0.11, 0.022), scale)
        self.cap = caption(fig, "")

    def frame(self, i):
        tv = i / FPS
        t = min(self.t0 + tv * self.speed, self.sim_end)
        if tv * self.speed + self.t0 >= self.sim_end:          # holding on the last frame
            t = max(r.t_end for r in self.runs) + 1 if self.hold else t
        for p in self.panels:
            p.update(t)
        for h in self.headers:
            h.update(t)
        self.cap.set_text(max((c for c in self.captions if c[0] <= tv), key=lambda c: c[0])[1])
        return grab(self.cv)


def seg_flocking():
    return SimSegment(7, ("baseline",), "single", 1, 6.0, 7.0,
                      [(0, "Flocking: robots move together as a group, using only local rules.")])


def seg_baseline_wind():
    return SimSegment(7, ("baseline",), "battery", 15, 0.0, 14.0, [
        (0, "Robots in front face the full wind and drain their battery fastest."),
        (6.5, "A classic flocking model keeps the same robots in front..."),
        (11.5, "...so the whole group must stop when the first robot runs out."),
    ], show_final=True, hold=2.5)


def seg_evolved_realtime():
    return SimSegment(20, ("baseline", "evolved"), "single", 1, 20.0, 9.0, [
        (0, "Our evolved controller: robots keep looping around each other..."),
        (4.5, "...and keep swapping which robot is exposed at the front."),
    ])


def seg_evolved_full():
    return SimSegment(20, ("baseline", "evolved"), "battery", 20, 0.0, 408 / 20, [
        (0, "20 robots, same wind, same start, until the first robot runs out of battery."),
        (9.0, "The evolved swarm runs out sooner, but covers much more ground."),
        (408 / 20, "The evolved swarm travels almost twice as far."),
    ], show_final=True, hold=3.0)


# ---------------------------------------------------------------- 03 wind explainer

class WindExplainer:
    """Two robots and the simulator's own wind model (RayTraceCircularRobots): the second
    robot slides into the first one's wake. Streaks move with the local wind speed."""

    VIEW = (-2.3, 2.3)

    def __init__(self):
        self.n_frames = int(round(11.0 * FPS))
        fig, self.cv = new_fig()
        self.ax = fig.add_axes((0.025, 0.135, 0.95, 0.73))
        ax = self.ax
        ax.set_facecolor(rv.BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#c9ced6")
        vw = self.VIEW[1] - self.VIEW[0]
        vh = vw * (0.73 * H) / (0.95 * W)
        ax.set_xlim(*self.VIEW)
        ax.set_ylim(-vh / 2, vh / 2)
        self.vh = vh
        self.im = ax.imshow(np.full((2, 2), 100.0), extent=(-5, 5, -5, 5), origin="lower", cmap=rv.WIND_CMAP,
                            vmin=rv.WIND_VMIN, vmax=rv.WIND_VMAX, interpolation="bilinear", zorder=0,
                            aspect="auto")
        rng = np.random.default_rng(1)
        self.p = np.column_stack([rng.uniform(-2.6, 2.6, 150), rng.uniform(-vh / 2, vh / 2, 150)])
        self.streaks = LineCollection([], colors="#a3adb9", linewidths=1.6, zorder=1, capstyle="round")
        ax.add_collection(self.streaks)
        self.robots = EllipseCollection(widths=0.11, heights=0.11, angles=0, units="xy",
                                        offsets=np.zeros((2, 2)), offset_transform=ax.transData,
                                        facecolors=rv.SINGLE_COLOR, edgecolors=INK, linewidths=1.2, zorder=3)
        ax.add_collection(self.robots)
        self.heads = LineCollection([], colors="white", linewidths=3, zorder=4, capstyle="round")
        ax.add_collection(self.heads)
        self.labels = [ax.text(0, 0, "", ha="center", va="bottom", fontsize=24, color=INK, zorder=5,
                               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c9ced6", alpha=0.9))
                       for _ in range(2)]
        fig.text(0.03, 0.925, "How wind costs energy", fontsize=34, fontweight="bold", color=INK, va="center")
        fig.text(0.97, 0.925, "wind model used in the simulations", fontsize=19, color=MUTED, va="center",
                 ha="right")
        ax.add_patch(FancyArrow(-2.2, vh / 2 - 0.22, 0.45, 0, width=0.035, head_width=0.11, head_length=0.08,
                                color="#7b8794", zorder=6, length_includes_head=True))
        ax.text(-2.2, vh / 2 - 0.12, "wind", fontsize=20, color=MUTED, zorder=6)
        rv._wind_legend(fig, (0.44, 0.885, 0.11, 0.022))
        self.cap = caption(fig, "")
        self.prev_t = 0.0

    def robots_at(self, t):
        a = np.clip((t - 3.0) / 3.5, 0, 1)
        a = a * a * (3 - 2 * a)                          # smoothstep
        front = (-0.55, 0.0)
        back = (-0.55 + 0.65 * a, 0.75 * (1 - a))        # from beside the front robot into its wake
        agents = np.zeros((2, 4))
        agents[:, :2] = [front, back]
        agents[:, 2] = np.pi / 2                         # heading (-sin, cos) -> facing upwind (-x)
        return agents

    def frame(self, i):
        t = i / FPS
        agents = self.robots_at(t)
        yv, xv, pw = RayTraceCircularRobots(agents, config.WIND_RAD, config.UINF, [-5.0, 5.0], [-5.0, 5.0], 50, 50)
        dx, dy = xv[1] - xv[0], yv[1] - yv[0]
        self.im.set_data(pw.T)
        self.im.set_extent((xv[0] - dx / 2, xv[-1] + dx / 2, yv[0] - dy / 2, yv[-1] + dy / 2))
        u = agent_wind_percentage(agents, config.WIND_RAD, xv, yv, pw, 2)

        # streaks advected by the local wind
        ix = np.clip(np.searchsorted(xv, self.p[:, 0]), 0, len(xv) - 1)
        iy = np.clip(np.searchsorted(yv, self.p[:, 1]), 0, len(yv) - 1)
        speed = pw[ix, iy] / 100.0
        self.p[:, 0] += speed * 1.4 / FPS
        wrap = self.p[:, 0] > 2.6
        self.p[wrap, 0] -= 5.2
        near = np.min(np.linalg.norm(self.p[:, None, :] - agents[None, :, :2], axis=2), axis=1) < 0.09
        segs = np.stack([self.p, self.p + np.column_stack([0.07 + 0.16 * speed, 0 * speed])], axis=1)
        self.streaks.set_segments(segs[~near])

        xy = agents[:, :2]
        self.robots.set_offsets(xy)
        hd = np.array([[-1.0, 0.0]] * 2)
        self.heads.set_segments(np.stack([xy + 0.008 * hd, xy + 0.052 * hd], axis=1))
        for k, lab in enumerate(self.labels):
            lab.set_position((xy[k, 0], xy[k, 1] + 0.1))
            lab.set_text(f"wind {u[k]:.0f} %")
        self.cap.set_text("Grey: air slowed down behind a robot (its wake)." if t < 3.0 else
                          "A robot in another's wake feels less wind, and uses less than half the battery."
                          if t < 7.5 else "The robot in front pays the most.")
        return grab(self.cv)


# ---------------------------------------------------------------- 05 approach card

def seg_approach():
    fig, cv = new_fig()
    fig.text(0.05, 0.86, "Our approach", fontsize=40, fontweight="bold", color=INK)
    lines = [
        ("Each robot senses only:", True),
        ("•  the nearest neighbour in each of 4 directions\n    (distance and bearing)", False),
        ("•  its own battery level", False),
        ("•  a compass", False),
        ("No wind sensor  ·  no communication  ·  no memory", True),
        ("A small neural network whose learning rules\nare evolved, with 20 robots in simulation.", False),
    ]
    y = 0.73
    for text, bold in lines:
        fig.text(0.05, y, text, fontsize=27, color=INK if bold else "#33404f",
                 fontweight="bold" if bold else "normal", va="top", linespacing=1.35)
        y -= 0.115 if "\n" in text else 0.085
        if bold and text.startswith("No wind"):
            y -= 0.02

    # sensing diagram: four quadrants centred on the robot's front, nearest neighbour in each
    ax = fig.add_axes((0.63, 0.13, 0.33, 0.33 * W / H))
    ax.set_aspect("equal")
    ax.axis("off")
    R = 2.0
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)
    names = {0: "front", 90: "left", 180: "back", 270: "right"}
    shades = ["#eef2f7", "#f7f8fa"]
    for k, c in enumerate((180, 90, 0, 270)):   # robot faces left (-x), so its front quadrant is centred on 180 deg
        ax.add_patch(Wedge((0, 0), R, c - 45, c + 45, fc=shades[k % 2], ec="#c9ced6", lw=1.5))
    for c, label in ((180, "front"), (90, "right"), (0, "back"), (270, "left")):
        r = R + 0.22
        ax.text(r * np.cos(np.radians(c)), r * np.sin(np.radians(c)), label, ha="center", va="center",
                fontsize=19, color=MUTED, rotation=0)
    nbrs = np.array([[-1.2, 0.35], [-1.7, -0.6], [0.3, 1.1], [0.9, 1.6], [1.3, -0.2], [1.6, 0.9],
                     [-0.2, -1.4], [0.6, -1.7]])
    nearest = [0, 2, 4, 6]
    for j, (x, y) in enumerate(nbrs):
        hit = j in nearest
        if hit:
            ax.plot([0, x], [0, y], color=C_EVO, lw=3, zorder=2)
        ax.add_patch(Circle((x, y), 0.13, fc=rv.SINGLE_COLOR if hit else "#aab3bf", ec=INK, lw=1.2, zorder=3))
    ax.add_patch(Circle((0, 0), 0.17, fc=C_EVO, ec=INK, lw=1.5, zorder=4))
    ax.plot([-0.03, -0.16], [0, 0], color="white", lw=4, zorder=5, solid_capstyle="round")
    ax.text(0.32, -0.3, "robot", ha="center", va="top", fontsize=18, color=INK, zorder=5)
    return Static([(0, grab(cv))], 10.0)


# ---------------------------------------------------------------- 07 result plots

def _plot_fig(title):
    fig, cv = new_fig()
    fig.text(0.05, 0.9, title, fontsize=36, fontweight="bold", color=INK, va="center")
    return fig, cv


def seg_scatter():
    d = json.load(open(os.path.join(SUMMARY_DIR, "paper_results.json")))["controllers"]
    base, evo = d["LJ baseline"]["per_run"], d["Stage 2"]["per_run"]
    states = []
    for show_evo in (False, True):
        fig, cv = _plot_fig("100 simulations with 20 robots each")
        ax = fig.add_axes((0.09, 0.2, 0.55, 0.62))
        style_axes(ax)
        ax.grid(axis="x", color="#e3e7ec", lw=1)
        ax.scatter(base["dist"], base["batt"], s=70, color=C_BASE, alpha=0.75, lw=0)
        if show_evo:
            ax.scatter(evo["dist"], evo["batt"], s=70, color=C_EVO, alpha=0.75, lw=0)
        ax.set_xlim(5, 26)
        ax.set_ylim(7, 37)
        ax.set_xlabel("distance travelled against the wind [m]", fontsize=24, color=INK)
        ax.set_ylabel("battery left at the end [%]", fontsize=24, color=INK)
        fig.text(0.68, 0.70, "Baseline flocking model", fontsize=28, color=C_BASE, fontweight="bold")
        fig.text(0.68, 0.655, f"{np.mean(base['dist']):.1f} m  ·  {np.mean(base['batt']):.0f} % battery left",
                 fontsize=24, color=INK)
        if show_evo:
            fig.text(0.68, 0.53, "Evolved controller", fontsize=28, color=C_EVO, fontweight="bold")
            fig.text(0.68, 0.485, f"{np.mean(evo['dist']):.1f} m  ·  {np.mean(evo['batt']):.0f} % battery left",
                     fontsize=24, color=INK)
            fig.text(0.68, 0.36, "127 % further,\nwith more battery left", fontsize=34, color=INK,
                     fontweight="bold", linespacing=1.3)
        caption(fig, "Each dot is one simulation, run until the first robot's battery is empty.", size=26)
        states.append(grab(cv))
    return Static([(0, states[0]), (2.0, states[1])], 8.5)


def seg_front_changes():
    rows = json.load(open(os.path.join(FINAL_DIR, "leadership_analysis", "seed_sweep_table_n20.json")))
    vals = {c: [60 * r["persistence_filtered_rate_per_sec"] for r in rows if r["condition"] == c]
            for c in ("lj_baseline", "plain_seed123")}
    fig, cv = _plot_fig("Who is at the front?  (20 robots)")
    ax = fig.add_axes((0.12, 0.2, 0.4, 0.62))
    style_axes(ax)
    rng = np.random.default_rng(0)
    for k, (c, col) in enumerate((("lj_baseline", C_BASE), ("plain_seed123", C_EVO))):
        v = np.array(vals[c])
        ax.scatter(k + rng.uniform(-0.17, 0.17, len(v)), v, s=80, color=col, alpha=0.8, lw=0, zorder=3)
        ax.hlines(np.median(v), k - 0.3, k + 0.3, color=INK, lw=4, zorder=4)
        ax.text(k + 0.34, np.median(v), f"{np.median(v):.1f}", fontsize=26, color=INK, va="center",
                fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Evolved"], fontsize=26, color=INK)
    ax.set_xlim(-0.6, 1.8)
    ax.set_ylim(0, 16)
    ax.set_ylabel("changes of the front robot per minute", fontsize=24, color=INK)
    fig.text(0.6, 0.62, "The robot at the exposed\nfront changes about\n10 times per minute,", fontsize=32,
             color=INK, linespacing=1.35)
    fig.text(0.6, 0.43, "almost 5× as often as\nwith the baseline.", fontsize=32, color=INK, fontweight="bold",
             linespacing=1.35)
    caption(fig, "Each dot is one simulation (30 per controller). Only front positions held for at least 1.5 s count.",
            size=24)
    return Static([(0, grab(cv))], 7.5)


def seg_swarm_size():
    d = json.load(open(os.path.join(SUMMARY_DIR, "n_agents_sweep_comparison_plain_seed123_no_clamp.json")))
    states = []
    for show_note in (False, True):
        fig, cv = _plot_fig("It pays off in larger swarms")
        ax = fig.add_axes((0.09, 0.2, 0.55, 0.62))
        style_axes(ax)
        for key, col, label in (("lj_baseline", C_BASE, "Baseline"), ("plain_seed123", C_EVO, "Evolved")):
            ns = sorted(int(k) for k in d[key])
            m = np.array([d[key][str(n)]["dist_mean"] for n in ns])
            s = np.array([d[key][str(n)]["dist_std"] for n in ns])
            ax.fill_between(ns, m - s, m + s, color=col, alpha=0.18, lw=0)
            ax.plot(ns, m, "o-", color=col, lw=3.5, ms=9)
            ax.text(ns[-1] + 0.4, m[-1], label, color=col, fontsize=26, fontweight="bold", va="center")
        if show_note:
            ax.axvspan(12.5, 20.5, color="#fbe3d6", alpha=0.6, zorder=0)
        ax.set_xlim(0.5, 22.8)
        ax.set_ylim(0, 28)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel("number of robots", fontsize=24, color=INK)
        ax.set_ylabel("distance travelled [m]", fontsize=24, color=INK)
        fig.text(0.68, 0.64, "Evolved with 20 robots.", fontsize=30, color=INK)
        if show_note:
            fig.text(0.68, 0.49, "It overtakes the baseline\nat about 12 robots.", fontsize=30, color=INK,
                     fontweight="bold", linespacing=1.35)
            fig.text(0.68, 0.36, "In small swarms the\nbaseline goes further.", fontsize=30, color=INK,
                     linespacing=1.35)
        caption(fig, "Mean ± one standard deviation over 30 simulations per swarm size.", size=24)
        states.append(grab(cv))
    return Static([(0, states[0]), (2.0, states[1])], 8.0)


# ---------------------------------------------------------------- 08-10 hardware

def seg_hw_intro():
    fig, cv = new_fig()
    fig.text(0.5, 0.74, "Real robots", ha="center", fontsize=52, fontweight="bold", color=INK)
    items = ["7 Thymio-II robots, each running the controller on board",
             "Positions from Lighthouse tracking, shared between robots by radio",
             "Wind and battery are simulated on each robot",
             "30 trials of 60 s per controller"]
    for k, s in enumerate(items):
        fig.text(0.5, 0.58 - k * 0.09, s, ha="center", fontsize=30, color="#33404f")
    return Static([(0, grab(cv))], 6.0)


def _ffmpeg():
    return rv._ffmpeg()


HW_FILES = ("baseline_13.mp4", "evolved_24.mp4")


def _track_robots(path, min_w=760, margin=110):
    """Crop window (cx, cy, w) at 2 Hz of source time that follows the robots: they are found
    by differencing each frame against the clip's median frame (the robots move, so the
    median is the empty arena), and the window is their bounding box plus a margin, never
    narrower than min_w, widened by a rolling max and then smoothed so it stays steady."""
    from scipy import ndimage
    w, h, f = 480, 270, 4
    raw = subprocess.run([_ffmpeg(), "-v", "error", "-t", str(HW_T + 1), "-i", path, "-an", "-vf",
                          f"fps=2,scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.int16)
    bg = np.median(frames, axis=0)
    boxes = []
    for fr in frames:
        m = np.abs(fr - bg).sum(axis=2) > 90
        m[:, :40] = False                                  # dark curtain at the left edge
        m = ndimage.binary_opening(m)
        lab, n = ndimage.label(m)
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        objs = [o for o, sz in zip(ndimage.find_objects(lab), sizes) if sz >= 6]
        boxes.append((min(o[1].start for o in objs) * f, max(o[1].stop for o in objs) * f,
                      min(o[0].start for o in objs) * f, max(o[0].stop for o in objs) * f))
    b = np.array(boxes, float)
    cx, cy = (b[:, 0] + b[:, 1]) / 2, (b[:, 2] + b[:, 3]) / 2
    need = np.maximum(b[:, 1] - b[:, 0] + 2 * margin, (b[:, 3] - b[:, 2] + 2 * margin) * 16 / 9)
    need = np.maximum(need, min_w)
    need = ndimage.maximum_filter1d(need, size=7)          # +-1.5 s look-ahead/behind
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2)
    k /= k.sum()
    sm = lambda v: np.convolve(np.pad(v, 6, mode="edge"), k, mode="valid")
    return np.column_stack([sm(cx), sm(cy), sm(need) * 1.04])


class HardwareSegment:
    PANEL_W, PANEL_H = 940, 528

    def __init__(self):
        dur = HW_T / HW_SPEED
        self.n_frames = int(round(dur * FPS))
        self.top = 236
        self.lefts = (13, W - 13 - self.PANEL_W)
        self.caps = [(0, "Baseline: the robots keep their places in the group."),
                     (dur * 0.5, "Evolved: the robots loop around each other and take turns at the front.")]
        self.backgrounds = []
        for _, text in self.caps:
            fig, cv = new_fig()
            for left, name in zip(self.lefts, ("Baseline flocking model", "Evolved controller")):
                fig.text(left / W + 0.006, 0.875, name, fontsize=30 * 0.85, fontweight="bold", color=INK,
                         va="center")
                fig.text(left / W + 0.006, 0.83, f"7 real Thymio-II robots  ·  {HW_SPEED:g}× speed  ·  "
                         "camera follows the robots", fontsize=18 * 0.85, color=MUTED, va="center")
            caption(fig, text, y=0.19)
            self.backgrounds.append(grab(cv))
        self.tracks = [_track_robots(os.path.join(SCRIPT_DIR, f)) for f in HW_FILES]
        self.readers = None

    def _open(self):
        vf = f"setpts=PTS/{HW_SPEED},fps={FPS}"
        self.readers = [subprocess.Popen([_ffmpeg(), "-v", "error", "-t", str(HW_T), "-i",
                                          os.path.join(SCRIPT_DIR, f), "-an", "-vf", vf, "-f", "rawvideo",
                                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
                        for f in HW_FILES]
        self.last = [np.full((self.PANEL_H, self.PANEL_W, 3), 255, np.uint8)] * 2

    def _crop(self, img, track, ts):
        import cv2
        s = min(ts * 2, len(track) - 1.001)
        k = int(s)
        cx, cy, cw = track[k] * (1 - (s - k)) + track[k + 1] * (s - k)
        cw = min(cw, 1920.0)
        ch = cw * self.PANEL_H / self.PANEL_W
        x0 = int(round(np.clip(cx - cw / 2, 0, 1920 - cw)))
        y0 = int(round(np.clip(cy - ch / 2, 0, 1080 - ch)))
        sub = img[y0:y0 + int(round(ch)), x0:x0 + int(round(cw))]
        return cv2.resize(sub, (self.PANEL_W, self.PANEL_H), interpolation=cv2.INTER_AREA)

    def frame(self, i):
        if self.readers is None:
            self._open()
        tv = i / FPS
        k = max(j for j, (t, _) in enumerate(self.caps) if t <= tv)
        out = self.backgrounds[k].copy()
        size = 1920 * 1080 * 3
        for j, proc in enumerate(self.readers):
            buf = proc.stdout.read(size)
            if len(buf) == size:
                img = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3)
                self.last[j] = self._crop(img, self.tracks[j], tv * HW_SPEED)
            out[self.top:self.top + self.PANEL_H, self.lefts[j]:self.lefts[j] + self.PANEL_W] = self.last[j]
        return out


def seg_hw_plot():
    d = json.load(open(os.path.join(SUMMARY_DIR, "paper_hardware_lighthouse_summary.json")))["n30"]
    fig, cv = _plot_fig("Real robots behave like the simulation")
    specs = (("front_occupancy_entropy", "how evenly the front is shared\n(0 = one robot always leads, 1 = all equally)",
              (0, 1.05)),
             ("front_changes_per_min", "changes of the front robot\nper minute", (0, 13.5)))
    rng = np.random.default_rng(0)
    for p, (key, ylabel, ylim) in enumerate(specs):
        ax = fig.add_axes((0.08 + p * 0.47, 0.2, 0.38, 0.6))
        style_axes(ax, 20)
        for g, (c, col) in enumerate((("lj", C_BASE), ("plain", C_EVO))):
            real = np.array([r[key] for r in d[c]["per_run"]])
            sim = d[c]["sim_n7"]["full"][key]
            xr, xs = g * 2.4, g * 2.4 + 1
            ax.scatter(xr + rng.uniform(-0.2, 0.2, len(real)), real, s=55, color=col, alpha=0.8, lw=0, zorder=3)
            ax.hlines(np.median(real), xr - 0.32, xr + 0.32, color=INK, lw=4, zorder=4)
            ax.bar(xs, sim["q75"] - sim["q25"], bottom=sim["q25"], width=0.55, color=col, alpha=0.25,
                   edgecolor=col, hatch="//", lw=1.5, zorder=2)
            ax.hlines(sim["median"], xs - 0.32, xs + 0.32, color=INK, lw=4, zorder=4)
            ax.text(g * 2.4 + 0.5, ylim[1] * 0.985, "Baseline" if c == "lj" else "Evolved", ha="center",
                    va="top", fontsize=24, color=col, fontweight="bold")
        ax.set_xticks([0, 1, 2.4, 3.4])
        ax.set_xticklabels(["real", "sim.", "real", "sim."], fontsize=21, color=INK)
        ax.set_xlim(-0.6, 4.0)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=21, color=INK, linespacing=1.3)
    caption(fig, "On real robots, as in simulation, the evolved controller shares the exposed front far more evenly.",
            size=26)
    fig.text(0.5, 0.115, "Dots: 30 real trials per controller.  Hatched: middle 50 % of 30 matching simulations with 7 robots.",
             ha="center", fontsize=19, color=MUTED)
    return Static([(0, grab(cv))], 9.0)


# ---------------------------------------------------------------- 11 closing

def seg_closing():
    fig, cv = new_fig()
    fig.text(0.5, 0.86, "Key findings", ha="center", fontsize=44, fontweight="bold", color=INK)
    items = ["Turn-taking at the windy front emerges without a wind sensor or communication.",
             "With 20 simulated robots: 127 % further, with more battery left.",
             "On real robots, the front is shared far more evenly than with classic flocking."]
    for k, s in enumerate(items):
        fig.text(0.5, 0.72 - k * 0.085, s, ha="center", fontsize=29, color="#33404f")
    fig.text(0.5, 0.40, TITLE, ha="center", fontsize=30, fontweight="bold", color=INK)
    fig.text(0.5, 0.345, AUTHORS, ha="center", fontsize=21, color=MUTED)
    logos(fig, 0.09, 0.12)
    return Static([(0, grab(cv))], 9.0)


SEGMENTS = [
    ("01_title", seg_title),
    ("02_flocking", seg_flocking),
    ("03_wind", WindExplainer),
    ("04_baseline_wind", seg_baseline_wind),
    ("05_approach", seg_approach),
    ("06_evolved_realtime", seg_evolved_realtime),
    ("07_evolved_full", seg_evolved_full),
    ("08_scatter", seg_scatter),
    ("09_front_changes", seg_front_changes),
    ("10_swarm_size", seg_swarm_size),
    ("11_hw_intro", seg_hw_intro),
    ("12_hardware", HardwareSegment),
    ("13_hw_plot", seg_hw_plot),
    ("14_closing", seg_closing),
]


def render_segment(name, preview=False):
    seg = dict(SEGMENTS)[name]()
    if preview:
        os.makedirs(os.path.join(SEG_DIR, "previews"), exist_ok=True)
        path = os.path.join(SEG_DIR, "previews", f"{name}.png")
        mid = seg.n_frames - 1 if name != "12_hardware" else seg.n_frames // 2
        if isinstance(seg, (HardwareSegment, WindExplainer)):
            for i in range(mid):
                seg.frame(i)
        mpimg.imsave(path, seg.frame(mid))
        return path
    os.makedirs(SEG_DIR, exist_ok=True)
    path = os.path.join(SEG_DIR, f"{name}.mp4")
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", "-r", str(FPS), path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    nf, nfade = seg.n_frames, int(FADE_S * FPS)
    for i in range(nf):
        f = seg.frame(i)
        a = min(1.0, (i + 1) / nfade, (nf - i) / nfade)
        if a < 1.0:
            f = (f.astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    return f"{name}: {nf / FPS:.1f} s"


def concat():
    lst = os.path.join(SEG_DIR, "list.txt")
    with open(lst, "w") as fh:
        for name, _ in SEGMENTS:
            fh.write(f"file '{name}.mp4'\n")
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-an", "-c", "copy", "-movflags", "+faststart", OUT_PATH], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--jobs", type=int, default=14)
    args = ap.parse_args()
    names = args.only or [n for n, _ in SEGMENTS]
    with Pool(min(args.jobs, len(names))) as pool:
        for msg in pool.starmap(render_segment, [(n, args.preview) for n in names]):
            print(msg, flush=True)
    if not args.preview:
        concat()
        print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
