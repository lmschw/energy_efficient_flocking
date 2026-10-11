"""Renders promo-video clips from the episodes recorded by record_runs.py.

For each swarm size (7, 20) and colour mode (battery: blue = full -> red = empty, as a
percentage of each controller's own starting charge; single: one colour throughout):
  realtime_60s/   first 60 s of simulated time, played in real time
  full_episode/   the whole episode until the first robot's battery is empty, sped up
                  by the same factor for both controllers (so their speeds stay comparable)
each as individual clips (baseline, evolved) and as a side-by-side comparison.

Frames are interpolated between the simulator's 2 Hz steps for smooth 30 fps playback
(positions/headings/battery linearly; the wind field is taken from the nearest step).
The grey shading is the simulated wind field: white = free-stream wind, grey = sheltered.

Usage: python render_videos.py [--jobs 16] [--only realtime_60s] [--preview]
"""
import argparse
import os
import subprocess
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.collections import EllipseCollection, LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgba  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyArrow, Rectangle  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
OUT_DIR = os.path.join(SCRIPT_DIR, "videos")
SEED = 42

FPS = 30
W_PX, H_PX = 1920, 1080
ROBOT_RAD = 0.055
TRAIL_S = 4.0
HOLD_S = 2.5              # freeze on the last frame of a full episode
FULL_SPEEDUP = {7: 4, 20: 7}

# Width of the camera view [m] per swarm size: (single clip, side-by-side panel)
VIEW_W = {7: (5.5, 3.6), 20: (8.0, 5.0)}

BG = "#f6f7f9"
INK = "#1d2733"
MUTED = "#5f6b7a"
SINGLE_COLOR = "#2d3e50"
BATTERY_CMAP = LinearSegmentedColormap.from_list(
    "battery", ["#d7301f", "#8e3fa0", "#2166c4"])        # 0 % red -> 50 % purple -> 100 % blue
WIND_CMAP = LinearSegmentedColormap.from_list("wind", ["#8d97a3", BG])  # sheltered -> free stream
WIND_VMIN, WIND_VMAX = 10.0, 100.0

TITLES = {"baseline": "Baseline flocking model", "evolved": "Evolved controller"}


class Run:
    def __init__(self, condition, n):
        d = np.load(os.path.join(RUNS_DIR, f"{condition}_n{n}_seed{SEED}.npz"))
        self.condition, self.n = condition, n
        self.states = d["states"]
        self.x_vals, self.y_vals, self.wind = d["x_vals"], d["y_vals"], d["wind"]
        self.dt = float(d["dt"])
        self.batt_pct = np.clip(self.states[:, :, 3] / float(d["battery_scale"]) * 100.0, 0, 100)
        self.walls_y = d["walls_y"]
        self.n_steps = len(self.states) - 1
        self.t_end = self.n_steps * self.dt
        # Unwrapped headings for interpolation; heading vector is (-sin th, cos th).
        self.theta = np.unwrap(self.states[:, :, 2], axis=0)
        # Camera: median robot position (robust to stragglers), smoothed over ~2 s.
        med = np.median(self.states[:, :, :2], axis=1)
        k = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2)
        k /= k.sum()
        pad = np.pad(med, ((12, 12), (0, 0)), mode="edge")
        self.cam = np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(2)], axis=1)

    def _idx(self, t):
        s = min(max(t / self.dt, 0.0), self.n_steps)
        k = min(int(s), self.n_steps - 1)
        return k, s - k

    def _lerp(self, arr, t):
        k, a = self._idx(t)
        return arr[k] * (1 - a) + arr[k + 1] * a

    def xy(self, t):
        return self._lerp(self.states[:, :, :2], t)

    def heading(self, t):
        th = self._lerp(self.theta, t)
        return np.stack([-np.sin(th), np.cos(th)], axis=1)

    def battery(self, t):
        return self._lerp(self.batt_pct, t)

    def camera(self, t):
        return self._lerp(self.cam, t)

    def wind_at(self, t):
        k, a = self._idx(t)
        k = k + 1 if a >= 0.5 else k
        xv, yv = self.x_vals[k], self.y_vals
        dx, dy = xv[1] - xv[0], yv[1] - yv[0]
        extent = (xv[0] - dx / 2, xv[-1] + dx / 2, yv[0] - dy / 2, yv[-1] + dy / 2)
        return self.wind[k].T, extent

    def trail_segments(self, t):
        k, _ = self._idx(t)
        h = int(TRAIL_S / self.dt)
        pts = self.states[max(0, k - h + 1):k + 1, :, :2]
        pts = np.concatenate([pts, self.xy(t)[None]], axis=0)   # (L, n, 2)
        if len(pts) < 2:
            return np.zeros((0, 2, 2)), np.zeros(0)
        segs = np.stack([pts[:-1], pts[1:]], axis=2)            # (L-1, n, 2, 2)
        age = np.linspace(0.0, 1.0, len(pts) - 1)               # 0 oldest -> 1 newest
        alpha = np.repeat(age[:, None], self.n, axis=1)
        return segs.reshape(-1, 2, 2), alpha.reshape(-1)

    def distance(self, t):
        return -float(np.mean(self.xy(t)[:, 0]))               # the paper's distance metric


class Panel:
    """One arena view of one run, in figure-fraction rect (l, b, w, h)."""

    def __init__(self, fig, run, rect, view_w, mode, header_scale=1.0):
        self.run, self.mode = run, mode
        l, b, w, h = rect
        self.ax = fig.add_axes(rect)
        self.ax.set_facecolor(BG)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for s in self.ax.spines.values():
            s.set_color("#c9ced6")
        self.view_w = view_w
        self.view_h = view_w * (h * H_PX) / (w * W_PX)
        img, extent = run.wind_at(0)
        self.wind_im = self.ax.imshow(img, extent=extent, origin="lower", cmap=WIND_CMAP,
                                      vmin=WIND_VMIN, vmax=WIND_VMAX, interpolation="bilinear",
                                      zorder=0, aspect="auto")
        # 1 m ground grid, so movement is visible while the camera follows the swarm.
        self.grid = LineCollection([], colors="#d5dae1", linewidths=0.8, zorder=1)
        self.ax.add_collection(self.grid)
        self.start_line = self.ax.axvline(0.0, color="#9aa4b1", lw=1.4, ls=(0, (6, 4)), zorder=1)
        self.start_txt = self.ax.text(0.06, 0, "start", color="#8a94a1", fontsize=13 * header_scale,
                                      zorder=1, va="bottom", clip_on=True)
        for y in run.walls_y:  # corridor walls (10 m apart)
            self.ax.add_patch(Rectangle((-1e3, y if y > 0 else y - 50), 2e3, 50,
                                        color="#c3c9d1", zorder=1, lw=0))
        self.trails = LineCollection([], linewidths=1.6, zorder=2, capstyle="round")
        self.ax.add_collection(self.trails)
        xy = run.xy(0)
        self.robots = EllipseCollection(widths=2 * ROBOT_RAD, heights=2 * ROBOT_RAD, angles=0,
                                        units="xy", offsets=xy, offset_transform=self.ax.transData,
                                        edgecolors=INK, linewidths=0.9, zorder=3)
        self.ax.add_collection(self.robots)
        self.heads = LineCollection([], colors="white", linewidths=2.2, zorder=4, capstyle="round")
        self.ax.add_collection(self.heads)
        self.ax.set_aspect("equal", adjustable="box")
        # Wind indicator, top-left of the arena (axes coords).
        self.ax.add_patch(FancyArrow(0.022, 0.93, 0.07 * self.view_h / self.view_w * 1.6, 0,
                                     width=0.012, head_width=0.035, head_length=0.018,
                                     transform=self.ax.transAxes, color="#7b8794", zorder=6,
                                     length_includes_head=True))
        self.ax.text(0.022, 0.965, "wind", transform=self.ax.transAxes, color="#5f6b7a",
                     fontsize=15 * header_scale, zorder=6, va="bottom")
        self.final_txt = self.ax.text(0.5, 0.06, "", transform=self.ax.transAxes, ha="center",
                                      va="bottom", fontsize=20 * header_scale, color=INK, zorder=7,
                                      bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#c9ced6",
                                                alpha=0.95))
        self.final_txt.set_visible(False)

    def update(self, t):
        run = self.run
        t_eff = min(t, run.t_end)
        cx, cy = run.camera(t_eff)
        cx += 0.08 * self.view_w   # a little more room for the wake trailing downwind
        x0, x1 = cx - self.view_w / 2, cx + self.view_w / 2
        y0, y1 = cy - self.view_h / 2, cy + self.view_h / 2
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)

        img, extent = run.wind_at(t_eff)
        self.wind_im.set_data(img)
        self.wind_im.set_extent(extent)

        gx = np.arange(np.floor(x0), np.ceil(x1) + 1)
        gy = np.arange(np.floor(y0), np.ceil(y1) + 1)
        self.grid.set_segments([[(x, y0 - 1), (x, y1 + 1)] for x in gx] +
                               [[(x0 - 1, y), (x1 + 1, y)] for y in gy])
        self.start_txt.set_position((0.06, y0 + 0.03 * self.view_h))

        segs, age = run.trail_segments(t_eff)
        base = np.array(to_rgba("#6b7a8c"))
        cols = np.tile(base, (len(age), 1))
        cols[:, 3] = 0.55 * age ** 1.5
        self.trails.set_segments(segs)
        self.trails.set_colors(cols)

        xy = run.xy(t_eff)
        self.robots.set_offsets(xy)
        if self.mode == "battery":
            self.robots.set_facecolors(BATTERY_CMAP(run.battery(t_eff) / 100.0))
        else:
            self.robots.set_facecolors(SINGLE_COLOR)
        hd = run.heading(t_eff)
        self.heads.set_segments(np.stack([xy + 0.15 * ROBOT_RAD * hd, xy + 0.95 * ROBOT_RAD * hd], axis=1))

        if t >= run.t_end and getattr(self, "show_final", False):
            self.final_txt.set_text(f"First robot out of battery after {run.t_end:.0f} s\n"
                                    f"Distance travelled upwind: {run.distance(run.t_end):.1f} m")
            self.final_txt.set_visible(True)


class Header:
    def __init__(self, fig, run, rect, mode, speed, scale=1.0):
        self.run = run
        l, b, w, h = rect
        self.ax = fig.add_axes(rect)
        self.ax.axis("off")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.text(0.0, 0.62, TITLES[run.condition], fontsize=30 * scale, fontweight="bold",
                     color=INK, va="center")
        sub = f"{run.n} robots  ·  simulation"
        if speed != 1:
            sub += f"  ·  {speed}× speed"
        self.ax.text(0.0, 0.16, sub, fontsize=18 * scale, color=MUTED, va="center")
        self.stats = self.ax.text(1.0, 0.40, "", fontsize=19 * scale, color=INK, va="center",
                                  ha="right", family="DejaVu Sans Mono", linespacing=1.45)

    def update(self, t):
        run = self.run
        te = min(t, run.t_end)
        self.stats.set_text(f"t = {te:5.1f} s\n"
                            f"upwind distance {run.distance(te):5.1f} m\n"
                            f"mean battery {np.mean(run.battery(te)):5.0f} %")


def _colorbar(fig, rect, scale=1.0):
    ax = fig.add_axes(rect)
    grad = np.linspace(0, 1, 256)[None, :]
    ax.imshow(grad, aspect="auto", cmap=BATTERY_CMAP, extent=(0, 100, 0, 1))
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0 %", "50 %", "100 %"], fontsize=13 * scale, color=MUTED)
    ax.tick_params(length=0, pad=4)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("battery", fontsize=14 * scale, color=MUTED, pad=5)


def _wind_legend(fig, rect, scale=1.0):
    ax = fig.add_axes(rect)
    grad = np.linspace(WIND_VMIN, WIND_VMAX, 256)[None, :]
    ax.imshow(grad, aspect="auto", cmap=WIND_CMAP, vmin=WIND_VMIN, vmax=WIND_VMAX,
              extent=(WIND_VMIN, WIND_VMAX, 0, 1))
    ax.set_yticks([])
    ax.set_xticks([10, 100])
    ax.set_xticklabels(["sheltered", "full wind"], fontsize=13 * scale, color=MUTED)
    ax.tick_params(length=0, pad=4)
    for s in ax.spines.values():
        s.set_edgecolor("#c9ced6")
    ax.set_title("wind speed", fontsize=14 * scale, color=MUTED, pad=5)


def build_job_list():
    jobs = []
    for n in (7, 20):
        for mode in ("battery", "single"):
            for kind in ("realtime_60s", "full_episode"):
                for cond in ("baseline", "evolved"):
                    jobs.append(dict(kind=kind, n=n, mode=mode, layout="single", conds=(cond,)))
                jobs.append(dict(kind=kind, n=n, mode=mode, layout="side_by_side",
                                 conds=("baseline", "evolved")))
    return jobs


def job_path(job):
    name = (f"{job['conds'][0]}" if job["layout"] == "single" else "side_by_side")
    return os.path.join(OUT_DIR, job["kind"], f"n{job['n']}_{name}_{job['mode']}.mp4")


def _ffmpeg():
    sys.path.insert(0, os.environ.get("IMAGEIO_FFMPEG_PATH", ""))
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def render(job, preview_t=None):
    runs = [Run(c, job["n"]) for c in job["conds"]]
    speed = 1 if job["kind"] == "realtime_60s" else FULL_SPEEDUP[job["n"]]
    if job["kind"] == "realtime_60s":
        sim_dur, hold = 60.0, 0.0
    else:
        sim_dur, hold = max(r.t_end for r in runs), HOLD_S
    n_frames = int(round(sim_dur / speed * FPS + hold * FPS))

    fig = Figure(figsize=(W_PX / 100, H_PX / 100), dpi=100, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    panels, headers = [], []
    if job["layout"] == "single":
        panels.append(Panel(fig, runs[0], (0.025, 0.035, 0.95, 0.80), VIEW_W[job["n"]][0], job["mode"]))
        headers.append(Header(fig, runs[0], (0.03, 0.865, 0.94, 0.11), job["mode"], speed))
        legend_y, scale = 0.885, 1.0
        leg_x = 0.44
    else:
        scale = 0.85
        for i, r in enumerate(runs):
            l = 0.02 + i * 0.49
            panels.append(Panel(fig, r, (l, 0.035, 0.47, 0.76), VIEW_W[job["n"]][1], job["mode"],
                                header_scale=scale))
            headers.append(Header(fig, r, (l + 0.005, 0.815, 0.46, 0.1), job["mode"], speed, scale=scale))
        legend_y = 0.945
        leg_x = 0.33
    for p in panels:
        p.show_final = job["kind"] == "full_episode"
    if job["mode"] == "battery":
        _colorbar(fig, (leg_x, legend_y, 0.11, 0.022), scale)
        _wind_legend(fig, (leg_x + 0.17, legend_y, 0.11, 0.022), scale)
    else:
        _wind_legend(fig, (leg_x + 0.07, legend_y, 0.11, 0.022), scale)

    def draw(t):
        for p in panels:
            p.update(t)
        for h in headers:
            h.update(t)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())

    if preview_t is not None:
        out = os.path.join(OUT_DIR, "previews", os.path.relpath(job_path(job), OUT_DIR)
                           .replace(os.sep, "_").replace(".mp4", f"_t{preview_t:.0f}.png"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        import matplotlib.image as mpimg
        mpimg.imsave(out, draw(preview_t))
        return out

    out = job_path(job)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
           "-s", f"{W_PX}x{H_PX}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
           "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(n_frames):
        t = min(f / FPS * speed, sim_dur)
        proc.stdin.write(draw(t).tobytes())
    proc.stdin.close()
    proc.wait()
    return f"{os.path.relpath(out, SCRIPT_DIR)}  ({n_frames / FPS:.1f} s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--only", default=None, help="realtime_60s | full_episode")
    ap.add_argument("--preview", type=float, default=None, help="write one PNG per clip at this sim time")
    args = ap.parse_args()
    jobs = [j for j in build_job_list() if args.only is None or j["kind"] == args.only]
    if args.preview is not None:
        for j in jobs:
            print(render(j, preview_t=args.preview))
        return
    with Pool(min(args.jobs, len(jobs))) as pool:
        for msg in pool.imap_unordered(render, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
