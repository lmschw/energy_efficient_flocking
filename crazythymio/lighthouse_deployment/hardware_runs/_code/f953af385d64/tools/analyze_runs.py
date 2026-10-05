#!/usr/bin/env python3
"""Preliminary / summary analysis of the Lighthouse experiment runs.
usage: tools/analyze_runs.py [hardware_runs dir]     (default: ../hardware_runs next to this tools/ folder)

Per valid run (folder <condition>_rep<k>_* with run_info.json, see hardware_runs/README.md) it puts all robots on one
0.5 s time grid (laptop clock, via pi_clock_offset_s) and reports:
  dist      swarm displacement toward -x: mean(x_start) - mean(x_end)  [m]   (NOT -mean(final x): runs start near x=+1)
  speed     dist / duration [m/s]
  t_arena   time until the swarm centroid passes the arena's -x end (x = -1.84) [s] (nan = never)
  order     mean heading alignment (polar order 0..1, 1 = all facing the same way)
  nn        mean nearest-neighbour centre distance [m]
  contact   % of time some pair of centres is closer than 0.15 m (~touching; robots are 0.11 m wide)
  avoid     % of robot-ticks in which the collision avoidance braked (scale_avoid < 1)
  batt      mean simulated battery at the end [%];  wind  mean simulated wind exposure [%] (100 = unsheltered)
  track     % of robot-ticks with a valid position
Then a per-condition summary (mean +- sd over repetitions).
"""
import csv, glob, json, math, os, re, sys
import numpy as np

ARENA_X_MIN = -1.84
here = os.path.dirname(os.path.abspath(__file__))
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "hardware_runs")


def load(run):
    info = json.load(open(os.path.join(run, "run_info.json")))
    off = info.get("pi_clock_offset_s", {})
    rows = {}
    for n in info["robots"].split():
        f = glob.glob(os.path.join(run, f"robot-{n}_*.csv"))
        if not f:
            return info, None
        r = [{k: (float(v) if v not in ("", None) else float("nan")) for k, v in q.items()} for q in csv.DictReader(open(f[0]))]
        if not r:
            return info, None
        o = off.get(f"robot-{n}", 0.0)
        for q in r:
            q["t"] = q["timestamp"] - o
        rows[n] = r
    return info, rows


def analyse(run):
    info, rows = load(run)
    if rows is None:
        return None
    names = list(rows)
    t0 = max(r[0]["t"] for r in rows.values()); t1 = min(r[-1]["t"] for r in rows.values())
    grid = np.arange(t0, t1, 0.5)
    if len(grid) < 4:
        return None
    def I(n, k, unwrap=False):
        t = np.array([q["t"] for q in rows[n]]); v = np.array([q.get(k, np.nan) for q in rows[n]])
        if unwrap: v = np.unwrap(v)
        return np.interp(grid, t, v)
    X = np.stack([I(n, "x") for n in names], 1); Y = np.stack([I(n, "y") for n in names], 1)
    H = np.stack([I(n, "heading", True) for n in names], 1)
    ok = (np.abs(X) < 50).all(1)
    X, Y, H = X[ok], Y[ok], H[ok]
    P = np.stack([X, Y], 2); D = np.linalg.norm(P[:, :, None] - P[:, None], axis=3)
    D[:, np.arange(len(names)), np.arange(len(names))] = 9
    cx = X.mean(1)
    t_arena = next((i * 0.5 for i, x in enumerate(cx) if x < ARENA_X_MIN), float("nan"))
    allq = [q for r in rows.values() for q in r]
    sa = np.array([q.get("scale_avoid", np.nan) for q in allq])
    wind = np.array([q.get("wind_pct", np.nan) for q in allq])
    dur = (len(cx) - 1) * 0.5
    return dict(
        cond=re.match(r"([a-z]+)_rep", os.path.basename(run)).group(1),
        rep=int(re.match(r"[a-z]+_rep(\d+)_", os.path.basename(run)).group(1)),
        dist=cx[0] - cx[-1], speed=(cx[0] - cx[-1]) / dur if dur > 0 else np.nan, t_arena=t_arena,
        order=float(np.abs(np.exp(1j * (H + math.pi / 2)).mean(1)).mean()), nn=float(D.min(2).mean()),
        contact=float(np.mean(D.min((1, 2)) < 0.15) * 100),
        avoid=float(np.mean(sa[~np.isnan(sa)] < 0.999) * 100) if (~np.isnan(sa)).any() else np.nan,
        batt=float(np.mean([r[-1]["battery"] for r in rows.values()])),
        wind=float(np.nanmean(wind)) if (~np.isnan(wind)).any() else np.nan,
        track=float(np.nanmean([q.get("tracked", 1) for q in allq]) * 100), dur=dur,
        x_end=float(cx[-1]), spread_y=float(np.ptp(Y[-1])))


runs = sorted(d for d in glob.glob(os.path.join(root, "*_rep*_*")) if os.path.exists(os.path.join(d, "run_info.json")))
res = [r for r in (analyse(d) for d in runs) if r]
cols = ("dist", "speed", "t_arena", "order", "nn", "contact", "avoid", "batt", "wind", "track", "x_end")
fmt = {"dist": "{:6.2f}", "speed": "{:6.3f}", "t_arena": "{:6.1f}", "order": "{:6.2f}", "nn": "{:6.2f}", "contact": "{:6.1f}",
       "avoid": "{:6.1f}", "batt": "{:6.1f}", "wind": "{:6.1f}", "track": "{:6.1f}", "x_end": "{:6.2f}"}
print(f"{'run':14s}" + "".join(f"{c:>8s}" for c in cols))
for r in sorted(res, key=lambda r: (r["cond"], r["rep"])):
    print(f"{r['cond'] + '_rep' + str(r['rep']):14s}" + "".join(f"{fmt[c].format(r[c]):>8s}" for c in cols))
print()
for cond in sorted({r["cond"] for r in res}):
    sub = [r for r in res if r["cond"] == cond]
    print(f"{cond:8s} n={len(sub):2d} | " + " | ".join(
        f"{c} {np.nanmean([r[c] for r in sub]):.2f}+-{np.nanstd([r[c] for r in sub]):.2f}" for c in ("dist", "t_arena", "order", "nn", "contact", "avoid", "batt", "wind")))
