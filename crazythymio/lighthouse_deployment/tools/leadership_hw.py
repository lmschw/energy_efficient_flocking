#!/usr/bin/env python3
"""Leadership / position-exchange metrics for the Lighthouse hardware runs, computed with EXACTLY the functions used for the
simulation results (hardware_transfer_test/final/leadership_metrics.py), so hardware and simulation numbers are comparable:
  1. Voelkl reciprocity r (lead time vs follow time across robots)
  2. Nagy directional-correlation network: leadership entropy + hierarchy steepness
  3. front-rank occupancy entropy + position-exchange rate (per s, per m)
  4. persistence-filtered leadership-switch rate (per s)
Trajectories: all robots on one 0.5 s grid (the simulation's dt), Lighthouse glitches removed (see analyze_runs.py).
usage: tools/leadership_hw.py [hardware_runs dir]
"""
import glob, os, re, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "hardware_transfer_test", "final"))
import leadership_metrics as lm                       # noqa: E402  (the simulation's own implementation)
import importlib.util                                 # noqa: E402
spec = importlib.util.spec_from_file_location("ar", os.path.join(HERE, "analyze_runs.py"))
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "hardware_runs")
_argv = sys.argv; sys.argv = [sys.argv[0], os.path.join(root, "__none__")]   # load helpers without running the report
ar = importlib.util.module_from_spec(spec); spec.loader.exec_module(ar)
sys.argv = _argv
DT = 0.5


def trajectories(run):
    info, rows = ar.load(run)
    if rows is None:
        return None
    names = list(rows)
    t0 = max(r[0]["t"] for r in rows.values()); t1 = min(r[-1]["t"] for r in rows.values())
    grid = np.arange(t0, t1, DT)
    P = np.stack([np.stack([np.interp(grid, [q["t"] for q in rows[n]], [q[k] for q in rows[n]]) for k in ("x", "y")], 1)
                  for n in names], 1)                       # (T, N, 2)
    return P[(np.abs(P[..., 0]) < 50).all(1)]


def metrics(P):
    ranks, bearing = lm._front_rank_series(P)
    recip = lm.reciprocity_index(ranks, DT)
    occ = lm.occupancy_and_exchange(ranks, DT, P)
    path = np.sum(np.hypot(*np.diff(P.mean(1), axis=0).T))
    pers = lm.persistence_filtered_switches(occ["front_id"], DT, path)
    ls, _ = lm.directional_correlation_network(bearing, DT)
    hier = lm.hierarchy_summary(ls)
    return {"reciprocity_r": recip["r"], "front_occ_entropy": occ["normalized_entropy"],
            "exchange_per_s": occ["exchange_rate_per_sec"], "exchange_per_m": occ["exchange_rate_per_m"],
            "switch_rate_filtered_per_s": pers["rate_per_sec"], "nagy_entropy": hier["entropy"],
            "nagy_steepness": hier["steepness"]}


runs = sorted(d for d in glob.glob(os.path.join(root, "*_rep*_*")) if os.path.exists(os.path.join(d, "run_info.json")))
res = {}
for d in runs:
    P = trajectories(d)
    if P is None or len(P) < 20:
        continue
    cond = re.match(r"([a-z]+)_rep", os.path.basename(d)).group(1)
    res.setdefault(cond, []).append(metrics(P))
keys = ("reciprocity_r", "front_occ_entropy", "exchange_per_s", "exchange_per_m", "switch_rate_filtered_per_s",
        "nagy_entropy", "nagy_steepness")
print(f"{'metric':28s}" + "".join(f"{c + ' (n=' + str(len(v)) + ')':>34s}" for c, v in res.items()) + "   p (Mann-Whitney vs lj)")
for k in keys:
    line = f"{k:28s}"
    for c, v in res.items():
        a = np.array([m[k] for m in v], float); a = a[~np.isnan(a)]
        line += f"{f'{np.median(a):.3f} [{np.percentile(a, 25):.3f}, {np.percentile(a, 75):.3f}]' if len(a) else 'n/a':>34s}"
    ps = []
    for c, v in res.items():
        if c == "lj" or "lj" not in res:
            continue
        a = np.array([m[k] for m in v], float); b = np.array([m[k] for m in res["lj"]], float)
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) > 2 and len(b) > 2:
            ps.append(f"{c}: {stats.mannwhitneyu(a, b).pvalue:.1e}")
    print(line + "   " + ", ".join(ps))
