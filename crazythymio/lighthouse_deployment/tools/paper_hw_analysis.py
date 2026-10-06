#!/usr/bin/env python3
"""Paper analysis of the Lighthouse hardware runs (7 CrazyThymios) + the matched n=7 simulations.

Per controller the FIRST 30 USABLE trials in repetition order (usable = a valid window of >= 20 s, see below): trials
of the main block (reps 1..30, fixed order lj -> plain -> clamped) that are not usable are replaced by the next usable
repetitions 31.. (recorded afterwards, alternating lj / plain). Written to
hardware_transfer_test/final/overleaf_summary/paper_hardware_lighthouse_summary.json (key "n30"), with one figure per
paper version (figures/hardware_lighthouse.{pdf,png,svg} with clamped, figures/hardware_lighthouse_no_clamp.{...} without).

Leadership metrics: exactly the functions used for the simulation tables (hardware_transfer_test/final/
leadership_metrics.py), on all robots' positions on one 0.5 s grid (the simulation's dt).

Position-estimate failures: when robots collide, a robot's Lighthouse/Kalman estimate occasionally diverges for a few
seconds (its estimated height z drops by up to metres, x/y jump). A sample is flagged if |z - median z| > 0.15 m, the
x/y step from the previous sample is > 0.3 m (the robots cover <= 0.09 m per 0.5 s), or the position is outside the
arena; flags are widened by one sample. The z reference is the robot's median z over its first 5 s. Flagged stretches of <= 2 s are bridged by linear interpolation; the metrics
use the longest window in which no robot has a longer flagged stretch (runs with a window < 20 s are excluded).
The first sample of every robot is dropped (known first-tick glitch).

usage: tools/paper_hw_analysis.py
"""
import csv, glob, json, os, re, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "hardware_runs")
FINAL = os.path.abspath(os.path.join(HERE, "..", "..", "..", "hardware_transfer_test", "final"))
OUT_JSON = os.path.join(FINAL, "overleaf_summary", "paper_hardware_lighthouse_summary.json")
FIG_DIR = os.path.join(FINAL, "overleaf_summary", "figures")
SIM_CACHE = os.path.join(FINAL, "leadership_analysis", "seed_sweep_table_n7_first60s.json")
sys.path.insert(0, FINAL)
os.chdir(FINAL)                                     # leadership_metrics resolves genome/rule paths relative to itself
import leadership_metrics as lm                     # noqa: E402

DT = 0.5
Z_TOL, STEP_TOL, MAX_BRIDGE, MIN_WINDOW = 0.15, 0.30, 4, 40     # m, m, samples (2 s), grid points (20 s)
BOX = (-5.6, 2.5, -3.0, 3.0)                                    # plausible x_min, x_max, y_min, y_max [m]
CONTACT_M = 0.12                                                # centre distance counted as a collision (as in sim)
SIM_NAME = {"plain": "plain_seed123", "clamped": "plain_seed123_clamped", "lj": "lj_baseline"}
LABEL = {"plain": "Stage 2", "clamped": "Stage 2, clamped", "lj": "Baseline"}
COLOR = {"plain": "#D95319", "clamped": "#0072BD", "lj": "#77AC30"}


def _bad_mask(x, y, z, tracked):
    step = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
    z_ref = np.median(z[:10])          # first 5 s at the start position (a run-wide median fails if > 50% diverged)
    bad = (np.abs(z - z_ref) > Z_TOL) | (step > STEP_TOL) | (tracked < 0.5) \
        | (x < BOX[0]) | (x > BOX[1]) | (y < BOX[2]) | (y > BOX[3])
    return bad | np.r_[bad[1:], False] | np.r_[False, bad[:-1]]


def _stretches(mask):
    """(start, end_exclusive) of every True stretch."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def load_run(run):
    info = json.load(open(os.path.join(run, "run_info.json")))
    off = info.get("pi_clock_offset_s", {})
    robots = []
    for n in info["robots"].split():
        f = glob.glob(os.path.join(run, f"robot-{n}_*.csv"))[0]
        r = list(csv.DictReader(open(f)))[1:]
        g = lambda k: np.array([float(q[k]) if q[k] not in ("", None) else np.nan for q in r])
        t = g("timestamp") - off.get(f"robot-{n}", 0.0)
        x, y, z, trk = g("x"), g("y"), g("raw_z"), g("tracked")
        bad = _bad_mask(x, y, z, np.nan_to_num(trk, nan=1.0))
        unbridged = np.zeros(len(t), bool)
        for a, b in _stretches(bad):
            if b - a > MAX_BRIDGE:
                unbridged[a:b] = True
        good = ~bad
        robots.append(dict(t=t, x=x, y=y, good=good, unbridged=unbridged, heading=g("heading"),
                           battery=g("battery"), wind=g("wind_pct"), ir=np.fmax(g("ir_front_max"), g("ir_rear_max"))))
    t0 = max(r["t"][0] for r in robots); t1 = min(r["t"][-1] for r in robots)
    grid = np.arange(t0, t1 + 1e-9, DT)
    P = np.zeros((len(grid), len(robots), 2)); usable = np.ones((len(grid), len(robots)), bool)
    for j, r in enumerate(robots):
        tg = r["t"][r["good"]]
        P[:, j, 0] = np.interp(grid, tg, r["x"][r["good"]]); P[:, j, 1] = np.interp(grid, tg, r["y"][r["good"]])
        near = np.clip(np.searchsorted(r["t"], grid), 0, len(r["t"]) - 1)
        usable[:, j] = ~r["unbridged"][near]
    ok = usable.all(1); best = (0, 0)
    for a, b in _stretches(ok):
        if b - a > best[1] - best[0]:
            best = (a, b)
    return dict(info=info, robots=robots, grid=grid, P=P, usable=usable, window=best,
                flagged=float(np.mean(np.concatenate([~r["good"] for r in robots]))))


def leadership(P):
    ranks, bearing = lm._front_rank_series(P)
    recip = lm.reciprocity_index(ranks, DT)
    occ = lm.occupancy_and_exchange(ranks, DT, P)
    path = float(np.sum(np.hypot(*np.diff(P.mean(1), axis=0).T)))
    pers = lm.persistence_filtered_switches(occ["front_id"], DT, path)
    ls, _ = lm.directional_correlation_network(bearing, DT)
    hier = lm.hierarchy_summary(ls)
    return {"front_changes_per_min": pers["rate_per_sec"] * 60.0,
            "persist_share_pct": 100.0 * pers["real_switches"] / occ["raw_switches"] if occ["raw_switches"] else np.nan,
            "front_occupancy_entropy": occ["normalized_entropy"], "reciprocity_r": recip["r"],
            "hierarchy_entropy": hier["entropy"], "hierarchy_steepness": hier["steepness"]}


def movement(R):
    P, (a, b) = R["P"], R["window"]
    W = P[a:b]
    D = np.linalg.norm(W[:, :, None] - W[:, None], axis=3); D[:, range(W.shape[1]), range(W.shape[1])] = 9
    H = np.stack([np.interp(R["grid"][a:b], r["t"], np.unwrap(r["heading"])) for r in R["robots"]], 1)
    cx = P.mean(1)[:, 0]
    return {"distance_m": float(cx[0] - cx[-1]),                      # displacement upwind (-x) over the whole run
            "alignment": float(np.abs(np.exp(1j * H).mean(1)).mean()),
            "nn_distance_m": float(D.min(2).mean()),
            "collision_time_pct": float(np.mean(D.min((1, 2)) < CONTACT_M) * 100),
            "wind_exposure_pct": float(np.nanmean(np.concatenate([r["wind"] for r in R["robots"]]))),
            "final_battery_pct": float(np.mean([r["battery"][-1] for r in R["robots"]]))}


def runs_of(cond):
    """All complete runs of a condition, in repetition order."""
    out = {}
    for d in glob.glob(os.path.join(RUNS, f"{cond}_rep*_*")):
        m = re.match(rf"{cond}_rep(\d+)_", os.path.basename(d))
        if m and os.path.exists(os.path.join(d, "run_info.json")):
            out[int(m.group(1))] = d
    return [out[k] for k in sorted(out)]


def summarise(vals, base=None):
    v = np.array(vals, float); v = v[np.isfinite(v)]
    e = {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
         "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "n": int(len(v))}
    if base is not None:
        b = np.array(base, float); b = b[np.isfinite(b)]
        e["p_vs_baseline"] = float(stats.mannwhitneyu(v, b).pvalue)
    return e


def sim_n7():
    """Matched simulations (n=7, the 30 seeds 1000..1029 of the paper's seed sweeps): full episode (from
    leadership_metrics.py's table) and the first 60 s only (the length of a hardware trial)."""
    full = json.load(open(os.path.join(FINAL, "leadership_analysis", "seed_sweep_table_n7.json")))
    if os.path.exists(SIM_CACHE):
        first60 = json.load(open(SIM_CACHE))
    else:
        first60 = []
        for c in SIM_NAME.values():
            for s in range(1000, 1030):
                pos, dt, _, _ = lm.run_trajectory(c, 7, seed=s)
                first60.append(dict(condition=c, seed=s, **leadership(np.asarray(pos)[:121])))
        json.dump(first60, open(SIM_CACHE, "w"), indent=1)
    per = {}
    for c, sc in SIM_NAME.items():
        F = [r for r in full if r["condition"] == sc]
        per[c] = {"full": {"front_changes_per_min": [r["persistence_filtered_rate_per_sec"] * 60 for r in F],
                           "persist_share_pct": [100 * r["real_switches"] / r["raw_switches"] if r["raw_switches"] else np.nan for r in F],
                           "front_occupancy_entropy": [r["front_occupancy_normalized_entropy"] for r in F],
                           "reciprocity_r": [r["reciprocity_r"] for r in F],
                           "hierarchy_entropy": [r["hierarchy_leadership_entropy"] for r in F],
                           "hierarchy_steepness": [r["hierarchy_steepness"] for r in F]},
                  "first60s": {k: [r[k] for r in first60 if r["condition"] == sc] for k in METRICS}}
    return per


METRICS = ("front_changes_per_min", "persist_share_pct", "front_occupancy_entropy", "reciprocity_r",
           "hierarchy_entropy", "hierarchy_steepness")
MOVE = ("distance_m", "alignment", "nn_distance_m", "collision_time_pct", "wind_exposure_pct", "final_battery_pct")


def analyse(conds, n_trials, sim):
    res, raw = {}, {}
    for c in conds:
        L, M, Q, excluded = [], [], [], []
        for d in runs_of(c):
            if len(L) == n_trials:
                break
            R = load_run(d); a, b = R["window"]
            if b - a < MIN_WINDOW:                  # not usable -> replaced by the next usable repetition
                excluded.append({"run": os.path.basename(d), "window_s": (b - a) * DT}); continue
            M.append(movement(R))
            Q.append({"run": os.path.basename(d), "window_s": (b - a) * DT, "flagged_pct": 100 * R["flagged"]})
            L.append(dict(run=os.path.basename(d), **leadership(R["P"][a:b])))
        assert len(L) == n_trials, f"{c}: only {len(L)} usable trials"
        raw[c] = {"lead": L, "move": M, "quality": Q, "excluded": excluded}
    for c in conds:
        L, M, Q = raw[c]["lead"], raw[c]["move"], raw[c]["quality"]
        bl, bm = raw["lj"]["lead"], raw["lj"]["move"]
        res[c] = {"n_runs": len(M), "runs_used": [l["run"] for l in L], "excluded": raw[c]["excluded"],
                  "window_s": summarise([q["window_s"] for q in Q]),
                  "full_window_runs": int(sum(q["window_s"] >= 59 for q in Q)),
                  "flagged_samples_pct": float(np.mean([q["flagged_pct"] for q in Q])),
                  "hardware": {k: summarise([l[k] for l in L], None if c == "lj" else [l[k] for l in bl]) for k in METRICS},
                  "movement": {k: summarise([m[k] for m in M], None if c == "lj" else [m[k] for m in bm]) for k in MOVE},
                  "sim_n7": {v: {k: summarise(sim[c][v][k], None if c == "lj" else sim["lj"][v][k]) for k in METRICS}
                             for v in ("full", "first60s")},
                  "per_run": L}
        res[c]["hw_vs_sim_full_p"] = {k: float(stats.mannwhitneyu(
            [l[k] for l in L if np.isfinite(l[k])], [s for s in sim[c]["full"][k] if np.isfinite(s)]).pvalue) for k in METRICS}
    return res, raw


def figure(raw, sim, conds, stem):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.45, wspace=0.38)
    # top: one representative run per condition (median front-occupancy entropy), usable window only
    top = fig.add_gridspec(1, len(conds), left=0.08, right=0.98, top=0.95, bottom=0.6, wspace=0.08)
    for i, c in enumerate(conds):
        L = raw[c]["lead"]; e = np.array([l["front_occupancy_entropy"] for l in L])
        rep = L[int(np.argsort(e)[len(e) // 2])]["run"]
        R = load_run(os.path.join(RUNS, rep)); a, b = R["window"]
        ax = fig.add_subplot(top[0, i])
        for j in range(R["P"].shape[1]):
            ax.plot(R["P"][a:b, j, 0], R["P"][a:b, j, 1], lw=0.9)
            ax.plot(*R["P"][a, j], "k.", ms=3)
        ax.set_aspect("equal"); ax.set_xlim(-5.2, 1.6); ax.set_ylim(-2.7, 2.7)
        ax.set_title(f"{LABEL[c]} ({rep.split('_')[1]})", fontsize=8); ax.tick_params(labelsize=7)
        ax.set_xlabel("x [m]  (wind blows to +x)", fontsize=7)
        if i == 0: ax.set_ylabel("y [m]", fontsize=7)
        else: ax.set_yticklabels([])
    bot = fig.add_gridspec(1, 3, left=0.08, right=0.98, top=0.47, bottom=0.12, wspace=0.42)
    for i, (k, lab) in enumerate((("front_occupancy_entropy", "Front occupancy entropy"),
                                  ("front_changes_per_min", "Front changes per minute"),
                                  ("reciprocity_r", "Reciprocity $r$"))):
        ax = fig.add_subplot(bot[0, i]); pos = 0; ticks = []
        rng = np.random.default_rng(0)
        for c in conds:
            for src, alpha in (("hw", 0.45), ("sim", 0.15)):
                v = np.array([l[k] for l in raw[c]["lead"]] if src == "hw" else sim[c]["full"][k], float)
                v = v[np.isfinite(v)]
                bp = ax.boxplot([v], positions=[pos], widths=0.6, showfliers=False, patch_artist=True)
                bp["boxes"][0].set(facecolor=COLOR[c], alpha=alpha, hatch=None if src == "hw" else "///")
                bp["medians"][0].set(color="k")
                ax.scatter(pos + rng.uniform(-0.15, 0.15, len(v)), v, s=4, color=COLOR[c], lw=0, zorder=3)
                pos += 1
            pos += 0.5
        ax.set_xticks([0.5 + 2.5 * i for i in range(len(conds))], [LABEL[c].replace(", ", ",\n") for c in conds], fontsize=7)
        ax.set_ylabel(lab, fontsize=8); ax.tick_params(axis="y", labelsize=7); ax.grid(True, axis="y", ls=":", alpha=0.6)
    fig.text(0.5, 0.01, "left box of each pair: real robots; hatched right box: matched simulation (n = 7)",
             ha="center", fontsize=7)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def main():
    sim = sim_n7()
    out = {"method": __doc__, "sim_n7_note": "seeds 1000-1029, random start in a 3x3 m square, paper LJ rules (r0 0.7 m); "
           "'full' = whole episode until the first battery is empty, 'first60s' = first 121 steps"}
    raws = {}
    out["n30"], raws["n30"] = analyse(("lj", "plain", "clamped"), 30, sim)
    json.dump(out, open(OUT_JSON, "w"), indent=1, default=float)
    figure(raws["n30"], sim, ("plain", "clamped", "lj"), os.path.join(FIG_DIR, "hardware_lighthouse"))
    figure(raws["n30"], sim, ("plain", "lj"), os.path.join(FIG_DIR, "hardware_lighthouse_no_clamp"))
    for key in ("n30",):
        print(f"\n===== {key}")
        for c, r in out[key].items():
            print(f"{c:8s} runs {r['n_runs']} (excluded/replaced: {[e['run'] for e in r['excluded']]}, full 60 s: {r['full_window_runs']}), "
                  f"window {r['window_s']['median']:.1f} s median, flagged {r['flagged_samples_pct']:.1f}% of samples")
            for k in METRICS:
                h, s, s6 = r["hardware"][k], r["sim_n7"]["full"][k], r["sim_n7"]["first60s"][k]
                p = f"p={h['p_vs_baseline']:.1e}" if "p_vs_baseline" in h else ""
                print(f"   {k:24s} hw {h['median']:7.2f} [{h['q25']:6.2f},{h['q75']:6.2f}] {p:10s} | sim {s['median']:6.2f} "
                      f"[{s['q25']:6.2f},{s['q75']:6.2f}] | sim60 {s6['median']:6.2f} | hw-vs-sim p={r['hw_vs_sim_full_p'][k]:.1e}")
            for k in MOVE:
                m = r["movement"][k]; p = f"p={m['p_vs_baseline']:.1e}" if "p_vs_baseline" in m else ""
                print(f"   {k:24s} {m['mean']:7.2f} +- {m['sd']:5.2f} {p}")


if __name__ == "__main__":
    main()
