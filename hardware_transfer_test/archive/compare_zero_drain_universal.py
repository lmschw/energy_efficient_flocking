"""Re-evaluates EVERY genome produced this session (plus LJ baseline) under ONE uniform
protocol: n=10, standard KAPPA=10, 30 seeds (1000-1029), and -- regardless of what battery-
drain regime each genome was actually TRAINED under -- ZERO battery drain above the idle
floor while colliding at evaluation time (drain_fraction=0.0). Each genome still uses its OWN
correct clamp/inflation/resolve settings (the ones it was actually trained with -- that part
is NOT normalized, only the collision-drain question is). This isolates "how good is this
genome's behavior" from "how much did we choose to charge it for collision at eval time,"
putting every genome on the same footing regardless of its training-time fitness function.

Includes a from-scratch LJ-baseline drain-holiday wrapper (LJ has no equivalent in
simulation_hebbian_no_drain.py) since LJ shares the same battery-drain physics and should get
the same treatment for a genuinely uniform comparison.
"""
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # ./hardware_transfer_test/archive -> repo root
EXPERIMENT_DIR = os.path.join(REPO_ROOT, "ants26_replication", "experiment")
sys.path.insert(0, EXPERIMENT_DIR)

import config  # noqa: E402
from hebbian_controller import unflatten_abcd  # noqa: E402
from simulation_hebbian_no_drain import simulate_hebbian_episode_partial_drain_on_collision as sim_hebbian_nd  # noqa: E402
from lj_baseline import _flocking_velocity_command, _resolve_rules, move as lj_move  # noqa: E402
from wind_physics import RayTraceCircularRobots, dragforce, batterydrainage, _spawn_agents  # noqa: E402

N_AGENTS = 10
SEEDS = list(range(1000, 1030))
KAPPA_STANDARD = 10.0


def simulate_lj_baseline_no_drain(rules, seed, n_agents, drain_fraction=0.0):
    """LJ-baseline equivalent of simulate_hebbian_episode_partial_drain_on_collision -- same
    floor-protected refund logic (only the above-idle-floor portion of drain is refundable,
    the unconditional BATTERY_MIN_DRAIN floor is always paid), same MAX_STEPS safety cap."""
    if seed is not None:
        np.random.seed(seed)

    dt = config.DT
    robot_rad = config.ROBOT_RAD
    wind_rad = config.WIND_RAD
    xRange = list(config.X_RANGE)
    yRange = list(config.Y_RANGE)
    v_wind = config.V_WIND
    Uinf, Nx, Ny, kappa = config.UINF, config.HEBBIAN_NX, config.HEBBIAN_NY, config.KAPPA
    spawn_square_size = config.SPAWN_SQUARE_SIZE
    midpoint = list(config.SPAWN_MIDPOINT)
    min_battery, max_battery = config.MIN_BATTERY, config.MAX_BATTERY

    walls = [xRange[0] + robot_rad, xRange[1] - robot_rad, yRange[1] - robot_rad, yRange[0] + robot_rad]
    min_dist = config.COLLISION_MIN_DIST_SLACK + 2.0 * robot_rad
    min_dist_initial = config.SPAWN_MIN_DIST_SLACK + 2.0 * robot_rad

    agents = _spawn_agents(n_agents, midpoint, spawn_square_size, min_dist_initial, max_battery, min_battery)
    vel = np.zeros((n_agents, 2))
    batteryEmpty = False
    collision_counter = 0
    r0, epsilon, k_align, k_goal, K1, K2, U = _resolve_rules(rules)
    r_cut, r_min, R_align = config.R_CUT, config.R_MIN, config.R_ALIGN

    step_count = 0
    MAX_STEPS = 5000
    while not batteryEmpty and step_count < MAX_STEPS:
        step_count += 1
        vel[:, :] = _flocking_velocity_command(agents, n_agents, r0, epsilon, k_align, k_goal, K1, K2, U,
                                                r_cut, r_min, R_align)
        vel_actual, agents, xRange, collision_counter = lj_move(agents, vel, dt, n_agents, min_dist, walls,
                                                                 collision_counter)
        yVals, xVals, powerVals = RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny)
        F_drag = dragforce(agents, wind_rad, xVals, yVals, powerVals, n_agents, vel_actual, v_wind, kappa)
        agents, batt_drain = batterydrainage(agents, vel_actual, F_drag, robot_rad, dt)

        agents_xy = agents[:, 0:2]
        D = np.linalg.norm(agents_xy[:, None, :] - agents_xy[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        colliding_agent = D.min(axis=1) < min_dist if n_agents > 1 else np.zeros(n_agents, dtype=bool)
        if np.any(colliding_agent):
            floor = config.BATTERY_MIN_DRAIN * dt
            refundable = np.maximum(batt_drain - floor, 0.0)
            agents[colliding_agent, 3] += (1.0 - drain_fraction) * refundable[colliding_agent]

        batteryEmpty = np.any(agents[:, 3] <= 0.0)

    average_batt = np.mean(agents[:, 3])
    dist_travelled = -np.mean(agents[:, 0])
    collision_time = collision_counter * dt
    return dist_travelled, average_batt, collision_time


def _cfg(safety_clamp, min_dist_inflation, resolve_collisions, resolve_strength=None, resolve_max_iter=None, kappa=KAPPA_STANDARD):
    d = dict(HEBBIAN_SAFETY_CLAMP_ENABLED=safety_clamp, HEBBIAN_MIN_DIST_INFLATION=min_dist_inflation,
             HEBBIAN_RESOLVE_COLLISIONS=resolve_collisions, KAPPA=kappa)
    if resolve_strength is not None:
        d["HEBBIAN_RESOLVE_COLLISIONS_STRENGTH"] = resolve_strength
        d["HEBBIAN_RESOLVE_COLLISIONS_MAX_ITER"] = resolve_max_iter
    return d


NO_CLAMP = _cfg(False, 1.0, False)
WITH_CLAMP = _cfg(True, 1.3, True, 0.5, 2)
NO_CLAMP_KAPPA20 = _cfg(False, 1.0, False, kappa=20.0)

HT = os.path.join(SCRIPT_DIR)
RESULTS = os.path.join(REPO_ROOT, "results")

CONDITIONS = {
    "lj_baseline":            ("lj", None, NO_CLAMP),
    "pre_clamp_best":         ("hebbian", f"{HT}/pre_clamp_best/genome_trained_n20.npy", NO_CLAMP),
    "safety_clamp_best":      ("hebbian", f"{HT}/safety_clamp_best/genome_trained_n20.npy", WITH_CLAMP),
    "upwind_clamp":           ("hebbian", f"{HT}/upwind_clamp/genome_trained_n10.npy", WITH_CLAMP),
    "upwind_no_clamp":        ("hebbian", f"{HT}/upwind_no_clamp/genome_trained_n10.npy", NO_CLAMP),
    "walk_left_n10_clamp":    ("hebbian", f"{HT}/walk_left_n10_clamp/genome_trained_n10.npy", WITH_CLAMP),
    "walk_left_n10_no_clamp": ("hebbian", f"{HT}/walk_left_n10_no_clamp/genome_trained_n10.npy", NO_CLAMP),
    "job5_zero_signal":       ("hebbian", f"{RESULTS}/hebbian_results_v2_zero_collision_signal/n10_seed42/hebbian_save_battery_avoid_all_best.npy", NO_CLAMP),
    "job6_drain0":            ("hebbian", f"{RESULTS}/hebbian_results_v2_no_drain_on_collision/n10_seed42/hebbian_save_battery_avoid_all_best.npy", NO_CLAMP),
    "job7_drain10":           ("hebbian", f"{RESULTS}/hebbian_results_v2_partial_drain_on_collision_10pct/n10_seed42/hebbian_save_battery_avoid_all_best.npy", NO_CLAMP),
    "job8_drain25":           ("hebbian", f"{RESULTS}/hebbian_results_v2_partial_drain_on_collision_25pct/n10_seed42/hebbian_save_battery_avoid_all_best.npy", NO_CLAMP),
    "job9_drain50":           ("hebbian", f"{RESULTS}/hebbian_results_v2_partial_drain_on_collision_50pct/n10_seed42/hebbian_save_battery_avoid_all_best.npy", NO_CLAMP),
    "job10_drain75":          ("hebbian", f"{RESULTS}/hebbian_results_v2_partial_drain_on_collision_75pct/n10_seed42/hebbian_save_battery_avoid_all_best.npy", NO_CLAMP),
    "job11_kappa20":          ("hebbian", f"{RESULTS}/hebbian_results_v2_kappa20/n10_seed42/hebbian_save_battery_avoid_all_best.npy", NO_CLAMP),
    "job12_kappa20_drain0":   ("hebbian", f"{RESULTS}/hebbian_results_v2_kappa20_drain_holiday/n10_seed42/hebbian_save_battery_avoid_all_best.npy", NO_CLAMP),
    "job13_2stage_upwind":    ("hebbian", f"{RESULTS}/hebbian_results_v2_2stage_upwind/n10_seed42/hebbian_save_battery_avoid_all_best.npy", NO_CLAMP),
}

_BASELINE_RULES_PATH = os.path.join(HT, "lj_baseline", "paper_baseline_rules.json")


def evaluate(label, kind, genome_path, cfg):
    saved = {k: getattr(config, k) for k in cfg}
    for k, v in cfg.items():
        setattr(config, k, v)
    try:
        d, b, c, steps = [], [], [], []
        if kind == "lj":
            rules = json.load(open(_BASELINE_RULES_PATH))
            for s in SEEDS:
                dist, batt, ct = simulate_lj_baseline_no_drain(rules, seed=s, n_agents=N_AGENTS, drain_fraction=0.0)
                d.append(dist); b.append(batt / config.MAX_BATTERY * 100.0); c.append(ct)
        else:
            genome = np.load(genome_path)
            rules = unflatten_abcd(genome)
            for s in SEEDS:
                dist, batt, ct, wct, coh, prox, tele = sim_hebbian_nd(
                    rules, seed=s, n_agents=N_AGENTS, wind_enabled=True, nx=50, ny=50,
                    drain_fraction=0.0, record_battery=True)
                d.append(dist); b.append(batt); c.append(ct); steps.append(tele["battery"].shape[0] - 1)
        return {
            "dist_mean": float(np.mean(d)), "dist_std": float(np.std(d)),
            "batt_mean": float(np.mean(b)), "batt_std": float(np.std(b)),
            "ct_mean": float(np.mean(c)), "ct_std": float(np.std(c)),
            "n_steps_max": max(steps) if steps else None,
        }
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def main():
    results = {}
    for label, (kind, genome_path, cfg) in CONDITIONS.items():
        r = evaluate(label, kind, genome_path, cfg)
        results[label] = r
        cap_note = f" [CAP HIT n_steps={r['n_steps_max']}]" if r["n_steps_max"] == 5000 else ""
        print(f"{label:26s} dist={r['dist_mean']:6.2f}+-{r['dist_std']:5.2f}  "
              f"batt={r['batt_mean']:6.2f}+-{r['batt_std']:5.2f}%  "
              f"ct={r['ct_mean']:7.1f}+-{r['ct_std']:6.1f}{cap_note}")

    with open(os.path.join(SCRIPT_DIR, "overleaf_summary", "zero_drain_universal_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved overleaf_summary/zero_drain_universal_comparison.json")


if __name__ == "__main__":
    main()
