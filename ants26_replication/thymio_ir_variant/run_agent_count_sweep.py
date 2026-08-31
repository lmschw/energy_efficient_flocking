"""Sweeps optimize_hebbian.py (this directory's original, paper-faithful 3-stage/
10-input architecture) across multiple swarm sizes and multiple seeds per size, to
check whether/how results generalize across agent counts rather than trusting a single
(n_agents, seed) combination.

Pure orchestration: every training run is a separate `python optimize_hebbian.py ...`
subprocess, invoked exactly as you would by hand. This file adds no coupling to
optimize_hebbian.py's internals -- it only depends on its existing CLI. Copied from
../wall_sensor_variant/run_agent_count_sweep.py (same mechanism), with defaults
switched to this directory's default 3-stage curriculum and a 2-10 agent range.

Usage:
    python run_agent_count_sweep.py --wind-grid 50 --output-dir ../../hebbian_results_v2_thymio_ir_sweep
    python run_agent_count_sweep.py --n-agents-list 3 5 7 --seeds 42 123 --parallel 4   # smaller/faster look
"""
import argparse
import concurrent.futures
import datetime
import os
import subprocess
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIMIZE_SCRIPT = os.path.join(THIS_DIR, "optimize_hebbian.py")

DEFAULT_N_AGENTS_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# First 5 of optimize_hebbian.py's own 10 canonical seeds (config.HEBBIAN_BATCH_SEEDS) --
# duplicated here as a literal default rather than importing config, so this script's
# default doesn't silently drift if that constant is ever tuned independently.
DEFAULT_SEEDS = [42, 123, 777, 2026, 888]


def run_one(n_agents, seed, output_dir, stages, wind_grid, popsize, maxiter, n_repeats, battery):
    run_dir = os.path.join(output_dir, f"n{n_agents}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.txt")

    # "uv run" (not sys.executable) -- this script's own interpreter might be whatever
    # was active in the caller's shell (conda base, system python, etc.), which has no
    # reason to have cma/numpy/scipy/etc. installed. "uv run" always resolves to this
    # project's own environment (pyproject.toml/uv.lock at the repo root), regardless of
    # what launched this orchestration script or what cwd it's invoked from.
    cmd = ["uv", "run", OPTIMIZE_SCRIPT,
           "--n-agents", str(n_agents), "--seed", str(seed),
           "--stages", *stages,
           "--output-dir", run_dir]
    if wind_grid is not None:
        cmd += ["--wind-grid", str(wind_grid)]
    if popsize is not None:
        cmd += ["--popsize", str(popsize)]
    if maxiter is not None:
        cmd += ["--maxiter", str(maxiter)]
    if n_repeats is not None:
        cmd += ["--n-repeats", str(n_repeats)]
    if battery is not None:
        cmd += ["--battery", str(battery)]

    start = time.time()
    with open(log_path, "w") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n\n")
        log_file.flush()
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    return {"n_agents": n_agents, "seed": seed, "run_dir": run_dir, "log_path": log_path,
            "returncode": result.returncode, "elapsed_s": elapsed}


def main():
    parser = argparse.ArgumentParser(
        description="Sweep optimize_hebbian.py across multiple agent counts x seeds.")
    parser.add_argument("--n-agents-list", type=int, nargs="+", default=DEFAULT_N_AGENTS_LIST,
                         help=f"Agent counts to sweep (default: {DEFAULT_N_AGENTS_LIST}).")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                         help=f"Seeds to repeat per agent count (default: {DEFAULT_SEEDS}).")
    parser.add_argument("--stages", nargs="+", default=["walk_left", "save_battery_avoid_wall",
                                                          "save_battery_avoid_all"],
                         help="Passed through to optimize_hebbian.py --stages (default: the "
                              "paper's full 3-stage curriculum).")
    parser.add_argument("--output-dir", default="hebbian_results_v2_thymio_ir_sweep",
                         help="Base directory; each run gets its own "
                              "<output-dir>/n<N>_seed<S>/ subdirectory.")
    parser.add_argument("--wind-grid", type=int, default=None, help="Passed through.")
    parser.add_argument("--popsize", type=int, default=None, help="Passed through.")
    parser.add_argument("--maxiter", type=int, default=None, help="Passed through.")
    parser.add_argument("--n-repeats", type=int, default=None, help="Passed through.")
    parser.add_argument("--battery", type=float, default=None, help="Passed through.")
    parser.add_argument("--parallel", type=int, default=1,
                         help="How many runs to execute concurrently (default: 1, sequential). "
                              "Each run is CPU-heavy (the wind-marching loop), so don't set "
                              "this higher than your actual core count minus a little headroom.")
    args = parser.parse_args()

    combos = [(n, s) for n in args.n_agents_list for s in args.seeds]
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"🚀 Sweeping {len(args.n_agents_list)} agent counts x {len(args.seeds)} seeds = "
          f"{len(combos)} runs, stages={args.stages}, parallel={args.parallel}")
    print(f"   Results in '{args.output_dir}/n<N>_seed<S>/', each with its own log.txt.")

    results = []
    start_all = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(run_one, n, s, args.output_dir, args.stages, args.wind_grid,
                        args.popsize, args.maxiter, args.n_repeats, args.battery): (n, s)
            for n, s in combos
        }
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            n, s = futures[future]
            done_count += 1
            try:
                r = future.result()
                status = "OK" if r["returncode"] == 0 else f"FAILED (exit {r['returncode']})"
                print(f"[{done_count}/{len(combos)}] n_agents={n} seed={s} -> {status} "
                      f"({r['elapsed_s']/60:.1f} min) -- {r['log_path']}")
                results.append(r)
            except Exception as e:
                print(f"[{done_count}/{len(combos)}] n_agents={n} seed={s} -> CRASHED: {e}")
                results.append({"n_agents": n, "seed": s, "returncode": -1, "elapsed_s": None})

    total_elapsed = time.time() - start_all
    failed = [r for r in results if r["returncode"] != 0]
    print(f"\n🎉 Sweep complete in {total_elapsed/3600:.2f} hours "
          f"({datetime.timedelta(seconds=int(total_elapsed))}).")
    print(f"   {len(results) - len(failed)}/{len(results)} runs succeeded.")
    if failed:
        print("   Failed runs (check their log.txt for the actual error):")
        for r in failed:
            print(f"     n_agents={r['n_agents']} seed={r['seed']} -> {r.get('log_path', '?')}")


if __name__ == "__main__":
    main()
