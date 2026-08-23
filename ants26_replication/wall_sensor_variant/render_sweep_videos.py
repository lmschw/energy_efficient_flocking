"""Renders one video per result directory produced by run_agent_count_sweep.py (or any
directory laid out the same way: <sweep-dir>/<run>/hebbian_<stage>_best.npy).

Pure orchestration, like run_agent_count_sweep.py: every video is rendered by a separate
`uv run visualize_hebbian.py ...` subprocess -- "uv run" (not sys.executable) so this
still works no matter what environment launched this script itself (see
run_agent_count_sweep.py's docstring for why that distinction matters). Deliberately
stdlib-only so it can be invoked from any Python, including one without numpy/cv2/etc.
installed at all.

For each run directory, renders the LAST stage that actually has a saved genome (same
"use whatever's actually there" logic as analyze_hebbian_results.py's
_discover_available_stages) -- e.g. save_battery_avoid_all if present, else walk_left.

Usage:
    python render_sweep_videos.py --sweep-dir ../../hebbian_results_v2_wallsensor_sweep
    python render_sweep_videos.py --sweep-dir ../../hebbian_results_v2_wallsensor_sweep \\
        --battery 30 --parallel 4   # shorter videos, several at once
"""
import argparse
import concurrent.futures
import os
import subprocess
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALIZE_SCRIPT = os.path.join(THIS_DIR, "visualize_hebbian.py")

# Same stage list/order as config.HEBBIAN_STAGES -- duplicated as a literal here (not
# imported) so this script stays stdlib-only; see run_agent_count_sweep.py's DEFAULT_SEEDS
# for the same reasoning.
STAGE_PRIORITY = ["save_battery_avoid_all", "save_battery_avoid_wall", "walk_left"]


def find_genome(run_dir):
    """Returns (stage, genome_path) for the most-complete stage that has a saved genome
    in run_dir, or (None, None) if it has none at all (e.g. a sweep run that crashed
    before finishing any stage)."""
    for stage in STAGE_PRIORITY:
        path = os.path.join(run_dir, f"hebbian_{stage}_best.npy")
        if os.path.exists(path):
            return stage, path
    return None, None


def render_one(run_name, run_dir, stage, genome_path, battery, wind_grid, seed):
    video_path = os.path.join(run_dir, f"video_{stage}.mp4")
    log_path = os.path.join(run_dir, "render_log.txt")

    cmd = ["uv", "run", VISUALIZE_SCRIPT, genome_path, "--video-path", video_path]
    if battery is not None:
        cmd += ["--battery", str(battery)]
    if wind_grid is not None:
        cmd += ["--wind-grid", str(wind_grid)]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    start = time.time()
    with open(log_path, "w") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n\n")
        log_file.flush()
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    return {"run_name": run_name, "stage": stage, "video_path": video_path,
            "log_path": log_path, "returncode": result.returncode, "elapsed_s": elapsed}


def main():
    parser = argparse.ArgumentParser(
        description="Render a video for each result directory in a sweep.")
    parser.add_argument("--sweep-dir", required=True,
                         help="Directory containing one subdirectory per run (as produced "
                              "by run_agent_count_sweep.py).")
    parser.add_argument("--battery", type=float, default=None,
                         help="Override starting battery for the render (shorter battery "
                              "-> shorter video). Default: whatever the genome was trained "
                              "with (auto-detected from its _history.json).")
    parser.add_argument("--wind-grid", type=int, default=None,
                         help="Override wind grid resolution for the render (lower -> "
                              "faster render, coarser wake visual). Default: auto-detected.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Playback seed (default: visualize_hebbian.py's own default "
                              "for all renders, i.e. the same spawn config every video).")
    parser.add_argument("--parallel", type=int, default=1,
                         help="How many videos to render concurrently (default: 1). Each "
                              "render is CPU-heavy if wind is enabled -- don't set this "
                              "higher than your core count minus a little headroom.")
    args = parser.parse_args()

    run_dirs = sorted(
        d for d in (os.path.join(args.sweep_dir, name) for name in os.listdir(args.sweep_dir))
        if os.path.isdir(d)
    )
    if not run_dirs:
        raise FileNotFoundError(f"No subdirectories found in '{args.sweep_dir}'.")

    jobs = []
    skipped = []
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        stage, genome_path = find_genome(run_dir)
        if stage is None:
            skipped.append(run_name)
        else:
            jobs.append((run_name, run_dir, stage, genome_path))

    print(f"🎬 Rendering {len(jobs)} videos from '{args.sweep_dir}' (parallel={args.parallel}).")
    if skipped:
        print(f"   Skipping {len(skipped)} run(s) with no saved genome at all "
              f"(likely crashed before finishing any stage): {skipped}")

    results = []
    start_all = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(render_one, run_name, run_dir, stage, genome_path,
                        args.battery, args.wind_grid, args.seed): run_name
            for run_name, run_dir, stage, genome_path in jobs
        }
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            run_name = futures[future]
            done_count += 1
            try:
                r = future.result()
                status = "OK" if r["returncode"] == 0 else f"FAILED (exit {r['returncode']})"
                print(f"[{done_count}/{len(jobs)}] {run_name} ({r['stage']}) -> {status} "
                      f"({r['elapsed_s']:.1f}s) -- {r['video_path']}")
                results.append(r)
            except Exception as e:
                print(f"[{done_count}/{len(jobs)}] {run_name} -> CRASHED: {e}")
                results.append({"run_name": run_name, "returncode": -1})

    total_elapsed = time.time() - start_all
    failed = [r for r in results if r["returncode"] != 0]
    print(f"\n🎉 Done in {total_elapsed:.1f}s. {len(results) - len(failed)}/{len(results)} "
          f"videos rendered.")
    if failed:
        print("   Failed (check render_log.txt in each run directory):")
        for r in failed:
            print(f"     {r['run_name']} -> {r.get('log_path', '?')}")


if __name__ == "__main__":
    main()
