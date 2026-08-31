"""For every n<N>_seed<S>/ run directory produced by run_agent_count_sweep.py, runs
analyze_hebbian_results.py (the paper's Fig. 5a/5b/6 validation plots) and renders one
playback video of that run's most-trained (save_battery_avoid_all) genome.

Pure orchestration, same pattern as run_agent_count_sweep.py: every unit of work is a
separate subprocess, invoked exactly as you would by hand. Discovers run directories by
their 'n<N>_seed<S>' naming convention rather than taking an explicit list, since that's
exactly what run_agent_count_sweep.py produces.

Usage:
    python run_analysis_and_video_sweep.py --sweep-dir ../../hebbian_results_v2_thymio_ir_sweep --parallel 8
"""
import argparse
import concurrent.futures
import datetime
import os
import re
import subprocess
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYZE_SCRIPT = os.path.join(THIS_DIR, "analyze_hebbian_results.py")
VISUALIZE_SCRIPT = os.path.join(THIS_DIR, "visualize_hebbian.py")
# NOT "../visualize_hebbian.py" (the shared base script, as ../experiment/'s and
# ../upwind_variant/'s copies of this file both still point to) -- that one hardcodes
# `from experiment import config`, which would load ../experiment/config.py's 10-input
# architecture instead of this variant's own 9-input one (HEBBIAN_N_ABCD would mismatch
# and unflatten_abcd() would immediately raise -- confirmed by inspection, not yet hit
# in practice). Harmless for ../upwind_variant/ (its config only differs from
# ../experiment/'s in ways that don't affect rendering an already-trained
# save_battery_avoid_all genome), but NOT harmless here, since the whole point of this
# variant is a different sensor_model.py/config.HEBBIAN_N_INPUTS. Use this variant's
# own visualize_hebbian.py copy instead (same pattern wall_sensor_variant/ and
# straight_reward_variant/ use, even though neither of those has its own copy of this
# particular sweep script).

RUN_DIR_RE = re.compile(r"^n(\d+)_seed(\d+)$")
VIDEO_GENOME = "hebbian_save_battery_avoid_all_best.npy"


def discover_runs(sweep_dir):
    runs = []
    for name in sorted(os.listdir(sweep_dir)):
        m = RUN_DIR_RE.match(name)
        if m and os.path.isdir(os.path.join(sweep_dir, name)):
            runs.append((int(m.group(1)), int(m.group(2)), os.path.join(sweep_dir, name)))
    return runs


def run_one(n_agents, seed, run_dir, n_sims, n_trajectory_repeats):
    result = {"n_agents": n_agents, "seed": seed, "run_dir": run_dir}

    analysis_out = os.path.join(run_dir, "hebbian_analysis")
    os.makedirs(analysis_out, exist_ok=True)
    analysis_log = os.path.join(run_dir, "analysis_log.txt")
    cmd = ["uv", "run", ANALYZE_SCRIPT,
           "--results-dir", run_dir, "--n-agents", str(n_agents),
           "--output-dir", analysis_out,
           "--n-sims", str(n_sims), "--n-trajectory-repeats", str(n_trajectory_repeats)]
    start = time.time()
    with open(analysis_log, "w") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n\n")
        log_file.flush()
        r = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    result["analysis_returncode"] = r.returncode
    result["analysis_elapsed_s"] = time.time() - start
    result["analysis_log"] = analysis_log

    genome_path = os.path.join(run_dir, VIDEO_GENOME)
    video_path = os.path.join(run_dir, "hebbian_video.mp4")
    video_log = os.path.join(run_dir, "video_log.txt")
    cmd = ["uv", "run", VISUALIZE_SCRIPT, genome_path,
           "--seed", str(seed), "--video-path", video_path]
    start = time.time()
    with open(video_log, "w") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n\n")
        log_file.flush()
        r = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    result["video_returncode"] = r.returncode
    result["video_elapsed_s"] = time.time() - start
    result["video_log"] = video_log

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate analysis plots + a playback video for every run in a sweep directory.")
    parser.add_argument("--sweep-dir", default="hebbian_results_v2_thymio_ir_sweep",
                         help="Directory containing n<N>_seed<S>/ run subdirectories.")
    parser.add_argument("--n-sims", type=int, default=100,
                         help="Simulations per controller/condition for analyze_hebbian_results.py "
                              "(paper default: 100).")
    parser.add_argument("--n-trajectory-repeats", type=int, default=10,
                         help="Seeds for the trajectory-repeats sanity check (default: 10).")
    parser.add_argument("--parallel", type=int, default=1,
                         help="How many runs to process concurrently.")
    args = parser.parse_args()

    runs = discover_runs(args.sweep_dir)
    print(f"🚀 Found {len(runs)} run directories in '{args.sweep_dir}'. "
          f"Generating analysis plots (n_sims={args.n_sims}) + 1 video each, parallel={args.parallel}")

    results = []
    start_all = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(run_one, n, s, run_dir, args.n_sims, args.n_trajectory_repeats): (n, s)
            for n, s, run_dir in runs
        }
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            n, s = futures[future]
            done_count += 1
            try:
                r = future.result()
                a_status = "OK" if r["analysis_returncode"] == 0 else f"FAILED({r['analysis_returncode']})"
                v_status = "OK" if r["video_returncode"] == 0 else f"FAILED({r['video_returncode']})"
                print(f"[{done_count}/{len(runs)}] n_agents={n} seed={s} -> "
                      f"analysis={a_status} ({r['analysis_elapsed_s']/60:.1f}min), "
                      f"video={v_status} ({r['video_elapsed_s']:.0f}s)")
                results.append(r)
            except Exception as e:
                print(f"[{done_count}/{len(runs)}] n_agents={n} seed={s} -> CRASHED: {e}")
                results.append({"n_agents": n, "seed": s, "analysis_returncode": -1, "video_returncode": -1})

    total_elapsed = time.time() - start_all
    failed = [r for r in results
              if r.get("analysis_returncode") != 0 or r.get("video_returncode") != 0]
    print(f"\n🎉 Done in {total_elapsed/3600:.2f} hours ({datetime.timedelta(seconds=int(total_elapsed))}).")
    print(f"   {len(results) - len(failed)}/{len(results)} runs fully succeeded (analysis + video).")
    if failed:
        print("   Runs with a failure (check their analysis_log.txt/video_log.txt):")
        for r in failed:
            print(f"     n_agents={r['n_agents']} seed={r['seed']} -> {r.get('run_dir', '?')}")


if __name__ == "__main__":
    main()
