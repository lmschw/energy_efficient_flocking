"""Play back a swarm run for a trained Hebbian ABCD genome, rendering a real video from
the actual controller (hebbian_controller.py's MLP forward pass + Hebbian update).

NOT interchangeable with visualize_best.py: that script only understands 7-parameter LJ
genomes. A Hebbian ABCD genome (880 floats, from optimize_hebbian.py) fed into it doesn't
error out -- config.genome_to_rules() used to silently zip the first 7 raw ABCD weights
into (r0, epsilon, k_align, k_goal, K1, K2, U) and render meaningless garbage (now raises
a clear error instead). Use this script for Hebbian genomes.
"""
import argparse
import json
import os

import numpy as np

from experiment import config
from experiment.battery_plot import plot_battery_levels
from experiment.hebbian_controller import unflatten_abcd
from experiment.simulation_hebbian import render_hebbian_episode_video


def load_rules(genome_path):
    flat_genome = np.load(genome_path)
    if len(flat_genome) != config.HEBBIAN_N_ABCD:
        raise ValueError(
            f"'{genome_path}' has length {len(flat_genome)}, expected {config.HEBBIAN_N_ABCD} "
            f"(a Hebbian ABCD genome from optimize_hebbian.py). A length-7 genome is an LJ "
            f"genome instead -- use visualize_best.py for those.")
    print(f"Loaded Hebbian ABCD genome from '{genome_path}'")
    return unflatten_abcd(flat_genome)


def _load_history(genome_path):
    """optimize_hebbian.py saves 'hebbian_<stage>[_nosensor]_best.npy' alongside a
    sibling '..._history.json' recording the actual n_agents/battery/wind-grid/stage it
    was trained with. Returns that dict, or None if there's no sibling file (e.g. an
    older genome saved before this was recorded, or a hand-placed file) -- callers fall
    back to filename-pattern guessing in that case."""
    history_path = genome_path.replace("_best.npy", "_history.json")
    if history_path == genome_path or not os.path.exists(history_path):
        return None
    with open(history_path) as f:
        return json.load(f)


def _infer_wind_enabled(genome_path, history):
    """Stage 1 (walk_left) trains with wind disabled; stages 2/3 train with it enabled
    (config.HEBBIAN_STAGE_WIND_ENABLED). Prefers the history file's recorded 'stage'
    (authoritative); falls back to guessing the stage from the filename if there's no
    history file. Either way, this exists so forgetting --wind/--no-wind can't silently
    play back the wrong condition -- exactly the kind of mismatch that produced
    meaningless-looking video before (see module docstring)."""
    stage = (history or {}).get("stage")
    if stage is None:
        name = os.path.basename(genome_path)
        stage = next((s for s in config.HEBBIAN_STAGES if s in name), None)
    if stage is None:
        return True  # unrecognized -- default to the more common (wind-enabled) case
    return config.HEBBIAN_STAGE_WIND_ENABLED[stage]


def _infer_battery_sensor(genome_path, history):
    """Prefers the history file's recorded 'battery_sensor'; falls back to
    optimize_hebbian.py --no-battery-sensor's own '_nosensor' filename suffix
    convention if there's no history file."""
    if history is not None and "battery_sensor" in history:
        return history["battery_sensor"]
    return "_nosensor" not in os.path.basename(genome_path)


def _infer_from_history(history, key):
    """n_agents/max_battery/wind-grid: only recorded in the history file (no filename
    convention to fall back on), so this is None -- meaning 'use the simulation's own
    default' -- for older genomes saved before this was tracked."""
    return (history or {}).get(key)


def run_live_visualization(genome_path, seed, n_agents, wind_enabled, battery, wind_grid,
                            use_battery_sensor, video_path, battery_plot_path):
    rules = load_rules(genome_path)
    video_path = video_path or config.HEBBIAN_VIDEO_PATH
    print(f"n_agents={n_agents or config.HEBBIAN_N_AGENTS}, wind_enabled={wind_enabled}, "
          f"battery_sensor={use_battery_sensor}")

    result = render_hebbian_episode_video(
        rules, seed=seed, n_agents=n_agents, wind_enabled=wind_enabled,
        max_battery=battery, min_battery=battery, nx=wind_grid, ny=wind_grid,
        use_battery_sensor=use_battery_sensor, video_path=video_path,
        record_battery=battery_plot_path is not None)

    if battery_plot_path is not None:
        dist, batt, ct, wct, telemetry = result
    else:
        dist, batt, ct, wct = result

    print(f"Playback complete -- dist_travelled={dist:.4f}, avg_battery={batt:.4f}, "
          f"collision_time={ct:.2f}, wall_collision_time={wct:.2f}")
    print(f"Video saved to '{video_path}'.")

    if battery_plot_path is not None:
        plot_battery_levels(telemetry["battery"], config.DT, battery_plot_path,
                             title="Battery Level per Agent (Hebbian ABCD controller)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play back a swarm run for a saved Hebbian ABCD genome.")
    parser.add_argument("genome_path", nargs="?",
                         default="hebbian_results/hebbian_save_battery_avoid_all_best.npy",
                         help="Path to a .npy Hebbian ABCD genome (880 floats) from optimize_hebbian.py.")
    parser.add_argument("--seed", type=int, default=config.OPTIMIZE_SEED, help="Random seed for the run.")
    parser.add_argument("--n-agents", type=int, default=None,
                         help="Override the number of agents. Default: auto-detected from the "
                              "genome's sibling _history.json if present (the n_agents it was "
                              "actually trained with), else config.HEBBIAN_N_AGENTS. Flocking "
                              "behavior is visibly agent-count-sensitive, so playing back at the "
                              "wrong count can look like a bug in the controller when it's really "
                              "just a scale mismatch -- pass this explicitly to override.")
    parser.add_argument("--wind", choices=["auto", "on", "off"], default="auto",
                         help="Whether wind is enabled during playback. 'auto' (default) infers this "
                              "from the genome filename's stage name (walk_left -> off, "
                              "save_battery_avoid_wall/avoid_all -> on) -- override explicitly if you "
                              "renamed the file, since the wrong setting silently produces misleading "
                              "playback rather than an error.")
    parser.add_argument("--battery", type=float, default=None,
                         help=f"Starting battery for all agents (default: {config.HEBBIAN_MAX_BATTERY}).")
    parser.add_argument("--wind-grid", type=int, default=None,
                         help=f"Wind grid resolution, both axes (default: {config.HEBBIAN_NX}). Lower "
                              "this (e.g. 50) for a much faster render at the cost of a coarser wake.")
    parser.add_argument("--battery-sensor", choices=["auto", "on", "off"], default="auto",
                         help="Whether the NN's battery input reads the real value. 'auto' (default) "
                              "infers this from the genome filename ('_nosensor' suffix -> off, "
                              "matching optimize_hebbian.py --no-battery-sensor's own naming).")
    parser.add_argument("--video-path", default=None,
                         help=f"Where to save the video (default: {config.HEBBIAN_VIDEO_PATH}).")
    parser.add_argument("--plot-battery", action="store_true",
                         help="Also save a per-agent battery-vs-time plot.")
    parser.add_argument("--battery-plot-path", default="hebbian_battery_levels.png", metavar="PATH",
                         help="Where to save the battery plot (only used with --plot-battery).")
    args = parser.parse_args()

    history = _load_history(args.genome_path)
    if history is None:
        print(f"No sibling _history.json found for '{args.genome_path}' -- falling back to "
              f"filename-pattern guessing for wind/battery-sensor, and simulation defaults for "
              f"n_agents/battery/wind-grid. Pass --n-agents etc. explicitly if that's wrong.")

    wind_enabled = (_infer_wind_enabled(args.genome_path, history) if args.wind == "auto"
                    else args.wind == "on")
    battery_sensor = (_infer_battery_sensor(args.genome_path, history) if args.battery_sensor == "auto"
                       else args.battery_sensor == "on")
    n_agents = args.n_agents if args.n_agents is not None else _infer_from_history(history, "n_agents")
    battery = args.battery if args.battery is not None else _infer_from_history(history, "max_battery")
    wind_grid = args.wind_grid if args.wind_grid is not None else _infer_from_history(history, "wind_grid_nx")

    run_live_visualization(
        args.genome_path, args.seed, n_agents, wind_enabled=wind_enabled,
        battery=battery, wind_grid=wind_grid, use_battery_sensor=battery_sensor,
        video_path=args.video_path,
        battery_plot_path=args.battery_plot_path if args.plot_battery else None)
