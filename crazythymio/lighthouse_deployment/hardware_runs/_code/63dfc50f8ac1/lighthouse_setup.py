"""Lighthouse set-up without the cfclient GUI (its 3D tab needs OpenGL): check what a board
sees, estimate the base-station geometry once, SAVE it, and write the saved system to any
number of other boards. Built on cflib's own estimator (the same code behind cfclient's
Lighthouse wizard, cflib/localization/); the guided procedure follows cflib's
examples/lighthouse/multi_bs_geometry_estimation.py.

  python lighthouse_setup.py status    --uri radio://0/100/2M/E7E7E7E701
  python lighthouse_setup.py calibrate --uri radio://0/100/2M/E7E7E7E701      # once per arena
  python lighthouse_setup.py upload    --ids 2,3,4,5,6,7                      # every other board

The result goes to lighthouse_config/lighthouse_system.yaml (geometry AND the base stations'
calibration data, the same file format cfclient's "Save/Load system config" uses, so the GUI
can load it too). One file is valid for all robots as long as the base stations don't move
and keep their channels: commit it, and re-run `calibrate` only if you move a base station.
Raw samples of each calibration run are kept in lighthouse_config/sessions/.

Coordinate frame defined by `calibrate`: origin where you put the board first, +x through the
second point (exactly 1 m away), +y by the third (in the floor plane, z up). Pick the origin
at the arena center (or note its position for run_hebbian.py --origin).
"""
import argparse
import os
import sys
import time
from threading import Event

import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.mem import LighthouseBsGeometry
from cflib.crazyflie.mem.lighthouse_memory import LighthouseMemHelper
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.localization.lighthouse_cf_pose_sample import LhCfPoseSample
from cflib.localization.lighthouse_config_manager import LighthouseConfigFileManager
from cflib.localization.lighthouse_config_manager import LighthouseConfigWriter
from cflib.localization.lighthouse_geo_estimation_manager import LhGeoEstimationManager
from cflib.localization.lighthouse_geo_estimation_manager import LhGeoInputContainer
from cflib.localization.lighthouse_geometry_solution import LighthouseGeometrySolution
from cflib.localization.lighthouse_sweep_angle_reader import LighthouseMatchedSweepAngleReader
from cflib.localization.lighthouse_sweep_angle_reader import LighthouseSweepAngleAverageReader
from cflib.localization.lighthouse_types import LhDeck4SensorPositions
from cflib.localization.user_action_detector import UserActionDetector

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(HERE, "lighthouse_config")
DEFAULT_FILE = os.path.join(CONFIG_DIR, "lighthouse_system.yaml")
REFERENCE_DIST = 1.0
MAX_BS = 4   # base-station slots the firmware supports (CONFIG_DECK_LIGHTHOUSE_MAX_N_BS in app-config);
             # writing more makes the board reject the extra slots and the write is reported as failed


def _geo_vec(geo):
    return np.r_[geo.origin, np.ravel(geo.rotation_matrix)]


def uri_for(robot_id, channel, datarate):
    return f"radio://0/{channel}/{datarate}/E7E7E7E7{robot_id:02X}"


def bitmask(value):
    return [i + 1 for i in range(16) if value & (1 << i)]   # base-station numbers, 1-indexed


def deck_report(scf):
    deck = scf.cf.param.get_value("deck.bcLighthouse4")
    system_type = scf.cf.param.get_value("lighthouse.systemType")
    print(f"  Lighthouse deck detected: {'yes' if deck == '1' else 'NO (deck.bcLighthouse4=' + deck + ')'}")
    print(f"  system type: {system_type} (1 = V1 base stations, 2 = V2)")
    return deck == "1", int(system_type)


def read_status(scf, seconds=3.0):
    lg = LogConfig("lhstatus", period_in_ms=200)
    for var in ("bsAvailable", "bsActive", "bsCalVal", "bsGeoVal", "bsReceive"):
        lg.add_variable(f"lighthouse.{var}", "uint16_t")
    lg.add_variable("lighthouse.status", "uint8_t")
    last = None
    t_end = time.time() + seconds
    with SyncLogger(scf, lg) as logger:
        for entry in logger:
            last = entry[1]
            if time.time() > t_end:
                break
    return last


def print_status(scf):
    deck_report(scf)
    s = read_status(scf)
    print(f"  base stations seen (available): {bitmask(s['lighthouse.bsAvailable'])}")
    print(f"  base stations in use (active):  {bitmask(s['lighthouse.bsActive'])}")
    print(f"  with calibration data:          {bitmask(s['lighthouse.bsCalVal'])}")
    print(f"  with geometry:                  {bitmask(s['lighthouse.bsGeoVal'])}")
    print(f"  status: {s['lighthouse.status']}")
    return s


def cmd_status(args):
    with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        print_status(scf)


def record_average(scf, timeout=5.0):
    result = {}
    done = Event()

    def ready(averages):
        result.update(averages)
        done.set()

    reader = LighthouseSweepAngleAverageReader(scf.cf, ready)
    reader.start_angle_collection()
    if not done.wait(timeout):
        print("  Recording timed out (no base station seen?).")
        return None
    sample = LhCfPoseSample({bs: data[1] for bs, data in result.items()})
    print(f"  recorded, base stations visible: {', '.join(str(b + 1) for b in result)}")
    if len(result) < 2:
        print("  Need >= 2 base stations in view -- try again.")
        return None
    return sample


def get_recording(scf):
    while True:
        input("  Place the board, keep it still and press Enter... ")
        sample = record_average(scf)
        if sample is not None:
            return sample
        time.sleep(1)


def get_recordings(scf):
    out = []
    while True:
        prompt = "  Enter = record a point, 'q' = done: " if out else "  Enter = record a point: "
        if input(prompt).strip().lower() == "q" and out:
            return out
        sample = record_average(scf)
        if sample is not None:
            out.append(sample)


def upload_geometry(scf, bs_poses):
    geos = {}
    for bs_id, pose in bs_poses.items():
        geo = LighthouseBsGeometry()
        geo.origin = pose.translation.tolist()
        geo.rotation_matrix = pose.rot_matrix.tolist()
        geo.valid = True
        geos[bs_id] = geo
    done = Event()
    LighthouseConfigWriter(scf.cf, nr_of_base_stations=MAX_BS).write_and_store_config(lambda ok: done.set(), geos=geos)
    done.wait()


def read_back(scf):
    """Geometry + calibration currently stored on the board, as objects for the config file."""
    helper = LighthouseMemHelper(scf.cf)
    out = {}
    for name, fn in (("geos", helper.read_all_geos), ("calibs", helper.read_all_calibs)):
        done = Event()
        fn(lambda data, name=name, done=done: (out.__setitem__(name, data), done.set()))
        if not done.wait(20):
            raise RuntimeError(f"timed out reading {name} back from the board")
    return out["geos"], out["calibs"]


def cmd_calibrate(args):
    os.makedirs(os.path.join(CONFIG_DIR, "sessions"), exist_ok=True)
    print(f"Connecting to {args.uri} ...")
    with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        has_deck, system_type = deck_report(scf)
        if not has_deck:
            sys.exit("No Lighthouse deck detected on this board -- fix that first (is it seated "
                     "on the Crazyflie, and is the firmware built with CONFIG_DECK_LIGHTHOUSE?).")
        print("Waiting for the board to receive the base stations' calibration data "
              "(needs line of sight, can take up to a minute) ...")
        t_end = time.time() + args.cal_timeout
        s = None
        while time.time() < t_end:
            s = read_status(scf, 1.0)
            seen = bitmask(s["lighthouse.bsAvailable"])
            have = bitmask(s["lighthouse.bsCalVal"])
            print(f"  seen {seen}, calibration received for {have}")
            if len(seen) >= 2 and set(seen) <= set(have):
                break
        else:
            sys.exit("Calibration data for >= 2 base stations never arrived. Check line of sight, "
                     "base-station channels and that the deck faces up.")

        container = LhGeoInputContainer(LhDeck4SensorPositions.positions)
        container.enable_auto_save(os.path.join(CONFIG_DIR, "sessions"))
        latest = {}

        def on_solution(solution: LighthouseGeometrySolution):
            print(f"      [solver] ok={solution.progress_is_ok} converged={solution.has_converged} "
                  f"info={solution.progress_info} error={solution.error_stats}")
            if solution.progress_is_ok:
                latest["solution"] = solution

        thread = LhGeoEstimationManager.SolverThread(container, is_done_cb=on_solution)
        thread.start()

        print("\nStep 1/4  ORIGIN: put the board where (0, 0) of the arena should be.")
        container.set_origin_sample(get_recording(scf))
        print(f"\nStep 2/4  +X AXIS: put the board EXACTLY {REFERENCE_DIST} m from the origin in the "
              "direction of +x (also sets the scale -- measure it).")
        container.set_x_axis_sample(get_recording(scf))
        print("\nStep 3/4  XY PLANE: put the board on the floor at one or more other points, NOT on the "
              "x axis (e.g. in the +y direction, away from the origin). Same height as the others.")
        container.set_xy_plane_samples(get_recordings(scf))

        print("\nStep 4/4  SPACE SAMPLES (improves accuracy): carry the board around the arena at "
              "robot height; at each spot, QUICKLY ROTATE it about the vertical axis to trigger a "
              "sample. Do 15-30 spots spread over the whole arena.")

        def matched_cb(sample):
            print("    sample stored")
            container.append_xyz_space_samples([sample])

        reader = LighthouseMatchedSweepAngleReader(
            scf.cf, matched_cb, timeout_cb=lambda: print("    timeout, no angles -- try again"))
        detector = UserActionDetector(scf.cf, cb=lambda: (print("    sampling..."), reader.start(timeout=1.0)))
        detector.start()
        input("  Press Enter when you have covered the arena ... ")
        detector.stop()
        time.sleep(1.5)            # let the solver finish the last version
        thread.stop()

        solution = latest.get("solution")
        if solution is None:
            sys.exit("No valid solution -- see the solver messages above (too few base stations in "
                     "view at origin/x-axis/plane points is the usual cause).")
        print("\nSolution:")
        for bs_id, pose in sorted(solution.bs_poses.items()):
            print(f"  base station {bs_id + 1} at {pose.translation.round(3).tolist()}")
        print(f"  error: {solution.error_stats}")
        if input("Write this to the board and save it? [y/N] ").strip().lower() != "y":
            sys.exit("Aborted, nothing written.")

        upload_geometry(scf, solution.bs_poses)
        geos, calibs = read_back(scf)
        os.makedirs(os.path.dirname(args.file), exist_ok=True)
        LighthouseConfigFileManager.write(args.file, geos=geos, calibs=calibs, system_type=system_type)
        print(f"\nSaved to {args.file}. Use `upload` for the other boards.")


def cmd_upload(args):
    geos, calibs, system_type = LighthouseConfigFileManager.read(args.file)
    print(f"Loaded {args.file}: base stations {sorted(k + 1 for k in geos)}, system type {system_type}")
    uris = list(args.uri or [])
    if args.ids:
        uris += [uri_for(int(i), args.channel, args.datarate) for i in args.ids.split(",")]
    if not uris:
        sys.exit("Give --uri (repeatable) and/or --ids.")
    failed = []
    for uri in uris:
        print(f"-> {uri}")
        try:
            with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:
                deck_report(scf)
                done, ok = Event(), []
                LighthouseConfigWriter(scf.cf, nr_of_base_stations=MAX_BS).write_and_store_config(
                    lambda success: (ok.append(success), done.set()),
                    geos=geos, calibs=calibs, system_type=system_type)
                if not done.wait(60) or not ok[0]:
                    raise RuntimeError("write/persist did not complete successfully")
                back_geos, _ = read_back(scf)
                for k, g in geos.items():
                    b = back_geos.get(k)
                    if b is None or not b.valid or np.abs(_geo_vec(g) - _geo_vec(b)).max() > 1e-4:
                        raise RuntimeError(f"read-back of base station {k + 1} does not match the file")
                print("   written, stored and verified by read-back")
        except Exception as e:
            print(f"   FAILED: {e}")
            failed.append(uri)
    if failed:
        sys.exit(f"Failed for: {failed}")
    print("All boards updated. Verify each with `status` (geometry for all base stations).")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("status", "calibrate", "upload"):
        sp = sub.add_parser(name)
        sp.add_argument("--file", default=DEFAULT_FILE)
        if name == "upload":
            sp.add_argument("--uri", action="append")
            sp.add_argument("--ids", help="robot radio ids, e.g. 2,3,4 -> radio://0/<channel>/<datarate>/E7E7E7E7<id>")
            sp.add_argument("--channel", type=int, default=100)
            sp.add_argument("--datarate", default="2M")
        else:
            sp.add_argument("--uri", required=True)
        if name == "calibrate":
            sp.add_argument("--cal-timeout", type=float, default=120.0)
    args = p.parse_args()
    cflib.crtp.init_drivers()
    {"status": cmd_status, "calibrate": cmd_calibrate, "upload": cmd_upload}[args.cmd](args)


if __name__ == "__main__":
    main()
