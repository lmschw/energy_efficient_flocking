"""Controller-side launcher for diagnostics/calibrate_speed_experiment.py -- run this
BEFORE run_hardware_trial.py, from an environment where `swarm_platform` is importable
(e.g. thymio_swarm_platform's own venv), not from this repo's environment.

Deploys the `calibrate_speed` experiment (see README.md's "Calibration" section) to all
HOSTS at once, waits for the sweep to finish, collects logs, and prints each robot's
measured MOTOR_UNITS_PER_MPS plus the mean across all of them.

Prerequisite: `experiments/calibrate_speed/` (controller_config.py +
calibrate_speed_experiment.py) must already be committed and pushed in your
thymio_raspberry_swarm_control checkout, and registered in its swarm_project.yaml as:

    calibrate_speed:
      class: experiments.calibrate_speed.calibrate_speed_experiment.CalibrateSpeedExperiment
      tracking: true

Physical setup: give every robot in HOSTS a clear, straight runway (a couple of meters,
see calibrate_speed_experiment.py's docstring) and make sure none of their runways cross
-- they calibrate independently and don't sense each other.
"""
import asyncio

import pandas as pd

from swarm_platform.config import COORDINATOR_IP
from swarm_platform.controller.client import SwarmClient
from swarm_platform.utils.unpack_results import unpack_and_aggregate

REPOSITORY = "https://github.com/lmschw/thymio_raspberry_swarm_control.git"
HOSTS = ["thymio-15", "thymio-16", "thymio-17"]
SESSION_NAME = "calibrate-speed-run"
EXPERIMENT_NAME = "calibrate_speed"

MOTOR_TARGETS = [100, 200, 300, 400, 500]
HOLD_SECONDS = 3.0
SETTLE_SECONDS = 2.0
# Wall-clock budget: len(targets) * (hold + settle) per robot, run in parallel across
# HOSTS, plus a fixed buffer for install/activate/start round-trip latency.
RUN_SECONDS = len(MOTOR_TARGETS) * (HOLD_SECONDS + SETTLE_SECONDS) + 15


async def main():
    client = SwarmClient(COORDINATOR_IP)
    project = client.project(REPOSITORY, HOSTS)

    print("Installing...")
    await project.install()
    print("Updating...")
    await project.update()
    print("Activating...")
    await project.activate()

    session = project.session(SESSION_NAME)

    shared_config = {
        "motor_targets": MOTOR_TARGETS,
        "hold_seconds": HOLD_SECONDS,
        "settle_seconds": SETTLE_SECONDS,
    }
    host_configs = {h: {"self_hostname": h} for h in HOSTS}

    print(f"Starting calibration sweep on {HOSTS} (~{RUN_SECONDS:.0f}s)...")
    await session.start(EXPERIMENT_NAME, config=shared_config, host_configs=host_configs)

    await asyncio.sleep(RUN_SECONDS)

    print("Stopping...")
    await session.stop()

    print("Collecting logs...")
    await session.collect_logs(output_dir="results")
    df = unpack_and_aggregate(f"results/{SESSION_NAME}", f"results/{SESSION_NAME}/processed")

    print("Deleting remote logs...")
    await session.delete_logs()

    summary = df[df["phase"] == "summary"][["hostname", "recommended_units_per_mps"]]
    if summary.empty:
        print("\nNo summary rows found -- check the per-robot output above/logs for "
              "tracking or runway issues before retrying.")
        return

    print("\n=== Per-robot MOTOR_UNITS_PER_MPS ===")
    print(summary.to_string(index=False))
    mean_k = summary["recommended_units_per_mps"].mean()
    print(f"\nMean across {len(summary)} robot(s): {mean_k:.2f}")
    print("\nUpdate MOTOR_UNITS_PER_MPS in controller_config.py (both here in "
          "hardware_deployment/ AND in your thymio_raspberry_swarm_control checkout's "
          "experiments/hebbian_swarm/controller_config.py) to this value, then "
          "commit+push before running run_hardware_trial.py. If individual robots differ "
          "by a lot more than measurement noise would explain, consider per-robot "
          "calibration instead of one shared constant.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
