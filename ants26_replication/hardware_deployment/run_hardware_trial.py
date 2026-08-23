"""Controller-side launcher for the real 3-agent Hebbian ABCD hardware trial. Run from an
environment where `swarm_platform` is importable (e.g. thymio_swarm_platform's own venv),
not from this repo's environment -- same pattern as
thymio_swarm_platform/examples/decision_external_repo.py, adapted to this project's
experiment (host_configs instead of one session.start() call per host, which is the
simpler way to give each robot a different self_hostname -- see
swarm_platform.controller.session.SwarmSession.start()).

Prerequisites (see README.md) -- do NOT run this until all of these are true:
1. `python local_test_harness.py` passes locally.
2. Calibrated on your actual rig: diagnostics/print_poses_experiment.py (POSITION_AXES,
   HEADING_OFFSET_RAD) and run_speed_calibration.py (MOTOR_UNITS_PER_MPS) --
   controller_config.py's hardware-calibration constants are no longer placeholders.
3. `experiments/hebbian_swarm/` (this package's *.py files, minus local_test_harness.py,
   plus GENOME_PATH_ON_PI's .npy) exists in your thymio_raspberry_swarm_control checkout,
   is registered in its swarm_project.yaml as:
       hebbian_swarm:
         class: experiments.hebbian_swarm.hebbian_swarm_experiment.HebbianSwarmExperiment
         tracking: true
   and has been committed and pushed -- the Pis pull via git, they never see this local
   checkout of energy_efficient_flocking.

Genome: hebbian_save_battery_avoid_all_best.npy -- trained through all 3 curriculum
stages (walk_left -> save_battery_avoid_wall -> save_battery_avoid_all) at the paper's
default n_agents=20, NOT re-trained at n_agents=3. Running it with only 3 real neighbors
is a real, disclosed sim-to-real gap (much sparser than what it was evolved against) --
see README.md. controller_config.BATTERY_MODE must be "simulated": this genome was
trained WITH the battery sensor, so "none" mode (constant placeholder) would feed it an
input it never learned to use.
"""
import asyncio
import time

from swarm_platform.config import COORDINATOR_IP
from swarm_platform.controller.client import SwarmClient
from swarm_platform.utils.unpack_results import unpack_and_aggregate

REPOSITORY = "https://github.com/lmschw/thymio_raspberry_swarm_control.git"
HOSTS = ["thymio-15", "thymio-16", "thymio-17"]
SESSION_NAME = "hebbian-swarm-3agent-run"
EXPERIMENT_NAME = "hebbian_swarm"
# hebbian_swarm_experiment.py resolves a relative path like this one against its OWN
# file's directory if it doesn't already resolve against the daemon's cwd (which is not
# reliably project_root -- see _resolve_genome_path()'s docstring), so this just needs to
# be the basename of the .npy sitting next to hebbian_swarm_experiment.py on the Pi.
GENOME_PATH_ON_PI = "hebbian_save_battery_avoid_all_best.npy"
EXPERIMENT_DURATION_SECONDS = 120  # first live trial: keep this short, extend once you've

# watched it behave sanely for a couple of minutes.


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
        "genome_path": GENOME_PATH_ON_PI,
        "hostnames": HOSTS,  # identical list/order on every robot
    }
    host_configs = {h: {"self_hostname": h} for h in HOSTS}  # differs per robot

    print(f"Starting (duration={EXPERIMENT_DURATION_SECONDS}s)...")
    start_time = time.monotonic()
    await session.start(EXPERIMENT_NAME, config=shared_config, host_configs=host_configs)

    try:
        while True:
            remaining = EXPERIMENT_DURATION_SECONDS - (time.monotonic() - start_time)
            try:
                cmd = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, input, "\n[p]ause  [r]esume  [s]top > "
                    ),
                    timeout=max(0, remaining),
                )
            except asyncio.TimeoutError:
                print("Experiment duration elapsed. Stopping...")
                break

            cmd = cmd.strip().lower()
            if cmd == "p":
                print("Pausing...")
                await session.pause()
            elif cmd == "r":
                print("Resuming...")
                await session.resume()
            elif cmd == "s":
                print("Stopping...")
                break
    finally:
        print("Stopping...")
        try:
            await session.stop()
        except Exception as e:
            print(f"Failed to stop swarm: {e}")

        print("Collecting logs...")
        await session.collect_logs(output_dir="results")
        unpack_and_aggregate(f"results/{SESSION_NAME}", f"results/{SESSION_NAME}/processed")

        print("Deleting remote logs...")
        await session.delete_logs()
        print("Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
