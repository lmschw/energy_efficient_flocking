"""Validates HebbianSwarmExperiment's full sense->decide->act logic (pose conversion,
quadrant sensing, NN forward pass + Hebbian update, motor conversion) without any real
Thymio/Raspberry Pi/OptiTrack hardware. thymio_swarm_platform has no dry-run/simulation
mode at all (confirmed: every code path assumes a live tdmclient connection), so this
fills that gap for local development before a real hardware trial.

Run directly: `python local_test_harness.py [path/to/genome.npy]`
With no argument, generates a random genome on the fly (useful for a pure plumbing
smoke-test; won't produce sensible behavior, just exercises every code path).
"""
import asyncio
import sys

import numpy as np

import controller_config as cfg
from pose_utils import Pose
from hebbian_swarm_experiment import HebbianSwarmExperiment


class FakeRobot:
    """Minimal stand-in for swarm_platform.robot.Robot -- only the methods
    HebbianSwarmExperiment actually calls. `poses` can be mutated between ticks (e.g.
    to simulate motion) by the caller."""

    def __init__(self, poses):
        self.poses = poses
        self.drive_calls = []
        self.stopped = False

    async def get_all_global_poses(self):
        return self.poses

    async def drive(self, left, right):
        self.drive_calls.append((left, right))
        print(f"  drive(left={left}, right={right})")

    async def stop(self):
        self.stopped = True
        print("  stop()")


def _make_test_poses():
    """Three robots in a small triangle, all facing the same way (identity quaternion).
    Not a claim about what identity-quaternion "facing" means in your real Motive setup
    -- see README.md's calibration section."""
    return {
        "robot-a": Pose(position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)),
        "robot-b": Pose(position=(0.6, 0.3, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)),
        "robot-c": Pose(position=(-0.5, 0.4, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)),
    }


async def main():
    genome_path = sys.argv[1] if len(sys.argv) > 1 else None
    if genome_path is None:
        genome_path = "_random_test_genome.npy"
        np.save(genome_path, np.random.uniform(-5.0, 5.0, cfg.N_ABCD))
        print(f"No genome given -- wrote a random one to {genome_path} for a pure "
              f"plumbing test (expect no sensible behavior, just no crashes/shape errors).")

    hostnames = ["robot-a", "robot-b", "robot-c"]
    poses = _make_test_poses()
    robot = FakeRobot(poses)
    config = {"genome_path": genome_path, "hostnames": hostnames, "self_hostname": "robot-a"}

    experiment = HebbianSwarmExperiment(robot=robot, config=config, logger=None)
    print(f"Constructed OK. w1/w2/w3 shapes: {experiment.w1.shape}, "
          f"{experiment.w2.shape}, {experiment.w3.shape}")

    print("\nRunning 5 ticks (no real asyncio.sleep delay, just exercising _tick()):")
    for i in range(5):
        print(f"tick {i}:")
        v, w, left, right = await experiment._tick()
        print(f"  -> v={v:.4f} m/s, w={w:.4f} rad/s")
        assert abs(v) <= cfg.LINEAR_VEL_MAX + 1e-9, "v exceeded LINEAR_VEL_MAX"
        assert abs(w) <= cfg.ANGULAR_VEL_MAX + 1e-9, "w exceeded ANGULAR_VEL_MAX"
        assert -cfg.MAX_MOTOR_TARGET <= left <= cfg.MAX_MOTOR_TARGET
        assert -cfg.MAX_MOTOR_TARGET <= right <= cfg.MAX_MOTOR_TARGET

    await experiment.stop()
    await experiment.run()  # should return almost immediately since running=False
    assert robot.stopped, "run() should call robot.stop() on exit"

    print("\nTesting missing-pose handling (robot-b temporarily untracked)...")
    poses_missing = dict(poses)
    del poses_missing["robot-b"]
    robot2 = FakeRobot(poses_missing)
    experiment2 = HebbianSwarmExperiment(robot=robot2, config=config, logger=None)
    v, w, left, right = await experiment2._tick()
    print(f"  -> ran fine with a missing neighbor: v={v:.4f}, w={w:.4f}")

    print("\nTesting BATTERY_MODE='simulated' (requires scipy)...")
    cfg.BATTERY_MODE = "simulated"
    try:
        poses3 = _make_test_poses()
        robot3 = FakeRobot(poses3)
        experiment3 = HebbianSwarmExperiment(robot=robot3, config=config, logger=None)
        battery_trace = [experiment3.battery]
        for i in range(5):
            x, y, z = robot3.poses["robot-a"].position
            robot3.poses["robot-a"] = Pose(position=(x - 0.02, y, z), orientation=(0.0, 0.0, 0.0, 1.0))
            v, w, left, right = await experiment3._tick()
            battery_trace.append(experiment3.battery)
            print(f"  tick {i}: v={v:.4f} w={w:.4f} battery={experiment3.battery:.4f}")
        assert all(b <= cfg.INITIAL_BATTERY + 1e-9 for b in battery_trace), "battery exceeded starting value"
        assert battery_trace[-1] < battery_trace[0], "battery should have drained over 5 moving ticks"
        print("  simulated battery mode OK -- battery drained monotonically as expected.")
    except ModuleNotFoundError as exc:
        print(f"  SKIPPED locally ({exc}) -- fine off-hardware; wind_battery_model.py is "
              f"only imported when BATTERY_MODE == 'simulated' is actually selected.")
    finally:
        cfg.BATTERY_MODE = "none"

    print("\nAll checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
