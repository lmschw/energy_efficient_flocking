"""Hardware-free check of the Lighthouse -> HebbianSwarmExperiment wiring (fake Crazyflie
board and Thymio): python crazythymio/lighthouse_deployment/test_offline.py"""
import asyncio
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import controller_config as cfg
from lighthouse_robot import LighthouseRobot, NO_NEIGHBOR
from hebbian_swarm_experiment import HebbianSwarmExperiment
import pose_utils
for m in (cfg, pose_utils, HebbianSwarmExperiment.__module__ and sys.modules[HebbianSwarmExperiment.__module__]):
    assert os.path.dirname(os.path.abspath(m.__file__)) == os.path.abspath(HERE), m.__file__  # isolation



class FakeThymio:
    def __init__(self): self.cmds = []
    async def drive(self, l, r): self.cmds.append((l, r))
    def proximity_horizontal(self): return [0] * 7


class FakeBoard:
    def __init__(self):
        self.own = {"ctr.x": 2.0, "ctr.y": 1.5, "stateEstimate.z": 0.05,
                    "stateEstimate.yaw": 90.0}   # facing +y in the Lighthouse frame
        self.own_time = time.time()
        self.neighbors = {2: (2.0, 2.0), 3: (1.0, 1.5)}   # 0.5 m ahead, 1 m behind-left


async def main():
    ids = {"a": 1, "b": 2, "c": 3}
    robot = LighthouseRobot("a", ids, origin_xy=(2.0, 1.5))
    robot.thymio, robot.board = FakeThymio(), FakeBoard()
    poses = await robot.get_all_global_poses()
    assert set(poses) == {"a", "b", "c"}, poses
    path = os.path.join(HERE, "_t.npy")
    np.save(path, np.random.uniform(-5, 5, cfg.N_ABCD))
    exp = HebbianSwarmExperiment(robot, {"genome_path": path, "hostnames": list(ids), "self_hostname": "a"})
    for _ in range(3):
        robot.board.own_time = time.time()
        v, w, l, r = await exp._tick()
    from pose_utils import poses_to_agents
    agents, i = poses_to_agents(poses, list(ids), "a")
    print("agents (x, y, heading, batt):\n", agents.round(3))
    assert abs(agents[0, 2]) < 1e-6, "facing +y must be sim heading 0"
    assert abs(agents[1, 1] - 0.5) < 1e-6
    print("ticks ok; last command", v, w, l, r)
    robot.board.own_time = 0  # stale -> untracked: must still run, capped slow
    v, *_ = await exp._tick()
    assert abs(v) <= cfg.UNTRACKED_SAFE_V_CAP + 1e-9
    os.remove(path)
    print("OFFLINE TEST PASSED")

asyncio.run(main())
