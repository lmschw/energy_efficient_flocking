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
        self.neighbors = {233: (2.0, 2.0), 235: (1.0, 1.5)}   # 0.5 m ahead, 1 m behind-left


async def main():
    ids = {"a": 1, "b": 233, "c": 235}
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
    robot.board.own_time = 0  # stale -> untracked: must still run
    cfg.SAFETY_LAYERS_ENABLED = True       # with the safety layers ON the crawl cap applies ...
    v, *_ = await exp._tick()
    assert abs(v) <= cfg.UNTRACKED_SAFE_V_CAP + 1e-9
    cfg.SAFETY_LAYERS_ENABLED = False      # ... and with them OFF (the current default) it must not
    exp2 = HebbianSwarmExperiment(robot, {"genome_path": path, "hostnames": list(ids), "self_hostname": "a"})
    v2, *_ = await exp2._tick()
    print("untracked tick, safety OFF: v =", round(v2, 3), "(not capped)")
    os.remove(path)
    print("OFFLINE TEST PASSED")

asyncio.run(main())


# ---- added 2026-10-04: LJ baseline tick, x governor, IR backoff disabled ----------------------------
async def extra():
    import hebbian_swarm_experiment as hse
    from lj_baseline_experiment import LJBaselineExperiment
    names = ["a", "b", "c"]
    robot = LighthouseRobot("a", {"a": 1, "b": 233, "c": 235})
    robot.thymio, robot.board = FakeThymio(), FakeBoard()
    robot.board.neighbors = {233: (2.0, 2.0), 235: (1.0, 1.5)}
    exp = LJBaselineExperiment(robot, {"hostnames": names, "self_hostname": "a"})
    for _ in range(3):
        robot.board.own_time = time.time()
        v, w, l, r = await exp._tick()
    print("LJ tick ok:", round(v, 3), round(w, 3), l, r)
    # IR backoff must be a no-op with the switch off, even for a huge reading
    assert cfg.IR_BACKOFF_ENABLED is False
    assert hse._apply_ir_backoff(0.1, [9999] * 7) == 0.1
    # x governor: disabled by default, full slowdown at the limit when set
    assert hse._corridor_x_speed_scale(-5.0) == 1.0
    cfg.CORRIDOR_X_MIN, cfg.CORRIDOR_X_MAX = -2.0, 1.5
    assert hse._corridor_x_speed_scale(0.0) == 1.0 and hse._corridor_x_speed_scale(-2.0) == 0.0
    assert abs(hse._corridor_x_speed_scale(-1.75) - 0.5) < 1e-9
    cfg.CORRIDOR_X_MIN = cfg.CORRIDOR_X_MAX = None
    print("EXTRA OFFLINE TESTS PASSED")

asyncio.run(extra())


# ---- direction-aware governor (added 2026-10-04 after the first real run trapped robots past a limit) ----
def governor_checks():
    import hebbian_swarm_experiment as hse
    cfg.CORRIDOR_X_MIN, cfg.CORRIDOR_X_MAX = -1.64, 1.20
    cfg.CORRIDOR_Y_MIN, cfg.CORRIDOR_Y_MAX = -2.04, 1.50
    def agents(x, y, face_deg):
        a = np.zeros((1, 4)); a[0, 0], a[0, 1] = x, y; a[0, 2] = math.radians(face_deg) - math.pi / 2; return a
    # past the +x limit facing +x (outward): blocked; facing -x (inward): free
    assert hse._corridor_scale(agents(1.30, 0, 0), 0, 0.1) == 0.0
    assert hse._corridor_scale(agents(1.30, 0, 180), 0, 0.1) == 1.0
    # reversing out while facing outward is allowed too (velocity points inward)
    assert hse._corridor_scale(agents(1.30, 0, 0), 0, -0.1) == 1.0
    # mid-arena: no slowdown; approaching the -x limit slows linearly (half way through the 0.5 m margin)
    assert hse._corridor_scale(agents(0.0, 0, 180), 0, 0.1) == 1.0
    assert abs(hse._corridor_scale(agents(-1.39, 0, 180), 0, 0.1) - 0.5) < 1e-6
    # y walls behave the same way
    assert hse._corridor_scale(agents(0, 1.6, 90), 0, 0.1) == 0.0 and hse._corridor_scale(agents(0, 1.6, 270), 0, 0.1) == 1.0
    cfg.CORRIDOR_X_MIN = cfg.CORRIDOR_X_MAX = cfg.CORRIDOR_Y_MIN = cfg.CORRIDOR_Y_MAX = None
    print("GOVERNOR TESTS PASSED")

governor_checks()
