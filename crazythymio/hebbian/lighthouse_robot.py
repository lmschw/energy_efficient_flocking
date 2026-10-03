"""Drop-in replacement for thymio_swarm_platform's Robot, for the CrazyThymio hardware stack
(Thymio + Raspberry Pi + Crazyflie board with a Lighthouse deck, code from
https://github.com/fudavd/CrazyThymio), so ants26_replication/hardware_deployment's
HebbianSwarmExperiment runs on it unchanged.

It implements exactly the four methods HebbianSwarmExperiment calls:
  get_all_global_poses() -> {hostname: Pose}   (own pose from the Lighthouse Kalman estimate,
                                                neighbors from the board's radio (P2P) table)
  drive(left, right), stop()                    (Thymio motors via the Thymio Device Manager)
  proximity_horizontal()                        (Thymio IR sensors)

Needs the firmware in crazythymio/firmware/app_share_pos_hebbian/ on every Crazyflie board:
it broadcasts each robot's position over the radio and exposes the neighbor table through the
`ctr`, `nbA` and `nbB` log groups read below.
"""
import asyncio
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "ants26_replication", "hardware_deployment"))
from pose_utils import Pose  # noqa: E402

NO_NEIGHBOR = -32768          # int16 sentinel written by the firmware: not heard recently
POSE_TIMEOUT_S = 1.0          # own pose older than this counts as untracked
N_PEERS = 10                  # firmware neighbor table size (radio ids 1..10)


class ThymioLink:
    """Minimal tdmclient wrapper, same pattern as thymio_swarm_platform's ThymioConnection."""

    def __init__(self):
        self.client = None
        self.node = None
        self._poll_task = None

    async def connect(self, timeout=10.0):
        from tdmclient import ClientAsync
        self.client = ClientAsync()
        self.client.__enter__()
        loop = asyncio.get_running_loop()
        start, seen, seen_since = loop.time(), None, None
        while loop.time() - start < timeout:
            self.client.process_waiting_messages()
            nodes = list(self.client.nodes)
            if nodes:
                if nodes[0] != seen:
                    seen, seen_since = nodes[0], loop.time()
                elif loop.time() - seen_since > 0.5:
                    break
            else:
                seen = None
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError("No Thymio found -- is thymio-device-manager running and the "
                               "Thymio connected over USB?")
        self.node = seen
        await self.node.lock()
        await self.node.watch(variables=True, events=True)
        for _ in range(50):
            self.client.process_waiting_messages()
            if self.node.var.get("prox.horizontal") is not None:
                break
            await asyncio.sleep(0.05)
        self._poll_task = asyncio.create_task(self._poll())

    async def _poll(self):
        while True:
            self.client.process_waiting_messages()
            await asyncio.sleep(0.01)

    async def drive(self, left, right):
        await asyncio.wait_for(self.node.set_variables({
            "motor.left.target": [int(left)], "motor.right.target": [int(right)]}), timeout=2.0)

    def proximity_horizontal(self):
        return list(self.node.var.get("prox.horizontal"))

    async def disconnect(self):
        try:
            await self.drive(0, 0)
        except Exception:
            pass
        if self._poll_task is not None:
            self._poll_task.cancel()
        try:
            await self.node.unlock()
        except Exception:
            pass
        try:
            self.client.__exit__(None, None, None)
        except Exception:
            pass


class LighthouseLink:
    """Crazyflie board over USB: own pose + neighbor table, via cflib log callbacks."""

    def __init__(self, uri="usb://0"):
        self.uri = uri
        self.scf = None
        self.own = {}                       # latest values from the log callbacks
        self.own_time = 0.0                 # wall-clock time of the last `ctr` update
        self.neighbors = {}                 # radio id -> (x_m, y_m), only fresh ones

    def connect(self, reset_estimator=True):
        import cflib.crtp
        from cflib.crazyflie import Crazyflie
        from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
        from cflib.crazyflie.log import LogConfig

        cflib.crtp.init_drivers()
        self.scf = SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache="./cache"))
        self.scf.open_link()
        cf = self.scf.cf
        if reset_estimator:
            # Robot must be standing still, in view of the base stations.
            cf.param.set_value("kalman.resetEstimation", "1")
            time.sleep(0.1)
            cf.param.set_value("kalman.resetEstimation", "0")
            time.sleep(1.0)

        own = LogConfig(name="own", period_in_ms=100)
        for var in ("ctr.x", "ctr.y"):
            own.add_variable(var, "float")
        for var in ("stateEstimate.z", "stateEstimate.yaw"):
            own.add_variable(var, "float")
        own.data_received_cb.add_callback(self._on_own)
        cf.log.add_config(own)
        own.start()

        for name, ids in (("nbA", range(1, 6)), ("nbB", range(6, 11))):
            cfg = LogConfig(name=name, period_in_ms=100)
            for i in ids:
                cfg.add_variable(f"{name}.x{i}", "int16_t")
                cfg.add_variable(f"{name}.y{i}", "int16_t")
            cfg.data_received_cb.add_callback(self._on_neighbors)
            cf.log.add_config(cfg)
            cfg.start()

    def _on_own(self, timestamp, data, logconf):
        self.own = dict(data)
        self.own_time = time.time()

    def _on_neighbors(self, timestamp, data, logconf):
        for key, value in data.items():
            group, name = key.split(".")
            peer, axis = int(name[1:]), name[0]
            if axis != "x":
                continue
            y = data[f"{group}.y{peer}"]
            if value == NO_NEIGHBOR or y == NO_NEIGHBOR:
                self.neighbors.pop(peer, None)
            else:
                self.neighbors[peer] = (value / 1000.0, y / 1000.0)

    def close(self):
        if self.scf is not None:
            self.scf.close_link()


class LighthouseRobot:
    """See module docstring. `ids` maps hostname -> radio id (1..10) of that robot's board;
    `origin_xy` is subtracted from every Lighthouse position, so the simulation frame's
    (0, 0) can be put at the arena center (Lighthouse's own origin is wherever the geometry
    wizard put it)."""

    def __init__(self, self_hostname, ids, uri="usb://0", origin_xy=(0.0, 0.0)):
        self.self_hostname = self_hostname
        self.ids = dict(ids)
        self.origin_xy = origin_xy
        self.thymio = ThymioLink()
        self.board = LighthouseLink(uri)

    async def connect(self):
        await self.thymio.connect()
        self.board.connect()

    def own_pose_record(self):
        """(x, y, z, yaw_rad) in the origin-shifted frame, or None if the board has not
        reported recently."""
        b = self.board
        if not b.own or time.time() - b.own_time > POSE_TIMEOUT_S:
            return None
        return (b.own["ctr.x"] - self.origin_xy[0], b.own["ctr.y"] - self.origin_xy[1],
                b.own["stateEstimate.z"], math.radians(b.own["stateEstimate.yaw"]))

    async def get_all_global_poses(self):
        poses = {}
        own = self.own_pose_record()
        z = 0.0
        if own is not None:
            x, y, z, yaw = own
            poses[self.self_hostname] = Pose(
                position=(x, y, z),
                orientation=(0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)))
        by_id = dict(self.board.neighbors)
        for host, rid in self.ids.items():
            if host == self.self_hostname or rid not in by_id:
                continue
            nx, ny = by_id[rid]
            # Heading of neighbors is not used by the controller (only its position is).
            poses[host] = Pose(position=(nx - self.origin_xy[0], ny - self.origin_xy[1], z),
                               orientation=(0.0, 0.0, 0.0, 1.0))
        return poses

    async def drive(self, left, right):
        await self.thymio.drive(left, right)

    async def stop(self):
        await self.thymio.drive(0, 0)

    async def proximity_horizontal(self):
        return self.thymio.proximity_horizontal()

    async def disconnect(self):
        await self.thymio.disconnect()
        self.board.close()
