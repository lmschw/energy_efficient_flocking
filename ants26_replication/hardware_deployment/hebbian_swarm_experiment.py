"""Experiment class deploying a genome trained with optimize_hebbian.py
(energy_efficient_flocking/experiment/optimize_hebbian.py) onto real Thymio+Pi hardware
via thymio_swarm_platform.

Matches that platform's de-facto experiment contract (there is no formal base class --
see README.md): __init__(robot, config, logger), async run()/pause()/resume()/stop().

The whole energy_efficient_flocking repo IS the deployable project: /swarm_project.yaml
(at the REPO ROOT -- see that file's header comment for why it can't live in this
directory despite this being the only code it actually needs) registers this class, so
thymio_swarm_platform's controller-side scripts
(thymio_swarm_platform/examples/hebbian_swarm_trial.py) can point client.project()
straight at this repo's GitHub remote -- no copying into another repo. Start a session
with a config dict providing at least genome_path, hostnames, and self_hostname (see
README.md for the full walkthrough).
"""
import asyncio
import math
import os
import sys

# thymio_swarm_platform's ProjectLoader only adds the project's ROOT directory to
# sys.path (see loader.py) -- since project root is now the whole repo (swarm_project.yaml
# lives at its top level, not in this directory -- see that file's header comment), this
# file's own directory is NOT on sys.path by default, so the bare `import controller_config`
# below (and the same pattern in sensor_model.py/hebbian_controller.py/pose_utils.py/
# motor_utils.py/wind_battery_model.py) would raise ModuleNotFoundError without this.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import controller_config as cfg
import wind_battery_model
from sensor_model import get_sensor_data
from hebbian_controller import init_weights, hebbian_step, unflatten_abcd
from pose_utils import poses_to_agents
from motor_utils import velocity_to_motor_targets


def _corridor_speed_scale(y):
    """Deployment-only wall-safety governor -- see CORRIDOR_Y_MIN/MAX's comment in
    controller_config.py for why this exists (the genome has no wall sense of its own).
    Returns a [0, 1] multiplier for v: 1.0 away from both walls, scaling linearly down to
    0.0 over the last CORRIDOR_SLOWDOWN_MARGIN_M before either wall. Disabled (always
    1.0) until both CORRIDOR_Y_MIN and CORRIDOR_Y_MAX are set."""
    if cfg.CORRIDOR_Y_MIN is None or cfg.CORRIDOR_Y_MAX is None:
        return 1.0
    margin = min(y - cfg.CORRIDOR_Y_MIN, cfg.CORRIDOR_Y_MAX - y)
    if margin <= 0.0:
        return 0.0
    return min(1.0, margin / cfg.CORRIDOR_SLOWDOWN_MARGIN_M)


class HebbianSwarmExperiment:
    # NOTE: the parameter must be named exactly `config` (not e.g. config_dict) --
    # thymio_swarm_platform's daemon instantiates every experiment with the keyword
    # call experiment_cls(robot=self.robot, config=config, logger=self.logger).
    def __init__(self, robot, config=None, logger=None):
        self.robot = robot
        self.config = config or {}
        self.logger = logger
        self.running = True
        self.paused = False

        if "genome_path" not in self.config:
            raise ValueError("config['genome_path'] is required -- point it at a "
                              "hebbian_<stage>_best.npy from optimize_hebbian.py "
                              "(ideally one trained with --no-battery-sensor, since no "
                              "real battery reading exists on this hardware).")
        if "hostnames" not in self.config or "self_hostname" not in self.config:
            raise ValueError("config['hostnames'] (the full participating swarm, same "
                              "list/order on every robot) and config['self_hostname'] "
                              "are both required.")

        self.rules = unflatten_abcd(np.load(self._resolve_genome_path(self.config["genome_path"])))
        self.hostnames = list(self.config["hostnames"])
        self.self_hostname = self.config["self_hostname"]
        self.w1, self.w2, self.w3 = init_weights()

        # Simulated battery state (BATTERY_MODE == "simulated") -- see wind_battery_model.py.
        self.battery = cfg.INITIAL_BATTERY
        self._prev_position = None  # (x, y) as of the previous _tick(), for the
                                     # position-delta velocity estimate batterydrainage()
                                     # needs (no wheel encoders exist on this platform).
                                     # Only ever set from a TRACKED pose -- see
                                     # _is_tracked()'s use in _tick(); never the
                                     # pose_utils.py untracked (1e4, 1e4) sentinel, or one
                                     # stale tick later that sentinel becomes a ~14000m,
                                     # one-tick "delta" that instantly zeroes the battery.
        self._last_w = None         # commanded angular velocity that was actually active
                                     # over the interval since _prev_position was recorded.
        # wind_battery_model has no extra dependencies beyond numpy (already required
        # regardless of BATTERY_MODE), so it's imported unconditionally at the top of
        # this file -- no lazy-import/try-except needed here anymore.
        self._wind_battery_model = wind_battery_model if cfg.BATTERY_MODE == "simulated" else None

    @staticmethod
    def _resolve_genome_path(genome_path):
        """Resolves a relative genome_path against THIS FILE's own directory if it
        doesn't already exist relative to the process's current working directory.
        thymio_swarm_platform's ProjectManager never chdir()s into the active project
        directory before running an experiment (confirmed: manager.py just clones/pulls
        it, and daemon/server.py's own cwd print shows whatever directory the daemon
        process happened to be started from) -- so a bare filename is NOT guaranteed to
        resolve relative to os.getcwd() at runtime. An absolute path, or a path that
        already resolves from the real cwd, is used as-is.
        """
        if os.path.isabs(genome_path) or os.path.exists(genome_path):
            return genome_path
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  os.path.basename(genome_path))
        if os.path.exists(candidate):
            return candidate
        return genome_path  # let np.load() raise its own clear FileNotFoundError

    async def _tick(self):
        """One full sense -> decide -> act step. Factored out from run()'s loop so
        local_test_harness.py can drive it directly without needing an infinite loop or
        real hardware."""
        poses = await self.robot.get_all_global_poses()
        agents, self_index = poses_to_agents(poses, self.hostnames, self.self_hostname)
        current_position = (float(agents[self_index, 0]), float(agents[self_index, 1]))
        # pose_utils.py places an untracked robot at (1e4, 1e4) rather than raising -- a
        # real, expected state right after a session starts, before OptiTrack has locked
        # onto every rigid body. self_tracked guards the position-delta speed estimate
        # below against it: without this check, a sentinel position gets stored as
        # _prev_position, and the instant real tracking kicks in the next tick, the
        # ~14000m "delta" over one CONTROL_TICK_SECONDS computes as a ~28000 m/s "speed",
        # which floods straight into batterydrainage() and zeroes the battery in a single
        # tick -- a confirmed real failure mode (all robots driving for ~2s then stopping
        # simultaneously, no exception, just a quiet "battery depleted" print, because
        # OptiTrack takes about that long to lock onto all three robots after start).
        self_tracked = abs(current_position[0]) < cfg.UNTRACKED_XY_THRESHOLD

        if cfg.BATTERY_MODE == "simulated":
            if self._prev_position is not None and self_tracked:
                dt = cfg.CONTROL_TICK_SECONDS
                dx = current_position[0] - self._prev_position[0]
                dy = current_position[1] - self._prev_position[1]
                dist = math.hypot(dx, dy)
                speed = dist / dt
                travel_heading = 0.0 if dist < 1e-9 else math.atan2(dy, dx) - math.pi / 2.0
                angular_vel = self._last_w if self._last_w is not None else 0.0
                self.battery, _batt_drain, _wind_pct = self._wind_battery_model.compute_virtual_battery_update(
                    agents, self_index, self.battery, (speed, angular_vel, travel_heading), dt)
            agents[self_index, 3] = self.battery
            if self_tracked:
                self._prev_position = current_position
            # else: leave _prev_position at its last real value, so the delta computed
            # once tracking resumes is still measured from a real prior position instead
            # of silently skipping straight past the gap.

        sensor_inputs = get_sensor_data(agents)  # (10, n_agents)
        x_in = sensor_inputs[:, self_index].copy()
        if cfg.BATTERY_MODE == "none":
            x_in[8] = cfg.BATTERY_SENSOR_PLACEHOLDER

        v, w, self.w1, self.w2, self.w3 = hebbian_step(x_in, self.w1, self.w2, self.w3, self.rules)
        if self_tracked:
            v *= _corridor_speed_scale(current_position[1])
        left, right = velocity_to_motor_targets(v, w)
        await self.robot.drive(left, right)
        self._last_w = w

        if cfg.BATTERY_MODE == "simulated" and self.battery <= 0.0:
            print(f"[{self.self_hostname}] simulated battery depleted (<= 0) -- stopping, "
                  f"matching the simulation's own termination condition.")
            await self.stop()

        if self.logger:
            self.logger.log(
                state={"x": float(agents[self_index, 0]), "y": float(agents[self_index, 1]),
                       "heading": float(agents[self_index, 2]), "battery": float(self.battery)},
                command={"v": float(v), "w": float(w), "left": left, "right": right},
            )
        return v, w, left, right

    async def run(self):
        while self.running:
            if self.paused:
                await self.robot.stop()
                await asyncio.sleep(0.1)
                continue
            await self._tick()
            await asyncio.sleep(cfg.CONTROL_TICK_SECONDS)
        await self.robot.stop()

    async def pause(self):
        self.paused = True

    async def resume(self):
        self.paused = False

    async def stop(self):
        self.running = False
