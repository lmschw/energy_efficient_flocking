"""Experiment class deploying a genome trained with optimize_hebbian.py
(energy_efficient_flocking/experiment/optimize_hebbian.py) onto real Thymio+Pi hardware
via thymio_swarm_platform / thymio_raspberry_swarm_control.

Matches that platform's de-facto experiment contract (there is no formal base class --
see README.md): __init__(robot, config, logger), async run()/pause()/resume()/stop().

To actually deploy: copy every *.py file in this directory except local_test_harness.py
(and diagnostics/, which are standalone tools, not part of the experiment) into
thymio_raspberry_swarm_control/experiments/hebbian_swarm/, register it in that repo's
swarm_project.yaml, and start a session with a config dict providing at least
genome_path, hostnames, and self_hostname (see README.md for the full walkthrough).
"""
import asyncio
import math

import numpy as np

import controller_config as cfg
from sensor_model import get_sensor_data
from hebbian_controller import init_weights, hebbian_step, unflatten_abcd
from pose_utils import poses_to_agents
from motor_utils import velocity_to_motor_targets


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

        self.rules = unflatten_abcd(np.load(self.config["genome_path"]))
        self.hostnames = list(self.config["hostnames"])
        self.self_hostname = self.config["self_hostname"]
        self.w1, self.w2, self.w3 = init_weights()

        # Simulated battery state (BATTERY_MODE == "simulated") -- see wind_battery_model.py.
        self.battery = cfg.INITIAL_BATTERY
        self._prev_position = None  # (x, y) as of the previous _tick(), for the
                                     # position-delta velocity estimate batterydrainage()
                                     # needs (no wheel encoders exist on this platform).
        self._last_w = None         # commanded angular velocity that was actually active
                                     # over the interval since _prev_position was recorded.
        self._wind_battery_model = None
        if cfg.BATTERY_MODE == "simulated":
            try:
                import wind_battery_model
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "controller_config.BATTERY_MODE == 'simulated' requires scipy "
                    "(wind_battery_model.py uses scipy.signal.convolve2d) -- "
                    "pip install scipy on this Pi, or set BATTERY_MODE = 'none'."
                ) from exc
            self._wind_battery_model = wind_battery_model

    async def _tick(self):
        """One full sense -> decide -> act step. Factored out from run()'s loop so
        local_test_harness.py can drive it directly without needing an infinite loop or
        real hardware."""
        poses = await self.robot.get_all_global_poses()
        agents, self_index = poses_to_agents(poses, self.hostnames, self.self_hostname)
        current_position = (float(agents[self_index, 0]), float(agents[self_index, 1]))

        if cfg.BATTERY_MODE == "simulated":
            if self._prev_position is not None:
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
            self._prev_position = current_position

        sensor_inputs = get_sensor_data(agents)  # (10, n_agents)
        x_in = sensor_inputs[:, self_index].copy()
        if cfg.BATTERY_MODE == "none":
            x_in[8] = cfg.BATTERY_SENSOR_PLACEHOLDER

        v, w, self.w1, self.w2, self.w3 = hebbian_step(x_in, self.w1, self.w2, self.w3, self.rules)
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
