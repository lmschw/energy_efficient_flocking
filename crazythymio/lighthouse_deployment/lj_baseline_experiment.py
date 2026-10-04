"""Deploys the paper's Table 3 "standard collective motion baseline" (LJ spacing +
heading alignment + goal-pull, Fig. 5a's cluster-4 comparison point) onto real
Thymio+Pi hardware, via thymio_swarm_platform. Sibling to hebbian_swarm_experiment.py's
HebbianSwarmExperiment -- same experiment contract (__init__(robot, config, logger),
async run()/pause()/resume()/stop()), same OptiTrack pose pipeline
(pose_utils.poses_to_agents()), and the SAME deployment-only safety reflexes (corridor/
agent-safety clamps, IR backoff, untracked-speed cap, imported from
hebbian_swarm_experiment.py rather than duplicated) -- only the "decide" step differs:
_lj_velocity_command() below is a fixed, stateless control law (no genome, no per-tick
weight update), ported from ants26_replication/experiment/lj_baseline.py's
_flocking_velocity_command() to run per-tick from live poses instead of inside a batch
simulation loop. Rule gains come from LJ_* in controller_config.py, copied verbatim
from hardware_transfer_test/final/lj_baseline/paper_baseline_rules.json -- the exact
baseline evaluated in the paper, not re-tuned here.
"""
import asyncio
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import controller_config as cfg
import wind_battery_model
from pose_utils import poses_to_agents
from sensor_model import get_sensor_data
from motor_utils import velocity_to_motor_targets
from hebbian_swarm_experiment import (
    _corridor_speed_scale,
    _corridor_scale,
    _agent_safety_speed_scale,
    _apply_obstacle_backoff,
    _apply_ir_backoff,
)


def _lj_velocity_command(agents, self_index):
    """Verbatim port of ants26_replication/experiment/lj_baseline.py's
    _flocking_velocity_command(), fully vectorized over every agent (the LJ spacing and
    heading-alignment terms are pairwise, so -- like sensor_model.get_sensor_data() for
    the Hebbian controller -- the full agents array is needed even though only this
    robot's own (v, w) is used). Returns (v, w) for agents[self_index]."""
    n_agents = agents.shape[0]
    r0, epsilon = cfg.LJ_R0, cfg.LJ_EPSILON
    k_align, k_goal = cfg.LJ_K_ALIGN, cfg.LJ_K_GOAL
    K1, K2, U = cfg.LJ_K1, cfg.LJ_K2, cfg.LJ_U
    r_cut, r_min, R_align = cfg.LJ_R_CUT, cfg.LJ_R_MIN, cfg.LJ_R_ALIGN

    sigma = r0 / np.sqrt(2.0)
    X = agents[:, 0:2]
    th = agents[:, 2] + np.pi / 2.0

    Dx = np.subtract.outer(X[:, 0], X[:, 0])
    Dy = np.subtract.outer(X[:, 1], X[:, 1])
    R2 = Dx**2 + Dy**2 + np.eye(n_agents)
    R = np.sqrt(R2)
    # UNLIKE the sim this was ported from, two DIFFERENT (off-diagonal) agents can end up
    # at the exact same point here: pose_utils.poses_to_agents() places EVERY currently-
    # untracked robot at the identical (1e4, 1e4) sentinel, and losing tracking on 2+
    # robots at once is routine on this rig (simultaneous marker occlusion when robots
    # cluster -- confirmed from real trial data). The +eye(n_agents) above only guards
    # the diagonal (self-distance); an off-diagonal R of exactly 0 divides-by-zero into
    # NaN below, and since NaN * 0 is still NaN, the (R > r_min) mask further down does
    # NOT clean this up -- confirmed by a real crash in local_test_harness-style testing
    # the very first time two agents shared a position. Clamping the denominator here
    # (not the mask) is what actually prevents it.
    R = np.maximum(R, 1e-6)

    ex, ey = Dx / R, Dy / R
    cos_th_mat = np.tile(np.cos(th)[:, None], (1, n_agents))
    sin_th_mat = np.tile(np.sin(th)[:, None], (1, n_agents))

    exr = ex * cos_th_mat + ey * sin_th_mat
    eyr = ey * cos_th_mat - ex * sin_th_mat

    mask = (R > r_min) & (R < r_cut)
    np.fill_diagonal(mask, False)

    sig_over_r6 = (sigma**2) / (R**2 + 1e-6)
    sig_over_r12 = sig_over_r6**2
    Fmag = 8.0 * epsilon * (2.0 * sig_over_r12 - sig_over_r6) / (R + 1e-6) * mask

    Fp_x = np.sum(Fmag * exr, axis=1)
    Fp_y = np.sum(Fmag * eyr, axis=1)

    align_mask = (R < R_align) & (~np.eye(n_agents, dtype=bool))
    Th1 = np.tile(th, (n_agents, 1))
    Th2 = np.tile(th[:, None], (1, n_agents))

    H_x = align_mask * np.cos(Th1)
    H_y = align_mask * np.sin(Th1)
    Hb_x = H_x * np.cos(Th2) + H_y * np.sin(Th2)
    Hb_y = H_y * np.cos(Th2) - H_x * np.sin(Th2)

    Fa_x = np.sum(Hb_x, axis=1)
    Fa_y = np.sum(Hb_y, axis=1)
    A = np.maximum(np.sqrt(Fa_x**2 + Fa_y**2), 1e-9)
    Fa_x = Fa_x / A
    Fa_y = Fa_y / A

    # Goal direction is a constant pull in -x (sim-frame "upwind"), same convention as
    # every Hebbian genome in this repo -- see simulation_hebbian.py's dist_travelled.
    Fg_gx = -1.0
    Fg_x = Fg_gx * np.cos(th)
    Fg_y = Fg_gx * -np.sin(th)

    F_x = Fp_x + k_align * Fa_x + k_goal * Fg_x
    F_y = Fp_y + k_align * Fa_y + k_goal * Fg_y

    v = float(np.clip(K1 * F_x[self_index] + U, -cfg.LINEAR_VEL_MAX, cfg.LINEAR_VEL_MAX))
    w = float(np.clip(K2 * F_y[self_index], -cfg.ANGULAR_VEL_MAX, cfg.ANGULAR_VEL_MAX))
    return v, w


class LJBaselineExperiment:
    # NOTE: the parameter must be named exactly `config` -- see HebbianSwarmExperiment's
    # identical note for why.
    def __init__(self, robot, config=None, logger=None):
        self.robot = robot
        self.config = config or {}
        self.logger = logger
        self.running = True
        self.paused = False
        self._tick_count = 0

        if "hostnames" not in self.config or "self_hostname" not in self.config:
            raise ValueError("config['hostnames'] (the full participating swarm, same "
                              "list/order on every robot) and config['self_hostname'] "
                              "are both required.")
        self.hostnames = list(self.config["hostnames"])
        self.self_hostname = self.config["self_hostname"]

        # Simulated battery state -- purely descriptive telemetry here (matching
        # simulate_lj_baseline()'s own recording): the LJ control law above never reads
        # battery, unlike the Hebbian controller, so this cannot affect its behaviour.
        self.battery = cfg.INITIAL_BATTERY
        self._prev_position = None
        self._last_w = None
        self._wind_battery_model = wind_battery_model if cfg.BATTERY_MODE == "simulated" else None

        # OBSTACLE_BACKOFF_* state -- see hebbian_swarm_experiment.py's
        # _apply_obstacle_backoff() for what these mean; reused unchanged here.
        self._front_close_streak = 0
        self._back_close_streak = 0
        self._backoff_ticks_remaining = 0
        self._backoff_v = 0.0

    async def _tick(self):
        self._tick_count += 1
        scale_corridor = scale_agent = 1.0
        poses = await self.robot.get_all_global_poses()
        agents, self_index = poses_to_agents(poses, self.hostnames, self.self_hostname)
        current_position = (float(agents[self_index, 0]), float(agents[self_index, 1]))
        self_pose = poses.get(self.self_hostname)
        if self_pose is not None:
            raw_position = tuple(float(c) for c in self_pose.position)
            raw_orientation = tuple(float(c) for c in self_pose.orientation)
        else:
            raw_position = (None, None, None)
            raw_orientation = (None, None, None, None)
        self_tracked = abs(current_position[0]) < cfg.UNTRACKED_XY_THRESHOLD

        if cfg.BATTERY_MODE == "simulated":
            if self._prev_position is not None and self_tracked:
                dt = cfg.CONTROL_TICK_SECONDS
                dx = current_position[0] - self._prev_position[0]
                dy = current_position[1] - self._prev_position[1]
                dist = math.hypot(dx, dy)
                speed = dist / dt
                if speed > cfg.MAX_PLAUSIBLE_SPEED_MPS:
                    print(f"[{self.self_hostname}] implausible {speed:.2f} m/s ({dist:.2f} m in "
                          f"one tick) -- treating as a manual relocation, no battery drain.")
                else:
                    travel_heading = 0.0 if dist < 1e-9 else math.atan2(dy, dx) - math.pi / 2.0
                    angular_vel = self._last_w if self._last_w is not None else 0.0
                    self.battery, _batt_drain, _wind_pct = self._wind_battery_model.compute_virtual_battery_update(
                        agents, self_index, self.battery, (speed, angular_vel, travel_heading), dt)
            agents[self_index, 3] = self.battery
            self._prev_position = current_position if self_tracked else None

        v, w = _lj_velocity_command(agents, self_index)
        v_policy = float(v)    # logged: LJ law output before any deployment-side reflex/governor

        # Only used here, to feed the same OptiTrack-derived obstacle-backoff reflex the
        # Hebbian deployment uses -- the LJ control law itself never sees these.
        sensor_inputs = get_sensor_data(agents)
        front_d, back_d = float(sensor_inputs[0, self_index]), float(sensor_inputs[2, self_index])
        v = _apply_obstacle_backoff(self, v, front_d, back_d)

        if cfg.SAFETY_LAYERS_ENABLED:
            if self_tracked:
                scale_corridor = _corridor_scale(agents, self_index, v)
                scale_agent = _agent_safety_speed_scale(agents, self_index)
                v *= min(scale_corridor, scale_agent)
            else:
                v = max(-cfg.UNTRACKED_SAFE_V_CAP, min(cfg.UNTRACKED_SAFE_V_CAP, v))

        ir = await self.robot.proximity_horizontal()
        v = _apply_ir_backoff(v, ir)

        left, right = velocity_to_motor_targets(v, w)
        await self.robot.drive(left, right)
        self._last_w = w

        if cfg.BATTERY_MODE == "simulated" and self.battery <= 0.0:
            print(f"[{self.self_hostname}] simulated battery depleted (<= 0) -- stopping.")
            await self.stop()

        if self.logger:
            self.logger.log(
                state={"tick": self._tick_count, "timestamp": time.time(),
                       "x": float(agents[self_index, 0]), "y": float(agents[self_index, 1]),
                       "heading": float(agents[self_index, 2]), "battery": float(self.battery),
                       "raw_x": raw_position[0], "raw_y": raw_position[1], "raw_z": raw_position[2],
                       "qx": raw_orientation[0], "qy": raw_orientation[1],
                       "qz": raw_orientation[2], "qw": raw_orientation[3],
                       "tracked": int(self_tracked),
                       "n_neighbors_seen": sum(1 for h in self.hostnames
                                               if h != self.self_hostname and poses.get(h) is not None),
                       "front_d": front_d, "back_d": back_d,
                       "ir_front_max": max(ir[0:5]), "ir_rear_max": max(ir[5:7])},
                command={"v": float(v), "w": float(w), "left": left, "right": right,
                         "v_policy": v_policy, "scale_corridor": float(scale_corridor),
                         "scale_agent": float(scale_agent)},
            )
        return v, w, left, right

    async def run(self):
        # FIXED-RATE loop: one tick every CONTROL_TICK_SECONDS measured from tick START to tick START, like the simulation's
        # dt. (Sleeping CONTROL_TICK_SECONDS AFTER each tick made the real period ~0.6 s: every command was held 20% longer
        # than in simulation and the battery model, which divides by CONTROL_TICK_SECONDS, overestimated speed by 20%.)
        next_t = time.monotonic()
        while self.running:
            if self.paused:
                await self.robot.stop()
                await asyncio.sleep(0.1)
                next_t = time.monotonic()
                continue
            await self._tick()
            next_t += cfg.CONTROL_TICK_SECONDS
            now = time.monotonic()
            if next_t < now:                 # a tick overran: don't try to catch up with a burst of ticks
                next_t = now
            await asyncio.sleep(next_t - now)
        await self.robot.stop()

    async def pause(self):
        self.paused = True

    async def resume(self):
        self.paused = False

    async def stop(self):
        self.running = False
