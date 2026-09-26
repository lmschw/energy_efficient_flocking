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
import time

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


def _apply_obstacle_backoff(state, v, front_d, back_d):
    """Deployment-only reflex -- see OBSTACLE_BACKOFF_* in controller_config.py for the
    full rationale (direct answer to AGENT_SAFETY_CLAMP_INNER_GAP's deadlock history and
    to OptiTrack tracking degrading when robots cluster tightly). `state` is the owning
    HebbianSwarmExperiment instance -- reads/writes its _front_close_streak,
    _back_close_streak, _backoff_ticks_remaining, _backoff_v. Returns the (possibly
    overridden) v; w is never touched."""
    if not cfg.OBSTACLE_BACKOFF_ENABLED:
        return v

    state._front_close_streak = (state._front_close_streak + 1
                                  if front_d <= cfg.OBSTACLE_TRIGGER_DIST else 0)
    state._back_close_streak = (state._back_close_streak + 1
                                 if back_d <= cfg.OBSTACLE_TRIGGER_DIST else 0)

    if state._backoff_ticks_remaining > 0:
        state._backoff_ticks_remaining -= 1
        return state._backoff_v

    if state._front_close_streak >= cfg.OBSTACLE_TRIGGER_TICKS:
        state._backoff_v = -cfg.OBSTACLE_BACKOFF_SPEED
        state._backoff_ticks_remaining = cfg.OBSTACLE_BACKOFF_TICKS - 1
        state._front_close_streak = 0
        return state._backoff_v
    if state._back_close_streak >= cfg.OBSTACLE_TRIGGER_TICKS:
        state._backoff_v = cfg.OBSTACLE_BACKOFF_SPEED
        state._backoff_ticks_remaining = cfg.OBSTACLE_BACKOFF_TICKS - 1
        state._back_close_streak = 0
        return state._backoff_v
    return v


def _agent_safety_speed_scale(agents, self_index):
    """Deployment-ENFORCED port of simulation_hebbian.py's _apply_safety_clamp() (agent-agent
    half only -- the wall half is _corridor_speed_scale() above, a separately-calibrated
    mechanism for the real corridor). See controller_config.py's "Inter-agent safety clamp"
    section for why this MUST be enforced here, not just trained in: plain_seed123_clamped's
    weights were shaped by this clamp being active during training, but the clamp itself is
    not part of what the network learned -- without this function, deploying that genome
    would provide no more actual collision safety than the unclamped one.

    Returns a [0, 1] multiplier for v, scaling from 1.0 (gap >= AGENT_SAFETY_CLAMP_OUTER_GAP)
    linearly down to 0.0 (gap <= AGENT_SAFETY_CLAMP_INNER_GAP, i.e. contact), using the same
    formula and thresholds simulation_hebbian.py trained this genome under. `agents` is the
    full (n_agents, 4) array from poses_to_agents() -- untracked robots sit at pose_utils.py's
    (1e4, 1e4) sentinel, which yields a huge gap here (no braking effect), the same
    "can't protect against what we can't see" default every other neighbor-facing computation
    in this package already accepts (see sensor_model.py)."""
    agents_xy = agents[:, 0:2]
    n_agents = agents_xy.shape[0]
    if n_agents < 2:
        return 1.0
    deltas = agents_xy[self_index] - agents_xy
    dists = np.linalg.norm(deltas, axis=1)
    dists[self_index] = np.inf
    nearest_gap = dists.min() - 2.0 * cfg.ROBOT_RAD
    outer, inner = cfg.AGENT_SAFETY_CLAMP_OUTER_GAP, cfg.AGENT_SAFETY_CLAMP_INNER_GAP
    return float(np.clip((nearest_gap - inner) / (outer - inner), 0.0, 1.0))


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
        self._tick_count = 0  # incremented once per _tick() call, logged alongside a
                               # wall-clock timestamp so post-hoc analysis (leadership
                               # metrics, collision/wall proximity vs. time) can align
                               # different robots' logs precisely instead of assuming a
                               # clean, gap-free CONTROL_TICK_SECONDS cadence per robot.

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
        # OBSTACLE_BACKOFF_* state (see controller_config.py and _apply_obstacle_backoff()
        # above) -- persistence-tick counters and the active reflex's remaining duration/v.
        self._front_close_streak = 0
        self._back_close_streak = 0
        self._backoff_ticks_remaining = 0
        self._backoff_v = 0.0
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
        self._tick_count += 1
        poses = await self.robot.get_all_global_poses()
        agents, self_index = poses_to_agents(poses, self.hostnames, self.self_hostname)
        current_position = (float(agents[self_index, 0]), float(agents[self_index, 1]))
        # Raw pose (untranslated, unrotated OptiTrack reading) for THIS robot, logged
        # alongside the derived x/y/heading below -- poses_to_agents() only keeps the
        # POSITION_AXES-selected 2D ground-plane projection plus a yaw angle, discarding
        # the third (height, here Y since this rig is Y-up -- see POSITION_AXES's comment)
        # position component and the full orientation quaternion entirely. Logged raw so
        # nothing is thrown away that a later analysis might want (tilt/roll from the
        # quaternion, height drift, an independent recomputation of heading, etc.) --
        # None (empty in the CSV) if this robot isn't currently tracked, matching
        # poses_to_agents()'s own `poses.get(host)` handling of the same case.
        self_pose = poses.get(self.self_hostname)
        if self_pose is not None:
            raw_position = tuple(float(c) for c in self_pose.position)
            raw_orientation = tuple(float(c) for c in self_pose.orientation)
        else:
            raw_position = (None, None, None)
            raw_orientation = (None, None, None, None)
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
                if speed > cfg.MAX_PLAUSIBLE_SPEED_MPS:
                    # Not driven -- a hand relocation (see MAX_PLAUSIBLE_SPEED_MPS). Skip
                    # this tick's drain and re-baseline from the new position below.
                    print(f"[{self.self_hostname}] implausible {speed:.2f} m/s ({dist:.2f} m in "
                          f"one tick) -- treating as a manual relocation, no battery drain.")
                else:
                    travel_heading = 0.0 if dist < 1e-9 else math.atan2(dy, dx) - math.pi / 2.0
                    angular_vel = self._last_w if self._last_w is not None else 0.0
                    self.battery, _batt_drain, _wind_pct = self._wind_battery_model.compute_virtual_battery_update(
                        agents, self_index, self.battery, (speed, angular_vel, travel_heading), dt)
            agents[self_index, 3] = self.battery
            # Only a position from the IMMEDIATELY previous tick is a valid baseline: after
            # any tracking gap (including a stale-feed rejection), the next tracked tick
            # re-baselines instead of charging the whole gap's displacement as one tick of
            # driving -- that is what drained thymio-08's battery to -39% on 2026-09-26
            # (4.26 m "in 0.5 s" after being carried back to the middle) and stopped it.
            self._prev_position = current_position if self_tracked else None

        sensor_inputs = get_sensor_data(agents)  # (10, n_agents)
        x_in = sensor_inputs[:, self_index].copy()
        if cfg.BATTERY_MODE == "none":
            x_in[8] = cfg.BATTERY_SENSOR_PLACEHOLDER

        # TEMPORARY DEBUG (2026-09-20) -- diagnosing "robot behaves the same regardless
        # of real inter-robot distance" by checking what the network actually receives,
        # instead of inferring it from code review. front/back/right/left_d are
        # sensor_model.py's normalized quadrant distances: +1.0 means "no neighbor found
        # in that quadrant within SENSING_RADIUS" (either truly out of range, or agents
        # array effectively has this robot alone in it); anything less than +1.0 means a
        # real neighbor was sensed there. n_agents_seen is agents.shape[0] as a sanity
        # check that the array itself has all 3 rows. Remove once diagnosed.
        _debug_n_agents = int(agents.shape[0])
        _debug_front_d, _debug_back_d = float(x_in[0]), float(x_in[2])
        _debug_right_d, _debug_left_d = float(x_in[4]), float(x_in[6])

        v, w, self.w1, self.w2, self.w3 = hebbian_step(x_in, self.w1, self.w2, self.w3, self.rules)
        v = _apply_obstacle_backoff(self, v, _debug_front_d, _debug_back_d)
        if self_tracked:
            # min(), not product -- matches simulation_hebbian.py's own
            # vel[:,0] *= np.minimum(agent_scale, wall_scale) combination exactly: whichever
            # constraint (nearest neighbor or nearest wall) is more restrictive wins, rather
            # than compounding both into an even smaller scale.
            v *= min(_corridor_speed_scale(current_position[1]),
                     _agent_safety_speed_scale(agents, self_index))
        else:
            # CANNOT compute _agent_safety_speed_scale/_corridor_speed_scale without a real
            # position -- self is at pose_utils.py's (1e4,1e4) sentinel, which also makes
            # every OTHER agent read as "far away" to THIS robot's own sensing (see
            # sensor_model.py), so hebbian_step tends to output an open-field v here, not a
            # braked one. Un-tracked used to mean UNCLAMPED (full v straight to the
            # motors) -- confirmed from real trial data as a real collision mechanism:
            # tracking degrades/drops exactly when robots cluster tightly (marker
            # occlusion -- see pose_utils.py's stale-freeze/up-axis-outlier detectors), so
            # the moment we can least verify it's safe to keep going is the same moment
            # this branch used to apply NO speed limit at all. Not knowing where we are is
            # itself a reason to slow down, not a reason to skip the clamp -- cap to a
            # slow crawl instead (direction/w still untouched, same as every other
            # deployment-only reflex here).
            v = max(-cfg.UNTRACKED_SAFE_V_CAP, min(cfg.UNTRACKED_SAFE_V_CAP, v))
        left, right = velocity_to_motor_targets(v, w)
        await self.robot.drive(left, right)
        self._last_w = w

        if cfg.BATTERY_MODE == "simulated" and self.battery <= 0.0:
            print(f"[{self.self_hostname}] simulated battery depleted (<= 0) -- stopping, "
                  f"matching the simulation's own termination condition.")
            await self.stop()

        if self.logger:
            self.logger.log(
                state={"tick": self._tick_count, "timestamp": time.time(),
                       "x": float(agents[self_index, 0]), "y": float(agents[self_index, 1]),
                       "heading": float(agents[self_index, 2]), "battery": float(self.battery),
                       "raw_x": raw_position[0], "raw_y": raw_position[1], "raw_z": raw_position[2],
                       "qx": raw_orientation[0], "qy": raw_orientation[1],
                       "qz": raw_orientation[2], "qw": raw_orientation[3],
                       "n_agents_seen": _debug_n_agents, "front_d": _debug_front_d,
                       "back_d": _debug_back_d, "right_d": _debug_right_d,
                       "left_d": _debug_left_d},
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
