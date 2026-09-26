"""Calibration helper -- NOT the real controller. Point-and-sample pose reader: place the
robot at a specific spot, step clear of the tracked volume so your own body isn't occluding
its markers, then trigger a reading from the controller terminal. Replaces an earlier
continuous "walk it around and watch the numbers scroll" version -- that approach requires
staying bent over the robot while it's tracked, which reliably occludes it from some camera
angles and not others, producing a frozen/stale-looking reading in whichever direction happens
to be blocked. A real code bug and this occlusion artifact look identical from the printed
numbers alone (both show "no change"), which is exactly what caused a false "POSITION_AXES
must be wrong" diagnosis before this rewrite -- sampling only while stationary and clear of
the robot removes the ambiguity entirely.

How to use, deployed via `hebbian_pose_calibration.py` (same launcher as before -- no changes
needed there beyond its printed instructions):
1. Deploy this experiment (config = {"hostnames": [...], "self_hostname": "..."}, no
   genome_path needed) -- see ../README.md.
2. Physically place the robot at the first point you want to measure (e.g. against one
   wall), then STEP AWAY from the tracked volume entirely.
3. In the controller terminal's `[p]ause  [r]esume  [s]top >` prompt, press `p` (or `r` --
   both do the same thing here, see below). This takes exactly one fresh pose reading right
   then and prints/logs it as a numbered sample.
4. Move the robot to the next point you want to measure (e.g. the opposite wall), step away
   again, and press `p`/`r` again for sample #2. Repeat for as many points as you need.
5. Press `s` to stop when done.

For the corridor walls specifically: sample at each wall (step 2-4 above), then set this
robot's CORRIDOR_Y_BOUNDS entry to (smaller, larger) of the two printed "sim frame y" values,
each with a little headroom inward (see controller_config.py's comment on
CORRIDOR_SLOWDOWN_MARGIN_M for how much margin makes sense for your setup). Do this per
robot -- different robots report different sim_y at the same wall.

Why `pause()` and `resume()` both trigger a sample rather than actually pausing/resuming
anything: this experiment doesn't drive the robot or run any continuous loop that needs
pausing -- `run()` just idles. Reusing the existing pause/resume session messages as the
"take a sample now" trigger means the already-existing `[p]/[r]/[s]` interactive prompt in
every controller-side launcher works as-is, with no changes needed to the daemon/coordinator
protocol or to hebbian_pose_calibration.py's actual logic (only its printed instructions were
updated to describe this new usage).

Also still useful for POSITION_AXES/HEADING_OFFSET_RAD calibration (the original purpose):
place the robot at a known position/heading, sample, and compare the printed raw position/
raw_yaw to what you expect -- just do it as discrete stationary samples now, not a continuous
walk.
"""
import asyncio
import os
import sys

# See hebbian_swarm_experiment.py's identical comment: the daemon only puts the
# project ROOT on sys.path, not this file's own directory, so the bare imports below
# need this once the file is deployed inside a subpackage.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import controller_config as cfg
from pose_utils import quaternion_to_yaw, poses_to_agents


class PrintPosesExperiment:
    def __init__(self, robot, config=None, logger=None):
        self.robot = robot
        self.config = config or {}
        self.logger = logger
        self.running = True
        self.hostnames = list(self.config.get("hostnames", []))
        self.self_hostname = self.config.get("self_hostname")
        self._sample_count = 0

    async def run(self):
        # All real work happens in _sample() below, triggered by pause()/resume() (i.e. the
        # controller terminal's 'p'/'r' keys) -- see module docstring for why. Idle otherwise,
        # deliberately not polling/printing continuously: this experiment is meant to be used
        # while standing clear of the robot, not while watching a live feed over its shoulder.
        while self.running:
            await asyncio.sleep(0.1)

    async def _sample(self, trigger):
        self._sample_count += 1
        n = self._sample_count
        poses = await self.robot.get_all_global_poses()
        own_pose = poses.get(self.self_hostname)

        if own_pose is None:
            print(f"[{self.self_hostname}] SAMPLE #{n} (via '{trigger}'): NOT TRACKED -- "
                  f"check the robot is inside the tracked volume and its markers aren't "
                  f"occluded, then retry.")
            return

        raw_yaw = quaternion_to_yaw(*own_pose.orientation)
        line = (f"[{self.self_hostname}] SAMPLE #{n} (via '{trigger}'): "
                f"raw position={own_pose.position} raw_yaw={raw_yaw:+.3f} rad "
                f"({raw_yaw * 180 / 3.14159:+.1f} deg)")

        sim_x = sim_y = sim_heading = None
        if self.hostnames:
            agents, self_index = poses_to_agents(poses, self.hostnames, self.self_hostname)
            sim_x, sim_y, sim_heading = (float(agents[self_index, 0]),
                                          float(agents[self_index, 1]),
                                          float(agents[self_index, 2]))
            applied_offset = cfg.HEADING_OFFSET_RAD.get(
                self.self_hostname, cfg.HEADING_OFFSET_RAD_DEFAULT)
            line += (f" | with POSITION_AXES={cfg.POSITION_AXES}, "
                     f"HEADING_OFFSET_RAD[{self.self_hostname}]={applied_offset} "
                     f"(dict has {list(cfg.HEADING_OFFSET_RAD.keys())}), "
                     f"ROTATION_SIGN={cfg.ROTATION_SIGN} -> sim frame "
                     f"x={sim_x:.3f} y={sim_y:.3f} heading={sim_heading:+.3f} rad")
            if sim_y is not None and abs(sim_y) < cfg.UNTRACKED_XY_THRESHOLD:
                line += (f" | for this robot's CORRIDOR_Y_BOUNDS entry: use this y value if this point is "
                         f"at (or just inside) a wall")

        print(line)
        if self.logger:
            self.logger.log(
                state={"sample": n, "trigger": trigger, "position": own_pose.position,
                       "raw_yaw": raw_yaw, "sim_x": sim_x, "sim_y": sim_y,
                       "sim_heading": sim_heading},
                command={})

    async def pause(self):
        await self._sample("p")

    async def resume(self):
        await self._sample("r")

    async def stop(self):
        self.running = False
