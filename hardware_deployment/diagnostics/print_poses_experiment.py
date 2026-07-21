"""Calibration helper -- NOT the real controller. An experiment (same contract as
HebbianSwarmExperiment) that just prints/logs each robot's raw OptiTrack pose and
derived yaw every tick, so you can determine controller_config.py's POSITION_AXES and
HEADING_OFFSET_RAD before trusting the real controller.

How to calibrate:
1. Deploy this experiment the same way you would the real one (see ../README.md), with
   config = {"hostnames": [...], "self_hostname": "..."} (no genome_path needed).
2. Physically point the robot in the direction this codebase's simulation calls
   heading=0 -- "facing +y" in whichever 2D plane you decide POSITION_AXES selects. If
   you don't have an independent reference for that, an easier equivalent calibration:
   point the robot in whatever direction you want to DEFINE as heading=0 for your
   experiments, note the raw yaw printed here, and set HEADING_OFFSET_RAD to minus that
   value -- the simulation's heading=0 is just a convention, what matters is that your
   real robots' heading=0 all agrees with each other and with "the goal direction" you
   want them walking toward (recall: the trained controller always walks toward -x in
   its own frame, so whichever physical direction you calibrate as heading=0 is the
   direction the swarm will try to migrate away from).
3. Try both POSITION_AXES = (0, 1) and (0, 2) (and (1, 2) if neither looks right) --
   whichever pair produces (x, y) values that visibly change the way you'd expect as you
   physically move the robot around your tracked volume is the correct one for your
   Motive calibration.
4. If the robot turns the wrong way once you deploy the real controller (spins away
   from, rather than toward, where the sensed neighbors/geometry should steer it),
   flip ROTATION_SIGN and re-test -- this script only helps calibrate position axes and
   the zero-heading offset, not rotation sign (that's easiest to just observe directly
   from the real controller's behavior).
"""
import asyncio

import controller_config as cfg
from pose_utils import quaternion_to_yaw, poses_to_agents


class PrintPosesExperiment:
    def __init__(self, robot, config=None, logger=None):
        self.robot = robot
        self.config = config or {}
        self.logger = logger
        self.running = True
        self.paused = False
        self.hostnames = list(self.config.get("hostnames", []))
        self.self_hostname = self.config.get("self_hostname")

    async def run(self):
        while self.running:
            if self.paused:
                await asyncio.sleep(0.1)
                continue

            poses = await self.robot.get_all_global_poses()
            own_pose = poses.get(self.self_hostname)
            if own_pose is None:
                print(f"[{self.self_hostname}] not currently tracked (outside volume, "
                      f"or tracking hasn't started yet)")
            else:
                raw_yaw = quaternion_to_yaw(*own_pose.orientation)
                line = (f"[{self.self_hostname}] raw position={own_pose.position} "
                        f"raw_yaw={raw_yaw:+.3f} rad ({raw_yaw * 180 / 3.14159:+.1f} deg)")
                if self.hostnames:
                    agents, self_index = poses_to_agents(poses, self.hostnames, self.self_hostname)
                    x, y, heading = agents[self_index, 0], agents[self_index, 1], agents[self_index, 2]
                    line += (f" | with POSITION_AXES={cfg.POSITION_AXES}, "
                             f"HEADING_OFFSET_RAD={cfg.HEADING_OFFSET_RAD}, "
                             f"ROTATION_SIGN={cfg.ROTATION_SIGN} -> sim frame "
                             f"x={x:.3f} y={y:.3f} heading={heading:+.3f} rad")
                print(line)
                if self.logger:
                    self.logger.log(state={"raw_yaw": raw_yaw, "position": own_pose.position}, command={})

            await asyncio.sleep(0.5)  # matches OptiTrack's own push rate -- no point polling faster

    async def pause(self):
        self.paused = True

    async def resume(self):
        self.paused = False

    async def stop(self):
        self.running = False
