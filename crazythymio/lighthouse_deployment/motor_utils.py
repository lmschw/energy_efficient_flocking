"""Converts the Hebbian controller's (v [m/s], w [rad/s]) output into raw Thymio
drive(left, right) motor-target integers (thymio_swarm_platform.robot.Robot.drive()'s
expected input), via standard differential-drive kinematics.

Robot.drive() applies NO clamping itself (confirmed: thymio_swarm_platform's
RobotConfig.max_motor is declared but never enforced in code) -- clamping here is the
only safety net against sending the firmware an out-of-range target.
"""
import controller_config as cfg


def velocity_to_motor_targets(v, w):
    """v: forward speed [m/s]. w: angular rate [rad/s] (positive = the simulation's
    convention for increasing heading; flip controller_config.ROTATION_SIGN if the
    robot turns the wrong way on hardware). Returns (left, right) as ints, clamped to
    [-MAX_MOTOR_TARGET, MAX_MOTOR_TARGET]."""
    half_track = cfg.WHEEL_DISTANCE_M / 2.0
    v_left = v - w * half_track
    v_right = v + w * half_track

    left = int(round(v_left * cfg.MOTOR_UNITS_PER_MPS))
    right = int(round(v_right * cfg.MOTOR_UNITS_PER_MPS))

    left = max(-cfg.MAX_MOTOR_TARGET, min(cfg.MAX_MOTOR_TARGET, left))
    right = max(-cfg.MAX_MOTOR_TARGET, min(cfg.MAX_MOTOR_TARGET, right))
    return left, right
