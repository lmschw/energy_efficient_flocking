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
    if cfg.PRESERVE_TURN_ON_SATURATION:
        # A Thymio wheel tops out at MAX_MOTOR_TARGET (~0.17 m/s), so a forward speed v and a turn rate w can only be
        # commanded together while |v| + |w|*half_track fits. The simulation lets v (0.2 m/s) and w (0.63 rad/s) reach
        # their limits independently; clipping each wheel separately then throws away the wheel-speed DIFFERENCE, i.e. the
        # turn (measured: the clipped wheel speeds encoded only ~50% of the commanded turn; a wheel was saturated in 52% of
        # all ticks of the first 7-robot LJ trial). Give the turn priority and reduce the forward speed to fit instead.
        room = cfg.MAX_MOTOR_TARGET / cfg.MOTOR_UNITS_PER_MPS - abs(w) * half_track
        v = max(-room, min(room, v))
    v_left = v - w * half_track
    v_right = v + w * half_track

    left = int(round(v_left * cfg.MOTOR_UNITS_PER_MPS))
    right = int(round(v_right * cfg.MOTOR_UNITS_PER_MPS))

    left = max(-cfg.MAX_MOTOR_TARGET, min(cfg.MAX_MOTOR_TARGET, left))
    right = max(-cfg.MAX_MOTOR_TARGET, min(cfg.MAX_MOTOR_TARGET, right))
    return left, right
