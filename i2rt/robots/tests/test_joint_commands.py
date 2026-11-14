from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace
from typing import NoReturn
from unittest import TestCase

import numpy as np

from i2rt.robots.motor_chain_robot import JointCommands, MotorChainRobot


def make_robot() -> MotorChainRobot:
    robot = MotorChainRobot.__new__(MotorChainRobot)
    robot._command_lock = Lock()
    robot._joint_limits = None
    robot.motor_chain = [None, None]
    robot._kp = np.array([10.0, 20.0])
    robot._kd = np.array([1.0, 2.0])
    robot._commands = JointCommands.init_all_zero(2)
    robot._commands.pos[:] = [0.1, 0.2]
    robot._commands.kp[:] = robot._kp
    robot.remapper = SimpleNamespace(
        to_robot_joint_pos_space=lambda q: q.copy(),
        to_robot_joint_vel_space=lambda q: q.copy(),
    )
    return robot


def test_reader_never_observes_an_empty_command_during_update() -> None:
    robot = make_robot()
    entered, release = Event(), Event()
    previous = robot._commands

    def convert(q: np.ndarray) -> np.ndarray:
        entered.set()
        assert release.wait(5)
        return q.copy()

    robot.remapper.to_robot_joint_pos_space = convert
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(robot.command_joint_state, {"pos": np.array([0.3, 0.4]), "vel": np.array([0.5, 0.6])})
        try:
            assert entered.wait(5)
            with robot._command_lock:
                assert robot._commands is previous
                np.testing.assert_array_equal(robot._commands.kp, [10, 20])
        finally:
            release.set()
        future.result()
    np.testing.assert_array_equal(robot._commands.pos, [0.3, 0.4])
    np.testing.assert_array_equal(robot._commands.vel, [0.5, 0.6])
    np.testing.assert_array_equal(robot._commands.kp, [10, 20])
    np.testing.assert_array_equal(robot._commands.kd, [1, 2])


def test_conversion_failure_preserves_last_command() -> None:
    robot = make_robot()
    previous = robot._commands
    cause = ValueError("invalid velocity")

    def fail(q: np.ndarray) -> NoReturn:
        raise cause

    robot.remapper.to_robot_joint_vel_space = fail
    with TestCase().assertRaises(ValueError) as caught:
        robot.command_joint_state({"pos": np.array([0.3, 0.4]), "vel": np.array([0.5, 0.6])})
    assert caught.exception is cause
    assert robot._commands is previous
