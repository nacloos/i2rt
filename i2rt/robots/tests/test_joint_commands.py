from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace
from typing import NoReturn
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from i2rt.robots.motor_chain_robot import JointCommands, MotorChainRobot


def make_robot() -> MotorChainRobot:
    robot = MotorChainRobot.__new__(MotorChainRobot)
    robot._command_lock = Lock()
    robot._state_lock = Lock()
    robot._joint_limits = None
    robot.motor_chain = [None, None]
    robot._kp = np.array([10.0, 20.0])
    robot._kd = np.array([1.0, 2.0])
    robot._commands = JointCommands.init_all_zero(2)
    robot._commands.pos[:] = [0.1, 0.2]
    robot._commands.kp[:] = robot._kp
    robot._integral_torque = np.zeros(2)
    robot._joint_state = SimpleNamespace(pos=np.zeros(2))
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


def test_integral_is_bounded_unwinds_and_clears_when_position_control_stops() -> None:
    robot = make_robot()
    with patch("i2rt.robots.motor_chain_robot.time.monotonic", return_value=0):
        robot.configure_integral(np.array([2.0, 0.0]), np.array([0.3, 0.0]))
    commands = JointCommands.init_all_zero(2)
    commands.kp[:] = 10
    commands.pos[:] = 1
    for tick in range(1, 101):
        with patch("i2rt.robots.motor_chain_robot.time.monotonic", return_value=tick * 0.01):
            torque = robot._integral_compensation(commands)
        assert np.all(np.abs(torque) <= [0.3, 0.0])
    np.testing.assert_allclose(torque, [0.3, 0.0])
    commands.pos[:] = -1
    with patch("i2rt.robots.motor_chain_robot.time.monotonic", return_value=1.01):
        np.testing.assert_allclose(robot._integral_compensation(commands), [0.28, 0.0])
    commands.kp[:] = 0
    with patch("i2rt.robots.motor_chain_robot.time.monotonic", return_value=1.02):
        np.testing.assert_array_equal(robot._integral_compensation(commands), 0)


def test_integral_does_not_accumulate_across_stalls() -> None:
    robot = make_robot()
    with patch("i2rt.robots.motor_chain_robot.time.monotonic", return_value=0):
        robot.configure_integral(np.ones(2), np.ones(2))
    commands = JointCommands.init_all_zero(2)
    commands.kp[:] = 10
    commands.pos[:] = 1
    with patch("i2rt.robots.motor_chain_robot.time.monotonic", return_value=10):
        np.testing.assert_array_equal(robot._integral_compensation(commands), 0)


def test_invalid_integral_configuration_preserves_state() -> None:
    robot = make_robot()
    robot.configure_integral(np.ones(2), np.ones(2))
    for gains in (np.array([-1, 0]), np.array([np.nan, 0]), np.zeros(1)):
        with TestCase().assertRaises(ValueError):
            robot.configure_integral(gains, np.ones(2))
        np.testing.assert_array_equal(robot._ki, 1)
