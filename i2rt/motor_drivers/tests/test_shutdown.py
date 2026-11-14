from threading import Event, Thread
from types import SimpleNamespace
from unittest import TestCase

from i2rt.motor_drivers.dm_driver import DMChainCanInterface


def test_close_waits_for_can_control_loop_before_closing_socket() -> None:
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.running = True
    started, stopped = Event(), Event()
    closed = []

    def control() -> None:
        started.set()
        while chain.running:
            stopped.wait(0.001)
        stopped.set()

    def close_socket() -> None:
        assert stopped.is_set()
        closed.append(True)

    chain.motor_interface = SimpleNamespace(close=close_socket)
    chain._thread = Thread(target=control)
    chain._thread.start()
    assert started.wait(5)
    try:
        chain.close()
    finally:
        chain.running = False
        chain._thread.join(timeout=5)
    assert closed == [True]


def test_stuck_control_loop_does_not_close_its_socket() -> None:
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.running = True
    closed = []
    chain.motor_interface = SimpleNamespace(close=lambda: closed.append(True))
    chain._thread = SimpleNamespace(join=lambda timeout: None, is_alive=lambda: True)
    with TestCase().assertRaises(TimeoutError):
        chain.close()
    assert not closed
