"""Borrowed CAN transports preserve startup validation and connection ownership."""

from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import can
import numpy as np
import pytest
from can.interfaces.virtual import VirtualBus

from i2rt.motor_config_tool.dm_motor_registers import CMD_READ, REG_BY_ADDR, DMRegAddr, _value_to_bytes
from i2rt.motor_config_tool.utils import RawCanInterface
from i2rt.motor_drivers import dm_driver, motor_check
from i2rt.motor_drivers.can_interface import CanInterface
from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface, PassiveEncoderReader
from i2rt.motor_drivers.utils import MotorType
from i2rt.robots.utils import ArmType
from i2rt.utils import encoder_manager


@pytest.fixture
def buses(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[VirtualBus, VirtualBus]]:
    channel = str(uuid4())
    with VirtualBus(channel) as bus, VirtualBus(channel) as peer:

        def forbid_ambient_bus(*_args: Any, **_kwargs: Any) -> None:
            pytest.fail("An injected transport must not open another CAN connection")

        monkeypatch.setattr(can.interface, "Bus", forbid_ambient_bus)
        yield bus, peer


@pytest.mark.parametrize("wrapper", [CanInterface, RawCanInterface, DMSingleMotorCanInterface])
def test_borrowed_transport_remains_usable_after_wrapper_close(
    buses: tuple[VirtualBus, VirtualBus], wrapper: type[CanInterface] | type[RawCanInterface]
) -> None:
    bus, peer = buses
    connection = wrapper(channel="selected-connection", bus=bus)
    connection.close()
    peer.send(can.Message(arbitration_id=1, data=[0, 255], is_extended_id=False))
    reply = bus.recv(timeout=1)
    assert reply is not None
    assert reply.data == bytes([0, 255])


@pytest.mark.parametrize("wrapper", [CanInterface, RawCanInterface, DMSingleMotorCanInterface])
def test_wrapper_closes_a_transport_it_created(
    monkeypatch: pytest.MonkeyPatch, wrapper: type[CanInterface] | type[RawCanInterface]
) -> None:
    with VirtualBus(str(uuid4())) as bus:
        monkeypatch.setattr(can.interface, "Bus", lambda **_kwargs: bus)
        wrapper().close()
        with pytest.raises(can.CanOperationError):
            bus.send(can.Message(arbitration_id=1))


class BeforeMotion(Exception):
    """The test ends before any motor enable or control command."""


@pytest.mark.parametrize("matching_type", [True, False])
def test_injected_bus_checks_real_register_frames_before_motor_enable(
    buses: tuple[VirtualBus, VirtualBus], monkeypatch: pytest.MonkeyPatch, matching_type: bool
) -> None:
    bus, peer = buses
    values = {
        DMRegAddr.GR: MotorType.get_gear_ratio("DM4340") if matching_type else 1000.0,
        DMRegAddr.HW_VER: 1,
        DMRegAddr.SW_VER: 925970741,
        DMRegAddr.CTRL_MODE: 1,
        **motor_check.scaling_for("DM4340"),
    }
    reads = []
    violations = []

    def reply(message: can.Message) -> None:
        data = message.data
        if len(data) != 8 or data[2] != CMD_READ:
            violations.append(message)
            return
        register = DMRegAddr(data[3])
        reads.append(register)
        payload = bytes(data[:4]) + _value_to_bytes(REG_BY_ADDR[int(register)], values[register])
        peer.send(can.Message(arbitration_id=0x7FF, data=payload, is_extended_id=False))

    def before_motion(_chain: dm_driver.DMChainCanInterface) -> None:
        assert {DMRegAddr.GR, DMRegAddr.CTRL_MODE, *motor_check.SCALING_REGISTERS}.issubset(reads)
        raise BeforeMotion

    monkeypatch.setattr(dm_driver.DMChainCanInterface, "_motor_on", before_motion)
    notifier = can.Notifier(peer, [reply])
    try:
        failure = BeforeMotion if matching_type else RuntimeError
        with pytest.raises(failure):
            dm_driver.DMChainCanInterface(
                [(1, "DM4340")],
                np.zeros(1),
                np.ones(1),
                channel="selected-connection",
                start_thread=False,
                check_motor_types=True,
                check_motor_config=True,
                bus=bus,
            )
        assert DMRegAddr.GR in reads
        assert not violations, "Startup validation must not enable motors or repair healthy configuration"
        assert not bus._is_shutdown
    finally:
        notifier.stop(timeout=1)


@pytest.mark.parametrize("borrowed", [True, False])
def test_encoder_failure_closes_only_a_connection_it_created(
    buses: tuple[VirtualBus, VirtualBus], monkeypatch: pytest.MonkeyPatch, borrowed: bool
) -> None:
    bus, _peer = buses
    monkeypatch.setattr(encoder_manager, "get_can_bus", lambda _channel: bus)
    monkeypatch.setattr(encoder_manager.PassiveJointEncoder, "get_version", lambda *_args, **_kwargs: [])
    with pytest.raises(RuntimeError, match="No encoders found"):
        encoder_manager.PassiveJointEncoder.validate_encoders(
            "selected-connection",
            encoder_manager.EncoderConfig(adc_freq=255, report_freq=0, firmware=">=2.2.12"),
            bus=bus if borrowed else None,
        )
    assert bus._is_shutdown is not borrowed


def test_encoder_reader_reuses_its_existing_bus(
    buses: tuple[VirtualBus, VirtualBus], monkeypatch: pytest.MonkeyPatch
) -> None:
    bus, _peer = buses
    seen = []

    def validate(_channel: str, _config: encoder_manager.EncoderConfig, *, bus: can.BusABC) -> dict[int, dict]:
        seen.append(bus)
        return {}

    monkeypatch.setattr(encoder_manager.PassiveJointEncoder, "validate_encoders", validate)
    connection = CanInterface(channel="selected-connection", bus=bus)
    PassiveEncoderReader(connection)
    assert seen == [bus]
    assert not bus._is_shutdown


@pytest.mark.parametrize("arm", [ArmType.YAM, ArmType.NO_ARM])
def test_robot_factory_preserves_the_supplied_transport(
    buses: tuple[VirtualBus, VirtualBus], monkeypatch: pytest.MonkeyPatch, arm: ArmType
) -> None:
    from i2rt.robots import get_robot

    bus, _peer = buses

    def stop_before_construction(*_args: Any, **kwargs: Any) -> None:
        assert kwargs["bus"] is bus
        assert kwargs["check_motor_types"]
        assert kwargs["check_motor_config"]
        raise BeforeMotion

    monkeypatch.setattr(get_robot, "DMChainCanInterface", stop_before_construction)
    with pytest.raises(BeforeMotion):
        get_robot.get_yam_robot(arm_type=arm, bus=bus)
    assert not bus._is_shutdown
