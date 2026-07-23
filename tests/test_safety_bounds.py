from reachy_ai.motion.safety import Pose3D, gate_check


def _valid_pose() -> Pose3D:
    return Pose3D(x=0.5, y=0.0, z=0.10)


def test_gate_passes_when_all_conditions_met():
    result = gate_check(
        target_pose=_valid_pose(),
        operator_approved=True,
    )
    assert result.passed
    assert result.failures == []


def test_gate_fails_without_operator_approval():
    result = gate_check(target_pose=_valid_pose(), operator_approved=False)
    assert not result.passed
    assert any("approval" in f.lower() for f in result.failures)


def test_gate_fails_when_target_outside_workspace():
    out_of_range = Pose3D(x=2.0, y=0.0, z=0.10)
    result = gate_check(target_pose=out_of_range, operator_approved=True)
    assert not result.passed
    assert any("workspace" in f.lower() for f in result.failures)


def test_gate_fails_when_human_hand_detected():
    result = gate_check(
        target_pose=_valid_pose(),
        operator_approved=True,
        human_hand_detected=True,
    )
    assert not result.passed
    assert any("human hand" in f.lower() for f in result.failures)


def test_gate_fails_when_force_sensor_not_clear():
    result = gate_check(
        target_pose=_valid_pose(),
        operator_approved=True,
        force_sensor_clear=False,
    )
    assert not result.passed
    assert any("force sensor" in f.lower() for f in result.failures)


def test_gate_fails_destination_out_of_workspace():
    result = gate_check(
        target_pose=_valid_pose(),
        destination_pose=Pose3D(x=5.0, y=0.0, z=0.0),
        operator_approved=True,
    )
    assert not result.passed


def test_gate_collects_multiple_failures():
    result = gate_check(
        target_pose=Pose3D(x=5.0, y=0.0, z=0.0),
        operator_approved=False,
        human_hand_detected=True,
    )
    assert not result.passed
    assert len(result.failures) >= 3
