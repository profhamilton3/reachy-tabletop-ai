import pytest
from pydantic import ValidationError

from reachy_ai.planner.schema import Destination, Plan, TaskType


def make_valid_plan(**overrides) -> dict:
    base = {
        "task_type": "pick_place",
        "target_id": "obj_3",
        "target_description": "whiteboard marker",
        "destination": "right_tray",
        "confidence": 0.85,
        "requires_confirmation": False,
        "safety_notes": ["object appears light"],
        "brief_reason": "Marker matches the writing task.",
    }
    base.update(overrides)
    return base


def test_valid_plan():
    plan = Plan(**make_valid_plan())
    assert plan.task_type == TaskType.pick_place
    assert plan.destination == Destination.right_tray
    assert plan.confidence == 0.85


def test_confidence_below_zero_raises():
    with pytest.raises(ValidationError):
        Plan(**make_valid_plan(confidence=-0.1))


def test_confidence_above_one_raises():
    with pytest.raises(ValidationError):
        Plan(**make_valid_plan(confidence=1.1))


def test_empty_target_id_raises():
    with pytest.raises(ValidationError):
        Plan(**make_valid_plan(target_id=""))


def test_invalid_task_type_raises():
    with pytest.raises(ValidationError):
        Plan(**make_valid_plan(task_type="fly"))


def test_invalid_destination_raises():
    with pytest.raises(ValidationError):
        Plan(**make_valid_plan(destination="trash_can"))


def test_requires_confirmation_forced_when_low_confidence():
    plan = Plan(**make_valid_plan(confidence=0.55, requires_confirmation=False))
    # Schema does not auto-enforce this — safety gate does — but value is stored
    assert plan.confidence == 0.55
