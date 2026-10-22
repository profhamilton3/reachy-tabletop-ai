"""ReachyClient sends no arm or gripper motion (#68): live, every such method raises MotionRefused and the
SDK sees nothing; dry-run logs. Offline, with a fake SDK that fails on any arm or gripper write."""

import logging
import runpy
import sys
import types
from pathlib import Path

import pytest

from reachy_ai.motion.safety import MotionRefused
from reachy_ai.reachy_client import ReachyClient


class Tripwire:
    """Any attribute write, or call, on the arm or gripper fails the test."""

    def __init__(self, name):
        object.__setattr__(self, "_name", name)

    def __getattr__(self, attr):
        if attr in ("goal_position", "compliant"):
            return 0.0
        return Tripwire(f"{self._name}.{attr}")

    def __setattr__(self, attr, value):
        raise AssertionError(f"direct write {self._name}.{attr} = {value!r}")

    def __call__(self, *args, **kwargs):
        raise AssertionError(f"direct call {self._name}({args}, {kwargs})")


class FakeReachy:
    def __init__(self):
        self.r_arm = Tripwire("r_arm")
        self.r_gripper = Tripwire("r_gripper")

    def turn_on(self, *args):
        raise AssertionError(f"turn_on{args}")

    def turn_off(self, *args):
        raise AssertionError(f"turn_off{args}")


def live_client():
    client = ReachyClient(dry_run=True)  # no SDK import; then go live with the fake
    client.dry_run = False
    client._reachy = FakeReachy()
    return client


LIVE_CALLS = [
    ("set_joint_goal", ("r_elbow_pitch", 30.0, 2.0), "arm gate"),
    ("open_gripper", (), "capture procedures"),
    ("close_gripper", (), "capture procedures"),
    ("turn_on_arm", (), "power changes"),
    ("turn_off_arm", (), "can drop it"),
]


@pytest.mark.parametrize("method,args,why", LIVE_CALLS)
def test_live_arm_and_gripper_methods_refuse(method, args, why):
    with pytest.raises(MotionRefused, match=why) as exc:
        getattr(live_client(), method)(*args)
    assert "nothing was sent" in str(exc.value)


@pytest.mark.parametrize("method,args,why", LIVE_CALLS)
def test_dry_run_logs_and_does_not_raise(method, args, why, caplog):
    client = ReachyClient(dry_run=True)
    with caplog.at_level(logging.INFO):
        getattr(client, method)(*args)
    assert f"DRY-RUN {method}" in caplog.text


def test_get_joint_positions_is_still_a_read():
    client = live_client()
    client._reachy = types.SimpleNamespace(
        r_arm=types.SimpleNamespace(joints={"r_elbow_pitch": types.SimpleNamespace(present_position=12.5)}))
    assert client.get_joint_positions() == {"r_elbow_pitch": 12.5}


def test_run_demo_dry_still_completes(monkeypatch):
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_demo_dry.py"
    monkeypatch.setattr(sys, "argv", [str(path)])
    runpy.run_path(str(path), run_name="__main__")
