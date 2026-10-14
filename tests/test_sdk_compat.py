"""reachy_sdk 0.7.0 command-poll leak and its fix (sdk_compat), issue #23.
Offline: the SDK's own poll runs against stand-in joints on a local event
loop.  Ported from reachy-1-2-sim tests/unit/test_sdk_compat.py.

These need `reachy_sdk` installed.  Without it they SKIP -- unless
REQUIRE_REACHY_SDK=1, which CI sets, where a missing SDK must FAIL.
"""
import ast
import asyncio
import hashlib
import os

import pytest

if os.environ.get("REQUIRE_REACHY_SDK") == "1":
    import reachy_sdk.reachy_sdk as sdk  # fail, never skip
else:
    sdk = pytest.importorskip("reachy_sdk.reachy_sdk")
from reachy_sdk_api.joint_pb2 import JointCommand, JointId  # noqa: E402

from reachy_ai import reachy_client, sdk_compat  # noqa: E402

JOINTS = 21          # Reachy 1.2 joint count (as in the simulator)
COMMANDED = 7        # a right-arm trajectory streams the 7 right-arm joints
STOCK = sdk.ReachySDK.__dict__["_poll_waiting_commands"]


class _Joint:
    def __init__(self, uid):
        self.uid = uid
        self._need_sync = asyncio.Event()

    def _pop_command(self):
        self._need_sync.clear()
        return JointCommand(id=JointId(uid=self.uid))


class _SDK:
    pass


def _drive(poll, pushes, commanded=lambda i: range(COMMANDED)):
    """`pushes` command pushes; returns (uids sent per push, pending tasks)."""
    async def main():
        s = _SDK()
        s._joints = [_Joint(u) for u in range(JOINTS)]
        sent = []
        for i in range(pushes):
            for u in commanded(i):
                s._joints[u]._need_sync.set()
            cmd = await poll(s)
            sent.append(sorted(c.id.uid for c in cmd.commands))
        await asyncio.sleep(0)                    # let cancellations land
        me = asyncio.current_task()
        return sent, sum(1 for t in asyncio.all_tasks() if t is not me and not t.done())
    return asyncio.run(main())


def test_stock_poll_leaves_a_task_per_uncommanded_joint_per_push():
    """Pins the diagnosis: the heap growth of any long-lived commanding client."""
    _, pending = _drive(STOCK, 50)
    assert pending == 50 * (JOINTS - COMMANDED)


def test_fixed_poll_leaves_nothing_pending():
    _, pending = _drive(sdk_compat._poll_waiting_commands, 50)
    assert pending == 0


def test_fixed_poll_sends_exactly_what_the_stock_poll_sends():
    pattern = lambda i: [u for u in range(JOINTS) if (u * 7 + i) % 5 == 0] or [0]  # noqa: E731
    stock, _ = _drive(STOCK, 40, pattern)
    fixed, _ = _drive(sdk_compat._poll_waiting_commands, 40, pattern)
    assert fixed == stock


def test_a_release_other_than_the_measured_one_is_left_alone(monkeypatch):
    monkeypatch.setattr(sdk.ReachySDK, "_poll_waiting_commands", STOCK)
    monkeypatch.setattr(sdk_compat, "_installed_release", lambda: "0.8.0")
    assert sdk_compat.fix_command_poll() is False
    assert sdk.ReachySDK.__dict__["_poll_waiting_commands"] is STOCK


def test_the_measured_release_is_fixed_once(monkeypatch):
    monkeypatch.setattr(sdk.ReachySDK, "_poll_waiting_commands", STOCK)
    monkeypatch.setattr(sdk_compat, "_installed_release", lambda: sdk_compat.FIXED_RELEASE)
    assert sdk_compat.fix_command_poll() is True
    assert sdk_compat.fix_command_poll() is True
    assert sdk.ReachySDK.__dict__["_poll_waiting_commands"] is sdk_compat._poll_waiting_commands


#: sha256 of the stock `ReachySDK._poll_waiting_commands` source in
#: reachy-sdk 0.7.0 (the code `sdk_compat` replaces).  If this changes, the
#: SDK changed under the fix: re-read the new poll before trusting either.
STOCK_POLL_SHA256 = "fc2c2ab7183cff4ef863d07c7312d21a735cdf55360918e065ba6ca2eaeb1d3f"


def _stock_poll_source():
    """Read from the installed file, so a fix already applied in this
    process cannot hide a change."""
    src = open(sdk.__file__).read()
    cls = next(n for n in ast.parse(src).body
               if isinstance(n, ast.ClassDef) and n.name == "ReachySDK")
    fn = next(n for n in cls.body if getattr(n, "name", "") == "_poll_waiting_commands")
    return ast.get_source_segment(src, fn)


def test_the_stock_poll_is_the_one_the_fix_was_written_against():
    assert sdk_compat._installed_release() == sdk_compat.FIXED_RELEASE
    digest = hashlib.sha256(_stock_poll_source().encode()).hexdigest()
    assert digest == STOCK_POLL_SHA256, (
        "reachy_sdk's _poll_waiting_commands changed: review sdk_compat before "
        "updating this hash")


def test_reachy_client_fixes_the_poll_before_it_connects(monkeypatch):
    import reachy_sdk
    order = []

    class _Robot:
        def __init__(self, host):
            order.append("connect")

    monkeypatch.setattr(reachy_sdk, "ReachySDK", _Robot)
    monkeypatch.setattr(sdk_compat, "fix_command_poll",
                        lambda: order.append("fix") or True)
    client = reachy_client.ReachyClient(host="localhost", dry_run=False)
    assert order == ["fix", "connect"]
    assert client.command_poll_fixed is True


def test_dry_run_client_neither_connects_nor_patches(monkeypatch):
    monkeypatch.setattr(sdk_compat, "fix_command_poll",
                        lambda: pytest.fail("dry run must not touch the SDK"))
    client = reachy_client.ReachyClient(host="localhost", dry_run=True)
    assert not client.connected and client.command_poll_fixed is False
