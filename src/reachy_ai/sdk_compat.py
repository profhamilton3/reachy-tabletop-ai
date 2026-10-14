"""Fixes for the installed `reachy_sdk` (v1, 0.7.0), applied once per process.

THE COMMAND-POLL LEAK
---------------------
`ReachySDK._poll_waiting_commands` runs once per command push (up to 100 Hz
while commands flow).  Each call creates one asyncio task per joint, waiting
on that joint's `_need_sync` event, returns as soon as ONE fires, and never
cancels the rest.  A task for a joint that is never commanded -- on this
robot the left arm, the neck and the antennas, 13-14 of 21 joints when only
the right arm moves -- never completes, so every push leaves 13-14 tasks
pending on the SDK's loop for the life of the process.

Measured (2026-10-09, offline with the installed SDK code, emulated amd64
container): 14 pending tasks and 112 GC-tracked objects per push, linear;
4,000 pushes = +448k objects and a full collection 26 -> 75 ms.  A lift
streams at least ~5,000 pushes, so a process that is kept between lifts (the
panel's motion worker) collects ever longer: generation-2 pauses grew 7 ->
124 ms inside one lift, and a 0.67 s pause halted attempt 5 of the
2026-10-08 measurement on stale feedback (consistent; not proven).

The fix is the same function with the losers cancelled.  Cancelling an
`asyncio.Event.wait()` removes its waiter, so nothing is retained.  Which
commands are sent is unchanged: they are built from the events, not from the
tasks.

PORTING NOTE: any long-lived `reachy_sdk` 0.7.0 client that commands a
subset of the joints has this leak, on the real robot as on the simulator.

Ported unchanged from reachy-1-2-sim (#167, #168) for issue #23.  Call
`fix_command_poll()` once per process, before `ReachySDK(...)` is
constructed: `ReachyClient` does, and so do the notebooks that send commands.
"""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)

#: The SDK release this fix was written against and measured on.
FIXED_RELEASE = "0.7.0"


async def _poll_waiting_commands(self):
    """`ReachySDK._poll_waiting_commands`, 0.7.0, with the waits that did not
    fire cancelled."""
    from reachy_sdk_api import joint_pb2

    tasks = [asyncio.create_task(joint._need_sync.wait()) for joint in self._joints]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
    return joint_pb2.JointsCommand(
        commands=[joint._pop_command() for joint in self._joints
                  if joint._need_sync.is_set()],
    )


def _installed_release() -> str:
    try:
        from importlib.metadata import version
        return version("reachy-sdk")
    except Exception:                             # noqa: BLE001 - not installed
        return ""


def fix_command_poll() -> bool:
    """Replace the leaking poll on `ReachySDK`, before any instance runs it.

    Returns whether the fixed poll is in place.  A release other than the
    one measured is left alone and logged: its poll may differ, and a blind
    replacement could drop commands.
    """
    from reachy_sdk import reachy_sdk as sdk_module
    cls = sdk_module.ReachySDK
    if cls._poll_waiting_commands is _poll_waiting_commands:
        return True
    release = _installed_release()
    if release != FIXED_RELEASE:
        log.warning("reachy_sdk %s is not %s: command-poll fix not applied",
                    release or "(unknown)", FIXED_RELEASE)
        return False
    cls._poll_waiting_commands = _poll_waiting_commands
    return True
