"""Shared helpers for the read-only operator scripts (zoom_fov_check, ruler_depth_test). Not run on its own.

These scripts connect with reachy_sdk (v1) using REACHY_IP from the environment; no address
is stored in code or printed. They read camera frames and, only where a script says so and
the operator passes the explicit flag, send zoom commands through reachy_ai.camera_zoom.
No arm, head or gripper command is ever sent. Each starts with the read-only pre-capture health check
(reachy_ai.health.precapture) and stops if it fails.
"""

import os
import sys
import time
from datetime import datetime, timezone


def connect():
    host = os.environ.get("REACHY_IP", "")
    if not host:
        sys.exit("REACHY_IP is not set. export REACHY_IP=<ip> first (never put the address in code).")
    from reachy_sdk import ReachySDK
    return ReachySDK(host=host)


def fresh_frame(camera, timeout_s=5.0, previous=None):
    """The next decoded frame that is not `previous` (SDK 0.7 replaces the array per stream message)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = camera.last_frame
        if frame is not None and frame.size and frame is not previous:
            return frame.copy()
        time.sleep(0.02)
    raise TimeoutError("no new camera frame")


def fresh_pair(client, previous=(None, None)):
    """(left, right) BGR frames, each a new decode (not synchronized exposure)."""
    return (fresh_frame(client.left_camera, previous=previous[0]),
            fresh_frame(client.right_camera, previous=previous[1]))


def health_check(robot):
    """The pre-capture health check (read-only). Prints each check; exits with the failed ones if any failed.
    Returns the report as a dict for the run (CaptureRun.precapture)."""
    from reachy_ai.health import precapture

    report = precapture.check(robot).to_dict()
    for item in report["checks"]:
        mark = {True: "ok", False: "FAIL", None: "not run"}[item["passed"]]
        print(f"health {item['name']:>13}: {mark:7} {item['detail']}")
    if not report["passed"]:
        sys.exit(precapture.refusal_text(report))
    return report


def fresh_pair_received(client, previous=(None, None)):
    """(left, right, frames_received): fresh_pair, with the wall-clock time each new frame was seen here."""
    frames, received = [], {}
    for eye, camera, before in (("left", client.left_camera, previous[0]), ("right", client.right_camera, previous[1])):
        frames.append(fresh_frame(camera, previous=before))
        now = time.time_ns()
        received[eye] = {"observed_time_ns": None, "received_unix_ns": now,
                         "received_utc": datetime.fromtimestamp(now / 1e9, timezone.utc).strftime(
                             "%Y-%m-%dT%H:%M:%S.%fZ")}
    return frames[0], frames[1], received
