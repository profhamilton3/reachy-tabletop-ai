"""Shared helpers for the read-only operator scripts (zoom_fov_check, ruler_depth_test). Not run on its own.

These scripts connect with reachy_sdk (v1) using REACHY_IP from the environment; no address
is stored in code or printed. They read camera frames and, only where a script says so and
the operator passes the explicit flag, send zoom commands through reachy_ai.camera_zoom.
No arm, head or gripper command is ever sent.
"""

import os
import sys
import time


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
