"""Deterministic motion primitives. All physical robot movement goes through this module.

HEAD ONLY, for vision data capture (#28). Owner decision 2026-10-20: "Neck motion is required for
vision data capture. Capture procedures may articulate the head through the neck for viewing and
perception. Arm motion stays forbidden in capture procedures."

These functions touch `robot.head` (the three neck joints) and `robot.turn_on('head')` /
`robot.turn_off_smoothly('head')`, nothing else: never an arm, never the part 'reachy' as a whole.
`robot` is a connected reachy_sdk 0.7.0 `ReachySDK` (or None in dry-run without a connection).

Every move:
  1. knows its neck angles before moving: a preset is a set of neck angles (config.HEAD_PRESETS);
     a look_at point is turned into angles by the SDK's own head IK (`head._look_at`, which is
     what `head.look_at` sends: reachy_sdk/head.py, `inverse_kinematics` -> ComputeHeadIK);
  2. passes `safety.head_gate_check` (limits, duration, look_at box, operator approval,
     REACHY_ENABLE_MOTION), else it is refused and sends nothing;
  3. is live only when the caller also allows it (`allow_live`: the viewer's --allow-head-motion),
     otherwise a logged dry run that sends nothing;
  4. is streamed as minimum-jerk neck goals that a Stop interrupts (the SDK's `goto` cannot be
     interrupted), and ends with a neck read-back.
Every attempt is logged (preset, target, angles, duration, live or dry-run) and returned as a record.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from reachy_ai import config
from reachy_ai.config import (
    HEAD_DEFAULT_DURATION_S,
    HEAD_PRESETS,
    HEAD_READBACK_SETTLE_S,
    NECK_JOINTS,
)
from reachy_ai.motion.safety import MOTION_DISABLED, Pose3D, head_gate_check

LOG = logging.getLogger(__name__)
HEAD_PART = "head"
STREAM_HZ = 50


def preset_angles(preset: str) -> dict[str, float]:
    """The neck angles (degrees) of a named preset; KeyError for an unknown name."""
    try:
        return dict(zip(NECK_JOINTS, HEAD_PRESETS[preset]))
    except KeyError:
        raise KeyError(f"unknown head preset {preset!r}; choose one of {sorted(HEAD_PRESETS)}") from None


def look_at_preset(robot, preset: str, duration: float = HEAD_DEFAULT_DURATION_S, *, operator_approved: bool,
                   allow_live: bool, motion_enabled: bool | None = None, stop_event=None,
                   sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> dict:
    """Move the neck to a named preset (config.HEAD_PRESETS). Returns the move record."""
    return _move_neck(robot, preset_angles(preset), duration, preset=preset, look_at=None,
                      operator_approved=operator_approved, allow_live=allow_live, motion_enabled=motion_enabled,
                      stop_event=stop_event, sleep=sleep, clock=clock)


def return_to_neutral(robot, duration: float = HEAD_DEFAULT_DURATION_S, **kwargs) -> dict:
    """Neck to (0, 0, 0). The head stays powered (stiff) afterwards."""
    return look_at_preset(robot, "neutral", duration, **kwargs)


def look_at_point(robot, x: float, y: float, z: float, duration: float = HEAD_DEFAULT_DURATION_S, *,
                  operator_approved: bool, allow_live: bool, motion_enabled: bool | None = None, stop_event=None,
                  sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> dict:
    """Look at (x, y, z), SDK torso frame (X forward, Y left, Z up, metres), through the same gate.

    The angles come from the SDK head IK before anything moves; without a connection they are
    unknown and the gate refuses the move."""
    target = None
    if robot is not None:
        goals = robot.head._look_at(x, y, z)  # IK only (a kinematics-service call); nothing moves
        target = {joint.name: float(angle) for joint, angle in goals.items()}
    return _move_neck(robot, target, duration, preset=None, look_at=Pose3D(x, y, z),
                      operator_approved=operator_approved, allow_live=allow_live, motion_enabled=motion_enabled,
                      stop_event=stop_event, sleep=sleep, clock=clock)


def hold_head(robot, *, allow_live: bool, motion_enabled: bool | None = None) -> dict:
    """Stop: hold the neck where it is now (goal = present position). Needs no approval: it only ever
    cancels motion. Live under the same flags as a move; otherwise logged only."""
    live = allow_live and _motion_enabled(motion_enabled) and robot is not None
    record = {"action": "stop", "mode": "live" if live else "dry-run", "readback_deg": None}
    if live:
        for joint in _neck(robot).values():
            joint.goal_position = joint.present_position
        LOG.info("HEAD STOP: holding the present neck position")
    else:
        LOG.info("DRY-RUN head stop (nothing sent)")
    record["readback_deg"] = read_neck(robot)
    record["outcome"] = "held" if live else "dry-run"
    return record


def release_head(robot, duration: float = 2.0, *, operator_approved: bool, allow_live: bool,
                 motion_enabled: bool | None = None) -> dict:
    """Turn the head compliant smoothly (`turn_off_smoothly('head')`). Needs the operator's approval."""
    failures = [] if operator_approved else ["Operator approval not given"]
    if not _motion_enabled(motion_enabled):
        failures.append(MOTION_DISABLED)
    blocking = [f for f in failures if f != MOTION_DISABLED]
    live = not failures and allow_live and robot is not None
    record = {"action": "release", "duration_s": duration, "gate_failures": failures,
              "mode": "live" if live else "dry-run"}
    if blocking:
        LOG.warning("HEAD RELEASE refused: %s", "; ".join(blocking))
        record["outcome"] = "refused"
    elif live:
        LOG.info("HEAD RELEASE: turning the head compliant over %.1f s", duration)
        robot.turn_off_smoothly(HEAD_PART, duration=duration)
        record["outcome"] = "released"
    else:
        LOG.info("DRY-RUN head release over %.1f s (nothing sent)", duration)
        record["outcome"] = "dry-run"
    record["readback_deg"] = read_neck(robot)
    return record


def read_neck(robot) -> dict[str, float] | None:
    """Present neck angles in degrees, or None without a connection or on a failed read."""
    if robot is None:
        return None
    try:
        return {name: float(joint.present_position) for name, joint in _neck(robot).items()}
    except Exception as exc:  # noqa: BLE001 - a read-back is informational; report the type only
        LOG.warning("neck read-back failed: %s", type(exc).__name__)
        return None


#: IK-only probes of the neck's sign convention: (look_at point in metres, joint, what a positive angle means).
SIGN_PROBES = (((0.3, 0.0, -0.4), "neck_pitch", "looks down"), ((0.5, 0.3, 0.0), "neck_yaw", "turns left"))


def neck_sign_check(robot) -> dict:
    """Compute the SDK head IK for two known points and check the angle signs. Nothing moves.

    Returns {"result": "passed" | "failed", "detail": str}; a failed IK call fails the check."""
    for point, joint_name, meaning in SIGN_PROBES:
        try:
            goals = robot.head._look_at(*point)  # IK only (a kinematics-service call)
            angles = {joint.name: float(angle) for joint, angle in goals.items()}
        except Exception as exc:  # noqa: BLE001 - report the type only
            return {"result": "failed", "detail": f"IK call failed ({type(exc).__name__})", "ik_failed": True}
        value = angles.get(joint_name)
        if value is None or not value > 0:
            return {"result": "failed",
                    "detail": f"look_at{point} gave {joint_name} {value}; expected > 0 ({meaning})"}
    return {"result": "passed",
            "detail": "; ".join(f"look_at{p} -> {n} > 0 ({m})" for p, n, m in SIGN_PROBES)}


def _neck(robot) -> dict[str, Any]:
    joints = robot.head.joints
    return {name: getattr(joints, name) for name in NECK_JOINTS}


def _motion_enabled(value: bool | None) -> bool:
    return config.REACHY_ENABLE_MOTION if value is None else value


def _move_neck(robot, target: dict[str, float] | None, duration: float, *, preset: str | None,
               look_at: Pose3D | None, operator_approved: bool, allow_live: bool, motion_enabled: bool | None,
               stop_event, sleep, clock) -> dict:
    present = read_neck(robot)  # before the move: the gate bounds the speed from here to the target
    gate = head_gate_check(target, duration, operator_approved, look_at, motion_enabled, present)
    blocking = [f for f in gate.failures if f != MOTION_DISABLED]
    live = gate.passed and allow_live and robot is not None
    record = {"action": "move", "preset": preset,
              "look_at_m": None if look_at is None else [look_at.x, look_at.y, look_at.z],
              "target_deg": target, "duration_s": duration, "mode": "live" if live else "dry-run",
              "gate": {"passed": gate.passed, "failures": gate.failures}}
    what = preset or f"look_at {record['look_at_m']}"
    if blocking:
        LOG.warning("HEAD MOVE refused: %s -> %s over %.1f s: %s", what, target, duration, "; ".join(blocking))
        record["outcome"] = "refused"
    elif not live:
        LOG.info("DRY-RUN head move %s -> %s over %.1f s (nothing sent)", what, target, duration)
        record["outcome"] = "dry-run"
    else:
        LOG.info("HEAD MOVE %s -> %s over %.1f s (live)", what, target, duration)
        record["outcome"] = _stream(robot, target, duration, stop_event, sleep, clock)
        sleep(HEAD_READBACK_SETTLE_S)
    record["readback_deg"] = read_neck(robot)
    LOG.info("head move %s: %s; neck read-back %s", what, record["outcome"], record["readback_deg"])
    return record


def _stream(robot, target: dict[str, float], duration: float, stop_event, sleep, clock) -> str:
    """Minimum-jerk neck goals from the present position, at STREAM_HZ. 'reached' or 'stopped'."""
    if any(getattr(robot.head.joints, name).compliant for name in NECK_JOINTS):
        robot.turn_on(HEAD_PART)  # stiff; the SDK sets goal = present first, so nothing jumps
    joints = _neck(robot)
    start = {name: float(joint.present_position) for name, joint in joints.items()}
    began = clock()
    while True:
        if stop_event is not None and stop_event.is_set():
            for joint in joints.values():
                joint.goal_position = joint.present_position
            return "stopped"
        s = min(1.0, (clock() - began) / duration)
        blend = s ** 3 * (10 - 15 * s + 6 * s * s)
        for name, joint in joints.items():
            joint.goal_position = start[name] + (target[name] - start[name]) * blend
        if s >= 1.0:
            return "reached"
        sleep(1 / STREAM_HZ)
