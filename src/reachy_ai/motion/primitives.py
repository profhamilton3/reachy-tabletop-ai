"""Deterministic motion primitives. All physical robot movement goes through this module.

HEAD (the neck) and the right GRIPPER, for data capture only; never an arm.

HEAD, for vision data capture (#28). Owner decision 2026-10-20: "Neck motion is required for
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

GRIPPER, for gripper calibration (#27 Part A). Owner decision 2026-10-22: "Gripper-only motion is allowed
in capture procedures for gripper calibration: only `r_gripper` is powered, the arm stays at REST, every
move needs a confirm and passes a gripper gate (travel limits, low speed and torque, a force-limited close,
`REACHY_ENABLE_MOTION`), and the gripper returns to compliant at the end."

`gripper_read`, `gripper_step` and `gripper_release` touch `robot.r_arm.joints.r_gripper` (power, limits,
goal) and read `robot.force_sensors.r_force_gripper`, nothing else: never `turn_on('r_arm')` (reachy_sdk
0.7.0 powers a single joint through `joint.compliant`), `goto`, `look_at` or another joint. A step is one
bounded move toward a target (`safety.gripper_gate_check`: limits, step size, speed, torque, approval,
REACHY_ENABLE_MOTION), polled at config.GRIPPER_SAMPLE_HZ. It ends 'reached' (the gripper stays powered at
the target) or, for any of end_stop / temperature / load / force / stop / timeout / error, with
`gripper_release` (compliant). Dry-run sends nothing: reads only.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from reachy_ai import config
from reachy_ai.config import (
    GRIPPER_FOLLOW_TOL_DEG,
    GRIPPER_MAX_FORCE,
    GRIPPER_MAX_LOAD,
    GRIPPER_MAX_TEMP_C,
    GRIPPER_SAMPLE_CAP,
    GRIPPER_SAMPLE_HZ,
    GRIPPER_SETTLE_S,
    GRIPPER_STALL_MIN_PROGRESS_DEG,
    GRIPPER_STALL_WINDOW_S,
    GRIPPER_STEP_TIMEOUT_S,
    HEAD_DEFAULT_DURATION_S,
    HEAD_PRESETS,
    HEAD_READBACK_SETTLE_S,
    NECK_JOINTS,
)
from reachy_ai.motion.safety import MOTION_DISABLED, Pose3D, gripper_gate_check, head_gate_check

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


# --- Gripper (#27 Part A) --------------------------------------------------------------------------------

GRIPPER_FORCE_SENSOR = "r_force_gripper"
#: How a move ended. Only "reached" leaves the gripper powered; every other outcome ends with gripper_release.
GRIPPER_ABORTS = ("temperature", "load", "force", "stop", "timeout", "error")
GRIPPER_END_OUTCOMES = ("reached", "end_stop", *GRIPPER_ABORTS)


def _gripper_joint(robot):
    """The r_gripper joint: the only joint these functions power or command."""
    return robot.r_arm.joints.r_gripper


def _try(read) -> tuple[Any, str | None]:
    """(value, None), or (None, the exception's type name): a failed read is data, not a crash."""
    try:
        return read(), None
    except Exception as exc:  # noqa: BLE001 - report the type only
        return None, type(exc).__name__


def gripper_read(robot) -> dict:
    """Everything Stage 0 reads of the gripper, each read on its own: position, goal, compliance, torque
    and speed limits, temperature, present_load and the gripper force sensor. Nothing is written.

    Keys are None where a read failed; `errors` maps them to the exception type. `sdk_open_close` says
    whether this SDK has a gripper object with open() and close() (reachy_sdk 0.7.0 has not)."""
    names = ("position_deg", "goal_deg", "compliant", "torque_limit_pct", "speed_limit_deg_s", "temperature_c",
             "present_load", "force")
    if robot is None:
        return {**dict.fromkeys(names), "connected": False, "errors": {}, "sdk_open_close": None}
    joint, joint_error = _try(lambda: _gripper_joint(robot))
    reads = {
        "position_deg": lambda: float(joint.present_position),
        "goal_deg": lambda: float(joint.goal_position),
        "compliant": lambda: bool(joint.compliant),
        "torque_limit_pct": lambda: float(joint.torque_limit),
        "speed_limit_deg_s": lambda: float(joint.speed_limit),
        "temperature_c": lambda: float(joint.temperature),
        "present_load": lambda: float(joint.present_load),
        "force": lambda: float(getattr(robot.force_sensors, GRIPPER_FORCE_SENSOR).force),
    }
    values, errors = {}, {}
    for name in names:
        values[name], error = _try(reads[name]) if joint_error is None or name == "force" else (None, joint_error)
        if error:
            errors[name] = error
    gripper, _ = _try(lambda: getattr(robot.r_arm, "gripper"))
    values["sdk_open_close"] = {"gripper_object": gripper is not None,
                                "open": callable(getattr(gripper, "open", None)),
                                "close": callable(getattr(gripper, "close", None))}
    LOG.info("gripper read: %s%s", {k: v for k, v in values.items() if k != "sdk_open_close"},
             f" (read errors: {errors})" if errors else "")
    return {**values, "connected": True, "errors": errors}


def gripper_release(robot, *, allow_live: bool, motion_enabled: bool | None = None, reason: str = "release") -> dict:
    """Turn the gripper compliant: `joint.compliant = True` on r_gripper alone. Needs no approval (it only
    removes power); live under the same flags as a move, otherwise a logged dry run. The torque and speed
    limits of the last move stay as they were (low); nothing here raises them."""
    live = allow_live and _motion_enabled(motion_enabled) and robot is not None
    record = {"action": "release", "reason": reason, "mode": "live" if live else "dry-run"}
    if live:
        LOG.info("GRIPPER RELEASE (%s): r_gripper compliant", reason)
        error = _try(lambda: setattr(_gripper_joint(robot), "compliant", True))[1]
        record["outcome"] = "released" if error is None else "error"
        if error:
            record["error"] = error
            LOG.error("GRIPPER RELEASE failed: %s", error)
    else:
        LOG.info("DRY-RUN gripper release (%s; nothing sent)", reason)
        record["outcome"] = "dry-run"
    record["readback"] = gripper_read(robot)
    return record


def gripper_step(robot, target_deg: float, speed_limit: float, torque_limit: float, *, operator_approved: bool,
                 allow_live: bool, motion_enabled: bool | None = None, force_zero: float | None = None,
                 stop_event=None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> dict:
    """One bounded move of r_gripper toward `target_deg` (degrees), then the measured state.

    The gate (`safety.gripper_gate_check`) sees the present position first; a refusal sends nothing. Live
    (gate passed, `allow_live`) the limits are set BEFORE the joint is powered, the goal is sent, and the
    move is polled at GRIPPER_SAMPLE_HZ until it ends:

      reached      within GRIPPER_FOLLOW_TOL_DEG of the target for GRIPPER_SETTLE_S; the gripper stays powered
      end_stop     the position stopped following (GRIPPER_STALL_WINDOW_S without GRIPPER_STALL_MIN_PROGRESS_DEG)
      temperature / load / force   a reading over its limit (force is read against `force_zero`, the
                   Stage 0 zero; an unreadable sensor cannot trip)
      stop         `stop_event` was set
      timeout      GRIPPER_STEP_TIMEOUT_S passed
      error        the position could not be read, or the SDK raised

    Everything but "reached" ends with `gripper_release`. The record carries `samples` (time, target,
    present, load, force, temperature, limits), the outcome and the read-back."""
    present, _ = _try(lambda: float(_gripper_joint(robot).present_position)) if robot is not None else (None, None)
    gate = gripper_gate_check(target_deg, speed_limit, torque_limit, operator_approved, motion_enabled, present)
    blocking = [f for f in gate.failures if f != MOTION_DISABLED]
    live = gate.passed and allow_live and robot is not None
    record = {"action": "move", "target_deg": target_deg, "speed_limit_deg_s": speed_limit,
              "torque_limit_pct": torque_limit, "present_before_deg": present, "force_zero": force_zero,
              "mode": "live" if live else "dry-run", "gate": {"passed": gate.passed, "failures": gate.failures},
              "samples": []}
    if blocking:
        LOG.warning("GRIPPER MOVE refused: -> %s deg: %s", target_deg, "; ".join(blocking))
        record["outcome"] = "refused"
    elif not live:
        LOG.info("DRY-RUN gripper move %s -> %s deg at %s deg/s, %s %% (nothing sent)", present, target_deg,
                 speed_limit, torque_limit)
        record["outcome"] = "dry-run"
    else:
        LOG.info("GRIPPER MOVE %s -> %s deg at %s deg/s, %s %% (live)", present, target_deg, speed_limit,
                 torque_limit)
        try:
            record["outcome"] = _drive_gripper(robot, target_deg, speed_limit, torque_limit, force_zero, stop_event,
                                               sleep, clock, record)
        except Exception as exc:  # noqa: BLE001 - report the type only; the release below is the safe state
            record["outcome"], record["error"] = "error", type(exc).__name__
            LOG.error("GRIPPER MOVE failed: %s", type(exc).__name__)
        if record["outcome"] != "reached":
            record["release"] = gripper_release(robot, allow_live=allow_live, motion_enabled=motion_enabled,
                                                reason=record["outcome"])
    record["readback"] = gripper_read(robot)
    LOG.info("gripper move -> %s deg: %s; read-back %s", target_deg, record["outcome"],
             {k: record["readback"].get(k) for k in ("position_deg", "present_load", "force", "temperature_c")})
    return record


def _gripper_sample(joint, robot, target, speed, torque, now) -> dict:
    """One poll. The position must be readable; the other sensors are None where they are not."""
    sample = {"t": now, "target_deg": target, "present_deg": float(joint.present_position),
              "torque_limit_pct": torque, "speed_limit_deg_s": speed}
    sample["load"], _ = _try(lambda: float(joint.present_load))
    sample["force"], _ = _try(lambda: float(getattr(robot.force_sensors, GRIPPER_FORCE_SENSOR).force))
    sample["temperature_c"], _ = _try(lambda: float(joint.temperature))
    return sample


def _gripper_limit_hit(sample: dict, force_zero: float | None) -> str | None:
    if sample["temperature_c"] is not None and sample["temperature_c"] > GRIPPER_MAX_TEMP_C:
        return "temperature"
    if sample["load"] is not None and abs(sample["load"]) > GRIPPER_MAX_LOAD:
        return "load"
    if sample["force"] is not None and abs(sample["force"] - (force_zero or 0.0)) > GRIPPER_MAX_FORCE:
        return "force"
    return None


def _drive_gripper(robot, target, speed, torque, force_zero, stop_event, sleep, clock, record) -> str:
    joint = _gripper_joint(robot)
    joint.torque_limit = torque       # limits first, then power: nothing moves at the old (possibly high) limits
    joint.speed_limit = speed
    joint.compliant = False           # r_gripper only; the SDK sets goal = present first, so nothing jumps
    joint.goal_position = target
    began = clock()
    anchor_t, anchor_pos = began, float(joint.present_position)
    arrived = None  # when the position came inside the tolerance
    while True:
        now = clock()
        sample = _gripper_sample(joint, robot, target, speed, torque, now)
        if len(record["samples"]) < GRIPPER_SAMPLE_CAP:
            record["samples"].append(sample)
        if stop_event is not None and stop_event.is_set():
            return "stop"
        hit = _gripper_limit_hit(sample, force_zero)
        if hit:
            return hit
        if abs(sample["present_deg"] - target) <= GRIPPER_FOLLOW_TOL_DEG:
            arrived = now if arrived is None else arrived
            if now - arrived >= GRIPPER_SETTLE_S:  # let it settle, so the read-back is the arrival, not the approach
                return "reached"
            sleep(1 / GRIPPER_SAMPLE_HZ)
            continue
        arrived = None
        if abs(sample["present_deg"] - anchor_pos) >= GRIPPER_STALL_MIN_PROGRESS_DEG:
            anchor_t, anchor_pos = now, sample["present_deg"]
        elif now - anchor_t >= GRIPPER_STALL_WINDOW_S:
            return "end_stop"
        if now - began > GRIPPER_STEP_TIMEOUT_S:
            return "timeout"
        sleep(1 / GRIPPER_SAMPLE_HZ)
