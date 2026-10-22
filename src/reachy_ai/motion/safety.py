"""Safety gate — all checks must pass before any motion is allowed."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from reachy_ai import config
from reachy_ai.config import (
    GRIPPER_LIMITS_DEG,
    GRIPPER_MAX_SPEED_DEG_S,
    GRIPPER_MAX_STEP_DEG,
    GRIPPER_MAX_TORQUE_PCT,
    GRIPPER_STEP_MARGIN_DEG,
    HEAD_LOOK_AT_X_MAX,
    HEAD_LOOK_AT_X_MIN,
    HEAD_LOOK_AT_Y_MAX,
    HEAD_LOOK_AT_Y_MIN,
    HEAD_LOOK_AT_Z_MAX,
    HEAD_LOOK_AT_Z_MIN,
    HEAD_MAX_AVG_SPEED_DEG_S,
    HEAD_MIN_DURATION_S,
    HEAD_STEEP_MIN_DURATION_S,
    HEAD_STEEP_PITCH_DEG,
    HEAD_UNKNOWN_START_MIN_DURATION_S,
    NECK_JOINTS,
    NECK_LIMITS_DEG,
    WORKSPACE_X_MAX,
    WORKSPACE_X_MIN,
    WORKSPACE_Y_MAX,
    WORKSPACE_Y_MIN,
    WORKSPACE_Z_MAX,
    WORKSPACE_Z_MIN,
)

logger = logging.getLogger(__name__)

#: The one head-gate failure that leaves a move as a logged dry run rather than a refusal.
MOTION_DISABLED = "REACHY_ENABLE_MOTION is not true"


@dataclass
class Pose3D:
    x: float
    y: float
    z: float


@dataclass
class GateResult:
    passed: bool
    failures: list[str]


WORKSPACE = ((WORKSPACE_X_MIN, WORKSPACE_X_MAX), (WORKSPACE_Y_MIN, WORKSPACE_Y_MAX),
             (WORKSPACE_Z_MIN, WORKSPACE_Z_MAX))
HEAD_LOOK_AT_BOX = ((HEAD_LOOK_AT_X_MIN, HEAD_LOOK_AT_X_MAX), (HEAD_LOOK_AT_Y_MIN, HEAD_LOOK_AT_Y_MAX),
                    (HEAD_LOOK_AT_Z_MIN, HEAD_LOOK_AT_Z_MAX))


def gate_check(
    target_pose: Pose3D,
    destination_pose: Pose3D | None = None,
    human_hand_detected: bool = False,
    operator_approved: bool = False,
    force_sensor_clear: bool = True,
) -> GateResult:
    failures: list[str] = []

    if not _in_workspace(target_pose):
        failures.append(f"Target pose {target_pose} outside allowed workspace")

    if destination_pose is not None and not _in_workspace(destination_pose):
        failures.append(f"Destination pose {destination_pose} outside allowed workspace")

    if human_hand_detected:
        failures.append("Human hand detected near target — motion refused")

    if not operator_approved:
        failures.append("Operator approval not given")

    if not force_sensor_clear:
        failures.append("Force sensor not clear — unexpected contact before motion")

    return _report(failures)


def head_gate_check(
    target_deg: dict[str, float] | None,
    duration_s: float,
    operator_approved: bool = False,
    look_at: Pose3D | None = None,
    motion_enabled: bool | None = None,
    present_deg: dict[str, float] | None = None,
) -> GateResult:
    """Head (neck) moves only: limits, duration, speed, look_at box, approval and REACHY_ENABLE_MOTION.

    `target_deg` maps neck_roll / neck_pitch / neck_yaw to the goal in degrees (None when the
    angles could not be found before moving). `look_at` is the torso-frame point a look_at move
    aims at; presets have none. `motion_enabled` defaults to config.REACHY_ENABLE_MOTION.
    `present_deg` is the neck read-back taken before the move: the largest joint change divided by
    the duration may not exceed HEAD_MAX_AVG_SPEED_DEG_S. Without it (None, or any joint missing or
    not finite) the speed is unknown and the duration must be at least HEAD_UNKNOWN_START_MIN_DURATION_S.
    """
    failures: list[str] = []

    if look_at is not None and not _in_workspace(look_at, HEAD_LOOK_AT_BOX):
        failures.append(f"Look-at point {look_at} outside the head look-at box")

    if target_deg is None:
        failures.append("Neck angles unknown before the move — motion refused")
    else:
        for name in NECK_JOINTS:
            low, high = NECK_LIMITS_DEG[name]
            value = target_deg.get(name)
            if value is None or not math.isfinite(value) or not low <= value <= high:
                failures.append(f"{name} {value} deg outside the head limits [{low}, {high}]")

    if not math.isfinite(duration_s) or duration_s < HEAD_MIN_DURATION_S:
        failures.append(f"Duration {duration_s} s is shorter than {HEAD_MIN_DURATION_S} s")
    elif (target_deg is not None and (target_deg.get("neck_pitch") or 0) > HEAD_STEEP_PITCH_DEG
          and duration_s < HEAD_STEEP_MIN_DURATION_S):
        failures.append(f"A move to pitch above {HEAD_STEEP_PITCH_DEG} deg needs at least "
                        f"{HEAD_STEEP_MIN_DURATION_S} s (got {duration_s} s)")

    if target_deg is not None and math.isfinite(duration_s) and duration_s >= HEAD_MIN_DURATION_S:
        change = _largest_change(target_deg, present_deg)
        if change is None:
            if duration_s < HEAD_UNKNOWN_START_MIN_DURATION_S:
                failures.append(f"Present neck angles unknown: duration {duration_s} s is shorter than the "
                                f"{HEAD_UNKNOWN_START_MIN_DURATION_S} s minimum without them")
        elif change / duration_s > HEAD_MAX_AVG_SPEED_DEG_S:
            failures.append(f"Average neck speed {change / duration_s:.1f} deg/s ({change:.1f} deg in "
                            f"{duration_s} s) exceeds {HEAD_MAX_AVG_SPEED_DEG_S} deg/s")

    if not operator_approved:
        failures.append("Operator approval not given")

    if not (config.REACHY_ENABLE_MOTION if motion_enabled is None else motion_enabled):
        failures.append(MOTION_DISABLED)

    return _report(failures)


def gripper_gate_check(
    target_deg: float | None,
    speed_limit: float | None,
    torque_limit: float | None,
    operator_approved: bool = False,
    motion_enabled: bool | None = None,
    present_deg: float | None = None,
) -> GateResult:
    """Gripper-only moves (#27 Part A): travel limits, the step size, low speed and torque, approval and
    REACHY_ENABLE_MOTION. Owner decision 2026-10-22: every move needs a confirm and passes this gate.

    `target_deg` is the r_gripper goal in degrees; it must lie inside config.GRIPPER_LIMITS_DEG.
    `speed_limit` (deg/s) and `torque_limit` (percent) must be above zero and no higher than the caps.
    `present_deg` is the read-back taken before the move: |target - present| may not exceed
    GRIPPER_MAX_STEP_DEG (plus GRIPPER_STEP_MARGIN_DEG of read-back noise). Without it (None, or not finite)
    the step size is unknown and the move is refused.
    """
    failures: list[str] = []
    low, high = GRIPPER_LIMITS_DEG

    if not _finite(target_deg) or not low <= target_deg <= high:
        failures.append(f"Gripper target {target_deg} deg outside the gripper limits [{low}, {high}]")

    if not _finite(present_deg):
        failures.append("Present gripper position unknown — step size cannot be checked, motion refused")
    elif (_finite(target_deg)
          and abs(target_deg - present_deg) > GRIPPER_MAX_STEP_DEG + GRIPPER_STEP_MARGIN_DEG):
        failures.append(f"Gripper step {abs(target_deg - present_deg):.1f} deg ({present_deg:.1f} -> {target_deg}) "
                        f"exceeds {GRIPPER_MAX_STEP_DEG} deg")

    if not _finite(speed_limit) or not 0 < speed_limit <= GRIPPER_MAX_SPEED_DEG_S:
        failures.append(f"Gripper speed limit {speed_limit} deg/s outside (0, {GRIPPER_MAX_SPEED_DEG_S}]")

    if not _finite(torque_limit) or not 0 < torque_limit <= GRIPPER_MAX_TORQUE_PCT:
        failures.append(f"Gripper torque limit {torque_limit} % outside (0, {GRIPPER_MAX_TORQUE_PCT}]")

    if not operator_approved:
        failures.append("Operator approval not given")

    if not (config.REACHY_ENABLE_MOTION if motion_enabled is None else motion_enabled):
        failures.append(MOTION_DISABLED)

    return _report(failures)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _largest_change(target_deg: dict[str, float], present_deg: dict[str, float] | None) -> float | None:
    """Largest neck-joint change from the present to the target angles; None if either is unusable."""
    if present_deg is None:
        return None
    changes = []
    for name in NECK_JOINTS:
        goal, present = target_deg.get(name), present_deg.get(name)
        if goal is None or present is None or not math.isfinite(goal) or not math.isfinite(present):
            return None
        changes.append(abs(goal - present))
    return max(changes)


def _report(failures: list[str]) -> GateResult:
    if failures:
        for f in failures:
            logger.warning("Safety gate FAIL: %s", f)
    else:
        logger.info("Safety gate PASS")
    return GateResult(passed=len(failures) == 0, failures=failures)


def _in_workspace(pose: Pose3D, bounds=WORKSPACE) -> bool:
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    return (
        x_min <= pose.x <= x_max
        and y_min <= pose.y <= y_max
        and z_min <= pose.z <= z_max
    )
