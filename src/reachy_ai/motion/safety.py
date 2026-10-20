"""Safety gate — all checks must pass before any motion is allowed."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from reachy_ai import config
from reachy_ai.config import (
    HEAD_LOOK_AT_X_MAX,
    HEAD_LOOK_AT_X_MIN,
    HEAD_LOOK_AT_Y_MAX,
    HEAD_LOOK_AT_Y_MIN,
    HEAD_LOOK_AT_Z_MAX,
    HEAD_LOOK_AT_Z_MIN,
    HEAD_MIN_DURATION_S,
    HEAD_STEEP_MIN_DURATION_S,
    HEAD_STEEP_PITCH_DEG,
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
) -> GateResult:
    """Head (neck) moves only: limits, duration, look_at box, approval and REACHY_ENABLE_MOTION.

    `target_deg` maps neck_roll / neck_pitch / neck_yaw to the goal in degrees (None when the
    angles could not be found before moving). `look_at` is the torso-frame point a look_at move
    aims at; presets have none. `motion_enabled` defaults to config.REACHY_ENABLE_MOTION.
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

    if not operator_approved:
        failures.append("Operator approval not given")

    if not (config.REACHY_ENABLE_MOTION if motion_enabled is None else motion_enabled):
        failures.append(MOTION_DISABLED)

    return _report(failures)


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
