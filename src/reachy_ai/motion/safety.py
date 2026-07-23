"""Safety gate — all checks must pass before any motion is allowed."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from reachy_ai.config import (
    WORKSPACE_X_MAX,
    WORKSPACE_X_MIN,
    WORKSPACE_Y_MAX,
    WORKSPACE_Y_MIN,
    WORKSPACE_Z_MAX,
    WORKSPACE_Z_MIN,
)

logger = logging.getLogger(__name__)


@dataclass
class Pose3D:
    x: float
    y: float
    z: float


@dataclass
class GateResult:
    passed: bool
    failures: list[str]


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

    if failures:
        for f in failures:
            logger.warning("Safety gate FAIL: %s", f)
    else:
        logger.info("Safety gate PASS")

    return GateResult(passed=len(failures) == 0, failures=failures)


def _in_workspace(pose: Pose3D) -> bool:
    return (
        WORKSPACE_X_MIN <= pose.x <= WORKSPACE_X_MAX
        and WORKSPACE_Y_MIN <= pose.y <= WORKSPACE_Y_MAX
        and WORKSPACE_Z_MIN <= pose.z <= WORKSPACE_Z_MAX
    )
