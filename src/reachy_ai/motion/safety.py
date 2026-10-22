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
from reachy_ai.motion import arm_routes

logger = logging.getLogger(__name__)

#: The one head-gate failure that leaves a move as a logged dry run rather than a refusal.
MOTION_DISABLED = "REACHY_ENABLE_MOTION is not true"


class MotionRefused(PermissionError):
    """A live arm or gripper command was refused because no gate covers it yet; nothing was sent."""


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


def arm_gate_check(
    doc: dict | None,
    route_name: str,
    leg_index: int,
    record: dict | None,
    present_deg: dict[str, float] | None = None,
    temperatures_c: dict[str, float] | None = None,
    duration_s: float | None = None,
    max_goal_step_deg: float | None = None,
    operator_approved: bool = False,
    motion_enabled: bool | None = None,
    motion_lock_free: bool = False,
    stop_unresolved: bool = True,
    rig_check_done: bool = False,
    empty_confirmed: tuple[str, ...] | list[str] | set[str] = (),
    board_reads: dict[str, str] | None = None,
    fk_tip_m: tuple[float, float, float] | None = None,
    rest_tip_m: tuple[float, float, float] | None = None,
    hand_in_view: bool | None = None,
) -> GateResult:
    """One arm leg (#66, docs/design/arm-motion.md section 5, items 1-9). Plain values in, `GateResult` out.

    Every item can only add a failure: a pass needs all nine, and nothing a caller passes removes another item's
    failure. Defaults refuse. `MOTION_DISABLED` is the one failure that leaves a leg a logged dry run.

    `doc` is the validated route document (`arm_routes.load_routes_doc`), `route_name` and `leg_index` pick the
    flown leg (`arm_routes.expand_legs`: out, then back), `record` the clearance record the loader returned (or
    None). The rig ids and the purposes the owner decisions cover are read from `config`, not passed in.

    1 route allowed: in the file, a covered purpose, every pose taught (OD-A5).
    2 record: matches the route's hash, verdict pass, this rig and mount, and describes this leg.
    3 limits: the leg's poses and gripper inside ARM_LIMITS_DEG; `duration_s` no shorter than the speed rule asks;
      `max_goal_step_deg` (the planned largest per-goal change) within ARM_MAX_GOAL_STEP_DEG; `present_deg` (the
      seven arm joints) within ARM_START_TOL_DEG of the leg's start; `temperatures_c` (the seven) not over
      ARM_MAX_TEMP_C. Anything unknown refuses.
    4 `operator_approved`: a fresh confirm for this leg.
    5 `motion_enabled` (default config.REACHY_ENABLE_MOTION).
    6 `motion_lock_free` and not `stop_unresolved`: nothing else moves, no Stop or abort is unresolved.
    7 board: when any leg of the route comes within ARM_BOARD_CLEARANCE_CM of it in the record, `rig_check_done`,
      every `must_be_empty` cell in `empty_confirmed`, and no cell in `board_reads` ({cell: status}) read as
      anything but "empty". A read can only refuse; a missing read does not.
    8 FK check: `fk_tip_m` (forward_kinematics() of the REST read-back) within ARM_FK_TOL_M of `rest_tip_m`.
    9 `hand_in_view`: None = not checked (not "allowed"), False = clear, anything else refuses.
    """
    failures: list[str] = []
    leg = None
    route = (doc or {}).get("routes", {}).get(route_name) if isinstance(doc, dict) else None

    # 1. Route allowed
    if route is None:
        failures.append(f"Route {route_name!r} is not in the route file")
    else:
        if route.get("purpose") not in config.ARM_ALLOWED_PURPOSES:
            failures.append(f"Route purpose {route.get('purpose')!r} is not covered by an owner decision "
                            f"(allowed: {list(config.ARM_ALLOWED_PURPOSES)})")
        taught, why = arm_routes.pose_rule(doc, route_name)
        if not taught:
            failures.append(why)
        try:
            legs = arm_routes.expand_legs(doc, route_name)
            leg = legs[leg_index] if isinstance(leg_index, int) and not isinstance(leg_index, bool) \
                and 0 <= leg_index < len(legs) else None
        except (KeyError, TypeError, ValueError):
            legs = []
        if leg is None:
            failures.append(f"Leg {leg_index!r} is not a leg of route {route_name!r}")

    # 2. Record valid
    record_leg = None
    if not isinstance(record, dict):
        failures.append("No clearance record for this route")
    else:
        try:
            current = arm_routes.route_hashes(doc)[route_name]
        except (KeyError, TypeError, ValueError):
            current = None
        if record.get("route") != route_name or current is None or record.get("route_hash") != current:
            failures.append("The record does not match the route's current hash")
        if record.get("verdict") != "pass":
            failures.append(f"Record verdict is {record.get('verdict')!r}, not 'pass'")
        rig = record.get("rig") if isinstance(record.get("rig"), dict) else {}
        if rig.get("rig_layout_id") != config.ARM_RIG_LAYOUT_ID or rig.get("mount_id") != config.ARM_MOUNT_ID:
            failures.append(f"Record is for rig {rig.get('rig_layout_id')!r} / mount {rig.get('mount_id')!r}, "
                            f"not {config.ARM_RIG_LAYOUT_ID!r} / {config.ARM_MOUNT_ID!r}")
        rows = record.get("legs") if isinstance(record.get("legs"), list) else []
        if leg is not None:
            record_leg = rows[leg.index] if leg.index < len(rows) and isinstance(rows[leg.index], dict) else None
            if (record_leg is None or (record_leg.get("from"), record_leg.get("to"), record_leg.get("direction"))
                    != (leg.start, leg.to, leg.direction)):
                failures.append("The record has no row for this leg")
                record_leg = None

    # 3. Limits, speed, step, start, temperature
    if leg is not None:
        for pose, q in ((leg.start, leg.q_from), (leg.to, leg.q_to)):
            for joint in arm_routes.ARM_JOINTS:
                low, high = arm_routes.ARM_LIMITS_DEG[joint]
                if not _finite(q.get(joint)) or not low <= q[joint] <= high:
                    failures.append(f"Waypoint {pose}: {joint} {q.get(joint)} deg outside the arm limits "
                                    f"[{low}, {high}]")
        low, high = arm_routes.ARM_LIMITS_DEG[arm_routes.GRIPPER_JOINT]
        if not _finite(leg.gripper_deg) or not low <= leg.gripper_deg <= high:
            failures.append(f"Leg gripper {leg.gripper_deg} deg outside the gripper limits [{low}, {high}]")
        if not _finite(duration_s) or duration_s <= 0:
            failures.append(f"Leg duration {duration_s} s is not a positive number")
        else:
            try:
                change = max(abs(leg.q_to[j] - leg.q_from[j]) for j in arm_routes.ARM_JOINTS)
            except (KeyError, TypeError):
                change = float("inf")
            if change / duration_s > config.ARM_MAX_AVG_SPEED_DEG_S + 1e-9:
                failures.append(f"Average arm speed {change / duration_s:.1f} deg/s ({change:.1f} deg in "
                                f"{duration_s} s) exceeds {config.ARM_MAX_AVG_SPEED_DEG_S} deg/s")
            if duration_s + 1e-9 < leg.min_duration_s:
                failures.append(f"Leg duration {duration_s} s is shorter than the route's {leg.min_duration_s} s")
        if not _finite(max_goal_step_deg):
            failures.append("Largest streamed goal step unknown — step size cannot be checked")
        elif max_goal_step_deg > config.ARM_MAX_GOAL_STEP_DEG + 1e-9:
            failures.append(f"A streamed goal moves a joint {max_goal_step_deg:.2f} deg, over "
                            f"{config.ARM_MAX_GOAL_STEP_DEG} deg")
        for joint in arm_routes.ARM_JOINTS:
            present = (present_deg or {}).get(joint)
            if not _finite(present):
                failures.append(f"Present {joint} unknown — the start cannot be checked")
            elif _finite(leg.q_from.get(joint)) and abs(present - leg.q_from[joint]) > config.ARM_START_TOL_DEG:
                failures.append(f"{joint} is {abs(present - leg.q_from[joint]):.1f} deg from {leg.start}'s "
                                f"{leg.q_from[joint]}, over {config.ARM_START_TOL_DEG} deg")
    for joint in arm_routes.ARM_JOINTS:
        temperature = (temperatures_c or {}).get(joint)
        if not _finite(temperature):
            failures.append(f"Temperature of {joint} unknown — motion refused")
        elif temperature > config.ARM_MAX_TEMP_C:
            failures.append(f"{joint} is {temperature:.0f} C, over {config.ARM_MAX_TEMP_C:.0f} C")

    # 4. Approval
    if operator_approved is not True:
        failures.append("Operator approval not given")

    # 5. Motion disabled (a dry run, not a refusal)
    if not (config.REACHY_ENABLE_MOTION if motion_enabled is None else motion_enabled):
        failures.append(MOTION_DISABLED)

    # 6. Something else running
    if motion_lock_free is not True:
        failures.append("Another command holds the motion lock")
    if stop_unresolved is not False:
        failures.append("A head, gripper or arm Stop or abort is unresolved")

    # 7. The board, where the route touches it
    rows = record.get("legs") if isinstance(record, dict) and isinstance(record.get("legs"), list) else []
    touches = (not rows) or any(not isinstance(r, dict) or not _finite(((r.get("kinematic") or {}).get("board_cm")))
                                or r["kinematic"]["board_cm"] < config.ARM_BOARD_CLEARANCE_CM for r in rows)
    if touches:
        if rig_check_done is not True:
            failures.append("No rig check this session — the route comes within "
                            f"{config.ARM_BOARD_CLEARANCE_CM:g} cm of the board")
        occupancy = record.get("occupancy") if isinstance(record, dict) and isinstance(record.get("occupancy"), dict) \
            else None
        if occupancy is None:
            failures.append("The record has no occupancy map — the cells to keep empty are unknown")
        else:
            confirmed = set(empty_confirmed or ())
            for cell in sorted(c for c, status in occupancy.items() if status == "must_be_empty"):
                if cell not in confirmed:
                    failures.append(f"Cell {cell} must be empty and the operator has not confirmed it")
        for cell, status in sorted((board_reads or {}).items()):
            if status != "empty" and isinstance(occupancy, dict) and occupancy.get(cell) == "must_be_empty":
                failures.append(f"Board read: cell {cell} is {status}, and it must be empty")

    # 8. Arm FK check
    if not _vec3(fk_tip_m) or not _vec3(rest_tip_m):
        failures.append("Arm FK check not done: the REST tip (read-back FK or the record's) is unknown")
    else:
        gap = math.dist(fk_tip_m, rest_tip_m)
        if gap > config.ARM_FK_TOL_M:
            failures.append(f"Arm FK tip is {gap * 100:.1f} cm from the record's REST tip, over "
                            f"{config.ARM_FK_TOL_M * 100:.0f} cm")

    # 9. Hand-freeze hook (None: not checked, which is not "allowed")
    if hand_in_view is not None and hand_in_view is not False:
        failures.append("A hand is in view — arm motion refused")

    return _report(failures)


def _vec3(value) -> bool:
    return isinstance(value, (tuple, list)) and len(value) == 3 and all(_finite(v) for v in value)


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
