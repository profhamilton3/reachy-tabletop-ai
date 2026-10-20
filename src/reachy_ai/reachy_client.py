"""
Wraps reachy_sdk (v1) for Reachy 1.2.
Dry-run by default — set REACHY_ENABLE_MOTION=true to allow motion.
"""
import logging
from typing import Any

from reachy_ai.config import HEAD_DEFAULT_DURATION_S, REACHY_ENABLE_MOTION, REACHY_IP
from reachy_ai.motion import primitives

logger = logging.getLogger(__name__)


class ReachyClient:
    def __init__(self, host: str = REACHY_IP, dry_run: bool = not REACHY_ENABLE_MOTION):
        self.host = host
        self.dry_run = dry_run
        self._reachy: Any = None
        self.command_poll_fixed = False

        if not dry_run:
            from reachy_sdk import ReachySDK  # type: ignore[import]

            from reachy_ai.sdk_compat import fix_command_poll

            # Before the SDK is constructed: 0.7.0 leaks pending waits on every command push (#23).
            self.command_poll_fixed = fix_command_poll()
            self._reachy = ReachySDK(host=host)
            logger.info("Connected to Reachy 1.2; reachy_sdk command-poll fix applied=%s", self.command_poll_fixed)
        else:
            logger.info("DRY-RUN mode — no robot connection (REACHY_ENABLE_MOTION not set)")

    @property
    def connected(self) -> bool:
        return self._reachy is not None

    def get_joint_positions(self) -> dict[str, float]:
        if self.dry_run:
            logger.debug("DRY-RUN get_joint_positions")
            return {}
        return {name: j.present_position for name, j in self._reachy.r_arm.joints.items()}

    def set_joint_goal(self, joint_name: str, goal_deg: float, duration: float = 1.0) -> None:
        if self.dry_run:
            logger.info("DRY-RUN set_joint_goal: %s -> %.1f deg over %.1fs", joint_name, goal_deg, duration)
            return
        joint = getattr(self._reachy.r_arm, joint_name)
        joint.goal_position = goal_deg

    def open_gripper(self) -> None:
        if self.dry_run:
            logger.info("DRY-RUN open_gripper")
            return
        self._reachy.r_arm.gripper.open()

    def close_gripper(self) -> None:
        if self.dry_run:
            logger.info("DRY-RUN close_gripper")
            return
        self._reachy.r_arm.gripper.close()

    def look_at(self, x: float, y: float, z: float, duration: float = HEAD_DEFAULT_DURATION_S,
                operator_approved: bool = False) -> dict:
        """Look at (x, y, z), SDK torso frame, through motion.primitives and the head gate.

        There is no other head path: the angles come from the SDK IK before moving, and the move is
        refused, or only logged, unless the gate passes and this client is not dry-run."""
        return primitives.look_at_point(self._reachy, x, y, z, duration,
                                        operator_approved=operator_approved, allow_live=not self.dry_run)

    def turn_on_arm(self) -> None:
        if self.dry_run:
            logger.info("DRY-RUN turn_on_arm")
            return
        self._reachy.turn_on("r_arm")

    def turn_off_arm(self) -> None:
        if self.dry_run:
            logger.info("DRY-RUN turn_off_arm")
            return
        self._reachy.turn_off("r_arm")
