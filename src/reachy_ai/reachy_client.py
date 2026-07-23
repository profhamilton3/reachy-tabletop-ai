"""
Wraps reachy_sdk (v1) for Reachy 1.2.
Dry-run by default — set REACHY_ENABLE_MOTION=true to allow motion.
"""
import logging
from typing import Any

from reachy_ai.config import REACHY_ENABLE_MOTION, REACHY_IP

logger = logging.getLogger(__name__)


class ReachyClient:
    def __init__(self, host: str = REACHY_IP, dry_run: bool = not REACHY_ENABLE_MOTION):
        self.host = host
        self.dry_run = dry_run
        self._reachy: Any = None

        if not dry_run:
            from reachy_sdk import ReachySDK  # type: ignore[import]
            self._reachy = ReachySDK(host=host)
            logger.info("Connected to Reachy 1.2 at %s", host)
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

    def look_at(self, x: float, y: float, z: float, duration: float = 1.0) -> None:
        if self.dry_run:
            logger.info("DRY-RUN look_at: (%.2f, %.2f, %.2f)", x, y, z)
            return
        self._reachy.head.look_at(x=x, y=y, z=z, duration=duration)

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
