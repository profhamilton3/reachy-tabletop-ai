#!/usr/bin/env python3
"""
Full pipeline dry-run — no robot connection, no network calls.
Tests that all modules import and wire together without errors.
"""
import logging
import sys

from reachy_ai.motion.safety import Pose3D, gate_check
from reachy_ai.planner.schema import Plan
from reachy_ai.reachy_client import ReachyClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")


def main():
    print("\n=== Reachy 1.2 Dry-Run Demo ===\n")

    client = ReachyClient(dry_run=True)
    print(f"Client mode: {'DRY-RUN' if client.dry_run else 'LIVE'}")

    # Simulate a planner response
    plan = Plan(
        task_type="pick_place",
        target_id="obj_2",
        target_description="whiteboard eraser",
        destination="left_tray",
        confidence=0.88,
        requires_confirmation=False,
        safety_notes=["object appears light", "inside reachable zone"],
        brief_reason="Sponge-like object matches erasing task.",
    )
    print(
        f"\nPlanner output:\n  target: {plan.target_description}"
        f"\n  action: {plan.task_type} → {plan.destination}"
        f"\n  confidence: {plan.confidence}"
    )

    # Safety gate
    target_pose = Pose3D(x=0.5, y=0.05, z=0.08)
    result = gate_check(
        target_pose=target_pose,
        operator_approved=True,
    )
    print(f"\nSafety gate: {'PASS' if result.passed else 'FAIL'}")
    if not result.passed:
        for f in result.failures:
            print(f"  - {f}")
        sys.exit(1)

    # Motion (dry-run)
    print("\nMotion primitives (dry-run):")
    client.look_at(x=1, y=0, z=-0.3)
    client.open_gripper()
    client.turn_on_arm()
    client.close_gripper()
    client.turn_off_arm()

    print("\n=== Dry-run complete ===")


if __name__ == "__main__":
    main()
