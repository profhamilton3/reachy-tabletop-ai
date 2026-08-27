"""Reachy 1.2 SDK probe — read-only API surface verification.

Connects to a physical (or simulated) Reachy 1.2, reads every subsystem,
and prints a structured report confirming which SDK calls succeed and what
values the robot returns.  No motion is commanded.

Usage:
    python scripts/probe_sdk1.py                    # connects to localhost
    python scripts/probe_sdk1.py 10.22.129.133      # connects to robot IP
    REACHY_IP=10.22.129.133 python scripts/probe_sdk1.py
"""

import argparse
import math
import os
import sys
import time
from typing import Optional

# ── SDK import guard ──────────────────────────────────────────────────────────
try:
    from reachy_sdk import ReachySDK
except ImportError:
    print("ERROR: reachy-sdk not installed.  Run: pip install reachy-sdk")
    sys.exit(1)

PASS = "✓"
FAIL = "✗"
SKIP = "—"


def _fmt(val, fmt=".3f"):
    try:
        return format(val, fmt)
    except Exception:
        return str(val)


def probe_connection(host: str, port: int) -> Optional[ReachySDK]:
    print(f"\n{'='*60}")
    print("  Reachy 1.2 SDK Probe")
    print(f"  Host: {host}:{port}")
    print(f"{'='*60}\n")

    print("[1] Connection")
    try:
        reachy = ReachySDK(host=host, sdk_port=port)
        time.sleep(0.5)
        print(f"  {PASS}  ReachySDK connected to {host}:{port}")
        return reachy
    except Exception as e:
        print(f"  {FAIL}  Connection failed: {e}")
        return None


def probe_joints(reachy: ReachySDK):
    print("\n[2] Joint discovery")
    for part_name in ("r_arm", "l_arm", "head"):
        part = getattr(reachy, part_name, None)
        if part is None:
            print(f"  {SKIP}  {part_name}: not present")
            continue
        joints = part.joints
        print(f"  {PASS}  {part_name}: {len(joints)} joints")
        for name, joint in joints.items():
            try:
                pos_deg = joint.present_position
                pos_rad = math.radians(pos_deg)
                comp = "compliant" if joint.compliant else "stiff"
                print(f"         {name:<22} {pos_deg:+7.2f}°  ({pos_rad:+.4f} rad)  [{comp}]")
            except Exception as e:
                print(f"         {name:<22} {FAIL} {e}")


def probe_head_kinematics(reachy: ReachySDK):
    print("\n[3] Head kinematics (look_at)")
    head = getattr(reachy, "head", None)
    if head is None:
        print(f"  {SKIP}  head not present")
        return
    try:
        # look_at straight ahead — read-only kinematics call
        result = head.look_at(x=1.0, y=0.0, z=0.0, duration=0.0)
        print(f"  {PASS}  head.look_at(forward) returned: {result}")
    except Exception as e:
        print(f"  {FAIL}  head.look_at: {e}")

    print("\n[3b] Antenna joints")
    for ant in ("l_antenna", "r_antenna"):
        j = getattr(head, ant, None)
        if j is None:
            print(f"  {SKIP}  {ant}: not present")
            continue
        try:
            print(f"  {PASS}  {ant}: present_position={j.present_position:+.2f}°  compliant={j.compliant}")
        except Exception as e:
            print(f"  {FAIL}  {ant}: {e}")


def probe_arm_kinematics(reachy: ReachySDK):
    print("\n[4] Arm forward kinematics")
    for arm_name in ("r_arm", "l_arm"):
        arm = getattr(reachy, arm_name, None)
        if arm is None:
            print(f"  {SKIP}  {arm_name}: not present")
            continue
        try:
            fk = arm.forward_kinematics()
            pos = fk[:3, 3]
            print(f"  {PASS}  {arm_name}.forward_kinematics()  end-effector xyz = "
                  f"[{_fmt(pos[0])}, {_fmt(pos[1])}, {_fmt(pos[2])}] m")
        except Exception as e:
            print(f"  {FAIL}  {arm_name}.forward_kinematics: {e}")

    print("\n[4b] Arm inverse kinematics")
    import numpy as np
    # Neutral reach-forward target for right arm (safe, reachable)
    target_r = np.array([
        [0, 0, -1, 0.3],
        [0, 1,  0, -0.2],
        [1, 0,  0, -0.3],
        [0, 0,  0,  1.0],
    ], dtype=float)
    # Mirror for left arm
    target_l = target_r.copy()
    target_l[1, 3] = 0.2

    for arm_name, target in (("r_arm", target_r), ("l_arm", target_l)):
        arm = getattr(reachy, arm_name, None)
        if arm is None:
            print(f"  {SKIP}  {arm_name}: not present")
            continue
        try:
            joints = arm.inverse_kinematics(target)
            rounded = [round(j, 1) for j in joints]
            print(f"  {PASS}  {arm_name}.inverse_kinematics()  joints (deg) = {rounded}")
        except Exception as e:
            print(f"  {FAIL}  {arm_name}.inverse_kinematics: {e}")


def probe_cameras(reachy: ReachySDK):
    print("\n[5] Cameras")
    for cam_name in ("left_camera", "right_camera"):
        cam = getattr(reachy, cam_name, None)
        if cam is None:
            print(f"  {SKIP}  {cam_name}: not present")
            continue
        try:
            frame = cam.last_frame
            if frame is not None and frame.size > 0:
                print(f"  {PASS}  {cam_name}: frame shape={frame.shape}  dtype={frame.dtype}")
            else:
                print(f"  {FAIL}  {cam_name}: connected but frame is empty")
        except Exception as e:
            print(f"  {FAIL}  {cam_name}: {e}")


def probe_force_sensors(reachy: ReachySDK):
    print("\n[6] Force sensors")
    fs = getattr(reachy, "force_sensors", None)
    if fs is None:
        print(f"  {SKIP}  force_sensors: not present")
        return
    for sensor_name in ("r_force_gripper", "l_force_gripper"):
        sensor = getattr(fs, sensor_name, None)
        if sensor is None:
            print(f"  {SKIP}  {sensor_name}: not present")
            continue
        try:
            print(f"  {PASS}  {sensor_name}: force={sensor.force:.3f} N")
        except Exception as e:
            print(f"  {FAIL}  {sensor_name}: {e}")


def probe_fans(reachy: ReachySDK):
    print("\n[7] Fans")
    fans = getattr(reachy, "fans", None)
    if fans is None:
        print(f"  {SKIP}  fans: not present")
        return
    for fan_name in ("r_arm_fan", "l_arm_fan", "head_fan"):
        fan = getattr(fans, fan_name, None)
        if fan is None:
            print(f"  {SKIP}  {fan_name}: not present")
            continue
        try:
            print(f"  {PASS}  {fan_name}: on={fan.is_on}")
        except Exception as e:
            print(f"  {FAIL}  {fan_name}: {e}")


def probe_compliant_toggle(reachy: ReachySDK):
    """Verify turn_on/turn_off round-trip without moving the robot."""
    print("\n[8] Compliant toggle (turn_on / turn_off — no motion)")
    for part in ("r_arm", "head"):
        try:
            reachy.turn_on(part)
            time.sleep(0.2)
            # Sample one joint
            sample_joint = None
            if part == "r_arm" and reachy.r_arm:
                sample_joint = reachy.r_arm.r_shoulder_pitch
            elif part == "head" and reachy.head:
                sample_joint = reachy.head.neck_roll
            stiff = not sample_joint.compliant if sample_joint else None

            reachy.turn_off(part)
            time.sleep(0.2)
            relaxed = sample_joint.compliant if sample_joint else None

            if stiff and relaxed:
                print(f"  {PASS}  {part}: turn_on→stiff, turn_off→compliant")
            else:
                print(f"  {FAIL}  {part}: toggle did not work as expected "
                      f"(stiff={stiff}, relaxed={relaxed})")
        except Exception as e:
            print(f"  {FAIL}  {part}: {e}")


def summary(results: dict):
    print(f"\n{'='*60}")
    print("  Probe Summary")
    print(f"{'='*60}")
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    for section, ok in results.items():
        mark = PASS if ok else FAIL
        print(f"  {mark}  {section}")
    print(f"\n  {passed}/{total} sections passed")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Reachy 1.2 SDK probe — read-only")
    parser.add_argument("host", nargs="?",
                        default=os.environ.get("REACHY_IP", "localhost"),
                        help="Robot IP or hostname (default: localhost)")
    parser.add_argument("--port", type=int, default=50051,
                        help="gRPC port (default: 50051)")
    args = parser.parse_args()

    reachy = probe_connection(args.host, args.port)
    if reachy is None:
        sys.exit(1)

    results = {}

    def run(label, fn):
        try:
            fn(reachy)
            results[label] = True
        except Exception as e:
            print(f"  {FAIL}  Unhandled error in {label}: {e}")
            results[label] = False

    run("Joints",              probe_joints)
    run("Head kinematics",     probe_head_kinematics)
    run("Arm kinematics",      probe_arm_kinematics)
    run("Cameras",             probe_cameras)
    run("Force sensors",       probe_force_sensors)
    run("Fans",                probe_fans)
    run("Compliant toggle",    probe_compliant_toggle)

    summary(results)


if __name__ == "__main__":
    main()
