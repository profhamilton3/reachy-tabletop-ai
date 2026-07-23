#!/usr/bin/env python3
"""
Smoke test all hardware channels on the physical Reachy 1.2.
Run with: python scripts/smoke_test_all.py
Requires: REACHY_IP set in environment.
"""
import os
import sys


def check(label: str, fn) -> bool:
    try:
        fn()
        print(f"  PASS  {label}")
        return True
    except Exception as e:
        print(f"  FAIL  {label}: {e}")
        return False


def main():
    ip = os.environ.get("REACHY_IP", "")
    if not ip:
        print("ERROR: REACHY_IP not set")
        sys.exit(1)

    print(f"\nReachy 1.2 Smoke Test — {ip}\n")
    results = []

    # SDK connection
    def test_sdk():
        from reachy_sdk import ReachySDK  # type: ignore[import]
        r = ReachySDK(host=ip)
        assert r is not None

    results.append(check("reachy_sdk connection", test_sdk))

    # Arm joints
    def test_arm():
        from reachy_sdk import ReachySDK  # type: ignore[import]
        r = ReachySDK(host=ip)
        joints = r.r_arm.joints
        assert len(joints) > 0

    results.append(check("Right arm joints readable", test_arm))

    # Head
    def test_head():
        from reachy_sdk import ReachySDK  # type: ignore[import]
        r = ReachySDK(host=ip)
        _ = r.head.joints

    results.append(check("Head joints readable", test_head))

    # Cameras
    def test_cameras():
        import cv2
        for idx in [0, 2]:
            cap = cv2.VideoCapture(idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"         Camera {idx}: {frame.shape}")
                return
        raise RuntimeError("No camera frame captured on indices 0 or 2")

    results.append(check("Camera capture", test_cameras))

    # Google Coral
    def test_coral():
        from pycoral.utils import edgetpu  # type: ignore[import]
        tpus = edgetpu.list_edge_tpus()
        assert len(tpus) > 0, f"No Coral TPU found: {tpus}"

    results.append(check("Google Coral TPU", test_coral))

    # ReSpeaker
    def test_audio():
        import pyaudio  # type: ignore[import]
        pa = pyaudio.PyAudio()
        found = False
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if "respeaker" in info["name"].lower() or info["maxInputChannels"] >= 4:
                print(f"         Audio device {i}: {info['name']} ({info['maxInputChannels']}ch)")
                found = True
                break
        pa.terminate()
        assert found, "ReSpeaker not found"

    results.append(check("ReSpeaker mic array", test_audio))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} checks passed\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
