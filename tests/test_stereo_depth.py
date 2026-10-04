"""Bounded geometry and pair-binding checks; no calibration or hardware experiments."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from reachy_ai.perception.stereo_depth import DepthUnavailable, PairDepth, camera_to_torso
from reachy_ai.stereo import StereoService, StereoSettings
from reachy_ai.stereo_app import create_app


def plane():
    depth = PairDepth(sim_lens="pinhole")
    g = depth.geometry
    g.size = (320, 240)
    g.K = [np.array([[400., 0, 160], [0, 400, 120], [0, 0, 1]]) for _ in range(2)]
    left = np.random.default_rng(4).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-16] = left[:, 16:]
    boxes = [{"label": "cube", "box_xywh": [130, 95, 40, 40]},
             {"label": "empty", "box_xywh": [130, 95, 40, 40]}]
    return depth, left, right, boxes


def test_known_disparity_metric_right_camera_and_urdf_transform():
    depth, left, right, boxes = plane()
    original = right.copy()
    result, image = depth.analyze(left, right, boxes)
    item = result["measurements"][0]
    assert item["status"] == "measured"
    assert item["forward_depth_m"] == pytest.approx(400 * .0725 / 16, abs=.005)
    # Right-camera X (not rectified left X, which differs by the baseline).
    assert item["xyz_camera_m"][0] == pytest.approx((149.5 - 160) * 1.8125 / 400, abs=.004)
    assert result["measurements"][1]["status"] == "unavailable"
    assert image.startswith(b"\x89PNG")
    np.testing.assert_array_equal(original, right)
    pose = {n: {"present_deg": 0} for n in ("neck_roll", "neck_pitch", "neck_yaw")}
    # A point on the optical axis is forward and below the unpitched head due to URDF pitch.
    point = camera_to_torso([0, 0, 1], pose)
    c, s = np.cos(.174), np.sin(.174)
    np.testing.assert_allclose(point, [c * 1.0033 + s * .061 + .015, -.03625,
                                      -s * 1.0033 + c * .061 + .095])


def test_textureless_and_wrong_shape_are_not_metric_positions():
    depth, left, right, boxes = plane()
    result, _ = depth.analyze(left * 0, right * 0, boxes)
    assert result["measurements"][0]["status"] == "unavailable"
    with pytest.raises(DepthUnavailable, match="size differs"):
        depth.analyze(left[:100], right[:100], boxes)


def test_depth_routes_cache_identity_expiry_and_raw_preservation(monkeypatch):
    depth, left, right, boxes = plane()
    class Detector:
        def status(self):
            return {"available": True}

        def analyze(self, pixels):
            return {"detections": boxes, "model": {"name": "fixture"}}

    service = StereoService(StereoSettings())
    monkeypatch.setattr(service, "start", lambda: None)
    monkeypatch.setattr(service, "stop", lambda: None)
    service._new_session()
    for eye, pixels in zip(("left", "right"), (left, right)):
        service._accept({"kind": "frame", "eye": eye, "sequence": 1,
                         "observed_ns": time.monotonic_ns(), "pixels": pixels})
    with TestClient(create_app(service.settings, service, detector=Detector(), depth_analyzer=depth)) as client:
        pair = client.post("/api/stereo/pairs").json()
        url = f"/api/stereo/pairs/{pair['pair_id']}"
        raw = client.get(url + "/right.jpg").content
        assert client.post(url + "/depth").status_code == 409
        analysis = client.post(url + "/analysis").json()
        result = client.post(url + "/depth").json()
        assert result["frame_ids"] == {e: pair["eyes"][e]["frame_id"] for e in ("left", "right")}
        assert result["analysis_id"] == analysis["analysis_id"]
        assert result["head_pose_used"] is None
        assert client.get(result["image_url"]).status_code == 200
        assert client.get(url + "/right.jpg").content == raw
        assert client.post(url + "/depth").json() == result
        assert "depth_image" not in client.get(url).json()
        pose = {n: {"present_deg": 0} for n in ("neck_roll", "neck_pitch", "neck_yaw")}
        observed = time.monotonic_ns()
        # Refresh the two frames so this checks capture association, not test duration.
        for eye, pixels in zip(("left", "right"), (left, right)):
            service._accept({"kind": "frame", "eye": eye, "sequence": 2,
                             "observed_ns": observed, "pixels": pixels})
        service._accept({"kind": "head", "observed_ns": observed, "values": pose})
        posed = client.post("/api/stereo/pairs").json()
        posed_url = f"/api/stereo/pairs/{posed['pair_id']}"
        client.post(posed_url + "/analysis")
        service.head["joints"]["neck_pitch"]["present_deg"] = 30
        measured = client.post(posed_url + "/depth").json()
        assert measured["head_pose_used"]["neck_pitch"]["present_deg"] == 0
        assert "xyz_torso_m" in measured["measurements"][0]
        other = client.post("/api/stereo/pairs").json()
        next_url = f"/api/stereo/pairs/{other['pair_id']}"
        client.post(next_url + "/analysis")
        analyze = depth.analyze
        def evict(*args):
            value = analyze(*args)
            service.snapshots.clear()
            return value
        monkeypatch.setattr(depth, "analyze", evict)
        assert client.post(next_url + "/depth").status_code == 410
        assert client.get(result["image_url"]).status_code == 410
