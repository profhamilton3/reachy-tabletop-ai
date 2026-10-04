"""Focused checks for box mapping and binding results to immutable captures."""

import time

import numpy as np
from fastapi.testclient import TestClient

from reachy_ai.perception.pair_detector import DetectorUnavailable, decode_predictions
from reachy_ai.stereo import StereoService, StereoSettings
from reachy_ai.stereo_app import create_app


def prepared_client(monkeypatch, detector):
    service = StereoService(StereoSettings())
    monkeypatch.setattr(service, "start", lambda: None)
    monkeypatch.setattr(service, "stop", lambda: None)
    service._new_session()
    for eye in ("left", "right"):
        service._accept({"kind": "frame", "eye": eye, "sequence": 1,
                         "observed_ns": time.monotonic_ns(),
                         "pixels": np.zeros((48, 64, 3), dtype=np.uint8)})
    return service, TestClient(create_app(service.settings, service, detector=detector))


class ExampleDetector:
    def status(self):
        return {"available": True}

    def analyze(self, pixels):
        self.seen = pixels
        return {"model": {"name": "test", "sha256": "test-only"}, "elapsed_ms": 1,
                "detections": [{"class_id": 0, "label": "cube", "confidence": 0.9,
                                "box_xywh": [10, 10, 20, 20]}]}


def test_box_mapping_and_class_aware_nms():
    rows = np.array([[0.5, 0.5, 0.4, 0.4, 0.9, 0.01, 0.01],
                     [0.5, 0.5, 0.4, 0.4, 0.8, 0.01, 0.01],
                     [0.5, 0.5, 0.4, 0.4, 0.01, 0.85, 0.01]], dtype=np.float32)
    result = decode_predictions(rows.T[None], ["cube", "cylinder", "empty"], 640, 480)
    assert [r["label"] for r in result] == ["cube", "cylinder"]
    assert result[0]["box_xywh"] == [192, 144, 256, 192]
    assert decode_predictions(np.zeros((1, 7, 1)), ["cube", "cylinder", "empty"], 640, 480) == []


def test_analysis_retains_captured_pixels_and_raw_view(monkeypatch):
    detector = ExampleDetector()
    service, client = prepared_client(monkeypatch, detector)
    with client:
        pair = client.post("/api/stereo/pairs").json()
        raw_before = client.get(pair["eyes"]["right"]["url"]).content
        # New live source/session must not be substituted for this capture.
        original = service.frames["right"]
        service._new_session()
        response = client.post(f"/api/stereo/pairs/{pair['pair_id']}/analysis")
        assert response.status_code == 200
        result = response.json()
        assert result["frame_id"] == original.frame_id
        assert result["session_id"] == pair["session_id"]
        assert detector.seen is original.pixels
        assert result["camera_context"]["calibration_id"] == pair["camera_context"]["calibration_id"]
        assert client.get(result["overlay_url"]).status_code == 200
        assert client.get(result["raw_url"]).content == raw_before
        assert not np.any(original.pixels)
        assert client.post(f"/api/stereo/pairs/{pair['pair_id']}/analysis").json() == result


def test_expired_during_analysis_and_unavailable_detector_keep_raw_safe(monkeypatch):
    detector = ExampleDetector()
    service, client = prepared_client(monkeypatch, detector)
    with client:
        pair = client.post("/api/stereo/pairs").json()

        def unavailable(pixels):
            raise DetectorUnavailable("Install the analysis runtime")

        monkeypatch.setattr(detector, "analyze", unavailable)
        assert client.post(f"/api/stereo/pairs/{pair['pair_id']}/analysis").status_code == 503
        assert client.get(pair["eyes"]["right"]["url"]).status_code == 200

        def evict_while_running(pixels):
            for _ in range(4):
                service.capture()
            return {"detections": []}

        monkeypatch.setattr(detector, "analyze", evict_while_running)
        assert client.post(f"/api/stereo/pairs/{pair['pair_id']}/analysis").status_code == 410
        assert all("analysis" not in record for record in service.snapshots.values())
