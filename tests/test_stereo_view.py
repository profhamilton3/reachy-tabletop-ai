"""Small offline checks for the new viewer; no robot or motion calls."""

import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from reachy_ai.stereo import StereoService, StereoSettings
from reachy_ai.stereo_app import create_app


def feed(service, eye, sequence=1, age=0, color=(0, 0, 255)):
    pixels = np.full((48, 64, 3), color, dtype=np.uint8)
    service._accept({"kind": "frame", "eye": eye, "sequence": sequence,
                     "observed_ns": time.monotonic_ns() - int(age * 1e9), "pixels": pixels})


def test_pair_retains_images_and_identity_across_new_frames_and_reconnect():
    service = StereoService(StereoSettings())
    service._new_session()
    feed(service, "left")
    feed(service, "right", color=(255, 0, 0))
    pair = service.capture()
    original = service.snapshots[pair["pair_id"]]["frames"]["left"]
    feed(service, "left", sequence=2, color=(0, 255, 0))
    service._new_session()
    assert service.session != pair["session_id"]
    assert original.pixels[0, 0].tolist() == [0, 0, 255]
    assert not original.pixels.flags.writeable
    assert service.snapshots[pair["pair_id"]]["frames"]["left"] is original
    assert service.status()["can_capture"] is False


def test_stale_or_skewed_eye_blocks_capture_and_healthy_eye_remains_available():
    service = StereoService(StereoSettings())
    service._new_session()
    feed(service, "left", age=3)
    feed(service, "right")
    status = service.status()
    assert status["eyes"]["left"]["state"] == "stale"
    assert status["eyes"]["right"]["state"] == "live"
    assert not status["can_capture"]
    feed(service, "left", sequence=2, age=0.5)
    assert service.status()["eyes"]["left"]["state"] == "live"
    assert not service.status()["can_capture"]


def test_web_snapshot_expiry_color_and_read_only_routes(monkeypatch):
    service = StereoService(StereoSettings())
    lifecycle = []
    monkeypatch.setattr(service, "start", lambda: lifecycle.append("start"))
    monkeypatch.setattr(service, "stop", lambda: lifecycle.append("stop"))
    with TestClient(create_app(service.settings, service)) as client:
        assert "LEFT CAMERA" in client.get("/").text
        assert client.post("/api/stereo/pairs").status_code == 409
        service._new_session()
        feed(service, "left")
        feed(service, "right")
        pair = client.post("/api/stereo/pairs").json()
        image = client.get(pair["eyes"]["left"]["url"])
        decoded = cv2.imdecode(np.frombuffer(image.content, np.uint8), cv2.IMREAD_COLOR)
        assert decoded[0, 0, 2] > 245  # Red stays red through BGR -> JPEG.
        assert decoded[0, 0, 0] < 5
        for _ in range(4):
            service.capture()
        assert client.get(pair["eyes"]["left"]["url"]).status_code == 410
        assert client.get('/api/stereo/pairs/' + pair["pair_id"]).status_code == 410
        assert len(service.snapshots) == 4
        assert client.get("/stream/other").status_code == 422
        assert client.post("/tasks").status_code == 404
        assert client.get("/api/stereo/status").json()["motion_controls"] is False
    assert lifecycle == ["start", "stop"]
