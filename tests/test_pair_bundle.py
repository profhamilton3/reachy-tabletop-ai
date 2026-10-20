"""One-click bundle: contents per state, expiry, size bound, and nothing written server-side."""

import builtins
import hashlib
import io
import json
import time
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from reachy_ai import pair_bundle
from reachy_ai.stereo import StereoService, StereoSettings
from reachy_ai.stereo_app import create_app

BOXES = [{"label": "cube", "class_id": 0, "box_xywh": [20, 10, 20, 20], "confidence": 0.9}]


class Detector:
    def status(self):
        return {"available": True}

    def analyze(self, pixels):
        return {"detections": BOXES, "model": {"name": "fixture", "version": "1"}}


class Depth:
    def status(self):
        return {"available": True, "profile": {"id": "cal-123"}}

    def analyze(self, left, right, detections, head=None):
        return {"profile": {"id": "cal-123"}, "measurements": []}, b"\x89PNG fixture"


@pytest.fixture
def client_and_service(monkeypatch):
    service = StereoService(StereoSettings())
    monkeypatch.setattr(service, "start", lambda: None)
    monkeypatch.setattr(service, "stop", lambda: None)
    service._new_session()
    rng = np.random.default_rng(1)
    for eye in ("left", "right"):
        service._accept({"kind": "frame", "eye": eye, "sequence": 1, "observed_ns": time.monotonic_ns(),
                         "pixels": rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)})
    app = create_app(service.settings, service, detector=Detector(), depth_analyzer=Depth())
    with TestClient(app) as client:
        yield client, service


def open_zip(response):
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "attachment" in response.headers["content-disposition"]
    return zipfile.ZipFile(io.BytesIO(response.content))


def check_manifest(archive, expected):
    assert sorted(archive.namelist()) == sorted(expected + ["manifest.json"])
    manifest = json.loads(archive.read("manifest.json"))
    assert [f["name"] for f in manifest["files"]] == expected
    for entry in manifest["files"]:
        data = archive.read(entry["name"])
        assert entry["sha256"] == hashlib.sha256(data).hexdigest() and entry["bytes"] == len(data)
    return manifest


def test_bundle_grows_with_analysis_and_depth(client_and_service):
    client, _ = client_and_service
    pair = client.post("/api/stereo/pairs").json()
    url = f"/api/stereo/pairs/{pair['pair_id']}"

    archive = open_zip(client.get(url + "/bundle.zip"))
    manifest = check_manifest(archive, ["pair.json", "left.jpg", "right.jpg"])
    assert json.loads(archive.read("pair.json"))["pair_id"] == pair["pair_id"]
    assert archive.read("left.jpg") == client.get(url + "/left.jpg").content
    assert archive.read("right.jpg") == client.get(url + "/right.jpg").content
    assert manifest["calibration_id"] == "cal-123"
    assert set(manifest["versions"]) == {"viewer", "sdk", "detector_model"}

    client.post(url + "/analysis")
    archive = open_zip(client.get(url + "/bundle.zip"))
    manifest = check_manifest(archive, ["pair.json", "left.jpg", "right.jpg", "analysis.json", "analysis_overlay.jpg"])
    assert archive.read("analysis_overlay.jpg") == client.get(url + "/analysis/image.jpg").content
    assert manifest["versions"]["detector_model"] == {"name": "fixture", "version": "1"}

    client.post(url + "/depth")
    archive = open_zip(client.get(url + "/bundle.zip"))
    check_manifest(archive, ["pair.json", "left.jpg", "right.jpg", "analysis.json", "analysis_overlay.jpg",
                             "depth.json", "depth.png"])
    assert archive.read("depth.png") == client.get(url + "/depth/image.png").content
    assert client.get(url + "/bundle.zip").content == client.get(url + "/bundle.zip").content


def test_expired_or_unknown_pair_is_410(client_and_service):
    client, service = client_and_service
    pair = client.post("/api/stereo/pairs").json()
    for _ in range(4):
        service.capture()
    assert client.get(f"/api/stereo/pairs/{pair['pair_id']}/bundle.zip").status_code == 410
    assert client.get("/api/stereo/pairs/nope/bundle.zip").status_code == 410


def test_oversized_pair_is_413(client_and_service, monkeypatch):
    client, _ = client_and_service
    pair = client.post("/api/stereo/pairs").json()
    monkeypatch.setattr(pair_bundle, "MAX_BUNDLE_BYTES", 100)
    assert client.get(f"/api/stereo/pairs/{pair['pair_id']}/bundle.zip").status_code == 413


def test_bundle_writes_nothing_on_the_server(client_and_service, monkeypatch, tmp_path):
    client, _ = client_and_service
    pair = client.post("/api/stereo/pairs").json()
    url = f"/api/stereo/pairs/{pair['pair_id']}"
    client.post(url + "/analysis")
    client.post(url + "/depth")
    monkeypatch.chdir(tmp_path)
    real_open = builtins.open

    def read_only(file, mode="r", *args, **kwargs):
        if any(c in mode for c in "wax+"):
            raise AssertionError(f"server tried to write {file!r}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", read_only)
    monkeypatch.setattr(zipfile.ZipFile, "__init__", _memory_only(zipfile.ZipFile.__init__))
    assert client.get(url + "/bundle.zip").status_code == 200
    assert list(tmp_path.iterdir()) == []


def _memory_only(original):
    def init(self, file, *args, **kwargs):
        assert isinstance(file, io.BytesIO), "zip must be built in memory"
        original(self, file, *args, **kwargs)
    return init


def test_page_offers_the_save_button(client_and_service):
    client, _ = client_and_service
    page = client.get("/").text
    assert 'id="save-zip"' in page and "Save pair (zip)" in page and "/bundle.zip" in page
