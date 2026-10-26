"""Zoom policy, the dry-run-by-default setter (fake client) and the viewer's zoom warning (#28 B). Offline."""

import logging
import time
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

from reachy_ai import camera_zoom
from reachy_ai.camera_zoom import SCENE_ZOOM, ZoomSetter, calibration_id, normalize_zoom, zoom_status
from reachy_ai.stereo import StereoService, StereoSettings
from reachy_ai.stereo_app import create_app


class FakeCamera:
    def __init__(self, level="INTER", fail=False, stuck=False):
        self.level, self.fail, self.stuck, self.commands = level, fail, stuck, []

    @property
    def zoom_level(self):
        return types.SimpleNamespace(name=self.level)

    @zoom_level.setter
    def zoom_level(self, value):
        self.commands.append(value)
        if self.fail:
            raise ValueError("Could not set new zoom level!")
        if not self.stuck:
            self.level = value


def fake_client(left="INTER", right="INTER", **kw):
    return types.SimpleNamespace(left_camera=FakeCamera(left, **kw), right_camera=FakeCamera(right, **kw))


def setter(client, **kw):
    return ZoomSetter(client, resolve=lambda name: name, sleep=lambda s: None, **kw)


def test_policy_is_the_widest_level_and_names_normalize():
    assert camera_zoom.ZOOM_ORDER[0] == SCENE_ZOOM == "OUT"
    assert normalize_zoom(types.SimpleNamespace(name="IN")) == "IN"
    assert normalize_zoom("inter") == "INTER"
    assert normalize_zoom("auto") is None and normalize_zoom(None) is None
    # ZERO is the SDK's unset level: a real read-back value, never mapped to another level, not a policy level.
    assert normalize_zoom("zero") == "ZERO" and normalize_zoom(types.SimpleNamespace(name="ZERO")) == "ZERO"
    assert "ZERO" not in camera_zoom.ZOOM_ORDER


def test_setter_is_dry_run_by_default_and_sends_nothing(caplog):
    client = fake_client()
    with caplog.at_level(logging.INFO, logger=camera_zoom.LOG.name):
        records = ZoomSetter(client).set_both("OUT")
    assert client.left_camera.commands == client.right_camera.commands == []
    assert all(r["dry_run"] and r["before"] == "INTER" and r["after"] == "INTER" and not r["ok"] for r in records)
    assert "DRY-RUN zoom left: INTER -> OUT (not sent)" in caplog.text


def test_live_setter_sends_a_logged_command_and_verifies_by_readback(caplog):
    client = fake_client()
    with caplog.at_level(logging.INFO, logger=camera_zoom.LOG.name):
        records = setter(client, dry_run=False).set_both("OUT")
    assert client.left_camera.commands == client.right_camera.commands == ["OUT"]
    assert all(r["ok"] and r["after"] == "OUT" for r in records)
    assert "ZOOM COMMAND left: INTER -> OUT" in caplog.text and "read back: OUT (ok)" in caplog.text


def test_live_setter_reports_a_failed_or_unconfirmed_command():
    failed = setter(fake_client(fail=True), dry_run=False).set_level("left", "OUT")
    assert failed["ok"] is False and failed["error"] == "ValueError"
    unconfirmed = setter(fake_client(stuck=True), dry_run=False).set_level("right", "OUT")
    assert unconfirmed["ok"] is False and unconfirmed["after"] == "INTER"


def test_setter_rejects_unknown_levels_and_eyes_without_commanding():
    client = fake_client()
    s = setter(client, dry_run=False)
    for bad in ("wide", ""):
        with pytest.raises(ValueError):
            s.set_level("left", bad)
    with pytest.raises(ValueError):
        s.set_level("middle", "OUT")
    assert client.left_camera.commands == []


def test_zoom_status_warns_on_unrecorded_mismatched_unreadable_and_non_widest():
    ok = zoom_status("OUT", {"left": "OUT", "right": "OUT"})
    assert ok["matches_profile"] is True and ok["warning"] is None and ok["captured"] == "OUT"
    unrecorded = zoom_status(None, {"left": {"zoom_level": "OUT"}, "right": {"zoom_level": "OUT"}})
    assert unrecorded["matches_profile"] is None and "never recorded" in unrecorded["warning"]
    mismatch = zoom_status("OUT", {"left": "IN", "right": "IN"})
    assert mismatch["matches_profile"] is False and "differs from the profile" in mismatch["warning"]
    assert "widest zoom" in mismatch["warning"]
    split = zoom_status("OUT", {"left": "OUT", "right": "IN"})
    assert split["captured"] is None and "differ" in split["warning"]
    unreadable = zoom_status("OUT", {"left": "auto", "right": None})
    assert unreadable["captured"] is None and "could not be read" in unreadable["warning"]


def test_zoom_is_folded_into_the_calibration_id():
    assert calibration_id("abc", "OUT") == "abc@zoom=OUT"
    assert calibration_id("abc", "IN") != calibration_id("abc", "OUT")
    assert calibration_id("abc", None) == "abc@zoom=unknown"
    assert calibration_id(None, "OUT") is None


def feed(service, eye):
    service._accept({"kind": "frame", "eye": eye, "sequence": 1, "observed_ns": time.monotonic_ns(),
                     "pixels": np.zeros((640, 480, 3), np.uint8)})


def robot_app(readings):
    service = StereoService(StereoSettings(source="robot"))
    service._new_session()
    for eye, level in readings.items():
        service._accept({"kind": "diagnostics", "section": eye, "values": {"zoom_level": level}})
    feed(service, "left")
    feed(service, "right")
    service.start = service.stop = lambda: None
    return service, TestClient(create_app(service.settings, service))


def test_viewer_status_and_capture_carry_the_zoom_warning_and_id():
    service, http = robot_app({"left": "IN", "right": "IN"})
    with http as client:
        zoom = client.get("/api/stereo/status").json()["zoom"]
        assert zoom["captured"] == "IN" and "never recorded" in zoom["warning"] and "widest zoom" in zoom["warning"]
        pair = client.post("/api/stereo/pairs").json()
        context = pair["camera_context"]
        assert context["zoom"]["captured"] == "IN"
        assert context["calibration_id"].endswith("@zoom=IN,focus=unknown")  # no focus read-back in this fixture
        assert context["calibration_id"].startswith(context["analysis_profile"]["id"])


def test_sim_source_reports_the_readback_only_and_keeps_the_plain_calibration_id():
    service = StereoService(StereoSettings())
    service._new_session()
    feed(service, "left")
    feed(service, "right")
    service.start = service.stop = lambda: None
    with TestClient(create_app(service.settings, service)) as client:
        zoom = client.get("/api/stereo/status").json()["zoom"]
        assert zoom["levels"] == {"left": None, "right": None} and zoom["warning"] is None  # nothing read yet
        service._accept({"kind": "diagnostics", "section": "left", "values": {"zoom_level": "ZERO"}})
        service._accept({"kind": "diagnostics", "section": "right", "values": {"zoom_level": "ZERO"}})
        context = client.post("/api/stereo/pairs").json()["camera_context"]
        assert context["zoom"]["captured"] == "ZERO" and context["zoom"]["warning"] is None
        assert "@zoom" not in context["calibration_id"]


def test_zero_is_a_readback_and_a_restore_command_but_the_sdk_may_refuse_it():
    assert zoom_status("OUT", {"left": "ZERO", "right": "ZERO"})["captured"] == "ZERO"
    client = fake_client(left="ZERO", right="ZERO")
    assert [r["level"] for r in setter(client).read_both()] == ["ZERO", "ZERO"]
    assert client.left_camera.commands == []  # read-back only
    accepted = setter(fake_client(left="INTER", right="INTER"), dry_run=False).set_level("left", "ZERO")
    assert accepted["ok"] and accepted["after"] == "ZERO"
    refused = setter(fake_client(fail=True), dry_run=False).set_level("left", "ZERO")
    assert refused["ok"] is False and refused["error"] == "ValueError"


def test_zoom_read_warning_compares_with_the_profile_zoom():
    ok = camera_zoom.zoom_read_warning("OUT", {"left": "OUT", "right": "OUT"})
    assert ok is None
    mismatch = camera_zoom.zoom_read_warning("OUT", {"left": "IN", "right": "IN"})
    assert "read IN" in mismatch and "valid at OUT" in mismatch
    unknown = camera_zoom.zoom_read_warning(None, {"left": "IN", "right": "IN"})
    assert "profile's zoom is unknown" in unknown and "only at the zoom recorded in the bundle (IN)" in unknown
    assert "could not be read" in camera_zoom.zoom_read_warning("OUT", {"left": "OUT", "right": None})
