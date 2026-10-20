"""Data-capture procedures, run bundle, zoom-after-confirm, and both offline analysers (#28). Offline, no robot."""

import builtins
import io
import json
import logging
import sys
import time
import types
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from reachy_ai.camera_zoom import ZoomSetter
from reachy_ai.pair_bundle import BadBundle, read_bundle
from reachy_ai.perception import procedures as proc
from reachy_ai.stereo import StereoService, StereoSettings
from reachy_ai.stereo_app import create_app

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_capture_bundle as analyse  # noqa: E402

PICTURES = ROOT / "tests" / "Pictures"


def jpeg(value=0):
    return proc.encode_jpeg(np.full((16, 16, 3), value, np.uint8))


def meta(pair_id, zoom=None, calibration="cal-1"):
    return {"pair_id": pair_id, "observation_skew_ms": 3.0,
            "camera_context": {"zoom": {"captured": zoom, "levels": {"left": zoom, "right": zoom}},
                               "calibration_id": calibration}}


class RecordingSetter:
    def __init__(self, dry_run=True):
        self.dry_run, self.calls = dry_run, []

    def set_both(self, level):
        self.calls.append(level)
        return [{"eye": e, "requested": level, "dry_run": self.dry_run, "after": level, "ok": True}
                for e in ("left", "right")]


# --- step logic -----------------------------------------------------------------------------------

def test_procedures_describe_numbered_instructions_and_steps():
    zoom, depth = proc.get_procedure("zoom_fov"), proc.get_procedure("depth_ruler")
    assert [s.zoom for s in zoom.steps] == ["OUT", "INTER", "IN"]
    assert [s.distance_m for s in depth.steps] == [0.3, 0.5, 0.8, 1.2]
    text = zoom.describe()["instructions"]
    assert text[0].startswith("1. ") and text[-1].startswith(f"{len(text)}. ")
    assert len(text) == len(zoom.setup) + len(zoom.steps)
    assert all(s.zoom is None for s in depth.steps)  # the depth test never touches the lens
    with pytest.raises(KeyError):
        proc.get_procedure("arm_sweep")


def test_run_state_counts_pairs_and_enforces_limits():
    run = proc.CaptureRun(proc.get_procedure("depth_ruler"))
    step = run.procedure.steps[0].id
    for n in range(proc.MAX_CAPTURES_PER_STEP):
        run.add_capture(step, meta(f"p{n}"), jpeg(), jpeg())
    assert run.state()["steps"][0]["captured"] == proc.MAX_CAPTURES_PER_STEP
    with pytest.raises(proc.RunLimit):
        run.add_capture(step, meta("extra"), jpeg(), jpeg())
    with pytest.raises(KeyError):
        run.add_capture("nope", meta("x"), jpeg(), jpeg())


def test_zoom_only_after_confirm_and_only_on_zoom_steps():
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    setter = RecordingSetter()
    with pytest.raises(proc.ZoomNotConfirmed):
        run.apply_zoom("zoom_inter", setter, confirmed=False)
    assert setter.calls == [] and run.state()["steps"][1]["zoom_commands"] == []
    run.apply_zoom("zoom_inter", setter, confirmed=True)
    assert setter.calls == ["INTER"] and len(run.state()["steps"][1]["zoom_commands"]) == 2
    with pytest.raises(ValueError):
        proc.CaptureRun(proc.get_procedure("depth_ruler")).apply_zoom("dist_0_3m", setter, confirmed=True)


def test_confirmed_zoom_through_the_real_setter_is_dry_run_by_default(caplog):
    cameras = {e: types.SimpleNamespace(zoom_level=types.SimpleNamespace(name="OUT")) for e in ("left", "right")}
    client = types.SimpleNamespace(left_camera=cameras["left"], right_camera=cameras["right"])
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    with caplog.at_level(logging.INFO):
        records = run.apply_zoom("zoom_in", ZoomSetter(client, resolve=lambda n: n, sleep=lambda s: None), True)
    assert all(r["dry_run"] for r in records) and "DRY-RUN" in caplog.text
    assert all(c.zoom_level.name == "OUT" for c in cameras.values())  # nothing was sent


# --- bundle ----------------------------------------------------------------------------------------

def zoom_run():
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    for n, step in enumerate(run.procedure.steps):
        run.apply_zoom(step.id, RecordingSetter(), True)
        run.add_capture(step.id, meta(f"pair{n}", zoom=step.zoom), jpeg(n), jpeg(n + 1))
    return run


def test_bundle_has_every_pair_labels_zoom_calibration_and_sha256():
    data = zoom_run().build_zip({"viewer": "9.9", "sdk": "0.7.0"})
    manifest, files = read_bundle(data)
    assert manifest["procedure"] == {"id": "zoom_fov", "name": "Zoom field-of-view check", "version": "1"}
    assert manifest["calibration_id"] == "cal-1" and manifest["versions"]["sdk"] == "0.7.0"
    for n, level in enumerate(("out", "inter", "in")):
        folder = f"steps/{n + 1:02d}_zoom_{level}/cap1"
        assert {f"{folder}/left.jpg", f"{folder}/right.jpg", f"{folder}/pair.json"} <= set(files)
    capture = json.loads(files["capture.json"])
    assert [s["label"] for s in capture["steps"]] == ["Zoom OUT", "Zoom INTER", "Zoom IN"]
    assert [s["captures"][0]["zoom_level"] for s in capture["steps"]] == ["OUT", "INTER", "IN"]
    assert capture["procedure"]["version"] == "1" and capture["calibration_id"] == "cal-1"
    assert capture["steps"][0]["zoom_commands"][0]["requested"] == "OUT"
    assert {e["name"] for e in manifest["files"]} == set(files)


def test_bundle_is_deterministic_and_tampering_is_detected():
    assert zoom_run().build_zip({}) == zoom_run().build_zip({})
    data = zoom_run().build_zip({})
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            blob = src.read(info.filename)
            dst.writestr(info.filename, b"x" + blob[1:] if info.filename.endswith("cap1/left.jpg") else blob)
    with pytest.raises(BadBundle, match="SHA-256"):
        read_bundle(out.getvalue())
    with pytest.raises(proc.RunLimit):
        proc.CaptureRun(proc.get_procedure("zoom_fov")).build_zip({})


# --- viewer API ------------------------------------------------------------------------------------

def feed(service, eye):
    service._accept({"kind": "frame", "eye": eye, "sequence": 1, "observed_ns": time.monotonic_ns(),
                     "pixels": np.zeros((64, 48, 3), np.uint8)})


def viewer(source="robot", level="OUT"):
    service = StereoService(StereoSettings(source=source))
    service._new_session()
    for eye in ("left", "right"):
        service._accept({"kind": "diagnostics", "section": eye, "values": {"zoom_level": level}})
        feed(service, eye)
    service.start = service.stop = lambda: None
    service.zoom_requests = []

    def request_zoom(level):
        service.zoom_requests.append(level)
        return [{"eye": e, "requested": level, "dry_run": True, "after": "OUT", "ok": False} for e in ("left", "right")]

    service.request_zoom = request_zoom
    return service, TestClient(create_app(service.settings, service))


def test_viewer_zoom_needs_confirm_then_is_dry_run_and_capture_and_download_work(monkeypatch, tmp_path):
    service, http = viewer()
    with http as client:
        listing = client.get("/api/stereo/procedures").json()
        assert [p["id"] for p in listing["procedures"]] == ["zoom_fov", "depth_ruler"]
        assert listing["zoom_commands"] == "dry-run"
        assert client.get("/api/stereo/capture-run").status_code == 409  # nothing chosen yet
        assert client.post("/api/stereo/capture-run", json={"procedure": "nope"}).status_code == 404
        assert client.post("/api/stereo/capture-run", json={"procedure": "zoom_fov"}).status_code == 200
        url = "/api/stereo/capture-run/steps/zoom_inter"
        assert client.post(url + "/zoom", json={}).status_code == 409
        assert client.post(url + "/zoom", json={"confirm": False}).status_code == 409
        assert service.zoom_requests == []
        result = client.post(url + "/zoom", json={"confirm": True}).json()
        assert service.zoom_requests == ["INTER"] and result["dry_run"] is True
        state = client.post(url + "/capture").json()
        assert state["steps"][1]["captured"] == 1 and state["pairs"] == 1
        monkeypatch.chdir(tmp_path)
        real_open = builtins.open

        def read_only(file, mode="r", *args, **kwargs):
            if any(c in mode for c in "wax+"):
                raise AssertionError(f"server tried to write {file!r}")
            return real_open(file, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", read_only)
        response = client.get("/api/stereo/capture-run/bundle.zip")
        assert response.status_code == 200 and "attachment" in response.headers["content-disposition"]
        manifest, files = read_bundle(response.content)
        capture = json.loads(files["capture.json"])
        assert capture["steps"][1]["captures"][0]["zoom_level"] == "OUT"  # the real read-back, not the request
        assert capture["steps"][1]["requested_zoom"] == "INTER"
        assert capture["calibration_id"].endswith("@zoom=OUT") and manifest["procedure"]["id"] == "zoom_fov"
        assert list(tmp_path.iterdir()) == []
        assert client.delete("/api/stereo/capture-run").status_code == 200
        assert client.get("/api/stereo/capture-run").status_code == 409


def test_simulator_refuses_zoom_commands_but_depth_steps_capture():
    service, http = viewer(source="sim")
    with http as client:
        client.post("/api/stereo/capture-run", json={"procedure": "zoom_fov"})
        assert client.post("/api/stereo/capture-run/steps/zoom_out/zoom", json={"confirm": True}).status_code == 409
        assert service.zoom_requests == []
        client.post("/api/stereo/capture-run", json={"procedure": "depth_ruler"})
        assert client.post("/api/stereo/capture-run/steps/dist_0_5m/zoom", json={"confirm": True}).status_code == 409
        assert client.post("/api/stereo/capture-run/steps/dist_0_5m/capture").json()["pairs"] == 1


def test_zoom_result_updates_the_readback_only_when_really_sent():
    service = StereoService(StereoSettings(source="robot"))
    service._new_session()
    service._accept({"kind": "diagnostics", "section": "left", "values": {"zoom_level": "OUT"}})
    record = {"eye": "left", "after": "IN", "ok": True}
    service._accept({"kind": "zoom_result", "id": "a", "dry_run": True, "records": [record]})
    assert service.diagnostics["left"]["zoom_level"] == "OUT"
    service._accept({"kind": "zoom_result", "id": "b", "dry_run": False, "records": [record]})
    assert service.diagnostics["left"]["zoom_level"] == "IN"


def test_page_offers_the_data_capture_dropdown():
    _, http = viewer()
    with http as client:
        page = client.get("/").text
    assert 'id="capture-select"' in page and "Download capture bundle" in page and "I confirm: command zoom" in page


# --- offline analysers -----------------------------------------------------------------------------

def scaled(name, factor):
    image = cv2.imread(str(PICTURES / name))
    return image if factor == 1 else cv2.resize(image, None, fx=factor, fy=factor)


def zoom_fixture_bundle(factors=(1.0, 1.25, 1.5)):
    """The August board pair, enlarged to stand in for the zoomed-in levels (cell pitch scales with the zoom)."""
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    for step, factor in zip(run.procedure.steps, factors):
        run.add_capture(step.id, meta(step.id, zoom=step.zoom),
                        proc.encode_jpeg(scaled("Left Camera_screenshot_20.08.2026.png", factor)),
                        proc.encode_jpeg(scaled("Right Camera_screenshot_20.08.2026.png", factor)))
    return run.build_zip({})


def test_zoom_analyser_reports_pitch_per_level_from_a_bundle():
    _, files, capture = analyse.load(zoom_fixture_bundle())
    result = analyse.analyze_zoom(capture, files)
    pitch = {(r["level"], r["eye"]): r for r in result["rows"]}
    for eye in ("left", "right"):
        assert pitch[("OUT", eye)]["pitch_px"] > 40
        assert pitch[("OUT", eye)]["pitch_vs_widest"] == 1.0
        assert pitch[("INTER", eye)]["pitch_vs_widest"] == pytest.approx(1.25, abs=0.03)
        assert pitch[("IN", eye)]["pitch_vs_widest"] == pytest.approx(1.5, abs=0.04)
    assert result["scene_zoom_is_widest"] is True and result["findings"] == []


def test_zoom_analyser_flags_a_zoom_that_never_changed():
    _, files, capture = analyse.load(zoom_fixture_bundle((1.0, 1.0, 1.0)))
    findings = analyse.analyze_zoom(capture, files)["findings"]
    assert any("pitch did not change" in f for f in findings)


def depth_fixture_bundle():
    run = proc.CaptureRun(proc.get_procedure("depth_ruler"))
    for step in run.procedure.steps:
        for n in range(2):
            run.add_capture(step.id, meta(f"{step.id}-{n}", zoom="OUT"), jpeg(), jpeg())
    return run.build_zip({})


def test_depth_analyser_judges_each_distance_against_the_analytic_sensitivity():
    _, files, capture = analyse.load(depth_fixture_bundle())
    measured = {0.3: 0.302, 0.5: 0.505, 0.8: 0.806, 1.2: 1.246}  # error 2 / 5 / 6 / 46 mm
    ruler_by_call = iter(m for d in (0.3, 0.5, 0.8, 1.2) for m in (measured[d],) * 2)
    result = analyse.analyze_depth(capture, files, depth=None,
                                   measure=lambda depth, left, right, roi: (next(ruler_by_call), {}))
    rows = {r["ruler_m"]: r for r in result["rows"]}
    assert [rows[d]["sensitivity_mm_per_px"] for d in (0.3, 0.5, 0.8, 1.2)] == [2.9, 8.0, 20.5, 46.2]
    assert rows[0.5]["error_mm"] == 5.0 and rows[0.5]["within_target"] and rows[0.5]["verdict"] == "within 15 mm"
    assert rows[0.8]["within_target"]  # 6 mm at 0.8 m is 0.29 px of disparity
    far = rows[1.2]
    assert not far["within_target"] and far["error_mm"] == 46.0
    assert far["implied_disparity_px"] == pytest.approx(1.0, abs=0.01)
    assert far["target_needs_disparity_px"] == pytest.approx(0.33, abs=0.01)
    assert "tighter than sub-pixel" in far["verdict"] and "Record the measured bound" in far["verdict"]


def test_depth_analyser_reports_a_close_range_miss_as_a_real_problem():
    _, files, capture = analyse.load(depth_fixture_bundle())
    ruler_m = capture["steps"][0]["distance_m"]
    result = analyse.analyze_depth(capture, files, depth=None,
                                   measure=lambda depth, left, right, roi: (ruler_m + 0.05, {}))
    near = result["rows"][0]
    assert not near["within_target"] and "calibration/matching problem" in near["verdict"]


def test_depth_analyser_handles_no_measurement():
    _, files, capture = analyse.load(depth_fixture_bundle())
    result = analyse.analyze_depth(capture, files, depth=None, measure=lambda *a: (None, {}))
    assert all(r["measured_z_m"] is None and r["verdict"] == "no stereo measurement" for r in result["rows"])
