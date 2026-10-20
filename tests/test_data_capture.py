"""Data-capture procedures, run bundle, zoom/head-after-confirm, settled captures, and the offline analysers (#28).

Offline, no robot: the SDK process is replaced by fakes on the StereoService."""

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
from reachy_ai.perception.stereo_depth import PairDepth
from reachy_ai.stereo import StereoService, StereoSettings
from reachy_ai.stereo_app import create_app

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_capture_bundle as analyse  # noqa: E402

PICTURES = ROOT / "tests" / "Pictures"


def jpeg(value=0):
    return proc.encode_jpeg(np.full((16, 16, 3), value, np.uint8))


def meta(pair_id, zoom=None, calibration="cal-1", neck=None):
    record = {"pair_id": pair_id, "observation_skew_ms": 3.0,
              "camera_context": {"zoom": {"captured": zoom, "levels": {"left": zoom, "right": zoom}},
                                 "calibration_id": calibration}}
    if neck is not None:
        record["head_context"] = {"joints": {n: {"present_deg": v} for n, v in
                                             zip(("neck_roll", "neck_pitch", "neck_yaw"), neck)}}
        record["head_command"] = {"action": "move", "mode": "live", "outcome": "reached"}
    return record


def steps_with(procedure, attribute):
    return [s for s in procedure.steps if getattr(s, attribute) is not None]


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
    assert [s.zoom for s in steps_with(zoom, "zoom")] == ["OUT", "INTER", "IN"]
    assert [s.distance_m for s in steps_with(depth, "distance_m")] == [0.3, 0.5, 0.8, 1.2]
    text = zoom.describe()["instructions"]
    assert text[0].startswith("1. ") and text[-1].startswith(f"{len(text)}. ")
    assert len(text) == len(zoom.setup) + len(zoom.steps)
    assert all(s.zoom is None for s in depth.steps)  # the depth test never touches the lens
    with pytest.raises(KeyError):
        proc.get_procedure("arm_sweep")


def test_head_steps_open_and_close_the_procedures():
    zoom, depth = proc.get_procedure("zoom_fov"), proc.get_procedure("depth_ruler")
    # Zoom check: step 0 points the head at the board view; the run ends with the head at neutral.
    assert (zoom.steps[0].head, zoom.steps[0].capture) == ("board_view", False)
    assert (zoom.steps[-1].head, zoom.steps[-1].capture) == ("neutral", False)
    # Depth ruler: step 0 sets neutral and nothing moves the head after it.
    assert (depth.steps[0].head, depth.steps[0].capture) == ("neutral", False)
    assert all(s.head is None for s in depth.steps[1:])
    sweep = proc.get_procedure("head_sweep")
    assert sweep.kind == "head" and [s.head for s in sweep.steps] == [*proc.SWEEP_PRESETS, "neutral"]
    assert sweep.steps[1].describe()["head_deg"] == {"neck_roll": 0.0, "neck_pitch": 64.0, "neck_yaw": 0.0}
    approach = proc.get_procedure("board_view_approach")
    assert [s.head for s in approach.steps] == ["neutral", "board_approach_55", "board_approach_60", "board_view",
                                                "neutral"]
    assert all("Point head at" in s.instruction for p in proc.PROCEDURES.values() for s in p.steps if s.head)


def test_run_state_counts_pairs_and_enforces_limits():
    run = proc.CaptureRun(proc.get_procedure("depth_ruler"))
    step = run.procedure.steps[1].id
    for n in range(proc.MAX_CAPTURES_PER_STEP):
        run.add_capture(step, meta(f"p{n}"), jpeg(), jpeg())
    assert run.state()["steps"][1]["captured"] == proc.MAX_CAPTURES_PER_STEP
    with pytest.raises(proc.RunLimit):
        run.add_capture(step, meta("extra"), jpeg(), jpeg())
    with pytest.raises(KeyError):
        run.add_capture("nope", meta("x"), jpeg(), jpeg())
    with pytest.raises(ValueError, match="nothing to capture"):
        run.add_capture("head_neutral", meta("x"), jpeg(), jpeg())


def test_zoom_only_after_confirm_and_only_on_zoom_steps():
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    setter = RecordingSetter()
    with pytest.raises(proc.ZoomNotConfirmed):
        run.apply_zoom("zoom_inter", setter, confirmed=False)
    assert setter.calls == [] and run.state()["steps"][2]["zoom_commands"] == []
    run.apply_zoom("zoom_inter", setter, confirmed=True)
    assert setter.calls == ["INTER"] and len(run.state()["steps"][2]["zoom_commands"]) == 2
    with pytest.raises(ValueError):
        proc.CaptureRun(proc.get_procedure("depth_ruler")).apply_zoom("dist_0_3m", setter, confirmed=True)


class RecordingMover:
    def __init__(self, outcome="dry-run"):
        self.calls, self.outcome = [], outcome

    def move(self, preset):
        self.calls.append(preset)
        failures = ["neck_pitch 80 deg outside the head limits"] if self.outcome == "refused" else []
        return {"action": "move", "preset": preset, "target_deg": proc.head_deg(preset), "mode": "dry-run",
                "outcome": self.outcome, "gate": {"passed": not failures, "failures": failures},
                "readback_deg": {"neck_roll": 0.0, "neck_pitch": 0.0, "neck_yaw": 0.0}}


def test_head_moves_only_after_confirm_and_every_attempt_is_recorded():
    run = proc.CaptureRun(proc.get_procedure("head_sweep"))
    mover = RecordingMover()
    with pytest.raises(proc.HeadNotConfirmed):
        run.apply_head("head_board_view", mover, confirmed=False)
    assert mover.calls == []  # nothing sent
    run.apply_head("head_board_view", mover, confirmed=True)
    with pytest.raises(proc.HeadRefused, match="outside the head limits"):
        run.apply_head("head_board_view", RecordingMover("refused"), confirmed=True)
    attempts = run.state()["steps"][1]["head_commands"]
    assert [a["outcome"] for a in attempts] == ["refused", "dry-run", "refused"]
    assert attempts[0]["gate"]["failures"] == ["Operator approval not given"]
    assert mover.calls == ["board_view"]
    with pytest.raises(ValueError, match="does not move the head"):
        proc.CaptureRun(proc.get_procedure("depth_ruler")).apply_head("dist_0_3m", mover, True)


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
    for n, step in enumerate(steps_with(run.procedure, "zoom")):
        run.apply_zoom(step.id, RecordingSetter(), True)
        run.add_capture(step.id, meta(f"pair{n}", zoom=step.zoom), jpeg(n), jpeg(n + 1))
    return run


def test_bundle_has_every_pair_labels_zoom_calibration_and_sha256():
    data = zoom_run().build_zip({"viewer": "9.9", "sdk": "0.7.0"})
    manifest, files = read_bundle(data)
    assert manifest["procedure"] == {"id": "zoom_fov", "name": "Zoom field-of-view check", "version": "2"}
    assert manifest["calibration_id"] == "cal-1" and manifest["versions"]["sdk"] == "0.7.0"
    for n, level in enumerate(("out", "inter", "in")):
        folder = f"steps/{n + 2:02d}_zoom_{level}/cap1"
        assert {f"{folder}/left.jpg", f"{folder}/right.jpg", f"{folder}/pair.json"} <= set(files)
    capture = json.loads(files["capture.json"])
    assert [s["label"] for s in capture["steps"]] == ["Head: board_view", "Zoom OUT", "Zoom INTER", "Zoom IN",
                                                      "End: head to neutral"]
    zoom_steps = capture["steps"][1:4]
    assert [s["captures"][0]["zoom_level"] for s in zoom_steps] == ["OUT", "INTER", "IN"]
    assert capture["procedure"]["version"] == "2" and capture["calibration_id"] == "cal-1"
    assert zoom_steps[0]["zoom_commands"][0]["requested"] == "OUT"
    assert capture["steps"][0]["requested_head"] == "board_view" and capture["steps"][0]["captures"] == []
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

def feed(service, eye, observed_ns=None):
    service._accept({"kind": "frame", "eye": eye, "sequence": 1, "observed_ns": observed_ns or time.monotonic_ns(),
                     "pixels": np.zeros((64, 48, 3), np.uint8)})


def neck_readback(service, roll=0.0, pitch=0.0, yaw=0.0, observed_ns=None):
    service._accept({"kind": "head", "observed_ns": observed_ns or time.monotonic_ns(), "values": {
        n: {"present_deg": v, "goal_deg": v, "compliant": False}
        for n, v in (("neck_roll", roll), ("neck_pitch", pitch), ("neck_yaw", yaw))}})


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
    service.head_requests_seen = []

    def request_head(action, preset=None, approved=False):
        # What the SDK process would answer in dry-run, fed back through the real result handler.
        service.head_requests_seen.append((action, preset, approved))
        record = {"action": action, "preset": preset, "target_deg": proc.head_deg(preset), "mode": "dry-run",
                  "outcome": "dry-run", "gate": {"passed": False, "failures": ["REACHY_ENABLE_MOTION is not true"]},
                  "readback_deg": {"neck_roll": 0.0, "neck_pitch": 0.0, "neck_yaw": 0.0}}
        service._accept({"kind": "head_result", "id": "x", "record": record, "finished_ns": time.monotonic_ns()})
        return record

    def request_head_stop():
        service.head_requests_seen.append(("stop", None, None))
        return {"action": "stop", "mode": "dry-run", "outcome": "dry-run", "readback_deg": None}

    service.request_head, service.request_head_stop = request_head, request_head_stop
    return service, TestClient(create_app(service.settings, service))


def settle(service):
    """Fresh frames and neck read-back, observed after every command so far plus the settle time."""
    later = (service.command_done_ns or time.monotonic_ns()) + int(1e9)
    for eye in ("left", "right"):
        feed(service, eye, later)
    neck_readback(service, observed_ns=later)


def test_viewer_zoom_needs_confirm_then_is_dry_run_and_capture_and_download_work(monkeypatch, tmp_path):
    service, http = viewer()
    with http as client:
        listing = client.get("/api/stereo/procedures").json()
        assert [p["id"] for p in listing["procedures"]] == ["zoom_fov", "depth_ruler", "board_view_approach",
                                                            "head_sweep"]
        assert listing["zoom_commands"] == "dry-run" and listing["head_motion"] == "dry-run"
        assert listing["head_presets"]["board_view"] == [0.0, 64.0, 0.0]
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
        assert state["steps"][2]["captured"] == 1 and state["pairs"] == 1
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
        assert capture["steps"][2]["captures"][0]["zoom_level"] == "OUT"  # the real read-back, not the request
        assert capture["steps"][2]["requested_zoom"] == "INTER"
        assert capture["calibration_id"].endswith("@zoom=OUT") and manifest["procedure"]["id"] == "zoom_fov"
        assert list(tmp_path.iterdir()) == []
        assert client.delete("/api/stereo/capture-run").status_code == 200
        assert client.get("/api/stereo/capture-run").status_code == 409


def test_viewer_head_moves_need_a_fresh_confirm_and_land_in_the_bundle():
    service, http = viewer()
    with http as client:
        client.post("/api/stereo/capture-run", json={"procedure": "head_sweep"})
        url = "/api/stereo/capture-run/steps/head_board_view"
        assert client.post(url + "/head", json={}).status_code == 409
        assert service.head_requests_seen == []  # unconfirmed: nothing reaches the SDK process
        result = client.post(url + "/head", json={"confirm": True}).json()
        assert result["dry_run"] is True and service.head_requests_seen == [("move", "board_view", True)]
        assert client.post("/api/stereo/capture-run/steps/end_neutral/capture").status_code == 409  # no capture
        assert client.post("/api/stereo/capture-run/steps/dist_0_3m/head", json={"confirm": True}).status_code == 404
        settle(service)
        assert client.post(url + "/capture").json()["pairs"] == 1
        stop = client.post("/api/stereo/head/stop").json()
        assert stop["dry_run"] is True and stop["state"]["head_events"][0]["action"] == "stop"
        assert client.post("/api/stereo/head/release", json={}).status_code == 409
        assert client.post("/api/stereo/head/release", json={"confirm": True}).status_code == 200
        assert [a for a, *_ in service.head_requests_seen] == ["move", "stop", "release"]
        _, files = read_bundle(client.get("/api/stereo/capture-run/bundle.zip").content)
    capture = json.loads(files["capture.json"])
    step = capture["steps"][1]
    assert step["requested_head"] == "board_view" and step["requested_head_deg"]["neck_pitch"] == 64.0
    assert [c["outcome"] for c in step["head_commands"]] == ["refused", "dry-run"]  # every attempt, refused too
    assert step["head_commands"][1]["target_deg"] == {"neck_roll": 0.0, "neck_pitch": 64.0, "neck_yaw": 0.0}
    assert step["captures"][0]["neck_deg"] == {"neck_roll": 0.0, "neck_pitch": 0.0, "neck_yaw": 0.0}
    assert step["captures"][0]["head_command"]["preset"] == "board_view"
    assert [e["action"] for e in capture["head_events"]] == ["stop", "release", "release"]
    assert capture["head_events"][1]["outcome"] == "refused"


def test_capture_is_refused_before_the_last_command_has_settled():
    service, http = viewer()
    with http as client:
        client.post("/api/stereo/capture-run", json={"procedure": "zoom_fov"})
        client.post("/api/stereo/capture-run/steps/zoom_out/zoom", json={"confirm": True})
        service._accept({"kind": "zoom_result", "id": "z", "dry_run": True, "records": [],
                         "finished_ns": time.monotonic_ns()})
        refused = client.post("/api/stereo/capture-run/steps/zoom_out/capture")
        assert refused.status_code == 409 and "settling" in refused.json()["detail"]
        assert client.post("/api/stereo/pairs").status_code == 200  # a plain viewer capture is not a procedure
        settle(service)
        assert client.post("/api/stereo/capture-run/steps/zoom_out/capture").json()["pairs"] == 1


def test_capture_is_refused_while_the_neck_reads_back_off_its_command():
    service, http = viewer()
    with http as client:
        client.post("/api/stereo/capture-run", json={"procedure": "head_sweep"})
        live = {"action": "move", "preset": "board_view", "mode": "live", "outcome": "reached",
                "target_deg": {"neck_roll": 0.0, "neck_pitch": 64.0, "neck_yaw": 0.0}}
        service._accept({"kind": "head_result", "id": "m", "record": live, "finished_ns": time.monotonic_ns()})
        url = "/api/stereo/capture-run/steps/head_board_view/capture"
        later = service.command_done_ns + int(1e9)
        for eye in ("left", "right"):
            feed(service, eye, later)
        neck_readback(service, pitch=58.0, observed_ns=later)  # 6 deg short of the command
        refused = client.post(url)
        assert refused.status_code == 409 and "read-back" in refused.json()["detail"]
        neck_readback(service, pitch=63.0, observed_ns=later)  # within the tolerance
        assert client.post(url).json()["pairs"] == 1
        stopped = {"action": "stop", "mode": "live", "outcome": "held"}
        service._accept({"kind": "head_result", "id": "stop", "record": stopped, "finished_ns": time.monotonic_ns()})
        settle(service)
        assert "stopped" in client.post(url).json()["detail"]


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
    for step, factor in zip(steps_with(run.procedure, "zoom"), factors):
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
    assert {r["step"]: r["recorded_zoom"] for r in result["rows"]}["Zoom INTER"] == ["INTER"]  # not emptied


def test_zoom_analyser_flags_a_zoom_that_never_changed():
    _, files, capture = analyse.load(zoom_fixture_bundle((1.0, 1.0, 1.0)))
    findings = analyse.analyze_zoom(capture, files)["findings"]
    assert any("pitch did not change" in f for f in findings)


def readback_bundle(levels_per_step):
    """A zoom run whose captures read back the given levels (None = unreadable), board pair at x1."""
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    left = proc.encode_jpeg(scaled("Left Camera_screenshot_20.08.2026.png", 1))
    right = proc.encode_jpeg(scaled("Right Camera_screenshot_20.08.2026.png", 1))
    for step, levels in zip(steps_with(run.procedure, "zoom"), levels_per_step):
        for n, level in enumerate(levels):
            run.add_capture(step.id, meta(f"{step.id}{n}", zoom=level), left, right)
    return run.build_zip({})


def test_zoom_analyser_never_falls_back_to_the_requested_level():
    _, files, capture = analyse.load(readback_bundle([["OUT"], ["INTER", None], ["IN", "OUT"]]))
    result = analyse.analyze_zoom(capture, files)
    rows = {(r["step"], r["eye"]): r for r in result["rows"]}
    assert rows[("Zoom OUT", "left")]["level"] == "OUT"
    missing, mixed = rows[("Zoom INTER", "left")], rows[("Zoom IN", "left")]
    assert missing["level"] is None and missing["readback_missing"] == 1 and missing["recorded_zoom"] == ["INTER"]
    assert mixed["level"] is None and mixed["recorded_zoom"] == ["IN", "OUT"]
    findings = " | ".join(result["findings"])
    assert "Zoom INTER: zoom read-back missing for 1 of 2" in findings
    assert "Zoom IN: mixed zoom read-backs within the step (IN, OUT)" in findings
    assert "requested" not in findings.replace("not assumed to be the requested one", "")


def test_zoom_analyser_reports_a_step_without_captures():
    _, files, capture = analyse.load(readback_bundle([["OUT"], ["INTER"], []]))
    assert "Zoom IN: no capture." in analyse.analyze_zoom(capture, files)["findings"]


def depth_fixture_bundle(size=(16, 16)):
    run = proc.CaptureRun(proc.get_procedure("depth_ruler"))
    image = proc.encode_jpeg(np.zeros((*size, 3), np.uint8))
    for step in steps_with(run.procedure, "distance_m"):
        for n in range(2):
            run.add_capture(step.id, meta(f"{step.id}-{n}", zoom="OUT"), image, image)
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
    ruler_m = capture["steps"][1]["distance_m"]
    result = analyse.analyze_depth(capture, files, depth=None,
                                   measure=lambda depth, left, right, roi: (ruler_m + 0.05, {}))
    near = result["rows"][0]
    assert not near["within_target"] and "calibration/matching problem" in near["verdict"]


def test_depth_analyser_handles_no_measurement():
    _, files, capture = analyse.load(depth_fixture_bundle())
    result = analyse.analyze_depth(capture, files, depth=None, measure=lambda *a: (None, {}))
    assert all(r["measured_z_m"] is None and r["verdict"] == "no stereo measurement" for r in result["rows"])
    assert result["sensitivity_source"] == "nominal (no profile)"


def test_depth_sensitivity_takes_f_and_b_from_the_profile_unless_overridden():
    _, files, capture = analyse.load(depth_fixture_bundle(size=(640, 480)))  # the profile's 480 x 640 portrait
    depth = PairDepth("robot")
    stub = lambda *a: (1.2, {})  # noqa: E731
    result = analyse.analyze_depth(capture, files, depth, measure=stub)
    f, b = analyse.profile_constants(depth, (640, 480, 3))
    assert result["focal_px"] == round(f, 1) and result["baseline_m"] == round(b, 4)
    assert 0.07 < b < 0.09 and 380 < f < 480 and result["sensitivity_source"].startswith("profile ")
    far = {r["ruler_m"]: r for r in result["rows"]}[1.2]
    assert far["sensitivity_mm_per_px"] == round(1.2 ** 2 / (f * b) * 1000, 1)
    overridden = analyse.analyze_depth(capture, files, depth, focal_px=400.0, measure=stub)
    assert overridden["focal_px"] == 400.0 and overridden["baseline_m"] == round(b, 4)
    assert overridden["sensitivity_source"].endswith("with override")
    both = analyse.analyze_depth(capture, files, depth, focal_px=400.0, baseline_m=0.078, measure=stub)
    assert both["sensitivity_source"] == "override" and both["rows"][3]["sensitivity_mm_per_px"] == 46.2


# --- head sweep -------------------------------------------------------------------------------------

def head_fixture_bundle(necks, blank_last=False):
    """August board pair under the head-sweep poses with the given recorded neck angles (deg)."""
    run = proc.CaptureRun(proc.get_procedure("head_sweep"))
    left = proc.encode_jpeg(scaled("Left Camera_screenshot_20.08.2026.png", 1))
    right = proc.encode_jpeg(scaled("Right Camera_screenshot_20.08.2026.png", 1))
    for step, neck in zip(steps_with(run.procedure, "head"), necks):
        run.add_capture(step.id, meta(step.id, zoom="OUT", neck=neck), left, right)
    if blank_last:
        blank = proc.encode_jpeg(np.full((640, 480, 3), 200, np.uint8))
        run.add_capture("head_sweep_56_left", meta("blank", zoom="OUT", neck=(0, 56, 15)), blank, blank)
    return run.build_zip({})


def test_head_sweep_analyser_places_the_board_with_the_recorded_neck_angles():
    _, files, capture = analyse.load(head_fixture_bundle([(0, 0, 0), (0, 0, 0)]))
    result = analyse.analyze_head(capture, files, PairDepth("robot"))
    assert capture["kind"] == "head" and list(result["poses"]) == ["Head: neutral", "Head: board_view"]
    first = result["rows"][0]
    assert first["corners"] >= 24 and 0.5 < np.linalg.norm(first["centre_camera_m"]) < 0.9
    assert result["spread"]["max_deviation_mm"] == 0.0
    assert any("same neck read-back" in f for f in result["findings"])  # identical pictures: a dry run


def test_head_sweep_analyser_flags_a_spread_the_head_transform_does_not_explain():
    _, files, capture = analyse.load(head_fixture_bundle([(0, 0, 0), (0, 0, 15)], blank_last=True))
    result = analyse.analyze_head(capture, files, PairDepth("robot"))
    # The same picture recorded at two yaws cannot be the same board: the analyser must say so.
    assert result["spread"]["max_deviation_mm"] > analyse.HEAD_SPREAD_LIMIT_MM
    assert any("board centre moves by up to" in f for f in result["findings"])
    assert any("grid not found" in f for f in result["findings"])
    torso = [result["poses"][p]["centre_torso_m"] for p in ("Head: neutral", "Head: board_view")]
    assert abs(np.linalg.norm(torso[0]) - np.linalg.norm(torso[1])) < 0.05  # yaw turns it, roughly in place
