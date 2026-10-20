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
    def __init__(self, dry_run=True, start="INTER", accepts=True, reads=None):
        self.dry_run, self.calls, self.start, self.accepts = dry_run, [], start, accepts
        self.reads = reads or {"left": start, "right": start}

    def read_both(self):
        return [{"eye": e, "level": self.reads[e]} for e in ("left", "right")]

    def set_both(self, level):
        self.calls.append(level)
        if not self.accepts:
            return [{"eye": e, "requested": level, "before": "OUT", "dry_run": self.dry_run, "after": "OUT",
                     "ok": False, "error": "ValueError"} for e in ("left", "right")]
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
    # Zoom check: step 0 reads the zoom, points the head at the board view and captures the untouched starting
    # pair; the head returns to neutral, and the last step restores the starting zoom.
    assert (zoom.steps[0].zoom_role, zoom.steps[0].head, zoom.steps[0].capture) == ("start", "board_view", True)
    assert zoom.steps[0].zoom is None  # no zoom command in the start step
    assert [s.head for s in zoom.steps[1:-2]] == [None, None, None]
    assert (zoom.steps[-2].head, zoom.steps[-2].capture) == ("neutral", False)
    assert (zoom.steps[-1].zoom_role, zoom.steps[-1].capture) == ("restore", False)
    # Depth ruler: step 0 reads the zoom, step 1 sets neutral and nothing moves the head after it.
    assert (depth.steps[0].zoom_role, depth.steps[0].capture) == ("check", False)
    assert (depth.steps[1].head, depth.steps[1].capture) == ("neutral", False)
    assert all(s.head is None for s in depth.steps[2:])
    sweep = proc.get_procedure("head_sweep")
    assert sweep.kind == "head" and [s.head for s in sweep.steps] == [*proc.SWEEP_PRESETS, "neutral"]
    assert sweep.steps[1].describe()["head_deg"] == {"neck_roll": 0.0, "neck_pitch": 64.0, "neck_yaw": 0.0}
    approach = proc.get_procedure("board_view_approach")
    assert [s.head for s in approach.steps] == ["neutral", "board_approach_20", "board_approach_55",
                                                "board_approach_60", "board_view", "neutral"]
    assert all("Point head at" in s.instruction for p in proc.PROCEDURES.values() for s in p.steps if s.head)


def test_run_state_counts_pairs_and_enforces_limits():
    run = proc.CaptureRun(proc.get_procedure("depth_ruler"))
    step = run.procedure.steps[2].id
    for n in range(proc.MAX_CAPTURES_PER_STEP):
        run.add_capture(step, meta(f"p{n}"), jpeg(), jpeg())
    assert run.state()["steps"][2]["captured"] == proc.MAX_CAPTURES_PER_STEP
    with pytest.raises(proc.RunLimit):
        run.add_capture(step, meta("extra"), jpeg(), jpeg())
    with pytest.raises(KeyError):
        run.add_capture("nope", meta("x"), jpeg(), jpeg())
    with pytest.raises(ValueError, match="nothing to capture"):
        run.add_capture("head_neutral", meta("x"), jpeg(), jpeg())
    with pytest.raises(ValueError, match="nothing to capture"):
        run.add_capture("zoom_check", meta("x"), jpeg(), jpeg())


def test_zoom_only_after_confirm_and_only_on_zoom_steps():
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    setter = RecordingSetter()
    with pytest.raises(proc.ZoomNotConfirmed):
        run.apply_zoom("zoom_inter", setter, confirmed=False)
    by_id = {s["id"]: s for s in run.state()["steps"]}
    assert setter.calls == [] and by_id["zoom_inter"]["zoom_commands"] == []
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
    run.apply_zoom("zoom_start", RecordingSetter(), False)  # read-back only: no confirm
    for n, step in enumerate(steps_with(run.procedure, "zoom")):
        run.apply_zoom(step.id, RecordingSetter(), True)
        run.add_capture(step.id, meta(f"pair{n}", zoom=step.zoom), jpeg(n), jpeg(n + 1))
    run.apply_zoom("zoom_restore", RecordingSetter(), True)
    return run


def test_bundle_has_every_pair_labels_zoom_calibration_and_sha256():
    data = zoom_run().build_zip({"viewer": "9.9", "sdk": "0.7.0"})
    manifest, files = read_bundle(data)
    assert manifest["procedure"] == {"id": "zoom_fov", "name": "Zoom field-of-view check", "version": "4"}
    assert manifest["calibration_id"] == "cal-1" and manifest["versions"]["sdk"] == "0.7.0"
    for n, level in enumerate(("out", "inter", "in")):
        folder = f"steps/{n + 2:02d}_zoom_{level}/cap1"
        assert {f"{folder}/left.jpg", f"{folder}/right.jpg", f"{folder}/pair.json"} <= set(files)
    capture = json.loads(files["capture.json"])
    assert [s["label"] for s in capture["steps"]] == [
        "Zoom: starting level", "Zoom OUT", "Zoom INTER", "Zoom IN", "End: head to neutral",
        "Zoom: restore starting level"]
    zoom_steps = capture["steps"][1:4]
    assert [s["captures"][0]["zoom_level"] for s in zoom_steps] == ["OUT", "INTER", "IN"]
    assert capture["procedure"]["version"] == "4" and capture["calibration_id"] == "cal-1"
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


def test_simulator_zoom_follows_the_robot_rules_confirm_then_dry_run():
    service, http = viewer(source="sim", level="ZERO")
    with http as client:
        client.post("/api/stereo/capture-run", json={"procedure": "zoom_fov"})
        url = "/api/stereo/capture-run/steps/zoom_out/zoom"
        assert client.post(url, json={}).status_code == 409  # the confirm is still needed
        assert service.zoom_requests == []
        result = client.post(url, json={"confirm": True}).json()
        assert service.zoom_requests == ["OUT"] and result["dry_run"] is True  # no --allow-zoom-commands
        client.post("/api/stereo/capture-run", json={"procedure": "depth_ruler"})
        assert client.post("/api/stereo/capture-run/steps/dist_0_5m/zoom", json={"confirm": True}).status_code == 422
        settle(service)
        assert client.post("/api/stereo/capture-run/steps/dist_0_5m/capture").json()["pairs"] == 1


def test_simulator_zoom_bundle_records_source_zero_and_the_start_and_restore_steps():
    service, http = viewer(source="sim", level="ZERO")
    with http as client:
        client.post("/api/stereo/capture-run", json={"procedure": "zoom_fov"})
        base = "/api/stereo/capture-run/steps/"
        read = client.post(base + "zoom_start/zoom", json={}).json()  # read-back only: no confirm, no command
        assert [r["level"] for r in read["records"]] == ["ZERO", "ZERO"] and read["dry_run"] is False
        assert service.zoom_requests == []
        settle(service)
        assert client.post(base + "zoom_out/zoom", json={"confirm": True}).status_code == 200
        settle(service)
        client.post(base + "zoom_out/capture")
        assert client.post(base + "zoom_restore/zoom", json={}).status_code == 409  # a command: needs the confirm
        assert client.post(base + "zoom_restore/zoom", json={"confirm": True}).status_code == 200
        assert service.zoom_requests == ["OUT", "ZERO"]  # ZERO commanded as the restore target
        capture = json.loads(read_bundle(client.get(base[:-6] + "bundle.zip").content)[1]["capture.json"])
    assert capture["source"] == "sim" and capture["start_zoom"] == "ZERO"
    assert capture["steps"][1]["captures"][0]["zoom_level"] == "ZERO"  # recorded as ZERO, not unknown
    assert [r["level"] for r in capture["steps"][0]["zoom_commands"]] == ["ZERO", "ZERO"]
    restore = capture["steps"][-1]
    assert restore["zoom_role"] == "restore" and all(r["target"] == "ZERO" for r in restore["zoom_commands"])


def test_depth_ruler_setup_step_warns_when_the_zoom_differs_or_the_profile_zoom_is_unknown():
    service, http = viewer(level="IN")
    with http as client:
        client.post("/api/stereo/capture-run", json={"procedure": "depth_ruler"})
        records = client.post("/api/stereo/capture-run/steps/zoom_check/zoom", json={}).json()["records"]
        assert service.zoom_requests == []  # read-back only
        assert [r["level"] for r in records] == ["IN", "IN"]
        warning = records[0]["warning"]
        assert "profile's zoom is unknown" in warning and "recorded in the bundle (IN)" in warning
    run = proc.CaptureRun(proc.get_procedure("depth_ruler"))
    ok = run.apply_zoom("zoom_check", RecordingSetter(start="OUT"), False, profile_zoom="OUT")
    assert ok[0]["warning"] is None
    bad = run.apply_zoom("zoom_check", RecordingSetter(start="IN"), False, profile_zoom="OUT")
    assert "valid at OUT" in bad[0]["warning"] and bad[0]["profile_zoom"] == "OUT"


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
    run.apply_zoom("zoom_start", RecordingSetter(start="INTER"), False)
    for step, factor in zip(steps_with(run.procedure, "zoom"), factors):
        run.add_capture(step.id, meta(step.id, zoom=step.zoom),
                        proc.encode_jpeg(scaled("Left Camera_screenshot_20.08.2026.png", factor)),
                        proc.encode_jpeg(scaled("Right Camera_screenshot_20.08.2026.png", factor)))
    run.apply_zoom("zoom_restore", RecordingSetter(start="INTER", dry_run=False), True)
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
    assert {r["level"]: r["step"] for r in result["rows"]}["INTER"] == "Zoom INTER"  # grouped by read-back
    assert result["restore_status"] == "zoom restored to INTER (read back INTER)"


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
    rows = {(r["level"], r["eye"]): r for r in result["rows"]}
    assert rows[("OUT", "left")]["captures"] == 2 and rows[("OUT", "left")]["step"] == "Zoom OUT, Zoom IN"
    unknown = rows[(None, "left")]  # the pair with no read-back is its own group, not the requested INTER
    assert unknown["captures"] == 1 and unknown["readback_missing"] == 1 and unknown["step"] == "Zoom INTER"
    assert rows[("IN", "left")]["captures"] == 1 and rows[("INTER", "left")]["captures"] == 1
    findings = " | ".join(result["findings"])
    assert "Zoom INTER: zoom read-back missing for 1 of 2" in findings
    assert "Zoom IN: mixed zoom read-backs within the step (IN, OUT)" in findings
    assert "Zoom IN (pair 2): requested IN but the cameras read back OUT; its pitch is counted under OUT" in findings
    assert "Zoom INTER (pair 2)" not in findings  # no read-back: never compared with, or filled in from, the request


def test_zoom_analyser_counts_a_mixed_step_under_each_pairs_real_level():
    """The sim test of 2026-10-20: the OUT step held a pair read back at IN, taken after the IN step."""
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    run.apply_zoom("zoom_start", RecordingSetter(start="IN"), False)
    for step, level, factor in ((s, z, f) for s, z, f in zip(steps_with(run.procedure, "zoom"),
                                                              ("OUT", "INTER", "IN"), (1.0, 1.25, 1.5))):
        run.add_capture(step.id, meta(f"{step.id}-a", zoom=level),
                        proc.encode_jpeg(scaled("Left Camera_screenshot_20.08.2026.png", factor)),
                        proc.encode_jpeg(scaled("Right Camera_screenshot_20.08.2026.png", factor)))
    run.add_capture("zoom_out", meta("late", zoom="IN"),  # a second OUT-step pair, really at IN
                    proc.encode_jpeg(scaled("Left Camera_screenshot_20.08.2026.png", 1.5)),
                    proc.encode_jpeg(scaled("Right Camera_screenshot_20.08.2026.png", 1.5)))
    run.apply_zoom("zoom_restore", RecordingSetter(start="IN", dry_run=False), True)
    _, files, capture = analyse.load(run.build_zip({}))
    result = analyse.analyze_zoom(capture, files)
    by_level = {(r["level"], r["eye"]): r for r in result["rows"]}
    assert all(r["level"] in ("OUT", "INTER", "IN") for r in result["rows"])  # no unknown group
    for eye in ("left", "right"):
        assert by_level[("OUT", eye)]["pitch_vs_widest"] == 1.0 and by_level[("OUT", eye)]["captures"] == 1
        assert by_level[("IN", eye)]["captures"] == 2 and by_level[("IN", eye)]["step"] == "Zoom OUT, Zoom IN"
        assert by_level[("IN", eye)]["pitch_vs_widest"] == pytest.approx(1.5, abs=0.04)
        assert by_level[("INTER", eye)]["pitch_vs_widest"] == pytest.approx(1.25, abs=0.03)
    findings = " | ".join(result["findings"])
    assert "Zoom OUT: mixed zoom read-backs within the step (IN, OUT)" in findings
    assert "Zoom OUT (pair 2): requested OUT but the cameras read back IN; its pitch is counted under IN" in findings
    assert result["scene_zoom_is_widest"] is True and result["restore_status"] == "zoom restored to IN (read back IN)"


def test_zoom_analyser_notes_a_start_that_was_read_twice_with_different_results():
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    run.apply_zoom("zoom_start", RecordingSetter(start="ZERO"), False)
    run.apply_zoom("zoom_start", RecordingSetter(start="IN"), False)
    run.add_capture("zoom_out", meta("a", zoom="OUT"), jpeg(), jpeg())
    _, files, capture = analyse.load(run.build_zip({}))
    assert capture["start_zoom"] == "ZERO"  # the first complete read; the later one is an extra read-back
    reads = capture["steps"][0]["zoom_commands"]
    assert [r.get("extra") for r in reads] == [None, None, True, True]
    findings = analyse.analyze_zoom(capture, files)["findings"]
    assert any("read 2 times with different results (ZERO, IN)" in f and "first complete read" in f
               for f in findings)


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
    ruler_m = capture["steps"][2]["distance_m"]
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
        run.add_capture("head_sweep_roll_left_5", meta("blank", zoom="OUT", neck=(5, 64, 0)), blank, blank)
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


def restore_bundle(start_setter=None, restore_setter=None, restore=True, sim=False):
    """A zoom run (captures at x1) with a chosen start read and restore outcome."""
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    left = proc.encode_jpeg(scaled("Left Camera_screenshot_20.08.2026.png", 1))
    right = proc.encode_jpeg(scaled("Right Camera_screenshot_20.08.2026.png", 1))
    run.apply_zoom("zoom_start", start_setter or RecordingSetter(start="ZERO"), False)
    for step in steps_with(run.procedure, "zoom"):
        run.add_capture(step.id, meta(step.id, zoom=step.zoom), left, right)
    if restore:
        run.apply_zoom("zoom_restore", restore_setter or RecordingSetter(start="ZERO", dry_run=False), True)
    if sim:
        first = next(iter(run.captures.values()))[0]["meta"]["camera_context"]
        first["analysis_profile"] = {"source": "sim", "lens": "distorted"}
    return run.build_zip({})


def test_zoom_analyser_reports_the_starting_level_and_a_clean_restore():
    _, files, capture = analyse.load(restore_bundle())
    result = analyse.analyze_zoom(capture, files)
    assert result["start_zoom"] == "ZERO"
    assert not any("estore" in f for f in result["findings"])


def test_zoom_analyser_flags_a_failed_missing_dry_run_or_unknown_restore():
    def findings(**kw):
        _, files, capture = analyse.load(restore_bundle(**kw))
        return " | ".join(analyse.analyze_zoom(capture, files)["findings"])
    failed = findings(restore_setter=RecordingSetter(start="ZERO", dry_run=False, accepts=False))
    assert "Restore FAILED (left): requested ZERO, read back OUT (ValueError)" in failed
    assert "restore step was not run" in findings(restore=False)
    assert "Restore to ZERO (left) was a dry run" in findings(restore_setter=RecordingSetter(start="ZERO"))
    unknown = findings(start_setter=RecordingSetter(reads={"left": None, "right": None}))
    assert "starting zoom is unknown" in unknown and "by hand" in unknown


def test_unknown_start_restore_sends_nothing_and_records_the_manual_step():
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    run.apply_zoom("zoom_start", RecordingSetter(reads={"left": "IN", "right": "OUT"}), False)  # eyes differ
    setter = RecordingSetter()
    record = run.apply_zoom("zoom_restore", setter, True)[0]
    assert setter.calls == [] and record["manual"] and record["ok"] is False and "by hand" in record["note"]
    failed = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    failed.apply_zoom("zoom_start", RecordingSetter(start="ZERO"), False)
    refused = failed.apply_zoom("zoom_restore", RecordingSetter(dry_run=False, accepts=False), True)
    assert all(r["manual"] and r["note"] == proc.ZERO_RESTORE_NOTE for r in refused)
    assert "restarting the robot's camera service is expected to re-home the lens (unconfirmed)" in refused[0]["note"]
    other = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    other.apply_zoom("zoom_start", RecordingSetter(start="INTER"), False)
    manual = other.apply_zoom("zoom_restore", RecordingSetter(dry_run=False, accepts=False), True)
    assert all("set the zoom to INTER by hand" in r["note"] for r in manual)


def test_zoom_analyser_notes_the_simulator_zoom_model():
    _, files, capture = analyse.load(restore_bundle(sim=True))
    findings = analyse.analyze_zoom(capture, files)["findings"]
    assert "Field of view from the simulator's zoom model, not the real lens." in findings
    _, files, capture = analyse.load(restore_bundle())
    assert not any("simulator" in f for f in analyse.analyze_zoom(capture, files)["findings"])


def test_depth_analyser_reports_the_setup_zoom_warning():
    run = proc.CaptureRun(proc.get_procedure("depth_ruler"))
    run.apply_zoom("zoom_check", RecordingSetter(start="IN"), False, profile_zoom=None)
    image = proc.encode_jpeg(np.zeros((16, 16, 3), np.uint8))
    run.add_capture("dist_0_3m", meta("p", zoom="IN"), image, image)
    _, files, capture = analyse.load(run.build_zip({}))
    result = analyse.analyze_depth(capture, files, depth=None, focal_px=400.0, baseline_m=0.078,
                                   measure=lambda *a, **k: (None, None))
    assert any("profile's zoom is unknown" in f for f in result["findings"])


def test_capture_is_refused_until_the_steps_zoom_command_has_completed():
    service, http = viewer()
    with http as client:
        client.post("/api/stereo/capture-run", json={"procedure": "zoom_fov"})
        capture = "/api/stereo/capture-run/steps/zoom_out/capture"
        settle(service)
        refused = client.post(capture)
        assert refused.status_code == 409 and "has not been commanded" in refused.json()["detail"]
        assert "Set zoom OUT" in refused.json()["detail"]
        client.post("/api/stereo/capture-run/steps/zoom_out/zoom", json={"confirm": False})  # refused: no command
        assert client.post(capture).status_code == 409
        client.post("/api/stereo/capture-run/steps/zoom_out/zoom", json={"confirm": True})  # dry-run still completes
        settle(service)
        assert client.post(capture).json()["pairs"] == 1
        # steps that command no zoom are not gated by it
        client.post("/api/stereo/capture-run", json={"procedure": "depth_ruler"})
        assert client.post("/api/stereo/capture-run/steps/dist_0_5m/capture").json()["pairs"] == 1


def test_a_failed_zoom_command_does_not_open_the_capture_gate():
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    assert "has not been commanded" in run.capture_blocker("zoom_in")
    run.zoom_commands["zoom_in"] = [{"eye": "left", "requested": "IN", "ok": False, "error": "ValueError"}]
    assert run.capture_blocker("zoom_in")
    run.apply_zoom("zoom_in", RecordingSetter(), True)
    assert run.capture_blocker("zoom_in") is None
    assert run.capture_blocker("end_neutral") is None and run.capture_blocker("zoom_start") is None


def test_start_zoom_is_the_first_complete_read_and_is_fixed_once_a_command_was_attempted():
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    run.apply_zoom("zoom_start", RecordingSetter(reads={"left": None, "right": "IN"}), False)  # incomplete read
    assert run.start_zoom() is None
    run.apply_zoom("zoom_start", RecordingSetter(start="ZERO"), False)
    run.apply_zoom("zoom_start", RecordingSetter(start="INTER"), False)
    assert run.start_zoom() == "ZERO"
    run.apply_zoom("zoom_out", RecordingSetter(), True)
    run.apply_zoom("zoom_start", RecordingSetter(start="OUT"), False)
    assert run.start_zoom() == "ZERO"
    records = run.state()["steps"][0]["zoom_commands"]
    assert [r.get("extra") for r in records] == [None, None, None, None, True, True, True, True]
    late = proc.CaptureRun(proc.get_procedure("zoom_fov"))  # read only after a command: no start level
    late.apply_zoom("zoom_out", RecordingSetter(), True)
    late.apply_zoom("zoom_start", RecordingSetter(start="OUT"), False)
    assert late.start_zoom() is None


def start_pair_bundle(factor, start="ZERO"):
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    run.apply_zoom("zoom_start", RecordingSetter(start=start), False)
    left, right = "Left Camera_screenshot_20.08.2026.png", "Right Camera_screenshot_20.08.2026.png"
    run.add_capture("zoom_start", meta("start", zoom=start),
                    proc.encode_jpeg(scaled(left, factor)), proc.encode_jpeg(scaled(right, factor)))
    for step, f in zip(steps_with(run.procedure, "zoom"), (1.0, 1.25, 1.5)):
        run.add_capture(step.id, meta(step.id, zoom=step.zoom), proc.encode_jpeg(scaled(left, f)),
                        proc.encode_jpeg(scaled(right, f)))
    return run.build_zip({})


def test_zoom_analyser_reports_which_level_the_starting_pair_matches():
    _, files, capture = analyse.load(start_pair_bundle(1.25))
    result = analyse.analyze_zoom(capture, files)
    match = result["start_match"]
    assert match["level"] == "INTER" and match["text"].startswith("start ZERO matches INTER (ratio 1.0")
    assert not any(r["step"] == "Zoom: starting level" for r in result["rows"])  # not counted as a level
    _, files, capture = analyse.load(start_pair_bundle(1.12))
    none = analyse.analyze_zoom(capture, files)["start_match"]
    assert none["level"] is None and "matches none of the tested levels (closest" in none["text"]
    _, files, capture = analyse.load(readback_bundle([["OUT"], ["INTER"], ["IN"]]))
    assert "no pair captured at the starting zoom" in analyse.analyze_zoom(capture, files)["start_match"]["text"]


def test_zoom_analyser_explains_a_refused_zero_restore():
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    run.apply_zoom("zoom_start", RecordingSetter(start="ZERO"), False)
    run.add_capture("zoom_out", meta("a", zoom="OUT"), jpeg(), jpeg())
    run.apply_zoom("zoom_restore", RecordingSetter(dry_run=False, accepts=False), True)
    _, files, capture = analyse.load(run.build_zip({}))
    findings = " | ".join(analyse.analyze_zoom(capture, files)["findings"])
    assert "Restore FAILED (left): requested ZERO" in findings
    assert "ZERO cannot be commanded through the SDK; restarting the robot's camera service is expected to " \
           "re-home the lens (unconfirmed)." in findings
