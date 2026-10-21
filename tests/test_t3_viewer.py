"""T3-Analysis in the stereo viewer (issue #52): reference, analyze, confirm, history, labels. Offline: a fake camera
service, a stub detector and the sim fixtures; nothing moves."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from reachy_ai.games import t3_viewer
from reachy_ai.games import tictactoe as t3
from reachy_ai.pair_bundle import read_bundle
from reachy_ai.stereo import StereoService, StereoSettings
from reachy_ai.stereo_app import create_app

ROOT = Path(__file__).resolve().parents[1]
EMPTY = ROOT / "tests" / "fixtures" / "sim_head_sweep"
D_FULL = ROOT / "tests" / "fixtures" / "t3" / "D_full"
NECK = ("neck_roll", "neck_pitch", "neck_yaw")


class StubDetector:
    def __init__(self):
        self.detections = []

    def status(self):
        return {"available": True}

    def analyze(self, pixels):
        return {"model": {"name": "stub", "sha256": "test-only"}, "elapsed_ms": 1, "detections": self.detections}


def feed_images(service, folder, sequence=1):
    for eye in ("left", "right"):
        service._accept({"kind": "frame", "eye": eye, "sequence": sequence, "observed_ns": time.monotonic_ns(),
                         "pixels": cv2.imread(str(folder / f"{eye}.jpg"))})


def neck(service, pitch=64.78):
    service._accept({"kind": "head", "observed_ns": time.monotonic_ns(), "values": {
        n: {"present_deg": v, "goal_deg": v, "compliant": False} for n, v in zip(NECK, (0.0, pitch, 0.0))}})


@pytest.fixture
def viewer(tmp_path):
    service = StereoService(StereoSettings(source="sim", t3_history_dir=str(tmp_path / "history")))
    service._new_session()
    for eye in ("left", "right"):
        service._accept({"kind": "diagnostics", "section": eye, "values": {"zoom_level": "ZERO"}})
    feed_images(service, EMPTY)
    neck(service)
    service.start = service.stop = lambda: None
    service.capture_blocker = lambda *a, **k: None       # the settling rule has its own tests (#28)
    service.moves = []

    def request_head(action, preset=None, approved=False):
        service.moves.append((action, preset, approved))
        record = {"action": action, "preset": preset, "mode": "dry-run", "outcome": "dry-run",
                  "gate": {"passed": False, "failures": ["REACHY_ENABLE_MOTION is not true"]}}
        service._accept({"kind": "head_result", "id": "x", "record": record, "finished_ns": time.monotonic_ns()})
        return record

    service.request_head = request_head
    detector = StubDetector()
    app = create_app(service.settings, service, detector=detector)
    app.state.t3_settle_timeout_s = 0
    with TestClient(app) as client:
        client.service, client.detector, client.history = service, detector, tmp_path / "history"
        yield client


def populate(client, folder=D_FULL):
    """Put a populated board (and its detector output) in front of the camera."""
    feed_images(client.service, folder, sequence=2)
    client.detector.detections = json.loads((folder / "analysis.json").read_text())["detections"]


def build_reference(client):
    return client.post("/api/t3/reference", json={"confirm": True})


def analyze(client):
    populate(client)
    return client.post("/api/t3/analyze", json={"confirm": True})


def history_lines(client):
    files = list(client.history.glob("*.jsonl"))
    return [json.loads(line) for f in files for line in f.read_text().splitlines()]


# --- the page, the route list and the layout -----------------------------------------------------------------------

def test_page_has_the_t3_panel_and_the_route_list_has_no_arm_route(viewer):
    page = viewer.get("/").text
    for text in ("T3-ANALYSIS", "Capture empty board", "Analyze board", "Accept suggestions", "Confirm board",
                 "Save labelled pair", "t3-grid-player"):
        assert text in page
    paths = {route.path for route in viewer.app.routes}
    t3_paths = {p for p in paths if p.startswith("/api/t3")}
    assert t3_paths == {"/api/t3/state", "/api/t3/reference", "/api/t3/analyze", "/api/t3/accept-suggestions",
                        "/api/t3/confirm", "/api/t3/reads/{read_id}/labelled-pair.zip"}
    for path in paths:
        assert not any(word in path for word in ("arm", "gripper", "grasp", "pick", "place", "task", "joint")), path
    assert viewer.post("/api/t3/move", json={}).status_code == 404
    assert viewer.post("/tasks").status_code == 404


def test_player_side_is_the_reachy_side_turned_180_degrees(viewer):
    views = viewer.get("/api/t3/state").json()["views"]
    assert views["reachy"][2] == ["r1c1", "r1c2", "r1c3"]     # r1 nearest Reachy at the bottom, c1 at the left
    assert views["player"][0][2] == "r1c1"                      # r1c1 at the player's far (top) right
    assert views["player"][2] == ["r3c3", "r3c2", "r3c1"]
    turned = [list(reversed(row)) for row in reversed(views["reachy"])]
    assert views["player"] == turned
    assert viewer.get("/api/t3/state").json()["cycle"] == ["X", "O", ".", "#"]


def test_history_dir_default_and_flag():
    assert StereoSettings().t3_history_dir == "outputs/t3_history"
    assert StereoSettings(t3_history_dir="elsewhere").t3_history_dir == "elsewhere"


# --- reference and staleness ---------------------------------------------------------------------------------------

def test_reference_needs_the_confirm_then_builds_with_all_nine_squares(viewer):
    assert viewer.get("/api/t3/state").json()["reference"] is None
    assert viewer.post("/api/t3/reference", json={}).status_code == 409
    assert viewer.post("/api/t3/reference", json={"confirm": False}).status_code == 409
    assert viewer.service.moves == []                      # unconfirmed: nothing reaches the head path
    state = build_reference(viewer).json()
    assert viewer.service.moves == [("move", "board_view", True)]
    ref = state["reference"]
    assert ref["squares_found"] == list(t3.CELLS) and ref["stale"] is False and ref["pose"] == "board_view"
    assert ref["calibration_id"] and ref["zoom"] == "ZERO" and ref["source"] == "sim"


def test_reference_refused_when_the_head_is_not_at_board_view(viewer):
    neck(viewer.service, pitch=10.0)
    response = build_reference(viewer)
    assert response.status_code == 409 and "not at board_view" in response.json()["detail"]
    assert viewer.get("/api/t3/state").json()["reference"] is None


def test_reference_refused_when_the_grid_is_not_found(viewer):
    for eye in ("left", "right"):
        viewer.service._accept({"kind": "frame", "eye": eye, "sequence": 3, "observed_ns": time.monotonic_ns(),
                                "pixels": np.full((480, 640, 3), 90, np.uint8)})
    response = build_reference(viewer)
    assert response.status_code == 422 and "did not find the board" in response.json()["detail"]


def test_analyze_without_a_reference_is_refused(viewer):
    assert viewer.post("/api/t3/analyze", json={"confirm": True}).status_code == 409


def test_reference_goes_stale_on_reconnect_zoom_change_or_head_away(viewer):
    build_reference(viewer)
    assert viewer.get("/api/t3/state").json()["reference"]["stale"] is False
    neck(viewer.service, pitch=20.0)                        # head away: stale for display, the move fixes it
    reasons = viewer.get("/api/t3/state").json()["reference"]["stale_reasons"]
    assert any("not at board_view" in r for r in reasons)
    neck(viewer.service)
    viewer.service._accept({"kind": "diagnostics", "section": "left", "values": {"zoom_level": "IN"}})
    viewer.service._accept({"kind": "diagnostics", "section": "right", "values": {"zoom_level": "IN"}})
    reasons = viewer.get("/api/t3/state").json()["reference"]["stale_reasons"]
    assert any(r.startswith("zoom changed") for r in reasons)
    refused = viewer.post("/api/t3/analyze", json={"confirm": True})
    assert refused.status_code == 409 and "stale" in refused.json()["detail"]
    for eye in ("left", "right"):
        viewer.service._accept({"kind": "diagnostics", "section": eye, "values": {"zoom_level": "ZERO"}})
    assert viewer.get("/api/t3/state").json()["reference"]["stale"] is False
    viewer.service._new_session()
    reasons = viewer.get("/api/t3/state").json()["reference"]["stale_reasons"]
    assert any("reconnected" in r for r in reasons)
    assert viewer.post("/api/t3/analyze", json={"confirm": True}).status_code == 409


def test_analyze_refused_when_the_head_did_not_reach_board_view(viewer):
    build_reference(viewer)
    populate(viewer)
    neck(viewer.service, pitch=10.0)                        # a dry-run move leaves the head where it was
    response = viewer.post("/api/t3/analyze", json={"confirm": True})
    assert response.status_code == 409 and "not at board_view" in response.json()["detail"]


# --- analyze, accept suggestions, confirm ---------------------------------------------------------------------------

def test_analyze_reads_the_board_with_evidence(viewer):
    build_reference(viewer)
    assert viewer.post("/api/t3/analyze", json={}).status_code == 409          # fresh confirm per move
    data = analyze(viewer).json()
    assert data["board"] == "?O?/O?O/?O?"
    assert data["bundle_sha256"] and data["pair_id"] and data["reference_id"]
    square = {s["cell"]: s for s in data["squares"]}["r3c1"]
    assert square["state"] == "?" and square["suggested"] == "X" and square["suggested_by"] == "shape"
    assert square["shape"]["class"] == "square"
    assert square["detections"][0]["label"] == "cylinder"
    assert data["settings"]["min_label_confidence"] == 0.75


def test_accept_suggestions_fills_only_unknowns_that_have_one(viewer):
    build_reference(viewer)
    data = analyze(viewer).json()
    filled = viewer.post("/api/t3/accept-suggestions",
                         json={"read_id": data["read_id"], "board": data["board"]}).json()
    assert filled["board"] == "?OX/OXO/XOX" and filled["filled"] == ["r1c3", "r2c2", "r3c1", "r3c3"]   # r1c1 has none
    kept = viewer.post("/api/t3/accept-suggestions", json={
        "read_id": data["read_id"], "board": "XOX/OXO/XOX"}).json()
    assert kept["board"] == "XOX/OXO/XOX" and kept["filled"] == []        # an edited square is not overwritten
    gone = viewer.post("/api/t3/accept-suggestions", json={"read_id": "gone", "board": data["board"]})
    assert gone.status_code == 410


def test_confirm_is_refused_while_any_square_is_unknown(viewer):
    build_reference(viewer)
    data = analyze(viewer).json()
    refused = viewer.post("/api/t3/confirm", json={"read_id": data["read_id"], "board": data["board"]})
    assert refused.status_code == 409 and "Still unknown: r1c1, r1c3, r2c2, r3c1, r3c3" in refused.json()["detail"]
    assert history_lines(viewer) == []                                     # nothing logged
    assert viewer.post("/api/t3/confirm", json={"read_id": data["read_id"], "board": "XOX/OXO/XO"}).status_code == 422
    assert viewer.post("/api/t3/confirm", json={"read_id": "gone", "board": "XOX/OXO/XOX"}).status_code == 410


def test_confirm_valid_board_gives_turn_move_and_rule(viewer):
    build_reference(viewer)
    data = analyze(viewer).json()
    done = viewer.post("/api/t3/confirm", json={"read_id": data["read_id"], "board": "XX./OO./..."}).json()
    result = done["result"]
    assert (result["valid"], result["turn"], result["move_cell"], result["rule_number"]) == (True, "X", "r1c3", 1)
    assert result["text"].startswith("X's turn: place a cube (X) at r1c3. Rule 1, win")
    assert result["lines"] == [["r1c1", "r1c2", "r1c3"]]


def test_confirm_illegal_board_gives_the_reasons_and_is_still_logged(viewer):
    build_reference(viewer)
    data = analyze(viewer).json()
    done = viewer.post("/api/t3/confirm", json={"read_id": data["read_id"], "board": "XXX/X../OO.",
                                                "tags": ["illegal-setup"]}).json()
    assert done["result"]["valid"] is False and done["result"]["move_cell"] is None
    assert done["result"]["text"] == "Invalid board: counts: 4 X and 2 O differ by more than 1."
    assert history_lines(viewer)[0]["tags"] == ["illegal-setup"]


def test_other_square_makes_the_board_invalid_with_its_reason(viewer):
    build_reference(viewer)
    data = analyze(viewer).json()
    done = viewer.post("/api/t3/confirm", json={"read_id": data["read_id"], "board": "XOX/O#O/XOX"}).json()
    assert "non-game object on r2c2" in done["result"]["reasons"]


# --- history ------------------------------------------------------------------------------------------------------

def test_history_line_schema_and_corrections(viewer):
    build_reference(viewer)
    data = analyze(viewer).json()
    done = viewer.post("/api/t3/confirm", json={
        "read_id": data["read_id"], "board": "XOX/OXO/XOX", "tags": ["demo", "  free text  ", ""],
        "note": " r3c1 was a cube "}).json()
    assert done["confirmed_board"] == "XOX/OXO/XOX"
    (line,) = history_lines(viewer)
    assert set(line) == {"schema", "time_utc", "session_id", "reference_id", "pair_id", "bundle_sha256", "board_read",
                         "squares", "corrections", "confirmed_board", "result", "tags", "note", "reader_settings",
                         "viewer_version"}
    assert line["schema"] == "t3_history.v1" and line["time_utc"].endswith("Z")
    assert line["session_id"] == viewer.service.session and line["pair_id"] == data["pair_id"]
    assert line["reference_id"] == data["reference_id"] and line["bundle_sha256"] == data["bundle_sha256"]
    assert line["board_read"] == "?O?/O?O/?O?" and line["confirmed_board"] == "XOX/OXO/XOX"
    assert {"square": "r3c1", "read": "?", "confirmed": "X"} in line["corrections"]
    assert {c["square"] for c in line["corrections"]} == {"r1c1", "r1c3", "r2c2", "r3c1", "r3c3"}
    assert all(c["read"] != c["confirmed"] for c in line["corrections"])
    assert line["tags"] == ["demo", "free text"] and line["note"] == "r3c1 was a cube"
    assert line["result"]["outcome"] == "win" and line["result"]["winner"] == "X"
    assert line["reader_settings"]["change_threshold"] == 0.35
    evidence = {s["cell"]: s for s in line["squares"]}["r3c1"]
    assert set(evidence) >= {"state", "suggested", "suggested_by", "shape", "detections", "change_score", "reason"}
    assert evidence["detections"][0]["label"] == "cylinder"
    assert evidence["detections"][0]["confidence"] == pytest.approx(0.5, abs=0.01)
    assert (viewer.history / f"{viewer.service.session}.jsonl").exists()


def test_each_confirmation_appends_one_line(viewer):
    build_reference(viewer)
    data = analyze(viewer).json()
    for board in ("XX./OO./...", "XXX/X../OO."):
        viewer.post("/api/t3/confirm", json={"read_id": data["read_id"], "board": board})
    lines = history_lines(viewer)
    assert [line["confirmed_board"] for line in lines] == ["XX./OO./...", "XXX/X../OO."]


def test_tags_and_note_are_bounded(viewer):
    build_reference(viewer)
    data = analyze(viewer).json()
    url, body = "/api/t3/confirm", {"read_id": data["read_id"], "board": "XOX/OXO/XOX"}
    assert viewer.post(url, json={**body, "tags": ["x"] * 11}).status_code == 422
    assert viewer.post(url, json={**body, "tags": "demo"}).status_code == 422
    assert viewer.post(url, json={**body, "note": "n" * 501}).status_code == 422
    assert history_lines(viewer) == []


# --- labelled pair -------------------------------------------------------------------------------------------------

def test_labelled_pair_zip_needs_a_confirmed_board_and_carries_labels(viewer):
    build_reference(viewer)
    data = analyze(viewer).json()
    url = f"/api/t3/reads/{data['read_id']}/labelled-pair.zip"
    assert viewer.get(url).status_code == 409
    viewer.post("/api/t3/confirm", json={"read_id": data["read_id"], "board": "XOX/OXO/XOX"})
    response = viewer.get(url)
    assert response.status_code == 200 and "attachment" in response.headers["content-disposition"]
    manifest, files = read_bundle(response.content)                    # every file still matches its SHA-256
    assert {"pair.json", "left.jpg", "right.jpg", "analysis.json", "labels.json"} <= set(files)
    labels = json.loads(files["labels.json"])
    assert labels["schema"] == "t3_labels.v1" and labels["board"] == "XOX/OXO/XOX"
    assert labels["pair_id"] == data["pair_id"] and labels["source"] == "operator_confirmed"
    by_cell = {s["cell"]: s for s in labels["squares"]}
    assert by_cell["r3c1"] == {"cell": "r3c1", "label": "cube", "state": "X", "read": "?", "corrected": True}
    assert by_cell["r1c2"] == {"cell": "r1c2", "label": "cylinder", "state": "O", "read": "O", "corrected": False}
    assert viewer.get("/api/t3/reads/nope/labelled-pair.zip").status_code == 410


def test_history_is_the_only_server_side_write(viewer, monkeypatch, tmp_path):
    import builtins
    build_reference(viewer)
    data = analyze(viewer).json()
    opened = []
    real_open = builtins.open

    def spy(file, mode="r", *args, **kwargs):
        if any(c in mode for c in "wax+"):
            opened.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    viewer.post("/api/t3/confirm", json={"read_id": data["read_id"], "board": "XOX/OXO/XOX"})
    viewer.get(f"/api/t3/reads/{data['read_id']}/labelled-pair.zip")
    viewer.get("/api/t3/state")
    assert len(opened) == 1 and opened[0].endswith(".jsonl") and str(tmp_path / "history") in opened[0]


def test_reads_held_in_memory_are_bounded(viewer):
    build_reference(viewer)
    first = analyze(viewer).json()["read_id"]
    for _ in range(t3_viewer.MAX_READS):
        analyze(viewer)
    assert viewer.post("/api/t3/confirm", json={"read_id": first, "board": "XOX/OXO/XOX"}).status_code == 410
