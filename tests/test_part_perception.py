"""#84: the Part presentation procedure, its analysers and the viewer routes. Offline: no robot, no simulator.

The frames in tests/fixtures/part_perception/ were rendered by the simulator's own classes (see PROVENANCE.md);
they are bundled here with the same CaptureRun the viewer uses (scripts/grade_part_perception.py `bundle`).
"""
import ast
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from reachy_ai.pair_bundle import build_bundle, read_bundle
from reachy_ai.perception import part_colour, part_hole
from reachy_ai.perception import part_geometry as pg
from reachy_ai.perception import part_perception as pp
from reachy_ai.perception import procedures as proc
from reachy_ai.reasoner import contract as c
from reachy_ai.reasoner import policy as p
from tests.test_data_capture import neck_readback, viewer

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "part_perception"
sys.path.insert(0, str(ROOT / "scripts"))
import grade_part_perception as grader  # noqa: E402

EMPTY_LABELS = {"design_id": "empty", "orientation": None, "top_face_visible": None, "observation": None}
PART_HEIGHT_MM = {"1": 25.4, "2": 69.85}
HEIGHT_BOUNDARY_MM = 47.6
#: The stated bounds cover 95 % of the development cases (docs/PART_PERCEPTION.md), so one fixture may fall outside
#: its own bound. The tests hold each to the issue's expected accuracy instead.
HEIGHT_ERROR_LIMIT_MM = 6.0


def assert_height(measured, truth):
    value, bound = (measured["value"], measured["bound"]) if isinstance(measured, dict) else (measured.value,
                                                                                              measured.bound)
    assert abs(value - truth) <= HEIGHT_ERROR_LIMIT_MM
    assert (value < HEIGHT_BOUNDARY_MM) == (truth < HEIGHT_BOUNDARY_MM)
    assert 4.0 <= bound <= 8.0


def case_of(name: str) -> dict:
    return {**json.loads((FIXTURES / name / "case.json").read_text()), "case": name}


def labels_of(case: dict) -> dict:
    lying = bool(case.get("placement", {}).get("lying"))
    return {"design_id": case["design"], "orientation": "lying" if lying else "upright",
            "top_face_visible": not lying, "observation": "clear", "spot": case["placement"]["cell"]}


def part_zip(name: str, **overrides) -> bytes:
    case = case_of(name)
    return grader.bundle(FIXTURES / name, case, {**labels_of(case), **overrides})


@pytest.fixture(scope="module")
def reference():
    return pp.load_run(grader.bundle(FIXTURES / "empty", case_of("empty"), EMPTY_LABELS))


@pytest.fixture(scope="module")
def analysed(reference):
    cache = {}

    def run(name):
        if name not in cache:
            cache[name] = pp.analyse(part_zip(name), reference)
        return cache[name]

    return run


def edit_bundle(data: bytes, change) -> bytes:
    """The bundle with `change(files, manifest)` applied and the manifest rebuilt."""
    manifest, files = read_bundle(data)
    change(files)
    return build_bundle(files, manifest["versions"], manifest["calibration_id"], manifest.get("procedure"))


def edit_pairs(data: bytes, change) -> bytes:
    """Every pair.json and capture.json entry changed by `change(meta, entry)`."""
    def apply(files):
        capture = json.loads(files["capture.json"])
        for step in capture["steps"]:
            for entry in step["captures"]:
                meta = json.loads(files[f"{entry['folder']}/pair.json"])
                change(meta, entry)
                files[f"{entry['folder']}/pair.json"] = json.dumps(meta).encode()
        files["capture.json"] = json.dumps(capture).encode()

    return edit_bundle(data, apply)


def edit_images(data: bytes, change) -> bytes:
    """Every JPEG replaced by a lossless PNG of change(image) (the name is kept; the decoder reads the bytes)."""
    def apply(files):
        for name in list(files):
            if name.endswith(".jpg"):
                image = cv2.imdecode(np.frombuffer(files[name], np.uint8), cv2.IMREAD_COLOR)
                files[name] = cv2.imencode(".png", change(image))[1].tobytes()

    return edit_bundle(data, apply)


# --- the procedure ---------------------------------------------------------------------------------

def test_part_procedure_copies_the_zoom_check_structure_and_moves_only_head_and_lens():
    part, zoom = proc.get_procedure("part_presentation"), proc.get_procedure("zoom_fov")
    assert part.steps[0] == zoom.steps[0] and part.steps[-1] == zoom.steps[-1]  # the exact lens restore
    assert part.steps[-2] == zoom.steps[-2]
    assert [(s.zoom, s.focus, s.capture) for s in part.steps[1:-2]] == [("OUT", "sweep", True),
                                                                        ("IN", "sweep", True)]
    assert part.captures_per_step == 2 and part.kind == "part"
    assert {s.head for s in part.steps if s.head} == {"board_view", "neutral"}
    assert not any(s.gripper or s.teach or s.sound or s.inventory for s in part.steps)


def test_ground_truth_is_checked_required_before_capture_and_frozen_after():
    run = proc.CaptureRun(proc.get_procedure("part_presentation"))
    assert "ground truth" in run.capture_blocker("zoom_start")
    for bad in ({"design_id": "PART999"}, {"design_id": "PART221"},
                {"design_id": "PART221", "orientation": "sideways", "top_face_visible": True,
                 "observation": "clear"},
                {"design_id": "PART221", "orientation": "upright", "top_face_visible": None,
                 "observation": "clear"},
                {"design_id": "PART221", "orientation": "upright", "top_face_visible": True,
                 "observation": "dim"},
                {"design_id": "empty", "orientation": "upright", "top_face_visible": None, "observation": None}):
        with pytest.raises(ValueError):
            run.set_ground_truth(**{"orientation": None, "top_face_visible": None, "observation": None, **bad})
    run.set_ground_truth(design_id="PART221", orientation="upright", top_face_visible=True, observation="clear")
    assert run.capture_blocker("zoom_start") is None
    assert "Zoom OUT has not been commanded" in run.capture_blocker("zoom_out")
    run.add_capture("zoom_start", {"pair_id": "a"}, b"l", b"r")
    with pytest.raises(proc.GroundTruthRequired):
        run.set_ground_truth(design_id="PART220", orientation="upright", top_face_visible=True,
                             observation="clear")
    with pytest.raises(ValueError):
        proc.CaptureRun(proc.get_procedure("zoom_fov")).set_ground_truth(**EMPTY_LABELS)


def test_the_bundle_keeps_the_labels_in_their_own_file():
    _, files = read_bundle(part_zip("black_221"))
    truth = json.loads(files["ground_truth.json"])
    assert truth["design_id"] == "PART221" and truth["orientation"] == "upright" and truth["spot"] == "r2c2"
    assert "PART221" not in files["capture.json"].decode()
    capture = json.loads(files["capture.json"])
    assert capture["kind"] == "part" and capture["source"] == "sim" and "ground_truth" not in capture


# --- the analysers, on rendered frames -------------------------------------------------------------

@pytest.mark.parametrize("name, kind, hole", [
    ("black_221", "cylinder", "present"), ("black_111", "block", "present"),
    ("family_210", "cylinder", "absent"), ("black_221_far", "cylinder", "present")])
def test_an_upright_part_is_measured(analysed, name, kind, hole):
    result = analysed(name)
    observation, design = result.observation, case_of(name)["design"]
    assert (observation.kind, observation.orientation, observation.hole.state) == (kind, "upright", hole)
    assert_height(observation.height_mm, PART_HEIGHT_MM[design[5]])
    assert abs(observation.width_mm.value - 49.2) <= observation.width_mm.bound <= 5.0
    if hole == "present":
        assert abs(observation.hole.diameter_mm.value - 25.4) <= observation.hole.diameter_mm.bound
    assert observation.sufficiency.raised() == ()
    assert result.details["measured_at"] == "IN"


def test_the_far_row_outline_on_black_tape_counts_only_as_a_lower_limit(reference):
    run = pp.load_run(part_zip("black_221_far"))
    views = pp.level_views("IN", run, reference)
    assert max(v.silhouette.hidden_share for v in views.views()) > 0.05


def test_a_lying_part_is_never_called_upright_and_hides_its_top_face(analysed):
    observation = analysed("black_220_lying").observation
    assert observation.orientation == "lying" and observation.hole.state == "unknown"
    assert observation.sufficiency.feature_not_visible


def test_the_observation_goes_into_the_request_and_configuration_e_reads_it(reference):
    data = part_zip("black_221")
    observation = pp.analyse(data, reference).observation
    assert observation.bundle.sha256 == hashlib.sha256(data).hexdigest()
    health = c.health_from_report({"format_version": 1, "passed": True, "checked_utc": "2026-11-28T00:00:00Z",
                                   "failed": []})
    request = c.Request("req-1", "2026-11-28T00:00:00Z", "perceived", "sim", (), 0,
                        c.Case("case-1", "development", "PART221", "PART221-A", "lot-1"), health, observation, None)
    assert c.loads_request(c.dumps(request)) == request
    decision = p.decide(request, p.load_policy(), "E", "0" * 40)
    assert decision.disposition in ("accept", "remeasure")  # its bounds decide which


def test_the_analysers_never_read_the_ground_truth(reference, analysed):
    expected = analysed("black_221").observation.to_dict()
    lied = part_zip("black_221", design_id="PART110", orientation="lying", top_face_visible=False,
                    observation="occluded")
    removed = edit_bundle(part_zip("black_221"), lambda files: files.pop("ground_truth.json"))
    for data in (lied, removed):
        got = pp.analyse(data, reference).observation.to_dict()
        assert {**got, "bundle": None} == {**expected, "bundle": None}
    tree = ast.parse((ROOT / "src" / "reachy_ai" / "perception" / "part_perception.py").read_text())
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                  and isinstance(n.body[0], ast.Expr)}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and id(n) not in docstrings
                and "ground_truth" in str(n.value)]


@pytest.mark.parametrize("order", [(2, 0, 1), (1, 2, 0), (2, 1, 0)])
def test_height_hole_and_orientation_do_not_depend_on_which_channel_is_which(order):
    """No colour shortcut (owner, 2026-11-27): reordering the channels of every picture changes the colour, and
    nothing that geometry reports."""
    def swap(image):
        return image[:, :, list(order)]

    empty = grader.bundle(FIXTURES / "empty", case_of("empty"), EMPTY_LABELS)
    plain = pp.analyse(part_zip("family_210"), empty).observation
    swapped = pp.analyse(edit_images(part_zip("family_210"), swap), edit_images(empty, swap)).observation
    assert (swapped.kind, swapped.orientation, swapped.hole) == (plain.kind, plain.orientation, plain.hole)
    assert swapped.height_mm.value == pytest.approx(plain.height_mm.value, abs=0.5)
    assert swapped.width_mm.value == pytest.approx(plain.width_mm.value, abs=0.5)
    assert swapped.colour.lab_median != plain.colour.lab_median


# --- sufficiency -------------------------------------------------------------------------------------

def test_a_zoom_that_did_not_read_back_raises_lens_unverified(reference):
    def change(meta, entry):
        if meta["pair_id"].endswith("OUT-2"):
            entry["zoom_level"] = "ZERO"

    observation = pp.analyse(edit_pairs(part_zip("black_111"), change), reference).observation
    assert observation.sufficiency.lens_unverified


@pytest.mark.parametrize("what", ["same_frames", "skew"])
def test_old_or_unpaired_frames_raise_stale(reference, what):
    def change(meta, entry):
        if what == "same_frames":
            meta["frames_received"] = {e: {"observed_time_ns": 1} for e in ("left", "right")}
        else:
            entry["observation_skew_ms"] = 900.0

    observation = pp.analyse(edit_pairs(part_zip("black_111"), change), reference).observation
    assert observation.sufficiency.stale


def test_a_blurred_look_raises_blurred_and_reads_no_hole(reference):
    data = edit_images(part_zip("black_221"), lambda image: cv2.GaussianBlur(image, (0, 0), 2.0))
    observation = pp.analyse(data, reference).observation
    assert observation.sufficiency.blurred and observation.hole.state == "unknown"


def test_a_head_that_moved_since_the_reference_measures_nothing(reference):
    def change(meta, entry):
        entry["neck_deg"]["neck_yaw"] += 3.0

    observation = pp.analyse(edit_pairs(part_zip("black_111"), change), reference).observation
    assert observation.sufficiency.depth_insufficient and observation.height_mm is None
    assert observation.kind == "unknown" and observation.colour is None


def test_the_robot_has_no_lens_model_at_out_or_in_so_nothing_metric_is_reported():
    with pytest.raises(pg.NoCameraModel):
        pg.eye_lens("robot", "IN", "left")
    with pytest.raises(pg.NoCameraModel):
        pg.eye_lens("robot", "IN", "left", profile_zoom="OUT")
    assert pg.eye_lens("robot", "OUT", "left", profile_zoom="OUT").source.shape == (3, 3)


def test_an_empty_board_is_required_as_the_reference(reference):
    assert pp.reference_problems(reference) == []
    occupied = pp.load_run(edit_images(grader.bundle(FIXTURES / "empty", case_of("empty"), EMPTY_LABELS),
                                       lambda image: np.full_like(image, 90)))
    assert pp.reference_problems(occupied)
    with pytest.raises(pp.PartRefused, match="not a part presentation"):
        other = io.BytesIO()
        with zipfile.ZipFile(other, "w") as archive:
            payload = json.dumps({"kind": "zoom"}).encode()
            archive.writestr("capture.json", payload)
            archive.writestr("manifest.json", json.dumps({"files": [
                {"name": "capture.json", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]}))
        pp.load_run(other.getvalue())


# --- colour and the hole test on their own ---------------------------------------------------------

@pytest.mark.parametrize("lab, name", [
    ((6, 0, 0), "black"), ((92, 0, 2), "white"), ((55, 2, -3), "grey"), ((55, 60, 40), "red"),
    ((66, 32, 64), "orange"), ((95, -17, 82), "yellow"), ((74, -51, 54), "green"), ((60, -4, -30), "blue"),
    ((45, 40, -40), "purple"), ((30, 18, 30), "brown"), ((20, 25, 30), "brown")])
def test_colour_names(lab, name):
    assert part_colour.name_of(np.array(lab, float))[0] == name
    assert name in c.COLOUR_NAMES


def colour_sample(part_rgb, white_level):
    image = np.zeros((60, 80, 3), np.uint8)
    image[:, 40:] = white_level
    image[10:50, 5:35] = part_rgb[::-1]
    part, white = np.zeros((60, 80), bool), np.zeros((60, 80), bool)
    part[10:50, 5:35], white[:, 40:] = True, True
    return [(image, part, white)]


def test_colour_is_balanced_to_the_white_and_reliable_only_with_it():
    good = part_colour.observed_colour(colour_sample((20, 140, 40), (200, 200, 200)))
    assert good.name == "green" and good.reliable and good.confidence == 1.0
    cast = part_colour.observed_colour(colour_sample((120, 120, 130), (215, 196, 196)))  # BGR: a bluish light
    assert cast.reliable and cast.lab_median[2] == pytest.approx(0.0, abs=4.0) and cast.lab_raw[2] < -3
    for white, why in (((255, 255, 255), "not visible"), ((112, 112, 112), "lighting check")):
        bad = part_colour.observed_colour(colour_sample((20, 140, 40), white))
        assert not bad.reliable and why in " ".join(bad.reasons) and bad.name == "green"
    clipped = part_colour.observed_colour(colour_sample((255, 200, 10), (200, 200, 200)))
    assert not clipped.reliable and clipped.clipped_share == 1.0


class FlatPose:
    """A camera straight above the face, 4 pixels per millimetre."""

    def project(self, points):
        return np.asarray(points)[:, :2] * 4000 + 200


@pytest.mark.parametrize("bore, state", [(20, "present"), (200, "present"), (None, "absent")])
def test_the_hole_test_reads_a_dim_or_a_see_through_bore_and_ignores_a_shadow_edge(bore, state):
    image = np.zeros((400, 400, 3), np.uint8)
    cv2.circle(image, (200, 200), 98, (90, 90, 90), -1)
    image[:200][image[:200].any(axis=2)] = 140          # a straight shadow edge across the face
    cv2.line(image, (200, 105), (200, 295), (60, 60, 60), 1)  # a faint seam
    if bore is not None:
        cv2.circle(image, (200, 200), 51, (bore, bore, bore), -1)  # 25.5 mm across
    face = part_hole.TopFace(0.0, 0.0, 0.03, 0.0, 0.0246, 0.0246, True)
    reading = part_hole.read_hole([(image, FlatPose())], face)
    assert reading.state == state
    if state == "present":
        assert abs(reading.diameter_mm - 25.5) <= reading.bound_mm


# --- the viewer ----------------------------------------------------------------------------------------

def show(service, name, zoom, sequence):
    """Make the fake cameras show a fixture's frames at `zoom`, observed after every command."""
    later = (service.command_done_ns or time.monotonic_ns()) + int(1e9) + sequence
    for eye in ("left", "right"):
        service._accept({"kind": "diagnostics", "section": eye, "values": {"zoom_level": zoom}})
        image = cv2.imread(str(FIXTURES / name / f"{'IN' if zoom == 'IN' else 'OUT'}_{eye}.jpg"))
        service._accept({"kind": "frame", "eye": eye, "sequence": sequence, "observed_ns": later, "pixels": image})
    neck = case_of(name)["neck_deg"]
    neck_readback(service, neck["neck_roll"], neck["neck_pitch"], neck["neck_yaw"], observed_ns=later)


def run_in_viewer(client, service, name, labels):
    base = "/api/stereo/capture-run"
    assert client.post(base, json={"procedure": "part_presentation"}).status_code == 200
    assert client.post(base + "/ground-truth", json=labels).status_code == 200
    sequence = 10
    for zoom in ("OUT", "IN"):
        step = f"{base}/steps/zoom_{zoom.lower()}"
        assert client.post(step + "/zoom", json={"confirm": True}).status_code == 200
        assert client.post(step + "/focus", json={"confirm": True}).status_code == 200
        for _ in range(2):
            sequence += 1
            show(service, name, zoom, sequence)
            response = client.post(step + "/capture")
            assert response.status_code == 200, response.json()


def test_viewer_runs_the_reference_then_a_part_then_decides_on_the_measured_look(tmp_path):
    service, http = viewer(source="sim", level="ZERO")
    http.app.state.disposition  # noqa: B018 - the panel is mounted
    with http as client:
        base = "/api/stereo/capture-run"
        client.post(base, json={"procedure": "zoom_fov"})
        assert client.post(base + "/ground-truth", json=EMPTY_LABELS).status_code == 409  # not a part run
        client.post(base, json={"procedure": "part_presentation"})
        assert client.post(base + "/steps/zoom_start/capture").status_code == 409  # no ground truth yet
        assert client.post(base + "/ground-truth", json={"design_id": "PART9"}).status_code == 422
        assert client.post("/api/stereo/part/analysis").status_code == 409  # no reference yet
        assert client.post("/api/stereo/part/reference").status_code == 409  # not recorded as the empty board

        run_in_viewer(client, service, "empty", EMPTY_LABELS)
        assert client.post("/api/stereo/part/analysis").status_code == 409
        kept = client.post("/api/stereo/part/reference")
        assert kept.status_code == 200 and kept.json()["reference"]["sha256"], kept.json()

        run_in_viewer(client, service, "black_221", labels_of(case_of("black_221")))
        assert client.post("/api/stereo/part/reference").status_code == 409  # a part is on the board
        result = client.post("/api/stereo/part/analysis")
        assert result.status_code == 200, result.json()
        observation = result.json()["analysis"]["observation"]
        assert (observation["kind"], observation["hole"]["state"]) == ("cylinder", "present")
        assert_height(observation["height_mm"], 69.85)
        data = client.get("/api/stereo/part/analysis/bundle.zip").content
        assert hashlib.sha256(data).hexdigest() == observation["bundle"]["sha256"]

        state = client.get("/api/disposition/state").json()
        assert state["part_note"]
        case = {"case_id": "case-part-1", "split": "development", "instance_id": "PART221-A", "lot": "lot-1",
                "expected_design_id": "PART221"}
        body = {"configuration": "E", "case": case, "input": {"mode": "part"}}
        typed = {**body, "input": {"mode": "part", "observation": {}}}
        assert client.post("/api/disposition/decide", json=typed).status_code == 422
        decided = client.post("/api/disposition/decide", json=body)
        assert decided.status_code == 200, decided.json()
        assert decided.json()["request"]["input_source"] == "perceived"
        held = client.get(decided.json()["bundle_url"]).content
        assert held == data
        again = client.post("/api/disposition/decide", json=body)  # the same look is not a new observation
        assert again.status_code == 409 and "already one of case" in again.json()["detail"]

        # an older look never passes for the present one: another pair, a new run or a discarded run drops it
        other = {**body, "case": {**case, "case_id": "case-part-2"}}

        def assert_not_served():
            assert client.get("/api/stereo/part").json()["analysis"] is None
            assert client.get("/api/stereo/part/analysis/bundle.zip").status_code == 404
            refused = client.post("/api/disposition/decide", json=other)
            assert refused.status_code == 409 and "Analyse a Part presentation run" in refused.json()["detail"]

        show(service, "black_221", "IN", 90)
        assert client.post(base + "/steps/zoom_in/capture").status_code == 200
        assert_not_served()
        assert client.post("/api/stereo/part/analysis").status_code == 200
        assert client.get("/api/stereo/part").json()["analysis"] is not None
        client.post(base, json={"procedure": "part_presentation"})
        assert_not_served()
        client.delete(base)
        assert_not_served()
    assert service.head_requests_seen == [] and all(level in ("OUT", "IN") for level in service.zoom_requests)


def test_the_part_analysis_modules_cannot_command_anything():
    for name in ("part_geometry", "part_hole", "part_colour", "part_perception", "part_viewer"):
        source = (ROOT / "src" / "reachy_ai" / "perception" / f"{name}.py").read_text()
        imports = [n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Import, ast.ImportFrom))]
        modules = {a.name for n in imports if isinstance(n, ast.Import) for a in n.names}
        modules |= {n.module or "" for n in imports if isinstance(n, ast.ImportFrom)}
        assert not [m for m in modules if m.startswith(("reachy_sdk", "reachy_ai.motion", "reachy_ai.stereo_app",
                                                         "reachy_ai.reachy_client"))], name
    html = (ROOT / "src" / "reachy_ai" / "web" / "stereo.html").read_text()
    assert 'id="part-box"' in html and "option('part'" in html
