"""P1: the PerceivedScene builder (docs/design/perceived-scene.md sections 1.4 to 1.6, 7). Offline; no robot, no sim.

Fixtures: tests/fixtures/perception_grade/ (copied from reachy-1-2-sim, see PROVENANCE.md). `pair_ok` has no
camera_context, so each test builds a COPY of it in tmp_path with one added; the committed files are never edited.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from reachy_ai.perception import procedures as proc
from reachy_ai.perception import scene_builder as builder
from reachy_ai.perception import scene_contract as sc
from reachy_ai.perception.stereo_depth import camera_to_torso

ROOT = Path(__file__).resolve().parents[1]
GRADE = ROOT / "tests" / "fixtures" / "perception_grade"
RESULT_186 = GRADE / "stage1-186" / "result.json"
sys.path.insert(0, str(ROOT / "scripts"))
import build_perceived_scene as cli  # noqa: E402

LAYOUT = "FWDCenterLabSivaPool"
#: the scene grid centre in the torso frame (ADR-0010): r2c2 of the taped grid, on the table plane.
GRID_CENTRE = (0.2258, 0.0, -0.4095)
RESULT_DETECTOR = "60dad10f8e6a29dc09daf1bc11608a1658d70b5b98ffbaa615d88f54e453ff7e"
RESULT_PROFILE = "771392183ddc6c62"


def make_bundle(tmp_path, *, zoom=("ZERO", "ZERO"), source="sim", matches_profile=None,
                calibration_id="fixture-profile", graded=False, extra_measured=None, name="pair"):
    """A copy of pair_ok with a camera_context and stub images. graded=True keys it to the #186 result's rows."""
    folder = tmp_path / name
    shutil.copytree(GRADE / "pair_ok", folder)
    pair = json.loads((folder / "pair.json").read_text())
    depth = json.loads((folder / "depth.json").read_text())
    analysis = json.loads((folder / "analysis.json").read_text())
    if graded:
        calibration_id = depth["profile"]["id"] = RESULT_PROFILE
        analysis["model"]["sha256"] = RESULT_DETECTOR
    pair["source"] = source
    pair["camera_context"] = {"calibration_id": calibration_id, "zoom": {
        "levels": {"left": zoom[0], "right": zoom[1]}, "matches_profile": matches_profile}}
    for item in extra_measured or ():
        index = len(analysis["detections"])
        analysis["detections"].append({"box_xywh": [10, 10, 40, 40], "class_id": 0, "confidence": 0.9,
                                       "label": item["label"]})
        depth["measurements"].append(dict(item, detection_index=index, samples=500, status="measured"))
    for fname, value in (("pair.json", pair), ("depth.json", depth), ("analysis.json", analysis)):
        (folder / fname).write_text(json.dumps(value))
    (folder / "left.jpg").write_bytes(b"left")
    (folder / "right.jpg").write_bytes(b"right")
    return folder


TWIN = sc.BoardPose(GRID_CENTRE, (1.0, 0.0, 0.0, 0.0), (0.003, 0.003, 0.003), "twin", 0, 0.0, 0.0)
OPTIONS = builder.RigOptions(rig_layout_id=LAYOUT, scene_name=LAYOUT, scene_sha256="7" * 64, scene_revision="initial")
CODE = ("7786a65", False)


def camera_of(tmp_path):
    """The right camera's torso-frame position for the fixture pair (neck 0, 64, 0)."""
    profile = json.loads((tmp_path / "pair" / "depth.json").read_text())["profile"]
    return builder.SimRig(profile, (0.0, 64.0, 0.0)).origin()


def build(tmp_path, bundle_kw=None, **kw):
    kw.setdefault("board", TWIN)
    kw.setdefault("options", OPTIONS)
    kw.setdefault("code", CODE)
    return builder.build_scene(builder.load_pair_bundle(make_bundle(tmp_path, **(bundle_kw or {}))), **kw)


@pytest.fixture(scope="module")
def model():
    return builder.bounds_from_grader_result(RESULT_186.read_bytes(), zoom_level="ZERO")


# ---------------------------------------------------------------- the bounds model from the #186 result

def test_the_186_rows_are_read_from_the_result_and_every_one_is_provisional(model):
    assert model.band_edges_m == (0.45, 0.5, 0.55, 0.6)
    pooled = [r for r in model.rows if r.band == "all"]
    assert len(pooled) == 1 and len(model.rows) > 1
    assert pooled[0].half_width_m == pytest.approx((0.0233, 0.0233, 0.0313))
    assert (pooled[0].detector_model_sha256, pooled[0].calibration_id, pooled[0].zoom_level, pooled[0].head_pose,
            pooled[0].kind, pooled[0].surface_correction_id) == (RESULT_DETECTOR, RESULT_PROFILE, "ZERO", "board_view",
                                                                 None, "none")
    assert pooled[0].n_distinct == 10   # 20 matched, but pair1 = pair2: 10 distinct (layout, object) sightings
    assert all(model.provisional(r) and r.n_distinct < builder.MIN_DISTINCT for r in model.rows)
    assert model.model_id.startswith("perception_grade.v1:")


def test_the_model_is_refused_for_a_result_that_is_not_a_grader_result():
    for raw in (b"{}", b"[]", b"not json", json.dumps({"schema_version": "other"}).encode()):
        with pytest.raises(builder.SceneRefused):
            builder.bounds_from_grader_result(raw, zoom_level="ZERO")


def test_lookup_needs_every_key_field_and_falls_back_from_the_band_to_the_pooled_row(model):
    key = dict(detector_model_sha256=RESULT_DETECTOR, depth_method=builder.DEPTH_METHOD, surface_correction_id="none",
               calibration_id=RESULT_PROFILE, zoom_level="ZERO", head_pose="board_view", kind="cube", distance_m=0.52)
    assert model.lookup(**key).band == "0.50-0.55 m"
    assert model.lookup(**dict(key, distance_m=0.40)).band == "all"      # no <0.45 m row: the pooled one
    for field, other in (("detector_model_sha256", "0" * 64), ("depth_method", "other"),
                         ("surface_correction_id", "class-half-extent.v1"), ("calibration_id", "x"),
                         ("zoom_level", "OUT"), ("head_pose", "neutral")):
        assert model.lookup(**dict(key, **{field: other})) is None, field


# ---------------------------------------------------------------- the scene: section 7's exit gate

def test_a_pair_that_matches_no_row_gets_infinite_provisional_bounds(tmp_path, model):
    scene = build(tmp_path, bounds=model)   # the fixture's detector and profile are not #186's
    assert scene.objects and all(o.bounds_provisional and o.bounds_m == (float("inf"),) * 3 for o in scene.objects)
    assert scene.pipeline.bounds_model_id == model.model_id


def test_every_186_row_gives_a_provisional_bound_and_never_a_permitting_one(tmp_path, model):
    scene = build(tmp_path, {"graded": True}, bounds=model)
    assert len(scene.objects) == 3
    for obj in scene.objects:
        assert obj.bounds_provisional and all(0 < b < 0.05 for b in obj.bounds_m), obj.bounds_m
        assert obj.bounds_coverage == 0.95
    bands = sorted(o.bounds_m[0] for o in scene.objects)
    assert bands[-1] == pytest.approx(0.0233)    # the near object falls below the 0.45 m band: the pooled row
    assert sc.loads(sc.dumps(scene)) == scene


def test_the_surface_correction_does_not_inherit_the_uncorrected_rows(tmp_path, model):
    scene = build(tmp_path, {"graded": True}, bounds=model, surface_correction=builder.SURFACE_CORRECTION)
    assert all(o.bounds_provisional and o.bounds_m == (float("inf"),) * 3 for o in scene.objects)
    assert scene.pipeline.surface_correction_id == builder.SURFACE_CORRECTION


def test_nine_cells_always_and_unknown_is_never_empty(tmp_path):
    extra = [{"label": "cube", "xyz_torso_m": [0.2, 0.0, -0.3]}]
    for i, kw in enumerate(({}, {"graded": True}, {"extra_measured": extra})):
        scene = build(tmp_path / str(i), kw)
        assert [c.name for c in scene.cells] == list(sc.CELL_NAMES)
    # no bounds: nothing can be placed, so no cell may claim to be empty or to hold an object
    assert {c.state for c in build(tmp_path / "plain").cells} == {"unknown"}


def test_with_finite_bounds_objects_land_in_their_cells_and_unseen_cells_stay_unknown(tmp_path, model):
    scene = build(tmp_path, {"graded": True}, bounds=model)
    held = {c.name: scene.objects[c.object_index].kind for c in scene.cells if c.object_index is not None}
    assert held == {"r3c3": "cylinder", "r1c2": "cube", "r2c2": "cylinder"}
    states = {c.name: c.state for c in scene.cells}
    assert sorted(states.values()) == ["cube", "cylinder", "cylinder"] + ["empty"] * 6   # the fixture's six empties
    assert all(states[n] in ("cube", "cylinder") for n in held)
    # drop the last empty detection: the cell it covered was never reported, so it is unknown and not empty
    folder = make_bundle(tmp_path / "less", graded=True)
    analysis = json.loads((folder / "analysis.json").read_text())
    depth = json.loads((folder / "depth.json").read_text())
    analysis["detections"].pop()
    depth["measurements"].pop()
    (folder / "analysis.json").write_text(json.dumps(analysis))
    (folder / "depth.json").write_text(json.dumps(depth))
    fewer = builder.build_scene(builder.load_pair_bundle(folder), options=OPTIONS, board=TWIN, bounds=model, code=CODE)
    changed = [c.name for c, d in zip(fewer.cells, scene.cells) if c.state != d.state]
    assert len(changed) == 1 and next(c for c in fewer.cells if c.name == changed[0]).state == "unknown"
    assert sorted(c.state for c in fewer.cells).count("empty") == 5


def test_an_empty_detection_over_an_occupied_cell_makes_it_unknown(tmp_path, model):
    folder = make_bundle(tmp_path, graded=True)
    bundle = builder.load_pair_bundle(folder)
    scene = builder.build_scene(bundle, options=OPTIONS, board=TWIN, bounds=model, code=CODE)
    cell = next(c for c in scene.cells if c.name == "r2c2")
    assert cell.state == "cylinder"
    rig = builder.SimRig(bundle.depth["profile"], (0.0, 64.0, 0.0))
    px = rig.project(builder._Board.of(TWIN).cell_centre("r2c2"))
    analysis = json.loads((folder / "analysis.json").read_text())
    analysis["detections"].append({"box_xywh": [px[0] - 20, px[1] - 20, 40, 40], "class_id": 2, "confidence": 0.9,
                                   "label": "empty"})
    (folder / "analysis.json").write_text(json.dumps(analysis))
    conflicted = builder.build_scene(builder.load_pair_bundle(folder), options=OPTIONS, board=TWIN, bounds=model,
                                     code=CODE)
    assert next(c for c in conflicted.cells if c.name == "r2c2").state == "unknown"
    assert any("r2c2" in w and "empty detection" in w for w in conflicted.warnings)
    assert all(o.cell != "r2c2" for o in conflicted.objects)


def test_off_board_detections_are_separated_and_never_get_a_cell(tmp_path, model):
    far = [{"label": "cube", "xyz_torso_m": [0.1, -0.9, -0.4], "coverage": 1.0, "depth_iqr_m": 0.01},
           {"label": "cylinder", "xyz_torso_m": [0.23, 0.0, -1.1], "coverage": 1.0, "depth_iqr_m": 0.01}]
    # one far to the side, one on the floor
    scene = build(tmp_path, {"graded": True, "extra_measured": far}, bounds=model)
    assert len(scene.objects) == 3 and len(scene.off_board) == 2
    assert all(o.cell is None for o in scene.off_board)
    assert {o.centre_m for o in scene.off_board}.isdisjoint({o.centre_m for o in scene.objects})
    assert [c.name for c in scene.cells] == list(sc.CELL_NAMES)
    assert any("off the board" in w for w in scene.warnings)


def test_without_a_board_there_are_no_cells_and_nothing_is_called_off_board(tmp_path):
    scene = build(tmp_path, board=None, fit_board_from_pair=False)
    assert scene.board is None and scene.cells == () and scene.off_board == ()
    assert len(scene.objects) == 3 and all(o.cell is None for o in scene.objects)
    assert any("no board" in w for w in scene.warnings)


def test_an_unmeasured_object_detection_is_not_an_object_and_its_cell_is_unknown(tmp_path, model):
    folder = make_bundle(tmp_path, graded=True)
    depth = json.loads((folder / "depth.json").read_text())
    depth["measurements"][1] = {"detection_index": 1, "label": "cube", "samples": 3, "status": "unavailable"}
    (folder / "depth.json").write_text(json.dumps(depth))
    scene = builder.build_scene(builder.load_pair_bundle(folder), options=OPTIONS, board=TWIN, bounds=model, code=CODE)
    assert len(scene.objects) == 2
    assert next(c for c in scene.cells if c.name == "r1c2").state == "unknown"
    assert any("detection 1 (cube) has no 3D measurement" in w for w in scene.warnings)


# ---------------------------------------------------------------- the surface correction

def test_the_class_sizes_use_the_midpoint_and_the_range_of_the_per_rig_table():
    cube, cylinder = builder.class_size(LAYOUT, "cube"), builder.class_size(LAYOUT, "cylinder")
    assert cube.half_m == pytest.approx((0.0277,) * 3)
    assert cylinder.half_m == pytest.approx((0.028925, 0.028925, 0.042775))
    straight_down = np.array([0.0, 0.0, -1.0])
    sideways = np.array([1.0, 0.0, 0.0])
    assert cube.along(straight_down) == pytest.approx((0.0277, 0.0023))              # +-2.3 mm
    assert cylinder.along(straight_down) == pytest.approx((0.042775, 0.007225))      # +-7.2 mm half-height
    assert cylinder.along(sideways) == pytest.approx((0.028925, 0.006075))           # +-6.1 mm radius
    assert cube.size_m == pytest.approx((0.0554,) * 3)


def test_the_correction_moves_the_centre_away_from_the_camera_along_the_view_ray(tmp_path):
    scene = build(tmp_path, surface_correction=builder.SURFACE_CORRECTION)
    camera = camera_of(tmp_path)
    for obj in scene.objects:
        surface, centre = np.array(obj.surface_point_m), np.array(obj.centre_m)
        ray = (surface - camera) / np.linalg.norm(surface - camera)
        moved = centre - surface
        expected, _ = builder.class_size(LAYOUT, obj.kind).along(ray)
        assert np.linalg.norm(moved) == pytest.approx(expected)
        assert np.dot(moved, ray) == pytest.approx(expected)    # along the ray, away from the camera
        assert np.linalg.norm(centre - camera) == pytest.approx(np.linalg.norm(surface - camera) + expected)
        assert obj.size_source == "class_prior" and obj.size_m == pytest.approx(
            builder.class_size(LAYOUT, obj.kind).size_m)
    uncorrected = build(tmp_path / "u")
    assert all(o.centre_m == o.surface_point_m and o.size_m is None and o.size_source == "unknown"
               for o in uncorrected.objects)
    assert [o.surface_point_m for o in uncorrected.objects] == [o.surface_point_m for o in scene.objects]


def test_the_range_of_the_known_sizes_is_added_to_a_looked_up_bound(tmp_path):
    row = lambda correction: builder.BoundsRow(  # noqa: E731
        "f" * 64, builder.DEPTH_METHOD, correction, "fixture-profile", "ZERO", "board_view", None, "all",
        (0.0, 0.0, 0.0), (0.010, 0.010, 0.012), 40, 0.95)
    own = builder.BoundsModel((row(builder.SURFACE_CORRECTION),), (0.45, 0.5, 0.55, 0.6), "a" * 64, "test:model")
    scene = build(tmp_path, bounds=own, surface_correction=builder.SURFACE_CORRECTION)
    camera = camera_of(tmp_path)
    for obj in scene.objects:
        ray = np.array(obj.surface_point_m) - camera
        _, spread = builder.class_size(LAYOUT, obj.kind).along(ray / np.linalg.norm(ray))
        assert spread > 0
        assert obj.bounds_m == pytest.approx((0.010 + spread, 0.010 + spread, 0.012 + spread))
        assert not obj.bounds_provisional   # 40 distinct sightings >= 30: a non-provisional row


def test_a_surface_correction_with_no_size_table_for_the_rig_is_refused(tmp_path):
    with pytest.raises(builder.SceneRefused, match="no size table"):
        build(tmp_path, options=builder.RigOptions(rig_layout_id="SomewhereElse", scene_name=LAYOUT),
              surface_correction=builder.SURFACE_CORRECTION)
    # without a correction the size table is not needed
    build(tmp_path / "ok", options=builder.RigOptions(rig_layout_id="SomewhereElse", scene_name=LAYOUT))


def test_the_sim_rig_agrees_with_the_viewers_torso_transform():
    profile = json.loads((GRADE / "pair_ok" / "depth.json").read_text())["profile"]
    neck = (3.0, 40.0, -12.0)
    rig = builder.SimRig(profile, neck)
    point = np.array([0.05, -0.1, 0.55])
    pose = {n: {"present_deg": v} for n, v in zip(("neck_roll", "neck_pitch", "neck_yaw"), neck)}
    np.testing.assert_allclose(rig.to_torso(point), camera_to_torso(point, pose), atol=1e-9)
    np.testing.assert_allclose(rig.to_camera(rig.to_torso(point)), point, atol=1e-9)


# ---------------------------------------------------------------- bad captures refuse

@pytest.mark.parametrize("zoom, why", [
    ((None, "ZERO"), "could not be read"), (("ZERO", None), "could not be read"),
    (("nope", "nope"), "could not be read"), (("OUT", "IN"), "eyes disagree"), (("ZERO", "OUT"), "eyes disagree")])
def test_a_bad_zoom_refuses(tmp_path, zoom, why):
    with pytest.raises(builder.SceneRefused, match=why):
        build(tmp_path, {"zoom": zoom})


def test_a_missing_zoom_read_back_refuses(tmp_path):
    folder = make_bundle(tmp_path)
    pair = json.loads((folder / "pair.json").read_text())
    del pair["camera_context"]["zoom"]
    (folder / "pair.json").write_text(json.dumps(pair))
    with pytest.raises(builder.SceneRefused, match="no camera_context.zoom"):
        builder.build_scene(builder.load_pair_bundle(folder), options=OPTIONS, board=TWIN, code=CODE)


@pytest.mark.parametrize("matches", [None, False])
def test_a_robot_capture_whose_zoom_does_not_match_its_profile_refuses(tmp_path, matches):
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    options = builder.RigOptions(rig_layout_id="FWDCenterLabMCC", profile_file=str(profile))
    with pytest.raises(builder.SceneRefused, match="matches_profile"):
        build(tmp_path, {"source": "robot", "zoom": ("OUT", "OUT"), "matches_profile": matches,
                         "calibration_id": "abc@zoom=OUT"}, options=options)


def test_a_robot_capture_has_no_torso_positions_so_it_has_no_objects(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text('{"id": "abc"}')
    folder = make_bundle(tmp_path, source="robot", zoom=("OUT", "OUT"), matches_profile=True,
                         calibration_id="abc@zoom=OUT")
    depth = json.loads((folder / "depth.json").read_text())
    for m in depth["measurements"]:
        m.pop("xyz_torso_m", None)
    (folder / "depth.json").write_text(json.dumps(depth))
    options = builder.RigOptions(rig_layout_id="FWDCenterLabMCC", profile_file=str(profile))
    scene = builder.build_scene(builder.load_pair_bundle(folder), options=options, code=CODE)
    assert scene.rig.source == "robot" and scene.rig.zoom_level == "OUT"
    assert scene.rig.intrinsics_sha256 == __import__("hashlib").sha256(profile.read_bytes()).hexdigest()
    assert scene.objects == () and scene.board is None and scene.cells == ()
    assert any("torso" in w for w in scene.warnings)
    with pytest.raises(builder.SceneRefused, match="torso frame"):
        builder.build_scene(builder.load_pair_bundle(folder), options=options, code=CODE,
                            surface_correction=builder.SURFACE_CORRECTION)


def test_a_robot_capture_needs_a_profile_file_and_a_layout(tmp_path):
    kw = {"source": "robot", "zoom": ("OUT", "OUT"), "matches_profile": True, "calibration_id": "abc@zoom=OUT"}
    with pytest.raises(builder.SceneRefused, match="--profile-file"):
        build(tmp_path, kw, options=builder.RigOptions(rig_layout_id="FWDCenterLabMCC"))


def test_files_that_disagree_refuse(tmp_path):
    with pytest.raises(builder.SceneRefused, match="disagree"):
        build(tmp_path, {"calibration_id": "someone-else"})
    folder = make_bundle(tmp_path / "n")
    depth = json.loads((folder / "depth.json").read_text())
    depth["head_pose_used"]["neck_pitch"]["present_deg"] = 30.0
    (folder / "depth.json").write_text(json.dumps(depth))
    with pytest.raises(builder.SceneRefused, match="neck"):
        builder.build_scene(builder.load_pair_bundle(folder), options=OPTIONS, board=TWIN, code=CODE)


def test_a_bundle_with_a_missing_file_or_an_altered_zip_refuses(tmp_path):
    folder = make_bundle(tmp_path)
    (folder / "depth.json").unlink()
    with pytest.raises(builder.SceneRefused, match="depth.json is missing"):
        builder.load_pair_bundle(folder)
    with pytest.raises(builder.SceneRefused):
        builder.load_pair_bundle(tmp_path / "does-not-exist.zip")
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(builder.SceneRefused, match="not a zip"):
        builder.load_pair_bundle(bad)


def test_the_head_pose_name_comes_from_the_read_back():
    assert builder.head_pose_name((0.0, 64.78, 0.0)) == "board_view"
    assert builder.head_pose_name((0.0, 64.0, -5.0)) == "sweep_yaw_right_5"
    assert builder.head_pose_name((10.0, 30.0, 0.0)) == "custom"


# ---------------------------------------------------------------- the board

def sweep_zip(poses):
    run = proc.CaptureRun(proc.get_procedure("head_sweep"))
    for label, folder in poses:
        meta = json.loads((folder / "pair.json").read_text())
        run.add_capture(label, meta, (folder / "left.jpg").read_bytes(), (folder / "right.jpg").read_bytes())
    return run.build_zip({})


def test_a_head_sweep_gives_a_board_pose_near_the_scene_grid(tmp_path):
    fixtures = ROOT / "tests" / "fixtures" / "sim_head_sweep_v2"
    names = ("sweep_yaw_right_5", "sweep_yaw_right_8", "sweep_yaw_left_8", "sweep_roll_right_5", "sweep_roll_left_5")
    path = tmp_path / "sweep.zip"
    path.write_bytes(sweep_zip([("head_" + n, fixtures / n) for n in names]))
    board, notes = builder.board_from_head_sweep(path)
    assert board.source == "head_sweep_fit" and board.poses_used == 5
    assert np.linalg.norm(np.array(board.position_m) - GRID_CENTRE) < 0.010
    assert 0.0 < board.spread_mm < 10.0 and board.fit_rms_mm >= 0.0
    assert board.quat_wxyz == (1.0, 0.0, 0.0, 0.0) and all(b > 0 for b in board.bounds_m)
    assert any("axis-aligned" in n for n in notes)


def test_a_one_pose_sweep_is_a_single_pair_fit_and_a_robot_sweep_is_refused_for_a_sim_pair(tmp_path):
    path = tmp_path / "one.zip"
    path.write_bytes(sweep_zip([("head_board_view", ROOT / "tests" / "fixtures" / "sim_head_sweep")]))
    board, _ = builder.board_from_head_sweep(path)
    assert board.source == "single_pair_fit" and board.poses_used == 1 and board.spread_mm == 0.0
    with pytest.raises(builder.SceneRefused, match="robot capture"):
        builder.board_from_head_sweep(path, expected_source="robot")
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    fixture = ROOT / "tests" / "fixtures" / "sim_head_sweep"
    run.add_capture("zoom_out", json.loads((fixture / "pair.json").read_text()), (fixture / "left.jpg").read_bytes(),
                    (fixture / "right.jpg").read_bytes())
    zoom = tmp_path / "x.zip"
    zoom.write_bytes(run.build_zip({}))
    with pytest.raises(builder.SceneRefused, match="not a head sweep"):
        builder.board_from_head_sweep(zoom)


def test_the_pair_s_own_images_can_fit_the_board(tmp_path):
    folder = make_bundle(tmp_path, graded=True)
    fixture = ROOT / "tests" / "fixtures" / "sim_head_sweep"
    for eye in ("left", "right"):
        shutil.copy(fixture / f"{eye}.jpg", folder / f"{eye}.jpg")
    bundle = builder.load_pair_bundle(folder)
    neck = (0.0, 64.78, 0.0)    # the fixture's own read-back
    board, notes = builder.board_from_pair(bundle, neck)
    assert board is not None, notes
    assert board.source == "single_pair_fit" and np.linalg.norm(np.array(board.position_m) - GRID_CENTRE) < 0.010
    scene = builder.build_scene(bundle, options=OPTIONS, code=CODE)     # no board given: it fits one
    # the pair's own neck read-back is (0, 64, 0)
    assert scene.board == builder.board_from_pair(bundle, (0.0, 64.0, 0.0))[0] and len(scene.cells) == 9
    stub = builder.load_pair_bundle(make_bundle(tmp_path, name="stub"))
    assert builder.board_from_pair(stub, (0.0, 64.0, 0.0))[0] is None    # stub images: no grid, no board, no failure


# ---------------------------------------------------------------- identity and the CLI

def test_the_rig_and_pipeline_come_from_the_bundle_and_the_options(tmp_path):
    scene = build(tmp_path, {"graded": True})
    assert scene.rig == sc.RigIdentity(1, "sim", LAYOUT, "october-2026", RESULT_PROFILE, "ZERO", RESULT_PROFILE, 0,
                                       LAYOUT, "7" * 64, "initial")
    assert scene.pipeline.detector_model_sha256 == RESULT_DETECTOR
    assert (scene.pipeline.code_sha, scene.pipeline.dirty) == CODE
    assert scene.objects[0].source.neck_deg == (0.0, 64.0, 0.0) and scene.objects[0].calibration_id == RESULT_PROFILE
    dirty = build(tmp_path / "d", code=("", True))
    assert dirty.pipeline.code_sha == "" and dirty.pipeline.dirty
    sha, is_dirty = builder.code_identity()
    assert isinstance(is_dirty, bool) and (sha == "" or len(sha) == 40)


def test_the_cli_writes_a_valid_scene_and_exits_2_on_a_refused_capture(tmp_path, capsys):
    folder = make_bundle(tmp_path)
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"scene_name": LAYOUT, "scene_sha256": "7" * 64, "scene_revision": "initial"}))
    out = tmp_path / "scene.json"
    args = [str(folder), "--truth", str(truth), "--out", str(out), "--captured-utc", "2026-10-21T03:00:19Z"]
    assert cli.main(args) == 0
    scene = sc.loads(out.read_text())
    assert scene.rig.scene_revision == "initial" and scene.objects[0].source.captured_utc == "2026-10-21T03:00:19Z"
    bad = make_bundle(tmp_path / "bad", zoom=("OUT", "IN"))
    assert cli.main([str(bad), "--truth", str(truth), "--out", str(tmp_path / "no.json")]) == 2
    assert "eyes disagree" in capsys.readouterr().err and not (tmp_path / "no.json").exists()
    with pytest.raises(SystemExit):
        cli.main([str(folder), "--bounds", str(RESULT_186)])      # --bounds needs --bounds-zoom
