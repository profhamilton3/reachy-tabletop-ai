"""The analysers on a real sim render: board_view of the 2026-10-20 head sweep (cleared board, 640 x 480).

Sim frames are 640 x 480 with the sim's own lens, not the robot's 480 x 640 calibration (#39)."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from reachy_ai.perception import procedures as proc
from reachy_ai.perception.stereo_depth import PairDepth

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_capture_bundle as analyse  # noqa: E402
import derive_stereo_calibration as grid  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "sim_head_sweep"
#: The scene grid centre in the torso frame (ADR-0010) and the agreement the analyser must reach.
SCENE_GRID_CENTRE_M = np.array([0.2258, 0.0, -0.4095])
CENTRE_TOLERANCE_MM = 10.0


def pair_files():
    return [FIXTURE / "left.jpg", FIXTURE / "right.jpg"]


def sim_bundle():
    run = proc.CaptureRun(proc.get_procedure("head_sweep"))
    meta = json.loads((FIXTURE / "pair.json").read_text())
    run.add_capture("head_board_view", meta, *(p.read_bytes() for p in pair_files()))
    return run.build_zip({})


def test_the_grid_is_found_in_both_eyes_with_the_sim_geometry():
    depth = PairDepth("sim", "distorted")
    for eye, path in zip(("left", "right"), pair_files()):
        found = grid.board_points(cv2.imread(str(path)), eye, depth.geometry)
        assert found is not None and len(found[0]) == 36  # 9 cells x 4 corners


def test_the_robot_calibration_misplaces_the_corners_of_a_sim_frame():
    geometry = PairDepth("sim", "distorted").geometry
    for eye, path in zip(("left", "right"), pair_files()):
        image = cv2.imread(str(path))
        sim, robot = grid.board_points(image, eye, geometry), grid.board_points(image, eye)
        # On the cleared board the finder still sees the cells with the wrong lens, but every corner is off.
        assert robot is None or np.linalg.norm(sim[1] - robot[1], axis=1).mean() > 5.0


def test_a_new_bundle_records_its_camera_source():
    _, files, capture = analyse.load(sim_bundle())
    assert capture["source"] == "sim" and capture["sim_lens"] == "distorted"
    assert analyse.bundle_geometry(capture, files) == ("sim", "distorted", "capture.json")


def test_an_older_bundle_gets_its_geometry_from_the_pair_json():
    _, files, capture = analyse.load(sim_bundle())
    capture.pop("source"), capture.pop("sim_lens")
    assert analyse.bundle_geometry(capture, files) == ("sim", "distorted", "pair.json analysis_profile")
    assert analyse.bundle_geometry({"steps": []}, {})[0] == "robot"


def test_the_head_sweep_analyser_reports_a_centre_for_board_view_close_to_the_scene_grid():
    _, files, capture = analyse.load(sim_bundle())
    source, lens, _ = analyse.bundle_geometry(capture, files)
    result = analyse.analyze_head(capture, files, PairDepth(source, lens))
    assert list(result["poses"]) == ["Head: board_view"]
    row = result["rows"][0]
    assert row["corners"] == 36 and "reason" not in row
    centre = np.array(result["poses"]["Head: board_view"]["centre_torso_m"])
    error_mm = np.linalg.norm(centre - SCENE_GRID_CENTRE_M) * 1000
    assert error_mm < CENTRE_TOLERANCE_MM, (centre, error_mm)


# --- #41: a partly hidden near row (Head sweep v2 sim re-shoot, cap1 of every pose that lost the grid) ---

FIXTURE_V2 = ROOT / "tests" / "fixtures" / "sim_head_sweep_v2"
V2_POSES = ("sweep_yaw_right_5", "sweep_yaw_right_8", "sweep_yaw_left_8", "sweep_roll_right_5", "sweep_roll_left_5")


def v2_images(pose):
    return [cv2.imread(str(FIXTURE_V2 / pose / f"{eye}.jpg")) for eye in ("left", "right")]


@pytest.mark.parametrize("pose", V2_POSES)
def test_the_grid_is_found_in_both_eyes_at_every_v2_sweep_pose(pose):
    geometry = PairDepth("sim", "distorted").geometry
    for eye, image in zip(("left", "right"), v2_images(pose)):
        found = grid.board_points_detail(image, eye, geometry, near_row_hidden=True)
        assert found is not None, (pose, eye)
        assert found["rows"] in ([1, 2], [0, 1, 2]) and len(found["obj"]) == 4 * found["cells"] in (24, 36)


@pytest.mark.parametrize("pose", V2_POSES)
def test_the_whole_rows_finder_still_loses_one_eye_unless_asked(pose):
    geometry = PairDepth("sim", "distorted").geometry
    lost = [grid.board_points(image, eye, geometry) is None for eye, image in zip(("left", "right"), v2_images(pose))]
    assert any(lost), "the strict finder is the default and must not change"


def v2_bundle():
    run = proc.CaptureRun(proc.get_procedure("head_sweep"))
    for pose in V2_POSES:
        meta = json.loads((FIXTURE_V2 / pose / "pair.json").read_text())
        eyes = ((FIXTURE_V2 / pose / f"{eye}.jpg").read_bytes() for eye in ("left", "right"))
        run.add_capture(f"head_{pose}", meta, *eyes)
    return run.build_zip({})


def test_the_board_centre_at_every_v2_sweep_pose_is_within_10_mm_of_the_scene_grid():
    _, files, capture = analyse.load(v2_bundle())
    source, lens, _ = analyse.bundle_geometry(capture, files)
    result = analyse.analyze_head(capture, files, PairDepth(source, lens))
    assert sorted(result["poses"]) == sorted(f"Head: {p}" for p in V2_POSES)
    for label, pose in result["poses"].items():
        error_mm = np.linalg.norm(np.array(pose["centre_torso_m"]) - SCENE_GRID_CENTRE_M) * 1000
        assert error_mm < CENTRE_TOLERANCE_MM, (label, error_mm)
    assert any(r.get("cells_note") == "6 of 9 cells (near row hidden)" for r in result["rows"])


def test_the_board_view_fixture_still_gives_36_corners_with_the_fallback_on():
    geometry = PairDepth("sim", "distorted").geometry
    for eye, path in zip(("left", "right"), pair_files()):
        found = grid.board_points_detail(cv2.imread(str(path)), eye, geometry, near_row_hidden=True)
        assert len(found["obj"]) == 36 and found["rows"] == [0, 1, 2]


def test_the_august_robot_pairs_give_the_same_corners_wherever_all_three_rows_are_found():
    compared = 0
    for _, left, right in grid.pair_files():
        for eye, path in (("left", left), ("right", right)):
            strict = grid.board_points_detail(path, eye)
            relaxed = grid.board_points_detail(path, eye, near_row_hidden=True)
            if strict is not None and strict["rows"] == [0, 1, 2]:
                compared += 1
                np.testing.assert_array_equal(relaxed["obj"], strict["obj"])
                np.testing.assert_array_equal(relaxed["img"], strict["img"])
            if strict is not None:  # a partial result of the old finder is kept too
                np.testing.assert_array_equal(relaxed["img"], strict["img"])
    assert compared >= 8
