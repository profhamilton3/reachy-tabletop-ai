"""The analysers on a real sim render: board_view of the 2026-10-20 head sweep (cleared board, 640 x 480).

Sim frames are 640 x 480 with the sim's own lens, not the robot's 480 x 640 calibration (#39)."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

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
