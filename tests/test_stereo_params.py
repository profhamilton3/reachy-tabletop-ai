"""stereo_params.json must agree with the versioned profile, never the rejected npz (#28 A); operator-script helpers."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_stereo_params as gen  # noqa: E402

PROFILE = json.loads(gen.PROFILE.read_text())
PARAMS = json.loads(gen.OUTPUT.read_text())


def test_committed_params_equal_what_the_generator_makes_from_the_profile():
    assert PARAMS == gen.params_from_profile(PROFILE)


def test_params_and_profile_agree_value_by_value():
    for side, eye in (("left", "camera_left"), ("right", "camera_right")):
        k = np.array(PROFILE[side]["K"])
        assert (PARAMS[eye]["fx"], PARAMS[eye]["fy"], PARAMS[eye]["cx"], PARAMS[eye]["cy"]) == (
            k[0, 0], k[1, 1], k[0, 2], k[1, 2])
        assert PARAMS[eye]["distortion"] == PROFILE[side]["dist"]
    np.testing.assert_array_equal(PARAMS["R"], PROFILE["R"])
    np.testing.assert_array_equal(PARAMS["T"], PROFILE["T"])
    assert PARAMS["baseline_m"] == PROFILE["baseline_m"] == pytest.approx(0.078, abs=0.001)
    assert np.linalg.norm(PARAMS["T"]) == pytest.approx(PARAMS["baseline_m"], abs=0.002)
    assert (PARAMS["image_width"], PARAMS["image_height"]) == tuple(PROFILE["image_size"]) == (480, 640)
    assert PARAMS["source_profile"] == PROFILE["id"]
    assert PARAMS["zoom_level"] == PROFILE["zoom_level"] and PARAMS["zoom_caveat"]


def test_params_are_not_the_rejected_npz_values():
    # The rejected right camera: fx 983, distortion up to 7.4, eye 0.31 m ahead of the left.
    assert PARAMS["camera_right"]["fx"] < 500 and max(abs(v) for v in PARAMS["camera_right"]["distortion"]) < 1
    assert abs(PARAMS["T"][2]) < 0.05


def test_generator_check_flags_a_stale_file(tmp_path):
    stale = tmp_path / "stereo_params.json"
    stale.write_text(json.dumps({**PARAMS, "baseline_m": 0.0621}))
    sys.argv = ["generate_stereo_params.py", "--check", "--output", str(stale)]
    assert gen.main() == 1


def test_a_non_profile_is_refused():
    with pytest.raises(ValueError):
        gen.params_from_profile({"format": "something_else"})


def test_cell_pitch_px_on_an_existing_picture_and_ratio_summary():
    import zoom_fov_check as zfc
    image = cv2.imread(str(ROOT / "tests" / "Pictures" / "Left Camera_screenshot_20.08.2026.png"))
    pitch = zfc.cell_pitch_px(image)
    assert 60 < pitch < 100  # ~79 px between adjacent 6 in cell centres at the August zoom
    assert zfc.cell_pitch_px(np.zeros((640, 480, 3), np.uint8)) is None
    rows = zfc.summarize([{"level": "OUT", "eye": "left", "pitch_px": 80.0},
                          {"level": "IN", "eye": "left", "pitch_px": 160.0},
                          {"level": "IN", "eye": "right", "pitch_px": None}])
    assert [r["pitch_vs_widest"] for r in rows] == [1.0, 2.0, None]


def test_ruler_evaluation_uses_the_15_mm_target():
    import ruler_depth_test as ruler
    assert ruler.evaluate(0.5, 0.512)["within_target"] is True
    assert ruler.evaluate(0.5, 0.516)["within_target"] is False
    assert ruler.evaluate(1.2, 1.2155)["within_target"] is False
    assert ruler.evaluate(0.8, 0.79)["error_mm"] == -10.0
    assert ruler.DISTANCES_M == (0.3, 0.5, 0.8, 1.2)
