"""T3 board reader (issue #51): reference, change score, square assignment, refusals. Sim fixtures; moves nothing."""
import ast
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from reachy_ai.games import t3_board_reader as reader
from reachy_ai.games import tictactoe as t3

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
EMPTY_REFERENCE = FIXTURES / "sim_head_sweep"      # an empty board at board_view
T3 = FIXTURES / "t3"


@pytest.fixture(scope="module")
def reference():
    return reader.build_reference(EMPTY_REFERENCE)


def truth_board(folder):
    cells = [t3.EMPTY] * 9
    for obj in json.loads((T3 / folder / "truth_before.json").read_text())["objects"]:
        if obj["cell"]:
            cells[t3.CELLS.index(obj["cell"])] = t3.X if obj["kind"] == "cube" else t3.O
    return tuple(cells)


def synthetic(tmp_path, detections=(), paint=None, mutate=None, shape="square"):
    """A copy of the empty reference pair with an analysis.json; `paint` = cell name gets a grey square on the image."""
    folder = tmp_path / "pair"
    shutil.copytree(EMPTY_REFERENCE, folder)
    reference = reader.build_reference(folder)
    if paint:
        image = cv2.imread(str(folder / "right.jpg"))
        geometry = reader._geometry(reader._identity(json.loads((folder / "pair.json").read_text())))
        cx, cy = reader.to_raw(reference.quads[paint].mean(axis=0, keepdims=True), geometry)[0].astype(int)
        if shape == "square":
            cv2.rectangle(image, (cx - 22, cy - 22), (cx + 22, cy + 22), (128, 128, 128), -1)
        elif shape == "round":
            cv2.circle(image, (cx, cy), 26, (128, 128, 128), -1)
        else:  # "thin": a bar too small to be a piece
            cv2.rectangle(image, (cx - 30, cy - 3), (cx + 30, cy + 3), (128, 128, 128), -1)
        cv2.imwrite(str(folder / "right.jpg"), image)
        box = [int(cx - 25), int(cy - 25), 50, 50]
    else:
        box = None
    resolved = [{**d, "box_xywh": box if d.get("box_xywh") == "paint" else d["box_xywh"], "class_id": 0}
                for d in detections]
    (folder / "analysis.json").write_text(json.dumps({"eye": "right", "detections": resolved}))
    if mutate:
        pair = json.loads((folder / "pair.json").read_text())
        mutate(pair)
        (folder / "pair.json").write_text(json.dumps(pair))
    return folder


# --- reference ---------------------------------------------------------------------------------------------------

def test_reference_builds_with_nine_quads(reference):
    assert set(reference.quads) == set(t3.CELLS)
    assert reference.pose == "board_view" and reference.source == "sim" and reference.zoom == "ZERO"
    assert reference.calibration_id == "771392183ddc6c62"
    r1c1, r1c3, r3c1 = (reference.quads[c].mean(axis=0) for c in ("r1c1", "r1c3", "r3c1"))
    assert r1c1[0] < r1c3[0]      # c1 is on Reachy's left, so on the image's left
    assert r1c1[1] > r3c1[1]      # r1 is nearest, so lower in the image


def test_reference_refuses_when_grid_not_found(tmp_path):
    folder = tmp_path / "blank"
    shutil.copytree(EMPTY_REFERENCE, folder)
    cv2.imwrite(str(folder / "right.jpg"), np.full((480, 640, 3), 90, np.uint8))
    with pytest.raises(reader.BoardReadRefused, match="did not find the board"):
        reader.build_reference(folder)


def test_pixel_spaces_round_trip(reference):
    geometry = reader._geometry(reader._identity({"source": "sim", "camera_context": {
        "analysis_profile": {"source": "sim", "lens": "distorted"},
        "settings_at_connection": {"left": {"zoom_level": "ZERO"}, "right": {"zoom_level": "ZERO"}}}}))
    raw = np.array([[40.0, 60.0], [320.0, 240.0], [600.0, 420.0], [114.0, 384.0]])
    back = reader.to_raw(reader.to_undistorted(raw, geometry), geometry)
    assert np.abs(back - raw).max() < 0.05
    assert np.abs(reader.to_undistorted(raw, geometry) - raw).max() > 1   # the lens really bends the corners


# --- reading ---------------------------------------------------------------------------------------------------

def test_empty_board_reads_all_empty(reference):
    result = reader.read_board(T3 / "A_default", reference)
    assert result.board_string() == ".../.../..."
    assert all(s.change_score < 0.1 and s.state == t3.EMPTY for s in result.squares)
    assert result.settings["change_threshold"] == reader.ReaderSettings().change_threshold


def suggestions(result):
    return {sq.cell: sq.suggested for sq in result.squares if sq.state == t3.UNKNOWN}


def test_populated_d_full_reads_truth_cells(reference):
    result = reader.read_board(T3 / "D_full", reference)
    truth = truth_board("D_full")
    assert truth == t3.parse("XOX/OXO/XOX")
    assert result.board_string() == "?O?/O?O/?O?"
    assert all(s.change_score > 0.5 for s in result.squares)    # every square holds a piece
    assert all(r != t3.EMPTY for r in result.board)             # never read as empty
    assert [c for c, got, want in zip(t3.CELLS, result.board, truth) if got in (t3.X, t3.O) and got != want] == []
    # the detector calls the cube on r3c1 a cylinder (0.50): the shape disagrees, so it is unknown, suggesting X
    square = result.squares[t3.CELLS.index("r3c1")]
    assert (square.state, square.suggested, square.suggested_by) == (t3.UNKNOWN, t3.X, "shape")
    assert "shape is square" in square.reason
    # every suggestion made is right
    assert all(got in (None, want) for got, want in zip(
        (suggestions(result).get(c) for c in t3.CELLS), truth))


def test_populated_e_full_swapped_reads_truth_cells(reference):
    result = reader.read_board(T3 / "E_full_swapped", reference)
    truth = truth_board("E_full_swapped")
    assert truth == t3.parse("OXO/XOX/OXO")
    assert result.board_string() == "???/???/??O"
    assert [c for c, got, want in zip(t3.CELLS, result.board, truth) if got in (t3.X, t3.O) and got != want] == []
    found = suggestions(result)
    assert found["r1c1"] is None                                  # shape unsure, detector 0.49
    assert all(found[c] == truth[t3.CELLS.index(c)] for c in found if found[c])
    assert sum(1 for v in found.values() if v) == 7               # the operator confirms these in 1c


def test_evidence_has_detections_shape_reason_and_score(reference):
    square = reader.read_board(T3 / "E_full_swapped", reference).squares[t3.CELLS.index("r3c3")]
    assert square.state == t3.O and square.change_score > 0.5
    assert square.detections[0]["label"] == "cylinder" and square.detections[0]["box_xywh"]
    assert square.shape["class"] == "round" and 0.5 < square.shape["rectangularity"] < 0.845
    assert square.shape["circularity"] > 0 and square.suggested is None
    assert "agree" in square.reason


def test_settings_recorded_with_shape_thresholds(reference):
    settings = reader.read_board(T3 / "A_default", reference).settings
    for key in ("min_label_confidence", "diff_threshold", "square_min_rect", "round_max_rect", "min_blob_area_px"):
        assert key in settings
    assert settings["min_label_confidence"] == 0.75


# --- the unknown rules ------------------------------------------------------------------------------------------

def test_grey_cube_called_empty_is_unknown_not_empty(tmp_path, reference):
    folder = synthetic(tmp_path, [{"label": "empty", "confidence": 0.8, "box_xywh": "paint"}], paint="r2c2")
    square = reader.read_board(folder, reference).squares[t3.CELLS.index("r2c2")]
    assert square.state == t3.UNKNOWN and square.change_score > 0.35
    assert "calls it empty" in square.reason


def test_changed_but_undetected_is_unknown(tmp_path, reference):
    folder = synthetic(tmp_path, [], paint="r2c2")
    result = reader.read_board(folder, reference)
    assert result.squares[t3.CELLS.index("r2c2")].state == t3.UNKNOWN
    assert result.board_string() == ".../.?./..."      # the other eight squares are unchanged, so empty


def test_cube_detection_and_square_shape_agree_is_x(tmp_path, reference):
    folder = synthetic(tmp_path, [{"label": "cube", "confidence": 0.9, "box_xywh": "paint"}], paint="r2c2")
    result = reader.read_board(folder, reference)
    assert result.board_string() == ".../.X./..."
    assert result.squares[4].shape["class"] == "square"


def test_cylinder_detection_and_round_shape_agree_is_o(tmp_path, reference):
    folder = synthetic(tmp_path, [{"label": "cylinder", "confidence": 0.9, "box_xywh": "paint"}], paint="r2c2",
                       shape="round")
    result = reader.read_board(folder, reference)
    assert result.squares[4].shape["class"] == "round"
    assert result.board_string() == ".../.O./..."


def test_detector_and_shape_disagree_is_unknown_with_shape_suggestion(tmp_path, reference):
    folder = synthetic(tmp_path, [{"label": "cylinder", "confidence": 0.95, "box_xywh": "paint"}], paint="r2c2")
    square = reader.read_board(folder, reference).squares[4]
    assert (square.state, square.suggested, square.suggested_by) == (t3.UNKNOWN, t3.X, "shape")


def test_low_confidence_label_is_unknown_with_shape_suggestion(tmp_path, reference):
    folder = synthetic(tmp_path, [{"label": "cube", "confidence": 0.6, "box_xywh": "paint"}], paint="r2c2")
    square = reader.read_board(folder, reference).squares[4]
    assert (square.state, square.suggested) == (t3.UNKNOWN, t3.X)
    assert "below 0.75" in square.reason


def test_clear_shape_without_detection_suggests_the_shape(tmp_path, reference):
    folder = synthetic(tmp_path, [], paint="r2c2", shape="round")
    square = reader.read_board(folder, reference).squares[4]
    assert (square.state, square.suggested, square.suggested_by) == (t3.UNKNOWN, t3.O, "shape")


def test_unsure_shape_confident_detector_suggests_detector(tmp_path, reference):
    folder = synthetic(tmp_path, [{"label": "cube", "confidence": 0.9, "box_xywh": "paint"}], paint="r2c2",
                       shape="thin")
    square = reader.read_board(folder, reference).squares[4]
    assert square.shape["class"] == "unsure"
    assert (square.state, square.suggested, square.suggested_by) == (t3.UNKNOWN, t3.X, "detector")


def test_cube_and_cylinder_on_one_square_is_unknown(tmp_path, reference):
    two = [{"label": "cube", "confidence": 0.7, "box_xywh": "paint"},
           {"label": "cylinder", "confidence": 0.6, "box_xywh": "paint"}]
    folder = synthetic(tmp_path, two, paint="r2c2")
    assert reader.read_board(folder, reference).board_string() == ".../.?./..."


def test_empty_detection_on_unchanged_square_stays_empty(tmp_path, reference):
    folder = synthetic(tmp_path, [{"label": "empty", "confidence": 0.9, "box_xywh": [200, 200, 80, 80]}])
    assert reader.read_board(folder, reference).board_string() == ".../.../..."


def test_piece_detected_on_unchanged_square_is_unknown(tmp_path, reference):
    cx, cy = reader.to_raw(reference.quads["r2c2"].mean(axis=0, keepdims=True),
                           reader._geometry({"source": "sim", "lens": "distorted"}))[0]
    folder = synthetic(tmp_path, [{"label": "cube", "confidence": 0.9,
                                   "box_xywh": [int(cx) - 20, int(cy) - 20, 40, 40]}])
    square = reader.read_board(folder, reference).squares[t3.CELLS.index("r2c2")]
    assert square.state == t3.UNKNOWN and "did not change" in square.reason


def test_low_confidence_detection_is_not_usable(tmp_path, reference):
    folder = synthetic(tmp_path, [{"label": "cube", "confidence": 0.1, "box_xywh": "paint"}], paint="r2c2")
    assert reader.read_board(folder, reference).board_string() == ".../.?./..."


def test_small_head_shift_is_aligned_away(tmp_path, reference):
    folder = synthetic(tmp_path)
    image = cv2.imread(str(folder / "right.jpg"))
    shifted = np.roll(image, (4, -6), axis=(0, 1))
    cv2.imwrite(str(folder / "right.jpg"), shifted)
    assert reader.read_board(folder, reference).board_string() == ".../.../..."


# --- refusals --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("what, mutate, message", [
    ("calibration", lambda p: p["camera_context"].update(calibration_id="deadbeef"), "calibration_id differs"),
    ("zoom", lambda p: p["camera_context"]["settings_at_connection"].update(
        left={"zoom_level": "IN"}, right={"zoom_level": "IN"}), "zoom differs"),
    ("source", lambda p: p["camera_context"]["analysis_profile"].update(source="robot"), "source differs"),
    ("pose", lambda p: p["head_context"]["joints"]["neck_pitch"].update(present_deg=50.0), "head pose differs"),
])
def test_mismatch_refuses(tmp_path, reference, what, mutate, message):
    folder = synthetic(tmp_path, mutate=mutate)
    with pytest.raises(reader.BoardReadRefused, match=message):
        reader.read_board(folder, reference)


def test_pair_without_analysis_refuses(reference):
    with pytest.raises(reader.BoardReadRefused, match="analysis.json is missing"):
        reader.read_board(EMPTY_REFERENCE, reference)


# --- the CLI and the guard ---------------------------------------------------------------------------------------

def test_cli(capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("t3_read_board", ROOT / "scripts" / "t3_read_board.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--reference", str(EMPTY_REFERENCE), "--pair", str(T3 / "E_full_swapped"), "--analyze"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "???/???/??O" and "Invalid board: unknown square(s)" in out
    assert "(suggest X by shape)" in out
    assert module.main(["--reference", str(EMPTY_REFERENCE), "--pair", str(EMPTY_REFERENCE)]) == 2


def test_import_guard_no_motion_code():
    tree = ast.parse((ROOT / "src" / "reachy_ai" / "games" / "t3_board_reader.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    banned = ("reachy_sdk", "reachy2_sdk", "reachy_ai.motion", "reachy_ai.reachy_client", "reachy_ai.stereo_app")
    assert not [name for name in imported if name.startswith(banned)], imported
