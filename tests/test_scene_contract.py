"""P0: the perceived_scene.v1 contract (docs/design/perceived-scene.md section 1). Offline; no robot, no sim."""
import ast
import copy
import dataclasses
import json
from pathlib import Path

import pytest

from reachy_ai.perception import scene_contract as sc

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "perceived_scene"
MODULE = ROOT / "src" / "reachy_ai" / "perception" / "scene_contract.py"

VALID = sorted((FIXTURES / "valid").glob("*.json"))
INVALID = sorted((FIXTURES / "invalid").glob("*.json"))

#: invalid fixture -> a phrase its refusal must contain (so each is refused for the stated reason).
REASONS = {
    "unknown_field": "unknown field(s): extra",
    "unknown_nested_field": "RigIdentity: unknown field(s)",
    "missing_field": "missing field(s): off_board",
    "wrong_schema_version": "schema_version",
    "object_quaternion_not_unit": "PerceivedObject.quat_wxyz: quaternion",
    "board_quaternion_not_unit": "BoardPose.quat_wxyz: quaternion",
    "eight_cells": "exactly the nine cells",
    "ten_cells": "exactly the nine cells",
    "cells_out_of_order": "exactly the nine cells",
    "cells_without_board": "has no cells",
    "bad_zoom_level": "zoom_level",
    "robot_calibration_id_zoom_mismatch": "does not carry the zoom_level",
    "sim_field_on_robot_rig": "sim-only",
    "nan_value": "NaN",
    "infinity_value": "Infinity",
    "overflow_value": "must be finite",
    "null_bound_not_provisional": "must be finite",
    "negative_bound": "negative",
    "boolean_as_number": "expected a number",
    "confidence_above_one": "outside [0, 1]",
    "bad_kind": "PerceivedObject.kind",
    "size_m_with_unknown_size_source": "size_source unknown cannot have a size_m",
    "cell_points_at_wrong_kind": "but the object is a",
    "cell_object_index_out_of_range": "out of range",
    "unknown_cell_with_object_index": "only for cube or cylinder",
    "object_disagrees_with_its_cell": "says cell",
    "off_board_object_with_cell": "has no cell",
    "object_calibration_differs_from_rig": "differs from the rig",
    "bad_sha256": "64 lowercase hex",
    "empty_code_sha_not_dirty": "needs dirty=true",
    "bad_utc_time": "UTC time",
    "single_pair_fit_with_spread": "single_pair_fit",
    "duplicate_key": "duplicate key",
    "not_an_object": "expected an object",
    "not_json": "not valid JSON",
}


def valid_dict(name="full_board_two_objects"):
    return json.loads((FIXTURES / "valid" / (name + ".json")).read_text())


def refused(data, phrase):
    with pytest.raises(sc.SceneContractError, match=phrase):
        sc.PerceivedScene.from_dict(data)


# ---------------------------------------------------------------- golden fixtures

@pytest.mark.parametrize("path", VALID, ids=lambda p: p.stem)
def test_a_valid_fixture_loads_and_round_trips_byte_for_byte(path):
    text = path.read_text()
    scene = sc.loads(text)
    assert sc.dumps(scene) == text
    assert sc.loads(sc.dumps(scene)) == scene


@pytest.mark.parametrize("path", INVALID, ids=lambda p: p.stem)
def test_an_invalid_fixture_is_refused_for_its_stated_reason(path):
    with pytest.raises(sc.SceneContractError, match=REASONS[path.stem].replace("[", r"\[").replace("(", r"\(")
                       .replace(")", r"\)")):
        sc.loads(path.read_text())


def test_every_invalid_fixture_has_a_stated_reason_and_the_readme_lists_every_fixture():
    assert {p.stem for p in INVALID} == set(REASONS)
    readme = (FIXTURES / "README.md").read_text()
    for path in VALID + INVALID:
        assert "`%s`" % path.name in readme, path.name


def test_the_full_fixture_covers_the_cases_the_sim_must_handle():
    scene = sc.loads((FIXTURES / "valid" / "full_board_two_objects.json").read_text())
    assert [c.name for c in scene.cells] == list(sc.CELL_NAMES)
    assert {c.state for c in scene.cells} == {"empty", "unknown", "cube", "cylinder"}
    finite, infinite = scene.objects
    assert not finite.bounds_provisional and all(b < 1 for b in finite.bounds_m)
    assert infinite.bounds_provisional and all(b == float("inf") for b in infinite.bounds_m)
    assert len(scene.off_board) == 1


# ---------------------------------------------------------------- refusals, one per reason (section 7 exit gate)

def test_an_unknown_field_is_refused_at_every_level():
    data = valid_dict()
    for path in ((), ("rig",), ("pipeline",), ("board",), ("cells", 0), ("objects", 0), ("objects", 0, "source")):
        bad = copy.deepcopy(data)
        node = bad
        for key in path:
            node = node[key]
        node["surprise"] = 1
        refused(bad, "unknown field")


def test_nan_and_infinity_are_refused_in_a_value_and_in_text():
    scene = sc.loads((FIXTURES / "valid" / "full_board_two_objects.json").read_text())
    for field, bad in (("kind_confidence", float("nan")), ("depth_iqr_m", float("inf")),
                       ("centre_m", (float("nan"), 0.0, 0.0)), ("surface_point_m", (0.0, float("inf"), 0.0))):
        with pytest.raises(sc.SceneContractError):
            dataclasses.replace(scene.objects[0], **{field: bad})
    with pytest.raises(sc.SceneContractError, match="NaN"):
        sc.loads('{"schema_version": NaN}')


def test_a_quaternion_must_be_unit_within_1e_minus_6():
    board = sc.loads((FIXTURES / "valid" / "full_board_two_objects.json").read_text()).board
    assert dataclasses.replace(board, quat_wxyz=(1.0 + 5e-7, 0.0, 0.0, 0.0))
    for q in ((1.0 + 2e-6, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)):
        with pytest.raises(sc.SceneContractError, match="unit norm"):
            dataclasses.replace(board, quat_wxyz=q)


@pytest.mark.parametrize("count", [0, 1, 8, 10])
def test_a_board_with_other_than_nine_cells_is_refused(count):
    data = valid_dict()
    data["cells"] = (data["cells"] * 2)[:count]
    refused(data, "exactly the nine cells")


def test_a_scene_without_a_board_may_have_no_cells_but_not_any():
    data = valid_dict("no_board_provisional_object")
    assert sc.PerceivedScene.from_dict(data).cells == ()
    data["cells"] = valid_dict()["cells"]
    refused(data, "has no cells")


@pytest.mark.parametrize("zoom", ["", "unknown", "out", "OUT ", None, 3, "UNREADABLE"])
def test_a_bad_zoom_is_refused(zoom):
    data = valid_dict()
    data["rig"]["zoom_level"] = zoom
    refused(data, "zoom_level")


def test_the_four_zoom_read_backs_are_accepted():
    for zoom in sc.ZOOM_LEVELS:
        data = valid_dict()
        data["rig"]["zoom_level"] = zoom
        assert sc.PerceivedScene.from_dict(data).rig.zoom_level == zoom


def test_an_infinite_bound_is_legal_only_for_a_provisional_object():
    scene = sc.loads((FIXTURES / "valid" / "full_board_two_objects.json").read_text())
    finite = scene.objects[0]
    with pytest.raises(sc.SceneContractError, match="finite"):
        dataclasses.replace(finite, bounds_m=(float("inf"),) * 3)
    assert dataclasses.replace(finite, bounds_m=(float("inf"),) * 3, bounds_provisional=True)
    # null on the wire is the same thing
    assert sc.PerceivedObject.from_dict(dict(finite.to_dict(), bounds_m=[None] * 3, bounds_provisional=True)
                                        ).bounds_m == (float("inf"),) * 3
    assert dumps_has_null_bounds(scene.objects[1])


def dumps_has_null_bounds(obj):
    return obj.to_dict()["bounds_m"] == [None, None, None]


def test_the_scene_is_frozen_and_hand_built_objects_are_validated_too():
    scene = sc.loads((FIXTURES / "valid" / "full_board_two_objects.json").read_text())
    with pytest.raises(dataclasses.FrozenInstanceError):
        scene.scene_id = "x"
    with pytest.raises(sc.SceneContractError):
        dataclasses.replace(scene.rig, rig_epoch=-1)
    with pytest.raises(sc.SceneContractError):
        dataclasses.replace(scene, scene_id="not-hex")


def test_match_fields_are_all_rig_identity_fields():
    assert set(sc.MATCH_FIELDS) <= {f.name for f in dataclasses.fields(sc.RigIdentity)}


# ---------------------------------------------------------------- Python 3.8 (the sim's ros:foxy container)

def module_tree():
    return ast.parse(MODULE.read_text(), feature_version=(3, 8))


def test_the_module_parses_as_python_3_8():
    module_tree()


def test_no_pep_604_unions_or_builtin_generics_that_python_3_8_cannot_evaluate():
    """ast feature_version does not reject `X | Y` or `list[int]`; both fail at runtime on 3.8."""
    tree = module_tree()
    unions = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr)]
    generics = [n.lineno for n in ast.walk(tree)
                if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and n.value.id in {"list", "dict", "tuple", "set", "frozenset", "type"}]
    assert not unions, "X | Y at line(s) %s" % unions
    assert not generics, "builtin generic at line(s) %s" % generics


def test_the_module_imports_only_the_standard_library():
    """It is vendored into the sim, which has no reachy_ai, numpy or cv2 in that container."""
    tree = module_tree()
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            modules.add(node.module.split(".")[0])
    assert modules <= {"json", "math", "re", "dataclasses", "typing"}, modules
