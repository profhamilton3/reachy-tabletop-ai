"""#81: the disposition_decision.v1 contract. Offline; no robot, no sim."""
import ast
import dataclasses
import json
from pathlib import Path

import pytest

from reachy_ai.health import precapture
from reachy_ai.planner.schema import Destination
from reachy_ai.reasoner import contract as c

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "disposition"
MODULE = ROOT / "src" / "reachy_ai" / "reasoner" / "contract.py"
LOADERS = {"request": c.loads_request, "decision": c.loads_decision}

VALID = sorted(FIXTURES.glob("valid/*/*.json"))
INVALID = sorted(FIXTURES.glob("invalid/*/*.json"))

#: invalid fixture -> a phrase its refusal must contain (so each is refused for the stated reason).
REASONS = {
    "request/unknown_field": "Request: unknown field(s): extra",
    "request/unknown_nested_field": "Colour: unknown field(s): extra",
    "request/missing_field": "Request: missing field(s): health",
    "request/missing_nullable_field": "Case: missing field(s): expected_design_id",
    "request/wrong_schema_version": "schema_version",
    "request/bad_split": "Case.split",
    "request/bad_kind": "Observation.kind",
    "request/bad_colour_name": "Colour.name",
    "request/bad_lab_lightness": "is outside [0, 100]",
    "request/nan_value": "NaN and infinity are refused",
    "request/infinite_bound": "NaN and infinity are refused",
    "request/negative_bound": "is negative",
    "request/boolean_as_number": "expected a number",
    "request/diameter_without_hole": "only for a present hole",
    "request/cell_ready_on_failed_report": "cell_ready needs a passed pre-capture report",
    "request/robot_sim_truth": "sim_truth input cannot come from the robot",
    "request/robot_degradation": "degradation flags are sim only",
    "request/robot_with_twin": "twin is null on the robot",
    "request/perceived_without_bundle": "needs a capture bundle",
    "request/bad_bundle_sha256": "64 lowercase hex",
    "request/negative_reobservation_count": "below the minimum",
    "request/duplicate_key": "duplicate key",
    "request/not_json": "not valid JSON",
    "decision/unknown_field": "Decision: unknown field(s): extra",
    "decision/disposition_and_refusal": "exactly one of disposition and refusal",
    "decision/neither_disposition_nor_refusal": "exactly one of disposition and refusal",
    "decision/bad_refusal": "Decision.refusal",
    "decision/remeasure_without_route": "Decision.remeasure_route",
    "decision/route_without_remeasure": "only for a remeasure",
    "decision/accept_to_reject_bin": "accept cannot go to right_tray",
    "decision/refusal_with_a_bin": "refusal cannot go to left_tray",
    "decision/no_reasons": "at least one reason",
    "decision/confidence_above_one": "outside [0, 1]",
    "decision/deterministic_with_model": "uses no model or prompt",
    "decision/ai_without_prompt": "needs a model_id and a prompt_sha256",
    "decision/bad_code_sha": "hex digits",
    "decision/operator_rating_out_of_range": "above the maximum 5",
    "decision/adjudication_missing_field": "missing field(s): human_adjudication_notes",
    "decision/not_an_object": "expected an object",
}


def name(path):
    return "%s/%s" % (path.parent.name, path.stem)


def valid_dict(kind="request", stem="robot_perceived_with_colour"):
    return json.loads((FIXTURES / "valid" / kind / (stem + ".json")).read_text())


# ---------------------------------------------------------------- golden fixtures

@pytest.mark.parametrize("path", VALID, ids=name)
def test_a_valid_fixture_loads_and_round_trips_byte_for_byte(path):
    text = path.read_text()
    document = LOADERS[path.parent.name](text)
    assert c.dumps(document) == text
    assert LOADERS[path.parent.name](c.dumps(document)) == document


@pytest.mark.parametrize("path", INVALID, ids=name)
def test_an_invalid_fixture_is_refused_for_its_stated_reason(path):
    with pytest.raises(c.DispositionContractError) as caught:
        LOADERS[path.parent.name](path.read_text())
    assert REASONS[name(path)] in str(caught.value)


def test_every_fixture_is_listed():
    assert {name(p) for p in INVALID} == set(REASONS)
    readme = (FIXTURES / "README.md").read_text()
    for path in VALID + INVALID:
        assert "`%s`" % path.name in readme, path.name


# ---------------------------------------------------------------- strictness

def test_an_unknown_field_is_refused_at_every_level_of_a_request():
    paths = ((), ("case",), ("health",), ("health", "precapture"), ("observation",), ("observation", "height_mm"),
             ("observation", "hole"), ("observation", "hole", "diameter_mm"), ("observation", "sufficiency"),
             ("observation", "bundle"), ("observation", "colour"))
    for path in paths:
        data = valid_dict()
        node = data
        for key in path:
            node = node[key]
        node["surprise"] = 1
        with pytest.raises(c.DispositionContractError, match="unknown field"):
            c.Request.from_dict(data)


def test_an_unknown_field_is_refused_in_the_twin_and_in_a_decision():
    for path in (("twin",), ("twin", "part_pose")):
        data = valid_dict(stem="sim_twin_degraded")
        node = data
        for key in path:
            node = node[key]
        node["surprise"] = 1
        with pytest.raises(c.DispositionContractError, match="unknown field"):
            c.Request.from_dict(data)
    for path in (("policy",), ("reasons", 0), ("adjudication",)):
        data = valid_dict("decision", "ai_configuration_adjudicated")
        node = data
        for key in path:
            node = node[key]
        node["surprise"] = 1
        with pytest.raises(c.DispositionContractError, match="unknown field"):
            c.Decision.from_dict(data)


@pytest.mark.parametrize("feasibility", [{"handover_zone": True}, {"none": True}, {"left_tray": 1}, [], "yes"])
def test_a_bad_path_feasibility_is_refused(feasibility):
    data = valid_dict(stem="sim_twin_degraded")
    data["twin"]["path_feasibility"] = feasibility
    with pytest.raises(c.DispositionContractError):
        c.Request.from_dict(data)
    with pytest.raises(c.DispositionContractError):
        c.Twin(part_pose=None, path_feasibility=3)


def test_a_non_unit_quaternion_and_a_huge_integer_are_refused():
    data = valid_dict(stem="sim_twin_degraded")
    data["twin"]["part_pose"]["quat_wxyz"] = [1.0, 1.0, 0.0, 0.0]
    with pytest.raises(c.DispositionContractError, match="unit norm"):
        c.Request.from_dict(data)
    with pytest.raises(c.DispositionContractError, match="finite"):
        c.Measurement(value=10 ** 400, bound=0.0)


def test_records_are_frozen_and_hand_built_ones_are_validated():
    request = c.loads_request(c.dumps(c.Request.from_dict(valid_dict(stem="sim_twin_degraded"))))
    decision = c.Decision.from_dict(valid_dict("decision", "accept"))
    for record in (request, request.case, request.observation, request.twin, decision, decision.policy):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(record, dataclasses.fields(record)[0].name, None)
    with pytest.raises(c.DispositionContractError):
        dataclasses.replace(decision, destination="right_tray")
    with pytest.raises(c.DispositionContractError):
        dataclasses.replace(request, source="robot")


def test_the_adjudication_has_the_nine_appendix_a_fields_and_is_null_at_decision_time():
    assert len(c.APPENDIX_A_FIELDS) == 9
    assert [f.name for f in dataclasses.fields(c.Adjudication)] == list(c.Adjudication._FIELDS)
    assert c.Decision.from_dict(valid_dict("decision", "accept")).adjudication is None


@pytest.mark.parametrize("record", [c.Case, c.PrecaptureSummary, c.Health, c.Measurement, c.Hole, c.Sufficiency,
                                    c.Bundle, c.Colour, c.Observation, c.PartPose, c.Twin, c.Request, c.Reason,
                                    c.PolicyHashes, c.Adjudication, c.Decision])
def test_each_records_wire_fields_are_its_dataclass_fields(record):
    assert sorted(record._FIELDS) == sorted(f.name for f in dataclasses.fields(record))


def test_contract_destinations_are_planner_destinations():
    assert set(c.DESTINATIONS) - {"none"} <= {d.value for d in Destination}
    assert Destination.rework_tray.value == "rework_tray"


# ---------------------------------------------------------------- health

def test_the_health_block_is_built_from_a_real_precapture_report():
    report = precapture.unreachable("no robot connection").to_dict()
    health = c.health_from_report(report)
    assert health.cell_ready is False
    assert health.precapture.failed == tuple(precapture.CHECKS)
    with pytest.raises(c.DispositionContractError, match="passed pre-capture report"):
        c.health_from_report(report, cell_ready=True)


def test_a_passed_report_may_still_be_marked_not_ready():
    report = {"format_version": precapture.FORMAT_VERSION, "passed": True,
              "checked_utc": "2026-11-27T12:00:00.000000Z", "failed": []}
    assert c.health_from_report(report).cell_ready is True
    assert c.health_from_report(report, cell_ready=False).cell_ready is False


@pytest.mark.parametrize("passed, failed", [(True, ["neck"]), (False, [])])
def test_a_self_contradicting_report_is_refused(passed, failed):
    with pytest.raises(c.DispositionContractError):
        c.PrecaptureSummary(format_version=1, passed=passed, checked_utc="2026-11-27T12:00:00Z", failed=failed)


def test_no_report_means_not_ready():
    with pytest.raises(c.DispositionContractError):
        c.Health(cell_ready=True, precapture=None)


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
    """It is vendored into the sim, so no reachy_ai, yaml, numpy or cv2."""
    tree = module_tree()
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            modules.add(node.module.split(".")[0])
    assert modules <= {"json", "math", "re", "dataclasses", "typing"}, modules
