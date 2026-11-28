"""#81: the policy files and configurations A, B and E (E under rules_v2). Offline; no robot, no sim, no model."""
import dataclasses
import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

from reachy_ai.health import precapture
from reachy_ai.reasoner import contract as c
from reachy_ai.reasoner import policy as p

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "disposition" / "valid" / "request"
CODE_SHA = "abcdef1"
POLICY = p.load_policy()

#: design -> the disposition in the #81 table
SPEC_TABLE = {"PART110": "remeasure", "PART111": "remeasure", "PART120": "remeasure", "PART121": "remeasure",
              "PART210": "reject", "PART211": "reject", "PART220": "rework", "PART221": "accept"}


def load(stem):
    return c.loads_request((FIXTURES / (stem + ".json")).read_text())


def spec_request(design="PART221"):
    return load("spec_" + design.lower())


def with_observation(request, **changes):
    return dataclasses.replace(request, observation=dataclasses.replace(request.observation, **changes))


def flagged(request, *flags):
    sufficiency = dataclasses.replace(request.observation.sufficiency, **{flag: True for flag in flags})
    return with_observation(request, sufficiency=sufficiency)


def height(value, bound=1.5):
    return c.Measurement(value=value, bound=bound)


def hole(state, diameter=None, bound=1.0):
    return c.Hole(state=state, diameter_mm=None if diameter is None else height(diameter, bound))


def expecting(request, design):
    return dataclasses.replace(request, case=dataclasses.replace(request.case, expected_design_id=design))


def with_twin(request, feasibility):
    return dataclasses.replace(request, twin=c.Twin(part_pose=None, path_feasibility=feasibility))


def rule_ids(decision):
    return [reason.rule_id for reason in decision.reasons]


def decide(request, configuration="E", policy=POLICY):
    return p.decide(request, policy, configuration, CODE_SHA)


def outcome(request, configuration="E"):
    decision = decide(request, configuration)
    return (decision.disposition or decision.refusal, decision.remeasure_route, decision.reasons[0].rule_id)


def not_ready(request):
    report = precapture.unreachable("injected outage").to_dict()
    return dataclasses.replace(request, health=c.health_from_report(report))


# ---------------------------------------------------------------- the files

def test_the_spec_holds_the_eight_designs_of_the_issue():
    spec = POLICY.spec
    assert spec.provisional is True
    assert {d.design: d.disposition for d in spec.designs} == SPEC_TABLE
    for d in spec.designs:
        shape, tall, holed = int(d.design[4]), int(d.design[5]), int(d.design[6])  # PART<S><H><F>
        assert d.shape == {1: "block", 2: "cylinder"}[shape]
        assert d.height_mm == {1: 25.4, 2: 69.85}[tall]
        assert d.footprint_mm == 49.2
        assert d.hole_diameter_mm == (25.4 if holed else None)
    assert (spec.height_class_boundary_mm, spec.hole_min_diameter_mm) == (47.6, 15.0)


def test_provisional_values_and_the_tbd_authority_stay_marked():
    authority = POLICY.spec.authority
    assert authority["provisional"] is True
    assert authority["quality_authority"] == "TBD (Willis)"
    assert authority["approved_rework_operation"] is False
    assert authority["scoring_costs"] == {
        "provisional": True, "false_accept": 10, "false_reject_or_scrap_of_good_part": 3,
        "wrong_rework_or_remeasure_routing": 2, "unnecessary_referral": 1,
        "correct_decision_or_justified_abstention": 0, "acted_when_not_ready": 10}
    text = (p.POLICY_DIR / p.SPEC_FILE).read_text()
    assert text.count("provisional: true") >= 7  # spec, five tolerances, authority (and the scoring costs)


def test_the_nominal_size_tolerances_are_the_owners_and_each_is_marked_provisional():
    spec = POLICY.spec
    assert (spec.width_tolerance_mm, spec.height_tolerance_mm, spec.hole_diameter_tolerance_mm) == (3.0, 5.0, 4.0)
    tolerances = yaml.safe_load((p.POLICY_DIR / p.SPEC_FILE).read_text())["tolerances"]
    assert set(tolerances) == set(p.TOLERANCES)
    assert all(entry["provisional"] is True for entry in tolerances.values())


#: rules_v1.yaml as merged in #89. It is kept for traceability and must not change.
RULES_V1_SHA256 = "d30778c78eb40d8f581bae1d83468354f9f24877ba3363fa5613d05048871d0d"
V1 = p.load_policy(rules_file=p.RULES_V1_FILE)


def test_v2_is_the_default_and_v1_is_unchanged():
    assert (p.RULES_FILE, POLICY.rules.version, V1.rules.version) == ("rules_v2.yaml", "rules_v2", "rules_v1")
    assert V1.rules.sha256 == RULES_V1_SHA256


def test_e_gains_four_rules_and_a_gate_and_a_b_are_as_stated():
    assert [r.id for r in POLICY.rules.sequence("E")] == ["R1", "R10", "R2", "R6", "R3", "R7", "R8", "R4", "R5"]
    assert [r.id for r in POLICY.rules.gates_for("E")] == ["R9"]
    assert [r.id for r in POLICY.rules.sequence("A")] == ["R4", "R5"]
    assert [r.id for r in POLICY.rules.sequence("B")] == ["R2", "R4", "R5"]
    assert POLICY.rules.gates_for("A") == POLICY.rules.gates_for("B") == ()
    assert POLICY.rules.max_reobservations == 2


def test_v2_leaves_the_first_five_rules_and_the_shared_settings_as_they_were():
    assert {i: POLICY.rules.rules[i] for i in V1.rules.rules} == V1.rules.rules
    assert (POLICY.rules.max_reobservations, POLICY.rules.hides) == (V1.rules.max_reobservations, V1.rules.hides)
    assert [r.id for r in V1.rules.sequence("E")] == ["R1", "R2", "R3", "R4", "R5"] and V1.rules.gates == {}


def test_the_split_is_proposed_and_not_frozen():
    split = POLICY.split
    assert split.frozen is False and split.status == "PROPOSED"
    assert {e["id"] for e in split.development} == {"dev_every_design_upright", "dev_lab_pool_pieces",
                                                    "dev_terrance_parts"}
    assert all(e["development_only"] for e in split.development if "parts" in e)
    assert set(split.development[0]["designs"]) == set(SPEC_TABLE)
    assert {e["id"] for e in split.held_out} == {
        "ho_holed_block_lying", "ho_tall_cylinder_partial_occlusion", "ho_twin_inconsistent",
        "ho_short_part_defocused", "ho_cell_motor_power_outage", "ho_cell_stale_clock"}


def test_every_decision_records_each_policy_files_hash():
    decision = decide(spec_request())
    for field, name in (("spec_sha256", p.SPEC_FILE), ("rules_sha256", p.RULES_FILE),
                        ("split_sha256", p.SPLIT_FILE)):
        assert getattr(decision.policy, field) == hashlib.sha256((p.POLICY_DIR / name).read_bytes()).hexdigest()
    assert decision.policy.prompt_sha256 is None and decision.model_id is None
    assert decision.code_sha == CODE_SHA and decision.confidence is None and decision.adjudication is None


def edited(tmp_path, name, old, new):
    directory = tmp_path / "disposition"
    shutil.copytree(p.POLICY_DIR, directory)
    text = (directory / name).read_text()
    assert old in text, old
    (directory / name).write_text(text.replace(old, new, 1))
    return directory


@pytest.mark.parametrize("name, old, new, phrase", [
    (p.SPEC_FILE, "version: spec_v1", "version: spec_v1\nsurprise: 1", "unknown key"),
    (p.SPEC_FILE, "version: spec_v1", "version: spec_v1\nversion: spec_v2", "duplicate key"),
    (p.SPEC_FILE, "design: PART111", "design: PART110", "listed twice"),
    (p.SPEC_FILE, "disposition: accept", "disposition: ship", "unknown disposition"),
    (p.SPEC_FILE, "value: 47.6", "value: -1", "positive number"),
    (p.SPEC_FILE, "value: 47.6", "value: 80.0", "is not tall at the class boundary"),
    (p.SPEC_FILE, "colour_requirement: null", "colour_requirement: green", "no colour requirement"),
    (p.SPEC_FILE, "version: spec_v1", "version: [", "spec_v1.yaml"),
    (p.RULES_FILE, "kind: classify", "kind: guess", "unknown rule kind"),
    (p.RULES_FILE, "A: [R4, R5]", "A: [R4]", "must end with the unmatched rule"),
    (p.RULES_FILE, "A: [R4, R5]", "A: [R99, R5]", "not listed"),
    (p.RULES_FILE, "A: [R4, R5]", "A: [R9, R4, R5]", "a gate belongs under gates"),
    (p.RULES_FILE, "E: [R9]", "E: [R6]", "only a gate may be listed"),
    (p.RULES_FILE, "E: [R9]", "E: [R99]", "not listed"),
    (p.RULES_FILE, "E: [R9]", "E: []", "non-empty list"),
    (p.RULES_FILE, "E: [R9]", "F: [R9]", "unknown key"),
    (p.SPEC_FILE, "  width_tolerance_mm:", "  width_tolerance:", "unknown key"),
    (p.SPEC_FILE, "value: 3.0", "value: 0", "positive number"),
    (p.RULES_FILE, "    blurred: [shape, height, hole]\n", "", "missing key"),
    (p.RULES_FILE, "feature_not_visible: [hole]", "feature_not_visible: []", "expected some of"),
    (p.SPLIT_FILE, "frozen: false", "frozen: true", "proposed split cannot be frozen"),
    (p.SPLIT_FILE, "id: ho_cell_stale_clock", "id: ho_holed_block_lying", "used twice"),
    (p.SPLIT_FILE, "[PART111, PART121]", "[PART999]", "must be spec designs"),
    (p.SPLIT_FILE, "view: defocused", "view: foggy", "unknown view"),
    (p.SPLIT_FILE, "    twin: inconsistent\n", "", "selects nothing"),
])
def test_a_bad_policy_file_is_refused(tmp_path, name, old, new, phrase):
    with pytest.raises(p.PolicyError, match=phrase):
        p.load_policy(edited(tmp_path, name, old, new))


def test_a_missing_policy_file_is_refused(tmp_path):
    with pytest.raises(p.PolicyError):
        p.load_policy(tmp_path)


# ---------------------------------------------------------------- one fixture per spec row

@pytest.mark.parametrize("design, expected", sorted(SPEC_TABLE.items()))
@pytest.mark.parametrize("configuration", ["A", "B", "E"])
def test_each_spec_row_gets_its_disposition(design, expected, configuration):
    decision = decide(spec_request(design), configuration)
    assert decision.disposition == expected
    assert decision.refusal is None
    assert design in decision.reasons[0].text
    assert decision.destination == {"accept": "left_tray", "reject": "right_tray", "rework": "rework_tray",
                                    "remeasure": "none"}[expected]


@pytest.mark.parametrize("design", ["PART110", "PART111", "PART120", "PART121"])
def test_a_spec_remeasure_reobserves_twice_then_refers(design):
    request = spec_request(design)
    for configuration in ("B", "E"):
        routes = [decide(dataclasses.replace(request, reobservation_count=n), configuration).remeasure_route
                  for n in range(4)]
        assert routes == ["reobserve", "reobserve", "refer", "refer"]
    assert decide(request, "A").remeasure_route == "refer"  # A cannot re-observe


def test_any_block_is_remeasure_even_with_nothing_else_known():
    request = with_observation(spec_request("PART110"), height_mm=None, hole=hole("unknown"))
    assert outcome(request, "B") == ("remeasure", "reobserve", "R4")
    assert outcome(request, "E") == ("remeasure", "reobserve", "R10")


def test_rework_says_it_is_only_recommended():
    text = decide(spec_request("PART220")).reasons[0].text
    assert "Rework recommended" in text and "no approved rework operation" in text and "TBD (Willis)" in text


# ---------------------------------------------------------------- Stage 0

def test_a_failed_health_report_is_refused_at_stage_0():
    decision = decide(not_ready(spec_request()))
    assert (decision.disposition, decision.refusal, decision.destination) == (None, "cell_not_ready", "none")
    assert decision.reasons[0].rule_id == "R1"
    assert "sdk_reachable" in decision.reasons[0].text


@pytest.mark.parametrize("report, phrase", [
    (None, "no pre-capture report"),
    (c.PrecaptureSummary(1, True, "2026-11-27T12:00:00Z", ()), "marked not ready"),
])
def test_stage_0_says_why_the_cell_is_not_ready(report, phrase):
    request = dataclasses.replace(spec_request(), health=c.Health(cell_ready=False, precapture=report))
    assert phrase in decide(request).reasons[0].text


def test_a_and_b_are_ablations_without_stage_0():
    assert decide(not_ready(spec_request()), "A").disposition == "accept"
    assert decide(not_ready(spec_request()), "B").disposition == "accept"


# ---------------------------------------------------------------- rule order: the first rule that fires wins

def test_stage_0_wins_over_every_later_rule():
    request = not_ready(flagged(with_observation(spec_request(), height_mm=height(47.0, 5.0)), "blurred"))
    assert outcome(request) == ("cell_not_ready", None, "R1")


def test_sufficiency_wins_over_the_bound_check_and_the_table():
    request = flagged(with_observation(spec_request(), height_mm=height(47.0, 5.0)), "blurred")
    assert outcome(request) == ("remeasure", "reobserve", "R2")


def test_the_bound_check_wins_over_the_table():
    request = with_observation(spec_request(), height_mm=height(49.0, 5.0))
    assert outcome(request, "E") == ("remeasure", "reobserve", "R3")
    assert outcome(request, "B") == ("accept", None, "R4")  # B has no bound check
    assert outcome(request, "A") == ("accept", None, "R4")


def test_reordering_the_rules_file_changes_which_rule_wins(tmp_path):
    reordered = p.load_policy(edited(tmp_path, p.RULES_FILE, "E: [R1, R10, R2, R6, R3, R7, R8, R4, R5]",
                                     "E: [R1, R10, R3, R2, R6, R7, R8, R4, R5]"))
    request = flagged(with_observation(spec_request(), height_mm=height(47.0, 5.0)), "blurred")
    decision = decide(request, policy=reordered)
    assert decision.reasons[0].rule_id == "R3"
    assert decision.policy.rules_sha256 != POLICY.rules.sha256


# ---------------------------------------------------------------- rule 2: sufficiency

@pytest.mark.parametrize("flag", ["blurred", "occluded", "stale", "lens_unverified", "depth_insufficient"])
@pytest.mark.parametrize("design", ["PART110", "PART210", "PART221"])
def test_each_general_flag_hides_a_deciding_feature(flag, design):
    request = flagged(spec_request(design), flag)
    for configuration in ("B", "E"):
        assert outcome(request, configuration) == ("remeasure", "reobserve", "R2")
    assert outcome(request, "A")[2] == "R4"  # A ignores sufficiency


@pytest.mark.parametrize("design, hidden", [("PART220", True), ("PART221", True), ("PART210", False),
                                            ("PART211", False), ("PART110", False), ("PART121", False)])
def test_a_hidden_top_face_matters_only_where_the_hole_decides(design, hidden):
    rule = outcome(flagged(spec_request(design), "feature_not_visible"))[2]
    assert rule == ("R2" if hidden else "R4")


def test_a_hidden_top_face_matters_when_the_family_is_not_known():
    for changes, rule_in_e in (({"kind": "unknown"}, "R2"), ({"kind": "cylinder", "height_mm": None}, "R10")):
        request = flagged(with_observation(spec_request("PART210"), **changes), "feature_not_visible")
        assert outcome(request, "B")[2] == "R2"
        assert outcome(request, "E")[2] == rule_in_e


def test_after_two_reobservations_a_hidden_feature_is_referred():
    request = flagged(spec_request(), "occluded")
    routes = [decide(dataclasses.replace(request, reobservation_count=n)).remeasure_route for n in range(4)]
    assert routes == ["reobserve", "reobserve", "refer", "refer"]


# ---------------------------------------------------------------- rule 3: bounds

@pytest.mark.parametrize("value, bound, straddles", [
    (47.0, 1.0, True), (48.0, 0.5, True), (47.6, 0.0, True), (46.6, 1.0, True), (48.6, 1.0, True),
    (46.0, 1.0, False), (49.0, 1.0, False), (25.4, 1.5, False), (69.85, 1.5, False)])
def test_a_height_bound_that_reaches_the_boundary_is_remeasured(value, bound, straddles):
    request = with_observation(spec_request(), height_mm=height(value, bound))
    assert (outcome(request)[2] == "R3") is straddles


def test_after_two_reobservations_a_straddling_height_is_referred():
    request = dataclasses.replace(with_observation(spec_request(), height_mm=height(47.6)), reobservation_count=2)
    assert outcome(request) == ("remeasure", "refer", "R3")


# ---------------------------------------------------------------- rules 4 and 5

@pytest.mark.parametrize("opening, expected", [
    (hole("present", 25.4), "accept"), (hole("present", 15.0), "accept"), (hole("absent"), "rework"),
    (hole("present", 14.9), "spec_unknown"), (hole("present"), "spec_unknown"), (hole("unknown"), "spec_unknown")])
def test_a_hole_is_present_only_with_a_measured_opening_of_15_mm_or_more(opening, expected):
    for configuration in ("A", "B"):
        assert outcome(with_observation(spec_request(), hole=opening), configuration)[0] == expected


@pytest.mark.parametrize("opening, expected", [
    (hole("present", 25.4), "accept"), (hole("present", 15.0), "spec_unknown"), (hole("absent"), "rework"),
    (hole("present", 14.9), "spec_unknown"), (hole("present"), "remeasure"), (hole("unknown"), "spec_unknown")])
def test_in_e_a_measured_hole_must_also_be_near_nominal(opening, expected):
    """No expected design here, so only the observation decides."""
    assert outcome(with_observation(expecting(spec_request(), None), hole=opening))[0] == expected


def test_the_presence_rule_and_the_size_rule_are_both_kept():
    """E5: 15 mm says whether an opening is seen; the size rule then judges the diameter."""
    opening = hole("present", 16.0, 0.0)
    request = with_observation(expecting(spec_request(), None), hole=opening)
    assert p._effective_hole(POLICY.spec, opening) == "present" and outcome(request, "A")[0] == "accept"
    assert outcome(request) == ("spec_unknown", None, "R7")  # seen, and the wrong size
    unseen = with_observation(request, hole=hole("present", 14.0, 0.0))
    assert p._effective_hole(POLICY.spec, unseen.observation.hole) == "unknown"
    assert outcome(unseen, "A")[0] == "spec_unknown"


def test_a_short_cylinder_is_rejected_whatever_the_hole():
    base = expecting(spec_request("PART210"), None)
    for opening in (hole("present", 25.4), hole("present", 5.0), hole("absent"), hole("unknown")):
        request = with_observation(base, hole=opening)
        assert outcome(request, "A")[0] == outcome(request, "B")[0] == "reject"
        wrong_size = opening.diameter_mm is not None and opening.diameter_mm.value == 5.0
        assert outcome(request, "E")[0] == ("spec_unknown" if wrong_size else "reject")


@pytest.mark.parametrize("changes, configurations", [({"kind": "unknown"}, "ABE"), ({"height_mm": None}, "AB")])
def test_anything_unmatched_is_spec_unknown(changes, configurations):
    for configuration in configurations:  # in E an unmeasured height is R10's: see the missing measurements
        decision = decide(with_observation(spec_request(), **changes), configuration)
        assert (decision.refusal, decision.reasons[0].rule_id) == ("spec_unknown", "R5")


def test_nothing_measured_is_a_refusal_in_every_configuration():
    request = load("nothing_measured_no_health_report")
    assert [outcome(request, name)[0] for name in "ABE"] == ["spec_unknown", "spec_unknown", "cell_not_ready"]


# ---------------------------------------------------------------- rule 10: missing measurements (E1)

def three_looks(request, configuration="E"):
    """The decisions with 0, 1 and 2 re-observations already made."""
    decisions = [decide(dataclasses.replace(request, reobservation_count=n), configuration) for n in range(3)]
    return [(d.disposition or d.refusal, d.remeasure_route, d.reasons[0].rule_id) for d in decisions]


def remeasured_by(rule_id):
    return [("remeasure", "reobserve", rule_id), ("remeasure", "reobserve", rule_id), ("remeasure", "refer", rule_id)]


def test_width_missing_on_an_otherwise_accept_tall_cylinder_is_remeasured_then_referred():
    request = with_observation(spec_request("PART221"), width_mm=None)
    assert outcome(request, "A")[0] == outcome(request, "B")[0] == "accept"
    assert three_looks(request) == remeasured_by("R10")
    assert decide(request).reasons[0].text.startswith("Missing measurement(s): width.")


def test_a_short_cylinder_with_width_missing_is_remeasured_not_rejected():
    request = with_observation(spec_request("PART210"), width_mm=None)
    assert outcome(request, "A")[0] == "reject"
    assert three_looks(request) == remeasured_by("R10")


MISSING = {
    "width": {"width_mm": None},
    "height": {"height_mm": None},
    "hole diameter": {"hole": c.Hole(state="present", diameter_mm=None)},
    "orientation": {"orientation": "unknown"},
}


@pytest.mark.parametrize("name", sorted(MISSING))
@pytest.mark.parametrize("design", sorted(SPEC_TABLE))
def test_each_missing_measurement_is_remeasured_for_every_design_and_named(name, design):
    request = with_observation(spec_request(design), **MISSING[name])
    assert three_looks(request) == remeasured_by("R10")
    assert f"Missing measurement(s): {name}." in decide(request).reasons[0].text


def test_every_missing_measurement_is_named_at_once():
    changes = {key: value for entry in MISSING.values() for key, value in entry.items()}
    text = decide(with_observation(spec_request(), **changes)).reasons[0].text
    assert "Missing measurement(s): width, height, hole diameter, orientation." in text


def test_a_hole_that_was_not_seen_needs_no_diameter():
    for state in ("absent", "unknown"):
        assert outcome(with_observation(spec_request("PART210"), hole=hole(state)))[2] == "R4"


def test_missing_is_recorded_even_when_another_rule_would_also_fire():
    request = flagged(with_observation(spec_request(), width_mm=None, orientation="lying",
                                       height_mm=height(120.0)), "blurred")
    assert outcome(request) == ("remeasure", "reobserve", "R10")
    assert outcome(not_ready(request)) == ("cell_not_ready", None, "R1")  # Stage 0 still comes first


def lying_short_cylinder():
    """A short cylinder on its side: its 49.2 mm width reads as a tall height, and no top-face hole shows."""
    return with_observation(expecting(spec_request("PART210"), None), orientation="lying", height_mm=height(49.2),
                            width_mm=height(49.2), hole=hole("absent"))


def test_a_lying_short_cylinder_is_remeasured_not_reworked():
    request = lying_short_cylinder()
    assert outcome(request, "A")[0] == outcome(request, "B")[0] == "rework"  # the baselines are fooled
    assert outcome(request) == ("remeasure", "reobserve", "R6")
    assert decide(request).destination == "none"


@pytest.mark.parametrize("orientation", ["lying", "rotated", "unknown"])
@pytest.mark.parametrize("design", sorted(SPEC_TABLE))
def test_a_part_not_upright_reobserves_twice_then_refers(orientation, design):
    request = with_observation(spec_request(design), orientation=orientation)
    # an unknown orientation is a missing measurement, so R10 reports it
    assert three_looks(request) == remeasured_by("R10" if orientation == "unknown" else "R6")


def test_orientation_is_checked_before_the_height_and_the_size():
    request = with_observation(spec_request(), orientation="lying", height_mm=height(47.6, 1.0))
    assert outcome(request)[2] == "R6"
    assert outcome(with_observation(request, width_mm=height(200.0)))[2] == "R6"
    assert outcome(flagged(request, "blurred"))[2] == "R2"  # evidence quality comes first


# ---------------------------------------------------------------- rule 7: nominal size (E)

def test_an_oversize_cylinder_with_a_hole_is_spec_unknown():
    request = with_observation(spec_request(), height_mm=height(120.0), width_mm=height(80.0))
    assert outcome(request, "A")[0] == outcome(request, "B")[0] == "accept"
    decision = decide(request)
    assert (decision.refusal, decision.destination, decision.reasons[0].rule_id) == ("spec_unknown", "none", "R7")
    assert "width 80 +- 1.5 mm" in decision.reasons[0].text
    assert "height 120 +- 1.5 mm" in decision.reasons[0].text
    assert "Refer." in decision.reasons[0].text


def sized(design, name, value, bound):
    changed = hole("present", value, bound) if name == "hole" else height(value, bound)
    return with_observation(spec_request(design), **{name: changed})


@pytest.mark.parametrize("design, name, value, inside", [
    ("PART221", "width_mm", 52.2, True), ("PART221", "width_mm", 52.3, False),
    ("PART221", "width_mm", 46.2, True), ("PART221", "width_mm", 46.1, False),
    ("PART221", "height_mm", 74.85, True), ("PART221", "height_mm", 74.95, False),
    ("PART221", "height_mm", 64.85, True), ("PART221", "height_mm", 64.75, False),
    ("PART210", "height_mm", 30.4, True), ("PART210", "height_mm", 30.5, False),
    ("PART210", "height_mm", 20.4, True), ("PART210", "height_mm", 20.3, False),
    ("PART221", "hole", 29.4, True), ("PART221", "hole", 29.5, False),
    ("PART221", "hole", 21.4, True), ("PART221", "hole", 21.3, False),
])
def test_the_nominal_limits_are_inside_and_anything_past_them_is_outside(design, name, value, inside):
    decision = decide(sized(design, name, value, 0.0))
    assert (decision.reasons[0].rule_id != "R7") is inside
    assert (decision.refusal == "spec_unknown") is not inside


@pytest.mark.parametrize("value, bound, expected", [
    # width: 49.2 +- 3, so the limits are 46.2 and 52.2
    (49.2, 3.0, "inside"), (51.7, 0.5, "inside"), (46.7, 0.5, "inside"),
    (51.9, 0.5, "straddles"),  # the owner's example
    (52.0, 0.5, "straddles"),  # 51.5 to 52.5 (owner, 2026-11-27: follow the rule)
    (52.7, 0.5, "straddles"), (46.4, 0.5, "straddles"), (45.7, 0.5, "straddles"),
    (49.2, 20.0, "straddles"),  # a bound wider than the whole band
    (53.0, 0.5, "outside"), (45.0, 0.5, "outside"), (80.0, 1.5, "outside"),
])
def test_the_size_rule_judges_the_whole_interval(value, bound, expected):
    request = sized("PART221", "width_mm", value, bound)
    if expected == "inside":
        assert outcome(request) == ("accept", None, "R4")
    elif expected == "straddles":
        assert three_looks(request) == remeasured_by("R7")
        assert f"width {value:g} +- {bound:g} mm" in decide(request).reasons[0].text
    else:
        assert outcome(request) == ("spec_unknown", None, "R7")
        assert "Refer." in decide(request).reasons[0].text
    assert outcome(request, "A")[0] == outcome(request, "B")[0] == "accept"  # the baselines read no width


@pytest.mark.parametrize("design, name, value, bound", [
    ("PART221", "height_mm", 74.5, 1.0), ("PART221", "height_mm", 65.0, 1.0), ("PART210", "height_mm", 30.0, 1.0),
    ("PART221", "hole", 29.0, 1.0), ("PART221", "hole", 21.0, 1.0)])
def test_a_straddling_height_or_hole_diameter_is_remeasured(design, name, value, bound):
    assert three_looks(sized(design, name, value, bound)) == remeasured_by("R7")


def test_one_size_wholly_outside_decides_even_if_another_straddles():
    request = with_observation(spec_request(), width_mm=height(51.9, 0.5), height_mm=height(120.0))
    assert outcome(request) == ("spec_unknown", None, "R7")


def test_an_oversize_block_is_spec_unknown_not_remeasure():
    assert outcome(with_observation(spec_request("PART121"), width_mm=height(70.0)))[:1] == ("spec_unknown",)


def test_a_straddling_height_is_reobserved_before_it_is_called_outside():
    assert outcome(with_observation(spec_request(), height_mm=height(47.6, 1.0))) == ("remeasure", "reobserve", "R3")
    assert outcome(with_observation(spec_request(), height_mm=height(40.0, 1.0)))[2] == "R7"


# ---------------------------------------------------------------- rule 8: the expected design (E)

def test_a_design_mismatch_reobserves_twice_then_refers():
    request = expecting(spec_request("PART221"), "PART220")  # a hole was seen; the expected part has none
    decisions = [decide(dataclasses.replace(request, reobservation_count=n)) for n in range(4)]
    assert [(d.disposition, d.remeasure_route, d.reasons[0].rule_id) for d in decisions] == [
        ("remeasure", "reobserve", "R8"), ("remeasure", "reobserve", "R8"),
        ("remeasure", "refer", "R8"), ("remeasure", "refer", "R8")]
    assert "expected design PART220: hole present, expected absent" in decisions[0].reasons[0].text
    assert outcome(request, "A")[0] == outcome(request, "B")[0] == "accept"


@pytest.mark.parametrize("design, expected, phrase", [
    ("PART221", "PART121", "kind cylinder, expected block"),
    ("PART221", "PART211", "height class tall, expected short"),
    ("PART220", "PART221", "hole absent, expected present"),
])
def test_each_known_feature_can_disagree_with_the_expected_design(design, expected, phrase):
    decision = decide(expecting(spec_request(design), expected))
    assert (decision.remeasure_route, decision.reasons[0].rule_id) == ("reobserve", "R8")
    assert phrase in decision.reasons[0].text


def test_an_unknown_feature_is_not_an_inconsistency():
    request = with_observation(expecting(spec_request("PART210"), "PART211"), hole=hole("unknown"))
    assert outcome(request) == ("reject", None, "R4")
    unknown = with_observation(spec_request(), kind="unknown")
    assert outcome(unknown) == ("spec_unknown", None, "R5")


def test_with_no_expected_design_nothing_is_compared():
    assert outcome(expecting(spec_request(), None)) == ("accept", None, "R4")


def test_an_expected_design_the_spec_does_not_list_is_spec_unknown():
    decision = decide(expecting(spec_request(), "PART999"))
    assert (decision.refusal, decision.reasons[0].rule_id) == ("spec_unknown", "R8")
    assert "PART999 is not in spec_v1" in decision.reasons[0].text


# ---------------------------------------------------------------- gate 9: twin feasibility (E)

@pytest.mark.parametrize("design, disposition, bin_name", [
    ("PART221", "accept", "left_tray"), ("PART210", "reject", "right_tray"), ("PART220", "rework", "rework_tray")])
def test_an_infeasible_destination_is_beyond_capability(design, disposition, bin_name):
    request = with_twin(spec_request(design), {bin_name: False})
    decision = decide(request)
    assert (decision.disposition, decision.refusal, decision.remeasure_route, decision.destination) == (
        None, "beyond_capability", None, "none")
    assert rule_ids(decision) == ["R9", "R4"]  # what decided, then the disposition it could not carry out
    assert bin_name in decision.reasons[0].text and f"The {disposition} that R4 reached" in decision.reasons[0].text
    assert design in decision.reasons[1].text
    assert decide(request, "A").disposition == decide(request, "B").disposition == disposition
    assert c.loads_decision(c.dumps(decision)) == decision


def test_a_feasible_destination_passes_and_only_that_destination_counts():
    request = with_twin(spec_request(), {"left_tray": True, "right_tray": False, "rework_tray": False})
    decision = decide(request)
    assert (decision.disposition, decision.destination, rule_ids(decision)) == ("accept", "left_tray", ["R4"])


def test_a_null_twin_skips_the_feasibility_check_and_records_it():
    request = load("robot_perceived_with_colour")
    assert request.source == "robot" and request.twin is None
    decision = decide(request)
    assert (decision.disposition, decision.destination, rule_ids(decision)) == ("accept", "left_tray", ["R4", "R9"])
    assert decision.reasons[1].text == "The twin block is null: the feasibility check for left_tray was skipped."
    assert rule_ids(decide(request, "A")) == rule_ids(decide(request, "B")) == ["R4"]


@pytest.mark.parametrize("feasibility", [None, {"right_tray": False}, {"right_tray": True}, {}])
def test_a_sim_twin_without_an_entry_for_the_bin_is_insufficient_evidence(feasibility):
    request = with_twin(spec_request(), feasibility)
    assert request.source == "sim"
    decision = decide(request)
    assert (decision.disposition, decision.refusal, decision.destination) == (None, "insufficient_evidence", "none")
    assert rule_ids(decision) == ["R9", "R4"]
    assert decision.reasons[0].text == "twin feasibility missing for left_tray"
    assert decide(request, "A").disposition == decide(request, "B").disposition == "accept"
    assert c.loads_decision(c.dumps(decision)) == decision


@pytest.mark.parametrize("request_", [
    spec_request("PART110"),                               # remeasure
    with_observation(spec_request(), kind="unknown"),      # a refusal
    not_ready(spec_request()),                             # Stage 0
])
def test_with_no_destination_there_is_no_feasibility_check(request_):
    everything_infeasible = with_twin(request_, {"left_tray": False, "right_tray": False, "rework_tray": False})
    assert decide(everything_infeasible) == decide(request_)
    assert "R9" not in rule_ids(decide(request_))


def test_the_old_rules_still_decide_as_they_did():
    assert decide(lying_short_cylinder(), policy=V1).disposition == "rework"
    assert rule_ids(decide(load("robot_perceived_with_colour"), policy=V1)) == ["R4"]


@pytest.mark.parametrize("configuration", ["A", "B"])
def test_a_and_b_decide_under_v2_exactly_as_under_v1(configuration):
    requests = [c.loads_request(path.read_text()) for path in sorted(FIXTURES.glob("*.json"))]
    requests += [lying_short_cylinder(), with_twin(spec_request(), {"left_tray": False}),
                 expecting(spec_request(), "PART220"), not_ready(spec_request()),
                 with_observation(spec_request(), width_mm=height(200.0), height_mm=height(47.0, 5.0))]
    for request in requests:
        new, old = decide(request, configuration), decide(request, configuration, policy=V1)
        assert new.policy.rules_sha256 != old.policy.rules_sha256
        assert dataclasses.replace(new, policy=old.policy) == old


# ---------------------------------------------------------------- colour, purity, scope

@pytest.mark.parametrize("design", sorted(SPEC_TABLE))
def test_colour_never_changes_a_disposition(design):
    plain = spec_request(design)
    for name in c.COLOUR_NAMES:
        coloured = with_observation(plain, colour=c.Colour(name=name, lab_median=(50.0, 0.0, 0.0),
                                                           confidence=1.0, reliable=True))
        assert decide(coloured) == decide(plain)


def test_a_and_b_still_read_no_orientation_width_expected_design_or_twin():
    base = spec_request()
    for configuration in ("A", "B"):
        plain = decide(base, configuration)
        for changed in (with_observation(base, orientation="lying"), with_observation(base, width_mm=height(200.0)),
                        expecting(base, "PART110"), with_twin(base, {"left_tray": False})):
            assert decide(changed, configuration) == plain


def test_no_rule_reads_the_part_pose():
    base = with_twin(spec_request(), {"left_tray": True})
    pose = c.PartPose(position_m=(0.4, -0.2, -0.3), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    posed = dataclasses.replace(base, twin=dataclasses.replace(base.twin, part_pose=pose))
    assert decide(posed) == decide(base)


def test_decide_is_pure(monkeypatch):
    request = spec_request()

    def refuse(*args, **kwargs):
        raise AssertionError("decide() touched a file")

    monkeypatch.setattr("builtins.open", refuse)
    monkeypatch.setattr(Path, "read_bytes", refuse)
    first, second = decide(request), decide(request)
    monkeypatch.undo()
    assert first == second


def test_a_decision_survives_the_wire():
    for design in SPEC_TABLE:
        decision = decide(spec_request(design))
        assert c.loads_decision(c.dumps(decision)) == decision


@pytest.mark.parametrize("configuration", ["C", "D", "F", "Z"])
def test_the_ai_configurations_are_not_run_here(configuration):
    with pytest.raises(p.PolicyError, match="not a deterministic configuration"):
        decide(spec_request(), configuration)


def test_the_reasoner_imports_no_motion_sdk_or_network_module():
    banned = ("reachy_sdk", "reachy_ai.motion", "reachy_client", "anthropic", "grpc", "socket", "urllib", "requests")
    for path in (ROOT / "src" / "reachy_ai" / "reasoner").glob("*.py"):
        imports = [line for line in path.read_text().splitlines() if line.lstrip().startswith(("import ", "from "))]
        if path.name == "ai.py":  # the AI reasoner (#83): the Anthropic SDK, imported only inside the provider call
            assert [line for line in imports if "anthropic" in line] == ["            import anthropic"]
            imports.remove("            import anthropic")
        assert not [line for line in imports for name in banned if name in line], path.name
