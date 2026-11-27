"""#81: the policy files and configurations A, B and E. Offline; no robot, no sim, no model."""
import dataclasses
import hashlib
import shutil
from pathlib import Path

import pytest

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


def hole(state, diameter=None):
    return c.Hole(state=state, diameter_mm=None if diameter is None else height(diameter, 1.0))


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
        "correct_decision_or_justified_abstention": 0}
    text = (p.POLICY_DIR / p.SPEC_FILE).read_text()
    assert text.count("provisional: true") >= 4  # spec, two tolerances, authority (and the scoring costs)


def test_the_rules_are_the_five_of_the_issue_and_a_b_e_are_as_stated():
    assert [r.id for r in POLICY.rules.sequence("E")] == ["R1", "R2", "R3", "R4", "R5"]
    assert [r.id for r in POLICY.rules.sequence("A")] == ["R4", "R5"]
    assert [r.id for r in POLICY.rules.sequence("B")] == ["R2", "R4", "R5"]
    assert POLICY.rules.max_reobservations == 2


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
    (p.RULES_FILE, "A: [R4, R5]", "A: [R9, R5]", "not listed"),
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
    assert outcome(request) == ("remeasure", "reobserve", "R4")


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
    reordered = p.load_policy(edited(tmp_path, p.RULES_FILE, "E: [R1, R2, R3, R4, R5]", "E: [R1, R3, R2, R4, R5]"))
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
    for changes in ({"kind": "unknown"}, {"kind": "cylinder", "height_mm": None}):
        request = flagged(with_observation(spec_request("PART210"), **changes), "feature_not_visible")
        assert outcome(request)[2] == "R2"


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
    for configuration in ("A", "B", "E"):
        assert outcome(with_observation(spec_request(), hole=opening), configuration)[0] == expected


def test_a_short_cylinder_is_rejected_whatever_the_hole():
    for opening in (hole("present", 25.4), hole("present", 5.0), hole("absent"), hole("unknown")):
        assert outcome(with_observation(spec_request("PART210"), hole=opening))[0] == "reject"


@pytest.mark.parametrize("changes", [{"kind": "unknown"}, {"height_mm": None}])
def test_anything_unmatched_is_spec_unknown(changes):
    for configuration in ("A", "B", "E"):
        decision = decide(with_observation(spec_request(), **changes), configuration)
        assert (decision.refusal, decision.reasons[0].rule_id) == ("spec_unknown", "R5")


def test_nothing_measured_is_a_refusal_in_every_configuration():
    request = load("nothing_measured_no_health_report")
    assert [outcome(request, name)[0] for name in "ABE"] == ["spec_unknown", "spec_unknown", "cell_not_ready"]


# ---------------------------------------------------------------- colour, purity, scope

@pytest.mark.parametrize("design", sorted(SPEC_TABLE))
def test_colour_never_changes_a_disposition(design):
    plain = spec_request(design)
    for name in c.COLOUR_NAMES:
        coloured = with_observation(plain, colour=c.Colour(name=name, lab_median=(50.0, 0.0, 0.0),
                                                           confidence=1.0, reliable=True))
        assert decide(coloured) == decide(plain)


def test_inputs_no_rule_reads_are_pinned_here():
    """#81 gives no rule for orientation, width, the expected design or the twin. This records that gap: when a
    rule is added, in a new rules version, this test changes with it."""
    base = spec_request()
    assert decide(with_observation(base, orientation="lying")) == decide(base)
    assert decide(with_observation(base, width_mm=height(200.0))) == decide(base)
    assert decide(dataclasses.replace(base, case=dataclasses.replace(base.case, expected_design_id="PART110"))
                  ) == decide(base)
    twin = c.Twin(part_pose=None, path_feasibility={"left_tray": False})
    assert decide(dataclasses.replace(base, twin=twin)) == decide(base)


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
        assert not [line for line in imports for name in banned if name in line], path.name
