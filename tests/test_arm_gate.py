"""A1 (#66): `safety.arm_gate_check`, design section 5 items 1-9. A refusal test per item, the dry run, and the
monotonicity property: no record, flag or argument combination turns a refused leg into a pass. Offline, no SDK."""

import ast
import copy
import itertools
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from reachy_ai import config
from reachy_ai.motion import arm_records, arm_routes
from reachy_ai.motion.safety import MOTION_DISABLED, GateResult, arm_gate_check
from tests.test_arm_routes import record as make_record

ROUTE = "B2_GRIP_TUNE"
REST_TIP = (0.141, -0.396, -0.397)


@pytest.fixture(autouse=True)
def covered_purpose(monkeypatch):
    """Part B's owner decision is not made: the packaged config covers no purpose. The tests assume it is."""
    monkeypatch.setattr(config, "ARM_ALLOWED_PURPOSES", ("gripper_calibration",))


def taught(doc):
    """The route file with REST and REST_OUT taught (OD-A5), so the pose rule is met."""
    out = copy.deepcopy(doc)
    for name in ("REST", "REST_OUT"):
        out["poses"][name]["provenance"] = copy.deepcopy(out["poses"]["PICK"]["provenance"])
    return out


def scenario(*defects, leg=0):
    """(doc, record, kwargs) for one leg: everything satisfied, then each named defect applied."""
    s = SimpleNamespace(doc=taught(arm_routes.load_routes_doc()), record=None, route=ROUTE, leg=leg)
    s.legs = arm_routes.expand_legs(s.doc, ROUTE)
    s.build = lambda: make_record(s.doc, verdict="pass")
    s.record = s.build()
    s.base = lambda: dict(
        present_deg=dict(s.legs[s.leg].q_from), temperatures_c={j: 40.0 for j in arm_routes.ARM_JOINTS},
        duration_s=arm_routes.leg_duration_s(s.legs[s.leg].q_from, s.legs[s.leg].q_to, s.legs[s.leg].min_duration_s),
        max_goal_step_deg=0.2, operator_approved=True, motion_enabled=True, motion_lock_free=True,
        stop_unresolved=False, rig_check_done=True,
        empty_confirmed=sorted(c for c, v in s.record["occupancy"].items() if v == "must_be_empty"),
        board_reads={}, fk_tip_m=REST_TIP, rest_tip_m=REST_TIP, hand_in_view=False)
    s.kw = s.base()
    for defect in sorted(defects, key=lambda d: d not in DOC_DEFECTS):   # a changed file first: its record follows it
        DEFECTS[defect](s)
    return s


def run(s):
    return arm_gate_check(s.doc, s.route, s.leg, s.record, **s.kw)


def mutate_doc(s, change):
    change(s.doc)
    s.legs = arm_routes.expand_legs(s.doc, s.route)
    s.record = s.build()


# One defect per item of section 5 (the text the failure must contain, in the second table).
def d_route_missing(s): s.route = "NO_SUCH_ROUTE"
def d_leg_index(s): s.leg = 99
def d_pose_not_taught(s): mutate_doc(s, lambda d: d["poses"]["REST"].update(provenance=copy.deepcopy(
    arm_routes.load_routes_doc()["poses"]["REST"]["provenance"])))
def d_no_record(s): s.record = None
def d_verdict_fail(s): s.record["verdict"] = "fail"
def d_hash(s): s.record["route_hash"] = "0" * 64
def d_mount(s): s.record["rig"]["mount_id"] = "september-2026"
def d_layout(s): s.record["rig"]["rig_layout_id"] = "other-lab"
def d_record_row(s): s.record["legs"][s.leg]["to"] = "ELSEWHERE"
def d_limit(s): mutate_doc(s, lambda d: d["poses"]["REST_OUT"]["joints_deg"].update(r_shoulder_roll=25.0))
def d_slow(s): s.kw["duration_s"] = 1.0
def d_no_duration(s): s.kw["duration_s"] = None
def d_step(s): s.kw["max_goal_step_deg"] = 0.6
def d_step_unknown(s): s.kw["max_goal_step_deg"] = None
def d_start(s): s.kw["present_deg"]["r_elbow_pitch"] += 5.0
def d_present_missing(s): s.kw["present_deg"].pop("r_wrist_roll")
def d_hot(s): s.kw["temperatures_c"]["r_arm_yaw"] = 61.0
def d_temperature_unknown(s): s.kw["temperatures_c"].pop("r_arm_yaw")
def d_approval(s): s.kw["operator_approved"] = False
def d_disabled(s): s.kw["motion_enabled"] = False
def d_lock(s): s.kw["motion_lock_free"] = False
def d_stop(s): s.kw["stop_unresolved"] = True
def d_rig_check(s): s.kw["rig_check_done"] = False
def d_not_empty(s): s.kw["empty_confirmed"] = ["r1c2", "r2c3", "r3c2"]
def d_read_occupied(s): s.kw["board_reads"] = {"r1c3": "occupied"}
def d_read_unknown(s): s.kw["board_reads"] = {"r2c3": "unknown"}
def d_fk_missing(s): s.kw["fk_tip_m"] = None
def d_fk_far(s): s.kw["fk_tip_m"] = (0.141, -0.396 + 0.05, -0.397)
def d_hand(s): s.kw["hand_in_view"] = True


DOC_DEFECTS = {"pose_not_taught", "limit"}
DEFECTS = {name[2:]: fn for name, fn in list(globals().items()) if name.startswith("d_")}

EXPECTED = {
    "route_missing": "is not in the route file", "leg_index": "is not a leg of route",
    "pose_not_taught": "sim pose(s)",
    "no_record": "No clearance record", "verdict_fail": "not 'pass'", "hash": "does not match the route's current hash",
    "mount": "mount 'september-2026'", "layout": "rig 'other-lab'", "record_row": "no row for this leg",
    "limit": "outside the arm limits", "slow": "exceeds", "no_duration": "not a positive number",
    "step": "over 0.5 deg", "step_unknown": "step size cannot be checked", "start": "from REST's",
    "present_missing": "Present r_wrist_roll unknown", "hot": "r_arm_yaw is 61 C",
    "temperature_unknown": "unknown — motion refused",
    "approval": "Operator approval not given", "disabled": MOTION_DISABLED, "lock": "motion lock",
    "stop": "Stop or abort is unresolved", "rig_check": "No rig check", "not_empty": "r1c3 must be empty",
    "read_occupied": "r1c3 is occupied", "read_unknown": "r2c3 is unknown", "fk_missing": "FK check not done",
    "fk_far": "from the record's REST tip", "hand": "A hand is in view",
}
assert set(EXPECTED) == set(DEFECTS)


def test_the_satisfied_scenario_passes_and_has_no_failures():
    result = run(scenario())
    assert isinstance(result, GateResult) and result.passed and result.failures == []


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_one_refusal_per_item(defect):
    result = run(scenario(defect))
    assert not result.passed
    assert any(EXPECTED[defect] in f for f in result.failures), result.failures


def test_the_purpose_item_refuses_when_no_owner_decision_covers_it(monkeypatch):
    monkeypatch.setattr(config, "ARM_ALLOWED_PURPOSES", ())
    result = run(scenario())
    assert not result.passed and any("not covered by an owner decision" in f for f in result.failures)


def test_every_item_is_covered():
    """Items 1-9 of section 5 each have a refusal test above."""
    items = {1: ("route_missing", "pose_not_taught"), 2: ("no_record", "verdict_fail", "hash", "mount"),
             3: ("limit", "slow", "step", "start", "hot"), 4: ("approval",), 5: ("disabled",),
             6: ("lock", "stop"), 7: ("rig_check", "not_empty", "read_occupied"), 8: ("fk_missing", "fk_far"),
             9: ("hand",)}
    assert all(d in DEFECTS for names in items.values() for d in names) and sorted(items) == list(range(1, 10))


def test_motion_disabled_is_a_dry_run_not_a_refusal():
    result = run(scenario("disabled"))
    assert not result.passed and result.failures == [MOTION_DISABLED]


def test_motion_enabled_defaults_to_the_config_flag(monkeypatch):
    s = scenario()
    s.kw["motion_enabled"] = None
    monkeypatch.setattr(config, "REACHY_ENABLE_MOTION", False)
    assert run(s).failures == [MOTION_DISABLED]
    monkeypatch.setattr(config, "REACHY_ENABLE_MOTION", True)
    assert run(s).passed


def test_hand_in_view_none_is_not_checked_and_does_not_mean_allowed():
    s = scenario()
    s.kw["hand_in_view"] = None
    assert run(s).passed                       # today: not checked
    for value in (True, 1, "no"):
        s.kw["hand_in_view"] = value
        assert not run(s).passed               # only None and False pass


def test_a_missing_board_read_does_not_refuse_but_a_clear_one_does_not_rescue():
    s = scenario()
    s.kw["board_reads"] = {"r1c3": "empty"}
    assert run(s).passed
    s.kw["empty_confirmed"] = []
    assert not run(s).passed                   # a read can only refuse; it never stands in for the operator


def test_the_board_item_applies_to_every_leg_of_a_route_that_touches_the_board():
    for leg in range(8):
        s = scenario("rig_check", leg=leg)
        assert any("No rig check" in f for f in run(s).failures), leg


def test_a_route_that_never_nears_the_board_needs_no_board_checks():
    s = scenario("rig_check", "not_empty")
    for row in s.record["legs"]:
        row["kinematic"]["board_cm"] = 9.0
    assert run(s).passed


def test_defaults_refuse_everything():
    doc = taught(arm_routes.load_routes_doc())
    result = arm_gate_check(doc, ROUTE, 0, make_record(doc))
    assert not result.passed
    for text in ("Present r_", "Temperature of", "Operator approval", "motion lock", "Stop or abort", "No rig check",
                 "FK check not done", "duration", "step size"):
        assert any(text.lower() in f.lower() for f in result.failures), text


# --- monotonicity ------------------------------------------------------------------------------------------------

# Defects whose failure is independent of the leg being found; the others (an unknown route or leg) remove the
# leg-dependent checks, so their failures are not additive.
# Defects that overwrite the same argument as another one (a second value for it) are left to the single tests.
ADDITIVE = sorted(set(DEFECTS) - {"route_missing", "leg_index", "no_record", "no_duration", "step_unknown",
                                  "present_missing", "temperature_unknown", "fk_missing", "read_unknown",
                                  "layout", "limit"})   # a changed pose changes the speed text


def test_no_combination_of_defects_passes_and_each_defect_keeps_its_own_failure():
    singles = {d: set(run(scenario(d)).failures) for d in ADDITIVE}
    combos = [c for n in (2, 3) for c in itertools.combinations(ADDITIVE, n)]
    combos.append(tuple(ADDITIVE))
    for combo in combos:
        result = run(scenario(*combo))
        assert not result.passed, combo
        for d in combo:
            assert singles[d] <= set(result.failures), (combo, d)


PERMISSIVE = {"operator_approved": [True, 1, "yes"], "motion_enabled": [True, 1, None], "motion_lock_free": [True],
              "stop_unresolved": [False], "rig_check_done": [True], "hand_in_view": [False, None],
              "board_reads": [{}, {"r1c3": "empty"}], "empty_confirmed": [["r1c2", "r1c3", "r2c3", "r3c2"]]}


# The argument a defect is about: setting that one right is the fix, not a rescue.
OWN_ARGUMENT = {"approval": "operator_approved", "disabled": "motion_enabled", "lock": "motion_lock_free",
                "stop": "stop_unresolved", "rig_check": "rig_check_done", "hand": "hand_in_view",
                "not_empty": "empty_confirmed", "read_occupied": "board_reads", "read_unknown": "board_reads"}


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_no_permissive_argument_rescues_a_refused_leg(defect):
    refused = run(scenario(defect))
    assert not refused.passed
    for name, values in PERMISSIVE.items():
        if OWN_ARGUMENT.get(defect) == name:
            continue
        for value in values:
            s = scenario(defect)
            s.kw[name] = value
            again = run(s)
            assert not again.passed, (defect, name, value)
            assert set(refused.failures) <= set(again.failures) | {f for f in refused.failures if name in f}, (
                defect, name, value)


def test_fuzzed_arguments_pass_only_when_every_necessary_condition_holds():
    rng = random.Random(66)
    pools = {
        "operator_approved": [True, False, None, 1, "yes"], "motion_enabled": [True, False, None, 1],
        "motion_lock_free": [True, False, None, 1], "stop_unresolved": [True, False, None, 0],
        "rig_check_done": [True, False, None, 1], "hand_in_view": [True, False, None, 1],
        "board_reads": [{}, {"r1c3": "occupied"}, {"r2c3": "unknown"}, {"r1c3": "empty"}, None],
        "empty_confirmed": [[], ["r1c2", "r1c3", "r2c3", "r3c2"], ["r1c3"], ()],
        "fk_tip_m": [REST_TIP, None, (0.0, 0.0, 0.0), (0.141, -0.396 + 0.02, -0.397), [0.141, -0.396, -0.397]],
        "max_goal_step_deg": [0.2, 0.5, 0.51, None, float("nan")], "duration_s": [4.0, 3.9, None, float("inf")],
    }
    passes = 0
    for _ in range(3000):
        s = scenario()
        for name, values in pools.items():
            if rng.random() < 0.2:                  # mostly the satisfied value, so some samples can pass
                s.kw[name] = rng.choice(values)
        if rng.random() < 0.1:
            s.record["verdict"] = rng.choice(["pass", "fail", None])
        if not run(s).passed:
            continue
        passes += 1
        kw = s.kw
        assert kw["operator_approved"] is True and kw["motion_lock_free"] is True
        assert kw["stop_unresolved"] is False and kw["rig_check_done"] is True
        assert kw["motion_enabled"] in (True, 1, None) and kw["hand_in_view"] in (None, False)
        assert s.record["verdict"] == "pass" and kw["fk_tip_m"] is not None
        assert kw["max_goal_step_deg"] is not None and kw["max_goal_step_deg"] <= 0.5
        assert kw["duration_s"] is not None and kw["duration_s"] >= 4.0
        assert set(kw["empty_confirmed"]) >= {"r1c2", "r1c3", "r2c3", "r3c2"}
        assert all(v == "empty" for c, v in (kw["board_reads"] or {}).items() if c in ("r1c2", "r1c3", "r2c3", "r3c2"))
    assert passes > 0


# --- today's routes -----------------------------------------------------------------------------------------------

def test_no_route_is_playable_today_whatever_is_passed():
    """Every committed route and leg is refused, for every combination of the boolean arguments, with or without the
    owner decision: no committed record passes, and the REST poses are not taught."""
    doc = arm_routes.load_routes_doc()
    records = arm_records.load_records()
    grid = list(itertools.product([True, False], [True, False], [True, False], [True, False], [True, False, None]))
    for route in sorted(doc["routes"]):
        mine = next((r for r in records if r["route"] == route), None)
        assert mine is None or mine["verdict"] == "fail"
        for leg in range(2 * len(doc["routes"][route]["legs"])):
            for approved, enabled, lock, rig, hand in grid:
                result = arm_gate_check(doc, route, leg, mine, operator_approved=approved, motion_enabled=enabled,
                                        motion_lock_free=lock, stop_unresolved=False, rig_check_done=rig,
                                        empty_confirmed=["r1c2", "r1c3", "r2c3", "r3c2"], hand_in_view=hand,
                                        fk_tip_m=REST_TIP, rest_tip_m=REST_TIP)
                assert not result.passed and result.failures != [MOTION_DISABLED], (route, leg)


def test_the_packaged_config_covers_no_purpose():
    """The owner decision (D1) is what adds a purpose, in a commit of its own; the committed default is empty."""
    tree = ast.parse(Path(config.__file__).read_text())
    values = [n.value for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
              and getattr(n.target, "id", None) == "ARM_ALLOWED_PURPOSES"]
    assert len(values) == 1 and ast.literal_eval(values[0]) == ()
