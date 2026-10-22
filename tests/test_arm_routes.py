"""Arm routes, A0 (#66): the route file, the strict loader, the limits, the hash, the records, playability.
Data and loading only: no motion code, and nothing here may import the SDK, stereo or primitives."""

import ast
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from reachy_ai import config
from reachy_ai.motion import arm_records, arm_route_hash, arm_routes
from reachy_ai.motion.arm_route_hash import route_hash
from reachy_ai.motion.arm_routes import ARM_LIMITS_DEG, RouteFileError, load_routes_doc, validate_routes_doc

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "fixtures" / "arm_routes"
MODULES = ("arm_routes", "arm_route_hash", "arm_records")
ARM = arm_routes.ARM_JOINTS


@pytest.fixture
def doc():
    return load_routes_doc()


def broken(doc, mutate):
    bad = copy.deepcopy(doc)
    mutate(bad)
    return bad


def test_packaged_file_loads_and_has_the_expected_content(doc):
    assert set(doc["poses"]) == {"PICK_SAFE", "PICK", "TRANSPORT", "B2_SAFE", "B2_PLACE", "REST", "REST_OUT"}
    route = doc["routes"]["B2_GRIP_TUNE"]
    assert route["purpose"] == "gripper_calibration" and route["start"] == route["end"] == "REST"
    assert [(leg["to"], leg["gripper_deg"], leg["min_duration_s"]) for leg in route["legs"]] == [
        ("REST_OUT", 20.0, 4.0), ("TRANSPORT", 20.0, 8.4), ("B2_SAFE", -29.0, 6.0), ("B2_PLACE", -29.0, 5.0)]
    assert route["legs"][-1]["near_board"] is True
    assert route["close_pose"] == "B2_PLACE" and route["return"] == "reverse"


def test_every_pose_has_provenance_and_the_joints_are_sdk_names(doc):
    for name, pose in doc["poses"].items():
        assert tuple(pose["joints_deg"]) == ARM, name
        assert pose["provenance"]["kind"] in ("taught", "sim"), name


def test_taught_poses_match_the_notebook_cell_and_hash(doc):
    notebook = json.loads((ROOT / "tests" / "MovePieceLogic.ipynb").read_text(encoding="utf-8"))
    for name in ("PICK_SAFE", "PICK", "TRANSPORT", "B2_SAFE", "B2_PLACE"):
        prov = doc["poses"][name]["provenance"]
        assert prov["kind"] == "taught" and prov["by"] == "Siva" and prov["source"] == "tests/MovePieceLogic.ipynb"
        source = "".join(notebook["cells"][prov["cell"]]["source"])
        assert hashlib.sha256(source.encode("utf-8")).hexdigest() == prov["sha256"], name
        for joint, deg in doc["poses"][name]["joints_deg"].items():
            assert "reachy.r_arm.%s: %.2f" % (joint, deg) in source, (name, joint)


def test_rest_poses_are_sim_provenance_citing_adr_and_step1(doc):
    for name in ("REST", "REST_OUT"):
        prov = doc["poses"][name]["provenance"]
        assert prov["kind"] == "sim" and prov["adr"] == "ADR-0010"
        assert prov["evidence"] == "docs/design/arm-motion-step1/"
        assert (ROOT / prov["evidence"]).is_dir()


# --- loader refusals, one per rule --------------------------------------------------------------------------------

def refuse(doc, mutate, text):
    with pytest.raises(RouteFileError, match=text):
        validate_routes_doc(broken(doc, mutate))


def test_refuses_unknown_top_level_field(doc):
    refuse(doc, lambda d: d.update(extra=1), "unknown field")


def test_refuses_wrong_schema(doc):
    refuse(doc, lambda d: d.update(schema="arm_routes.v2"), "schema")


def test_refuses_unknown_pose_field(doc):
    refuse(doc, lambda d: d["poses"]["REST"].update(extra=1), "poses.REST: unknown field")


def test_refuses_unknown_joint(doc):
    refuse(doc, lambda d: d["poses"]["REST"]["joints_deg"].update(r_gripper=1.0), "unknown field")


def test_refuses_missing_joint(doc):
    refuse(doc, lambda d: d["poses"]["REST"]["joints_deg"].pop("r_wrist_roll"), "missing field")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "10", True, None])
def test_refuses_non_finite_or_non_numeric_values(doc, value):
    refuse(doc, lambda d: d["poses"]["REST"]["joints_deg"].update(r_wrist_roll=value), "wrist_roll")


def test_refuses_nan_in_the_file_text(tmp_path, doc):
    path = tmp_path / "routes.json"
    path.write_text(json.dumps(doc).replace('"r_wrist_roll": -17.5', '"r_wrist_roll": NaN', 1))
    with pytest.raises(RouteFileError, match="NaN"):
        load_routes_doc(path)


def test_refuses_invalid_json(tmp_path):
    path = tmp_path / "routes.json"
    path.write_text("{")
    with pytest.raises(RouteFileError, match="not valid JSON"):
        load_routes_doc(path)


def test_refuses_unknown_pose_reference_in_each_place(doc):
    refuse(doc, lambda d: d["routes"]["B2_GRIP_TUNE"]["legs"][0].update(to="NOWHERE"), "legs\\[0\\].to")
    refuse(doc, lambda d: d["routes"]["B2_GRIP_TUNE"].update(start="NOWHERE"), "start")
    refuse(doc, lambda d: d["routes"]["B2_GRIP_TUNE"].update(end="NOWHERE"), "end")
    refuse(doc, lambda d: d["routes"]["B2_GRIP_TUNE"].update(close_pose="NOWHERE"), "close_pose")


def test_refuses_close_pose_outside_the_legs(doc):
    refuse(doc, lambda d: d["routes"]["B2_GRIP_TUNE"].update(close_pose="PICK"), "one of the route's legs")


def test_refuses_unknown_purpose(doc):
    refuse(doc, lambda d: d["routes"]["B2_GRIP_TUNE"].update(purpose="juggling"), "unknown purpose")


def test_refuses_unknown_route_and_leg_fields_and_bad_return(doc):
    refuse(doc, lambda d: d["routes"]["B2_GRIP_TUNE"].update(extra=1), "unknown field")
    refuse(doc, lambda d: d["routes"]["B2_GRIP_TUNE"]["legs"][0].update(extra=1), "unknown field")
    refuse(doc, lambda d: d["routes"]["B2_GRIP_TUNE"].update({"return": "forward"}), "reverse")


def test_refuses_missing_leg_gripper_and_bad_duration(doc):
    refuse(doc, lambda d: d["routes"]["B2_GRIP_TUNE"]["legs"][0].pop("gripper_deg"), "missing field")
    refuse(doc, lambda d: d["routes"]["B2_GRIP_TUNE"]["legs"][0].update(min_duration_s=0), "positive")
    refuse(doc, lambda d: d["routes"]["B2_GRIP_TUNE"]["legs"][0].update(near_board="yes"), "near_board")


def test_refuses_values_outside_the_limits(doc):
    refuse(doc, lambda d: d["poses"]["TRANSPORT"]["joints_deg"].update(r_wrist_roll=45.0), "outside the limits")
    refuse(doc, lambda d: d["poses"]["TRANSPORT"]["joints_deg"].update(r_shoulder_roll=20.4), "outside the limits")
    refuse(doc, lambda d: d["routes"]["B2_GRIP_TUNE"]["legs"][0].update(gripper_deg=21.0), "outside the limits")


def test_refuses_bad_provenance(doc):
    refuse(doc, lambda d: d["poses"]["PICK"].pop("provenance"), "missing field")
    refuse(doc, lambda d: d["poses"]["PICK"]["provenance"].update(kind="guess"), "taught' or 'sim")
    refuse(doc, lambda d: d["poses"]["PICK"]["provenance"].update(extra=1), "unknown field")
    refuse(doc, lambda d: d["poses"]["PICK"]["provenance"].pop("sha256"), "missing field")
    refuse(doc, lambda d: d["poses"]["PICK"]["provenance"].update(sha256="abc"), "sha256")
    refuse(doc, lambda d: d["poses"]["REST"]["provenance"].pop("evidence"), "missing field")


# --- limits -------------------------------------------------------------------------------------------------------

def test_limits_cover_the_arm_and_gripper_and_widen_only_shoulder_roll():
    urdf_deg = {  # reachy-1-2-sim native_mujoco/model/reachy.urdf @ 1948823, degrees
        "r_shoulder_pitch": (-150.00035, 89.95437), "r_shoulder_roll": (-179.90875, 9.96947),
        "r_arm_yaw": (-89.95437, 89.95437), "r_elbow_pitch": (-125.01939, 0.0),
        "r_forearm_yaw": (-99.98114, 99.98114), "r_wrist_pitch": (-44.97719, 44.97719),
        "r_wrist_roll": (-44.97719, 44.97719), "r_gripper": (-68.75494, 20.05352)}
    assert set(ARM_LIMITS_DEG) == set(urdf_deg)
    for joint, (lo, hi) in ARM_LIMITS_DEG.items():
        ulo, uhi = urdf_deg[joint]
        assert lo >= ulo - 1e-3 and hi <= (20.3 if joint == "r_shoulder_roll" else uhi + 1e-3), joint
    assert ARM_LIMITS_DEG["r_shoulder_roll"][1] == 20.3


# --- playability --------------------------------------------------------------------------------------------------

def test_route_with_a_sim_pose_is_not_playable(doc):
    ok, why = arm_routes.pose_rule(doc, "B2_GRIP_TUNE")
    assert not ok and "not playable on the robot until taught (OD-A5)" in why
    assert "REST" in why and "REST_OUT" in why
    assert arm_routes.sim_poses(doc, "B2_GRIP_TUNE") == ["REST", "REST_OUT"]


def test_all_taught_route_passes_the_pose_rule_but_still_needs_a_record(doc):
    def teach(d):
        for name in ("REST", "REST_OUT"):
            d["poses"][name]["provenance"] = copy.deepcopy(d["poses"]["PICK"]["provenance"])
    taught = broken(doc, teach)
    assert arm_routes.pose_rule(taught, "B2_GRIP_TUNE") == (True, None)
    playable, why = arm_records.playability(taught, "B2_GRIP_TUNE", [])
    assert not playable and "record status none" in why


def test_sim_pose_outside_the_route_does_not_count(doc):
    other = broken(doc, lambda d: d["routes"].update(
        PICKING={**copy.deepcopy(d["routes"]["B2_GRIP_TUNE"]), "start": "PICK", "end": "PICK",
                 "legs": [{"to": "PICK_SAFE", "gripper_deg": 20.0, "min_duration_s": 5.0}], "close_pose": "PICK_SAFE"}))
    assert arm_routes.pose_rule(other, "PICKING") == (True, None)


# --- the hash -----------------------------------------------------------------------------------------------------

def hashes(doc, limits=ARM_LIMITS_DEG):
    return route_hash("B2_GRIP_TUNE", doc, limits)


def test_hash_is_stable(doc):
    assert hashes(doc) == hashes(load_routes_doc()) == arm_routes.route_hashes(doc)["B2_GRIP_TUNE"]
    assert len(hashes(doc)) == 64


def test_hash_ignores_key_order_and_int_vs_float(doc):
    reordered = json.loads(json.dumps(doc), object_pairs_hook=lambda pairs: dict(reversed(pairs)))
    assert hashes(reordered) == hashes(doc)
    as_ints = broken(doc, lambda d: d["routes"]["B2_GRIP_TUNE"]["legs"][0].update(gripper_deg=20, min_duration_s=4))
    assert hashes(as_ints) == hashes(doc)


@pytest.mark.parametrize("mutate", [
    lambda d: d["poses"]["B2_PLACE"]["joints_deg"].update(r_elbow_pitch=-75.92),            # a pose, 0.02 deg
    lambda d: d["poses"]["REST_OUT"]["joints_deg"].update(r_shoulder_roll=-24.0),           # a pose on the route
    lambda d: d["routes"]["B2_GRIP_TUNE"]["legs"][0].update(min_duration_s=4.5),            # a duration
    lambda d: d["routes"]["B2_GRIP_TUNE"]["legs"][1].update(gripper_deg=19.0),              # a gripper value
    lambda d: d["routes"]["B2_GRIP_TUNE"]["legs"][3].update(near_board=False),              # a leg flag
    lambda d: d["routes"]["B2_GRIP_TUNE"]["legs"][0].update(to="TRANSPORT"),                # a leg target
    lambda d: d["routes"]["B2_GRIP_TUNE"].update(close_pose="B2_SAFE"),                     # the close pose
], ids=["pose", "named-pose", "duration", "gripper", "flag", "target", "close-pose"])
def test_hash_changes_with_the_route_or_its_poses(doc, mutate):
    assert hashes(broken(doc, mutate)) != hashes(doc)


def test_hash_changes_with_a_limit(doc):
    wider = dict(ARM_LIMITS_DEG, r_wrist_roll=(-44.97, 44.98))
    assert hashes(doc, wider) != hashes(doc)


def test_hash_rounds_poses_to_a_hundredth_of_a_degree(doc):
    same = broken(doc, lambda d: d["poses"]["B2_PLACE"]["joints_deg"].update(r_elbow_pitch=-75.9400001))
    assert hashes(same) == hashes(doc)


def test_hash_ignores_provenance_and_unnamed_poses(doc):
    def edit(d):
        d["poses"]["REST"]["provenance"]["source"] = "somewhere else"
        d["poses"]["PICK"]["provenance"]["by"] = "someone"
        d["poses"]["PICK"]["joints_deg"]["r_wrist_roll"] = 0.0    # PICK is not on B2_GRIP_TUNE
    assert hashes(broken(doc, edit)) == hashes(doc)


def test_golden_hashes():
    golden = json.loads((GOLDEN / "golden.json").read_text(encoding="utf-8"))
    expected = json.loads((GOLDEN / "golden_hashes.json").read_text(encoding="utf-8"))
    limits = {joint: tuple(pair) for joint, pair in expected["limits"].items()}
    validate_routes_doc(golden, limits)
    assert expected["hashes"] and set(expected["hashes"]) == set(golden["routes"])
    for name, digest in expected["hashes"].items():
        assert route_hash(name, golden, limits) == digest


def test_golden_hash_matches_a_hand_built_canonical_form():
    golden = json.loads((GOLDEN / "golden.json").read_text(encoding="utf-8"))
    expected = json.loads((GOLDEN / "golden_hashes.json").read_text(encoding="utf-8"))
    route = golden["routes"]["GOLDEN_SHORT"]
    text = ('{"limits":%s,"poses":{"A":%s,"B":%s},"route":{"close_pose":"B","end":"A","legs":'
            '[{"gripper_deg":20.0,"min_duration_s":3.0,"to":"B"}],"purpose":"gripper_calibration","return":"reverse",'
            '"start":"A"}}')
    limits = json.dumps({j: [float(a), float(b)] for j, (a, b) in sorted(expected["limits"].items())},
                        separators=(",", ":"))

    def pose(name):
        joints = golden["poses"][name]["joints_deg"]
        return json.dumps({j: round(v, 2) for j, v in sorted(joints.items())}, separators=(",", ":"))
    assert route["legs"][0]["gripper_deg"] == 20
    built = text % (limits, pose("A"), pose("B"))
    assert hashlib.sha256(built.encode()).hexdigest() == expected["hashes"]["GOLDEN_SHORT"]


def test_hash_module_is_stdlib_only_and_python_38_clean():
    path = Path(arm_route_hash.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 8))
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported <= {"hashlib", "json"}, imported
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
                or (isinstance(n, ast.FunctionDef) and n.returns)]


# --- records ------------------------------------------------------------------------------------------------------

RECORD_TEMPLATE = ROOT / "tests" / "fixtures" / "arm_route_records" / "B2_GRIP_TUNE_sim_output.json"


def record(doc, name="B2_GRIP_TUNE", verdict="pass", layout=config.ARM_RIG_LAYOUT_ID, mount=config.ARM_MOUNT_ID,
           digest=None):
    """A schema-valid record, from the sim's real B2_GRIP_TUNE output (a fail: the target square). A "pass" one
    has no reasons and a target placement that holds to 1 mm."""
    rec = json.loads(RECORD_TEMPLATE.read_text(encoding="utf-8"))
    rec.update(route=name, route_hash=digest or route_hash(name, doc, ARM_LIMITS_DEG), verdict=verdict)
    rec["rig"].update(rig_layout_id=layout, mount_id=mount)
    if verdict == "pass":
        rec["reasons"] = []
        for kind, piece in rec["target_placement"]["pieces"].items():
            piece["steps"] = [{"offset_mm": 0, "moved_mm": 0.0}, {"offset_mm": 1, "moved_mm": 0.2},
                              {"offset_mm": 2, "moved_mm": 1.3}]
            piece["tolerance_mm"] = rec["target_placement_tolerance_mm"][kind] = 1
    return rec


def write(directory, rec):
    path = directory / ("%s@%s.json" % (rec["route"], rec["route_hash"][:12]))
    path.write_text(json.dumps(rec))
    return path


def test_rig_identity_in_config():
    assert config.ARM_RIG_LAYOUT_ID == "fwd-center-lab" and config.ARM_MOUNT_ID == "october-2026"


def test_committed_records_directory_has_no_stale_record(doc):
    arm_records.check_no_stale(doc, arm_records.load_records())


def test_stale_record_fails(doc, tmp_path):
    write(tmp_path, record(doc, digest="f" * 64))
    records = arm_records.load_records(tmp_path)
    assert arm_records.stale_records(doc, records) == records
    with pytest.raises(arm_records.RecordError, match="stale"):
        arm_records.check_no_stale(doc, records)


def test_record_goes_stale_when_the_route_changes(doc, tmp_path):
    write(tmp_path, record(doc))
    records = arm_records.load_records(tmp_path)
    arm_records.check_no_stale(doc, records)
    changed = broken(doc, lambda d: d["routes"]["B2_GRIP_TUNE"]["legs"][0].update(min_duration_s=4.5))
    with pytest.raises(arm_records.RecordError, match="stale"):
        arm_records.check_no_stale(changed, records)


def test_record_loader_refusals(doc, tmp_path):
    good = record(doc)
    for bad, text in (({**good, "verdict": "maybe"}, "verdict"),
                      ({k: v for k, v in good.items() if k != "rig"}, "missing"),
                      ({**good, "rig": {"rig_layout_id": "x"}}, "rig must name"),
                      ({**good, "route_hash": "abc"}, "64-digit")):
        folder = tmp_path / text.replace(" ", "_")
        folder.mkdir()
        (folder / ("%s@%s.json" % (good["route"], good["route_hash"][:12]))).write_text(json.dumps(bad))
        with pytest.raises(arm_records.RecordError, match=text):
            arm_records.load_records(folder)
    (tmp_path / "wrongname.json").write_text(json.dumps(good))
    with pytest.raises(arm_records.RecordError, match="file name must be"):
        arm_records.load_records(tmp_path)


def test_committed_b2_grip_tune_record_is_valid_and_fails_on_the_target_square(doc):
    (rec,) = [r for r in arm_records.load_records() if r["route"] == "B2_GRIP_TUNE"]
    assert rec["route_hash"] == arm_routes.route_hashes(doc)["B2_GRIP_TUNE"]
    assert arm_records.record_status(doc, "B2_GRIP_TUNE", [rec]) == "fail"
    assert rec["reasons"] and all(r.startswith("target r2c2: a centred ") for r in rec["reasons"])
    assert rec["target_cell"] == "r2c2" and "r2c2" not in rec["occupancy"]
    assert rec["target_placement_tolerance_mm"] == {"cube": None, "cylinder": None}
    assert sorted(c for c, s in rec["occupancy"].items() if s == "must_be_empty") == ["r1c2", "r1c3", "r2c3", "r3c2"]
    conditional = {(d["pose"], d["pair"]): d["condition"] for d in rec["declared_contacts"]}
    assert conditional == {("B2_PLACE", "r_forearm_col/table_top"): None, ("REST", "r_forearm_col/table_top"): None,
                           ("REST", "r_hand_back_col/table_top"): {"max_joint_deviation_deg": 1.0},
                           ("REST", "r_palm_col/table_top"): {"max_joint_deviation_deg": 1.0}}


def test_the_template_is_the_committed_sim_output_modulo_nothing():
    committed = sorted(arm_records.RECORDS_DIR.glob("B2_GRIP_TUNE@*.json"))
    assert [p.read_bytes() for p in committed] == [RECORD_TEMPLATE.read_bytes()]


def _refused(rec, text, tmp_path):
    folder = tmp_path / str(abs(hash(text)))
    folder.mkdir()
    (folder / ("%s@%s.json" % (rec["route"], rec["route_hash"][:12]))).write_text(json.dumps(rec))
    with pytest.raises(arm_records.RecordError, match=text):
        arm_records.load_records(folder)


def _rest_palm(rec):
    for leg in rec["legs"]:
        for c in leg["physics"]["contacts"]:
            if c["pair"] == "r_palm_col/table_top" and c["at"] == "REST":
                return c
    raise AssertionError("no palm contact at REST in the template")


def test_record_schema_is_strict_and_self_consistent(doc, tmp_path):
    good = record(doc)
    arm_records.validate_record(copy.deepcopy(good))
    arm_records.validate_record(record(doc, verdict="fail"))
    cases = []
    cases.append(({**good, "extra": 1}, r"unknown field\(s\) \['extra'\]"))
    cases.append(({**good, "schema": "arm_route_record.v0"}, "must be 'arm_route_record.v1'"))
    cases.append(({**good, "reasons": ["x"]}, "'pass' needs no reasons"))
    cases.append(({**record(doc, verdict="fail"), "reasons": []}, "'fail' needs at least one"))
    bad = copy.deepcopy(good)
    bad["rig"]["extra"] = "x"
    cases.append((bad, r"rig: unknown field"))
    # a conditional declaration: the palm at REST is declared only within 1 deg of REST
    bad = copy.deepcopy(good)
    _rest_palm(bad)["deviation_deg"]["REST"] = 2.0
    cases.append((bad, "2.0 deg from REST \\(the declaration allows 1.0\\)"))
    bad = copy.deepcopy(good)
    bad["declared_contacts"] = [d for d in bad["declared_contacts"] if d["pair"] != "r_palm_col/table_top"]
    cases.append((bad, "nothing declares r_palm_col/table_top at REST"))
    bad = copy.deepcopy(good)
    bad["declared_contacts"][0]["condition"] = {"max_joint_deviation_deg": 0}
    cases.append((bad, "must be positive"))
    bad = copy.deepcopy(good)
    _rest_palm(bad)["declared"] = False
    cases.append((bad, "a passing record has an undeclared contact"))
    # the target square
    bad = copy.deepcopy(good)
    bad["occupancy"]["r2c2"] = "must_be_empty"
    cases.append((bad, "must not include the target square r2c2"))
    bad = copy.deepcopy(good)
    bad["target_placement_tolerance_mm"]["cube"] = 2
    cases.append((bad, "does not follow from the steps"))
    bad = copy.deepcopy(good)
    bad["target_placement"]["pieces"]["cube"]["steps"][1]["offset_mm"] = 3
    cases.append((bad, "1 mm steps from the centre"))
    bad = copy.deepcopy(good)
    cube = bad["target_placement"]["pieces"]["cube"]
    cube["steps"] = [{"offset_mm": 0, "moved_mm": 2.46}]
    cube["tolerance_mm"] = bad["target_placement_tolerance_mm"]["cube"] = None
    cases.append((bad, "a centred target piece was disturbed, so the verdict cannot pass"))
    bad = copy.deepcopy(good)
    del bad["target_placement_tolerance_mm"]["cylinder"]
    cases.append((bad, "must name exactly the pieces"))
    for rec, text in cases:
        _refused(rec, text, tmp_path)


def test_record_status_and_playability(doc):
    def taught(d):
        for name in ("REST", "REST_OUT"):
            d["poses"][name]["provenance"] = copy.deepcopy(d["poses"]["PICK"]["provenance"])
    ready = broken(doc, taught)
    status = lambda recs: arm_records.record_status(ready, "B2_GRIP_TUNE", recs)  # noqa: E731
    assert status([]) == "none"
    assert status([record(ready)]) == "pass"
    assert status([record(ready, verdict="fail")]) == "fail"
    assert status([record(ready, layout="other-lab")]) == "rig mismatch"
    assert status([record(ready, mount="august-2026")]) == "rig mismatch"
    assert status([record(ready, digest="f" * 64)]) == "none"
    assert arm_records.playability(ready, "B2_GRIP_TUNE", [record(ready)]) == (True, None)
    assert not arm_records.playability(ready, "B2_GRIP_TUNE", [record(ready, verdict="fail")])[0]
    # a passing record does not make a sim-pose route playable
    assert not arm_records.playability(doc, "B2_GRIP_TUNE", [record(doc)])[0]


def test_status_script_lists_the_route(doc):
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "arm_routes_status.py")],
                         capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "B2_GRIP_TUNE" in out.stdout and hashes(doc) in out.stdout
    assert "record    fail" in out.stdout and "OD-A5" in out.stdout


# --- no SDK, stereo or primitives --------------------------------------------------------------------------------

@pytest.mark.parametrize("module", MODULES)
def test_modules_import_no_sdk_stereo_or_primitives(module):
    tree = ast.parse((ROOT / "src" / "reachy_ai" / "motion" / (module + ".py")).read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names += [base] + [base + "." + a.name for a in node.names]
    banned = ("reachy_sdk", "reachy2_sdk", "reachy_ai.stereo", "reachy_ai.motion.primitives", "reachy_ai.sdk_compat")
    assert not [n for n in names if n.startswith(banned)], names
