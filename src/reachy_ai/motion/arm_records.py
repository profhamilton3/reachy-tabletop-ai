"""Sim clearance records for arm routes (arm-motion design section 3.2): loading them and the stale-record check.

A record is `arm_route_records/<route>@<hash12>.json`, produced by the sim's tools/route_clearance and committed
here. A0 only reads them. Whether a record lets the robot play a leg is the arm gate's job (A1); `record_status`
here answers the narrower question the status script prints. No motion code.

The format is `arm_route_record.v1`, checked strictly: every object has exactly its fields. Beyond the shape, the
loader checks what the record claims against itself: the verdict against its reasons; each contact marked declared
against a declaration for that pose and pair, and a conditional declaration's joint-deviation bound against the
contact's measured deviation; the target square kept out of the occupancy map; and each target-placement tolerance
against its 1 mm steps.
"""
import json
import math
import re
from pathlib import Path

from reachy_ai import config
from reachy_ai.motion.arm_routes import pose_rule, route_hashes

RECORDS_DIR = Path(__file__).with_name("arm_route_records")
SCHEMA = "arm_route_record.v1"
MOVE_LIMIT_MM = 1.0                   # a piece that moves this much is disturbed (design section 3.2)

_TOP = {"schema", "route", "route_hash", "verdict", "reasons", "rig", "sim", "routes_sha256", "legs",
        "physics_settle", "declared_contacts", "occupancy", "target_cell", "target_placement_tolerance_mm",
        "target_placement", "layouts_flown", "head", "method"}
_RIG = {"rig_layout_id", "mount_id", "scene_name", "scene_sha256", "scene_revision"}
_SIM = {"commit", "model_sha256", "mujoco_version", "tool_sha256"}
_LEG = {"from", "to", "direction", "duration_s", "gripper_deg", "kinematic", "physics"}
_KINEMATIC = {"board_cm", "board_geom", "board_t_s", "rig_cm", "left_arm_cm", "self_cm", "zones"}
_PHYSICS = {"follow_deg", "arrival_deg", "arrival_weak_deg", "contacts"}
_CONTACT = {"pair", "t_s", "steps", "at", "declared", "deviation_deg"}
_SETTLE = {"at", "judged", "contacts"}
_DECLARATION = {"pose", "pair", "condition"}
_CONDITION = {"max_joint_deviation_deg"}
_TARGET = {"close_pose", "approach_xy", "pieces"}
_TARGET_PIECE = {"body", "steps", "centred_piece_contacts", "tolerance_mm"}
_STEP = {"offset_mm", "moved_mm"}
_LAYOUT_REQUIRED = {"name", "pieces", "judged"}
_LAYOUT_KEYS = _LAYOUT_REQUIRED | {"moved_mm", "piece_contacts", "cell", "piece", "offset_mm", "clear_cm",
                                   "clear_geom", "status"}
_HEAD = {"neck_deg", "at_limit", "off_axis_deg", "tip_in_view_left", "tip_in_view_right", "tip_px_left",
         "tip_px_right"}
_NECK = {"neck_roll", "neck_pitch", "neck_yaw"}
_METHOD = {"board_distance", "piece_rig_left_self_distance", "cap_cm", "stream_hz", "settle_s", "hold_s",
           "max_avg_speed_deg_s", "leg_start", "moved", "near_board_cm", "contact_lag_s", "follow_max_deg",
           "arrival_max_deg", "follow_joints", "arrival_joints", "occupancy_clear_cm", "occupancy_move_mm",
           "occupancy_offsets_per_axis", "target_step_mm", "head_frame_px"}
_STATUSES = ("may_be_occupied", "must_be_empty")
_CELL = re.compile(r"^r[1-3]c[1-3]$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class RecordError(ValueError):
    """A record file is malformed, or a committed record is stale."""


def _fail(where, message):
    raise RecordError("%s: %s" % (where, message))


def _obj(where, value, keys, required=None):
    if not isinstance(value, dict):
        _fail(where, "must be an object")
    required = keys if required is None else required
    missing = sorted(required - set(value))
    if missing:
        _fail(where, "missing field(s) %s" % missing)
    unknown = sorted(set(value) - keys)
    if unknown:
        _fail(where, "unknown field(s) %s" % unknown)
    return value


def _num(where, value, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(where, "must be a finite number")
    if minimum is not None and value < minimum:
        _fail(where, "must be >= %s" % minimum)
    return value


def _str(where, value):
    if not isinstance(value, str) or not value:
        _fail(where, "must be a non-empty string")
    return value


def _bool(where, value):
    if not isinstance(value, bool):
        _fail(where, "must be true or false")
    return value


def _list(where, value):
    if not isinstance(value, list):
        _fail(where, "must be a list")
    return value


def _contact(where, c):
    _obj(where, c, _CONTACT)
    _str(where + ".pair", c["pair"])
    if c["pair"].count("/") != 1:
        _fail(where + ".pair", "must be 'geom/geom'")
    t_s = _list(where + ".t_s", c["t_s"])
    if len(t_s) != 2:
        _fail(where + ".t_s", "must be [first, last]")
    for t in t_s:
        _num(where + ".t_s", t, 0)
    _num(where + ".steps", c["steps"], 1)
    if c["at"] is not None:
        _str(where + ".at", c["at"])
    _bool(where + ".declared", c["declared"])
    if not isinstance(c["deviation_deg"], dict):
        _fail(where + ".deviation_deg", "must be an object")
    for pose, dev in c["deviation_deg"].items():
        _num("%s.deviation_deg.%s" % (where, pose), dev, 0)


def _check_declared(where, c, declarations):
    """A contact marked declared needs a declaration for its pose and pair; a conditional one also needs the
    measured deviation from that pose within the bound."""
    if not c["declared"]:
        return
    key = (c["at"], c["pair"])
    if key not in declarations:
        _fail(where, "marked declared, but nothing declares %s at %s" % (c["pair"], c["at"]))
    bound = declarations[key]
    if bound is not None:
        dev = c["deviation_deg"].get(c["at"])
        if dev is None or dev > bound:
            _fail(where, "marked declared, but %s is %s deg from %s (the declaration allows %s)"
                  % (c["pair"], dev, c["at"], bound))


def placement_tolerance(steps):
    """The largest offset up to which every 1 mm step moved the piece < MOVE_LIMIT_MM; None if the centred one
    moved. The sim tool's rule (tools/route_clearance evaluate.placement_tolerance)."""
    best = None
    for step in steps:
        if step["moved_mm"] >= MOVE_LIMIT_MM:
            break
        best = step["offset_mm"]
    return best


def validate_record(record, where="record"):
    """Raise RecordError unless `record` is a well-formed arm_route_record.v1 that agrees with itself."""
    _obj(where, record, _TOP)
    if record["schema"] != SCHEMA:
        _fail(where + ".schema", "must be %r" % SCHEMA)
    _str(where + ".route", record["route"])
    for key in ("route_hash", "routes_sha256"):
        if not (isinstance(record[key], str) and _HEX64.match(record[key])):
            _fail("%s.%s" % (where, key), "must be a 64-digit SHA-256")
    if record["verdict"] not in ("pass", "fail"):
        _fail(where + ".verdict", "verdict must be 'pass' or 'fail'")
    reasons = _list(where + ".reasons", record["reasons"])
    for reason in reasons:
        _str(where + ".reasons[]", reason)
    if (record["verdict"] == "pass") != (not reasons):
        _fail(where + ".verdict", "'pass' needs no reasons, and 'fail' needs at least one")
    rig = record["rig"]
    if not isinstance(rig, dict) or any(key not in rig for key in ("rig_layout_id", "mount_id", "scene_name")):
        _fail(where, "rig must name %s" % ["rig_layout_id", "mount_id", "scene_name"])
    _obj(where + ".rig", rig, _RIG)
    for key in sorted(_RIG):
        _str("%s.rig.%s" % (where, key), rig[key])
    sim = _obj(where + ".sim", record["sim"], _SIM)
    for key in sorted(_SIM):
        _str("%s.sim.%s" % (where, key), sim[key])

    declarations = {}
    for i, dec in enumerate(_list(where + ".declared_contacts", record["declared_contacts"])):
        dwhere = "%s.declared_contacts[%d]" % (where, i)
        _obj(dwhere, dec, _DECLARATION)
        _str(dwhere + ".pose", dec["pose"])
        _str(dwhere + ".pair", dec["pair"])
        bound = None
        if dec["condition"] is not None:
            cond = _obj(dwhere + ".condition", dec["condition"], _CONDITION)
            bound = _num(dwhere + ".condition.max_joint_deviation_deg", cond["max_joint_deviation_deg"], 0)
            if bound <= 0:
                _fail(dwhere + ".condition.max_joint_deviation_deg", "must be positive")
        key = (dec["pose"], dec["pair"])
        if key in declarations:
            _fail(dwhere, "declares %s at %s twice" % (dec["pair"], dec["pose"]))
        declarations[key] = bound

    legs = _list(where + ".legs", record["legs"])
    if not legs:
        _fail(where + ".legs", "must not be empty")
    for i, leg in enumerate(legs):
        lwhere = "%s.legs[%d]" % (where, i)
        _obj(lwhere, leg, _LEG)
        _str(lwhere + ".from", leg["from"])
        _str(lwhere + ".to", leg["to"])
        if leg["direction"] not in ("out", "return"):
            _fail(lwhere + ".direction", "must be 'out' or 'return'")
        _num(lwhere + ".duration_s", leg["duration_s"], 0)
        _num(lwhere + ".gripper_deg", leg["gripper_deg"])
        kin = _obj(lwhere + ".kinematic", leg["kinematic"], _KINEMATIC)
        for key in ("board_cm", "board_t_s", "rig_cm", "left_arm_cm", "self_cm"):
            _num("%s.kinematic.%s" % (lwhere, key), kin[key])
        if not isinstance(kin["zones"], dict):
            _fail(lwhere + ".kinematic.zones", "must be an object")
        if leg["physics"] is None:
            if record["verdict"] == "pass":
                _fail(lwhere + ".physics", "a passing record has every leg flown")
            continue
        phys = _obj(lwhere + ".physics", leg["physics"], _PHYSICS)
        for key in ("follow_deg", "arrival_deg", "arrival_weak_deg"):
            _num("%s.physics.%s" % (lwhere, key), phys[key], 0)
        for j, c in enumerate(_list(lwhere + ".physics.contacts", phys["contacts"])):
            cwhere = "%s.physics.contacts[%d]" % (lwhere, j)
            _contact(cwhere, c)
            _check_declared(cwhere, c, declarations)
            if record["verdict"] == "pass" and not c["declared"]:
                _fail(cwhere, "a passing record has an undeclared contact")

    settle = record["physics_settle"]
    if settle is not None:
        _obj(where + ".physics_settle", settle, _SETTLE)
        _bool(where + ".physics_settle.judged", settle["judged"])
        for j, c in enumerate(_list(where + ".physics_settle.contacts", settle["contacts"])):
            cwhere = "%s.physics_settle.contacts[%d]" % (where, j)
            _contact(cwhere, c)
            _check_declared(cwhere, c, declarations)

    occupancy = record["occupancy"]
    if not isinstance(occupancy, dict):
        _fail(where + ".occupancy", "must be an object")
    for cell, status in occupancy.items():
        if not _CELL.match(cell):
            _fail(where + ".occupancy", "unknown cell %r" % cell)
        if status not in _STATUSES:
            _fail("%s.occupancy.%s" % (where, cell), "must be one of %s" % list(_STATUSES))
    target = record["target_cell"]
    if target is not None:
        if not (isinstance(target, str) and _CELL.match(target)):
            _fail(where + ".target_cell", "must be a cell like 'r2c2', or null")
        if target in occupancy:
            _fail(where + ".occupancy", "must not include the target square %s" % target)

    placement = _obj(where + ".target_placement", record["target_placement"], _TARGET)
    _str(where + ".target_placement.close_pose", placement["close_pose"])
    pieces = placement["pieces"]
    if not isinstance(pieces, dict):
        _fail(where + ".target_placement.pieces", "must be an object")
    tolerances = record["target_placement_tolerance_mm"]
    if not isinstance(tolerances, dict) or set(tolerances) != set(pieces):
        _fail(where + ".target_placement_tolerance_mm", "must name exactly the pieces of target_placement")
    if target is not None and record["verdict"] == "pass" and not pieces:
        _fail(where + ".target_placement", "a passing record with a target square has its placement flown")
    for kind, piece in pieces.items():
        pwhere = "%s.target_placement.pieces.%s" % (where, kind)
        _obj(pwhere, piece, _TARGET_PIECE)
        _str(pwhere + ".body", piece["body"])
        steps = _list(pwhere + ".steps", piece["steps"])
        if not steps:
            _fail(pwhere + ".steps", "must not be empty")
        for k, step in enumerate(steps):
            _obj("%s.steps[%d]" % (pwhere, k), step, _STEP)
            if step["offset_mm"] != k:
                _fail("%s.steps[%d].offset_mm" % (pwhere, k), "must be %d (1 mm steps from the centre)" % k)
            _num("%s.steps[%d].moved_mm" % (pwhere, k), step["moved_mm"], 0)
        for text in _list(pwhere + ".centred_piece_contacts", piece["centred_piece_contacts"]):
            _str(pwhere + ".centred_piece_contacts[]", text)
        tol = tolerances[kind]
        if tol is not None:
            _num("%s.target_placement_tolerance_mm.%s" % (where, kind), tol, 0)
        if piece["tolerance_mm"] != tol or placement_tolerance(steps) != tol:
            _fail("%s.target_placement_tolerance_mm.%s" % (where, kind),
                  "%r does not follow from the steps (%r)" % (tol, placement_tolerance(steps)))
        if tol is None and record["verdict"] == "pass":
            _fail("%s.target_placement_tolerance_mm.%s" % (where, kind),
                  "a centred target piece was disturbed, so the verdict cannot pass")

    for i, layout in enumerate(_list(where + ".layouts_flown", record["layouts_flown"])):
        lwhere = "%s.layouts_flown[%d]" % (where, i)
        _obj(lwhere, layout, _LAYOUT_KEYS, _LAYOUT_REQUIRED)
        _str(lwhere + ".name", layout["name"])
        _bool(lwhere + ".judged", layout["judged"])
        if "status" in layout and layout["status"] not in _STATUSES:
            _fail(lwhere + ".status", "must be one of %s" % list(_STATUSES))
        if layout.get("cell") is not None and layout.get("cell") == target:
            _fail(lwhere, "the target square is not part of the occupancy flights")
    head = record["head"]
    if not isinstance(head, dict):
        _fail(where + ".head", "must be an object")
    for pose, row in head.items():
        hwhere = "%s.head.%s" % (where, pose)
        _obj(hwhere, row, _HEAD)
        neck = _obj(hwhere + ".neck_deg", row["neck_deg"], _NECK)
        for joint in sorted(_NECK):
            _num("%s.neck_deg.%s" % (hwhere, joint), neck[joint])
        for key in ("at_limit", "tip_in_view_left", "tip_in_view_right"):
            _bool("%s.%s" % (hwhere, key), row[key])
    _obj(where + ".method", record["method"], _METHOD)
    return record


def _load_one(path):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RecordError("%s: not valid JSON (%s)" % (path.name, error))
    validate_record(record, path.name)
    expected = "%s@%s.json" % (record["route"], record["route_hash"][:12])
    if path.name != expected:
        raise RecordError("%s: file name must be %s" % (path.name, expected))
    return record


def load_records(directory=None):
    """Every record in the directory, sorted by file name. Malformed ones raise RecordError."""
    directory = Path(directory or RECORDS_DIR)
    return [_load_one(path) for path in sorted(directory.glob("*.json"))]


def stale_records(doc, records, limits=None):
    """The records naming a hash that no route currently has (a changed route leaves its old record stale)."""
    current = set(route_hashes(doc, limits).values())
    return [r for r in records if r["route_hash"] not in current]


def check_no_stale(doc, records, limits=None):
    stale = stale_records(doc, records, limits)
    if stale:
        names = ["%s@%s" % (r["route"], r["route_hash"][:12]) for r in stale]
        raise RecordError("stale record(s), naming a hash no route has: %s" % ", ".join(names))


def record_status(doc, route_name, records, limits=None,
                  rig_layout_id=config.ARM_RIG_LAYOUT_ID, mount_id=config.ARM_MOUNT_ID):
    """One of 'none', 'fail', 'rig mismatch', 'pass' for the route's current hash."""
    current = route_hashes(doc, limits)[route_name]
    mine = [r for r in records if r["route"] == route_name and r["route_hash"] == current]
    if not mine:
        return "none"
    matching = [r for r in mine if r["rig"]["rig_layout_id"] == rig_layout_id and r["rig"]["mount_id"] == mount_id]
    if not matching:
        return "rig mismatch"
    return "pass" if any(r["verdict"] == "pass" for r in matching) else "fail"


def playability(doc, route_name, records, limits=None):
    """(playable, reason). Playable needs every pose taught AND a passing record for this rig. This is the
    A0 summary; the arm gate (A1) checks the rest of section 5."""
    ok, why = pose_rule(doc, route_name)
    if not ok:
        return False, why
    status = record_status(doc, route_name, records, limits)
    if status != "pass":
        return False, "not playable: record status %s" % status
    return True, None


__all__ = ["RecordError", "SCHEMA", "check_no_stale", "load_records", "placement_tolerance", "playability",
           "record_status", "stale_records", "validate_record"]
