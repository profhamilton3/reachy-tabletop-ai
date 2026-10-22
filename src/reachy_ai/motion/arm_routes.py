"""Named arm routes as data (arm-motion design section 3.1): the strict loader, the joint limits, the playability rule.

No motion code: this module reads `arm_routes.json` and validates it. It imports neither the SDK nor primitives; the
route player and the arm gate (A1, A2) build on it.
"""
import json
import math
from pathlib import Path

from reachy_ai.motion.arm_route_hash import named_poses, route_hash

SCHEMA = "arm_routes.v1"
ROUTES_PATH = Path(__file__).with_name("arm_routes.json")

ARM_JOINTS = (
    "r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw", "r_elbow_pitch",
    "r_forearm_yaw", "r_wrist_pitch", "r_wrist_roll",
)
GRIPPER_JOINT = "r_gripper"

# Degrees, (lower, upper), the joint names reachy_sdk 0.7.0 uses for r_arm. Source: the URDF limits in
# reachy-1-2-sim native_mujoco/model/reachy.urdf @ 1948823, rounded INWARD to 0.01 deg, so nothing is wider than
# the URDF. The one deliberate exception is r_shoulder_roll's upper bound: +20.3 against the URDF's +9.97, the
# stop the physical robot held at B2_SAFE in Siva's poses (#185; reachy-1-2-sim docs/adr/0010-october-layout.md
# section 4). Nothing else is widened (OD-A6: routes stay inside URDF + 20.3).
ARM_LIMITS_DEG = {
    "r_shoulder_pitch": (-150.0, 89.95),
    "r_shoulder_roll": (-179.90, 20.3),
    "r_arm_yaw": (-89.95, 89.95),
    "r_elbow_pitch": (-125.01, 0.0),
    "r_forearm_yaw": (-99.98, 99.98),
    "r_wrist_pitch": (-44.97, 44.97),
    "r_wrist_roll": (-44.97, 44.97),
    "r_gripper": (-68.75, 20.05),
}

PURPOSES = ("gripper_calibration",)
PLAYABLE_AFTER_TEACHING = "not playable on the robot until taught (OD-A5)"

_DOC_KEYS = {"schema", "poses", "routes"}
_POSE_KEYS = {"joints_deg", "provenance"}
_TAUGHT_REQUIRED = {"kind", "source", "cell", "sha256", "by"}
_TAUGHT_KEYS = _TAUGHT_REQUIRED | {"taught_utc"}
_SIM_KEYS = {"kind", "adr", "source", "evidence"}
_ROUTE_REQUIRED = {"purpose", "start", "end", "legs", "close_pose", "return"}
_LEG_REQUIRED = {"to", "gripper_deg", "min_duration_s"}
_LEG_KEYS = _LEG_REQUIRED | {"near_board"}


class RouteFileError(ValueError):
    """The route document breaks a rule; the message names where."""


def _fail(where, message):
    raise RouteFileError("%s: %s" % (where, message))


def _number(where, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(where, "must be a number, got %r" % (value,))
    if not math.isfinite(value):
        _fail(where, "must be finite, got %r" % (value,))
    return float(value)


def _keys(where, obj, required, allowed=None):
    if not isinstance(obj, dict):
        _fail(where, "must be an object")
    allowed = required if allowed is None else allowed
    unknown = sorted(set(obj) - allowed)
    if unknown:
        _fail(where, "unknown field(s) %s" % unknown)
    missing = sorted(required - set(obj))
    if missing:
        _fail(where, "missing field(s) %s" % missing)


def _string(where, value):
    if not isinstance(value, str) or not value:
        _fail(where, "must be a non-empty string")


def _check_provenance(where, prov):
    if not isinstance(prov, dict):
        _fail(where, "must be an object")
    kind = prov.get("kind")
    if kind == "taught":
        _keys(where, prov, _TAUGHT_REQUIRED, _TAUGHT_KEYS)
        if isinstance(prov["cell"], bool) or not isinstance(prov["cell"], int) or prov["cell"] < 0:
            _fail(where + ".cell", "must be a non-negative integer")
        sha = prov["sha256"]
        if not (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
            _fail(where + ".sha256", "must be 64 lower-case hex digits")
        for key in sorted(_TAUGHT_KEYS & set(prov) - {"cell", "sha256"}):
            _string(where + "." + key, prov[key])
    elif kind == "sim":
        _keys(where, prov, _SIM_KEYS)
        for key in sorted(_SIM_KEYS):
            _string(where + "." + key, prov[key])
    else:
        _fail(where + ".kind", "must be 'taught' or 'sim', got %r" % (kind,))


def _check_pose(name, pose, limits):
    where = "poses.%s" % name
    _keys(where, pose, _POSE_KEYS)
    joints = pose["joints_deg"]
    _keys(where + ".joints_deg", joints, set(ARM_JOINTS))
    for joint in ARM_JOINTS:
        deg = _number("%s.joints_deg.%s" % (where, joint), joints[joint])
        lo, hi = limits[joint]
        if not lo <= deg <= hi:
            _fail("%s.joints_deg.%s" % (where, joint), "%.2f is outside the limits [%s, %s]" % (deg, lo, hi))
    _check_provenance(where + ".provenance", pose["provenance"])


def _check_route(name, route, poses, limits):
    where = "routes.%s" % name
    _keys(where, route, _ROUTE_REQUIRED)
    if route["purpose"] not in PURPOSES:
        _fail(where + ".purpose", "unknown purpose %r (known: %s)" % (route["purpose"], list(PURPOSES)))
    for key in ("start", "end", "close_pose"):
        if route[key] not in poses:
            _fail("%s.%s" % (where, key), "unknown pose %r" % (route[key],))
    if route["return"] != "reverse":
        _fail(where + ".return", "must be 'reverse', got %r" % (route["return"],))
    legs = route["legs"]
    if not isinstance(legs, list) or not legs:
        _fail(where + ".legs", "must be a non-empty list")
    glo, ghi = limits["r_gripper"]
    for i, leg in enumerate(legs):
        lwhere = "%s.legs[%d]" % (where, i)
        _keys(lwhere, leg, _LEG_REQUIRED, _LEG_KEYS)
        if leg["to"] not in poses:
            _fail(lwhere + ".to", "unknown pose %r" % (leg["to"],))
        grip = _number(lwhere + ".gripper_deg", leg["gripper_deg"])
        if not glo <= grip <= ghi:
            _fail(lwhere + ".gripper_deg", "%.2f is outside the limits [%s, %s]" % (grip, glo, ghi))
        if _number(lwhere + ".min_duration_s", leg["min_duration_s"]) <= 0:
            _fail(lwhere + ".min_duration_s", "must be positive")
        if "near_board" in leg and not isinstance(leg["near_board"], bool):
            _fail(lwhere + ".near_board", "must be true or false")
    if route["close_pose"] not in [leg["to"] for leg in legs]:
        _fail(where + ".close_pose", "must be one of the route's legs")


def validate_routes_doc(doc, limits=None):
    """Raise RouteFileError unless `doc` follows arm_routes.v1 strictly. Returns `doc`."""
    limits = ARM_LIMITS_DEG if limits is None else limits
    _keys("document", doc, _DOC_KEYS)
    if doc["schema"] != SCHEMA:
        _fail("schema", "must be %r, got %r" % (SCHEMA, doc["schema"]))
    poses, routes = doc["poses"], doc["routes"]
    if not isinstance(poses, dict) or not poses:
        _fail("poses", "must be a non-empty object")
    if not isinstance(routes, dict) or not routes:
        _fail("routes", "must be a non-empty object")
    for name, pose in poses.items():
        _check_pose(name, pose, limits)
    for name, route in routes.items():
        _check_route(name, route, poses, limits)
    return doc


def _reject_constant(token):
    raise RouteFileError("document: %s is not valid JSON" % token)


def load_routes_doc(path=None, limits=None):
    """Read and validate a route document (the packaged one by default). NaN and Infinity are refused."""
    text = Path(path or ROUTES_PATH).read_text(encoding="utf-8")
    try:
        doc = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise RouteFileError("document: not valid JSON (%s)" % error)
    return validate_routes_doc(doc, limits)


def route_hashes(doc, limits=None):
    limits = ARM_LIMITS_DEG if limits is None else limits
    return {name: route_hash(name, doc, limits) for name in sorted(doc["routes"])}


def sim_poses(doc, route_name):
    """The route's poses whose provenance is 'sim' (not yet taught on the robot), sorted."""
    route = doc["routes"][route_name]
    return [n for n in named_poses(route) if doc["poses"][n]["provenance"]["kind"] == "sim"]


def pose_rule(doc, route_name):
    """(True, None) when every pose of the route was taught; else (False, why). One input to playability: the
    arm gate (A1) also needs a passing record that matches the rig."""
    sim = sim_poses(doc, route_name)
    if sim:
        return False, "%s: sim pose(s) %s" % (PLAYABLE_AFTER_TEACHING, ", ".join(sim))
    return True, None
