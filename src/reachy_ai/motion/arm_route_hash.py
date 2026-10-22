"""The route hash (arm-motion design section 3.1). Standard library only and Python 3.8-clean: the sim vendors this
file, and a route's record in either repo must name the same hash.

route_hash(route_name, routes_doc, limits) is the SHA-256 of the canonical JSON (sorted keys, separators (",", ":"),
ensure_ascii False) of {"route": the route, "poses": every pose it names (joints only, rounded to 0.01 deg),
"limits": the limits table}. Provenance is not hashed, so editing where a pose came from changes nothing; any change
to a pose, a leg, a duration, a gripper value or a limit changes the hash.
"""
import hashlib
import json


def _number(value):
    """A float, so 20 and 20.0 hash alike. -0.0 becomes 0.0."""
    return float(value) + 0.0


def _normalise(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _number(value)
    if isinstance(value, dict):
        return {key: _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    return value


def named_poses(route):
    """Every pose name a route refers to, sorted."""
    names = {route["start"], route["end"], route["close_pose"]}
    names.update(leg["to"] for leg in route["legs"])
    return sorted(names)


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def route_hash(route_name, routes_doc, limits):
    route = routes_doc["routes"][route_name]
    poses = {}
    for name in named_poses(route):
        joints = routes_doc["poses"][name]["joints_deg"]
        poses[name] = {joint: round(_number(deg), 2) + 0.0 for joint, deg in joints.items()}
    payload = {
        "route": _normalise(route),
        "poses": poses,
        "limits": {joint: [round(_number(lo), 2) + 0.0, round(_number(hi), 2) + 0.0]
                   for joint, (lo, hi) in limits.items()},
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
