"""Sim clearance records for arm routes (arm-motion design section 3.2): loading them and the stale-record check.

A record is `arm_route_records/<route>@<hash12>.json`, produced by the sim's tools/route_clearance and committed
here. A0 only reads them. Whether a record lets the robot play a leg is the arm gate's job (A1); `record_status`
here answers the narrower question the status script prints. No motion code.
"""
import json
from pathlib import Path

from reachy_ai import config
from reachy_ai.motion.arm_routes import pose_rule, route_hashes

RECORDS_DIR = Path(__file__).with_name("arm_route_records")

_REQUIRED = ("route", "route_hash", "verdict", "rig")
_RIG_REQUIRED = ("rig_layout_id", "mount_id", "scene_name")


class RecordError(ValueError):
    """A record file is malformed, or a committed record is stale."""


def _load_one(path):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RecordError("%s: not valid JSON (%s)" % (path.name, error))
    if not isinstance(record, dict):
        raise RecordError("%s: must be an object" % path.name)
    missing = [key for key in _REQUIRED if key not in record]
    if missing:
        raise RecordError("%s: missing field(s) %s" % (path.name, missing))
    if record["verdict"] not in ("pass", "fail"):
        raise RecordError("%s: verdict must be 'pass' or 'fail'" % path.name)
    rig = record["rig"]
    if not isinstance(rig, dict) or any(key not in rig for key in _RIG_REQUIRED):
        raise RecordError("%s: rig must name %s" % (path.name, list(_RIG_REQUIRED)))
    if not (isinstance(record["route_hash"], str) and len(record["route_hash"]) == 64):
        raise RecordError("%s: route_hash must be a 64-digit SHA-256" % path.name)
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


__all__ = ["RecordError", "check_no_stale", "load_records", "playability", "record_status",
           "stale_records"]
