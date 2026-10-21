"""Robot inventory (#62): the SDK reads, and the viewer's second redaction pass over the NUC probes.

`sdk_inventory(robot)` runs in the stereo viewer's SDK process, which owns the connection, as a read-only
"inventory" request on its bounded command channel. It only reads: the SDK version, every joint's present
position and compliance, the grippers, the force sensors, the fans, the cameras' zoom and focus, and the neck
sign check (IK only: `head._look_at` computes angles and moves nothing). Each read is bounded and wrapped on its
own: a failure records its error type and the rest continue. Nothing is ever assigned on an SDK object and no
command method is called.

`redact` repeats the audio helper's redaction (scripts/reachy_audio_helper.py) before anything is bundled:
IPv4/IPv6 and MAC addresses, home paths, serial-looking fields, and this workstation's host and user names
and the robot host. tests/test_robot_inventory.py checks the two give the same result.
"""
from __future__ import annotations

import getpass
import ipaddress
import json
import platform
import re
import socket
import threading
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable

READ_TIMEOUT_S = 5.0  # per read; lens reads can block in the SDK
PARTS = ("r_arm", "l_arm")
EYES = ("left", "right")
LENS_READS = ("zoom_level", "zoom", "focus")


class ReadTimeout(Exception):
    """A read did not return within READ_TIMEOUT_S."""


def _bounded(read: Callable[[], Any], timeout_s: float) -> Any:
    """`read()` in a daemon thread, waited for at most `timeout_s`; its exception is re-raised."""
    box: dict = {}

    def target():
        try:
            box["value"] = read()
        except BaseException as exc:  # noqa: BLE001 - handed back to the caller
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise ReadTimeout(f"no answer within {timeout_s:g} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def sdk_inventory(robot, neck_check: Callable | None = None, sdk_version: Callable[[], str] | None = None,
                  timeout_s: float = READ_TIMEOUT_S) -> dict:
    """Everything the inventory reads from the SDK. Only reads; each one on its own (see the module doc)."""
    errors: list[dict] = []

    def read(label: str, fn: Callable[[], Any]) -> Any:
        try:
            return _bounded(fn, timeout_s)
        except Exception as exc:  # noqa: BLE001 - recorded by type; the other reads continue
            errors.append({"read": label, "error": type(exc).__name__})
            return None

    def holder(label: str, get: Callable[[], Any]) -> list:
        items = read(label, lambda: list(get().items()))
        return items or []

    if sdk_version is None:
        def sdk_version():
            try:
                return version("reachy-sdk")
            except PackageNotFoundError:
                return None
    result: dict[str, Any] = {"sdk_version": read("sdk_version", sdk_version)}

    joints = {}
    for name, joint in holder("joints", lambda: robot.joints):
        present = read(f"joints.{name}.present_position", lambda j=joint: float(j.present_position))
        compliant = read(f"joints.{name}.compliant", lambda j=joint: bool(j.compliant))
        joints[name] = {"present_deg": present, "compliant": compliant}
    result["joints"], result["joint_count"] = joints, len(joints)

    grippers = {}
    for part in PARTS:
        arm = getattr(robot, part, None)
        if arm is None:
            continue
        name = f"{part[0]}_gripper"
        grippers[name] = {"present_deg": read(f"{part}.{name}.present_position",
                                              lambda a=arm, n=name: float(getattr(a.joints, n).present_position))}
    result["grippers"] = grippers

    result["force_sensors"] = {
        name: {"force": read(f"force_sensors.{name}.force", lambda s=sensor: float(s.force))}
        for name, sensor in holder("force_sensors", lambda: robot.force_sensors)}
    result["fans"] = {name: {"on": read(f"fans.{name}.is_on", lambda f=fan: bool(f.is_on))}
                      for name, fan in holder("fans", lambda: robot.fans)}

    cameras = {}
    for eye in EYES:
        camera = getattr(robot, f"{eye}_camera", None)
        if camera is None:
            cameras[eye] = None
            continue
        values = {}
        for field in LENS_READS:
            value = read(f"{eye}_camera.{field}", lambda c=camera, f=field: getattr(c, f))
            values[field] = getattr(value, "name", value)
        cameras[eye] = values
    result["cameras"] = cameras

    if neck_check is None:
        from reachy_ai.motion.primitives import neck_sign_check as neck_check  # IK only, nothing moves
    check = read("neck_sign_check", lambda: neck_check(robot))
    result["neck_sign_check"] = None if check is None else {"result": check.get("result"),
                                                            "detail": check.get("detail")}
    result["errors"] = errors
    return result


# --- redaction: the same rules as scripts/reachy_audio_helper.py ------------------------------------------

MAC_RE = re.compile(r"(?<![0-9A-Fa-f:-])[0-9A-Fa-f]{2}([:-])[0-9A-Fa-f]{2}(?:\1[0-9A-Fa-f]{2}){4}(?![0-9A-Fa-f:-])")
IPV6_RE = re.compile(r"(?<![0-9A-Za-z:.])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?:%[0-9A-Za-z_.-]+)?"
                     r"(?![0-9A-Za-z:])")
IPV4_RE = re.compile(r"(?<![0-9.])(?:\d{1,3}\.){3}\d{1,3}(?![0-9.])")
HOME_RE = re.compile(r"/(home|Users)/[^/\s:'\"]+")
SERIAL_RE = re.compile(r"(?i)\b([\w.-]*serial[\w.-]*(?:\s+number)?)(\s*[:=]\s*|\s+\d+\s+)([^\s,;]+)")
PLACEHOLDERS = ("<host>", "<ip>", "<mac>", "<user>", "<serial>")
#: An absolute ROS 2 topic name (the helper's TOPIC_PATTERN).
TOPIC_RE = re.compile(r"^(/[A-Za-z_][A-Za-z0-9_]{0,63}){1,8}$")


def topic_lines(text) -> set:
    """The lines of `ros2 topic list` output that are topic names."""
    return {line.strip() for line in (text or "").splitlines() if TOPIC_RE.fullmatch(line.strip())}


def _ipv6(match: re.Match) -> str:
    try:
        ipaddress.ip_address(match.group(0).split("%")[0])
    except ValueError:
        return match.group(0)  # a time, a USB id or a YAML key, not an address
    return "<ip>"


def redact_text(text: str, hosts=(), users=(), keep=()) -> str:
    """`text` with addresses, host and user names, home paths and serial-looking fields replaced; `keep` strings
    are spared the host and user names only (as in the helper)."""
    text = MAC_RE.sub("<mac>", text)
    text = IPV6_RE.sub(_ipv6, text)
    text = IPV4_RE.sub("<ip>", text)
    text = HOME_RE.sub(lambda m: f"/{m.group(1)}/<user>", text)
    kept = sorted({k for k in keep if k}, key=len, reverse=True)
    for index, name in enumerate(kept):
        text = text.replace(name, f"\0{index}\0")
    for name in sorted({h for h in hosts if h}, key=len, reverse=True):
        text = re.sub(r"(?i)(?<![\w-])" + re.escape(name) + r"(?![\w-])", "<host>", text)
    for name in sorted({u for u in users if u}, key=len, reverse=True):
        text = re.sub(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", "<user>", text)
    for index, name in enumerate(kept):
        text = text.replace(f"\0{index}\0", name)
    return SERIAL_RE.sub(lambda m: m.group(1) + m.group(2) + "<serial>", text)


def redact(value, hosts=(), users=(), keep=()):
    """redact_text over every string in a JSON-like value (keys included)."""
    if isinstance(value, str):
        return redact_text(value, hosts, users, keep)
    if isinstance(value, dict):
        return {redact(k, hosts, users, keep): redact(v, hosts, users, keep) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v, hosts, users, keep) for v in value]
    return value


def workstation_names(extra_hosts=()) -> tuple[set, set]:
    """(host names, user names) the viewer redacts: its own, plus `extra_hosts` (the robot host)."""
    hosts = {h for h in extra_hosts if h}
    for name in (socket.gethostname(), platform.node()):
        if name:
            hosts.update((name, name.split(".")[0]))
    try:
        users = {getpass.getuser()}
    except Exception:  # noqa: BLE001 - no user name is only less to redact
        users = set()
    return hosts, users


def count_replacements(before, after) -> int:
    """How many placeholders `after` has beyond `before`: what the viewer's pass caught that the helper missed."""
    old, new = json.dumps(before), json.dumps(after)
    return sum(new.count(p) - old.count(p) for p in PLACEHOLDERS)
