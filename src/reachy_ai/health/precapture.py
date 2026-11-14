"""Pre-capture health check: read-only reads that show the robot is really running before a capture procedure.

Why: the robot's ROS logs (Reachy-Lab reachy-1-2-ros, SUMMARY_REPORT_OF_FINDINGS.md) hold hours in which every
motor bus was failing and `ros2_control_node` had aborted while the cameras and the SDK server stayed up, and
boot-time clock steps of up to 64 s. A capture taken then looks valid. `check(robot)` runs these checks, each on
its own, so one failure does not hide another:

- `sdk_reachable`: a unary `GetJointsState` round trip to the SDK server answers (the joint values the SDK object
  holds are a cache fed by a stream, so they cannot show this);
- `left_frame`, `right_frame`: each camera returns a frame and then a new one within the frame timeout, and the new
  one is not pixel-identical to the one before it or to the one the previous check saw (the pixel rule only where
  `require_pixel_change`: the simulator renders a still scene identically, so the viewer asks for it on the robot);
- `joints`: every joint in the SDK's inventory (`robot.joints`) returns a finite `present_position`;
- `neck`: `neck_roll`, `neck_pitch` and `neck_yaw` return a finite `present_position`;
- `force_gripper`: `force_sensors.r_force_gripper.force` returns a finite value;
- `clock_offset`: the robot's clock (the `JointsState.timestamp` the server stamps at send time) minus the
  workstation's, taken at the midpoint of the best of three round trips, is within `max_clock_offset_s`. A server
  that sends no timestamp fails this check: the offset is unknown, not zero.

Read-only: nothing is assigned on an SDK object and no command method is called (no goal, `turn_on`/`turn_off`,
compliance, zoom or focus setter, homing or autofocus). tests/test_probe_read_only.py enforces it. Every read is
bounded by a timeout. The robot's current ROS log session folder cannot be read without new robot-side code, so
the report carries it as a documented null (`ros_log_session()`).

The viewer runs `check` in its SDK process (stereo.py, a "health" request) when a procedure starts; the CLI
fallbacks run it on their own connection.
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from reachy_ai.config import (
    NECK_JOINTS,
    PRECAPTURE_FRAME_TIMEOUT_S,
    PRECAPTURE_MAX_CLOCK_OFFSET_S,
    PRECAPTURE_READ_TIMEOUT_S,
)

FORMAT_VERSION = 1
EYES = ("left", "right")
CHECKS = ("sdk_reachable", "left_frame", "right_frame", "joints", "neck", "force_gripper", "clock_offset")
CLOCK_SAMPLES = 3
FRAME_POLL_S = 0.02
FORCE_SENSOR = "r_force_gripper"
ROS_LOG_SESSION_REASON = (
    "not readable without new robot-side code: reachy_sdk has no read for it, and the audio helper's fixed NUC "
    "probes do not list ~/.ros/log. Match a bundle to the robot's logs by its UTC time and the clock offset.")
CLOCK_METHOD = ("JointsState.timestamp of a unary GetJointsState (the robot's system clock when the SDK server "
                "sent the reply) minus the workstation's time.time() at the round trip's midpoint; best of "
                f"{CLOCK_SAMPLES} by round-trip time")


class ReadTimeout(Exception):
    """A read did not return within its timeout."""


def _utc(unix_s: float) -> str:
    return datetime.fromtimestamp(unix_s, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


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


def _finite(value) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"not finite ({number})")
    return number


def _why(exc: BaseException) -> str:
    """The exception's type, with its message only for the kinds raised here (a gRPC error's message can carry the
    robot's address, which is never recorded)."""
    text = str(exc)
    return type(exc).__name__ + (f": {text}" if text and isinstance(exc, (ReadTimeout, ValueError, AttributeError))
                                 else "")


def ros_log_session() -> dict:
    """The robot's current ROS log session folder: a documented null (see ROS_LOG_SESSION_REASON)."""
    return {"folder": None, "reason": ROS_LOG_SESSION_REASON}


@dataclass
class Check:
    name: str
    passed: bool | None  # None: not run
    detail: str
    values: dict = field(default_factory=dict)


@dataclass
class HealthReport:
    checks: list[Check]
    checked_utc: str
    elapsed_s: float
    settings: dict
    ros_log_session: dict = field(default_factory=ros_log_session)
    format_version: int = FORMAT_VERSION

    @property
    def passed(self) -> bool:
        """True only when every check ran and passed."""
        return bool(self.checks) and all(c.passed is True for c in self.checks)

    def failed(self) -> list[Check]:
        """The checks that failed or did not run."""
        return [c for c in self.checks if c.passed is not True]

    def check(self, name: str) -> Check:
        for item in self.checks:
            if item.name == name:
                return item
        raise KeyError(name)

    def to_dict(self) -> dict:
        return {"format_version": self.format_version, "passed": self.passed, "checked_utc": self.checked_utc,
                "elapsed_s": self.elapsed_s, "settings": self.settings,
                "failed": [c.name for c in self.failed()], "checks": [asdict(c) for c in self.checks],
                "ros_log_session": self.ros_log_session}


def failed_lines(report: dict) -> list[str]:
    """'name: detail' for each check of a report dict that failed or did not run."""
    return [f"{c['name']}: {c['detail']}" for c in report.get("checks", []) if c.get("passed") is not True]


def refusal_text(report: dict) -> str:
    """The operator's message when a capture procedure is refused, one failed check per line."""
    return "\n".join(["Pre-capture health check failed; the capture procedure was not started:",
                      *(f"✗ {line}" for line in failed_lines(report))])


def clock_offset(report: dict | None) -> dict | None:
    """The clock_offset check's values of a report dict (offset_s, rtt_s, ...), or None if it has none."""
    for item in (report or {}).get("checks", []):
        if item.get("name") == "clock_offset":
            return {"passed": item.get("passed"), "detail": item.get("detail"), **(item.get("values") or {})}
    return None


def unreachable(reason: str, settings: dict | None = None) -> HealthReport:
    """A report for a connection that could not be asked at all: sdk_reachable failed, the rest not run."""
    checks = [Check("sdk_reachable", False, f"no SDK connection: {reason}")]
    checks += [Check(name, None, "not run: no SDK connection") for name in CHECKS[1:]]
    now = time.time()
    return HealthReport(checks=checks, checked_utc=_utc(now), elapsed_s=0.0, settings=settings or {})


def grpc_clock_reader(robot, timeout_s: float = PRECAPTURE_READ_TIMEOUT_S) -> Callable[[], dict]:
    """One clock sample per call: a unary GetJointsState for one joint on the SDK's own channel (a read).

    Returns {"sent_unix_s", "received_unix_s", "robot_unix_s"}; robot_unix_s is None when the reply has no
    timestamp."""
    channel = getattr(robot, "_grpc_channel", None)
    if channel is None:
        raise AttributeError("this SDK object has no gRPC channel")
    from reachy_sdk_api import joint_pb2, joint_pb2_grpc

    joint = next(iter(dict(robot.joints.items()).values()))
    request = joint_pb2.JointsStateRequest(ids=[joint_pb2.JointId(uid=int(joint.uid))],
                                           requested_fields=[joint_pb2.JointField.PRESENT_POSITION])
    stub = joint_pb2_grpc.JointServiceStub(channel)

    def read() -> dict:
        sent = time.time()
        reply = stub.GetJointsState(request, timeout=timeout_s)
        received = time.time()
        stamp = reply.timestamp if reply.HasField("timestamp") else None
        robot_s = stamp.seconds + stamp.nanos / 1e9 if stamp is not None and (stamp.seconds or stamp.nanos) else None
        return {"sent_unix_s": sent, "received_unix_s": received, "robot_unix_s": robot_s}

    return read


def _frame_digest(frame) -> str:
    return hashlib.sha256(repr((frame.shape, str(frame.dtype))).encode() + frame.tobytes()).hexdigest()


def _frames(robot, previous: dict, require_change: bool, frame_timeout_s: float, read_timeout_s: float,
            sleep) -> list[Check]:
    """Both eyes: a first frame, then a new one (a new decode: SDK 0.7 replaces the array per stream message)."""
    first, results, values = {}, {}, {}
    for eye in EYES:
        camera = getattr(robot, f"{eye}_camera", None)
        if camera is None:
            results[eye] = Check(f"{eye}_frame", False, f"no {eye} camera on the SDK object")
            continue
        try:
            frame = _bounded(lambda c=camera: c.last_frame, read_timeout_s)
        except Exception as exc:  # noqa: BLE001 - recorded; the other eye continues
            results[eye] = Check(f"{eye}_frame", False, f"no frame ({_why(exc)})")
            continue
        if frame is None or not getattr(frame, "size", 0):
            results[eye] = Check(f"{eye}_frame", False, "empty frame")
            continue
        first[eye] = (camera, frame)
    started = time.monotonic()
    fresh: dict[str, tuple] = {}
    while len(fresh) < len(first):
        for eye, (camera, frame) in first.items():
            if eye in fresh:
                continue
            try:
                latest = camera.last_frame
            except Exception as exc:  # noqa: BLE001
                results[eye] = Check(f"{eye}_frame", False, f"frame read failed ({_why(exc)})")
                fresh[eye] = None
                continue
            if latest is not None and latest is not frame and getattr(latest, "size", 0):
                fresh[eye] = (latest, time.monotonic() - started, time.time())
        if len(fresh) >= len(first) or time.monotonic() - started > frame_timeout_s:
            break
        sleep(FRAME_POLL_S)
    for eye, (camera, frame) in first.items():
        if eye in results:
            continue
        if eye not in fresh:
            results[eye] = Check(f"{eye}_frame", False,
                                 f"no new frame within {frame_timeout_s:g} s: the stream is not delivering",
                                 {"shape": list(frame.shape)})
            continue
        latest, waited_s, seen_unix = fresh[eye]
        digest = _frame_digest(latest)
        same_as_last = _frame_digest(frame) == digest
        same_as_previous = previous.get(eye) == digest
        values[eye] = {"shape": list(latest.shape), "sha256": digest, "new_frame_after_s": round(waited_s, 3),
                       "received_utc": _utc(seen_unix), "pixels_identical_to_last": same_as_last,
                       "pixels_identical_to_previous_check": same_as_previous}
        if require_change and same_as_last:
            results[eye] = Check(f"{eye}_frame", False,
                                 "two consecutive frames are pixel-identical (a frozen camera)", values[eye])
        elif require_change and same_as_previous:
            results[eye] = Check(f"{eye}_frame", False,
                                 "the frame is pixel-identical to the previous check's", values[eye])
        else:
            note = " (pixel-identical: allowed, the source renders a still scene identically)" if (
                same_as_last or same_as_previous) else ""
            results[eye] = Check(f"{eye}_frame", True, f"new frame after {waited_s:.2f} s{note}", values[eye])
    return [results[eye] for eye in EYES]


def _joints(robot, timeout_s: float) -> Check:
    try:
        items = _bounded(lambda: list(robot.joints.items()), timeout_s)
    except Exception as exc:  # noqa: BLE001
        return Check("joints", False, f"the joint inventory could not be read ({_why(exc)})")
    if not items:
        return Check("joints", False, "the SDK lists no joints")
    positions, bad = {}, {}
    for name, joint in items:
        try:
            positions[name] = _bounded(lambda j=joint: _finite(j.present_position), timeout_s)
        except Exception as exc:  # noqa: BLE001 - recorded; the other joints continue
            positions[name], bad[name] = None, _why(exc)
    values = {"count": len(items), "present_deg": positions, "bad": bad}
    if bad:
        return Check("joints", False, f"{len(bad)} of {len(items)} joints without a finite present_position: "
                     + ", ".join(f"{n} ({why})" for n, why in bad.items()), values)
    return Check("joints", True, f"{len(items)} joints, all finite", values)


def _neck(robot, timeout_s: float) -> Check:
    head = getattr(robot, "head", None)
    if head is None:
        return Check("neck", False, "no head on the SDK object")
    angles, bad = {}, {}
    for name in NECK_JOINTS:
        try:
            angles[name] = _bounded(lambda n=name: _finite(getattr(head.joints, n).present_position), timeout_s)
        except Exception as exc:  # noqa: BLE001
            angles[name], bad[name] = None, _why(exc)
    if bad:
        return Check("neck", False, "neck orientation unreadable: "
                     + ", ".join(f"{n} ({why})" for n, why in bad.items()), {"neck_deg": angles, "bad": bad})
    return Check("neck", True, "roll {neck_roll:+.1f}, pitch {neck_pitch:+.1f}, yaw {neck_yaw:+.1f} deg".format(
        **angles), {"neck_deg": angles})


def _force(robot, timeout_s: float) -> Check:
    try:
        force = _bounded(lambda: _finite(getattr(robot.force_sensors, FORCE_SENSOR).force), timeout_s)
    except Exception as exc:  # noqa: BLE001
        return Check("force_gripper", False, f"{FORCE_SENSOR} unreadable ({_why(exc)})")
    return Check("force_gripper", True, f"{FORCE_SENSOR} reads {force:g}", {"force": force})


def _clock(reader: Callable[[], dict] | None, reader_error: BaseException | None, max_offset_s: float,
           timeout_s: float) -> tuple[Check, Check]:
    """(sdk_reachable, clock_offset) from up to CLOCK_SAMPLES round trips."""
    if reader is None:
        return (Check("sdk_reachable", False, f"no round trip possible ({_why(reader_error)})"),
                Check("clock_offset", None, "not run: the SDK server did not answer"))
    samples, error = [], None
    for _ in range(CLOCK_SAMPLES):
        try:
            samples.append(_bounded(reader, timeout_s))
        except Exception as exc:  # noqa: BLE001
            error = exc
            break
    if not samples:
        return (Check("sdk_reachable", False, f"the SDK server did not answer ({_why(error)})"),
                Check("clock_offset", None, "not run: the SDK server did not answer"))
    best = min(samples, key=lambda s: s["received_unix_s"] - s["sent_unix_s"])
    rtt = best["received_unix_s"] - best["sent_unix_s"]
    reachable = Check("sdk_reachable", True, f"GetJointsState answered in {rtt * 1000:.1f} ms",
                      {"rtt_s": round(rtt, 6), "round_trips": len(samples)})
    values = {"limit_s": max_offset_s, "method": CLOCK_METHOD, "rtt_s": round(rtt, 6),
              "uncertainty_s": round(rtt / 2, 6), "sign": "robot minus workstation (positive: the robot is ahead)",
              "workstation_utc": _utc((best["sent_unix_s"] + best["received_unix_s"]) / 2)}
    if best.get("robot_unix_s") is None:
        return reachable, Check("clock_offset", False, "the robot's clock is unknown: the SDK server's reply "
                                "carried no timestamp", {**values, "offset_s": None, "robot_utc": None})
    offset = best["robot_unix_s"] - (best["sent_unix_s"] + best["received_unix_s"]) / 2
    values.update(offset_s=round(offset, 6), robot_utc=_utc(best["robot_unix_s"]))
    shown = f"{offset:+.3f}" if max_offset_s >= 0.01 else f"{offset:+.6f}"  # enough digits to compare to the limit
    if abs(offset) > max_offset_s:
        return reachable, Check("clock_offset", False, f"robot clock {shown} s off the workstation's "
                                f"(limit {max_offset_s:g} s)", values)
    return reachable, Check("clock_offset", True, f"{shown} s (limit {max_offset_s:g} s, "
                            f"±{rtt / 2 * 1000:.1f} ms)", values)


def check(robot, *, previous: HealthReport | dict | None = None,
          max_clock_offset_s: float = PRECAPTURE_MAX_CLOCK_OFFSET_S, require_pixel_change: bool = True,
          frame_timeout_s: float = PRECAPTURE_FRAME_TIMEOUT_S, read_timeout_s: float = PRECAPTURE_READ_TIMEOUT_S,
          clock_reader: Callable[[], dict] | None = None, sleep: Callable[[float], None] | None = None
          ) -> HealthReport:
    """Run every check against `robot` (a ReachySDK, or a ReachyClient: a dry-run client has no connection).

    `previous`: the last report from this connection; its frame digests are compared with this check's.
    `clock_reader`: one clock sample per call (default: `grpc_clock_reader(robot)`). Only reads."""
    sleep = sleep or time.sleep
    started, started_unix = time.monotonic(), time.time()
    settings = {"max_clock_offset_s": max_clock_offset_s, "require_pixel_change": require_pixel_change,
                "frame_timeout_s": frame_timeout_s, "read_timeout_s": read_timeout_s}
    if hasattr(robot, "dry_run") and hasattr(robot, "_reachy"):  # a ReachyClient
        if robot._reachy is None:
            return unreachable("a dry-run client has no robot connection", settings)
        robot = robot._reachy
    if robot is None:
        return unreachable("no robot object", settings)
    reader_error = None
    if clock_reader is None:
        try:
            clock_reader = grpc_clock_reader(robot, read_timeout_s)
        except Exception as exc:  # noqa: BLE001 - becomes the sdk_reachable failure
            reader_error = exc
    reachable, clock = _clock(clock_reader, reader_error, max_clock_offset_s, read_timeout_s)
    if isinstance(previous, HealthReport):
        previous = previous.to_dict()
    digests = {}
    for item in (previous or {}).get("checks", []):
        for eye in EYES:
            if item.get("name") == f"{eye}_frame" and (item.get("values") or {}).get("sha256"):
                digests[eye] = item["values"]["sha256"]
    left, right = _frames(robot, digests, require_pixel_change, frame_timeout_s, read_timeout_s, sleep)
    checks = [reachable, left, right, _joints(robot, read_timeout_s), _neck(robot, read_timeout_s),
              _force(robot, read_timeout_s), clock]
    return HealthReport(checks=checks, checked_utc=_utc(started_unix),
                        elapsed_s=round(time.monotonic() - started, 3), settings=settings)
