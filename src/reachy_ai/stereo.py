"""Reachy v1 stereo acquisition, shared by all browser consumers. Camera and neck only, never arms.

The SDK owns its camera streaming threads. A single child process contains that
connection so its blocking startup/first-frame calls can be stopped on timeout.
No arm, gripper, focus, or robot-service command is issued here. Two commands exist, both requested
only by a data-capture procedure after the operator's explicit confirm, both run in the child
process that owns the connection, one at a time (bounded queues of 1):

- a camera zoom level (camera_zoom.ZoomSetter): dry-run (logged only) unless the viewer is started
  with --allow-zoom-commands;
- a named head pose, a Stop (hold the present position) and a smooth release
  (motion.primitives, behind safety.head_gate_check): dry-run unless REACHY_ENABLE_MOTION=true AND
  the viewer is started with --allow-head-motion. Owner decision 2026-10-20: "Neck motion is
  required for vision data capture. Capture procedures may articulate the head through the neck for
  viewing and perception. Arm motion stays forbidden in capture procedures."
"""

import argparse
import copy
import logging
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import cv2
import numpy as np

LOG = logging.getLogger(__name__)
EYES = ("left", "right")
OBSERVATION_QUEUE = 4  # frames and head readbacks in flight; the newest are kept
ZOOM_QUEUE = HEAD_QUEUE = 1  # commands in flight to the SDK process; a second one is refused


class CommandBusy(RuntimeError):
    """A zoom or head command is already pending or running."""


@dataclass(frozen=True)
class StereoSettings:
    source: str = "sim"
    robot_host: str = "localhost"
    sdk_port: int = 50051
    camera_port: int = 50051
    fps: float = 15
    preview_width: int = 640
    stale_seconds: float = 2
    restart_seconds: float = 12
    max_pair_skew_ms: float = 250
    zoom_commands: bool = False  # False: a requested zoom change is only logged (dry-run)
    head_motion: bool = False    # False: a head move is only logged; True still needs REACHY_ENABLE_MOTION

    def __post_init__(self):
        if self.source not in ("sim", "robot") or not self.robot_host:
            raise ValueError("Choose sim or robot and provide a robot host")
        if not 1 <= self.fps <= 30 or not 160 <= self.preview_width <= 1920:
            raise ValueError("Preview FPS must be 1–30 and width 160–1920")
        if not 0 < self.stale_seconds < self.restart_seconds or self.max_pair_skew_ms <= 0:
            raise ValueError("Require positive skew and 0 < stale interval < restart interval")


def _source(message):
    """The observation stream a frame or head message belongs to."""
    return message.get("eye") or message["kind"]


class _LatestObservations:
    """Producer side of the bounded observation queue: when it is full, discard
    the OLDEST message so the queue always holds the newest. Counts discards per
    source and stamps the running counts on every message it sends."""

    def __init__(self, messages, discard_wait=0.1):
        self.messages = messages
        self.discard_wait = discard_wait
        self.lock = threading.Lock()
        self.dropped: dict[str, int] = {}

    def publish(self, message):
        with self.lock:
            message["dropped"] = dict(self.dropped)
            try:
                self.messages.put_nowait(message)
                return
            except queue.Full:
                pass
            try:
                # Bounded wait: an mp.Queue reports full before its feeder thread
                # has flushed the oldest item. Only reached while the consumer stalls.
                old = self.messages.get(timeout=self.discard_wait)
                self.dropped[_source(old)] = self.dropped.get(_source(old), 0) + 1
            except queue.Empty:
                pass  # The consumer took it meanwhile.
            message["dropped"] = dict(self.dropped)
            try:
                self.messages.put_nowait(message)
            except queue.Full:
                self.dropped[_source(message)] = self.dropped.get(_source(message), 0) + 1


def _sdk_worker(settings: StereoSettings, messages, control, stop, commands=None, head=None):
    """Reuse the notebooks' ReachySDK + last_frame path, once per application.

    Frames and head readbacks go to the bounded `messages` queue, newest kept.
    Ready, error and diagnostics messages go to `control`, which is unbounded:
    they are few, and a failure report must never be dropped. `commands` carries explicit
    zoom requests from a data-capture procedure; `head` is (requests, halt): head moves and
    releases, and the Stop event. Results come back on `control`.
    """
    # Ctrl+C belongs to the web-server parent, which disposes this worker.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    observe = _LatestObservations(messages).publish  # A slow consumer must never block SDK acquisition.

    def publish(message):
        control.put_nowait(message)

    # The SDK reports lost streams through background-thread exceptions.
    # Keep the useful failure type without dumping gRPC endpoint addresses.
    def thread_error(args):
        publish({"kind": "error", "error": args.exc_type.__name__})

    threading.excepthook = thread_error

    try:
        from reachy_sdk import ReachySDK

        robot = ReachySDK(
            host=settings.robot_host, sdk_port=settings.sdk_port,
            camera_port=settings.camera_port,
        )
    except Exception as exc:
        publish({"kind": "error", "error": type(exc).__name__})
        return
    publish({"kind": "ready", "sdk_version": version("reachy-sdk")})

    def read_eye(eye):
        camera = getattr(robot, f"{eye}_camera", None)
        if camera is None:
            publish({"kind": "error", "eye": eye, "error": "CameraUnavailable"})
            return
        previous = None
        sequence = 0
        while not stop.is_set():
            try:
                frame = camera.last_frame
                # SDK 0.7 replaces the ndarray for each decoded stream message.
                # Re-reading the same cached object must NOT refresh its age.
                if frame is not previous and frame is not None and frame.size:
                    previous = frame
                    sequence += 1
                    observe({"kind": "frame", "eye": eye, "sequence": sequence,
                             "observed_ns": time.monotonic_ns(), "pixels": frame.copy()})
            except Exception as exc:
                publish({"kind": "error", "eye": eye, "error": type(exc).__name__})
                return
            stop.wait(1 / settings.fps)

    def lens_diagnostics(eye):
        # These are optional reads, off the frame path. A stuck read cannot
        # prevent viewing; the containing process is disposable on shutdown.
        values = {}
        for name in ("zoom_level", "zoom", "focus"):
            try:
                value = getattr(getattr(robot, f"{eye}_camera"), name)
                values[name] = getattr(value, "name", value)
            except Exception as exc:
                values[name] = None
                values[f"{name}_error"] = type(exc).__name__
            publish({"kind": "diagnostics", "section": eye, "values": dict(values)})

    def head_diagnostics():
        # Lens reads can block in the SDK. Keep pose readbacks independent.
        first = True
        while not stop.is_set():
            head = {}
            read_started = time.monotonic_ns()
            for name in ("neck_roll", "neck_pitch", "neck_yaw"):
                try:
                    joint = getattr(robot.head.joints, name)
                    head[name] = {"present_deg": float(joint.present_position),
                                  "goal_deg": float(joint.goal_position), "compliant": bool(joint.compliant)}
                except Exception as exc:
                    head[name] = {"error": type(exc).__name__}
            if first:
                publish({"kind": "diagnostics", "section": "head_at_connection", "values": head})
                first = False
            observe({"kind": "head", "values": head, "observed_ns": read_started})
            stop.wait(.2)

    def zoom_commands():
        from reachy_ai.camera_zoom import ZoomSetter
        setter = ZoomSetter(robot, dry_run=not settings.zoom_commands)
        while not stop.is_set():
            try:
                request = commands.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                records = setter.set_both(request["level"])
            except Exception as exc:  # noqa: BLE001 - report the type only
                publish({"kind": "zoom_result", "id": request["id"], "error": type(exc).__name__,
                         "finished_ns": time.monotonic_ns()})
                continue
            publish({"kind": "zoom_result", "id": request["id"], "dry_run": setter.dry_run, "records": records,
                     "finished_ns": time.monotonic_ns()})

    def head_commands():
        # The only thread that commands the neck. Stop is an event, so it reaches a running move.
        from reachy_ai.motion import primitives
        requests, halt = head
        live = settings.head_motion  # primitives also require REACHY_ENABLE_MOTION and a passed gate
        while not stop.is_set():
            if halt.is_set():
                record = primitives.hold_head(robot, allow_live=live)
                halt.clear()
                publish({"kind": "head_result", "id": "stop", "record": record, "finished_ns": time.monotonic_ns()})
                continue
            try:
                request = requests.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if request["action"] == "move":
                    record = primitives.look_at_preset(robot, request["preset"], operator_approved=request["approved"],
                                                       allow_live=live, stop_event=halt)
                else:
                    record = primitives.release_head(robot, operator_approved=request["approved"], allow_live=live)
            except Exception as exc:  # noqa: BLE001 - report the type only
                record = {"action": request["action"], "preset": request.get("preset"), "outcome": "error",
                          "error": type(exc).__name__}
            publish({"kind": "head_result", "id": request["id"], "record": record, "finished_ns": time.monotonic_ns()})

    if commands is not None:
        threading.Thread(target=zoom_commands, daemon=True).start()
    if head is not None:
        threading.Thread(target=head_commands, daemon=True).start()
    for eye in EYES:
        threading.Thread(target=read_eye, args=(eye,), daemon=True).start()
        threading.Thread(target=lens_diagnostics, args=(eye,), daemon=True).start()
    threading.Thread(target=head_diagnostics, daemon=True).start()
    stop.wait()


@dataclass(frozen=True)
class EyeFrame:
    frame_id: str
    sequence: int
    observed_ns: int
    pixels: np.ndarray
    preview: bytes

    def metadata(self):
        height, width = self.pixels.shape[:2]
        return {"frame_id": self.frame_id, "sequence": self.sequence,
                "width": width, "height": height, "encoding": "bgr8",
                "observed_time_ns": self.observed_ns, "capture_time_ns": None}


class StereoService:
    def __init__(self, settings: StereoSettings):
        self.settings = settings
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.session = ""
        self.frames: dict[str, EyeFrame] = {}
        self.snapshots: OrderedDict[str, dict] = OrderedDict()
        self.diagnostics: dict[str, Any] = {}
        self.errors: dict[str, str] = {}
        self.rates: dict[str, float] = {}
        self.dropped: dict[str, int] = {}     # lost to a full queue, counted by the SDK process
        self.superseded: dict[str, int] = {}  # delivered, skipped for a newer one of the same source
        self.state = "connecting"
        self.head = None
        self.commands = None
        self._zoom_waits: dict[str, dict] = {}
        self.head_requests = None
        self.head_halt = None
        self._head_waits: dict[str, dict] = {}
        self.head_busy = False
        self.command_done_ns: int | None = None  # when the last zoom or head command finished
        self.head_target: dict | None = None     # the last live head move's record, until released
        self.last_head_command: dict | None = None  # the last head command that was not refused, any mode

    def start(self):
        self.thread = threading.Thread(target=self._run, name="stereo-acquisition", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def _new_session(self):
        with self.lock:
            self.session = uuid.uuid4().hex
            self.frames.clear()
            self.diagnostics.clear()
            self.errors.clear()
            self.rates.clear()
            self.head = None
            self.dropped = {}
            self.superseded = {}
            self.head_busy = False
            self.head_target = self.last_head_command = None  # a new connection knows nothing of earlier commands

    def _run(self):
        context = mp.get_context("spawn")
        while not self.stop_event.is_set():
            self._new_session()
            messages = context.Queue(maxsize=OBSERVATION_QUEUE)
            control = context.Queue()
            commands = context.Queue(maxsize=ZOOM_QUEUE)
            head_requests, head_halt = context.Queue(maxsize=HEAD_QUEUE), context.Event()
            self.commands, self.head_requests, self.head_halt = commands, head_requests, head_halt
            worker_stop = context.Event()
            worker = context.Process(target=_sdk_worker,
                                     args=(self.settings, messages, control, worker_stop, commands,
                                           (head_requests, head_halt)),
                                     daemon=True)
            worker.start()
            started = time.monotonic()
            try:
                while not self.stop_event.is_set():
                    self._drain(messages, control)
                    with self.lock:
                        ages = [(time.monotonic_ns() - self.frames[e].observed_ns) / 1e9
                                if e in self.frames else time.monotonic() - started for e in EYES]
                    if not worker.is_alive() or max(ages) > self.settings.restart_seconds:
                        LOG.warning("Stereo acquisition reconnecting; source=%s", self.settings.source)
                        break
            finally:
                worker_stop.set()
                worker.join(timeout=0.5)
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=1)
                self._drain_control(control)  # e.g. the error that ended the worker
                messages.close()
                control.close()
                commands.close()
                head_requests.close()
                with self.lock:
                    self.state = "reconnecting"
                    self.frames.clear()
            self.stop_event.wait(2)

    def _drain(self, messages, control, timeout=0.2):
        """Apply everything already delivered: the newest frame per eye and the
        newest head readback, then control messages in order. Control goes last
        so an older frame from the same batch cannot clear a newer error."""
        batch = []
        try:
            batch.append(messages.get(timeout=timeout))
            while True:
                batch.append(messages.get_nowait())
        except queue.Empty:
            pass
        newest: dict[str, dict] = {}
        for message in batch:
            source = _source(message)
            if source in newest:
                with self.lock:
                    self.superseded[source] = self.superseded.get(source, 0) + 1
            newest[source] = message
        for message in newest.values():
            self._accept(message)
        self._drain_control(control)

    def _drain_control(self, control):
        while True:
            try:
                self._accept(control.get_nowait())
            except queue.Empty:
                return

    def _accept(self, message):
        kind = message["kind"]
        with self.lock:
            for source, count in message.get("dropped", {}).items():
                # Running totals from the SDK process; messages may apply out of order.
                self.dropped[source] = max(self.dropped.get(source, 0), count)
            if kind == "ready":
                self.diagnostics["sdk_version"] = message["sdk_version"]
                LOG.info("SDK connected; source=%s version=%s", self.settings.source, message["sdk_version"])
            elif kind == "diagnostics":
                self.diagnostics[message["section"]] = message["values"]
                LOG.info("SDK readback %s: %s", message["section"], message["values"])
            elif kind == "zoom_result":
                self._apply_zoom_result(message)
            elif kind == "head_result":
                self._apply_head_result(message)
            elif kind == "head":
                self.head = {"joints": message["values"], "observed_ns": message["observed_ns"]}
            elif kind == "error":
                self.errors[message.get("eye", "connection")] = message["error"]
                LOG.warning("SDK %s: %s", message.get("eye", "connection"), message["error"])
            elif kind == "frame":
                eye, pixels = message["eye"], message["pixels"]
                if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
                    self.errors[eye] = "UnsupportedFrameFormat"
                    return
                pixels.setflags(write=False)
                height, width = pixels.shape[:2]
                preview = pixels
                if width > self.settings.preview_width:
                    preview = cv2.resize(pixels, (self.settings.preview_width,
                                                max(1, round(height * self.settings.preview_width / width))))
                ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    self.errors[eye] = "PreviewEncodeFailed"
                    return
                previous = self.frames.get(eye)
                if previous:
                    elapsed = (message["observed_ns"] - previous.observed_ns) / 1e9
                    self.rates[eye] = 1 / elapsed if elapsed > 0 else 0
                else:
                    LOG.info("%s camera: %sx%s BGR; source timestamps unavailable", eye, width, height)
                self.frames[eye] = EyeFrame(
                    f"{self.session}:{eye}:{message['sequence']}", message["sequence"],
                    message["observed_ns"], pixels, encoded.tobytes(),
                )
                self.errors.pop(eye, None)
                self.state = "live"

    def _apply_zoom_result(self, message):
        """Fold a zoom command's read-back into the diagnostics, so later captures record the real level."""
        self._command_finished(message)
        for record in message.get("records", []):
            if not message.get("dry_run") and record.get("after"):
                self.diagnostics.setdefault(record["eye"], {})["zoom_level"] = record["after"]
        wait = self._zoom_waits.get(message["id"])
        if wait is not None:
            wait["message"] = message
            wait["event"].set()

    def request_zoom(self, level: str, timeout_s: float = 30.0) -> list[dict]:
        """Ask the SDK process to set both eyes to `level` and wait for the result records.

        Live or dry-run is `settings.zoom_commands`; the caller has already checked the confirm."""
        if self.commands is None or self.state != "live":
            raise RuntimeError("Camera service is not connected")
        request_id = uuid.uuid4().hex
        wait = {"event": threading.Event(), "message": None}
        with self.lock:
            self._zoom_waits[request_id] = wait
        try:
            try:
                self.commands.put_nowait({"id": request_id, "level": level})
            except queue.Full:
                raise CommandBusy("A zoom command is already pending") from None
            if not wait["event"].wait(timeout_s):
                raise TimeoutError("No zoom result from the camera service")
        finally:
            with self.lock:
                self._zoom_waits.pop(request_id, None)
        message = wait["message"]
        if message.get("error"):
            raise RuntimeError(f"Zoom command failed ({message['error']})")
        return message["records"]

    def _command_finished(self, message):
        finished = message.get("finished_ns")
        if finished is not None:
            self.command_done_ns = max(self.command_done_ns or 0, finished)

    def _apply_head_result(self, message):
        """Remember when the command ended and what the head was last live-commanded to."""
        self._command_finished(message)
        record = message.get("record") or {}
        if record.get("outcome") not in ("refused", "error"):
            self.last_head_command = {k: record.get(k) for k in ("action", "preset", "target_deg", "mode", "outcome")}
        if record.get("mode") == "live":
            if record.get("action") == "release":
                self.head_target = None
            elif record.get("action") in ("move", "stop") and record.get("outcome") != "refused":
                self.head_target = record
        if message["id"] != "stop":
            self.head_busy = False
        wait = self._head_waits.get(message["id"])
        if wait is not None:
            wait["message"] = message
            wait["event"].set()

    def head_motion_mode(self) -> str:
        from reachy_ai import config
        return "live" if self.settings.head_motion and config.REACHY_ENABLE_MOTION else "dry-run"

    def request_head(self, action: str, preset: str | None = None, approved: bool = False,
                     timeout_s: float = 60.0) -> dict:
        """Ask the SDK process for a head move ('move', with a preset) or a smooth 'release'; return its record.

        One at a time: refused with CommandBusy while another is pending or running. Live or dry-run is
        decided in the SDK process (primitives + gate); the caller has already checked the confirm."""
        if self.head_requests is None or self.state != "live":
            raise RuntimeError("Robot connection is not live")
        request_id = uuid.uuid4().hex
        wait = {"event": threading.Event(), "message": None}
        with self.lock:
            if self.head_busy:
                raise CommandBusy("A head move is already running; wait for it or press Stop")
            try:
                self.head_requests.put_nowait({"id": request_id, "action": action, "preset": preset,
                                               "approved": bool(approved)})
            except queue.Full:
                raise CommandBusy("A head command is already pending") from None
            self.head_busy = True
            self._head_waits[request_id] = wait
        try:
            if not wait["event"].wait(timeout_s):
                raise TimeoutError("No head result from the SDK process")
        finally:
            with self.lock:
                self._head_waits.pop(request_id, None)
        return wait["message"]["record"]

    def request_head_stop(self, timeout_s: float = 10.0) -> dict:
        """Stop: interrupt a running move and hold the present neck position. Needs no confirm."""
        if self.head_halt is None:
            raise RuntimeError("Robot connection is not live")
        with self.lock:
            wait = self._head_waits.setdefault("stop", {"event": threading.Event(), "message": None})
        self.head_halt.set()
        try:
            if not wait["event"].wait(timeout_s):
                raise TimeoutError("No stop result from the SDK process")
        finally:
            with self.lock:
                if self._head_waits.get("stop") is wait:
                    self._head_waits.pop("stop")
        return wait["message"]["record"]

    def capture_blocker(self, settle_s: float | None = None, tolerance_deg: float | None = None) -> str | None:
        """Why a procedure capture must wait, or None: frames (and the neck read-back) must be observed
        `settle_s` after the last zoom/head command ended, and the neck must read back within
        `tolerance_deg` of the last live head command."""
        from reachy_ai import config
        settle_s = config.CAPTURE_SETTLE_S if settle_s is None else settle_s
        tolerance_deg = config.NECK_READBACK_TOLERANCE_DEG if tolerance_deg is None else tolerance_deg
        with self.lock:
            if self.head_busy:
                return "A head move is running; capture after it has finished."
            ready_ns = None if self.command_done_ns is None else self.command_done_ns + int(settle_s * 1e9)
            if ready_ns is not None and any(f.observed_ns < ready_ns for f in self.frames.values()):
                return f"Frames predate the last command plus {settle_s} s of settling; try again in a moment."
            target = self.head_target
            if target is None:
                return None
            if target.get("action") == "stop":
                return "The head was stopped; move it to a preset before capturing."
            if target.get("outcome") != "reached":
                return f"The last head move did not complete ({target.get('outcome')}); move the head again."
            if self.head is None or (ready_ns is not None and self.head["observed_ns"] < ready_ns):
                return "No neck read-back since the last head move yet; try again in a moment."
            off = []
            for name, goal in target["target_deg"].items():
                present = (self.head["joints"].get(name) or {}).get("present_deg")
                if present is None or abs(present - goal) > tolerance_deg:
                    off.append(f"{name} {present} vs {goal}")
            if off:
                return (f"Neck read-back is more than {tolerance_deg} deg off the command ({'; '.join(off)}); "
                        "capture refused.")
            return None

    def status(self):
        with self.lock:
            eyes = {}
            now = time.monotonic_ns()
            for eye in EYES:
                frame = self.frames.get(eye)
                age = (now - frame.observed_ns) / 1e6 if frame else None
                eyes[eye] = {"state": ("stale" if age > self.settings.stale_seconds * 1000 else "live")
                             if frame else ("disconnected" if eye in self.errors else self.state),
                             "observed_age_ms": round(age, 1) if age is not None else None,
                             "sampled_fps": round(self.rates.get(eye, 0), 1),
                             "queue_drops": self.dropped.get(eye, 0),
                             "superseded": self.superseded.get(eye, 0), "error": self.errors.get(eye),
                             "frame": frame.metadata() if frame else None}
                if frame is None and eyes[eye]["state"] == "live":
                    eyes[eye]["state"] = "connecting"
            skew = None
            if all(e in self.frames for e in EYES):
                skew = abs(self.frames["left"].observed_ns - self.frames["right"].observed_ns) / 1e6
            can_capture = all(eyes[e]["state"] == "live" for e in EYES)
            can_capture = can_capture and skew is not None and skew <= self.settings.max_pair_skew_ms
            return {"schema_version": "1", "source": self.settings.source, "session_id": self.session,
                    "motion_controls": "head only", "head_motion": self.head_motion_mode(),
                    "head_busy": self.head_busy, "recording": False, "eyes": eyes,
                    "observation_skew_ms": round(skew, 1) if skew is not None else None,
                    "can_capture": can_capture, "diagnostics": dict(self.diagnostics),
                    "delivery": {"observation_queue": OBSERVATION_QUEUE, "policy": "newest kept",
                                 "head_queue_drops": self.dropped.get("head", 0),
                                 "head_superseded": self.superseded.get("head", 0)},
                    "connection_error": self.errors.get("connection"),
                    "timing": "Local SDK frame observation; capture timestamps and exposure sync unavailable",
                    "calibration_id": None, "simulator_backend": None,
                    "preview_fps_limit": self.settings.fps}

    def preview(self, eye):
        with self.lock:
            return self.frames.get(eye)

    def capture(self):
        with self.lock:
            if not self.status()["can_capture"]:
                raise ValueError("Both cameras must be fresh and within the observation-skew limit")
            pair_id = uuid.uuid4().hex
            record = {"pair_id": pair_id, "session_id": self.session, "source": self.settings.source,
                      "pairing": "latest observed frames; not synchronized exposure",
                      "observation_skew_ms": self.status()["observation_skew_ms"],
                      "camera_context": {"calibration_id": None,
                                         "settings_at_connection": copy.deepcopy(
                                             {e: self.diagnostics.get(e) for e in EYES})},
                      "frames": dict(self.frames)}
            record["head_command"] = copy.deepcopy(self.last_head_command)
            # Freeze observed readbacks with the pair; never use later live head angles.
            if self.head:
                age_ms = (time.monotonic_ns() - self.head["observed_ns"]) / 1e6
                skew_ms = max(abs(f.observed_ns - self.head["observed_ns"]) / 1e6 for f in self.frames.values())
                record["head_context"] = {**copy.deepcopy(self.head), "age_at_capture_ms": round(age_ms, 1),
                                          "frame_skew_ms": round(skew_ms, 1)}
            self.snapshots[pair_id] = record
            while len(self.snapshots) > 4:
                self.snapshots.popitem(last=False)
            return self.snapshot_metadata(record)

    @staticmethod
    def snapshot_metadata(record):
        return {**{k: v for k, v in record.items() if k not in ("frames", "analysis", "depth", "depth_image")},
                "eyes": {e: {**f.metadata(), "url": f"/api/stereo/pairs/{record['pair_id']}/{e}.jpg"}
                         for e, f in record["frames"].items()}}


def main():
    import uvicorn

    from reachy_ai.stereo_app import create_app

    parser = argparse.ArgumentParser(description="Reachy 1.2 read-only stereo viewer (workstation)")
    parser.add_argument("--source", choices=("sim", "robot"), default="sim")
    parser.add_argument("--robot-host", default=os.environ.get("REACHY_IP"))
    parser.add_argument("--sdk-port", type=int)
    parser.add_argument("--camera-port", type=int)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--preview-width", type=int, default=640)
    parser.add_argument("--detector-model", default=os.environ.get("REACHY_DETECTOR_MODEL"),
                        help="Existing tabletop TFLite detector; defaults to tests/best.tflite in this checkout")
    parser.add_argument("--sim-lens", choices=("distorted", "pinhole"), default="distorted",
                        help="Match the simulator's --distortion setting (default: distorted)")
    parser.add_argument("--allow-zoom-commands", action="store_true",
                        help="let a data-capture procedure send zoom commands to the cameras after the "
                             "operator's confirm in the page (default: dry-run, logged only)")
    parser.add_argument("--allow-head-motion", action="store_true",
                        help="let a data-capture procedure move the neck (head only) after the operator's confirm "
                             "in the page; also needs REACHY_ENABLE_MOTION=true (default: dry-run, logged only)")
    parser.add_argument("--stereo-calibration", default=os.environ.get("REACHY_STEREO_CALIBRATION"),
                        help="Physical source's calibration profile (JSON); "
                             "default calibration/reachy12_stereo_2026_08_27.json")
    args = parser.parse_args()
    if args.source == "robot" and not args.robot_host:
        parser.error("Robot mode requires --robot-host or REACHY_IP")
    settings = StereoSettings(
        source=args.source, robot_host=args.robot_host or "localhost",
        sdk_port=args.sdk_port or (50051 if args.source == "sim" else 50055),
        camera_port=args.camera_port or (50051 if args.source == "sim" else 50057),
        fps=args.fps, preview_width=args.preview_width, zoom_commands=args.allow_zoom_commands,
        head_motion=args.allow_head_motion,
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(settings, detector_model=args.detector_model, sim_lens=args.sim_lens,
                           stereo_calibration=args.stereo_calibration), host=args.bind, port=args.port,
                access_log=False, timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
