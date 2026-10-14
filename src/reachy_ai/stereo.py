"""Read-only Reachy v1 stereo acquisition, shared by all browser consumers.

The SDK owns its camera streaming threads. A single child process contains that
connection so its blocking startup/first-frame calls can be stopped on timeout.
No joint, compliance, zoom, focus, or robot-service commands are issued here.
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

    def __post_init__(self):
        if self.source not in ("sim", "robot") or not self.robot_host:
            raise ValueError("Choose sim or robot and provide a robot host")
        if not 1 <= self.fps <= 30 or not 160 <= self.preview_width <= 1920:
            raise ValueError("Preview FPS must be 1–30 and width 160–1920")
        if not 0 < self.stale_seconds < self.restart_seconds or self.max_pair_skew_ms <= 0:
            raise ValueError("Require positive skew and 0 < stale interval < restart interval")


def _sdk_worker(settings: StereoSettings, messages, stop):
    """Reuse the notebooks' ReachySDK + last_frame path, once per application."""
    # Ctrl+C belongs to the web-server parent, which disposes this worker.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    def publish(message):
        try:
            messages.put_nowait(message)
        except queue.Full:
            pass  # A slow consumer must never block SDK acquisition.

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
                    publish({"kind": "frame", "eye": eye, "sequence": sequence,
                             "observed_ns": time.monotonic_ns(), "pixels": frame.copy()})
            except Exception as exc:
                publish({"kind": "error", "eye": eye, "error": type(exc).__name__})
                return
            stop.wait(1 / settings.fps)

    def diagnostics():
        # These are optional reads, off the frame path. A stuck read cannot
        # prevent viewing; the containing process is disposable on shutdown.
        for eye in EYES:
            values = {}
            for name in ("zoom_level", "zoom", "focus"):
                try:
                    value = getattr(getattr(robot, f"{eye}_camera"), name)
                    values[name] = getattr(value, "name", value)
                except Exception as exc:
                    values[name] = None
                    values[f"{name}_error"] = type(exc).__name__
                publish({"kind": "diagnostics", "section": eye, "values": dict(values)})
        head = {}
        for name in ("neck_roll", "neck_pitch", "neck_yaw"):
            try:
                joint = getattr(robot.head.joints, name)
                head[name] = {"present_deg": float(joint.present_position),
                              "goal_deg": float(joint.goal_position), "compliant": bool(joint.compliant)}
            except Exception as exc:
                head[name] = {"error": type(exc).__name__}
        publish({"kind": "diagnostics", "section": "head_at_connection", "values": head})

    for eye in EYES:
        threading.Thread(target=read_eye, args=(eye,), daemon=True).start()
    threading.Thread(target=diagnostics, daemon=True).start()
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
        self.dropped = {eye: 0 for eye in EYES}
        self.state = "connecting"

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
            self.dropped = {eye: 0 for eye in EYES}

    def _run(self):
        context = mp.get_context("spawn")
        while not self.stop_event.is_set():
            self._new_session()
            messages = context.Queue(maxsize=4)
            worker_stop = context.Event()
            worker = context.Process(target=_sdk_worker, args=(self.settings, messages, worker_stop), daemon=True)
            worker.start()
            started = time.monotonic()
            try:
                while not self.stop_event.is_set():
                    try:
                        self._accept(messages.get(timeout=0.2))
                    except queue.Empty:
                        pass
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
                messages.close()
                with self.lock:
                    self.state = "reconnecting"
                    self.frames.clear()
            self.stop_event.wait(2)

    def _accept(self, message):
        kind = message["kind"]
        with self.lock:
            if kind == "ready":
                self.diagnostics["sdk_version"] = message["sdk_version"]
                LOG.info("SDK connected; source=%s version=%s", self.settings.source, message["sdk_version"])
            elif kind == "diagnostics":
                self.diagnostics[message["section"]] = message["values"]
                LOG.info("SDK readback %s: %s", message["section"], message["values"])
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
                    self.dropped[eye] += max(0, message["sequence"] - previous.sequence - 1)
                else:
                    LOG.info("%s camera: %sx%s BGR; source timestamps unavailable", eye, width, height)
                self.frames[eye] = EyeFrame(
                    f"{self.session}:{eye}:{message['sequence']}", message["sequence"],
                    message["observed_ns"], pixels, encoded.tobytes(),
                )
                self.errors.pop(eye, None)
                self.state = "live"

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
                             "queue_drops": self.dropped[eye], "error": self.errors.get(eye),
                             "frame": frame.metadata() if frame else None}
                if frame is None and eyes[eye]["state"] == "live":
                    eyes[eye]["state"] = "connecting"
            skew = None
            if all(e in self.frames for e in EYES):
                skew = abs(self.frames["left"].observed_ns - self.frames["right"].observed_ns) / 1e6
            can_capture = all(eyes[e]["state"] == "live" for e in EYES)
            can_capture = can_capture and skew is not None and skew <= self.settings.max_pair_skew_ms
            return {"schema_version": "1", "source": self.settings.source, "session_id": self.session,
                    "motion_controls": False, "recording": False, "eyes": eyes,
                    "observation_skew_ms": round(skew, 1) if skew is not None else None,
                    "can_capture": can_capture, "diagnostics": dict(self.diagnostics),
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
            self.snapshots[pair_id] = record
            while len(self.snapshots) > 4:
                self.snapshots.popitem(last=False)
            return self.snapshot_metadata(record)

    @staticmethod
    def snapshot_metadata(record):
        return {**{k: v for k, v in record.items() if k not in ("frames", "analysis")},
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
    args = parser.parse_args()
    if args.source == "robot" and not args.robot_host:
        parser.error("Robot mode requires --robot-host or REACHY_IP")
    settings = StereoSettings(
        source=args.source, robot_host=args.robot_host or "localhost",
        sdk_port=args.sdk_port or (50051 if args.source == "sim" else 50055),
        camera_port=args.camera_port or (50051 if args.source == "sim" else 50057),
        fps=args.fps, preview_width=args.preview_width,
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(settings, detector_model=args.detector_model), host=args.bind, port=args.port,
                access_log=False, timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
