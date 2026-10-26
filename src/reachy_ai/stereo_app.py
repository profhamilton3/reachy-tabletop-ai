"""Small stereo-viewer FastAPI application: cameras and, for data-capture procedures, the neck and r_gripper only.

No task or arm route is mounted. The motion routes (data capture, T3-Analysis) move the head through
motion.primitives behind the head safety gate (owner decision 2026-10-20: "Neck motion is required
for vision data capture. Capture procedures may articulate the head through the neck for viewing and
perception. Arm motion stays forbidden in capture procedures."). They are dry-run unless
REACHY_ENABLE_MOTION=true and the viewer runs with --allow-head-motion, and each move needs a fresh
operator confirm. The one gripper route (the Gripper range, #27 Part A; owner decision 2026-10-22) powers
r_gripper alone behind the gripper gate: dry-run unless REACHY_ENABLE_MOTION=true and --allow-gripper-motion, a
fresh confirm per move, a Stop that releases it. The Sound check's routes call only the audio helper
(REACHY_AUDIO_URL, #59): no motion.
The Robot inventory's route (#62) only reads: the SDK through the SDK process, the helper's fixed NUC probes.
The Focus check's route (#77) sends raw lens commands (focus, zoom, homing, autofocus) through the SDK process after
the operator's confirm: dry-run unless --allow-zoom-commands; it moves no joint.
"""

import asyncio
import threading
import uuid
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Literal

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from reachy_ai.audio import speech
from reachy_ai.audio.audio_client import AudioClient
from reachy_ai.camera_zoom import EYES, calibration_id, focus_by_eye, lens_readings, normalize_zoom, zoom_status
from reachy_ai.config import HEAD_PRESETS
from reachy_ai.games.t3_viewer import install_routes as install_t3_routes
from reachy_ai.pair_bundle import BundleTooLarge, build_bundle, json_bytes, viewer_version
from reachy_ai.perception import focus as focus_mod
from reachy_ai.perception.pair_detector import (
    DEFAULT_MODEL,
    DetectorBusy,
    DetectorUnavailable,
    PairDetector,
    draw_detections,
)
from reachy_ai.perception.procedures import (
    PROCEDURES,
    CaptureRun,
    FocusNotConfirmed,
    FocusRefused,
    GripperNotConfirmed,
    GripperRefused,
    HeadNotConfirmed,
    HeadRefused,
    InventoryFailed,
    RecordingNotConsented,
    RunLimit,
    SoundFailed,
    ZoomNotConfirmed,
    encode_jpeg,
    get_procedure,
    slim_focus,
    slim_gripper,
)
from reachy_ai.perception.stereo_depth import DepthUnavailable, PairDepth
from reachy_ai.stereo import CommandBusy, StereoService, StereoSettings


class _ServiceZoom:
    """`ZoomSetter`-shaped front for the SDK process: dry-run unless the viewer allows zoom commands."""

    def __init__(self, service):
        self.service = service
        self.dry_run = not service.settings.zoom_commands

    def set_both(self, level):
        return self.service.request_zoom(level)

    def read_both(self):
        """The cameras' last read-back (connection time, or after a live command); no command is sent."""
        with self.service.lock:
            levels = {eye: (self.service.diagnostics.get(eye) or {}).get("zoom_level") for eye in EYES}
        return [{"eye": eye, "level": normalize_zoom(levels[eye])} for eye in EYES]


class _ServiceLens:
    """Lens facade for `CaptureRun.apply_focus`: raw lens commands and fresh-frame sharpness, through the SDK process.

    Commands are live or a logged dry run by `--allow-zoom-commands` (`dry_run`); reads are always sent. `measure`
    waits for frames observed after the last command ended (the SDK process waits the settle delay first), then
    averages the sharpness of three of them on the board region, which is fixed per step (`reset_regions`), not
    chosen per frame."""

    def __init__(self, service, quads_provider=None):
        self.service = service
        self.dry_run = not service.settings.zoom_commands
        self.quads_provider = quads_provider or (lambda eye: None)
        self._regions: dict = {}

    def cached(self):
        """The last read-back (connection time, or after a live command); nothing is sent."""
        with self.service.lock:
            return lens_readings({eye: self.service.diagnostics.get(eye) for eye in EYES})

    def read(self):
        records = self.service.request_lens("read")
        return {r["eye"]: {k: r.get(k) for k in ("zoom_level", "zoom", "focus") if k in r} for r in records}

    def reset_regions(self):
        self._regions = {}

    def _command(self, op, eye, value=None):
        return self.service.request_lens(op, eye, value, timeout_s=90.0)[0]

    def set_focus(self, eye, value):
        return self._command("set_focus", eye, value)

    def set_zoom(self, eye, value):
        return self._command("set_zoom", eye, value)

    def homing(self, eye):
        return self._command("homing", eye)

    def autofocus(self, eye):
        return self._command("autofocus", eye)

    def set_level(self, level):
        return self.service.request_zoom(level)

    def measure(self, eye):
        # The SDK process waits the settle delay (`--focus-settle`) after a command before it reads back and stamps
        # the result, so frames observed after that stamp show the settled lens.
        with self.service.lock:
            done = self.service.command_done_ns or 0
        frames = self.service.fresh_frames(eye, focus_mod.FRAMES_PER_MEASURE, done)
        if eye not in self._regions:
            gray = focus_mod.to_gray(frames[0])
            self._regions[eye] = focus_mod.board_region(gray, self.quads_provider(eye))
        mask, source = self._regions[eye]
        return {**focus_mod.measure_frames(frames, mask), "region": source}


class _ServiceHead:
    """Head mover for `CaptureRun.apply_head`: the SDK process runs the primitive, gate included."""

    def __init__(self, service):
        self.service = service

    def move(self, preset):
        return self.service.request_head("move", preset, approved=True)


class _ServiceGripper:
    """Gripper mover for `CaptureRun.apply_gripper`: the SDK process runs the primitive, gate included."""

    def __init__(self, service):
        self.service = service

    def read(self):
        return self.service.request_gripper("read")

    def move(self, target_deg, speed_limit, torque_limit, force_zero=None):
        return self.service.request_gripper("move", target_deg, speed_limit, torque_limit, approved=True,
                                            force_zero=force_zero)

    def release(self):
        return self.service.request_gripper("release", approved=True)


def create_app(settings: StereoSettings, service: StereoService | None = None, detector_model=None, detector=None,
               sim_lens="distorted", stereo_calibration=None, depth_analyzer=None, audio_client=None):
    camera = service or StereoService(settings)
    audio = audio_client or AudioClient()
    analyzer = detector or PairDetector(detector_model or DEFAULT_MODEL)
    depth = depth_analyzer or PairDepth(settings.source, sim_lens, stereo_calibration)

    @asynccontextmanager
    async def lifespan(app):
        camera.start()
        try:
            yield
        finally:
            camera.stop()

    def zoom_context(profile):
        """Zoom read back from the cameras against the profile's.

        The simulator has no calibration profile at a zoom level, so it reports only the read-back levels."""
        with camera.lock:
            readbacks = {eye: camera.diagnostics.get(eye) for eye in EYES}
        if settings.source == "sim":
            levels = {eye: normalize_zoom((readbacks[eye] or {}).get("zoom_level")) for eye in EYES}
            captured = levels["left"] if levels["left"] == levels["right"] else None
            return {"levels": levels, "captured": captured, "profile": None, "scene_policy": None,
                    "matches_profile": None, "warnings": [], "warning": None}
        return zoom_status(profile.get("zoom_level"), readbacks, profile.get("lens_state"))

    def t3_quads(eye):
        """The T3 reference's board squares in raw right-eye pixels (the Focus check's sharpness region), or None."""
        held = getattr(app.state, "t3", {}).get("reference")
        if eye != "right" or held is None:
            return None
        try:
            from reachy_ai.games.t3_board_reader import _geometry, to_raw
            ref = held["reference"]
            geometry = _geometry({"source": ref.source, "lens": ref.lens})
            return [to_raw(q, geometry) for q in ref.quads.values()]
        except Exception:  # noqa: BLE001 - no reference region: the grid finder or a centre crop is used
            return None

    app = FastAPI(title="Reachy Stereo View", lifespan=lifespan)
    app.state.stereo = camera

    @app.get("/", response_class=HTMLResponse)
    def index():
        return files("reachy_ai").joinpath("web/stereo.html").read_text()

    @app.get("/api/stereo/status")
    def status():
        profile = depth.status()
        return {**camera.status(), "detector": analyzer.status(), "depth": profile,
                "calibration_id": profile.get("profile", {}).get("id"),
                "zoom": zoom_context(profile.get("profile", {}))}

    @app.get("/stream/{eye}")
    async def stream(eye: Literal["left", "right"], request: Request):
        async def frames():
            last_id = None
            while not await request.is_disconnected():
                frame = camera.preview(eye)
                if frame and frame.frame_id != last_id:
                    last_id = frame.frame_id
                    yield (b"--reachyframe\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(frame.preview)).encode() + b"\r\n\r\n" + frame.preview + b"\r\n")
                await asyncio.sleep(1 / settings.fps)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=reachyframe",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    def capture_record(settled=False):
        """Freeze the latest pair with its zoom, calibration id and profile. Returns (metadata, record).

        `settled` (procedure captures): refuse frames from before the last zoom/head command ended plus
        the settle time, or a neck read-back off the last live head command."""
        try:
            with camera.lock:
                blocker = camera.capture_blocker() if settled else None
                if blocker:
                    raise ValueError(blocker)
                pair = camera.capture()
                record = camera.snapshots[pair["pair_id"]]
                profile = depth.status().get("profile", {})
                zoom = zoom_context(profile)
                record["camera_context"]["zoom"] = zoom
                record["camera_context"]["calibration_id"] = (
                    calibration_id(profile.get("id"), zoom["captured"],
                                   focus_by_eye({eye: camera.diagnostics.get(eye) for eye in EYES}))
                    if zoom and settings.source != "sim" else profile.get("id"))
                record["camera_context"]["analysis_profile"] = dict(profile)
                return camera.snapshot_metadata(record), record
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/stereo/pairs")
    def capture():
        return capture_record()[0]

    # --- Data capture: operator procedures (camera and neck only, never arms; the run lives in memory) ---
    run_lock = threading.Lock()
    app.state.run = None

    def current_run():
        if app.state.run is None:
            raise HTTPException(409, "Choose a data-capture procedure first")
        return app.state.run

    @app.get("/api/stereo/procedures")
    def procedures():
        return {"procedures": [p.describe() for p in PROCEDURES.values()],
                "zoom_commands": "live" if camera.settings.zoom_commands else "dry-run",
                "head_motion": camera.head_motion_mode(),
                "gripper_motion": camera.gripper_motion_mode(),
                "head_presets": {name: list(angles) for name, angles in HEAD_PRESETS.items()}}

    @app.get("/api/stereo/capture-run")
    def run_state():
        return current_run().state()

    @app.post("/api/stereo/capture-run")
    def run_start(procedure: str = Body(embed=True)):
        try:
            # the robot host is one more name the inventory's redaction pass removes
            run = CaptureRun(get_procedure(procedure),
                             redact_hosts=(settings.robot_host,) if settings.source == "robot" else ())
        except KeyError as exc:
            raise HTTPException(404, exc.args[0]) from exc
        with run_lock:
            app.state.run = run
        return run.state()

    @app.delete("/api/stereo/capture-run")
    def run_discard():
        with run_lock:
            app.state.run = None
        return {"discarded": True}

    @app.post("/api/stereo/capture-run/steps/{step_id}/zoom")
    def run_zoom(step_id: str, confirm: bool = Body(False, embed=True)):
        """Set this step's zoom, on the simulator as on the robot. Refused without the operator's confirm;
        dry-run unless --allow-zoom-commands. A read step (start, check) only reads the zoom back."""
        run = current_run()
        try:
            with run_lock:
                profile_zoom = depth.status().get("profile", {}).get("zoom_level")
                records = run.apply_zoom(step_id, _ServiceZoom(camera), confirm, profile_zoom,
                                         lens=_ServiceLens(camera, t3_quads))
        except (ZoomNotConfirmed, CommandBusy) as exc:
            raise HTTPException(409, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, "Unknown step") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(503, str(exc)) from exc
        read_only = run.procedure.step(step_id).zoom_role in ("start", "check")  # nothing was commanded
        return {"dry_run": not read_only and not camera.settings.zoom_commands, "records": records,
                "state": run.state()}

    @app.post("/api/stereo/capture-run/steps/{step_id}/focus")
    def run_focus(step_id: str, confirm: bool = Body(False, embed=True), choice: str | None = Body(None, embed=True)):
        """Focus check (#77): this step's lens operation. A read or a sharpness measurement sends no command and needs
        no confirm; homing, a sweep, autofocus and the finish step are refused without the operator's confirm and are
        dry-run unless --allow-zoom-commands. A sweep is long (about two minutes): one confirm for its bounded
        commands. `choice` (finish only): keep, restore or sweep."""
        run = current_run()
        try:
            with run_lock:
                records = run.apply_focus(step_id, _ServiceLens(camera, t3_quads), confirm, choice)
        except (FocusNotConfirmed, FocusRefused, CommandBusy) as exc:
            raise HTTPException(409, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, "Unknown step") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(503, str(exc)) from exc
        send = run.procedure.step(step_id).focus not in ("read", "measure")
        return {"dry_run": send and not camera.settings.zoom_commands, "records": [slim_focus(r) for r in records],
                "state": run.state()}

    @app.post("/api/stereo/capture-run/steps/{step_id}/head")
    def run_head(step_id: str, confirm: bool = Body(False, embed=True)):
        """Point the head at this step's preset. Refused without the operator's confirm, while another head
        command runs, or by the gate; dry-run unless REACHY_ENABLE_MOTION=true and --allow-head-motion."""
        run = current_run()
        try:
            with run_lock:
                record = run.apply_head(step_id, _ServiceHead(camera), confirm)
        except (HeadNotConfirmed, HeadRefused, CommandBusy) as exc:
            raise HTTPException(409, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, "Unknown step") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"dry_run": record.get("mode") != "live", "record": record, "state": run.state()}

    @app.post("/api/stereo/capture-run/steps/{step_id}/gripper")
    def run_gripper(step_id: str, confirm: bool = Body(False, embed=True)):
        """Gripper range: this step's gripper operation. Stage 0 only reads and the end step only releases; a move
        is refused without the operator's confirm, before Stage 0, after an abort, while another command runs, or
        by the gate; dry-run unless REACHY_ENABLE_MOTION=true and --allow-gripper-motion."""
        run = current_run()
        try:
            with run_lock:
                record = run.apply_gripper(step_id, _ServiceGripper(camera), confirm)
        except (GripperNotConfirmed, GripperRefused, CommandBusy) as exc:
            raise HTTPException(409, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, "Unknown step") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"dry_run": record.get("mode") == "dry-run", "record": slim_gripper(record), "state": run.state()}

    @app.post("/api/stereo/gripper/stop")
    def gripper_stop():
        """Stop gripper: interrupt a running move and release (compliant). No confirm needed."""
        try:
            record = camera.request_gripper_stop()
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(503, str(exc)) from exc
        with run_lock:
            run = app.state.run
            if run is not None:
                run.record_gripper_event(record)
            state = None if run is None else run.state()
        return {"dry_run": record.get("mode") != "live", "record": record, "state": state}

    def record_head_event(record):
        with run_lock:
            run = app.state.run
            if run is not None:
                run.record_head_event(record)
                return run.state()
        return None

    @app.post("/api/stereo/head/stop")
    def head_stop():
        """Stop: interrupt a running head move and hold the present position. No confirm needed."""
        try:
            record = camera.request_head_stop()
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"dry_run": record.get("mode") != "live", "record": record, "state": record_head_event(record)}

    @app.post("/api/stereo/head/release")
    def head_release(confirm: bool = Body(False, embed=True)):
        """Turn the head off smoothly (compliant). Refused without the operator's confirm."""
        if not confirm:
            record_head_event({"action": "release", "outcome": "refused",
                               "gate_failures": ["Operator approval not given"]})
            raise HTTPException(409, "Releasing the head needs the operator's explicit confirm.")
        try:
            record = camera.request_head("release", approved=True)
        except CommandBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(503, str(exc)) from exc
        state = record_head_event(record)
        if record.get("outcome") == "refused":
            raise HTTPException(409, "Head release refused: " + "; ".join(record.get("gate_failures", [])))
        return {"dry_run": record.get("mode") != "live", "record": record, "state": state}

    @app.post("/api/stereo/capture-run/steps/{step_id}/capture")
    def run_capture(step_id: str):
        run = current_run()
        try:
            step = run.procedure.step(step_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown step") from exc
        if not step.capture:
            raise HTTPException(409, "This step has nothing to capture")
        blocker = run.capture_blocker(step_id)
        if blocker:
            raise HTTPException(409, blocker)
        with run_lock:
            meta, record = capture_record(settled=True)
            try:
                run.add_capture(step_id, meta, encode_jpeg(record["frames"]["left"].pixels),
                                encode_jpeg(record["frames"]["right"].pixels))
            except RunLimit as exc:
                raise HTTPException(409, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(500, str(exc)) from exc
            return run.state()

    @app.post("/api/stereo/capture-run/steps/{step_id}/sound")
    def run_sound(step_id: str, confirm: bool = Body(False, embed=True)):
        """Sound check: this step's audio-helper operation (status, tone, phrase or a recording). A recording is
        refused without the operator's tick that the speaker agreed. The WAV stays in the run's memory."""
        run = current_run()
        try:
            with run_lock:
                record = run.apply_sound(step_id, audio, confirm)
        except (RecordingNotConsented, RunLimit) as exc:
            raise HTTPException(409, str(exc)) from exc
        except SoundFailed as exc:
            raise HTTPException(409 if exc.code == "busy" else 503, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, "Unknown step") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"record": record, "state": run.state()}

    @app.post("/api/stereo/capture-run/steps/{step_id}/answer")
    def run_answer(step_id: str, heard: bool = Body(embed=True), note: str = Body("", embed=True)):
        """Sound check: the operator's 'heard it' / 'not heard' and volume note for a tone or phrase step."""
        run = current_run()
        try:
            with run_lock:
                run.record_answer(step_id, heard, note)
        except KeyError as exc:
            raise HTTPException(404, "Unknown step") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return run.state()

    @app.post("/api/stereo/capture-run/steps/{step_id}/inventory")
    def run_inventory(step_id: str, arm_resting: bool | None = Body(None, embed=True),
                      note: str = Body("", embed=True)):
        """Robot inventory: this step's SDK reads (in the SDK process), NUC probes (the audio helper) or operator
        answers. Reads only: nothing moves and nothing is written, here or on the robot."""
        run = current_run()
        try:
            with run_lock:
                record = run.apply_inventory(step_id, reader=camera.request_inventory, client=audio,
                                             arm_resting=arm_resting, note=note)
        except InventoryFailed as exc:
            raise HTTPException(409 if isinstance(exc.__cause__, CommandBusy) else 503, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, "Unknown step") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"record": record, "state": run.state()}

    @app.get("/api/stereo/capture-run/bundle.zip")
    def run_bundle():
        """Every pair (or audio clip) of the run, step labels, zoom per step and a SHA-256 manifest; in memory."""
        run = current_run()
        with camera.lock:
            versions = {"viewer": viewer_version(), "sdk": camera.diagnostics.get("sdk_version"),
                        "neck_sign_check": camera.diagnostics.get("neck_sign_check")}
        try:
            data = run.build_zip(versions)
        except RunLimit as exc:
            raise HTTPException(409, str(exc)) from exc
        except BundleTooLarge as exc:
            raise HTTPException(413, str(exc)) from exc
        return Response(data, media_type="application/zip", headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'attachment; filename="reachy-capture-{run.procedure.id}.zip"'})

    @app.get("/api/stereo/pairs/{pair_id}")
    def pair_metadata(pair_id: str):
        with camera.lock:
            record = camera.snapshots.get(pair_id)
            if record is None:
                raise HTTPException(410, "Pair unavailable or expired; capture a new pair")
            return camera.snapshot_metadata(record)

    @app.get("/api/stereo/pairs/{pair_id}/{eye}.jpg")
    def pair_image(pair_id: str, eye: Literal["left", "right"]):
        with camera.lock:
            record = camera.snapshots.get(pair_id)
            if record is None:
                raise HTTPException(410, "Pair unavailable or expired; capture a new pair")
            frame = record["frames"][eye]
        ok, jpeg = cv2.imencode(".jpg", frame.pixels, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise HTTPException(500, "Unable to encode snapshot")
        return Response(jpeg.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    def make_pair_bundle(pair_id):
        """Everything held for this pair, built on demand in memory (never written to the server's disk)."""
        with camera.lock:
            record = camera.snapshots.get(pair_id)
            if record is None:
                raise HTTPException(410, "Pair unavailable or expired; capture a new pair")
            metadata = camera.snapshot_metadata(record)
            frames = {e: record["frames"][e].pixels for e in ("left", "right")}
            analysis, depth_result = record.get("analysis"), record.get("depth")
            depth_png = record.get("depth_image")
            sdk_version = camera.diagnostics.get("sdk_version")
            neck_sign_check = camera.diagnostics.get("neck_sign_check")
            calibration_id = record["camera_context"].get("calibration_id")
        files = {"pair.json": json_bytes(metadata)}
        for eye, pixels in frames.items():
            ok, jpeg = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise HTTPException(500, "Unable to encode snapshot")
            files[f"{eye}.jpg"] = jpeg.tobytes()
        if analysis is not None:
            ok, jpeg = cv2.imencode(".jpg", draw_detections(frames["right"], analysis["detections"]),
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise HTTPException(500, "Unable to encode analysis image")
            files["analysis.json"] = json_bytes(analysis)
            files["analysis_overlay.jpg"] = jpeg.tobytes()
        if depth_result is not None and depth_png is not None:
            files["depth.json"] = json_bytes(depth_result)
            files["depth.png"] = depth_png
        versions = {"viewer": viewer_version(), "sdk": sdk_version, "neck_sign_check": neck_sign_check,
                    "detector_model": (analysis or {}).get("model")}
        try:
            return build_bundle(files, versions, calibration_id)
        except BundleTooLarge as exc:
            raise HTTPException(413, str(exc)) from exc

    @app.get("/api/stereo/pairs/{pair_id}/bundle.zip")
    def pair_bundle(pair_id: str):
        data = make_pair_bundle(pair_id)
        return Response(data, media_type="application/zip", headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'attachment; filename="reachy-pair-{pair_id[:8]}.zip"'})

    @app.post("/api/stereo/pairs/{pair_id}/analysis")
    def analyze_pair(pair_id: str):
        with camera.lock:
            record = camera.snapshots.get(pair_id)
            if record is None:
                raise HTTPException(410, "Pair unavailable or expired; capture a new pair")
            if "analysis" in record:
                return record["analysis"]
            frame = record["frames"]["right"]
        try:
            result = analyzer.analyze(frame.pixels)
        except DetectorBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        except DetectorUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        with camera.lock:
            if camera.snapshots.get(pair_id) is not record:
                raise HTTPException(410, "Pair expired during analysis; capture a new pair")
            result.update({"analysis_id": uuid.uuid4().hex, "pair_id": pair_id,
                           "session_id": record["session_id"], "source": record["source"],
                           "frame_id": frame.frame_id, "eye": "right",
                           "camera_context": record["camera_context"],
                           "raw_url": f"/api/stereo/pairs/{pair_id}/right.jpg",
                           "overlay_url": f"/api/stereo/pairs/{pair_id}/analysis/image.jpg"})
            record["analysis"] = result
            return result

    @app.get("/api/stereo/pairs/{pair_id}/analysis/image.jpg")
    def analysis_image(pair_id: str):
        with camera.lock:
            record = camera.snapshots.get(pair_id)
            if record is None:
                raise HTTPException(410, "Pair unavailable or expired; capture a new pair")
            result = record.get("analysis")
            if result is None:
                raise HTTPException(404, "Analyze this pair first")
            pixels = record["frames"]["right"].pixels
        annotated = draw_detections(pixels, result["detections"])
        ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise HTTPException(500, "Unable to encode analysis image")
        return Response(jpeg.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/stereo/pairs/{pair_id}/depth")
    def analyze_depth(pair_id: str):
        with camera.lock:
            record = camera.snapshots.get(pair_id)
            if record is None:
                raise HTTPException(410, "Pair unavailable or expired; capture a new pair")
            if "depth" in record:
                return record["depth"]
            analysis = record.get("analysis")
            if analysis is None:
                raise HTTPException(409, "Analyze this captured pair first")
            frames = record["frames"]
            context = record.get("head_context", {})
            head = context.get("joints")
            if (context.get("age_at_capture_ms", 9999) > 250
                or context.get("frame_skew_ms", 9999) > 250 or not head or not all(
                np.isfinite(head.get(n, {}).get("present_deg", float("nan")))
                for n in ("neck_roll", "neck_pitch", "neck_yaw")
            )):
                head = None
        try:
            result, image = depth.analyze(frames["left"].pixels, frames["right"].pixels,
                                          analysis["detections"], head)
        except DetectorBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        except DepthUnavailable as exc:
            raise HTTPException(422, str(exc)) from exc
        with camera.lock:
            if camera.snapshots.get(pair_id) is not record:
                raise HTTPException(410, "Pair expired during 3D analysis; capture a new pair")
            result.update({"pair_id": pair_id, "session_id": record["session_id"],
                           "source": record["source"], "analysis_id": analysis["analysis_id"],
                           "frame_ids": {e: f.frame_id for e, f in frames.items()},
                           "observation_skew_ms": record["observation_skew_ms"],
                           "head_context": context,
                           "image_url": f"/api/stereo/pairs/{pair_id}/depth/image.png"})
            record["depth"], record["depth_image"] = result, image
            return result

    @app.get("/api/stereo/pairs/{pair_id}/depth/image.png")
    def depth_image(pair_id: str):
        with camera.lock:
            record = camera.snapshots.get(pair_id)
            if record is None:
                raise HTTPException(410, "Pair unavailable or expired; capture a new pair")
            if "depth_image" not in record:
                raise HTTPException(404, "Run 3D analysis first")
            return Response(record["depth_image"], media_type="image/png", headers={"Cache-Control": "no-store"})

    install_t3_routes(app, camera=camera, settings=settings, zoom_context=zoom_context, depth=depth,
                      capture_record=capture_record, analyze_pair=analyze_pair, make_pair_bundle=make_pair_bundle,
                      history_dir=settings.t3_history_dir, speak=lambda text: speech.speak(text, client=audio))
    return app
