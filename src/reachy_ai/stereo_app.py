"""Small camera-only FastAPI application. No task or motion routes are mounted."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from reachy_ai.perception.pair_detector import (
    DEFAULT_MODEL,
    DetectorBusy,
    DetectorUnavailable,
    PairDetector,
    draw_detections,
)
from reachy_ai.perception.stereo_depth import DepthUnavailable, PairDepth
from reachy_ai.stereo import StereoService, StereoSettings


def create_app(settings: StereoSettings, service: StereoService | None = None, detector_model=None, detector=None,
               sim_lens="distorted", stereo_calibration=None, depth_analyzer=None):
    camera = service or StereoService(settings)
    analyzer = detector or PairDetector(detector_model or DEFAULT_MODEL)
    depth = depth_analyzer or PairDepth(settings.source, sim_lens, stereo_calibration)

    @asynccontextmanager
    async def lifespan(app):
        camera.start()
        try:
            yield
        finally:
            camera.stop()

    app = FastAPI(title="Reachy Stereo View", lifespan=lifespan)
    app.state.stereo = camera

    @app.get("/", response_class=HTMLResponse)
    def index():
        return files("reachy_ai").joinpath("web/stereo.html").read_text()

    @app.get("/api/stereo/status")
    def status():
        profile = depth.status()
        return {**camera.status(), "detector": analyzer.status(), "depth": profile,
                "calibration_id": profile.get("profile", {}).get("id")}

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

    @app.post("/api/stereo/pairs")
    def capture():
        try:
            with camera.lock:
                pair = camera.capture()
                record = camera.snapshots[pair["pair_id"]]
                profile = depth.status().get("profile", {})
                record["camera_context"]["calibration_id"] = profile.get("id")
                record["camera_context"]["analysis_profile"] = dict(profile)
                return camera.snapshot_metadata(record)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

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

    return app
