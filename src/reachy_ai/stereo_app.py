"""Small camera-only FastAPI application. No task or motion routes are mounted."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Literal

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from reachy_ai.perception.pair_detector import (
    DEFAULT_MODEL,
    DetectorBusy,
    DetectorUnavailable,
    PairDetector,
    draw_detections,
)
from reachy_ai.stereo import StereoService, StereoSettings


def create_app(settings: StereoSettings, service: StereoService | None = None, detector_model=None, detector=None):
    camera = service or StereoService(settings)
    analyzer = detector or PairDetector(detector_model or DEFAULT_MODEL)

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
        return {**camera.status(), "detector": analyzer.status()}

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
            return camera.capture()
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

    return app
