"""Small camera-only FastAPI application. No task or motion routes are mounted."""

import asyncio
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Literal

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from reachy_ai.stereo import StereoService, StereoSettings


def create_app(settings: StereoSettings, service: StereoService | None = None):
    camera = service or StereoService(settings)

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
        return camera.status()

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

    return app
