"""T3-Analysis routes for the stereo viewer (issue #52, epic #49): reference, analyze, confirm, history.

No arm, gripper or task route lives here. The only motion is the head, to board_view, through the viewer's gated neck
path (`camera.request_head`): a fresh operator confirm per move, then the SDK process's `head_gate_check`, the neck sign
check and the usual Stop. It is a logged dry run unless the viewer runs with --allow-head-motion and
REACHY_ENABLE_MOTION=true.

The viewer's only server-side write is the history log: one JSON line per confirmed analysis, appended to
<t3_history_dir>/<session_id>.jsonl (default outputs/t3_history, never committed). Everything else is held in memory.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, HTTPException
from fastapi.responses import Response

from reachy_ai.camera_zoom import calibration_id as zoomed_calibration_id
from reachy_ai.config import HEAD_PRESETS
from reachy_ai.games import t3_board_reader as reader
from reachy_ai.games import tictactoe as t3
from reachy_ai.pair_bundle import build_bundle, read_bundle, viewer_version
from reachy_ai.stereo import CommandBusy

POSE = "board_view"
CYCLE = ("X", "O", ".", "#")              # a click on a square: X, then O, then empty, then other
QUICK_TAGS = ("demo", "misread", "illegal-setup")
MAX_READS = 4                              # reads (with their pair bundle) kept in memory
MAX_TAGS, MAX_TAG_CHARS, MAX_NOTE_CHARS = 10, 60, 500
LABELS = {t3.X: "cube", t3.O: "cylinder", t3.EMPTY: "empty", t3.OTHER: "other"}
_SESSION_ID = re.compile(r"^[0-9a-f]{8,64}$")

# Square layout of the two views, rows top to bottom. Reachy's side is the camera's view: r1 (nearest Reachy) at the
# bottom, c1 (Reachy's left) at the left. The player sits opposite, so their view is that turned 180 degrees: r1 at
# the top (far from the player), c1 at the far right.
VIEWS = {
    "reachy": [[f"r{r}c{c}" for c in (1, 2, 3)] for r in (3, 2, 1)],
    "player": [[f"r{r}c{c}" for c in (3, 2, 1)] for r in (1, 2, 3)],
}
VIEW_LABELS = {"reachy": "Reachy's side (r1 nearest Reachy)", "player": "Player's side (turned 180 degrees)"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def head_off(joints: dict | None, tolerance_deg: float) -> str | None:
    """Why this neck read-back is not at board_view, or None."""
    target = dict(zip(("neck_roll", "neck_pitch", "neck_yaw"), HEAD_PRESETS[POSE]))
    if not joints:
        return "no neck read-back, so the head position is unknown"
    off = []
    for name, goal in target.items():
        present = (joints.get(name) or {}).get("present_deg")
        if present is None or abs(present - goal) > tolerance_deg:
            off.append(f"{name} {present} vs {goal}")
    return f"the head is not at {POSE} ({'; '.join(off)})" if off else None


def square_record(square: reader.SquareEvidence) -> dict:
    """One square's evidence as the page shows it and the history keeps it."""
    return {"cell": square.cell, "state": t3.symbol(square.state), "change_score": square.change_score,
            "visible_fraction": square.visible_fraction, "reason": square.reason, "shape": square.shape,
            "suggested": None if square.suggested is None else t3.symbol(square.suggested),
            "suggested_by": square.suggested_by,
            "detections": [{"label": d["label"], "confidence": d["confidence"], "box_xywh": d["box_xywh"]}
                           for d in square.detections]}


def corrections(read_board: tuple, confirmed: tuple) -> list:
    """[{square, read, confirmed}] for every square whose confirmed state differs from the read."""
    return [{"square": cell, "read": t3.symbol(a), "confirmed": t3.symbol(b)}
            for cell, a, b in zip(t3.CELLS, read_board, confirmed) if a != b]


def clean_tags(tags) -> list:
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise HTTPException(422, "tags must be a list of text")
    cleaned = [tag.strip()[:MAX_TAG_CHARS] for tag in tags if tag.strip()]
    if len(cleaned) > MAX_TAGS:
        raise HTTPException(422, f"at most {MAX_TAGS} tags")
    return cleaned


def history_line(read: dict, confirmed: tuple, result: t3.Result, tags: list, note: str) -> dict:
    return {
        "schema": "t3_history.v1", "time_utc": utc_now(), "session_id": read["session_id"],
        "reference_id": read["reference_id"], "pair_id": read["pair_id"], "bundle_sha256": read["bundle_sha256"],
        "board_read": read["board"], "squares": read["squares"],
        "corrections": corrections(read["board_tuple"], confirmed),
        "confirmed_board": t3.to_string(confirmed), "result": result.to_dict(),
        "tags": tags, "note": note, "reader_settings": read["settings"], "viewer_version": viewer_version(),
    }


def labels_document(read: dict, confirmed: tuple) -> dict:
    """labels.json of a labelled pair: the operator-confirmed state of every square (real-lab training labels)."""
    return {
        "schema": "t3_labels.v1", "source": "operator_confirmed", "pair_id": read["pair_id"],
        "reference_id": read["reference_id"], "confirmed_utc": read["confirmed_utc"], "pose": read["pose"],
        "board": t3.to_string(confirmed),
        "squares": [{"cell": cell, "label": LABELS[state], "state": t3.symbol(state),
                     "read": t3.symbol(was), "corrected": was != state}
                    for cell, state, was in zip(t3.CELLS, confirmed, read["board_tuple"])],
    }


def install_routes(app, *, camera, settings, zoom_context, depth, capture_record, analyze_pair, make_pair_bundle,
                   history_dir: str):
    """Mount /api/t3/* on the viewer app. The callables are the viewer's own (capture, analysis, bundle)."""
    lock = threading.Lock()
    app.state.t3 = {"reference": None, "reads": OrderedDict()}
    app.state.t3_settle_timeout_s = 8.0
    tolerance = reader.ReaderSettings().pose_tolerance_deg

    def live_identity() -> dict:
        profile = depth.status().get("profile", {})
        zoom = zoom_context(profile)
        captured = zoom.get("captured") if zoom else None
        cal = (zoomed_calibration_id(profile.get("id"), captured) if zoom and settings.source != "sim"
               else profile.get("id"))
        return {"calibration_id": cal, "zoom": captured, "source": settings.source}

    def stale_reasons(check_head: bool) -> list:
        held = app.state.t3["reference"]
        if held is None:
            return []
        ref, reasons = held["reference"], []
        if held["session_id"] != camera.session:
            reasons.append("the camera session reconnected since the reference was captured")
        live = live_identity()
        for key, ours in (("calibration_id", ref.calibration_id), ("zoom", ref.zoom), ("source", ref.source)):
            if live[key] != ours:
                reasons.append(f"{key} changed: {ours!r} when captured, {live[key]!r} now")
        if check_head:
            with camera.lock:
                joints = (camera.head or {}).get("joints")
            off = head_off(joints, tolerance)
            if off:
                reasons.append(off)
        return reasons

    def reference_summary():
        held = app.state.t3["reference"]
        if held is None:
            return None
        ref = held["reference"]
        reasons = stale_reasons(check_head=True)
        return {"id": held["id"], "built_utc": held["built_utc"], "pair_id": held["pair_id"],
                "session_id": held["session_id"], "pose": ref.pose, "calibration_id": ref.calibration_id,
                "zoom": ref.zoom, "source": ref.source, "squares_found": list(ref.quads),
                "stale": bool(reasons), "stale_reasons": reasons}

    def state():
        return {"head_motion": camera.head_motion_mode(), "pose": POSE, "reference": reference_summary(),
                "views": VIEWS, "view_labels": VIEW_LABELS, "cycle": list(CYCLE), "quick_tags": list(QUICK_TAGS),
                "history_dir": str(history_dir), "reader_settings": reader.ReaderSettings().__dict__}

    @app.get("/api/t3/state")
    def t3_state():
        return state()

    def move_head_and_capture(confirm: bool):
        """The gated head move to board_view, a settled capture, the detector, and the pair bundle."""
        if not confirm:
            raise HTTPException(409, "A head move needs the operator's explicit confirm.")
        try:
            record = camera.request_head("move", POSE, approved=True)
        except CommandBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(503, str(exc)) from exc
        if record.get("outcome") == "refused":
            failures = record.get("gate", {}).get("failures") or record.get("gate_failures") or ["refused"]
            raise HTTPException(409, "Head move refused: " + "; ".join(failures))
        deadline = time.monotonic() + app.state.t3_settle_timeout_s
        while True:
            try:
                meta, snapshot = capture_record(settled=True)
                break
            except HTTPException as exc:
                if exc.status_code != 409 or time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)
        pair_id = meta["pair_id"]
        off = head_off((snapshot.get("head_context") or {}).get("joints"), tolerance)
        if off:
            raise HTTPException(409, f"Not analysed: {off}. A dry-run head move does not move the head.")
        analyze_pair(pair_id)
        data = make_pair_bundle(pair_id)
        return meta, snapshot, data

    @app.post("/api/t3/reference")
    def t3_reference(confirm: bool = Body(False, embed=True)):
        """Capture the empty board at board_view and build the reference from it."""
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "Another T3 capture is running")
        try:
            meta, snapshot, data = move_head_and_capture(confirm)
            try:
                reference = reader.build_reference(data, POSE)
            except reader.BoardReadRefused as exc:
                raise HTTPException(422, f"Reference refused: {exc}") from exc
            app.state.t3["reference"] = {"id": uuid.uuid4().hex, "reference": reference, "built_utc": utc_now(),
                                         "pair_id": meta["pair_id"], "session_id": snapshot["session_id"]}
            return state()
        finally:
            lock.release()

    @app.post("/api/t3/analyze")
    def t3_analyze(confirm: bool = Body(False, embed=True)):
        """Capture at board_view, run the detector, and read the board against the reference."""
        held = app.state.t3["reference"]
        if held is None:
            raise HTTPException(409, "Capture the empty board first: there is no reference")
        reasons = stale_reasons(check_head=False)
        if reasons:
            raise HTTPException(409, "The reference is stale (" + "; ".join(reasons)
                                + "); capture the empty board again")
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "Another T3 capture is running")
        try:
            meta, snapshot, data = move_head_and_capture(confirm)
            try:
                result = reader.read_board(data, held["reference"])
            except reader.BoardReadRefused as exc:
                raise HTTPException(409, f"Not analysed: {exc}") from exc
            read_id = uuid.uuid4().hex
            squares = [square_record(s) for s in result.squares]
            reads = app.state.t3["reads"]
            reads[read_id] = {
                "id": read_id, "session_id": snapshot["session_id"], "pair_id": meta["pair_id"],
                "reference_id": held["id"], "bundle": data, "bundle_sha256": hashlib.sha256(data).hexdigest(),
                "board": result.board_string(), "board_tuple": result.board, "squares": squares,
                "settings": result.settings, "pose": result.pose, "confirmed": None, "confirmed_utc": None}
            while len(reads) > MAX_READS:
                reads.popitem(last=False)
            return {"read_id": read_id, "pair_id": meta["pair_id"], "bundle_sha256": reads[read_id]["bundle_sha256"],
                    "reference_id": held["id"], "board": result.board_string(), "squares": squares,
                    "settings": result.settings}
        finally:
            lock.release()

    def held_read(read_id):
        read = app.state.t3["reads"].get(read_id)
        if read is None:
            raise HTTPException(410, "That read is no longer held; analyze the board again")
        return read

    def parse_board(text):
        try:
            return t3.parse(text)
        except (ValueError, AttributeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/t3/accept-suggestions")
    def t3_accept(read_id: str = Body(embed=True), board: str = Body(embed=True)):
        """Fill every unknown square of `board` from its suggestion; squares without one stay unknown."""
        read, current = held_read(read_id), list(parse_board(board))
        filled = []
        for index, square in enumerate(read["squares"]):
            if current[index] == t3.UNKNOWN and square["suggested"]:
                current[index] = t3.state_of(square["suggested"])
                filled.append(square["cell"])
        return {"board": t3.to_string(tuple(current)), "filled": filled}

    @app.post("/api/t3/confirm")
    def t3_confirm(read_id: str = Body(embed=True), board: str = Body(embed=True), tags: list = Body([], embed=True),
                   note: str = Body("", embed=True)):
        """Confirm the (edited) board: refused while any square is unknown. Runs the engine and appends the history."""
        read, confirmed = held_read(read_id), parse_board(board)
        unknown = [cell for cell, state in zip(t3.CELLS, confirmed) if state == t3.UNKNOWN]
        if unknown:
            raise HTTPException(409, "Still unknown: " + ", ".join(unknown) + ". Set every square before confirming.")
        if not isinstance(note, str) or len(note) > MAX_NOTE_CHARS:
            raise HTTPException(422, f"note must be text of at most {MAX_NOTE_CHARS} characters")
        result = t3.recommend(confirmed)
        line = history_line(read, confirmed, result, clean_tags(tags), note.strip())
        if not _SESSION_ID.match(read["session_id"]):
            raise HTTPException(500, "Unexpected session id; history not written")
        path = Path(history_dir) / f"{read['session_id']}.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, sort_keys=True) + "\n")
        except OSError as exc:
            raise HTTPException(500, f"History not written ({exc.strerror}); confirmation not recorded") from exc
        read["confirmed"], read["confirmed_utc"] = confirmed, line["time_utc"]
        return {"result": result.to_dict(), "confirmed_board": t3.to_string(confirmed),
                "corrections": line["corrections"], "history_file": str(path), "history_time_utc": line["time_utc"]}

    @app.get("/api/t3/reads/{read_id}/labelled-pair.zip")
    def t3_labelled_pair(read_id: str):
        """The pair bundle plus labels.json of the confirmed squares, built in memory."""
        read = held_read(read_id)
        if read["confirmed"] is None:
            raise HTTPException(409, "Confirm the board before saving a labelled pair")
        manifest, files = read_bundle(read["bundle"])
        files = {**files, "labels.json": (json.dumps(labels_document(read, read["confirmed"]), indent=2,
                                                     sort_keys=True) + "\n").encode()}
        data = build_bundle(files, manifest.get("versions", {}), manifest.get("calibration_id"))
        return Response(data, media_type="application/zip", headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'attachment; filename="reachy-t3-labelled-{read["pair_id"][:8]}.zip"'})
