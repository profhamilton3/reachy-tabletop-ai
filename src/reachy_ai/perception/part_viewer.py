"""The viewer's part-presentation routes (#84): the operator's ground truth, the empty-board reference and the
analysis. They work on the current "Part presentation" capture run. Nothing here moves anything or sends a command,
and nothing is written on the server: the reference and the last analysis are held in memory.
"""
from __future__ import annotations

import contextlib
import threading

from fastapi import Body, HTTPException
from fastapi.responses import Response

from reachy_ai.pair_bundle import BundleTooLarge
from reachy_ai.perception import part_perception as pp
from reachy_ai.perception.procedures import (
    PART_DEFAULT_SPOT,
    PART_DESIGN_CHOICES,
    PART_EMPTY,
    PART_LABELS,
    PART_ORIENTATIONS,
    PART_SPOTS,
    GroundTruthRequired,
    RunLimit,
)

ANALYSIS_NOTE = ("Measured from the pictures only: the ground truth you recorded is never read by the analysers. "
                 "The bounds come from the simulator and are provisional.")


def install_part_routes(app, *, current_run, run_lock, versions, profile_zoom):
    """Mount the routes. `current_run()` gives the capture run (or raises 409), `versions()` the bundle's version
    block, and `profile_zoom()` the zoom level the lens calibration is valid at. `app.state.part_latest()` gives the
    decision panel the current run's analysis: (PartResult, bundle name, bundle bytes), or None."""
    held = {"reference": None, "analysis": None}
    busy = threading.Lock()

    def latest():
        """The analysis, only while it is of the CURRENT run as it stands. Once another run is started, the run is
        discarded or another pair is captured, it is not served: an older look must not pass for the present one."""
        if held["analysis"] is None:
            return None
        result, name, data, analysed_run, pairs = held["analysis"]
        try:
            run = current_run()
        except HTTPException:
            return None
        return (result, name, data) if run is analysed_run and run.pair_count() == pairs else None

    app.state.part_latest = latest

    def view() -> dict:
        reference, analysis = held["reference"], latest()
        return {
            "designs": list(PART_DESIGN_CHOICES), "orientations": list(PART_ORIENTATIONS),
            "labels": list(PART_LABELS), "spots": list(PART_SPOTS), "default_spot": PART_DEFAULT_SPOT,
            "reference": None if reference is None else {"sha256": reference.sha256, "name": reference.name},
            "analysis": None if analysis is None else {**analysis[0].to_dict(), "note": ANALYSIS_NOTE}}

    def part_run():
        run = current_run()
        if run.procedure.kind != "part":
            raise HTTPException(409, "Choose the Part presentation procedure first")
        return run

    def built_run(name: str) -> tuple:
        """(Run, bytes) of the current part run's bundle as it stands."""
        run = part_run()
        with run_lock:
            try:
                data = run.build_zip(versions())
            except RunLimit as exc:
                raise HTTPException(409, str(exc)) from exc
            except BundleTooLarge as exc:
                raise HTTPException(413, str(exc)) from exc
        try:
            return pp.load_run(data, name), data
        except pp.PartRefused as exc:
            raise HTTPException(409, str(exc)) from exc

    @contextlib.contextmanager
    def one_at_a_time():
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "A part analysis is already running")
        try:
            yield
        finally:
            busy.release()

    @app.get("/api/stereo/part")
    def part_state():
        return view()

    @app.post("/api/stereo/capture-run/ground-truth")
    def ground_truth(design_id: str = Body(embed=True), orientation: str | None = Body(None, embed=True),
                     top_face_visible: bool | None = Body(None, embed=True),
                     observation: str | None = Body(None, embed=True),
                     spot: str = Body(PART_DEFAULT_SPOT, embed=True), instance: str = Body("", embed=True),
                     note: str = Body("", embed=True)):
        """What the operator put on the board, before the first capture. For grading only."""
        run = part_run()
        try:
            with run_lock:
                run.set_ground_truth(design_id=design_id, orientation=orientation,
                                     top_face_visible=top_face_visible, observation=observation, spot=spot,
                                     instance=instance, note=note)
        except GroundTruthRequired as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return run.state()

    @app.post("/api/stereo/part/reference")
    def keep_reference():
        """Keep the current run, captured with the board EMPTY, as the reference part runs are compared with. The
        board must be found in each eye at each zoom."""
        with one_at_a_time():
            run = part_run()
            if run.ground_truth is None or run.ground_truth["design_id"] != PART_EMPTY:
                raise HTTPException(409, "Only a run recorded as the empty board can be the reference")
            loaded, _ = built_run("part-reference.zip")
            problems = pp.reference_problems(loaded, profile_zoom())
            if problems:
                raise HTTPException(409, "Not usable as the reference: " + "; ".join(problems))
            held["reference"], held["analysis"] = loaded, None
        return view()

    @app.delete("/api/stereo/part/reference")
    def drop_reference():
        held["reference"], held["analysis"] = None, None
        return view()

    @app.post("/api/stereo/part/analysis")
    def analyse_part():
        """Analyse the current part run against the reference. Takes a few seconds; reads frames already held."""
        with one_at_a_time():
            run = part_run()
            if held["reference"] is None:
                raise HTTPException(409, "No empty-board reference yet: run the procedure with the board empty and "
                                         "keep it as the reference")
            if run.ground_truth is not None and run.ground_truth["design_id"] == PART_EMPTY:
                raise HTTPException(409, "This run is recorded as the empty board: there is no part to analyse")
            held["analysis"] = None
            pairs = run.pair_count()
            loaded, data = built_run("part.zip")
            name = f"reachy-capture-part_presentation-{loaded.sha256[:8]}.zip"
            try:
                result = pp.analyse(loaded, held["reference"], bundle_path=name, profile_zoom=profile_zoom())
            except pp.PartRefused as exc:
                raise HTTPException(409, str(exc)) from exc
            held["analysis"] = (result, name, data, run, pairs)
        return view()

    @app.get("/api/stereo/part/analysis/bundle.zip")
    def analysed_bundle():
        """Exactly the bytes the observation's SHA-256 names."""
        analysis = latest()
        if analysis is None:
            raise HTTPException(404, "The current run has not been analysed")
        _, name, data = analysis
        return Response(data, media_type="application/zip", headers={
            "Cache-Control": "no-store", "Content-Disposition": f'attachment; filename="{name}"'})
