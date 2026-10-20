"""Operator data-capture procedures (#28): step definitions, the capture run, and its bundle.

Camera and neck only, never arms. A procedure is a list of numbered operator steps; the operator
captures a stereo pair per step. A procedure can ask for two robot commands, each only after an
explicit operator confirm: a camera zoom level (`CaptureRun.apply_zoom`, through
`camera_zoom.ZoomSetter`) and a named head pose (`CaptureRun.apply_head`, through
`motion.primitives` and the head safety gate). Owner decision 2026-10-20: "Neck motion is required
for vision data capture. Capture procedures may articulate the head through the neck for viewing and
perception. Arm motion stays forbidden in capture procedures." No arm or gripper command exists
here. Nothing is written to disk by this module: the run lives in memory and `CaptureRun.build_zip`
returns the bundle bytes.

The viewer (stereo_app.py) and the CLI fallbacks (scripts/zoom_fov_check.py,
scripts/ruler_depth_test.py) both drive these same objects; the offline analysers
(scripts/analyze_capture_bundle.py) read the bundle back.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from reachy_ai.camera_zoom import SCENE_ZOOM, ZOOM_ORDER
from reachy_ai.config import HEAD_PRESETS, NECK_JOINTS
from reachy_ai.pair_bundle import build_bundle, json_bytes

LOG = logging.getLogger(__name__)

MAX_CAPTURES_PER_STEP = 5
MAX_PAIRS_PER_RUN = 24
DISTANCES_M = (0.3, 0.5, 0.8, 1.2)
TOLERANCE_M = 0.015
#: Pinhole constants behind the analytic depth sensitivity of docs/PERCEPTION_STAGE1.md:
#: dZ = Z^2 / (f * B) * dd. Approximate (f ~ 400 px, B = 78 mm); the analyser can override them.
SENSITIVITY_FOCAL_PX = 400.0
SENSITIVITY_BASELINE_M = 0.078


class ZoomNotConfirmed(PermissionError):
    """A zoom command was requested without the operator's explicit confirm."""


class HeadNotConfirmed(PermissionError):
    """A head move was requested without the operator's explicit confirm."""


class HeadRefused(PermissionError):
    """The head safety gate refused the move; nothing was sent."""


class RunLimit(ValueError):
    pass


@dataclass(frozen=True)
class Step:
    id: str
    label: str
    instruction: str
    zoom: str | None = None          # camera zoom level this step asks for (needs operator confirm)
    distance_m: float | None = None  # ruler distance from the right camera's lens face
    head: str | None = None          # head preset this step points at (config.HEAD_PRESETS; needs confirm)
    capture: bool = True             # False for a head-only step (e.g. the end of a run)

    def describe(self) -> dict:
        return {"id": self.id, "label": self.label, "zoom": self.zoom, "distance_m": self.distance_m,
                "head": self.head, "head_deg": head_deg(self.head), "capture": self.capture,
                "instruction": self.instruction}


@dataclass(frozen=True)
class Procedure:
    id: str
    name: str
    version: str
    kind: str                        # "zoom" | "depth" | "head": which analyser reads it
    summary: str
    setup: tuple[str, ...]
    steps: tuple[Step, ...]
    captures_per_step: int = 1

    def step(self, step_id: str) -> Step:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(step_id)

    def describe(self) -> dict:
        """Numbered operator instructions for the UI."""
        numbered = [*self.setup, *(s.instruction for s in self.steps)]
        return {"id": self.id, "name": self.name, "version": self.version, "kind": self.kind,
                "summary": self.summary, "captures_per_step": self.captures_per_step,
                "instructions": [f"{i}. {text}" for i, text in enumerate(numbered, 1)],
                "steps": [s.describe() for s in self.steps]}


def head_deg(preset: str | None) -> dict | None:
    """A preset's neck angles in degrees (roll, pitch, yaw by joint name), or None."""
    return None if preset is None else dict(zip(NECK_JOINTS, HEAD_PRESETS[preset]))


def head_step(preset: str, capture: bool = True, step_id: str | None = None, label: str | None = None,
              warning: str = "") -> Step:
    then = "then press Capture." if capture else "nothing to capture."
    return Step(
        id=step_id or f"head_{preset}", label=label or f"Head: {preset}", head=preset, capture=capture,
        instruction=f"Point head at {preset} {HEAD_PRESETS[preset]} (roll, pitch, yaw deg): tick the confirm box "
                    f"and press 'Move head' (the neck moves; dry-run unless the viewer was started with "
                    f"--allow-head-motion and REACHY_ENABLE_MOTION=true), wait for the read-back, {then}"
                    + (f" {warning}" if warning else ""))


def end_step() -> Step:
    return head_step("neutral", capture=False, step_id="end_neutral", label="End: head to neutral")


def zoom_fov_procedure(levels=ZOOM_ORDER) -> Procedure:
    steps = tuple(Step(
        id=f"zoom_{level.lower()}", label=f"Zoom {level}", zoom=level,
        instruction=f"Zoom {level}: tick the confirm box and press 'Set zoom {level}' (the lens moves; dry-run "
                    f"unless the viewer was started with --allow-zoom-commands), wait for the read-back, "
                    f"then press Capture.") for level in levels)
    return Procedure(
        id="zoom_fov", name="Zoom field-of-view check", version="2", kind="zoom",
        summary="Head at the board view, board at each zoom level; the analyser reports the apparent cell pitch "
                "per level.",
        setup=("Arm at rest on the table; the head is moved only by the head steps below.",
               f"The widest level is expected to be {SCENE_ZOOM}; the analysis confirms it."),
        steps=(head_step("board_view", capture=False), *steps, end_step()))


def depth_ruler_procedure(distances=DISTANCES_M, captures_per_step=3) -> Procedure:
    steps = tuple(Step(
        id=f"dist_{str(d).replace('.', '_')}m", label=f"{d:.1f} m", distance_m=float(d),
        instruction=f"Hold the flat, textured target (the 60 mm cube face or a checker card) square to the right "
                    f"camera, centred in the image, {d:.1f} m from its lens face along its axis, then press "
                    f"Capture {captures_per_step} times.") for d in distances)
    return Procedure(
        id="depth_ruler", name="Depth ruler test", version="2", kind="depth",
        summary="Head at neutral, target at known ruler distances; the analyser compares stereo depth with the "
                "ruler.",
        setup=(f"Arm at rest at the scene zoom ({SCENE_ZOOM}); do not change the zoom during this test.",
               "The first step sets the head to neutral; it stays there for the whole test.",
               "Measure each distance from the right camera's lens face along its axis."),
        steps=(head_step("neutral", capture=False), *steps), captures_per_step=captures_per_step)


#: Head-sweep poses, in order: neutral, the board view, then the sweep around it.
SWEEP_PRESETS = ("neutral", "board_view", "sweep_yaw_right_5", "sweep_yaw_left_5", "sweep_yaw_right_8",
                 "sweep_yaw_left_8", "sweep_roll_right_5", "sweep_roll_left_5")


def head_sweep_procedure(presets=SWEEP_PRESETS, captures_per_step=2) -> Procedure:
    return Procedure(
        id="head_sweep", name="Head sweep", version="2", kind="head",
        summary="Board captured from each head pose; the analyser places the board centre in the torso frame "
                "from stereo depth and the recorded neck angles, and reports the spread across poses.",
        setup=("Arm at rest on the table, the taped 3x3 board clear and still; nobody near the head.",
               "Run 'Board-view approach' first in the first robot session.",
               f"At each pose press Capture {captures_per_step} times; a capture is refused until the neck has "
               "settled on its command."),
        steps=(*(head_step(p) for p in presets), end_step()), captures_per_step=captures_per_step)


def board_view_approach_procedure() -> Procedure:
    return Procedure(
        id="board_view_approach", name="Board-view approach", version="2", kind="head",
        summary="First robot session: step the neck up to the board view (20, 55, 60, then 64 deg pitch) with "
                "the operator watching, capturing at each pose.",
        setup=("Arm at rest on the table; Siva watches the head through every move, hand near the Stop button.",
               "Run this once before any head sweep or zoom check in the first robot session."),
        steps=(head_step("neutral"),
               head_step("board_approach_20", warning="The head must tilt DOWN. If it tilts up or sideways, "
                                                      "press Stop head and report."),
               *(head_step(p) for p in ("board_approach_55", "board_approach_60", "board_view")),
               end_step()))


PROCEDURES = {p.id: p for p in (zoom_fov_procedure(), depth_ruler_procedure(), board_view_approach_procedure(),
                                head_sweep_procedure())}


def get_procedure(procedure_id: str) -> Procedure:
    try:
        return PROCEDURES[procedure_id]
    except KeyError:
        raise KeyError(f"unknown procedure {procedure_id!r}; choose one of {sorted(PROCEDURES)}") from None


def encode_jpeg(pixels, quality: int = 95) -> bytes:
    import cv2
    ok, jpeg = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Unable to encode image")
    return jpeg.tobytes()


@dataclass
class CaptureRun:
    """One operator pass through a procedure, held in memory."""
    procedure: Procedure
    captures: dict[str, list[dict]] = field(default_factory=dict)
    zoom_commands: dict[str, list[dict]] = field(default_factory=dict)
    head_commands: dict[str, list[dict]] = field(default_factory=dict)
    head_events: list[dict] = field(default_factory=list)  # Stop and Release, not tied to a step

    def pair_count(self) -> int:
        return sum(len(v) for v in self.captures.values())

    def apply_zoom(self, step_id: str, setter, confirmed: bool) -> list[dict]:
        """Command this step's zoom level through `setter` (a `ZoomSetter`, or anything with `set_both`).

        Refused with ZoomNotConfirmed unless `confirmed`; whether the command is really sent or only
        logged is the setter's `dry_run`. Every attempt is recorded in the run (and so the bundle)."""
        step = self.procedure.step(step_id)
        if step.zoom is None:
            raise ValueError(f"step {step_id!r} does not change the zoom")
        if not confirmed:
            raise ZoomNotConfirmed("A zoom command needs the operator's explicit confirm.")
        records = setter.set_both(step.zoom)
        self.zoom_commands.setdefault(step_id, []).extend(records)
        return records

    def apply_head(self, step_id: str, mover, confirmed: bool) -> dict:
        """Point the head at this step's preset through `mover` (anything with `move(preset) -> record`).

        Without the confirm nothing is sent and HeadNotConfirmed is raised; a gate refusal raises
        HeadRefused. Whether a passed move is live or a logged dry run is the mover's. Every attempt,
        refused ones included, is recorded in the run (and so the bundle)."""
        step = self.procedure.step(step_id)
        if step.head is None:
            raise ValueError(f"step {step_id!r} does not move the head")
        attempts = self.head_commands.setdefault(step_id, [])
        if not confirmed:
            attempts.append({"action": "move", "preset": step.head, "target_deg": head_deg(step.head),
                             "outcome": "refused", "gate": {"passed": False,
                                                            "failures": ["Operator approval not given"]}})
            raise HeadNotConfirmed("A head move needs the operator's explicit confirm.")
        record = mover.move(step.head)
        attempts.append(record)
        if record.get("outcome") == "refused":
            raise HeadRefused("Head move refused: " + "; ".join(record.get("gate", {}).get("failures", [])))
        return record

    def record_head_event(self, record: dict) -> None:
        """A Stop or Release, kept in order with the run."""
        self.head_events.append(record)

    def add_capture(self, step_id: str, meta: dict, left_jpeg: bytes, right_jpeg: bytes) -> dict:
        """Record one captured pair for a step. `meta` is the viewer's pair metadata with its camera_context."""
        if not self.procedure.step(step_id).capture:
            raise ValueError(f"step {step_id!r} has nothing to capture")
        taken = self.captures.setdefault(step_id, [])
        if len(taken) >= MAX_CAPTURES_PER_STEP:
            raise RunLimit(f"At most {MAX_CAPTURES_PER_STEP} pairs per step; redo the run to start over.")
        if self.pair_count() >= MAX_PAIRS_PER_RUN:
            raise RunLimit(f"At most {MAX_PAIRS_PER_RUN} pairs per run.")
        record = {"pair_id": meta.get("pair_id"), "meta": meta, "left": left_jpeg, "right": right_jpeg}
        taken.append(record)
        return record

    def state(self) -> dict:
        return {"procedure": {k: v for k, v in self.procedure.describe().items() if k != "steps"},
                "steps": [{**s.describe(), "captured": len(self.captures.get(s.id, [])),
                           "zoom_commands": self.zoom_commands.get(s.id, []),
                           "head_commands": self.head_commands.get(s.id, [])}
                          for s in self.procedure.steps],
                "head_events": self.head_events, "pairs": self.pair_count(), "max_pairs": MAX_PAIRS_PER_RUN}

    def calibration_id(self) -> Any:
        for taken in self.captures.values():
            for record in taken:
                return (record["meta"].get("camera_context") or {}).get("calibration_id")
        return None

    def analysis_profile(self) -> dict:
        for taken in self.captures.values():
            for record in taken:
                return (record["meta"].get("camera_context") or {}).get("analysis_profile") or {}
        return {}

    def bundle_files(self) -> dict[str, bytes]:
        files: dict[str, bytes] = {}
        steps = []
        for index, step in enumerate(self.procedure.steps, 1):
            entries = []
            for n, record in enumerate(self.captures.get(step.id, []), 1):
                folder = f"steps/{index:02d}_{step.id}/cap{n}"
                files[f"{folder}/left.jpg"] = record["left"]
                files[f"{folder}/right.jpg"] = record["right"]
                files[f"{folder}/pair.json"] = json_bytes(record["meta"])
                context = record["meta"].get("camera_context") or {}
                zoom = context.get("zoom") or {}
                entries.append({"folder": folder, "pair_id": record["pair_id"],
                                "zoom_level": zoom.get("captured"), "zoom_levels": zoom.get("levels"),
                                "observation_skew_ms": record["meta"].get("observation_skew_ms"),
                                "neck_deg": neck_readback(record["meta"]),
                                "head_command": record["meta"].get("head_command")})
            steps.append({"index": index, "id": step.id, "label": step.label, "instruction": step.instruction,
                          "distance_m": step.distance_m, "requested_zoom": step.zoom,
                          "requested_head": step.head, "requested_head_deg": head_deg(step.head),
                          "zoom_commands": self.zoom_commands.get(step.id, []),
                          "head_commands": self.head_commands.get(step.id, []), "captures": entries})
        summary = {"schema_version": "2", "procedure": self.identity(), "kind": self.procedure.kind,
                   "calibration_id": self.calibration_id(), "steps": steps, "head_events": self.head_events}
        profile = self.analysis_profile()
        if profile.get("source"):
            summary["source"] = profile["source"]  # which camera geometry the analysers must use
            if profile["source"] == "sim":
                summary["sim_lens"] = profile.get("lens")
        return {"capture.json": json_bytes(summary), **files}

    def identity(self) -> dict:
        return {"id": self.procedure.id, "name": self.procedure.name, "version": self.procedure.version}

    def build_zip(self, versions: dict) -> bytes:
        if not self.pair_count():
            raise RunLimit("Nothing captured yet.")
        return build_bundle(self.bundle_files(), versions, self.calibration_id(), procedure=self.identity())


def neck_readback(meta: dict) -> dict | None:
    """Neck present angles (deg) frozen with a capture (the viewer's head_context), or None."""
    joints = (meta.get("head_context") or {}).get("joints") or {}
    try:
        return {name: float(joints[name]["present_deg"]) for name in NECK_JOINTS}
    except (KeyError, TypeError, ValueError):
        return None


def sensitivity_mm_per_px(z_m: float, focal_px: float = SENSITIVITY_FOCAL_PX,
                          baseline_m: float = SENSITIVITY_BASELINE_M) -> float:
    """Depth change in mm per pixel of disparity error: Z^2 / (f B)."""
    return z_m ** 2 / (focal_px * baseline_m) * 1000
