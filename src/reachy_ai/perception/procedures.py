"""Operator data-capture procedures (#28): step definitions, the capture run, and its bundle.

Camera and neck only, never arms. A procedure is a list of numbered operator steps; the operator
captures a stereo pair per step. A procedure can ask for two robot commands, each only after an
explicit operator confirm: a camera zoom level (`CaptureRun.apply_zoom`, through
`camera_zoom.ZoomSetter`) and a named head pose (`CaptureRun.apply_head`, through
`motion.primitives` and the head safety gate). The Sound check (#59) moves nothing at all: its steps call
the audio helper (`CaptureRun.apply_sound`, through `audio.audio_client`), and its recordings need an
operator tick that the speaker agreed. Owner decision 2026-10-20: "Neck motion is required
for vision data capture. Capture procedures may articulate the head through the neck for viewing and
perception. Arm motion stays forbidden in capture procedures." No arm or gripper command exists
here. Nothing is written to disk by this module: the run lives in memory (pairs and audio clips alike) and
`CaptureRun.build_zip` returns the bundle bytes.

The viewer (stereo_app.py) and the CLI fallbacks (scripts/zoom_fov_check.py,
scripts/ruler_depth_test.py) both drive these same objects; the offline analysers
(scripts/analyze_capture_bundle.py) read the bundle back.
"""
from __future__ import annotations

import io
import logging
import wave
from dataclasses import dataclass, field
from typing import Any

from reachy_ai.audio.audio_client import AudioHelperError
from reachy_ai.camera_zoom import SCENE_ZOOM, ZOOM_ORDER, zoom_read_warning
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
#: Sound check (#59): the tone, the phrase and the recordings.
SOUND_PHRASE = "Hello, I am Reachy"
SOUND_TONE = {"seconds": 1.0, "freq": 1000.0}
SOUND_RECORD_S = 2.0
SOUND_CHANNELS = 4
MAX_TAKES_PER_STEP = 3
MAX_NOTE_CHARS = 200
RECORDING_SOUNDS = ("loopback", "baseline")


class ZoomNotConfirmed(PermissionError):
    """A zoom command was requested without the operator's explicit confirm."""


class HeadNotConfirmed(PermissionError):
    """A head move was requested without the operator's explicit confirm."""


class HeadRefused(PermissionError):
    """The head safety gate refused the move; nothing was sent."""


class RunLimit(ValueError):
    pass


class RecordingNotConsented(PermissionError):
    """A recording was requested without the operator's tick that the speaker agreed."""


class SoundFailed(RuntimeError):
    """The audio helper refused or failed an operation; `code` is the helper's (busy, timeout, ...)."""

    def __init__(self, message: str, code: str = "failed"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Step:
    id: str
    label: str
    instruction: str
    zoom: str | None = None          # camera zoom level this step asks for (needs operator confirm)
    distance_m: float | None = None  # ruler distance from the right camera's lens face
    head: str | None = None          # head preset this step points at (config.HEAD_PRESETS; needs confirm)
    capture: bool = True             # False for a head-only step (e.g. the end of a run)
    zoom_role: str | None = None     # "start" (read-back only), "check" (read-back vs the profile) or "restore"
    sound: str | None = None         # "status", "tone", "say", "loopback" or "baseline" (the audio helper)

    def describe(self) -> dict:
        return {"id": self.id, "label": self.label, "zoom": self.zoom, "distance_m": self.distance_m,
                "head": self.head, "head_deg": head_deg(self.head), "capture": self.capture,
                "zoom_role": self.zoom_role, "sound": self.sound, "instruction": self.instruction}


@dataclass(frozen=True)
class Procedure:
    id: str
    name: str
    version: str
    kind: str                        # "zoom" | "depth" | "head" | "sound": which analyser reads it
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


#: Why ZERO cannot be restored: the SDK refuses it as a command (confirmed on the sim, 2026-10-20).
ZERO_RESTORE_NOTE = ("ZERO cannot be commanded through the SDK; restarting the robot's camera service is expected "
                     "to re-home the lens (unconfirmed).")


def zoom_start_step() -> Step:
    return Step(
        id="zoom_start", label="Zoom: starting level", zoom_role="start", head="board_view", capture=True,
        instruction=f"Starting zoom, untouched: press 'Read zoom' (read-back only, no zoom command is sent; the first "
                    f"complete read is the starting level). Then Point head at board_view "
                    f"{HEAD_PRESETS['board_view']} (roll, pitch, yaw deg): tick the confirm box and press 'Move head' "
                    f"(dry-run unless the viewer was started with --allow-head-motion and REACHY_ENABLE_MOTION=true), "
                    f"wait for the read-back, then press Capture. The last step puts the lens back at the starting "
                    f"level.")


def zoom_restore_step() -> Step:
    return Step(
        id="zoom_restore", label="Zoom: restore starting level", zoom_role="restore", capture=False,
        instruction="Restore the starting zoom: tick the confirm box and press 'Restore zoom' (the lens moves; "
                    "dry-run unless the viewer was started with --allow-zoom-commands), wait for the read-back. "
                    "If the starting level is unknown or cannot be commanded, set it by hand.")


def zoom_check_step() -> Step:
    return Step(
        id="zoom_check", label="Zoom: read and check", zoom_role="check", capture=False,
        instruction="Read the zoom: press 'Read zoom' (read-back only, no command). A warning shows if it differs "
                    "from the level the calibration profile is valid at, or if the profile's zoom is unknown.")


def zoom_fov_procedure(levels=ZOOM_ORDER) -> Procedure:
    steps = tuple(Step(
        id=f"zoom_{level.lower()}", label=f"Zoom {level}", zoom=level,
        instruction=f"Zoom {level}: tick the confirm box and press 'Set zoom {level}' (the lens moves; dry-run "
                    f"unless the viewer was started with --allow-zoom-commands), wait for the read-back, "
                    f"then press Capture.") for level in levels)
    return Procedure(
        id="zoom_fov", name="Zoom field-of-view check", version="4", kind="zoom",
        summary="Head at the board view, board at each zoom level; the analyser reports the apparent cell pitch "
                "per level. The starting zoom is read and captured first, untouched, and put back at the end.",
        setup=("Arm at rest on the table; the head is moved only by the head steps below.",
               f"The widest level is expected to be {SCENE_ZOOM}; the analysis confirms it.",
               "Run this check last in a robot session: it can leave the lens at a level the SDK cannot restore."),
        steps=(zoom_start_step(), *steps, end_step(), zoom_restore_step()))


def depth_ruler_procedure(distances=DISTANCES_M, captures_per_step=3) -> Procedure:
    steps = tuple(Step(
        id=f"dist_{str(d).replace('.', '_')}m", label=f"{d:.1f} m", distance_m=float(d),
        instruction=f"Hold the flat, textured target (the 60 mm cube face or a checker card) square to the right "
                    f"camera, centred in the image, {d:.1f} m from its lens face along its axis, then press "
                    f"Capture {captures_per_step} times.") for d in distances)
    return Procedure(
        id="depth_ruler", name="Depth ruler test", version="3", kind="depth",
        summary="Head at neutral, target at known ruler distances; the analyser compares stereo depth with the "
                "ruler.",
        setup=(f"Arm at rest at the scene zoom ({SCENE_ZOOM}); do not change the zoom during this test.",
               "The first step reads the zoom (no command) and warns if it is not the level the calibration "
               "profile is valid at.",
               "The second step sets the head to neutral; it stays there for the whole test.",
               "Measure each distance from the right camera's lens face along its axis."),
        steps=(zoom_check_step(), head_step("neutral", capture=False), *steps),
        captures_per_step=captures_per_step)


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


def sound_check_procedure() -> Procedure:
    tone = f"{SOUND_TONE['seconds']:g} s at {SOUND_TONE['freq']:g} Hz"
    record = f"{SOUND_RECORD_S:g} s on {SOUND_CHANNELS} channels"
    consent = "tick 'I confirm: the speaker agreed to this recording'"
    return Procedure(
        id="sound_check", name="Sound check", version="1", kind="sound",
        summary="Reachy's speaker and microphones through the audio helper: a tone and a phrase the operator "
                "confirms by ear, a loopback recording while the tone plays, and a room baseline. The analyser "
                "reports the level per channel.",
        setup=("No motion of any kind: no head, arm or zoom command is part of this check.",
               "Start the audio helper next to the speaker (the NUC, or the Mac for sim work) and, on the robot, "
               "the SSH tunnel (docs/SOUND_CHECK.md). Set the system volume low first; the helper never changes it.",
               "Recording needs the speaker's agreement: tick the consent box before each recording. Audio is held "
               "in memory and is kept only in the downloaded bundle."),
        steps=(Step(id="sound_status", label="Helper status", sound="status", capture=False,
                    instruction="Press 'Check helper': it shows whether the helper answers, its backend (alsa, "
                                "mac or fake), the TTS engine and the sound devices it found."),
               Step(id="sound_tone", label="Speaker: tone", sound="tone", capture=False,
                    instruction=f"Press 'Play tone' ({tone}, capped at -12 dBFS). Then answer 'Heard it' or "
                                f"'Not heard', with a volume note (e.g. quiet, comfortable, loud)."),
               Step(id="sound_phrase", label="Speaker: phrase", sound="say", capture=False,
                    instruction=f"Press 'Say phrase' (Reachy says \"{SOUND_PHRASE}\"). Then answer 'Heard it' or "
                                f"'Not heard', with a volume note. Without a TTS engine the step is skipped with a "
                                f"note."),
               Step(id="sound_loopback", label="Microphones: loopback", sound="loopback", capture=False,
                    instruction=f"Everyone quiet. {consent.capitalize()} and press 'Record loopback' ({record}; "
                                f"the recording starts, then the tone plays in it)."),
               Step(id="sound_baseline", label="Microphones: room baseline", sound="baseline", capture=False,
                    instruction=f"Everyone quiet, nothing playing. {consent.capitalize()} and press "
                                f"'Record baseline' ({record} of silence: the noise floor).")))


PROCEDURES = {p.id: p for p in (zoom_fov_procedure(), depth_ruler_procedure(), board_view_approach_procedure(),
                                head_sweep_procedure(), sound_check_procedure())}


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
    sound_results: dict[str, list[dict]] = field(default_factory=dict)  # every audio-helper attempt per step
    answers: dict[str, dict] = field(default_factory=dict)  # the operator's heard / not heard per step
    clips: dict[str, list[bytes]] = field(default_factory=dict)  # WAV takes per step, in memory only
    helper_status: dict | None = None  # the helper's last /status
    commands_attempted: bool = False  # a zoom command (live or dry-run) has been asked for
    _start: str | None = None  # the starting zoom: the first complete read, fixed once a command was attempted

    def pair_count(self) -> int:
        return sum(len(v) for v in self.captures.values())

    def apply_zoom(self, step_id: str, setter, confirmed: bool, profile_zoom: str | None = None) -> list[dict]:
        """Command this step's zoom level through `setter` (a `ZoomSetter`, or anything with `set_both`).

        Refused with ZoomNotConfirmed unless `confirmed`; whether the command is really sent or only
        logged is the setter's `dry_run`. Every attempt is recorded in the run (and so the bundle).
        A "start" or "check" step only reads the zoom back (`setter.read_both`, no command, no confirm);
        `profile_zoom` is the level the calibration profile is valid at, for the "check" warning.
        A "restore" step commands the level the "start" step read."""
        step = self.procedure.step(step_id)
        if step.zoom_role in ("start", "check"):
            return self._read_zoom(step, setter, profile_zoom)
        if step.zoom_role == "restore":
            return self._restore_zoom(step, setter, confirmed)
        if step.zoom is None:
            raise ValueError(f"step {step_id!r} does not change the zoom")
        if not confirmed:
            raise ZoomNotConfirmed("A zoom command needs the operator's explicit confirm.")
        self.commands_attempted = True
        records = setter.set_both(step.zoom)
        self.zoom_commands.setdefault(step_id, []).extend(records)
        return records

    def _read_zoom(self, step: Step, setter, profile_zoom: str | None) -> list[dict]:
        records = [{"action": "read", **r} for r in setter.read_both()]
        if step.zoom_role == "start":
            levels = {r["level"] for r in records}
            if self._start is not None or self.commands_attempted:
                # an extra read-back: recorded, but it never changes the starting level
                records = [{**r, "extra": True} for r in records]
            elif len(levels) == 1 and None not in levels:
                self._start = levels.pop()
        if step.zoom_role == "check":
            warning = zoom_read_warning(profile_zoom, {r["eye"]: r["level"] for r in records})
            records = [{**r, "profile_zoom": profile_zoom, "warning": warning} for r in records]
        self.zoom_commands.setdefault(step.id, []).extend(records)
        return records

    def capture_blocker(self, step_id: str) -> str | None:
        """Why a capture on this step must wait, or None. A step that commands a zoom is captured only after
        that command has completed (live or dry-run, not failed): a pair taken earlier shows another zoom."""
        step = self.procedure.step(step_id)
        if step.zoom and not any(not r.get("error") for r in self.zoom_commands.get(step_id, [])):
            return (f"Zoom {step.zoom} has not been commanded for this step yet: tick the confirm box and press "
                    f"'Set zoom {step.zoom}' (wait for it to finish) before capturing.")
        return None

    def start_zoom(self) -> str | None:
        """The starting level: the first complete read of the run (ZERO included), or None if there was no
        complete read before the first zoom command. Later reads never change it."""
        return self._start

    def _restore_zoom(self, step: Step, setter, confirmed: bool) -> list[dict]:
        """Put the lens back at the starting level. Unknown start: nothing is sent; the operator sets it by hand."""
        start = self.start_zoom()
        sent = self.zoom_commands.setdefault(step.id, [])
        if start is None:
            records = [{"action": "restore", "target": None, "ok": False, "manual": True,
                        "dry_run": getattr(setter, "dry_run", None),
                        "note": "The starting zoom is unknown, so it cannot be restored: set the zoom to its "
                                "starting level by hand."}]
            sent.extend(records)
            return records
        if not confirmed:
            raise ZoomNotConfirmed("A zoom command needs the operator's explicit confirm.")
        self.commands_attempted = True
        records = [{**r, "action": "restore", "target": start} for r in setter.set_both(start)]
        for r in records:
            if not r.get("ok") and not r.get("dry_run"):
                r.update(manual=True, note=ZERO_RESTORE_NOTE if start == "ZERO" else
                         f"The starting zoom {start} could not be commanded or confirmed by read-back: "
                         f"set the zoom to {start} by hand.")
        sent.extend(records)
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

    def apply_sound(self, step_id: str, client, confirmed: bool = False) -> dict:
        """Run this step's audio-helper operation through `client` (an `AudioClient`).

        A recording step is refused with RecordingNotConsented unless `confirmed` (the operator's tick that
        the speaker agreed). A helper failure raises SoundFailed; a phrase without a TTS engine is recorded
        as skipped. Every attempt is recorded in the run (and so the bundle); WAVs stay in memory."""
        step = self.procedure.step(step_id)
        if step.sound is None:
            raise ValueError(f"step {step_id!r} has no sound operation")
        attempts = self.sound_results.setdefault(step_id, [])
        if step.sound in RECORDING_SOUNDS:
            if not confirmed:
                attempts.append({"op": step.sound, "outcome": "refused", "consent": False})
                raise RecordingNotConsented("A recording needs the operator's tick that the speaker agreed.")
            if len(self.clips.get(step_id, [])) >= MAX_TAKES_PER_STEP:
                raise RunLimit(f"At most {MAX_TAKES_PER_STEP} recordings per step; redo the run to start over.")
        try:
            record = self._sound(step, client)
        except AudioHelperError as exc:
            if step.sound == "say" and exc.code == "unavailable":
                record = {"op": "say", "outcome": "skipped", "text": SOUND_PHRASE,
                          "note": "No TTS engine on the helper (piper or espeak-ng): the phrase step is skipped."}
                attempts.append(record)
                return record
            attempts.append({"op": step.sound, "outcome": "failed", "code": exc.code, "error": str(exc)})
            raise SoundFailed(str(exc), exc.code) from exc
        attempts.append(record)
        return record

    def _sound(self, step: Step, client) -> dict:
        if step.sound == "status":
            status = client.status()
            self.helper_status = status
            return {"op": "status", "outcome": "ok", "backend": status.get("backend"), "tts": status.get("tts"),
                    "version": status.get("version"), "rate": status.get("rate"), "devices": status.get("devices")}
        if step.sound == "tone":
            reply = client.tone(SOUND_TONE["seconds"], SOUND_TONE["freq"])
            return {"op": "tone", "outcome": "played", **{k: reply.get(k) for k in ("seconds", "freq", "dbfs",
                                                                                    "backend", "elapsed_s")}}
        if step.sound == "say":
            reply = client.say(SOUND_PHRASE)
            return {"op": "say", "outcome": "spoken", "text": SOUND_PHRASE, "tts": reply.get("tts"),
                    "backend": reply.get("backend"), "elapsed_s": reply.get("elapsed_s")}
        tone = SOUND_TONE if step.sound == "loopback" else None
        data = client.record(SOUND_RECORD_S, SOUND_CHANNELS, tone=tone)
        try:
            with wave.open(io.BytesIO(data), "rb") as clip:
                info = {"channels": clip.getnchannels(), "rate": clip.getframerate(),
                        "seconds": round(clip.getnframes() / clip.getframerate(), 3),
                        "sample_bits": clip.getsampwidth() * 8}
        except (wave.Error, EOFError, ZeroDivisionError) as exc:
            raise AudioHelperError(f"the recording is not a readable WAV ({exc})") from exc
        takes = self.clips.setdefault(step.id, [])
        takes.append(data)
        return {"op": step.sound, "outcome": "recorded", "consent": True, "take": len(takes),
                "requested_seconds": SOUND_RECORD_S, "requested_channels": SOUND_CHANNELS, "tone": tone, **info}

    def record_answer(self, step_id: str, heard: bool, note: str = "") -> dict:
        """The operator's answer for a tone or phrase step: heard or not heard, and a volume note."""
        step = self.procedure.step(step_id)
        if step.sound not in ("tone", "say"):
            raise ValueError(f"step {step_id!r} takes no heard / not heard answer")
        outcomes = {r.get("outcome") for r in self.sound_results.get(step_id, [])}
        if not outcomes & {"played", "spoken"}:
            raise ValueError("Play this step's sound first; there is nothing to answer yet.")
        answer = {"heard": bool(heard), "volume_note": str(note or "").strip()[:MAX_NOTE_CHARS]}
        self.answers[step_id] = answer
        return answer

    def sound_count(self) -> int:
        return sum(len(v) for v in self.sound_results.values())

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
                           "head_commands": self.head_commands.get(s.id, []),
                           "sound_results": self.sound_results.get(s.id, []), "answer": self.answers.get(s.id)}
                          for s in self.procedure.steps],
                "head_events": self.head_events, "pairs": self.pair_count(), "max_pairs": MAX_PAIRS_PER_RUN,
                "can_download": bool(self.pair_count() or self.sound_count())}

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
            row = {"index": index, "id": step.id, "label": step.label, "instruction": step.instruction,
                   "distance_m": step.distance_m, "requested_zoom": step.zoom,
                   "requested_head": step.head, "requested_head_deg": head_deg(step.head),
                   "zoom_role": step.zoom_role,
                   "zoom_commands": self.zoom_commands.get(step.id, []),
                   "head_commands": self.head_commands.get(step.id, []), "captures": entries}
            if step.sound:
                clips = []
                for n, data in enumerate(self.clips.get(step.id, []), 1):
                    clips.append(f"steps/{index:02d}_{step.id}/take{n}.wav")
                    files[clips[-1]] = data
                row.update(sound=step.sound, sound_results=self.sound_results.get(step.id, []),
                           answer=self.answers.get(step.id), clips=clips)
            steps.append(row)
        summary = {"schema_version": "2", "procedure": self.identity(), "kind": self.procedure.kind,
                   "calibration_id": self.calibration_id(), "steps": steps, "head_events": self.head_events}
        if self.procedure.kind == "zoom":
            summary["start_zoom"] = self.start_zoom()  # None: not read, unreadable or the eyes differed
        if self.procedure.kind == "sound":
            summary.update(helper_status=self.helper_status, backend=(self.helper_status or {}).get("backend"),
                           answers=self.answers, phrase=SOUND_PHRASE, tone=SOUND_TONE,
                           record={"seconds": SOUND_RECORD_S, "channels": SOUND_CHANNELS},
                           consent="The operator ticked 'the speaker agreed to this recording' before each "
                                   "recording.")
        profile = self.analysis_profile()
        if profile.get("source"):
            summary["source"] = profile["source"]  # which camera geometry the analysers must use
            if profile["source"] == "sim":
                summary["sim_lens"] = profile.get("lens")
        return {"capture.json": json_bytes(summary), **files}

    def identity(self) -> dict:
        return {"id": self.procedure.id, "name": self.procedure.name, "version": self.procedure.version}

    def build_zip(self, versions: dict) -> bytes:
        if not self.pair_count() and not self.sound_count():
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
