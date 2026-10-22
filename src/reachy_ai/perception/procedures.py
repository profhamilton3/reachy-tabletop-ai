"""Operator data-capture procedures (#28): step definitions, the capture run, and its bundle.

Camera, neck and (Gripper range) r_gripper only, never arms. A procedure is a list of numbered operator steps; the
operator captures a stereo pair per step. A procedure can ask for two robot commands, each only after an
explicit operator confirm: a camera zoom level (`CaptureRun.apply_zoom`, through
`camera_zoom.ZoomSetter`) and a named head pose (`CaptureRun.apply_head`, through
`motion.primitives` and the head safety gate). The Sound check (#59) moves nothing at all: its steps call
the audio helper (`CaptureRun.apply_sound`, through `audio.audio_client`), and its recordings need an
operator tick that the speaker agreed. The Robot inventory (#62) moves nothing and writes nothing either: it
reads the SDK through the viewer's SDK process (`robot_inventory.sdk_inventory`), runs the audio helper's fixed
NUC probes, and takes the operator's answers; the NUC output is redacted by the helper and again here.
Owner decision 2026-10-20: "Neck motion is required
for vision data capture. Capture procedures may articulate the head through the neck for viewing and
perception. Arm motion stays forbidden in capture procedures." No arm command exists here.
The Gripper range (#27 Part A) adds one more command after an explicit confirm: a bounded r_gripper move
(`CaptureRun.apply_gripper`, through `motion.primitives` and the gripper safety gate). Owner decision
2026-10-22: "Gripper-only motion is allowed in capture procedures for gripper calibration: only `r_gripper`
is powered, the arm stays at REST, every move needs a confirm and passes a gripper gate (travel limits, low
speed and torque, a force-limited close, `REACHY_ENABLE_MOTION`), and the gripper returns to compliant at the
end." Nothing is written to disk by this module: the run lives in memory (pairs and audio clips alike) and
`CaptureRun.build_zip` returns the bundle bytes.

The viewer (stereo_app.py) and the CLI fallbacks (scripts/zoom_fov_check.py,
scripts/ruler_depth_test.py) both drive these same objects; the offline analysers
(scripts/analyze_capture_bundle.py) read the bundle back.
"""
from __future__ import annotations

import csv
import hashlib
import io
import logging
import subprocess
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reachy_ai import config
from reachy_ai.audio.audio_client import AudioHelperError
from reachy_ai.camera_zoom import SCENE_ZOOM, ZOOM_ORDER, zoom_read_warning
from reachy_ai.config import (
    GRIPPER_LIMITS_DEG,
    GRIPPER_MAX_FORCE,
    GRIPPER_MAX_LOAD,
    GRIPPER_MAX_SPEED_DEG_S,
    GRIPPER_MAX_STEP_DEG,
    GRIPPER_MAX_TEMP_C,
    GRIPPER_MAX_TORQUE_PCT,
    GRIPPER_SAMPLE_HZ,
    GRIPPER_SWEEP_OPEN_DEG,
    GRIPPER_SWEEP_SHUT_DEG,
    GRIPPER_SWEEP_SPEED_DEG_S,
    GRIPPER_SWEEP_TORQUE_PCT,
    HEAD_PRESETS,
    NECK_JOINTS,
)
from reachy_ai.motion.primitives import GRIPPER_ABORTS
from reachy_ai.pair_bundle import build_bundle, json_bytes
from reachy_ai.robot_inventory import count_replacements, redact, topic_lines, workstation_names

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
SOUND_CHANNELS = 4  # without a channel layout from the helper ("plain")
MAX_TAKES_PER_STEP = 3
MAX_NOTE_CHARS = 200
RECORDING_SOUNDS = ("loopback", "baseline")
#: Robot inventory (#62): the audio helper's probe allow-list, in the order they run.
INVENTORY_PROBES = ("audio_devices", "audio_hw_params", "usb", "system", "ros2_topics", "imu")
PROBE_OUTCOMES = ("ok", "failed", "not_run")
HELPER_UNREACHABLE = "helper unreachable"


class ZoomNotConfirmed(PermissionError):
    """A zoom command was requested without the operator's explicit confirm."""


class HeadNotConfirmed(PermissionError):
    """A head move was requested without the operator's explicit confirm."""


class HeadRefused(PermissionError):
    """The head safety gate refused the move; nothing was sent."""


class GripperNotConfirmed(PermissionError):
    """A gripper move was requested without the operator's explicit confirm."""


class GripperRefused(PermissionError):
    """The gripper safety gate (or the run's own rules) refused the move; nothing was sent."""


class RunLimit(ValueError):
    pass


class RecordingNotConsented(PermissionError):
    """A recording was requested without the operator's tick that the speaker agreed."""


class InventoryFailed(RuntimeError):
    """The SDK reads could not be taken (no connection, a timeout, busy); the cause is chained."""


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
    inventory: str | None = None     # "sdk", "nuc" or "answers" (the Robot inventory, #62)
    gripper: str | None = None       # "read" (Stage 0, read-only), "move" (needs confirm) or "release"
    gripper_target_deg: float | None = None  # the r_gripper goal of a "move" step

    def describe(self) -> dict:
        return {"id": self.id, "label": self.label, "zoom": self.zoom, "distance_m": self.distance_m,
                "head": self.head, "head_deg": head_deg(self.head), "capture": self.capture,
                "zoom_role": self.zoom_role, "sound": self.sound, "inventory": self.inventory,
                "gripper": self.gripper, "gripper_target_deg": self.gripper_target_deg,
                "instruction": self.instruction}


@dataclass(frozen=True)
class Procedure:
    id: str
    name: str
    version: str
    kind: str                        # "zoom" | "depth" | "head" | "sound" | "inventory" | "gripper": its analyser
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
    record = (f"{SOUND_RECORD_S:g} s on every channel of the helper's layout: 4 plain, 6 for the ReSpeaker's "
              f"6-channel firmware, 1 for its 1-channel firmware")
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


def robot_inventory_procedure() -> Procedure:
    return Procedure(
        id="robot_inventory", name="Robot inventory", version="1", kind="inventory",
        summary="Read-only facts for #6, #7, #59 and #28 in one bundle: the SDK reads, the audio helper's fixed NUC "
                "probes, and two operator answers. The analyser answers the open questions (ReSpeaker and its "
                "firmware, the speaker device, the IMU topic, system versions).",
        setup=("No motion and no writes of any kind: the SDK is only read, and the NUC runs only the helper's "
               "fixed probes (nothing typed here is run).",
               "On the NUC, start the audio helper from a terminal with ROS sourced (with --imu-topic if the IMU "
               "topic is known), and open the SSH tunnel (docs/SOUND_CHECK.md). Without the helper the SDK step "
               "still runs and the NUC step is marked 'not run (helper unreachable)'.",
               "Host names, IP and MAC addresses, the user name and serial numbers are removed from the NUC output, "
               "by the helper and again by the viewer, before the bundle is built."),
        steps=(Step(id="inventory_sdk", label="SDK reads", inventory="sdk", capture=False,
                    instruction="Press 'Read robot': the SDK version, every joint's position and compliance, the "
                                "gripper, force sensor, fans, the cameras' zoom and focus, and the neck sign check "
                                "(IK only). Reads only; nothing moves."),
               Step(id="inventory_nuc", label="NUC probes", inventory="nuc", capture=False,
                    instruction="Press 'Run NUC probes': audio devices, the record device's hardware parameters, "
                                "USB devices, system versions, ROS 2 topics and the IMU topic, each with its own "
                                "result (ok, failed or not run, with the reason)."),
               Step(id="inventory_answers", label="Operator answers", inventory="answers", capture=False,
                    instruction="Answer 'Is the arm resting on the table?' (for the record only; nothing depends "
                                "on it), add a free note if you like, and press 'Save answers'.")))


#: Gripper range (#27 Part A): format_version 1 of docs/gripper-capture-format.md. Any change bumps it.
GRIPPER_FORMAT_VERSION = 1
GRIPPER_CSV_COLUMNS = ("step_id", "attempt", "time_s", "target_deg", "present_deg", "load", "force",
                       "temperature_c", "torque_limit_pct", "speed_limit_deg_s")
#: The offline sim check that gated Part A (reachy-1-2-sim @ 1948823, MuJoCo 3.11, REST keyframe, arm motors off,
#: only r_gripper actuated; docs/gripper-capture.md has the method). Recorded in every run record.
GRIPPER_SIM_CHECK = {
    "sim_commit": "1948823", "scene": "FWDCenterLabSivaPool",
    "finger_touches_table": False, "finger_clearance_mm": {"geom_distance": 37.0, "analytic": 32.8},
    "hand_contacts_at_rest": ["r_hand_back_col", "r_palm_col"],
    "gripper_reached_deg": -67.97,
    "arm_motion_deg": {"over_no_command_control": 0.48, "from_settled_arm": [0.43, 0.73],
                       "from_keyframe_incl_creep": 1.26},
    "tip_torso_m": [0.141, -0.397, -0.397], "aim_neck_deg": {"roll": 0.0, "pitch": 64.0, "yaw": -38.0},
    "fingers_in_view": True, "min_edge_margin_px": 167,
}


def gripper_step_id(direction: str, index: int) -> str:
    return f"gripper_{direction}_{index:02d}"


def gripper_range_procedure(speed=GRIPPER_SWEEP_SPEED_DEG_S, torque=GRIPPER_SWEEP_TORQUE_PCT) -> Procedure:
    def move(direction, index, target):
        return Step(
            id=gripper_step_id(direction, index), label=f"Gripper {direction} to {target:g}", gripper="move",
            gripper_target_deg=float(target),
            instruction=f"Hands clear of the gripper. Tick the confirm box and press 'Move gripper' (r_gripper to "
                        f"{target:g} deg at {speed:g} deg/s, {torque:g} % torque; dry-run unless the viewer was "
                        f"started with --allow-gripper-motion and REACHY_ENABLE_MOTION=true), wait for the "
                        f"read-back, then press Capture. Stop gripper releases it at any time.")
    steps = (
        Step(id="gripper_stage0", label="Stage 0: read the gripper", gripper="read", capture=False,
             instruction="Press 'Read gripper' (read-only, nothing is sent): position, goal, compliance, torque and "
                         "speed limits, temperature, load and the force sensor, and whether the SDK has a gripper "
                         "object with open() and close(). The force reading is the zero for the whole run."),
        head_step("gripper_view", step_id="gripper_head_aim", label="Head: aim at the hand",
                  warning="The hand should be in both eyes; if it is not, press Stop head and report."),
        *(move("open", i, t) for i, t in enumerate(GRIPPER_SWEEP_OPEN_DEG, 1)),
        *(move("shut", i, t) for i, t in enumerate(GRIPPER_SWEEP_SHUT_DEG, 1)),
        Step(id="gripper_release", label="End: release the gripper", gripper="release", capture=False,
             instruction="Press 'Release gripper' (r_gripper compliant, no confirm needed). The head stays where "
                         "it is; use 'Release head' or the next procedure's first step to move it."),
    )
    return Procedure(
        id="gripper_range", name="Gripper range (Part A)", version="1", kind="gripper",
        summary="The right gripper at REST, gripper only: Stage 0 reads it, the head aims at the hand, then an empty "
                "sweep open and shut in confirmed steps of at most 10 deg at low speed and torque, sampled at about "
                "20 Hz with a stereo pair per step. The analyser reports the range found, where -68 is reached, "
                "the force sensor's zero and drift, and any abort.",
        setup=("The arm rests on the table (REST). Only r_gripper is powered: no arm joint is ever turned on.",
               "Keep hands, cables and objects clear of the gripper and the hand. Siva stays at the robot with a "
               "hand near 'Stop gripper' and the hardware e-stop.",
               "Every move needs its own confirm tick. A temperature, load or force limit, the position not "
               "following (an end stop), a timeout or Stop releases the gripper; after an abort the run refuses "
               "further moves: start a new run.",
               "SDK 0.7.0 has no gripper.open() or close(): the bundle records that and skips them."),
        steps=steps)


PROCEDURES = {p.id: p for p in (zoom_fov_procedure(), depth_ruler_procedure(), board_view_approach_procedure(),
                                head_sweep_procedure(), sound_check_procedure(), robot_inventory_procedure(),
                                gripper_range_procedure())}


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
    sdk_inventory: dict | None = None  # Robot inventory: the SDK reads
    probes: dict[str, dict] = field(default_factory=dict)  # Robot inventory: each NUC probe's reply or why not run
    inventory_results: dict[str, list[dict]] = field(default_factory=dict)  # every inventory attempt per step
    inventory_answers: dict | None = None  # Robot inventory: the operator's answers
    redact_hosts: tuple[str, ...] = ()  # extra names the viewer's redaction pass removes (the robot host)
    gripper_commands: dict[str, list[dict]] = field(default_factory=dict)  # every gripper attempt per step
    gripper_events: list[dict] = field(default_factory=list)  # Stop, not tied to a step
    gripper_stage0: dict | None = None  # the first Stage 0 read that found the position
    gripper_aborted: str | None = None  # the outcome that ended the run's moves (an abort, not an end stop)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
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
        if step.gripper == "move" and not any(r.get("outcome") not in ("refused", "error")
                                              for r in self.gripper_commands.get(step_id, [])):
            return ("The gripper has not been moved for this step yet: tick the confirm box and press "
                    "'Move gripper' (wait for it to finish) before capturing.")
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

    def force_zero(self) -> float | None:
        """The force sensor's zero for the run: its Stage 0 reading, or None if it could not be read."""
        return None if self.gripper_stage0 is None else self.gripper_stage0.get("force")

    def apply_gripper(self, step_id: str, mover, confirmed: bool = False) -> dict:
        """Run this step's gripper operation through `mover` (`read()`, `move(target, speed, torque, force_zero)`
        and `release()`, each returning the primitives' record).

        'read' (Stage 0) and 'release' need no confirm: one only reads, the other only removes power. A 'move'
        is refused with GripperNotConfirmed without the confirm, and with GripperRefused before Stage 0 has
        read the gripper, after an abort (`gripper_aborted`: start a new run), or when the gate refuses; none
        of those sends anything. Whether a passed move is live or a logged dry run is the mover's. Every
        attempt, refused ones included, is recorded in the run (and so the bundle)."""
        step = self.procedure.step(step_id)
        if step.gripper is None:
            raise ValueError(f"step {step_id!r} does not command the gripper")
        attempts = self.gripper_commands.setdefault(step_id, [])
        if step.gripper == "read":
            record = mover.read()
            attempts.append(record)
            if self.gripper_stage0 is None and (record.get("readback") or {}).get("position_deg") is not None:
                self.gripper_stage0 = record["readback"]
            return record
        if step.gripper == "release":
            record = mover.release()
            attempts.append(record)
            return record
        refusal = {"action": "move", "target_deg": step.gripper_target_deg, "outcome": "refused", "mode": "dry-run"}
        if self.gripper_aborted:
            attempts.append({**refusal, "gate": {"passed": False, "failures": [
                f"Run aborted ({self.gripper_aborted}): start a new run"]}})
            raise GripperRefused(f"The run was aborted ({self.gripper_aborted}) and the gripper released: "
                                 "start a new run.")
        if self.gripper_stage0 is None:
            attempts.append({**refusal, "gate": {"passed": False, "failures": ["Stage 0 has not read the gripper"]}})
            raise GripperRefused("Run Stage 0 first: it reads the gripper and sets the force sensor's zero.")
        if not confirmed:
            attempts.append({**refusal, "gate": {"passed": False, "failures": ["Operator approval not given"]}})
            raise GripperNotConfirmed("A gripper move needs the operator's explicit confirm.")
        record = mover.move(step.gripper_target_deg, GRIPPER_SWEEP_SPEED_DEG_S, GRIPPER_SWEEP_TORQUE_PCT,
                            force_zero=self.force_zero())
        attempts.append(record)
        if record.get("outcome") == "refused":
            raise GripperRefused("Gripper move refused: " + "; ".join(record.get("gate", {}).get("failures", [])))
        if record.get("outcome") in GRIPPER_ABORTS:
            self.gripper_aborted = record["outcome"]
        return record

    def record_gripper_event(self, record: dict) -> None:
        """A Stop, kept in order with the run. A Stop is an abort: the run takes no further moves."""
        self.gripper_events.append(record)
        if record.get("action") == "stop":
            self.gripper_aborted = self.gripper_aborted or "stop"

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
        if self.helper_status is None:
            self.helper_status = client.status()  # the channel layout decides how many channels to record
        channels = self.layout_channels()
        data = client.record(SOUND_RECORD_S, channels, tone=tone)
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
                "requested_seconds": SOUND_RECORD_S, "requested_channels": channels, "tone": tone, **info}

    def apply_inventory(self, step_id: str, reader=None, client=None, arm_resting: bool | None = None,
                        note: str = "") -> dict:
        """Run a Robot inventory step; reads only. `reader()` returns the SDK reads (the viewer's SDK process);
        `client` (an `AudioClient`) runs the NUC probes. Every attempt is recorded in the run (and the bundle)."""
        step = self.procedure.step(step_id)
        if step.inventory is None:
            raise ValueError(f"step {step_id!r} is not a Robot inventory step")
        attempts = self.inventory_results.setdefault(step_id, [])
        if step.inventory == "sdk":
            try:
                inventory = reader()
            except Exception as exc:  # noqa: BLE001 - recorded, then raised for the route's status code
                attempts.append({"op": "sdk", "outcome": "failed", "error": str(exc)})
                raise InventoryFailed(f"SDK reads failed: {exc}") from exc
            self.sdk_inventory = inventory
            record = {"op": "sdk", "outcome": "ok", "sdk_version": inventory.get("sdk_version"),
                      "joint_count": inventory.get("joint_count"), "read_errors": len(inventory.get("errors") or []),
                      "neck_sign_check": (inventory.get("neck_sign_check") or {}).get("result")}
        elif step.inventory == "nuc":
            record = self._run_probes(client)
        else:
            self.inventory_answers = {"arm_resting": None if arm_resting is None else bool(arm_resting),
                                      "note": str(note or "").strip()[:MAX_NOTE_CHARS]}
            record = {"op": "answers", "outcome": "saved", **self.inventory_answers}
        attempts.append(record)
        return record

    def _run_probes(self, client) -> dict:
        """Every probe of INVENTORY_PROBES; an unreachable helper marks them all 'not run', never guessed."""
        probes: dict[str, dict] = {}
        try:
            self.helper_status = client.status()
            reachable, reason = True, None
        except AudioHelperError as exc:
            reachable = False
            reason = HELPER_UNREACHABLE if exc.code == "unreachable" else f"helper status failed: {exc}"
        for name in INVENTORY_PROBES:
            if not reachable:
                probes[name] = {"probe": name, "outcome": "not_run", "reason": reason}
                continue
            try:
                reply = client.probe(name)
            except AudioHelperError as exc:
                if exc.code == "unreachable":
                    reachable, reason = False, HELPER_UNREACHABLE
                    probes[name] = {"probe": name, "outcome": "not_run", "reason": reason}
                elif exc.code == "not_found":
                    probes[name] = {"probe": name, "outcome": "not_run",
                                    "reason": "the helper has no such probe: update scripts/reachy_audio_helper.py "
                                              "on the NUC"}
                else:
                    probes[name] = {"probe": name, "outcome": "failed", "reason": str(exc)}
                continue
            if reply.get("outcome") not in PROBE_OUTCOMES:
                reply = {**reply, "outcome": "failed", "reason": f"unexpected probe reply ({reply.get('outcome')})"}
            probes[name] = reply
        self.probes = probes
        outcomes = {p["outcome"] for p in probes.values()}
        overall = "ok" if outcomes == {"ok"} else "not_run" if outcomes == {"not_run"} else "partial"
        return {"op": "nuc", "outcome": overall,
                "probes": {n: {"outcome": p["outcome"], "reason": p.get("reason")} for n, p in probes.items()}}

    def inventory_count(self) -> int:
        return sum(len(v) for v in self.inventory_results.values())

    def redacted_nuc(self) -> tuple[list, int]:
        """The viewer's redaction pass over the NUC output, the helper status, the operator's answers and the
        step records: ([probes, helper_status, answers, {step: records}], replacements the helper had missed)."""
        hosts, users = workstation_names(self.redact_hosts)
        keep = {(self.helper_status or {}).get("imu_topic")}
        for command in (self.probes.get("ros2_topics") or {}).get("commands") or []:
            keep |= topic_lines(command.get("stdout"))
        before = [self.probes, self.helper_status, self.inventory_answers, self.inventory_results]
        after = redact(before, hosts, users, keep)
        return after, count_replacements(before, after)

    def channel_layout(self) -> dict:
        """The helper's channel layout ({name, channels, roles}); "plain" with 4 channels if it reports none."""
        layout = (self.helper_status or {}).get("channel_layout")
        return layout or {"name": "plain", "channels": SOUND_CHANNELS, "roles": ["mic"] * SOUND_CHANNELS}

    def layout_channels(self) -> int:
        return int(self.channel_layout()["channels"])

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
                           "gripper_commands": [slim_gripper(r) for r in self.gripper_commands.get(s.id, [])],
                           "sound_results": self.sound_results.get(s.id, []), "answer": self.answers.get(s.id),
                           "inventory_results": self.inventory_results.get(s.id, [])}
                          for s in self.procedure.steps],
                "head_events": self.head_events, "gripper_events": self.gripper_events,
                "gripper_aborted": self.gripper_aborted, "pairs": self.pair_count(), "max_pairs": MAX_PAIRS_PER_RUN,
                "can_download": bool(self.pair_count() or self.sound_count() or self.inventory_count())}

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

    def gripper_samples_csv(self) -> bytes:
        """Every sample of every move, in step order, as docs/gripper-capture-format.md section 'samples.csv'."""
        rows = []
        for step in self.procedure.steps:
            for attempt, record in enumerate(self.gripper_commands.get(step.id, []), 1):
                rows += [(step.id, attempt, sample) for sample in record.get("samples") or []]
        origin = min((sample["t"] for _, _, sample in rows), default=0.0)
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(GRIPPER_CSV_COLUMNS)
        for step_id, attempt, sample in rows:
            writer.writerow([step_id, attempt, f"{sample['t'] - origin:.3f}", sample["target_deg"],
                             sample["present_deg"], *(("" if sample[k] is None else sample[k]) for k in
                                                      ("load", "force", "temperature_c")),
                             sample["torque_limit_pct"], sample["speed_limit_deg_s"]])
        return out.getvalue().encode()

    def gripper_run_record(self, versions: dict | None) -> dict:
        """The JSON run record of docs/gripper-capture-format.md (format_version 1)."""
        versions = versions or {}
        stage0 = self.gripper_stage0
        sdk_api = (stage0 or {}).get("sdk_open_close")
        have = bool(sdk_api and sdk_api.get("gripper_object") and sdk_api.get("open") and sdk_api.get("close"))
        return {
            "format_version": GRIPPER_FORMAT_VERSION, "kind": "gripper", "units": GRIPPER_UNITS,
            "sdk_version": versions.get("sdk"), "viewer_version": versions.get("viewer"),
            "utc_time": self.created_utc, "script": "reachy_ai.perception.procedures:gripper_range_procedure",
            "script_sha256": file_sha256(Path(__file__)), "code_sha": git_sha(),
            "procedure": self.identity(),
            "stage_parameters": {
                "sweep_speed_limit_deg_s": GRIPPER_SWEEP_SPEED_DEG_S,
                "sweep_torque_limit_pct": GRIPPER_SWEEP_TORQUE_PCT,
                "sample_hz": GRIPPER_SAMPLE_HZ, "max_step_deg": GRIPPER_MAX_STEP_DEG,
                "travel_limits_deg": list(GRIPPER_LIMITS_DEG), "max_speed_limit_deg_s": GRIPPER_MAX_SPEED_DEG_S,
                "max_torque_limit_pct": GRIPPER_MAX_TORQUE_PCT,
                "abort_limits": {"temperature_c": GRIPPER_MAX_TEMP_C, "abs_load": GRIPPER_MAX_LOAD,
                                 "abs_force_minus_zero": GRIPPER_MAX_FORCE},
                "follow_tolerance_deg": config.GRIPPER_FOLLOW_TOL_DEG,
                "stall_window_s": config.GRIPPER_STALL_WINDOW_S,
                "stall_min_progress_deg": config.GRIPPER_STALL_MIN_PROGRESS_DEG,
                "step_timeout_s": config.GRIPPER_STEP_TIMEOUT_S,
                "sweep_open_deg": list(GRIPPER_SWEEP_OPEN_DEG), "sweep_shut_deg": list(GRIPPER_SWEEP_SHUT_DEG),
                "head_aim_neck_deg": head_deg("gripper_view")},
            "step0": {"stage0": stage0, "sim_check": GRIPPER_SIM_CHECK},
            "force_zero": self.force_zero(),
            "sdk_open_close": ({"outcome": "not_exercised",
                                "note": "this SDK has gripper.open() and close(), but Part A does not command them"}
                               if have else
                               {"outcome": "skipped",
                                "note": "the SDK has no gripper object with open() and close() "
                                        "(reachy_sdk 0.7.0 has joints.r_gripper only)"}),
            "aborted": self.gripper_aborted, "gripper_events": self.gripper_events,
            "files": {"samples": "gripper/samples.csv", "run_record": "gripper/run_record.json"},
        }

    def bundle_files(self, versions: dict | None = None) -> dict[str, bytes]:
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
                   "head_commands": self.head_commands.get(step.id, []),
                   "gripper": step.gripper, "gripper_target_deg": step.gripper_target_deg,
                   "gripper_commands": [slim_gripper(r) for r in self.gripper_commands.get(step.id, [])],
                   "captures": entries}
            if step.sound:
                clips = []
                for n, data in enumerate(self.clips.get(step.id, []), 1):
                    clips.append(f"steps/{index:02d}_{step.id}/take{n}.wav")
                    files[clips[-1]] = data
                row.update(sound=step.sound, sound_results=self.sound_results.get(step.id, []),
                           answer=self.answers.get(step.id), clips=clips)
            if step.inventory:
                row.update(inventory=step.inventory, inventory_results=self.inventory_results.get(step.id, []))
            steps.append(row)
        summary = {"schema_version": "2", "procedure": self.identity(), "kind": self.procedure.kind,
                   "calibration_id": self.calibration_id(), "steps": steps, "head_events": self.head_events}
        if self.procedure.kind == "zoom":
            summary["start_zoom"] = self.start_zoom()  # None: not read, unreadable or the eyes differed
        if self.procedure.kind == "gripper":
            files["gripper/samples.csv"] = self.gripper_samples_csv()
            files["gripper/run_record.json"] = json_bytes(self.gripper_run_record(versions))
            summary.update(format_version=GRIPPER_FORMAT_VERSION, gripper_aborted=self.gripper_aborted,
                           gripper_events=self.gripper_events)
        if self.procedure.kind == "sound":
            summary.update(helper_status=self.helper_status, backend=(self.helper_status or {}).get("backend"),
                           answers=self.answers, phrase=SOUND_PHRASE, tone=SOUND_TONE,
                           record={"seconds": SOUND_RECORD_S, "channels": self.layout_channels()},
                           channel_layout=self.channel_layout(),
                           consent="The operator ticked 'the speaker agreed to this recording' before each "
                                   "recording.")
        if self.procedure.kind == "inventory":
            (probes, helper_status, answers, results), missed = self.redacted_nuc()
            for name, reply in probes.items():
                files[f"probes/{name}.json"] = json_bytes(reply)
            for row in summary["steps"]:  # the step records carry probe reasons and the note too
                row["inventory_results"] = results.get(row["id"], [])
            summary.update(sdk=self.sdk_inventory, helper_status=helper_status, operator=answers,
                           probes={n: {"outcome": p["outcome"], "reason": p.get("reason"),
                                       "file": f"probes/{n}.json"} for n, p in probes.items()},
                           # replies built here (helper unreachable) carry no commands and need no helper pass
                           redaction={"helper": all(p.get("redacted") for p in probes.values() if "commands" in p),
                                      "viewer_pass": True, "viewer_replacements": missed})
        profile = self.analysis_profile()
        if profile.get("source"):
            summary["source"] = profile["source"]  # which camera geometry the analysers must use
            if profile["source"] == "sim":
                summary["sim_lens"] = profile.get("lens")
        return {"capture.json": json_bytes(summary), **files}

    def identity(self) -> dict:
        return {"id": self.procedure.id, "name": self.procedure.name, "version": self.procedure.version}

    def build_zip(self, versions: dict) -> bytes:
        if not (self.pair_count() or self.sound_count() or self.inventory_count() or self.gripper_commands):
            raise RunLimit("Nothing captured yet.")
        return build_bundle(self.bundle_files(versions), versions, self.calibration_id(), procedure=self.identity())


GRIPPER_UNITS = {"angle": "degrees (r_gripper, joint.goal_position / present_position as the SDK reports them)",
                 "load": "the SDK's present_load, unscaled", "force": "the gripper force sensor's reading, unscaled",
                 "temperature": "degrees Celsius", "torque_limit": "percent", "speed_limit": "degrees per second",
                 "time": "seconds since the first sample of the run (monotonic clock)"}


def slim_gripper(record: dict) -> dict:
    """A gripper record without its samples (they go to samples.csv), with their count."""
    slim = {k: v for k, v in record.items() if k != "samples"}
    if "samples" in record:
        slim["n_samples"] = len(record["samples"])
    if isinstance(slim.get("release"), dict):
        slim["release"] = slim_gripper(slim["release"])
    return slim


def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def git_sha() -> str | None:
    """The checkout's HEAD commit, or None when it is not a git checkout (read only: `git rev-parse`)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True,
                             text=True, timeout=3, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


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
