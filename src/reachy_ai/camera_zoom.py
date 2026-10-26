"""Zoom policy and the explicit camera-service zoom setter (#28 B).

Policy: scene capture uses the widest zoom (SCENE_ZOOM). The zoom level is recorded with
every capture and folded into its calibration_id, and a stereo profile is valid only at
the zoom it was derived at. A capture at another zoom shows a warning in the viewer.

The setter and the lens commander are the only places in the workstation code that command the lens.
Each is a logged camera-service call and dry-run by default, like other robot commands. They move
no joint and do not use the motion primitives. The commander (#77) reads and sets the raw `zoom` and
`focus` integers, runs `zoom_homing()` and Pollen's autofocus; it is only ever driven one command at a
time by a data-capture procedure after the operator's confirm, never from a per-frame loop.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

LOG = logging.getLogger(__name__)

#: Zoom levels of the v1 SDK (`reachy_sdk.camera.ZoomLevel`), widest field of view first.
ZOOM_ORDER = ("OUT", "INTER", "IN")
#: ZERO is the SDK's unset level (the sim starts there and treats it as INTER; what it means on the
#: robot is unconfirmed). It is a real read-back value, recorded as "ZERO" and never mapped to another
#: level. It is not a policy level: only a restore to the starting level ever tries to command it.
ZOOM_READBACK = ("ZERO", *ZOOM_ORDER)
#: Widest zoom, used for scene capture. That OUT is the widest is the SDK naming, not yet
#: measured on this robot: scripts/zoom_fov_check.py reports the cell pitch per level.
SCENE_ZOOM = "OUT"
EYES = ("left", "right")


def normalize_zoom(value: Any) -> str | None:
    """'ZERO' / 'OUT' / 'INTER' / 'IN' from an SDK enum or a string; None if unknown or unreadable."""
    name = str(getattr(value, "name", value) or "").strip().upper()
    return name if name in ZOOM_READBACK else None


def _raw_int(value: Any) -> int | None:
    """A lens motor position as an int, or None if unreadable (bool and NaN are not positions)."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(getattr(value, "name", value))
    except (TypeError, ValueError):
        return None
    return int(number) if number == number and abs(number) != float("inf") and number == int(number) else None


def focus_text(focus: dict | None) -> str:
    """The two eyes' raw focus as 'L331/R321' ('L?' or 'R?' for an unreadable eye)."""
    focus = focus or {}
    return "/".join(f"{eye[0].upper()}{'?' if focus.get(eye) is None else focus[eye]}" for eye in EYES)


def calibration_id(base_id: str | None, zoom: str | None, focus: dict | None = None) -> str | None:
    """The profile id with the capture's lens state folded in, e.g. 'abc123@zoom=OUT'.

    With `focus` (eye -> raw focus int, None if unreadable; #77) the id also carries it, e.g.
    'abc123@zoom=INTER,focus=L331/R321', so a refocused camera is never mistaken for the calibrated one."""
    if base_id is None:
        return None
    suffix = f"@zoom={zoom or 'unknown'}"
    if focus is not None:
        suffix += f",focus={focus_text(focus)}" if any(focus.get(e) is not None for e in EYES) else ",focus=unknown"
    return base_id + suffix


def lens_readings(readbacks: dict | None) -> dict:
    """eye -> {'zoom_level', 'zoom', 'focus'} from the stereo service's diagnostics (raw ints, None if unread)."""
    out = {}
    for eye in EYES:
        entry = (readbacks or {}).get(eye)
        entry = entry if isinstance(entry, dict) else {}
        out[eye] = {"zoom_level": normalize_zoom(entry.get("zoom_level")), "zoom": _raw_int(entry.get("zoom")),
                    "focus": _raw_int(entry.get("focus"))}
    return out


def focus_by_eye(readbacks: dict | None) -> dict:
    return {eye: r["focus"] for eye, r in lens_readings(readbacks).items()}


def zoom_status(profile_zoom: str | None, readbacks: dict | None, profile_lens_state: dict | None = None) -> dict:
    """Compare the zoom read from the cameras with the profile's and the scene policy.

    `readbacks` maps eye -> the eye's diagnostics dict (as stored by the stereo service),
    or eye -> level. Returns the captured level (None if unreadable or the eyes differ),
    `matches_profile` (None while the profile's zoom is unrecorded) and `warnings`.

    #77: when the readbacks carry the raw focus, the status also reports it and warns "unverified at this lens
    state" unless it equals the focus in `profile_lens_state` ({eye: {"focus": int}}, or None while the profile
    records none, as today)."""
    levels = {}
    for eye in EYES:
        entry = (readbacks or {}).get(eye)
        raw = entry.get("zoom_level") if isinstance(entry, dict) else entry
        levels[eye] = normalize_zoom(raw)
    profile = normalize_zoom(profile_zoom)
    captured = levels["left"] if levels["left"] == levels["right"] else None
    warnings = []
    if None in levels.values():
        warnings.append("Zoom level could not be read from "
                        + " and ".join(e for e in EYES if levels[e] is None)
                        + "; the profile's validity at this zoom cannot be confirmed.")
    elif captured is None:
        warnings.append(f"Left and right zoom differ ({levels['left']} vs {levels['right']}); "
                        "stereo geometry does not hold.")
    matches = None
    if captured is not None:
        if profile is None:
            warnings.append("The profile's zoom was never recorded (August pictures); this capture's zoom "
                            f"({captured}) cannot be confirmed to match it. Re-derive the profile at {SCENE_ZOOM}.")
        else:
            matches = captured == profile
            if not matches:
                warnings.append(f"Capture zoom {captured} differs from the profile's zoom {profile}: depth is "
                                "not valid. Re-derive the profile at this zoom or set the zoom back.")
        if captured != SCENE_ZOOM:
            warnings.append(f"Scene capture policy is the widest zoom ({SCENE_ZOOM}); this capture is at {captured}.")
    focus, focus_matches = None, None
    if any(isinstance((readbacks or {}).get(eye), dict) and "focus" in readbacks[eye] for eye in EYES):
        focus = focus_by_eye(readbacks)
        recorded = {eye: _raw_int(((profile_lens_state or {}).get(eye) or {}).get("focus")) for eye in EYES}
        now = focus_text(focus)
        if None in focus.values():
            warnings.append(f"Depth is unverified at this lens state: the focus could not be read ({now}).")
        elif None in recorded.values():
            warnings.append(f"Depth is unverified at this lens state (focus {now}): the profile records no lens "
                            "state. Run Focus check, then re-derive the profile with the lens state recorded.")
        else:
            focus_matches = focus == recorded
            if not focus_matches:
                warnings.append(f"Depth is unverified at this lens state: focus {now} differs from the profile's "
                                f"({focus_text(recorded)}).")
    return {"levels": levels, "captured": captured, "profile": profile, "scene_policy": SCENE_ZOOM,
            "matches_profile": matches, "focus": focus, "focus_matches_profile": focus_matches,
            "warnings": warnings, "warning": " ".join(warnings) or None}


def zoom_read_warning(profile_zoom: str | None, levels: dict | None) -> str | None:
    """Warning for a read-only zoom check (the Depth ruler test's setup step), or None if all is well.

    `levels` maps eye -> level as read back. Compares with the level the calibration profile is valid at."""
    read = {eye: normalize_zoom((levels or {}).get(eye)) for eye in EYES}
    if None in read.values():
        return ("Zoom level could not be read from " + " and ".join(e for e in EYES if read[e] is None)
                + "; the profile's validity at this zoom cannot be confirmed.")
    if read["left"] != read["right"]:
        return f"Left and right zoom differ ({read['left']} vs {read['right']}); stereo geometry does not hold."
    level, profile = read["left"], normalize_zoom(profile_zoom)
    if profile is None:
        return (f"The calibration profile's zoom is unknown, so this test's result applies only at the zoom "
                f"recorded in the bundle ({level}).")
    if level != profile:
        return (f"The cameras read {level}, but the calibration profile is valid at {profile}: depth is not "
                "valid at this zoom. Set the zoom back or re-derive the profile.")
    return None


class ZoomSetter:
    """Explicit, logged zoom commands on the camera service; dry-run unless told otherwise.

    `client` is a connected `ReachySDK` (anything with `left_camera` / `right_camera`).
    `resolve` turns a level name into the value the camera's `zoom_level` setter takes;
    the default imports the SDK's `ZoomLevel` lazily so tests can use a fake client.
    """

    def __init__(self, client, dry_run: bool = True, settle_s: float = 3.0,
                 resolve: Callable[[str], Any] | None = None, sleep: Callable[[float], None] = time.sleep):
        self.client, self.dry_run, self.settle_s = client, dry_run, settle_s
        self._resolve = resolve or self._sdk_level
        self._sleep = sleep

    @staticmethod
    def _sdk_level(name: str):
        from reachy_sdk.camera import ZoomLevel
        return ZoomLevel[name]

    def read(self, eye: str) -> str | None:
        return normalize_zoom(getattr(self._camera(eye), "zoom_level"))

    def read_both(self) -> list[dict]:
        """Read-back only, no command: one {"eye", "level"} per eye (level None if unreadable)."""
        records = []
        for eye in EYES:
            try:
                level = self.read(eye)
            except Exception as exc:  # noqa: BLE001 - report the failure type, not the endpoint
                LOG.warning("zoom %s read failed: %s", eye, type(exc).__name__)
                level = None
            records.append({"eye": eye, "level": level})
        return records

    def _camera(self, eye: str):
        if eye not in EYES:
            raise ValueError(f"eye must be one of {EYES}, not {eye!r}")
        camera = getattr(self.client, f"{eye}_camera", None)
        if camera is None:
            raise RuntimeError(f"{eye} camera unavailable")
        return camera

    def set_level(self, eye: str, level: str) -> dict:
        """Set one eye's zoom and read it back. In dry-run only logs what it would do."""
        level = str(level).upper()
        if level not in ZOOM_READBACK:
            raise ValueError(f"zoom level must be one of {ZOOM_READBACK}, not {level!r}")
        camera = self._camera(eye)
        before = normalize_zoom(camera.zoom_level)
        record = {"eye": eye, "requested": level, "before": before, "dry_run": self.dry_run,
                  "after": before, "ok": before == level}
        if self.dry_run:
            LOG.info("DRY-RUN zoom %s: %s -> %s (not sent)", eye, before, level)
            return record
        LOG.info("ZOOM COMMAND %s: %s -> %s", eye, before, level)
        try:
            camera.zoom_level = self._resolve(level)
        except Exception as exc:  # noqa: BLE001 - report the failure type, not the endpoint
            LOG.warning("zoom %s failed: %s", eye, type(exc).__name__)
            record.update(ok=False, error=type(exc).__name__)
            return record
        self._sleep(self.settle_s)
        record["after"] = normalize_zoom(camera.zoom_level)
        record["ok"] = record["after"] == level
        LOG.info("zoom %s read back: %s (%s)", eye, record["after"], "ok" if record["ok"] else "MISMATCH")
        return record

    def set_both(self, level: str) -> list[dict]:
        """Both eyes to the same level (stereo needs them equal)."""
        return [self.set_level(eye, level) for eye in EYES]


#: Raw lens motor ranges (Kurokesu controller, #77): zoom 0-600, focus 0-500.
ZOOM_RAW_RANGE = (0, 600)
FOCUS_RAW_RANGE = (0, 500)


class LensCommander:
    """Raw lens commands on the camera service (#77); dry-run unless told otherwise. Never in a per-frame loop.

    `read` is always allowed. `set_focus`, `set_zoom`, `homing` and the autofocus calls send a logged command
    and verify by read-back; in dry-run they only log and report `sent: False`. `client` is a connected
    `ReachySDK` (anything with `left_camera` / `right_camera`)."""

    def __init__(self, client, dry_run: bool = True, settle_s: float = 1.0, homing_wait_s: float = 20.0,
                 autofocus_s: float = 20.0, sleep: Callable[[float], None] = time.sleep):
        self.client, self.dry_run, self.settle_s = client, dry_run, settle_s
        self.homing_wait_s, self.autofocus_s, self._sleep = homing_wait_s, autofocus_s, sleep

    def _camera(self, eye: str):
        if eye not in EYES:
            raise ValueError(f"eye must be one of {EYES}, not {eye!r}")
        camera = getattr(self.client, f"{eye}_camera", None)
        if camera is None:
            raise RuntimeError(f"{eye} camera unavailable")
        return camera

    def read_eye(self, eye: str) -> dict:
        """{'zoom_level', 'zoom', 'focus'} for one eye; an unreadable field is None with a '<field>_error' type."""
        camera, values = self._camera(eye), {}
        for name in ("zoom_level", "zoom", "focus"):
            try:
                raw = getattr(camera, name)
            except Exception as exc:  # noqa: BLE001 - report the failure type, not the endpoint
                LOG.warning("lens %s %s read failed: %s", eye, name, type(exc).__name__)
                values[name], values[f"{name}_error"] = None, type(exc).__name__
                continue
            values[name] = normalize_zoom(raw) if name == "zoom_level" else _raw_int(raw)
        return values

    def read(self) -> dict:
        return {eye: self.read_eye(eye) for eye in EYES}

    def _command(self, eye: str, action: str, target: Any, send: Callable[[Any], None], field: str | None,
                 wait_s: float | None = None) -> dict:
        camera = self._camera(eye)
        before = self.read_eye(eye)
        record = {"eye": eye, "action": action, "target": target, "before": before, "after": before,
                  "dry_run": self.dry_run, "sent": False, "ok": False}
        if self.dry_run:
            LOG.info("DRY-RUN lens %s %s: %s (not sent)", eye, action, target)
            return record
        LOG.info("LENS COMMAND %s %s: %s (was %s)", eye, action, target,
                 before.get(field) if field else before)
        try:
            send(camera)
        except Exception as exc:  # noqa: BLE001 - report the failure type, not the endpoint
            LOG.warning("lens %s %s failed: %s", eye, action, type(exc).__name__)
            record.update(error=type(exc).__name__)
            return record
        record["sent"] = True
        self._sleep(self.settle_s if wait_s is None else wait_s)
        record["after"] = self.read_eye(eye)
        record["ok"] = True if field is None else record["after"].get(field) == target
        LOG.info("lens %s %s read back: %s (%s)", eye, action, record["after"], "ok" if record["ok"] else "MISMATCH")
        return record

    @staticmethod
    def _checked(name: str, value: Any, limits: tuple[int, int]) -> int:
        number = _raw_int(value)
        if number is None or not limits[0] <= number <= limits[1]:
            raise ValueError(f"{name} must be an integer in {limits[0]}-{limits[1]}, not {value!r}")
        return number

    def set_focus(self, eye: str, value: Any) -> dict:
        target = self._checked("focus", value, FOCUS_RAW_RANGE)
        return self._command(eye, "set_focus", target, lambda cam: setattr(cam, "focus", target), "focus")

    def set_zoom(self, eye: str, value: Any) -> dict:
        target = self._checked("zoom", value, ZOOM_RAW_RANGE)
        return self._command(eye, "set_zoom", target, lambda cam: setattr(cam, "zoom", target), "zoom")

    def homing(self, eye: str) -> dict:
        """`zoom_homing()`: the lens runs to its end stops and counts from there. Positions are relative to it."""
        return self._command(eye, "homing", None, lambda cam: cam.zoom_homing(), None, wait_s=self.homing_wait_s)

    def autofocus(self, eye: str) -> dict:
        """Pollen's `start_autofocus()`, left to run for `autofocus_s`, then `stop_autofocus()` (always)."""
        def run(camera):
            camera.start_autofocus()
            try:
                self._sleep(self.autofocus_s)
            finally:
                camera.stop_autofocus()
        return self._command(eye, "autofocus", None, run, None, wait_s=self.settle_s)
