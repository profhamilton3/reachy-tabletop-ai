"""Zoom policy and the explicit camera-service zoom setter (#28 B).

Policy: scene capture uses the widest zoom (SCENE_ZOOM). The zoom level is recorded with
every capture and folded into its calibration_id, and a stereo profile is valid only at
the zoom it was derived at. A capture at another zoom shows a warning in the viewer.

The setter is the only place in the workstation code that commands the lens. It is a
logged camera-service call and dry-run by default, like other robot commands. It moves
no joint and does not use the motion primitives.
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


def calibration_id(base_id: str | None, zoom: str | None) -> str | None:
    """The profile id with the capture's zoom folded in, e.g. 'abc123@zoom=OUT'."""
    if base_id is None:
        return None
    return f"{base_id}@zoom={zoom or 'unknown'}"


def zoom_status(profile_zoom: str | None, readbacks: dict | None) -> dict:
    """Compare the zoom read from the cameras with the profile's and the scene policy.

    `readbacks` maps eye -> the eye's diagnostics dict (as stored by the stereo service),
    or eye -> level. Returns the captured level (None if unreadable or the eyes differ),
    `matches_profile` (None while the profile's zoom is unrecorded) and `warnings`.
    """
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
    return {"levels": levels, "captured": captured, "profile": profile, "scene_policy": SCENE_ZOOM,
            "matches_profile": matches, "warnings": warnings, "warning": " ".join(warnings) or None}


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
