"""A fake lens with a synthetic sharpness model, for the Focus check tests (#77). No SDK, no camera."""
import math
import random

EYES = ("left", "right")


def peaked(left=332, right=322, width=25.0, floor=50.0, height=1000.0):
    """Sharpest at `left` / `right` (raw focus), falling off like a Gaussian."""
    peaks = {"left": left, "right": right}
    return lambda eye, focus, zoom: floor + height * math.exp(-(((focus - peaks[eye]) / width) ** 2))


def flat(level=200.0):
    return lambda eye, focus, zoom: level


class FakeLens:
    """The lens facade `CaptureRun.apply_focus` takes: every call is recorded in `calls`.

    `model(eye, focus, zoom)` is the Laplacian variance the camera would show. `noise` is added per measure
    (a seeded Gaussian). `dry_run` sends nothing (as the viewer without --allow-zoom-commands). `stuck` ignores
    focus and zoom commands (a dead motor); `fail_level` refuses a zoom level command."""

    def __init__(self, model=None, state=None, dry_run=False, noise=0.0, stuck=False, fail_level=False, seed=1):
        self.model = model or peaked()
        self.state = state or {"left": {"zoom_level": "INTER", "zoom": 270, "focus": 331},
                               "right": {"zoom_level": "INTER", "zoom": 270, "focus": 321}}
        self.state = {eye: dict(v) for eye, v in self.state.items()}
        self.dry_run, self.noise, self.stuck, self.fail_level = dry_run, noise, stuck, fail_level
        self.calls, self.rng, self.regions_reset = [], random.Random(seed), 0

    def cached(self):
        return {eye: dict(v) for eye, v in self.state.items()}

    def read(self):
        self.calls.append(("read",))
        return self.cached()

    def reset_regions(self):
        self.regions_reset += 1

    def measure(self, eye):
        self.calls.append(("measure", eye))
        s = self.state[eye]
        value = self.model(eye, s["focus"], s["zoom"]) + self.rng.gauss(0.0, self.noise)
        return {"laplacian_var": max(value, 0.0), "canny_density": max(value, 0.0) / 10000.0, "frames": 3,
                "region": "centre_crop"}

    def _record(self, eye, action, target, field):
        before = dict(self.state[eye])
        record = {"eye": eye, "action": action, "target": target, "before": before, "dry_run": self.dry_run,
                  "sent": False, "ok": False}
        if self.dry_run:
            record["after"] = before
            return record
        if field and not self.stuck:
            self.state[eye][field] = target
            if field == "zoom":  # a raw zoom on a preset's position reads as that level
                self.state[eye]["zoom_level"] = {60: "OUT", 270: "INTER", 457: "IN"}.get(target, "ZERO")
        record.update(sent=True, after=dict(self.state[eye]), ok=True if field is None else
                      self.state[eye][field] == target)
        return record

    def set_focus(self, eye, value):
        self.calls.append(("set_focus", eye, value))
        return self._record(eye, "set_focus", value, "focus")

    def set_zoom(self, eye, value):
        self.calls.append(("set_zoom", eye, value))
        return self._record(eye, "set_zoom", value, "zoom")

    def homing(self, eye):
        self.calls.append(("homing", eye))
        record = self._record(eye, "homing", None, None)
        if not self.dry_run:
            self.state[eye].update(zoom=0, focus=0, zoom_level="ZERO")  # a homed lens reads its origin
            record["after"] = dict(self.state[eye])
        return record

    def autofocus(self, eye):
        self.calls.append(("autofocus", eye))
        record = self._record(eye, "autofocus", None, None)
        if not self.dry_run:
            self.state[eye]["focus"] = 330  # Pollen's pick
            record["after"] = dict(self.state[eye])
        return record

    def set_level(self, level):
        self.calls.append(("set_level", level))
        presets = {"OUT": (60, 455), "INTER": (270, 331), "IN": (457, 70)}
        records = []
        for eye in EYES:
            ok = not self.dry_run and not self.fail_level and level in presets
            if ok:
                self.state[eye].update(zoom_level=level, zoom=presets[level][0], focus=presets[level][1])
            records.append({"eye": eye, "requested": level, "dry_run": self.dry_run, "ok": ok,
                            "after": self.state[eye]["zoom_level"]})
        return records
