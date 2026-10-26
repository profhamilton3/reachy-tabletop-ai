"""Focus measurement and the focus sweep (#77). Offline-testable: no SDK and no camera in this module.

Sharpness is measured on the board, not the whole frame (Pollen's autofocus scores the whole frame, so it finds
"clearest overall"): two numbers, the variance of the Laplacian and Pollen's Canny(50, 100) edge density, over
the board region, averaged over a few frames taken after a settle delay.

The sweep scans the focus motor over Pollen's zoom-dependent range, coarsely (steps of 10), then finely (+-10 in
steps of 2) around the best coarse value, picks the maximum, applies it and verifies. A FLAT curve (the
simulator, which renders sharp at every focus, or a motor that does not move) is reported as "no focus effect"
and the starting value is kept. The commands (`set_focus`) and the frame source (`measure`) are callables the
caller supplies; this module never sends one on its own.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import cv2
import numpy as np

COARSE_STEP = 10
FINE_SPAN = 10
FINE_STEP = 2
FRAMES_PER_MEASURE = 3
#: A curve whose (max - min) / mean is below this is flat: nothing the focus motor does shows in the image.
FLAT_RATIO = 0.08
#: The verified sharpness at the chosen value must reach this share of the sweep's own reading there.
VERIFY_SHARE = 0.8
FOCUS_LIMITS = (0, 500)
CENTRE_CROP = 0.5  # fallback region: the middle half of the width and of the height
NO_EFFECT = "no focus effect"


def pollen_range(zoom: int | None) -> tuple[int, int]:
    """Pollen's search range for a zoom position (reachy_focus/camera_focus.py), clamped to the motor's 0-500.

    min = 500 - (e^(0.01 zoom) + 25) * 5, max = 500 - (e^(zoom / 120) + 5) * 5. An unreadable zoom searches the
    whole range."""
    if zoom is None:
        return FOCUS_LIMITS
    low = 500 - (math.exp(0.01 * zoom) + 25) * 5
    high = 500 - (math.exp(zoom / 120) + 5) * 5
    low, high = sorted((int(round(min(max(low, 0), 500))), int(round(min(max(high, 0), 500)))))
    return low, high


def coarse_values(low: int, high: int, step: int = COARSE_STEP) -> list[int]:
    return list(range(low, high + 1, step)) or [low]


def fine_values(centre: int, low: int, high: int, span: int = FINE_SPAN, step: int = FINE_STEP) -> list[int]:
    """`centre` +- `span` in steps of `step`, inside the motor's range (and not the coarse points again)."""
    lo, hi = max(centre - span, FOCUS_LIMITS[0], 0), min(centre + span, FOCUS_LIMITS[1])
    return [v for v in range(lo, hi + 1, step) if v != centre]


# --- the board region and the sharpness numbers ----------------------------------------------------------

def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def centre_crop_mask(shape: tuple[int, int], share: float = CENTRE_CROP) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    y0, y1 = int(h * (1 - share) / 2), int(h * (1 + share) / 2)
    x0, x1 = int(w * (1 - share) / 2), int(w * (1 + share) / 2)
    mask[y0:y1, x0:x1] = 1
    return mask


def quads_mask(quads: Any, shape: tuple[int, int], margin_px: int = 0) -> np.ndarray | None:
    """A mask over the convex hull of 4-point quads (raw pixels): the board region. None if there are none."""
    points = [np.asarray(q, float).reshape(-1, 2) for q in (quads.values() if isinstance(quads, dict) else quads or [])]
    if not points:
        return None
    hull = cv2.convexHull(np.round(np.vstack(points)).astype(np.int32))
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    if margin_px:
        mask = cv2.dilate(mask, np.ones((2 * margin_px + 1, 2 * margin_px + 1), np.uint8))
    return mask


def grid_quads(gray: np.ndarray) -> list | None:
    """The taped board's cell quads from the head-sweep grid finder (raw pixels), or None (a blurry board often
    cannot be found, which is why the caller falls back to a centre crop)."""
    try:
        from reachy_ai.games.t3_board_reader import dsc as grid  # noqa: PLC0415 - puts scripts/ on sys.path
    except ImportError:
        return None
    try:
        quads, partial = grid.detect(gray)
        labelled = grid.pick_grid(quads, partial, gray.shape)
    except Exception:  # noqa: BLE001 - any finder failure means "use the fallback"
        return None
    return None if labelled is None else [quads[i][1] for i in labelled.values()]


def board_region(gray: np.ndarray, quads: Any = None, find_grid: Callable[[np.ndarray], Any] = grid_quads,
                 margin_px: int = 12) -> tuple[np.ndarray, str]:
    """(mask, source): the T3 reference quads if given, else the grid finder's, else a centre crop."""
    mask = quads_mask(quads, gray.shape, margin_px)
    if mask is not None and mask.any():
        return mask, "t3_reference"
    found = find_grid(gray)
    mask = quads_mask(found, gray.shape, margin_px)
    if mask is not None and mask.any():
        return mask, "grid"
    return centre_crop_mask(gray.shape), "centre_crop"


def sharpness(image: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """{'laplacian_var', 'canny_density'} of `image` over `mask` (the whole frame without one).

    Variance of the Laplacian of the grey image; Canny(50, 100) edge pixels over the region's pixels."""
    gray = to_gray(image)
    if mask is None:
        mask = np.ones(gray.shape[:2], np.uint8)
    inside = mask.astype(bool)
    pixels = int(inside.sum())
    if pixels == 0:
        raise ValueError("the sharpness region is empty")
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.Canny(gray, 50, 100)
    return {"laplacian_var": float(laplacian[inside].var()), "canny_density": float((edges[inside] > 0).mean())}


def average(readings: list[dict]) -> dict:
    """The mean of each sharpness number over the frames, plus how many frames."""
    if not readings:
        raise ValueError("no frames to average")
    return {"laplacian_var": float(np.mean([r["laplacian_var"] for r in readings])),
            "canny_density": float(np.mean([r["canny_density"] for r in readings])), "frames": len(readings)}


def measure_frames(frames: list[np.ndarray], mask: np.ndarray) -> dict:
    return average([sharpness(f, mask) for f in frames])


# --- the sweep -------------------------------------------------------------------------------------------

def is_flat(values: list[float], ratio: float = FLAT_RATIO) -> bool:
    """True if the curve's spread is below `ratio` of its mean: no focus effect (or a dead curve)."""
    if len(values) < 2:
        return True
    mean = float(np.mean(values))
    return mean <= 0 or (max(values) - min(values)) / mean < ratio


def run_sweep(eye: str, start_focus: int | None, zoom: int | None, set_focus: Callable[[int], dict],
              measure: Callable[[], dict], dry_run: bool = False, metric: str = "laplacian_var",
              flat_ratio: float = FLAT_RATIO) -> dict:
    """One eye's focus sweep.

    `set_focus(value)` sends one logged focus command and returns its record (`ok`, `after`...); `measure()`
    returns the averaged sharpness at the lens's present state (after the caller's settle delay). The result:

    - `outcome`: 'applied' (the maximum, applied and verified), 'no_focus_effect' (flat curve: the starting value
      is kept), 'dry_run' (nothing was sent: the plan only), 'command_failed' (a command did not take; the
      starting value is restored) or 'unmeasurable' (the measure failed).
    - `curve`: every point as {phase, focus, laplacian_var, canny_density}; `chosen`: the value kept.
    """
    low, high = pollen_range(zoom)
    plan = coarse_values(low, high)
    result: dict[str, Any] = {"eye": eye, "zoom": zoom, "start_focus": start_focus, "range": [low, high],
                              "coarse_step": COARSE_STEP, "fine_span": FINE_SPAN, "fine_step": FINE_STEP,
                              "metric": metric, "curve": [], "commands": 0, "chosen": start_focus,
                              "outcome": None, "note": None}
    if dry_run:
        result.update(outcome="dry_run", planned_coarse=plan,
                      planned_commands=len(plan) + 2 * FINE_SPAN // FINE_STEP + 2,
                      note="dry-run: no focus command was sent, so no curve was taken")
        try:
            result["before"] = measure()
        except Exception as exc:  # noqa: BLE001 - recorded by type
            result["note"] += f"; the sharpness could not be measured ({type(exc).__name__})"
        return result

    def point(phase: str, value: int) -> dict | None:
        command = set_focus(value)
        result["commands"] += 1
        if not command.get("ok"):
            result["failed_command"] = {k: command.get(k) for k in ("action", "target", "error", "after")}
            return None
        reading = measure()
        row = {"phase": phase, "focus": value, "laplacian_var": reading["laplacian_var"],
               "canny_density": reading["canny_density"]}
        result["curve"].append(row)
        return row

    def restore(note: str, outcome: str):
        result.update(outcome=outcome, note=note)
        if start_focus is not None:
            command = set_focus(start_focus)
            result["commands"] += 1
            result["chosen"] = start_focus
            result["restored"] = {"ok": bool(command.get("ok")), "after": command.get("after")}

    try:
        for value in plan:
            if point("coarse", value) is None:
                return restore("a focus command did not take (read-back mismatch or error): the starting focus was "
                               "restored", "command_failed") or result
        best = max(result["curve"], key=lambda r: r[metric])
        if is_flat([r[metric] for r in result["curve"]], flat_ratio):
            restore(f"{NO_EFFECT}: the sharpness varies by under {flat_ratio:.0%} over the range {low}-{high}; "
                    "the starting focus was kept (the simulator, or a lens that does not respond)",
                    "no_focus_effect")
            return result
        for value in fine_values(best["focus"], low, high):
            if point("fine", value) is None:
                return restore("a focus command did not take during the fine scan: the starting focus was "
                               "restored", "command_failed") or result
        best = max(result["curve"], key=lambda r: r[metric])
        result["peak_at_range_edge"] = best["focus"] in (low, high)
        command = set_focus(best["focus"])
        result["commands"] += 1
        if not command.get("ok"):
            return restore("the chosen focus could not be applied: the starting focus was restored",
                           "command_failed") or result
        verify = measure()
        result.update(outcome="applied", chosen=best["focus"], applied={"ok": True, "after": command.get("after")},
                      verified=verify, peak=best)
        result["verify_ok"] = verify[metric] >= VERIFY_SHARE * best[metric]
        result["note"] = ("the sharpest value is at the edge of Pollen's range: the true peak may lie outside it"
                          if result["peak_at_range_edge"] else None)
        if not result["verify_ok"]:
            result["note"] = ((result["note"] + "; ") if result["note"] else "") + (
                f"the verification reading ({verify[metric]:.1f}) is under {VERIFY_SHARE:.0%} of the sweep's reading "
                f"at {best['focus']} ({best[metric]:.1f}): the scene or the lighting changed during the sweep")
    except Exception as exc:  # noqa: BLE001 - recorded by type; the starting focus is put back
        try:
            restore(f"the sweep stopped ({type(exc).__name__}): the starting focus was restored", "unmeasurable")
        except Exception:  # noqa: BLE001
            result["note"] += " (and the restore failed: set the focus back by hand)"
    return result
