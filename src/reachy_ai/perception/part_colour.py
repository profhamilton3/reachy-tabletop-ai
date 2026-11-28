"""A part's observed colour (#84, #81; owner 2026-11-27): its own measured attribute, never an input to geometry.

CIELAB (D65) median over the object mask with clipped (highlight) pixels left out, a colour name with a confidence,
and a `reliable` flag. White balance is taken from a reference in view: the unclipped white of the board. If that
white is not there, or the lighting check fails, the colour is still recorded but `reliable` is false.

Pure arrays in, numbers out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

#: A channel at or above this is clipped: the pixel's colour is not known.
CLIPPED = 253
#: The white reference.
WHITE_MIN_LEVEL = 110          # darkest channel of a white-board pixel
WHITE_MAX_SPREAD = 28          # largest minus smallest channel: anything more is not the white board
WHITE_MIN_PIXELS = 400
#: The lighting check on the reference: its linear luminance, and how far from neutral it may be (CIELAB chroma).
WHITE_LUMINANCE_RANGE = (0.18, 0.93)
WHITE_MAX_CHROMA = 12.0
#: Colour is unreliable if more than this share of the object's pixels is clipped, or if fewer than this are left.
MAX_CLIPPED_SHARE = 0.5      # beyond this what is left is mostly the darker, less lit faces
MIN_OBJECT_PIXELS = 60

#: Naming limits (CIELAB). Generic limits, not fitted to the simulator's family colours.
BLACK_MAX_L = 22.0
BLACK_MAX_CHROMA = 18.0        # a dark but saturated colour is not black
NEUTRAL_MAX_CHROMA = 11.0
WHITE_MIN_L = 75.0
BROWN_MAX_L = 35.0
#: (upper hue angle, name), degrees
HUES = ((40.0, "red"), (75.0, "orange"), (110.0, "yellow"), (200.0, "green"), (305.0, "blue"), (345.0, "purple"),
        (360.0, "red"))

_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])
_D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_lab(linear: np.ndarray) -> np.ndarray:
    """Linear RGB (n, 3) -> CIELAB (n, 3), D65."""
    t = (np.asarray(linear, float) @ _TO_XYZ.T) / _D65
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def name_of(lab: np.ndarray) -> np.ndarray:
    """The colour name of each CIELAB value (n, 3)."""
    lab = np.atleast_2d(lab)
    lightness, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    names = np.full(len(lab), "other", dtype=object)
    lower = 0.0
    for upper, name in HUES:
        names[(hue >= lower) & (hue < upper)] = name
        lower = upper
    names[(hue >= 20) & (hue < 90) & (lightness < BROWN_MAX_L)] = "brown"
    neutral = chroma < NEUTRAL_MAX_CHROMA
    names[neutral] = "grey"
    names[neutral & (lightness > WHITE_MIN_L)] = "white"
    names[(lightness < BLACK_MAX_L) & (chroma < BLACK_MAX_CHROMA)] = "black"
    return names


@dataclass(frozen=True)
class ColourReading:
    name: str
    lab_median: tuple          # after white balance (as measured, if there is no reference)
    lab_raw: tuple             # as rendered / photographed, before white balance
    lab_raw_all: tuple         # the same with the clipped pixels kept (comparable with a rendered reference)
    confidence: float          # share of the object's pixels with the median's name
    reliable: bool
    reasons: tuple             # why it is not reliable
    pixels: int
    clipped_share: float
    reference: dict

    def to_dict(self) -> dict:
        return {"name": self.name, "lab_median": list(self.lab_median), "lab_raw": list(self.lab_raw),
                "lab_raw_all": list(self.lab_raw_all),
                "confidence": self.confidence, "reliable": self.reliable, "reasons": list(self.reasons),
                "pixels": self.pixels, "clipped_share": self.clipped_share, "reference": self.reference}


def _rgb(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[mask]


def white_pixels(image: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """The unclipped, near-neutral, bright pixels among `candidates` (the board's white, away from the part), RGB."""
    rgb = _rgb(image, candidates).astype(int)
    keep = ((rgb.min(axis=1) >= WHITE_MIN_LEVEL) & (rgb.max(axis=1) < CLIPPED)
            & (rgb.max(axis=1) - rgb.min(axis=1) <= WHITE_MAX_SPREAD))
    return rgb[keep]


def observed_colour(samples) -> ColourReading | None:
    """The colour over several images. `samples` is a list of (BGR image, object mask, white-candidate mask).
    None if no object pixel is left."""
    object_rgb, white_rgb = [], []
    for image, mask, candidates in samples:
        object_rgb.append(_rgb(image, mask))
        white_rgb.append(white_pixels(image, candidates))
    pixels = np.concatenate(object_rgb) if object_rgb else np.zeros((0, 3))
    if not len(pixels):
        return None
    clipped = (pixels >= CLIPPED).any(axis=1)
    kept = pixels[~clipped]
    share = float(clipped.mean())
    reasons = []
    if len(kept) < MIN_OBJECT_PIXELS:
        reasons.append(f"only {len(kept)} unclipped object pixels (at least {MIN_OBJECT_PIXELS})")
        if not len(kept):
            kept = pixels
    if share > MAX_CLIPPED_SHARE:
        reasons.append(f"{share:.0%} of the object's pixels are clipped")
    linear = srgb_to_linear(kept)
    raw = np.median(linear_to_lab(linear), axis=0)
    raw_all = np.median(linear_to_lab(srgb_to_linear(pixels)), axis=0)

    white = np.concatenate(white_rgb) if white_rgb else np.zeros((0, 3))
    reference = {"kind": "board_white", "pixels": int(len(white))}
    gains = np.ones(3)
    if len(white) < WHITE_MIN_PIXELS:
        reasons.append(f"white reference not visible: {len(white)} unclipped white pixels "
                       f"(at least {WHITE_MIN_PIXELS})")
    else:
        level = np.median(srgb_to_linear(white), axis=0)
        luminance = float(level @ _TO_XYZ[1])
        lab = linear_to_lab(level[None])[0]
        chroma = float(math.hypot(lab[1], lab[2]))
        reference.update(median_srgb=[int(v) for v in np.median(white, axis=0)], luminance=round(luminance, 3),
                         chroma=round(chroma, 1))
        if not WHITE_LUMINANCE_RANGE[0] <= luminance <= WHITE_LUMINANCE_RANGE[1]:
            reasons.append(f"lighting check failed: the white reference's luminance {luminance:.2f} is outside "
                           f"{WHITE_LUMINANCE_RANGE[0]}-{WHITE_LUMINANCE_RANGE[1]}")
        elif chroma > WHITE_MAX_CHROMA:
            reasons.append(f"lighting check failed: the white reference is not neutral (chroma {chroma:.0f})")
        else:
            gains = luminance / np.maximum(level, 1e-6)
            reference["gains"] = [round(float(g), 3) for g in gains]
    balanced = linear_to_lab(np.clip(linear * gains, 0.0, 1.0))
    median = np.median(balanced, axis=0)
    median[0] = min(max(median[0], 0.0), 100.0)
    name = str(name_of(median)[0])
    confidence = float((name_of(balanced) == name).mean())

    def triple(lab):
        return tuple(round(float(v), 2) for v in lab)

    return ColourReading(name=name, lab_median=triple(median), lab_raw=triple(raw), lab_raw_all=triple(raw_all),
                         confidence=round(confidence, 3), reliable=not reasons, reasons=tuple(reasons),
                         pixels=int(len(kept)), clipped_share=round(share, 3), reference=reference)
