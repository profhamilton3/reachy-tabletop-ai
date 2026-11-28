"""Is there an opening in the part's top face, and how wide? (#84)

The fitted solid gives the top face's place in space, so each image of it is rectified to millimetres on that face.
A bore's rim is then a circle, whatever the view. The test is the share of a circle's circumference with an edge on
it AND the face just outside it. A dim bore (a dark disc) passes, and so does a bore the lit board shows through
(bright, which a "dark ellipse" test would miss). A straight shadow edge or a seam across the face covers only a few
degrees of any circle.

TOP FACE ONLY: whether a hole goes through cannot be told from one view.

No colour: the edge strength is the largest over the three channels, so reordering them changes nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

RESOLUTION_M = 0.0004            # one rectified pixel
CROP_HALF_M = 0.036
FACE_REFERENCE_INSET_M = 0.004   # the face's own look is sampled on a ring this far inside its outline
FACE_INSET_M = 0.0025            # edges this near the face's own outline are the outline, not an opening
RADII_M = np.arange(0.006, 0.0212, 0.0004)
CENTRE_REACH_M = 0.006
CENTRE_STEP_M = 0.0008
ANGLES = np.linspace(0.0, 2 * math.pi, 90, endpoint=False)
EDGE_HALF_WIDTH_M = 0.0010       # the inside and outside samples sit this far either side of the circle
EDGE_MIN_LEVELS = 4.0            # gray levels (largest channel) across the circle that count as an edge there
EDGE_NOISE_FACTOR = 3.0
PRESENT_SUPPORT = 0.70           # this share of the circumference with an edge: an opening
ABSENT_SUPPORT = 0.45            # less than this for every circle: a clear face
CLIPPED_FACE_LIMIT = 0.97        # a face this saturated shows nothing
MIN_FACE_PIXELS_PER_MM = 0.9     # raw pixels per millimetre of face, below which a 15 mm opening is not resolved
DIAMETER_FLOOR_BOUND_MM = 2.0


@dataclass(frozen=True)
class TopFace:
    """The top face of a fitted upright solid, in the board frame."""
    x: float
    y: float
    height: float
    yaw: float
    half_a: float           # half-size along the face's own x (a disc: the radius)
    half_b: float
    round: bool

    def board_points(self, local: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        xy = local @ np.array([[c, s], [-s, c]]) + [self.x, self.y]
        return np.column_stack([xy, np.full(len(xy), self.height)])

    def inside(self, local: np.ndarray, inset: float) -> np.ndarray:
        if self.round:
            return np.hypot(local[..., 0], local[..., 1]) <= self.half_a - inset
        return (np.abs(local[..., 0]) <= self.half_a - inset) & (np.abs(local[..., 1]) <= self.half_b - inset)


@dataclass(frozen=True)
class HoleReading:
    state: str                    # "present" | "absent" | "unknown"
    diameter_mm: float | None
    bound_mm: float | None
    support: float                # the best circle's share of circumference with an edge
    offset_mm: tuple | None       # its centre from the face's centre, in face coordinates
    images: int
    reason: str

    def to_dict(self) -> dict:
        return {"state": self.state, "diameter_mm": self.diameter_mm, "bound_mm": self.bound_mm,
                "support": self.support, "offset_mm": None if self.offset_mm is None else list(self.offset_mm),
                "images": self.images, "reason": self.reason}


def _grid() -> np.ndarray:
    n = int(round(2 * CROP_HALF_M / RESOLUTION_M)) + 1
    axis = np.linspace(-CROP_HALF_M, CROP_HALF_M, n)
    xx, yy = np.meshgrid(axis, axis)
    return np.stack([xx, yy], axis=-1)


def rectify(image: np.ndarray, pose, face: TopFace) -> np.ndarray:
    """`image` sampled on a millimetre grid of the top face (face coordinates)."""
    grid = _grid()
    pixels = pose.project(face.board_points(grid.reshape(-1, 2))).astype(np.float32)
    maps = pixels.reshape(grid.shape[0], grid.shape[1], 2)
    out = cv2.remap(image, maps[..., 0], maps[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out if out.ndim == 3 else out[:, :, None]


def pixels_per_mm(pose, face: TopFace) -> float:
    ends = pose.project(face.board_points(np.array([[-0.01, 0.0], [0.01, 0.0], [0.0, -0.01], [0.0, 0.01]])))
    return float((np.linalg.norm(ends[1] - ends[0]) + np.linalg.norm(ends[3] - ends[2])) / 40.0)


def _candidates(face: TopFace):
    reach = np.arange(-CENTRE_REACH_M, CENTRE_REACH_M + 1e-9, CENTRE_STEP_M)
    for dx in reach:
        for dy in reach:
            yield float(dx), float(dy)


def _sample(crops: np.ndarray, points: np.ndarray) -> np.ndarray:
    """crops (size, size, channels) at face coordinates `points` (..., 2), bilinear."""
    scale = (crops.shape[0] - 1) / (2 * CROP_HALF_M)
    maps = ((points + CROP_HALF_M) * scale).astype(np.float32)
    flat = maps.reshape(1, -1, 2)
    columns = []
    for start in range(0, flat.shape[1], 30000):  # remap's destination is limited to 32767 columns
        part = flat[:, start:start + 30000]
        out = cv2.remap(crops, part[..., 0], part[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        columns.append(out.reshape(part.shape[1], -1))
    return np.concatenate(columns).reshape(*points.shape[:-1], -1)


def read_hole(images_and_poses, face: TopFace) -> HoleReading:
    """The opening in `face` from [(BGR image, that eye's board pose), ...]."""
    crops, unusable = [], []
    face_mask = face.inside(_grid(), FACE_INSET_M)
    for image, pose in images_and_poses:
        if pixels_per_mm(pose, face) < MIN_FACE_PIXELS_PER_MM:
            unusable.append("the face is too small in the image")
            continue
        crop = rectify(image, pose, face).astype(np.float32)
        if float((crop.min(axis=2)[face_mask] >= 253).mean()) > CLIPPED_FACE_LIMIT:
            unusable.append("the face is saturated")
            continue
        crops.append(cv2.GaussianBlur(crop, (3, 3), 0).reshape(crop.shape))
    if not crops:
        why = "; ".join(sorted(set(unusable))) or "no image of the top face"
        return HoleReading("unknown", None, None, 0.0, None, 0, why)

    size = face.half_a if face.round else min(face.half_a, face.half_b)
    centres = np.array(list(_candidates(face)))
    unit = np.column_stack([np.cos(ANGLES), np.sin(ANGLES)])
    noise = float(np.median([np.median(np.abs(np.diff(c, axis=1)).max(axis=2)[face_mask[:, 1:]]) for c in crops]))
    threshold = max(EDGE_MIN_LEVELS, EDGE_NOISE_FACTOR * noise)

    # the face's own look, direction by direction (a shadow edge may cross it), on a ring just inside its outline
    if face.round:
        ring = (face.half_a - FACE_REFERENCE_INSET_M) * unit
    else:
        reach_a, reach_b = face.half_a - FACE_REFERENCE_INSET_M, face.half_b - FACE_REFERENCE_INSET_M
        scale = np.minimum(reach_a / np.maximum(np.abs(unit[:, 0]), 1e-9),
                           reach_b / np.maximum(np.abs(unit[:, 1]), 1e-9))
        ring = unit * scale[:, None]
    surface = [_sample(crop, ring[None])[0] for crop in crops]

    supports = np.zeros((len(centres), len(RADII_M)))
    strengths = np.zeros((len(centres), len(RADII_M)))
    for j, radius in enumerate(RADII_M):
        fits = np.hypot(centres[:, 0], centres[:, 1]) + radius <= size - FACE_INSET_M
        if not fits.any():
            continue
        chosen = centres[fits][:, None, :]
        inner = chosen + unit[None] * (radius - EDGE_HALF_WIDTH_M)
        outer = chosen + unit[None] * (radius + EDGE_HALF_WIDTH_M)
        # per angle and image: the step across the circle, counted only where the outside looks like the face (so
        # that faint facets inside a dim bore are not a rim); then the median over the images
        measured = []
        for crop, look in zip(crops, surface):
            outside = _sample(crop, outer)
            step = np.abs(outside - _sample(crop, inner)).max(axis=2)
            from_face = np.abs(outside - look[None]).max(axis=2)
            measured.append(np.where(from_face <= np.maximum(0.6 * step, threshold), step, 0.0))
        steps = np.median(np.stack(measured), axis=0)
        supports[fits, j] = (steps >= threshold).mean(axis=1)
        strengths[fits, j] = np.minimum(steps, 6 * threshold).mean(axis=1) * supports[fits, j]

    best_support = float(supports.max())
    if best_support < ABSENT_SUPPORT:
        return HoleReading("absent", None, None, round(best_support, 3), None, len(crops),
                           "no circle on the top face has an edge around enough of it")
    if best_support < PRESENT_SUPPORT:
        return HoleReading("unknown", None, None, round(best_support, 3), None, len(crops),
                           f"a partial ring only ({best_support:.0%} of a circle; an opening needs "
                           f"{PRESENT_SUPPORT:.0%})")
    # The rim is the OUTERMOST well-supported circle: whatever shows inside a bore (its far wall, the board seen
    # through it) lies within the rim. Among the circles within one step of that radius, the strongest.
    good = supports >= max(PRESENT_SUPPORT, best_support - 0.08)
    good &= strengths >= 0.6 * float(strengths[good].max())
    outer = int(np.nonzero(good.any(axis=0))[0].max())
    score = np.where(good, strengths, -1.0)
    score[:, :max(outer - 1, 0)] = -1.0
    i, j = np.unravel_index(int(score.argmax()), score.shape)
    # The samples sit EDGE_HALF_WIDTH_M either side, so a rim answers over a BAND of radii about that wide on each
    # side of it: the rim is the band's middle (strength-weighted), not its outer end.
    band = strengths[i] >= 0.8 * strengths[i, j]
    low = high = int(j)
    while low > 0 and band[low - 1]:
        low -= 1
    while high < len(RADII_M) - 1 and band[high + 1]:
        high += 1
    radius = float(np.average(RADII_M[low:high + 1], weights=strengths[i, low:high + 1]))
    spread_mm = float((RADII_M[high] - RADII_M[low]) * 1000)
    diameter = 2000 * radius
    bound = DIAMETER_FLOOR_BOUND_MM + spread_mm
    return HoleReading("present", round(diameter, 1), round(bound, 1), round(float(supports[i, j]), 3),
                       (round(float(centres[i, 0]) * 1000, 1), round(float(centres[i, 1]) * 1000, 1)), len(crops),
                       "a round edge on the top face")
