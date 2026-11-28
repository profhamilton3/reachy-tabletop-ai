"""Part geometry from one head pose: each eye's board pose, the part's silhouette, and a solid fitted to it (#84).

Offline: arrays in, numbers out. No SDK, network or motion.

Why a fitted solid and not "pixel height x depth / focal length": from board_view the cameras look down at the board
(about 16 deg off straight down at the centre square), so a part's pixel extent is mostly its TOP face, and its height
adds only about 0.3 px per mm at zoom IN. The pinhole relation is the same one, applied to a solid that stands on the
board: the board's pose gives the depth, the silhouette gives the pixel extent, and the fit separates footprint from
height (the top face is nearer the camera than the base, so it also looks larger).

NO COLOUR (owner, 2026-11-27, #84): every step that measures geometry uses only the LARGEST difference over the three
channels, or the largest gradient over them, so reordering the channels changes nothing. Nothing here reads which
channel was largest. The grid finder runs on the empty, neutral board only.

Frames. Board: origin at the centre square's centre on the table top, +X toward the far row, +Y toward column 1,
+Z up (derive_stereo_calibration.board_cells). Each eye's pose is solved from its own empty-board image, so no neck
angle, camera mount or stereo calibration enters the part's size.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from reachy_ai.perception.stereo_depth import SIM_WARP, StereoGeometry

_SCRIPTS = str(Path(__file__).resolve().parents[3] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import derive_stereo_calibration as dsc  # noqa: E402

EYES = ("left", "right")
#: Half the side of a taped square, metres. The simulator's cells are 127 mm (reachy-1-2-sim truth.v1 cells); the
#: lab board's are the finder's own 5.08 in. With the wrong one the board fit is out by about 1 px.
CELL_HALF_M = {"sim": 0.0635, "robot": float(dsc.HALF)}
#: reachy-1-2-sim native_mujoco/zoom.py @ c46f4e4: OUT and IN are these DIAGONAL fields of view on 640 x 480, both
#: eyes alike, with fx == fy, and the same radial coefficients as at the calibrated level.
SIM_ZOOM_DIAGONAL_DEG = {"OUT": 125.0, "IN": 65.0}
SIM_SIZE = (640, 480)

BOARD_RMS_LIMIT_PX = 1.0     # a worse board fit is not used
CORNER_OUTLIER_PX = 1.2      # corners this far off the first fit (a clipped near row) are dropped from the second
MIN_CORNERS = 16             # four whole squares

DIFF_THRESHOLD = 24.0        # gray levels (largest channel) that count as changed
CONTOUR_SEARCH_PX = 4.0      # the edge is refined this far either side of the mask's contour
MIN_STEP = 20.0              # a contour point needs this much inside-outside difference to be used
MIN_CONTOUR_POINTS = 24
#: A cast shadow: every channel dimmed by about the same, modest factor, on a bright reference pixel.
SHADOW_MIN_RATIO = 0.62
SHADOW_MAX_SPREAD = 0.07
SHADOW_MIN_REFERENCE = 120.0
#: An outline point is only a LOWER limit where the board just outside is this dark (the tape) and the part is too:
#: a black part running onto black tape changes nothing, so the outline stops at the tape whatever the part does.
DARK_LEVEL = 75
CENSOR_PROBE_PX = 3.0


class NoCameraModel(ValueError):
    """There is no lens model for this source and zoom level, so nothing metric can be measured."""


class BoardNotFound(ValueError):
    """The empty-board image does not give a usable board pose."""


# ---------------------------------------------------------------- the lens at a zoom level

@dataclass(frozen=True)
class EyeLens:
    """One eye at one zoom: `source` and `distortion` describe the raw frame, `pinhole` the undistorted one."""
    eye: str
    source: np.ndarray
    distortion: np.ndarray
    pinhole: np.ndarray

    def board_lens(self, eye: str):  # the shape derive_stereo_calibration.lens_for expects of a sim geometry
        return self.source, self.distortion, self.pinhole


def _pinhole(fx: float, fy: float) -> np.ndarray:
    return np.array([[fx, 0.0, SIM_SIZE[0] / 2], [0.0, fy, SIM_SIZE[1] / 2], [0.0, 0.0, 1.0]])


def sim_zoom_focal_px(level: str) -> float:
    diagonal = math.hypot(*SIM_SIZE) / 2
    return diagonal / math.tan(math.radians(SIM_ZOOM_DIAGONAL_DEG[level]) / 2)


def eye_lens(source: str, zoom: str, eye: str, sim_lens: str = "distorted",
             profile_zoom: str | None = None) -> EyeLens:
    """The lens of `eye` at zoom level `zoom`. NoCameraModel when none is known (never guessed).

    sim: the simulator's zoom model (OUT, IN); ZERO and INTER are the measured calibration as the simulator renders it
    before any zoom command. robot: only the level the saved calibration profile is valid at."""
    i = EYES.index(eye)
    if source == "sim":
        if zoom in SIM_ZOOM_DIAGONAL_DEG:
            f = sim_zoom_focal_px(zoom)
            k = np.array([*SIM_WARP[eye][1], 0, 0, 0], float) if sim_lens == "distorted" else np.zeros(5)
            return EyeLens(eye, _pinhole(f, f), k, _pinhole(f, f))
        if zoom in ("ZERO", "INTER"):
            geometry = StereoGeometry("sim", sim_lens)
            src, k, pin = geometry.board_lens(eye)
            return EyeLens(eye, src, k, pin)
        raise NoCameraModel(f"no simulator lens model for zoom {zoom!r}")
    if profile_zoom is not None and zoom == profile_zoom:
        K, D, pin = dsc.lens_for(eye)
        return EyeLens(("left", "right")[i], K, D, pin)
    raise NoCameraModel(f"no robot lens calibration at zoom {zoom!r} (the saved profile is valid at "
                        f"{profile_zoom or 'an unrecorded level'}); calibrate the cameras at OUT and IN (#4)")


# ---------------------------------------------------------------- the board

@dataclass(frozen=True)
class BoardPoseEye:
    """One eye's pose of the board at one zoom: X_camera = R X_board + t."""
    lens: EyeLens
    rvec: np.ndarray
    tvec: np.ndarray
    rms_px: float
    corners: int
    half_m: float

    @property
    def rotation(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec)[0]

    def camera_position(self) -> np.ndarray:
        """The camera centre in the board frame."""
        return -self.rotation.T @ self.tvec.ravel()

    def project(self, points_board: np.ndarray) -> np.ndarray:
        """Board points (n, 3) -> RAW frame pixels (n, 2)."""
        pts = np.asarray(points_board, float).reshape(-1, 1, 3)
        return cv2.projectPoints(pts, self.rvec, self.tvec, self.lens.source, self.lens.distortion)[0].reshape(-1, 2)

    def on_plane(self, pixels: np.ndarray, z: float = 0.0) -> np.ndarray:
        """RAW pixels -> where their rays meet the board plane at height `z`."""
        und = cv2.undistortPoints(np.asarray(pixels, float).reshape(-1, 1, 2), self.lens.source,
                                  self.lens.distortion).reshape(-1, 2)
        rays = np.column_stack([und, np.ones(len(und))]) @ self.rotation  # camera -> board directions
        origin = self.camera_position()
        s = (z - origin[2]) / rays[:, 2]
        return origin + rays * s[:, None]

    def cell_quad(self, row: int, column: int) -> np.ndarray:
        """The raw-pixel outline of square (row, column), 1-based as r1c1 .. r3c3."""
        x, y = (row - 2) * dsc.PITCH, (2 - column) * dsc.PITCH
        h = self.half_m
        edge = np.linspace(-h, h, 9)
        ring = ([(x - h, y + e) for e in edge] + [(x + e, y + h) for e in edge]
                + [(x + h, y - e) for e in edge] + [(x - e, y - h) for e in edge])
        return self.project(np.array([(a, b, 0.0) for a, b in ring]))

    def board_mask(self, shape, margin_m: float = 0.0) -> np.ndarray:
        """A mask over the taped grid's outer square, grown by `margin_m` on the table plane."""
        reach = dsc.PITCH + self.half_m + margin_m
        edge = np.linspace(-reach, reach, 25)
        ring = ([(-reach, e) for e in edge] + [(e, reach) for e in edge]
                + [(reach, -e) for e in edge] + [(-e, -reach) for e in edge])
        outline = self.project(np.array([(a, b, 0.0) for a, b in ring]))
        mask = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(outline).astype(np.int32)], 1)
        return mask.astype(bool)


def fit_board(image: np.ndarray, lens: EyeLens, source: str) -> BoardPoseEye:
    """The board pose from an EMPTY-board image of one eye. The near row is often clipped by the robot's chest, so
    corners that do not fit the far and middle rows are dropped."""
    if image.ndim == 3:  # the channels' plain mean, so that even the board fit does not depend on their order
        image = np.round(image.mean(axis=2)).astype(np.uint8)
    found = dsc.board_points_detail(image, lens.eye, _AsGeometry(lens), near_row_hidden=True)
    if found is None:
        raise BoardNotFound(f"the grid finder did not find the board in the {lens.eye} eye (is the board empty?)")
    half = CELL_HALF_M[source]
    centres = dsc.board_cells()
    obj = found["obj"].astype(float).copy()
    for i in range(len(obj)):  # the finder's corners at this board's square size
        centre = min(centres.values(), key=lambda c: float(np.hypot(*(obj[i, :2] - c))))
        obj[i, :2] = centre + np.sign(obj[i, :2] - centre) * half
    pixels = found["img"].astype(float)
    keep = obj[:, 0] > -dsc.PITCH / 2 if (obj[:, 0] > -dsc.PITCH / 2).sum() >= MIN_CORNERS else obj[:, 0] > -1e9

    def solve(selection):
        ok, rvec, tvec = cv2.solvePnP(obj[selection], pixels[selection], lens.pinhole, None,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise BoardNotFound(f"the board pose did not solve in the {lens.eye} eye")
        projected = cv2.projectPoints(obj, rvec, tvec, lens.pinhole, None)[0].reshape(-1, 2)
        return rvec, tvec, np.linalg.norm(projected - pixels, axis=1)

    rvec, tvec, error = solve(keep)
    inliers = error <= CORNER_OUTLIER_PX
    if inliers.sum() >= MIN_CORNERS:
        keep = inliers
        rvec, tvec, error = solve(keep)
    rms = float(np.sqrt((error[keep] ** 2).mean()))
    pose = BoardPoseEye(lens, rvec, tvec, rms, int(keep.sum()), half)
    if rms > BOARD_RMS_LIMIT_PX or pose.camera_position()[2] <= 0:
        raise BoardNotFound(f"the board fit in the {lens.eye} eye is off by {rms:.2f} px "
                            f"(limit {BOARD_RMS_LIMIT_PX}): is the board empty and is this the stated zoom?")
    return pose


class _AsGeometry:
    """An EyeLens in the form the grid finder takes a sim geometry (it only asks for `source` and `board_lens`)."""
    source = "sim"

    def __init__(self, lens: EyeLens):
        self._lens = lens

    def board_lens(self, eye):
        return self._lens.board_lens(eye)


# ---------------------------------------------------------------- the silhouette

def channel_difference(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """The largest absolute difference over the channels, lightly smoothed. Channel order does not matter."""
    a = cv2.GaussianBlur(current, (3, 3), 0).astype(np.int16)
    b = cv2.GaussianBlur(reference, (3, 3), 0).astype(np.int16)
    diff = np.abs(a - b)
    return (diff.max(axis=2) if diff.ndim == 3 else diff).astype(np.float32)


def shadow_mask(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Pixels that look like the reference in shadow. Uses only the smallest and largest channel ratios, so channel
    order does not matter. A grey part much darker than the board is not a shadow (its ratio is lower)."""
    a = cv2.GaussianBlur(current, (3, 3), 0).astype(np.float32)
    b = cv2.GaussianBlur(reference, (3, 3), 0).astype(np.float32)
    if a.ndim == 2:
        a, b = a[:, :, None], b[:, :, None]
    ratio = a / np.maximum(b, 1.0)
    low, high = ratio.min(axis=2), ratio.max(axis=2)
    return ((low >= SHADOW_MIN_RATIO) & (high <= 1.02) & (high - low <= SHADOW_MAX_SPREAD)
            & (b.min(axis=2) >= SHADOW_MIN_REFERENCE))


def head_shift(reference: np.ndarray, current: np.ndarray, limit_px: float = 6.0) -> tuple[float, float]:
    """(dx, dy) of `current` against `reference` from phase correlation of gradient strength (head repeatability).
    Clamped to +-limit_px; (0, 0) when the images agree."""
    def edges(image):
        planes = image if image.ndim == 3 else image[:, :, None]
        grads = [np.hypot(cv2.Sobel(planes[:, :, i], cv2.CV_32F, 1, 0), cv2.Sobel(planes[:, :, i], cv2.CV_32F, 0, 1))
                 for i in range(planes.shape[2])]
        return np.maximum.reduce(grads)

    window = cv2.createHanningWindow((reference.shape[1], reference.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(edges(reference), edges(current), window)
    return (float(np.clip(dx, -limit_px, limit_px)), float(np.clip(dy, -limit_px, limit_px)))


@dataclass(frozen=True)
class Silhouette:
    """The part in one image: its filled mask, its outer contour refined to sub-pixel edge points, and how sharp
    that edge is (the 25-75 % width of the difference profile across it, in pixels)."""
    mask: np.ndarray
    points: np.ndarray           # (n, 2) raw pixels on the refined outline
    edge_width_px: float
    area_px: int
    other_area_px: int           # changed area on the board that is not this blob
    touches_frame: bool
    difference: np.ndarray = field(repr=False, default=None)
    hidden: np.ndarray = field(repr=False, default=None)  # (m, 2) outline points that are only a lower limit
    blind: np.ndarray = field(repr=False, default=None)   # pixels where a dark part on dark tape could not show

    @property
    def hidden_share(self) -> float:
        n = len(self.points) + (0 if self.hidden is None else len(self.hidden))
        return 0.0 if self.hidden is None else len(self.hidden) / max(n, 1)


def find_silhouette(reference: np.ndarray, current: np.ndarray, region: np.ndarray,
                    threshold: float = DIFF_THRESHOLD, outline_offset_px: float = 0.0) -> Silhouette | None:
    """The largest changed blob touching `region` (the board), or None. Class-agnostic: no detector.

    `outline_offset_px` moves every refined point outward by that much (used to measure how much the fit depends on
    where the edge is put)."""
    diff = channel_difference(reference, current)
    diff[shadow_mask(reference, current)] = 0.0
    changed = (diff > threshold).astype(np.uint8)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(changed)
    on_board = [i for i in range(1, count) if (region & (labels == i)).any()]
    min_area = 40
    on_board = [i for i in on_board if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not on_board:
        return None
    best = max(on_board, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    contour = max(cv2.findContours((labels == best).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    filled = np.zeros(changed.shape, np.uint8)
    cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)  # a see-through bore is part of the outline
    x, y, w, h = cv2.boundingRect(contour)
    touches = x <= 1 or y <= 1 or x + w >= diff.shape[1] - 1 or y + h >= diff.shape[0] - 1
    points, widths = _refine(diff, contour.reshape(-1, 2).astype(float), filled)
    if len(points) < MIN_CONTOUR_POINTS:
        return None
    centre = points.mean(axis=0)
    outward = points - centre
    outward /= np.maximum(np.linalg.norm(outward, axis=1)[:, None], 1e-9)
    hidden = _hidden(reference, current, points, outward)
    if outline_offset_px:
        points = points + outline_offset_px * outward
    others = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in on_board if i != best)
    if (~hidden).sum() < MIN_CONTOUR_POINTS:
        return None
    return Silhouette(filled.astype(bool), points[~hidden], float(np.nanmedian(widths)), int(filled.sum()), others,
                      touches, diff, points[hidden], _blind(reference, current))


def _brightest(image: np.ndarray) -> np.ndarray:
    return image.max(axis=2) if image.ndim == 3 else image


def _blind(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Pixels that are dark both with and without the part: whether the part covers them cannot be told."""
    return (_brightest(reference) < DARK_LEVEL) & (_brightest(current) < DARK_LEVEL)


def _hidden(reference: np.ndarray, current: np.ndarray, points: np.ndarray, outward: np.ndarray) -> np.ndarray:
    """Which outline points lie against dark tape with an equally dark part: no edge could be seen there."""
    def brightest(image, at):
        level = _brightest(image)
        x = np.clip(np.round(at[:, 0]).astype(int), 0, level.shape[1] - 1)
        y = np.clip(np.round(at[:, 1]).astype(int), 0, level.shape[0] - 1)
        return level[y, x]

    beyond = points + CENSOR_PROBE_PX * outward
    within = points - CENSOR_PROBE_PX * outward
    return (brightest(reference, beyond) < DARK_LEVEL) & (brightest(current, within) < DARK_LEVEL)


def _refine(diff: np.ndarray, contour: np.ndarray, filled: np.ndarray):
    """Each contour point moved along the outline's normal to where the difference falls to half its inside level.

    The reference cancels the board's own edges, so a tape line cannot pull the outline."""
    n = len(contour)
    span = 4
    tangent = np.roll(contour, -span, axis=0) - np.roll(contour, span, axis=0)
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-9)
    probe = np.round(contour + 3 * normal).astype(int)
    inside_height, inside_width = filled.shape
    outward_is_inside = filled[np.clip(probe[:, 1], 0, inside_height - 1),
                               np.clip(probe[:, 0], 0, inside_width - 1)].sum()
    if outward_is_inside > n / 2:
        normal = -normal
    offsets = np.arange(-CONTOUR_SEARCH_PX, CONTOUR_SEARCH_PX + 0.25, 0.25, dtype=np.float32)
    xs = (contour[:, 0:1] + normal[:, 0:1] * offsets[None, :]).astype(np.float32)
    ys = (contour[:, 1:2] + normal[:, 1:2] * offsets[None, :]).astype(np.float32)
    profile = cv2.remap(diff, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    points, widths = [], []
    for i in range(0, n, 2):
        row = profile[i]
        inside = float(np.median(row[offsets <= -2.0]))
        outside = float(np.median(row[offsets >= 2.0]))
        if inside - outside < MIN_STEP:
            continue

        def crossing(level):
            below = np.nonzero(row < level)[0]
            below = below[(below > 0) & (offsets[below] > -2.5)]
            if not len(below):
                return None
            j = int(below[0])
            a, b = float(row[j - 1]), float(row[j])
            if a < level:
                return None
            return float(offsets[j - 1] + (a - level) / max(a - b, 1e-6) * 0.25)

        half = crossing(outside + 0.5 * (inside - outside))
        if half is None:
            continue
        points.append(contour[i] + normal[i] * half)
        high, low = crossing(outside + 0.75 * (inside - outside)), crossing(outside + 0.25 * (inside - outside))
        if high is not None and low is not None:
            widths.append(low - high)
    return np.array(points).reshape(-1, 2), np.array(widths or [float("nan")])


# ---------------------------------------------------------------- the solids

#: model name -> its parameters, metres and radians. All stand on the board (z = 0).
MODELS = {
    "box": ("x", "y", "yaw", "a", "b", "h"),          # a rectangular block: footprint a x b, height h
    "cylinder": ("x", "y", "d", "h"),                 # axis vertical
    "lying_cylinder": ("x", "y", "yaw", "length", "d"),  # axis horizontal, along `yaw`
}
_CIRCLE = np.linspace(0.0, 2 * math.pi, 72, endpoint=False)
SIZE_LIMITS_M = (0.008, 0.20)


def model_points(model: str, p) -> np.ndarray:
    """Points on the solid whose projections' convex hull is its silhouette."""
    if model == "cylinder":
        x, y, d, h = p
        ring = np.column_stack([x + d / 2 * np.cos(_CIRCLE), y + d / 2 * np.sin(_CIRCLE)])
        return np.vstack([np.column_stack([ring, np.full(len(ring), z)]) for z in (0.0, h)])
    if model == "box":
        x, y, yaw, a, b, h = p
        t = np.linspace(-0.5, 0.5, 9)
        local = np.vstack([np.column_stack([t * a, np.full(9, s * b / 2)]) for s in (-1, 1)]
                          + [np.column_stack([np.full(9, s * a / 2), t * b]) for s in (-1, 1)])
        c, s_ = math.cos(yaw), math.sin(yaw)
        ring = local @ np.array([[c, s_], [-s_, c]]) + [x, y]
        return np.vstack([np.column_stack([ring, np.full(len(ring), z)]) for z in (0.0, h)])
    x, y, yaw, length, d = p
    axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    across = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    ring = np.outer(np.cos(_CIRCLE), across) + np.outer(np.sin(_CIRCLE), up)
    centre = np.array([x, y, d / 2])
    return np.vstack([centre + s * axis * length / 2 + ring * d / 2 for s in (-1, 1)])


def hull(pose: BoardPoseEye, model: str, params) -> np.ndarray:
    projected = pose.project(model_points(model, params)).astype(np.float32)
    return cv2.convexHull(projected).reshape(-1, 2)


def _distances(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Signed distance of each point to a convex polygon's boundary (positive outside)."""
    a = polygon.astype(float)
    b = np.roll(a, -1, axis=0)
    edge = b - a
    length2 = np.maximum((edge ** 2).sum(axis=1), 1e-12)
    rel = points[:, None, :] - a[None, :, :]
    s = np.clip((rel * edge[None]).sum(axis=2) / length2[None], 0.0, 1.0)
    nearest = np.linalg.norm(rel - s[..., None] * edge[None], axis=2).min(axis=1)
    cross = edge[None, :, 0] * rel[:, :, 1] - edge[None, :, 1] * rel[:, :, 0]
    inside = np.all(cross >= -1e-9, axis=1) | np.all(cross <= 1e-9, axis=1)
    return np.where(inside, -nearest, nearest)


@dataclass(frozen=True)
class View:
    """One image's evidence: the eye's board pose and the part's silhouette in it."""
    pose: BoardPoseEye
    silhouette: Silhouette


@dataclass(frozen=True)
class SolidFit:
    model: str
    params: dict
    sigma: dict               # one standard error per parameter, from the fit's Jacobian
    rms_px: float             # robust RMS of the outline distances
    cost: float               # mean soft-L1 cost per outline point (for comparing models)
    iou: float                # the worst view's overlap of the fitted silhouette with the observed one

    @property
    def height_m(self) -> float:
        return self.params["d"] if self.model == "lying_cylinder" else self.params["h"]

    @property
    def footprint_m(self) -> tuple[float, float]:
        """(longer, shorter) horizontal size."""
        if self.model == "cylinder":
            return self.params["d"], self.params["d"]
        if self.model == "box":
            pair = (self.params["a"], self.params["b"])
        else:
            pair = (self.params["length"], self.params["d"])
        return max(pair), min(pair)


F_SCALE_PX = 0.7


def _residuals(params, model: str, views) -> np.ndarray:
    out = []
    for view in views:
        polygon = hull(view.pose, model, params)
        d = _distances(view.silhouette.points, polygon)
        if view.silhouette.hidden is not None and len(view.silhouette.hidden):
            # a lower limit only: the solid must reach this far, and may reach further
            d = np.concatenate([d, np.maximum(_distances(view.silhouette.hidden, polygon), 0.0)])
        out.append(d / math.sqrt(len(d) / 100.0))  # each view weighs the same, whatever its outline's length
    return np.concatenate(out)


def _seeds(model: str, views):
    """Starting guesses: the footprint under the silhouette's middle, at three heights and a few yaws."""
    first = views[0]
    centroid = first.silhouette.points.mean(axis=0)
    spread = first.silhouette.points
    for h in (0.025, 0.050, 0.070):
        centre = first.pose.on_plane(centroid[None], h / 2)[0]
        edge = first.pose.on_plane(spread, h / 2)
        width = float(np.clip(np.ptp(edge[:, 1]), *SIZE_LIMITS_M))
        if model == "cylinder":
            yield [centre[0], centre[1], width, h]
        elif model == "box":
            for yaw in np.radians((0.0, 15.0, 30.0, 45.0, 60.0, 75.0)):  # a square footprint repeats every 90 deg
                for share in (0.75, 0.9):  # the outline's width over-reads a turned block's side
                    yield [centre[0], centre[1], yaw, width * share, width * share, h]
        else:
            depth = float(np.clip(np.ptp(edge[:, 0]), *SIZE_LIMITS_M))
            for yaw in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):
                yield [centre[0], centre[1], yaw, max(width, depth), min(width, depth)]


def _bounds(model: str):
    low, high = SIZE_LIMITS_M
    table = {"x": (-0.6, 0.6), "y": (-0.6, 0.6), "yaw": (-2 * math.pi, 2 * math.pi), "a": (low, high),
             "b": (low, high), "d": (low, high), "h": (low, high), "length": (low, high)}
    return (np.array([table[n][0] for n in MODELS[model]]), np.array([table[n][1] for n in MODELS[model]]))


def iou(pose: BoardPoseEye, model: str, params, silhouette: Silhouette) -> float:
    rendered = np.zeros(silhouette.mask.shape, np.uint8)
    polygon = np.round(hull(pose, model, params) * 8).astype(np.int32)
    cv2.fillConvexPoly(rendered, polygon, 1, lineType=cv2.LINE_8, shift=3)
    rendered, observed = rendered.astype(bool), silhouette.mask
    if silhouette.blind is not None:  # where nothing could show, neither side is counted
        unseen = silhouette.blind & ~observed
        rendered = rendered & ~unseen
    union = (rendered | observed).sum()
    return float((rendered & observed).sum() / union) if union else 0.0


def fit_solid(model: str, views, start: dict | None = None) -> SolidFit:
    """The solid of kind `model` whose silhouette best matches every view's outline (soft-L1 least squares).
    With `start` (a fit's params) the search begins there only: for refits that measure how much a fit moves."""
    low, high = _bounds(model)
    step = np.array([0.01 if n != "yaw" else 0.3 for n in MODELS[model]])

    def solve(seed, evaluations):
        seed = np.clip(np.array(seed, float), low, high)
        return least_squares(_residuals, seed, args=(model, views), loss="soft_l1", f_scale=F_SCALE_PX,
                             bounds=(low, high), x_scale=step, max_nfev=evaluations, diff_step=1e-6)

    if start is not None:
        best = solve([start[n] for n in MODELS[model]], 200)
    else:
        trials = sorted((solve(seed, 12) for seed in _seeds(model, views)), key=lambda r: r.cost)[:4]
        best = min((solve(t.x, 200) for t in trials), key=lambda r: r.cost)
    squared = (best.fun / F_SCALE_PX) ** 2
    cost = float(np.mean(2 * (np.sqrt(1 + squared) - 1)) * F_SCALE_PX ** 2 / 2)
    inlier = np.abs(best.fun) < 3 * F_SCALE_PX
    rms = float(np.sqrt((best.fun[inlier] ** 2).mean())) if inlier.any() else float("inf")
    variance = float((best.fun[inlier] ** 2).sum() / max(inlier.sum() - len(best.x), 1))
    jacobian = best.jac[inlier]
    covariance = np.linalg.pinv(jacobian.T @ jacobian) * variance
    sigma = {name: float(math.sqrt(max(covariance[i, i], 0.0))) for i, name in enumerate(MODELS[model])}
    params = {name: float(v) for name, v in zip(MODELS[model], best.x)}
    worst = min(iou(v.pose, model, best.x, v.silhouette) for v in views)
    return SolidFit(model, params, sigma, rms, cost, worst)


def fit_all(views) -> dict:
    """model -> SolidFit for every model."""
    return {model: fit_solid(model, views) for model in MODELS}
