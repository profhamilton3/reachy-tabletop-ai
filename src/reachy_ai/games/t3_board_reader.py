"""Read a tic-tac-toe board from a stereo pair into the 1a Board, square by square (issue #51, epic #49).

Pure perception: reads files, moves nothing. The head is moved elsewhere (the viewer, issue #37).

Phase 1 uses one head pose (board_view). At session start one EMPTY-board pair is the reference: each square's quad in
the right eye (from derive_stereo_calibration.board_points_detail) and the reference image. At each read, per square:

  a) the square's visible region (eroded, inside the frame) is compared with the reference after a small shift
     alignment; the result is a normalised change score (mean absolute difference of the z-scored patches);
  b) a cube or cylinder detection whose box centre is inside the square's quad gives X or O, if the square changed;
  c) no change and no cube or cylinder detection: empty (whatever the detector's "empty" class says);
  d) anything else is unknown, never empty: a change with no usable detection, only "empty" detections on a changed
     square, a cube and a cylinder on one square, a piece detected on an unchanged square, or too little of the square
     visible to judge. The operator decides unknowns (1c).

Pixel spaces: the detector's boxes are raw right-camera pixels; board_points gives undistorted (pinhole) pixels. Images
are undistorted for the comparison, box centres are converted raw -> undistorted, and to_raw() goes back (round trip is
tested). Thresholds are ReaderSettings and are recorded in every BoardRead.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from reachy_ai.games import tictactoe as t3
from reachy_ai.pair_bundle import BadBundle, read_bundle
from reachy_ai.perception.stereo_depth import StereoGeometry

_SCRIPTS = str(Path(__file__).resolve().parents[3] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import derive_stereo_calibration as dsc  # noqa: E402

EYE = "right"
BOARD_VIEW = "board_view"
NECK_JOINTS = ("neck_roll", "neck_pitch", "neck_yaw")


class BoardReadRefused(Exception):
    """The reference or the pair cannot be used. The message says why."""


@dataclass(frozen=True)
class ReaderSettings:
    erode_px: int = 4                 # shrink each square's quad inside its edges
    shift_px: int = 10                # head-pose repeatability: search +-this many px
    change_threshold: float = 0.35    # change score at or above this = the square changed
    min_visible_fraction: float = 0.35  # of the eroded quad that is inside the frame, else unknown
    min_confidence: float = 0.30      # a detection below this is not usable
    pose_tolerance_deg: float = 3.0   # neck read-back may differ from the reference's by this much


@dataclass(frozen=True)
class Reference:
    calibration_id: Optional[str]
    zoom: str
    source: str
    lens: Optional[str]
    pose: str
    neck_deg: tuple
    size: tuple                       # (width, height)
    gray: np.ndarray = field(repr=False, compare=False)   # undistorted reference, right eye
    # {"r1c1": 4x2 undistorted px: near-left, near-right, far-right, far-left}
    quads: dict = field(repr=False, compare=False)
    pair_path: str = ""


@dataclass(frozen=True)
class SquareEvidence:
    cell: str
    state: str                        # t3.X / t3.O / t3.EMPTY / t3.UNKNOWN
    change_score: Optional[float]
    visible_fraction: float
    detections: tuple                 # ({"label", "confidence", "box_xywh"}, ...) with centres inside the quad
    reason: str


@dataclass(frozen=True)
class BoardRead:
    board: tuple                      # 1a Board, CELLS order
    squares: tuple                    # SquareEvidence, CELLS order
    settings: dict
    pose: str
    reference_pair: str
    pair: str

    def board_string(self) -> str:
        return t3.to_string(self.board)

    def to_dict(self) -> dict:
        return {"board": self.board_string(), "pose": self.pose, "settings": self.settings, "pair": self.pair,
                "reference_pair": self.reference_pair,
                "squares": [{**asdict(s), "detections": list(s.detections)} for s in self.squares]}


# ---------------------------------------------------------------- loading

@dataclass(frozen=True)
class _Pair:
    path: str
    pair: dict
    analysis: Optional[dict]
    right: np.ndarray


def _load(path) -> _Pair:
    """A viewer pair bundle: the .zip (checked against its manifest) or an unpacked folder."""
    path = Path(path)
    if path.is_dir():
        files = {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}
    elif path.is_file():
        try:
            _, files = read_bundle(path.read_bytes())
        except BadBundle as exc:
            raise BoardReadRefused(f"{path.name}: {exc}") from exc
    else:
        raise BoardReadRefused(f"{path} does not exist")
    for name in ("pair.json", f"{EYE}.jpg"):
        if name not in files:
            raise BoardReadRefused(f"{name} is missing from {path.name}")
    image = cv2.imdecode(np.frombuffer(files[f"{EYE}.jpg"], np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise BoardReadRefused(f"{EYE}.jpg in {path.name} is not an image")
    analysis = json.loads(files["analysis.json"]) if "analysis.json" in files else None
    return _Pair(str(path), json.loads(files["pair.json"]), analysis, image)


def _identity(pair: dict) -> dict:
    context = pair.get("camera_context") or {}
    profile = context.get("analysis_profile") or {}
    zoom = context.get("zoom")
    levels = zoom.get("levels") if isinstance(zoom, dict) else None
    if not isinstance(levels, dict):  # a pair saved before the zoom read-back block: use the connection settings
        settings = context.get("settings_at_connection") or {}
        levels = {eye: (settings.get(eye) or {}).get("zoom_level") for eye in ("left", "right")}
    if levels.get("left") != levels.get("right") or not levels.get("left"):
        raise BoardReadRefused(f"the zoom is unreadable or the eyes disagree: {levels}")
    source = profile.get("source") or pair.get("source")
    if source not in ("sim", "robot"):
        raise BoardReadRefused(f"the pair does not say whether it is sim or robot (source={source!r})")
    return {"calibration_id": context.get("calibration_id"), "zoom": levels["left"], "source": source,
            "lens": profile.get("lens") if source == "sim" else None}


def _neck(pair: dict) -> tuple:
    try:
        joints = pair["head_context"]["joints"]
        return tuple(float(joints[n]["present_deg"]) for n in NECK_JOINTS)
    except (KeyError, TypeError, ValueError):
        raise BoardReadRefused("pair.json has no neck read-back: the head pose is unknown") from None


def _geometry(identity: dict):
    return StereoGeometry("sim", identity["lens"] or "distorted") if identity["source"] == "sim" else None


# ---------------------------------------------------------------- pixel spaces

def to_undistorted(points, geometry) -> np.ndarray:
    """Raw right-camera pixels (n, 2) -> undistorted (pinhole) pixels."""
    K, D, pinhole = dsc.lens_for(EYE, geometry)
    pts = np.asarray(points, float).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, D, P=pinhole).reshape(-1, 2)


def to_raw(points, geometry) -> np.ndarray:
    """Undistorted (pinhole) pixels (n, 2) -> raw right-camera pixels."""
    K, D, pinhole = dsc.lens_for(EYE, geometry)
    pts = np.asarray(points, float).reshape(-1, 2)
    rays = np.column_stack([(pts[:, 0] - pinhole[0, 2]) / pinhole[0, 0], (pts[:, 1] - pinhole[1, 2]) / pinhole[1, 1],
                            np.ones(len(pts))])
    return cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)


def _undistort_gray(image: np.ndarray, geometry) -> np.ndarray:
    K, D, pinhole = dsc.lens_for(EYE, geometry)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.GaussianBlur(cv2.undistort(gray, K, D, None, pinhole), (3, 3), 0)


# ---------------------------------------------------------------- reference

def build_reference(pair_bundle, pose: str = BOARD_VIEW) -> Reference:
    """The empty-board reference from a pair captured at `pose`. Refuses unless the grid finder sees all nine."""
    pair = _load(pair_bundle)
    identity = _identity(pair.pair)
    geometry = _geometry(identity)
    found = dsc.board_points_detail(pair.right, EYE, geometry)
    if found is None:
        raise BoardReadRefused("the grid finder did not find the board in the reference pair")
    if found["cells"] != 9:
        raise BoardReadRefused(f"the grid finder found {found['cells']} of 9 squares at {pose}; the reference needs "
                               "all nine (a second pose would be needed)")
    img = found["img"]
    quads = {}
    for index, name in enumerate(t3.CELLS):
        near_left, near_right, far_left, far_right = img[4 * index:4 * index + 4]
        quads[name] = np.array([near_left, near_right, far_right, far_left])
    neck = _neck(pair.pair)
    return Reference(identity["calibration_id"], identity["zoom"], identity["source"], identity["lens"], pose, neck,
                     (pair.right.shape[1], pair.right.shape[0]), _undistort_gray(pair.right, geometry), quads,
                     pair.path)


# ---------------------------------------------------------------- change score

def _square_mask(quad: np.ndarray, shape, erode_px: int):
    """(mask, visible_fraction): the quad eroded inside its edges, clipped to the frame."""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 1)
    whole = int(mask.sum())
    if erode_px > 0:
        mask = cv2.erode(mask, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
    # fillPoly clips to the frame, so the visible fraction is the filled area over the quad's own area
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    return mask.astype(bool), (whole / area if area > 0 else 0.0)


def _zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / max(values.std(), 1.0)


def change_score(reference: np.ndarray, current: np.ndarray, mask: np.ndarray, shift_px: int) -> float:
    """Mean absolute difference of the z-scored patches, minimised over shifts (coarse, then fine)."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return float("nan")
    ref = _zscore(reference[ys, xs].astype(float))
    height, width = current.shape

    def score(dy, dx):
        y, x = ys + dy, xs + dx
        if y.min() < 0 or x.min() < 0 or y.max() >= height or x.max() >= width:
            return None
        return float(np.abs(_zscore(current[y, x].astype(float)) - ref).mean())

    best = (float("inf"), 0, 0)
    for dy in range(-shift_px, shift_px + 1, 2):
        for dx in range(-shift_px, shift_px + 1, 2):
            value = score(dy, dx)
            if value is not None and value < best[0]:
                best = (value, dy, dx)
    if best[0] == float("inf"):
        return float("nan")
    for dy in range(best[1] - 1, best[1] + 2):
        for dx in range(best[2] - 1, best[2] + 2):
            value = score(dy, dx)
            if value is not None and value < best[0]:
                best = (value, dy, dx)
    return best[0]


# ---------------------------------------------------------------- read

def _detections(pair: _Pair, geometry) -> list:
    """[(label, confidence, box_xywh, centre_in_undistorted_px)] from the pair's right-eye analysis."""
    if pair.analysis is None:
        raise BoardReadRefused(f"analysis.json is missing from {Path(pair.path).name}: analyse the pair in the viewer")
    if pair.analysis.get("eye", EYE) != EYE:
        raise BoardReadRefused(f"analysis.json is for the {pair.analysis.get('eye')} eye, not the {EYE} eye")
    found = []
    for d in pair.analysis.get("detections", []):
        x, y, w, h = d["box_xywh"]
        centre = to_undistorted([[x + w / 2, y + h / 2]], geometry)[0]
        found.append((d["label"], float(d["confidence"]), [x, y, w, h], centre))
    return found


def _check_matches(reference: Reference, pair: _Pair, settings: ReaderSettings) -> dict:
    identity = _identity(pair.pair)
    for key, ours in (("calibration_id", reference.calibration_id), ("zoom", reference.zoom),
                      ("source", reference.source)):
        if identity[key] != ours:
            raise BoardReadRefused(f"{key} differs from the reference: {identity[key]!r} vs {ours!r}")
    neck = _neck(pair.pair)
    gap = max(abs(a - b) for a, b in zip(neck, reference.neck_deg))
    if gap > settings.pose_tolerance_deg:
        raise BoardReadRefused(f"the head pose differs from the reference's by {gap:.1f} deg "
                               f"(tolerance {settings.pose_tolerance_deg})")
    if (pair.right.shape[1], pair.right.shape[0]) != reference.size:
        raise BoardReadRefused("the image size differs from the reference's")
    return identity


def read_board(pair_bundle, reference: Reference, settings: ReaderSettings = ReaderSettings()) -> BoardRead:
    pair = _load(pair_bundle)
    identity = _check_matches(reference, pair, settings)
    geometry = _geometry(identity)
    current = _undistort_gray(pair.right, geometry)
    detections = _detections(pair, geometry)
    squares = []
    for name in t3.CELLS:
        quad = reference.quads[name]
        mask, visible = _square_mask(quad, current.shape, settings.erode_px)
        inside = tuple(
            {"label": label, "confidence": conf, "box_xywh": box}
            for label, conf, box, centre in detections
            if cv2.pointPolygonTest(quad.astype(np.float32), (float(centre[0]), float(centre[1])), False) >= 0)
        score = None
        if visible >= settings.min_visible_fraction and mask.any():
            score = change_score(reference.gray, current, mask, settings.shift_px)
            score = None if np.isnan(score) else score
        squares.append(_judge(name, score, visible, inside, settings))
    return BoardRead(tuple(s.state for s in squares), tuple(squares), asdict(settings), reference.pose,
                     reference.pair_path, pair.path)


def _judge(name: str, score: Optional[float], visible: float, inside: tuple,
           settings: ReaderSettings) -> SquareEvidence:
    def done(state, reason):
        return SquareEvidence(name, state, None if score is None else round(score, 4), round(visible, 3), inside,
                              reason)

    if score is None:
        return done(t3.UNKNOWN, f"only {visible:.0%} of the square is visible (needs "
                                f"{settings.min_visible_fraction:.0%}) so no change score")
    changed = score >= settings.change_threshold
    usable = [d for d in inside if d["confidence"] >= settings.min_confidence]
    pieces = {d["label"] for d in usable if d["label"] in ("cube", "cylinder")}
    if len(pieces) == 2:
        return done(t3.UNKNOWN, "a cube and a cylinder are both detected on this square")
    if not changed:
        if pieces:
            return done(t3.UNKNOWN, f"a {next(iter(pieces))} is detected but the square did not change")
        return done(t3.EMPTY, "no change from the empty board and no cube or cylinder detected")
    if pieces:
        label = next(iter(pieces))
        best = max((d for d in usable if d["label"] == label), key=lambda d: d["confidence"])
        return done(t3.X if label == "cube" else t3.O, f"changed; {label} detected ({best['confidence']:.2f})")
    if any(d["label"] == "empty" for d in usable):
        return done(t3.UNKNOWN, "the square changed but the detector calls it empty")
    return done(t3.UNKNOWN, "the square changed but nothing was detected on it")
