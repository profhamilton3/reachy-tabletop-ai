"""Build a PerceivedScene from a viewer pair bundle (docs/design/perceived-scene.md sections 1.4 to 1.6, P1).

Offline: reads files, writes nothing, moves nothing. The scene is a description of what the cameras saw and how
wrong that could be. It can only add refusals downstream (section 3.2), never permit a move.

Inputs
    a viewer single-pair bundle (a folder or the .zip): pair.json, analysis.json, depth.json (+ left/right.jpg);
    optionally a head-sweep bundle, whose analysis gives the BoardPose (analyze_capture_bundle.analyze_head);
    optionally a BoundsModel read from a sim grader result.json (perception_grade.v1).

What it refuses to build (SceneRefused)
    an unreadable zoom, eyes at different zooms, a robot capture whose zoom differs from its profile's,
    files that disagree about the profile or the neck, and a surface correction with no size table for the rig.

Bounds are looked up, never typed in. No matching row gives bounds_provisional=True and infinite bounds. A row with
fewer than MIN_DISTINCT distinct sightings is provisional too (OD-2): it still bounds the object, but only to refuse.
"""
import hashlib
import json
import math
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from reachy_ai.camera_zoom import EYES, normalize_zoom
from reachy_ai.config import HEAD_PRESETS, NECK_JOINTS
from reachy_ai.pair_bundle import BadBundle, read_bundle
from reachy_ai.perception import scene_contract as sc
from reachy_ai.perception.stereo_depth import PairDepth

REPO = Path(__file__).resolve().parents[3]

#: stereo_depth._analyze: SGBM, bidirectional check <= 1 px, median of the inner 60% of the box.
DEPTH_METHOD = "sgbm-bidir-1px/inner60-median"
#: OD-2: distinct sightings a bounds row needs before it may be non-provisional.
MIN_DISTINCT = 30
BOUNDS_COVERAGE = 0.95
DEFAULT_MOUNT_ID = "october-2026"
GRADER_SCHEMA = "perception_grade.v1"

#: The taped grid (reachy-1-2-sim truth.v1 cells): pitch between cell centres and the square's half-width.
CELL_PITCH_M = 0.1524
CELL_HALF_M = 0.0635
#: A detection this far from the table plane (board frame z) is not on the board: below, or above the tallest piece.
BOARD_Z_RANGE_M = (-0.05, 0.15)
#: A neck read-back this close to a preset on every joint takes the preset's name in a bounds key.
POSE_NAME_TOLERANCE_DEG = 2.0

#: Per-rig class sizes (design section 1.6). rig_layout_id -> kind -> pieces as (name, half-extents x, y, z).
#: FWD Center lab pieces are owner-measured (sim docs/scene-object-dimensions.md, 2026-10-19, #201); the sim's
#: red_cube and blue_cylinder come from the same file (owner confirmation, 2026-10-06).
SIZE_TABLE = {
    "FWDCenterLabSivaPool": {
        "cube": (("pool_box", (0.0254, 0.0254, 0.0254)), ("red_cube", (0.030, 0.030, 0.030))),
        "cylinder": (("pool_cyl", (0.02285, 0.02285, 0.03555)), ("blue_cylinder", (0.035, 0.035, 0.050))),
    },
}
SURFACE_CORRECTION = "class-half-extent.v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SceneRefused(Exception):
    """The capture cannot make a valid scene. The message says why."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- the pair bundle

@dataclass(frozen=True)
class PairBundle:
    pair: Dict[str, Any]
    analysis: Dict[str, Any]
    depth: Dict[str, Any]
    left_sha256: Optional[str]
    right_sha256: Optional[str]
    bundle_sha256: str
    left_jpg: Optional[bytes]
    right_jpg: Optional[bytes]
    file_time_utc: str
    path: str


def _load_json(name: str, raw: Optional[bytes]) -> Dict[str, Any]:
    if raw is None:
        raise SceneRefused("%s is missing from the pair bundle (analyze and measure the pair in the viewer first)"
                           % name)
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise SceneRefused("%s is not valid JSON: %s" % (name, exc)) from exc
    if not isinstance(value, dict):
        raise SceneRefused("%s is not a JSON object" % name)
    return value


def load_pair_bundle(path) -> PairBundle:
    """A viewer single-pair bundle: the .zip (checked against its manifest) or an unpacked folder."""
    path = Path(path)
    if path.is_dir():
        files = {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}
        mtime = path.stat().st_mtime
        manifest_raw = files.get("manifest.json")
    elif path.is_file():
        try:
            manifest, files = read_bundle(path.read_bytes())
        except BadBundle as exc:
            raise SceneRefused("%s: %s" % (path.name, exc)) from exc
        manifest_raw = json.dumps(manifest, sort_keys=True).encode()
        mtime = path.stat().st_mtime
    else:
        raise SceneRefused("%s does not exist" % path)
    pair = _load_json("pair.json", files.get("pair.json"))
    analysis = _load_json("analysis.json", files.get("analysis.json"))
    depth = _load_json("depth.json", files.get("depth.json"))
    hashed = manifest_raw if manifest_raw is not None else b"".join(
        files[n] for n in ("pair.json", "analysis.json", "depth.json"))
    return PairBundle(
        pair=pair, analysis=analysis, depth=depth,
        left_sha256=_sha256(files["left.jpg"]) if "left.jpg" in files else None,
        right_sha256=_sha256(files["right.jpg"]) if "right.jpg" in files else None,
        bundle_sha256=_sha256(hashed), left_jpg=files.get("left.jpg"), right_jpg=files.get("right.jpg"),
        file_time_utc=datetime.fromtimestamp(mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), path=str(path))


# ---------------------------------------------------------------- rig identity and zoom

@dataclass(frozen=True)
class RigOptions:
    """What the bundle does not say. Defaults suit the sim; a robot capture must name its layout and profile file."""
    rig_layout_id: Optional[str] = None      # sim: the truth export's scene_name when not given
    mount_id: str = DEFAULT_MOUNT_ID
    rig_epoch: int = 0
    scene_name: Optional[str] = None         # sim only (truth export)
    scene_sha256: Optional[str] = None
    scene_revision: Optional[str] = None
    profile_file: Optional[str] = None       # robot: the stereo profile whose SHA-256 is intrinsics_sha256
    captured_utc: Optional[str] = None


def neck_read_back(bundle: PairBundle) -> Tuple[float, float, float]:
    """(roll, pitch, yaw) degrees as READ BACK; the pair and the depth result must agree on it."""
    def joints(block, where):
        try:
            return tuple(float(block["joints"][n]["present_deg"]) for n in NECK_JOINTS)
        except (KeyError, TypeError, ValueError):
            raise SceneRefused("no neck read-back in %s: the pose cannot be placed in the torso frame" % where)
    neck = joints(bundle.pair.get("head_context") or {}, "pair.json head_context")
    used = bundle.depth.get("head_pose_used")
    if used is not None:
        try:
            depth_neck = tuple(float(used[n]["present_deg"]) for n in NECK_JOINTS)
        except (KeyError, TypeError, ValueError):
            raise SceneRefused("depth.json head_pose_used is unreadable")
        if max(abs(a - b) for a, b in zip(neck, depth_neck)) > 0.01:
            raise SceneRefused("depth.json was computed at neck %s but pair.json read back %s" % (depth_neck, neck))
    return neck


def head_pose_name(neck_deg: Sequence[float]) -> str:
    """The HEAD_PRESETS name this read-back is within POSE_NAME_TOLERANCE_DEG of, else "custom"."""
    best = ("custom", POSE_NAME_TOLERANCE_DEG)
    for name, preset in HEAD_PRESETS.items():
        gap = max(abs(a - b) for a, b in zip(neck_deg, preset))
        if gap <= best[1]:
            best = (name, gap)
    return best[0]


def read_zoom(bundle: PairBundle, source: str) -> str:
    """The captured zoom level, from the cameras' read-back. Refuses an unreadable zoom, disagreeing eyes, and
    (robot) a capture whose zoom differs from its profile's: none of them has valid depth."""
    zoom = (bundle.pair.get("camera_context") or {}).get("zoom")
    if not isinstance(zoom, dict) or not isinstance(zoom.get("levels"), dict):
        raise SceneRefused("pair.json has no camera_context.zoom read-back: the zoom is unreadable, so depth is "
                           "not valid")
    levels = {eye: normalize_zoom(zoom["levels"].get(eye)) for eye in EYES}
    unreadable = [eye for eye, level in levels.items() if level is None]
    if unreadable:
        raise SceneRefused("zoom_status: the zoom could not be read from the %s camera(s)" % " and ".join(unreadable))
    if levels["left"] != levels["right"]:
        raise SceneRefused("zoom_status: the eyes disagree (left %s, right %s); stereo geometry does not hold"
                           % (levels["left"], levels["right"]))
    if source == "robot" and zoom.get("matches_profile") is not True:
        raise SceneRefused("zoom_status: matches_profile is %r; the capture's zoom (%s) is not confirmed to match the "
                           "stereo profile, so depth is not valid" % (zoom.get("matches_profile"), levels["left"]))
    return levels["left"]


def build_rig(bundle: PairBundle, options: RigOptions) -> sc.RigIdentity:
    source = bundle.pair.get("source") or (bundle.depth.get("profile") or {}).get("source")
    if source not in sc.SOURCES:
        raise SceneRefused("the bundle does not say whether it is a sim or robot capture (source=%r)" % (source,))
    zoom_level = read_zoom(bundle, source)
    context = bundle.pair.get("camera_context") or {}
    calibration_id = context.get("calibration_id")
    profile_id = (bundle.depth.get("profile") or {}).get("id") or (context.get("analysis_profile") or {}).get("id")
    if not calibration_id:
        raise SceneRefused("pair.json has no camera_context.calibration_id")
    if source == "sim":
        if profile_id != calibration_id:
            raise SceneRefused("pair.json calibration_id %r and depth.json profile id %r disagree"
                               % (calibration_id, profile_id))
        intrinsics = calibration_id
        layout = options.rig_layout_id or options.scene_name
    else:
        if not options.profile_file:
            raise SceneRefused("a robot capture needs --profile-file: the intrinsics hash is the SHA-256 of that file")
        intrinsics = _sha256(Path(options.profile_file).read_bytes())
        layout = options.rig_layout_id
    if not layout:
        raise SceneRefused("no rig_layout_id: name the measured scene (sim: pass the truth export or --rig-layout-id)")
    try:
        return sc.RigIdentity(
            identity_version=sc.IDENTITY_VERSION, source=source, rig_layout_id=layout, mount_id=options.mount_id,
            calibration_id=calibration_id, zoom_level=zoom_level, intrinsics_sha256=intrinsics,
            rig_epoch=options.rig_epoch,
            scene_name=options.scene_name if source == "sim" else None,
            scene_sha256=options.scene_sha256 if source == "sim" else None,
            scene_revision=options.scene_revision if source == "sim" else None)
    except sc.SceneContractError as exc:
        raise SceneRefused(str(exc)) from exc


def code_identity() -> Tuple[str, bool]:
    """(HEAD sha, dirty) of this checkout. Untracked files do not make it dirty. Unknown means ("", True)."""
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO)] + list(args), capture_output=True, text=True, timeout=20)
    try:
        head = git("rev-parse", "HEAD")
        status = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return "", True
    if head.returncode != 0 or status.returncode != 0:
        return "", True
    return head.stdout.strip(), bool(status.stdout.strip())


# ---------------------------------------------------------------- camera geometry (sim)

_OPTICAL_TO_HEAD = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _rot(axis: str, degrees: float) -> np.ndarray:
    return Rotation.from_euler(axis, degrees, degrees=True).as_matrix()


class SimRig:
    """The sim stereo head at one neck pose, from the profile in depth.json: torso <-> camera, and the lens.

    Mirrors stereo_depth.camera_to_torso and the grader's Rig (reachy-1-2-sim scripts/perception_grade.py,
    @ 7786a65). Only the sim has a torso-frame conversion today (design section 7)."""

    def __init__(self, profile: Mapping[str, Any], neck_deg: Sequence[float]):
        try:
            self.distorted = profile["lens"] == "distorted"
            self.size = tuple(int(v) for v in profile["resolution"])
            self.focal = [float(v) for v in profile["focal_px"]]
            self.warp_fx = [float(v) for v in profile["warp_fx_px"]]
            self.radial = [[float(v) for v in k] for k in profile["radial"]]
            self.pivot = np.array(profile["neck_origin_xyz_m"], dtype=float)
            self._offset = {"right": np.array(profile["right_camera_in_head_m"], dtype=float)}
            baseline = np.array([-float(profile["baseline_m"]), 0.0, 0.0])
            self._offset["left"] = self._offset["right"] + _OPTICAL_TO_HEAD @ baseline
            roll, pitch, yaw = neck_deg
            self.R = (_rot("y", math.degrees(float(profile["neck_origin_pitch_rad"]))) @ _rot("x", roll)
                      @ _rot("y", pitch) @ _rot("z", yaw))
        except (KeyError, TypeError, ValueError) as exc:
            raise SceneRefused("depth.json profile cannot place the cameras: %r" % (exc,)) from exc

    def origin(self, eye: str = "right") -> np.ndarray:
        return self.R @ self._offset[eye] + self.pivot

    def to_torso(self, p_cam, eye: str = "right") -> np.ndarray:
        return self.R @ (_OPTICAL_TO_HEAD @ np.asarray(p_cam, dtype=float) + self._offset[eye]) + self.pivot

    def to_camera(self, p_torso, eye: str = "right") -> np.ndarray:
        head = self.R.T @ (np.asarray(p_torso, dtype=float) - self.pivot)
        return _OPTICAL_TO_HEAD.T @ (head - self._offset[eye])

    def project(self, p_torso, eye: str = "right") -> Optional[Tuple[float, float]]:
        x, y, z = self.to_camera(p_torso, eye)
        if z < 0.05:
            return None
        i = 0 if eye == "left" else 1
        x, y = x / z, y / z
        fx = self.warp_fx[i] if self.distorted else self.focal[i]
        if self.distorted:
            k1, k2 = self.radial[i]
            d = 1 + k1 * (x * x + y * y) + k2 * (x * x + y * y) ** 2
            x, y = x * d, y * d
        return fx * x + self.size[0] / 2, self.focal[i] * y + self.size[1] / 2


# ---------------------------------------------------------------- class sizes and the surface correction

def _support(kind: str, half: Sequence[float], ray: np.ndarray) -> float:
    """Half the extent of an upright piece along the unit ray: the distance from its centre to the visible surface.

    A cylinder is exact and yaw-free. A cube's yaw is unknown, so its horizontal reach is the yaw average,
    (4/pi) * half-side. The yaw error is NOT in the bound: boxes are refused while yaw is unknown (OD-5)."""
    up = abs(float(ray[2]))
    side = math.sqrt(max(0.0, 1.0 - up * up))
    if kind == "cube":
        return half[0] * up + half[0] * side * 4.0 / math.pi
    return half[2] * up + half[0] * side


@dataclass(frozen=True)
class ClassSize:
    """The known sizes of one kind on one rig. The correction uses their midpoint and adds their range."""
    kind: str
    pieces: Tuple[Tuple[str, Tuple[float, float, float]], ...]   # (name, half-extents x, y, z)

    @property
    def half_m(self) -> Tuple[float, float, float]:
        """Midpoint of the smallest and largest known half-extent, per axis."""
        halves = np.array([h for _, h in self.pieces], dtype=float)
        return tuple(float(v) for v in (halves.max(axis=0) + halves.min(axis=0)) / 2.0)

    @property
    def size_m(self) -> Tuple[float, float, float]:
        return tuple(2.0 * h for h in self.half_m)

    def along(self, ray: np.ndarray) -> Tuple[float, float]:
        """(half-extent along the unit ray at the midpoint size, half the spread of it over the known sizes)."""
        known = [_support(self.kind, h, ray) for _, h in self.pieces]
        return _support(self.kind, self.half_m, ray), (max(known) - min(known)) / 2.0


def class_size(rig_layout_id: str, kind: str) -> ClassSize:
    pieces = (SIZE_TABLE.get(rig_layout_id) or {}).get(kind)
    if not pieces:
        raise SceneRefused("no size table for %s on rig layout %r (known: %s): add the owner-measured sizes "
                           "before using %s" % (kind, rig_layout_id, ", ".join(sorted(SIZE_TABLE)), SURFACE_CORRECTION))
    return ClassSize(kind, tuple((name, tuple(half)) for name, half in pieces))


# ---------------------------------------------------------------- the bounds model

@dataclass(frozen=True)
class BoundsRow:
    """One looked-up row (design section 1.6). `kind` None means every kind: #186 supports no per-kind row."""
    detector_model_sha256: str
    depth_method: str
    surface_correction_id: str
    calibration_id: str
    zoom_level: str
    head_pose: str
    kind: Optional[str]
    band: str                                 # a distance band label, or "all" for the pooled row
    bias_m: Tuple[float, float, float]        # subtracted from the centre
    half_width_m: Tuple[float, float, float]  # per-axis half-width at `coverage`
    n_distinct: int
    coverage: float


@dataclass(frozen=True)
class BoundsModel:
    rows: Tuple[BoundsRow, ...]
    band_edges_m: Tuple[float, ...]
    source_sha256: str
    model_id: str
    min_distinct: int = MIN_DISTINCT

    def band(self, distance_m: float) -> str:
        """The grader's band label (perception_grade._band)."""
        lo = None
        for edge in self.band_edges_m:
            if distance_m < edge:
                return "<%.2f m" % edge if lo is None else "%.2f-%.2f m" % (lo, edge)
            lo = edge
        return ">=%.2f m" % self.band_edges_m[-1]

    def lookup(self, *, detector_model_sha256: str, depth_method: str, surface_correction_id: str,
               calibration_id: str, zoom_level: str, head_pose: str, kind: str,
               distance_m: Optional[float]) -> Optional[BoundsRow]:
        """The row for exactly this key. The camera-distance band is tried first, then the pooled row."""
        matching = [r for r in self.rows
                    if (r.detector_model_sha256, r.depth_method, r.surface_correction_id, r.calibration_id,
                        r.zoom_level, r.head_pose) == (detector_model_sha256, depth_method, surface_correction_id,
                                                        calibration_id, zoom_level, head_pose)
                    and r.kind in (None, kind)]
        wanted = [self.band(distance_m)] if distance_m is not None else []
        for band in wanted + ["all"]:
            for row in matching:
                if row.band == band:
                    return row
        return None

    def provisional(self, row: BoundsRow) -> bool:
        return row.n_distinct < self.min_distinct


def bounds_from_grader_result(raw: bytes, *, zoom_level: str, depth_method: str = DEPTH_METHOD,
                              surface_correction_id: str = "none", min_distinct: int = MIN_DISTINCT) -> BoundsModel:
    """Read a perception_grade.v1 result.json (reachy-1-2-sim scripts/perception_grade.py) into bounds rows.

    The result does not record the zoom, the depth method or whether a surface correction was applied, so the
    caller states them (the #186 run: ZERO, the viewer's SGBM, none). Everything else comes from the file:
    the detector, the calibration (profile) id, the head pose (from the graded pairs' neck read-backs), the
    distance bands, and per band and pooled: x and y half-width = the horizontal p95, z half-width = the 3D centre
    p95 (a norm bounds each axis). Bias is zero: the result has no per-axis signed error. `n_distinct` counts
    distinct (layout, object) sightings, because the sim renders are deterministic and a repeated pair adds nothing.
    """
    try:
        result = json.loads(raw)
        if result["schema_version"] != GRADER_SCHEMA:
            raise SceneRefused("grader result schema is %r, not %s" % (result["schema_version"], GRADER_SCHEMA))
        detectors, profiles = result["detector_models"], result["profile_ids"]
        if len(detectors) != 1 or len(profiles) != 1:
            raise SceneRefused("a bounds row needs exactly one detector and one profile; the result has %d and %d"
                               % (len(detectors), len(profiles)))
        edges = tuple(float(e) for e in result["config"]["distance_band_edges_m"])
        layouts = [lay for lay in result["layouts"] if lay.get("accepted")]
        necks = {tuple(round(float(p["neck_deg"][n]), 2) for n in NECK_JOINTS)
                 for lay in layouts for p in lay["pairs"]}
        if len(necks) != 1:
            raise SceneRefused("the graded pairs span %d head poses; a row needs one" % len(necks))
        overall = result["overall"]
        by_band, pooled = overall["error_3d_by_distance_mm"], overall["error_3d_mm"]
    except (KeyError, TypeError, ValueError) as exc:
        raise SceneRefused("not a usable %s result: %r" % (GRADER_SCHEMA, exc)) from exc
    pose = head_pose_name(next(iter(necks)))
    model = BoundsModel((), edges, _sha256(raw), "%s:%s" % (GRADER_SCHEMA, _sha256(raw)[:12]), min_distinct)

    sightings = {}   # band -> {(layout, object_id)}
    for layout in layouts:
        for pair in layout["pairs"]:
            for match in pair["matches"]:
                key = (layout["name"], match["object_id"])
                sightings.setdefault(model.band(match["distance_m"]), set()).add(key)
                sightings.setdefault("all", set()).add(key)

    def row(band: str, errors: Mapping[str, Any]) -> BoundsRow:
        horizontal, centre = errors["horizontal_mm"]["p95"] / 1000.0, errors["centre_mm"]["p95"] / 1000.0
        return BoundsRow(detectors[0], depth_method, surface_correction_id, profiles[0], zoom_level, pose, None, band,
                         (0.0, 0.0, 0.0), (horizontal, horizontal, centre), len(sightings.get(band, ())),
                         BOUNDS_COVERAGE)

    rows = [row(band, errors) for band, errors in by_band.items() if errors["horizontal_mm"].get("n")]
    rows.append(row("all", pooled))
    return BoundsModel(tuple(rows), edges, model.source_sha256, model.model_id, min_distinct)


# ---------------------------------------------------------------- the board

def _analyser():
    """scripts/analyze_capture_bundle.py, whose head analysis the board pose reuses."""
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import analyze_capture_bundle
    return analyze_capture_bundle


IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
BOARD_NOTE = ("board orientation is assumed axis-aligned with the torso (identity quaternion); only the centre is "
              "measured, and board bounds are the observed spread, not a coverage claim")


def board_from_head_sweep(path, expected_source: str = "sim") -> Tuple[Optional[sc.BoardPose], List[str]]:
    """BoardPose from a head-sweep bundle, by analyze_capture_bundle.analyze_head. (pose or None, warnings)."""
    analyse = _analyser()
    manifest, files, capture = analyse.load(Path(path))
    if capture.get("kind") != "head":
        raise SceneRefused("%s is a %r capture, not a head sweep" % (Path(path).name, capture.get("kind")))
    source, lens, _ = analyse.bundle_geometry(capture, files)
    if source != expected_source:
        raise SceneRefused("the head sweep is a %s capture but the pair is a %s capture" % (source, expected_source))
    if source != "sim":
        return None, ["no board pose: the robot torso-frame conversion is not available (design section 7)"]
    depth = PairDepth("sim", lens)
    if depth.geometry.error:
        raise SceneRefused(depth.geometry.error)
    result = analyse.analyze_head(capture, files, depth)
    poses = result["poses"]
    if not poses:
        return None, ["head sweep: the grid was found at no pose, so there is no board pose"]
    centres = np.array([p["centre_torso_m"] for p in poses.values()])
    rms = [r["fit_rms_mm"] for r in result["rows"] if "fit_rms_mm" in r and "centre_torso_m" in r]
    spread = float(result["spread"]["max_deviation_mm"]) if result.get("spread") else 0.0
    fit = float(np.mean(rms)) if rms else 0.0
    half = max(spread, fit) / 1000.0
    pose = sc.BoardPose(tuple(float(v) for v in centres.mean(axis=0)), IDENTITY_QUAT, (half,) * 3,
                        "head_sweep_fit" if len(poses) > 1 else "single_pair_fit", len(poses), fit,
                        spread if len(poses) > 1 else 0.0)
    return pose, [BOARD_NOTE] + list(result.get("findings", []))


def board_from_pair(bundle: PairBundle, neck: Sequence[float]) -> Tuple[Optional[sc.BoardPose], List[str]]:
    """A single-pair fit: the taped grid found in this pair's own two images (sim geometry only)."""
    if bundle.left_jpg is None or bundle.right_jpg is None:
        return None, ["no board pose: the pair bundle has no images to find the grid in"]
    import cv2

    from reachy_ai.perception.stereo_depth import camera_to_torso
    analyse = _analyser()
    profile = bundle.depth.get("profile") or {}
    depth = PairDepth("sim", profile.get("lens", "distorted"))
    left, right = (cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                   for raw in (bundle.left_jpg, bundle.right_jpg))
    if left is None or right is None:
        return None, ["no board pose: the pair images could not be decoded"]
    centre, detail = analyse.board_centre_camera(depth, left, right)
    if centre is None:
        return None, ["no board pose: " + detail.get("reason", "grid not found")]
    torso = camera_to_torso(centre, {n: {"present_deg": v} for n, v in zip(NECK_JOINTS, neck)})
    rms = float(detail["fit_rms_mm"])
    pose = sc.BoardPose(tuple(float(v) for v in torso), IDENTITY_QUAT, (rms / 1000.0,) * 3, "single_pair_fit", 1, rms,
                        0.0)
    return pose, [BOARD_NOTE]


@dataclass(frozen=True)
class _Board:
    pose: sc.BoardPose
    rotation: np.ndarray

    @classmethod
    def of(cls, pose: sc.BoardPose) -> "_Board":
        w, x, y, z = pose.quat_wxyz
        return cls(pose, Rotation.from_quat([x, y, z, w]).as_matrix())

    def to_board(self, p) -> np.ndarray:
        return self.rotation.T @ (np.asarray(p, dtype=float) - np.array(self.pose.position_m))

    def cell_centre(self, name: str) -> np.ndarray:
        """Torso-frame centre of a cell: board +x toward the far row (r3), board +y toward c1."""
        row, col = int(name[1]), int(name[3])
        local = np.array([(row - 2) * CELL_PITCH_M, (2 - col) * CELL_PITCH_M, 0.0])
        return self.rotation @ local + np.array(self.pose.position_m)

    def local_centre(self, name: str) -> np.ndarray:
        row, col = int(name[1]), int(name[3])
        return np.array([(row - 2) * CELL_PITCH_M, (2 - col) * CELL_PITCH_M])


# ---------------------------------------------------------------- the scene

DEFAULT_LABELS = ("cube", "cylinder", "empty")


def _inside(box: Sequence[float], px: Optional[Tuple[float, float]]) -> bool:
    if px is None:
        return False
    x, y, w, h = box
    return x <= px[0] <= x + w and y <= px[1] <= y + h


def build_scene(bundle: PairBundle, *, options: RigOptions = RigOptions(), board: Optional[sc.BoardPose] = None,
                board_warnings: Sequence[str] = (), fit_board_from_pair: bool = True,
                bounds: Optional[BoundsModel] = None, surface_correction: str = "none",
                scene_id: Optional[str] = None, created_utc: Optional[str] = None,
                code: Optional[Tuple[str, bool]] = None) -> sc.PerceivedScene:
    """One PerceivedScene from one pair. Raises SceneRefused for a capture that cannot make a valid one."""
    if surface_correction not in sc.SURFACE_CORRECTIONS:
        raise SceneRefused("unknown surface correction %r" % surface_correction)
    warnings = list(board_warnings)
    rig = build_rig(bundle, options)
    neck = neck_read_back(bundle)
    pose_name = head_pose_name(neck)
    model = bundle.analysis.get("model") or {}
    detector_sha = model.get("sha256")
    if not isinstance(detector_sha, str) or not _SHA256.match(detector_sha):
        raise SceneRefused("analysis.json has no usable detector model sha256 (got %r)" % (detector_sha,))
    labels = tuple(model.get("labels") or ())
    if not labels:
        labels = DEFAULT_LABELS
        warnings.append("analysis.json does not record the detector labels; assumed %s" % (list(DEFAULT_LABELS),))
    if bundle.left_sha256 is None or bundle.right_sha256 is None:
        raise SceneRefused("left.jpg and right.jpg are not in the bundle: the source images cannot be hashed")
    captured = options.captured_utc or bundle.pair.get("captured_utc")
    if not captured:
        captured = bundle.file_time_utc
        warnings.append("captured_utc is the bundle file's modification time: the v1 camera message has no source "
                        "timestamp")
    source_pair = sc.SourcePair(str(bundle.pair.get("pair_id") or bundle.analysis.get("pair_id") or "unknown"),
                                bundle.bundle_sha256, bundle.left_sha256, bundle.right_sha256, captured, neck)

    sim = SimRig(bundle.depth.get("profile") or {}, neck) if rig.source == "sim" else None
    if surface_correction != "none" and sim is None:
        raise SceneRefused("%s needs the camera position in the torso frame, which only the sim has today "
                           "(the robot torso-frame conversion is a prerequisite)" % surface_correction)
    if board is None and fit_board_from_pair and sim is not None:
        board, found = board_from_pair(bundle, neck)
        warnings.extend(found)
    elif board is None and sim is None:
        warnings.append("no board pose: the robot torso-frame conversion is not available (design section 7)")
    frame = _Board.of(board) if board is not None else None
    if frame is None:
        warnings.append("no board: no cells are reported and nothing is classified off-board")

    measurements = {m.get("detection_index"): m for m in bundle.depth.get("measurements") or []}
    detections = bundle.analysis.get("detections") or []
    camera = sim.origin("right") if sim is not None else None
    built, unmeasured, empties = [], [], []
    for index, det in enumerate(detections):
        label, box = det.get("label"), det.get("box_xywh")
        if label == "empty":
            empties.append(box)
            continue
        if label not in sc.KINDS:
            warnings.append("detection %d has the label %r, which is not in the vocabulary; ignored" % (index, label))
            continue
        m = measurements.get(index)
        if m is None or m.get("status") != "measured" or "xyz_torso_m" not in m:
            why = "no torso position" if m is not None and m.get("status") == "measured" else "no 3D measurement"
            warnings.append("detection %d (%s) has %s; it is not an object, and its cell is unknown"
                            % (index, label, why))
            unmeasured.append(box)
            continue
        if m.get("label") != label:
            raise SceneRefused("detection %d is %r in analysis.json but %r in depth.json"
                               % (index, label, m.get("label")))
        built.append(_perceive(index, det, m, rig, source_pair, camera, bounds, detector_sha, pose_name,
                               surface_correction, warnings))

    on_board, off_board = [], []
    for item in built:
        local = frame.to_board(item["centre"]) if frame is not None else None
        reach_m = CELL_PITCH_M + CELL_HALF_M    # the grid's outer edge
        on = frame is None or (abs(local[0]) <= reach_m and abs(local[1]) <= reach_m
                               and BOARD_Z_RANGE_M[0] <= local[2] <= BOARD_Z_RANGE_M[1])
        item["local"] = local
        (on_board if on else off_board).append(item)
    if off_board and frame is not None:
        warnings.append("%d detection(s) lie off the board and are listed in off_board" % len(off_board))

    cells = _cells(frame, sim, on_board, empties, unmeasured, warnings) if frame is not None else ()
    objects = tuple(_object(item, source_pair, rig) for item in on_board)
    off = tuple(_object(dict(item, cell=None), source_pair, rig) for item in off_board)
    sha, dirty = code if code is not None else code_identity()
    try:
        return sc.PerceivedScene(
            sc.SCHEMA_VERSION, scene_id or uuid.uuid4().hex, created_utc or utc_now(), rig,
            sc.PipelineIdentity(detector_sha, labels, DEPTH_METHOD, surface_correction,
                                bounds.model_id if bounds else "none", sha, dirty),
            board, cells, objects, off, tuple(warnings))
    except sc.SceneContractError as exc:
        raise SceneRefused("the scene is not valid: %s" % exc) from exc


def _perceive(index, det, m, rig, source_pair, camera, bounds, detector_sha, pose_name, correction, warnings):
    """Everything about one measured detection except its cell."""
    kind = det["label"]
    surface = np.array(m["xyz_torso_m"], dtype=float)
    centre, size, distance, ray_range = surface.copy(), None, None, 0.0
    if correction != "none":
        ray = surface - camera
        ray = ray / np.linalg.norm(ray)
        size = class_size(rig.rig_layout_id, kind)
        offset, ray_range = size.along(ray)
        centre = surface + ray * offset
    if camera is not None:
        distance = float(np.linalg.norm(centre - camera))
    row = None
    if bounds is not None:
        row = bounds.lookup(detector_model_sha256=detector_sha, depth_method=DEPTH_METHOD,
                            surface_correction_id=correction, calibration_id=rig.calibration_id,
                            zoom_level=rig.zoom_level, head_pose=pose_name, kind=kind, distance_m=distance)
    if row is None:
        half, provisional, coverage = (math.inf,) * 3, True, BOUNDS_COVERAGE
    else:
        centre = centre - np.array(row.bias_m)
        half = tuple(h + ray_range for h in row.half_width_m)
        provisional, coverage = bounds.provisional(row), row.coverage
    for field_name in ("coverage", "depth_iqr_m"):
        if field_name not in m:
            warnings.append("detection %d: depth.json has no %s; recorded as 0" % (index, field_name))
    return {"kind": kind, "confidence": det.get("confidence"), "centre": centre, "surface": surface, "half": half,
            "provisional": provisional, "coverage": coverage, "size": size, "samples": int(m.get("samples", 0)),
            "stereo_coverage": float(m.get("coverage", 0.0)), "iqr": float(m.get("depth_iqr_m", 0.0)), "cell": None}


def _object(item, source_pair, rig) -> sc.PerceivedObject:
    size = item["size"]
    return sc.PerceivedObject(
        kind=item["kind"], kind_confidence=item["confidence"], centre_m=tuple(float(v) for v in item["centre"]),
        surface_point_m=tuple(float(v) for v in item["surface"]), bounds_m=item["half"],
        bounds_coverage=item["coverage"],
        bounds_provisional=item["provisional"], quat_wxyz=None, yaw_bound_deg=None,
        size_m=None if size is None else size.size_m, size_source="unknown" if size is None else "class_prior",
        cell=item["cell"], stereo_samples=item["samples"], stereo_coverage=item["stereo_coverage"],
        depth_iqr_m=item["iqr"], source=source_pair, calibration_id=rig.calibration_id)


def _cells(frame, sim, on_board, empties, unmeasured, warnings) -> Tuple[sc.CellReport, ...]:
    """Nine cells, always. A cell is cube/cylinder only for one object that fits its footprint, `empty` only when an
    empty detection covers it and nothing is unsettled, and `unknown` otherwise: unknown is never empty."""
    names = sc.CELL_NAMES
    unknown, claimed = set(), {}
    rotation = np.abs(frame.rotation)
    margin = np.array(frame.pose.bounds_m)
    for item in on_board:
        half = np.array(item["half"])
        infinite = not np.all(np.isfinite(half))
        # Torso-axis half-widths (plus the board's own) as board-axis half-widths; an infinite bound has no reach.
        reach = rotation.T @ (np.where(np.isfinite(half), half, 0.0) + margin)
        item["reach"] = None if infinite else reach
        local = item["local"]
        inside = [n for n in names if np.all(np.abs(local[:2] - frame.local_centre(n)) <= CELL_HALF_M)]
        if inside and not infinite and np.all(
                np.abs(local[:2] - frame.local_centre(inside[0])) + reach[:2] <= CELL_HALF_M):
            claimed.setdefault(inside[0], []).append(item)
    for name, items in claimed.items():
        if len(items) == 1:
            items[0]["cell"] = name
        else:
            warnings.append("%d detections claim cell %s; none is assigned and the cell is unknown"
                            % (len(items), name))
            unknown.add(name)
    claimed = {n: i[0] for n, i in claimed.items() if len(i) == 1}
    # An object that is not assigned still occupies ground: every cell it could touch is unknown. With an infinite
    # bound it could be anywhere, so the whole board is.
    for item in on_board:
        if item["cell"] is None:
            for name in names:
                if item["reach"] is None or np.all(
                        np.abs(item["local"][:2] - frame.local_centre(name)) <= CELL_HALF_M + item["reach"][:2]):
                    unknown.add(name)
    pixels = {}
    if sim is not None:
        pixels = {n: sim.project(frame.cell_centre(n), "right") for n in names}
    for box in unmeasured:     # a cube or cylinder was seen here, but not measured
        unknown.update(n for n in names if _inside(box, pixels.get(n)))
    empty_cells = set()
    for box in empties:
        near = [(math.hypot(pixels[n][0] - (box[0] + box[2] / 2), pixels[n][1] - (box[1] + box[3] / 2)), n)
                for n in names if _inside(box, pixels.get(n))]
        if near:
            empty_cells.add(min(near)[1])
    for name in sorted(set(claimed) & empty_cells):
        warnings.append("cell %s has an object and an empty detection; it is unknown" % name)
        claimed[name]["cell"] = None
        unknown.add(name)
    order = {id(item): i for i, item in enumerate(on_board)}
    reports = []
    for name in names:
        item = claimed.get(name)
        if name in unknown or (item is None and name not in empty_cells):
            reports.append(sc.CellReport(name, "unknown", None))
        elif item is not None:
            reports.append(sc.CellReport(name, item["kind"], order[id(item)]))
        else:
            reports.append(sc.CellReport(name, "empty", None))
    return tuple(reports)
