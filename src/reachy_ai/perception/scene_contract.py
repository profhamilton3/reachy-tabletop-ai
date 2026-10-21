"""The perceived scene: one wire contract for #182 stage 2 and EPIC 9 (docs/design/perceived-scene.md, section 1).

Pure data and validation. No cv2, numpy, SDK or network, and no motion: nothing here can move the robot.

PYTHON 3.8: the sim vendors this file into its ros:foxy panel container (Python 3.8). Use typing.Tuple and
typing.Optional only: no "X | Y", no builtin generics, no walrus-free-only syntax later than 3.8.
tests/test_scene_contract.py parses this file as 3.8 and checks for both.

Strict by design, like the sim's truth.v1: unknown fields, missing fields, NaN/inf, a non-unit quaternion, a
board with other than nine cells and a bad zoom are each refused. Frozen dataclasses, validation in
__post_init__, so a hand-built object is as strict as a parsed one.

Wire form: `dumps()` is canonical (sorted keys, 2-space indent, ASCII, trailing newline), so two repos can
compare a fixture byte for byte. An infinite bound (a provisional object with no bounds row) travels as null,
because JSON has no infinity; from_dict turns it back into inf and accepts it only when the object is provisional.
"""
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

SCHEMA_VERSION = "perceived_scene.v1"
IDENTITY_VERSION = 1

ZOOM_LEVELS = ("OUT", "INTER", "IN", "ZERO")
SOURCES = ("sim", "robot")
KINDS = ("cube", "cylinder")
CELL_STATES = ("cube", "cylinder", "empty", "unknown")
CELL_NAMES = tuple("r%dc%d" % (row, col) for row in (1, 2, 3) for col in (1, 2, 3))
SURFACE_CORRECTIONS = ("none", "class-half-extent.v1")
BOARD_SOURCES = ("head_sweep_fit", "single_pair_fit", "twin", "taught")
SIZE_SOURCES = ("unknown", "class_prior", "measured")

#: A rig change makes memory stale: any difference in these is a rejection, never a score.
MATCH_FIELDS = ("source", "rig_layout_id", "mount_id", "calibration_id", "zoom_level",
                "intrinsics_sha256", "rig_epoch", "scene_sha256", "scene_revision")

#: |norm - 1| allowed for a quaternion.
QUAT_TOLERANCE = 1e-6

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CODE_SHA = re.compile(r"^[0-9a-f]{7,40}$")
_SCENE_ID = re.compile(r"^[0-9a-f]{32}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


class SceneContractError(ValueError):
    """A document or value that is not a valid perceived_scene.v1."""


# ---------------------------------------------------------------- validation helpers

def _fail(where: str, why: str) -> None:
    raise SceneContractError("%s: %s" % (where, why))


def _keys(where: str, data: Any, required: Tuple[str, ...], optional: Tuple[str, ...] = ()) -> Dict[str, Any]:
    if not isinstance(data, dict):
        _fail(where, "expected an object, got %s" % type(data).__name__)
    unknown = sorted(set(data) - set(required) - set(optional))
    if unknown:
        _fail(where, "unknown field(s): %s" % ", ".join(unknown))
    missing = [k for k in required if k not in data]
    if missing:
        _fail(where, "missing field(s): %s" % ", ".join(missing))
    return data


def _str(where: str, value: Any, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        _fail(where, "expected a string, got %r" % (value,))
    if not value and not allow_empty:
        _fail(where, "must not be empty")
    return value


def _choice(where: str, value: Any, choices: Tuple[str, ...]) -> str:
    if value not in choices:
        _fail(where, "%r is not one of %s" % (value, ", ".join(choices)))
    return value


def _bool(where: str, value: Any) -> bool:
    if not isinstance(value, bool):
        _fail(where, "expected true or false, got %r" % (value,))
    return value


def _int(where: str, value: Any, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(where, "expected an integer, got %r" % (value,))
    if value < minimum:
        _fail(where, "%d is below the minimum %d" % (value, minimum))
    return value


def _float(where: str, value: Any, allow_inf: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(where, "expected a number, got %r" % (value,))
    value = float(value)
    if math.isnan(value) or (math.isinf(value) and not allow_inf):
        _fail(where, "must be finite, got %r" % (value,))
    return value


def _unit_interval(where: str, value: Any) -> float:
    value = _float(where, value)
    if not 0.0 <= value <= 1.0:
        _fail(where, "%r is outside [0, 1]" % (value,))
    return value


def _non_negative(where: str, value: Any, allow_inf: bool = False) -> float:
    value = _float(where, value, allow_inf)
    if value < 0.0:
        _fail(where, "%r is negative" % (value,))
    return value


def _vec(where: str, value: Any, size: int, allow_inf: bool = False) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        _fail(where, "expected %d numbers, got %r" % (size, value))
    return tuple(_float("%s[%d]" % (where, i), v, allow_inf) for i, v in enumerate(value))


def _quat(where: str, value: Any) -> Tuple[float, ...]:
    q = _vec(where, value, 4)
    norm = math.sqrt(sum(c * c for c in q))
    if abs(norm - 1.0) > QUAT_TOLERANCE:
        _fail(where, "quaternion [w, x, y, z] must have unit norm within %g, norm is %r" % (QUAT_TOLERANCE, norm))
    return q


def _sha256(where: str, value: Any) -> str:
    if not isinstance(value, str) or not _SHA256.match(value):
        _fail(where, "expected 64 lowercase hex digits, got %r" % (value,))
    return value


def _utc(where: str, value: Any) -> str:
    if not isinstance(value, str) or not _UTC.match(value):
        _fail(where, "expected a UTC time like 2026-10-21T03:00:19Z, got %r" % (value,))
    return value


def _set(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


def _out_vec(values: Tuple[float, ...]) -> list:
    """A vector for JSON: an infinite entry becomes null (JSON has no infinity)."""
    return [None if math.isinf(v) else v for v in values]


def _opt(value: Any, convert) -> Any:
    return None if value is None else convert(value)


# ---------------------------------------------------------------- identities

@dataclass(frozen=True)
class RigIdentity:
    """The world the perception was made in. A field mismatch is a rejection, never a score."""
    identity_version: int
    source: str                      # "sim" | "robot"
    rig_layout_id: str               # measured scene the rig is modelled by, e.g. "FWDCenterLabMCC"
    mount_id: str                    # robot mount, e.g. "october-2026"
    calibration_id: str              # as recorded: robot "<profile>@zoom=OUT"; sim the plain profile id
    zoom_level: str                  # camera read-back: "OUT" | "INTER" | "IN" | "ZERO"; both eyes agree
    intrinsics_sha256: str           # robot: SHA-256 of the stereo profile file; sim: the geometry id
    rig_epoch: int
    scene_name: Optional[str] = None         # sim only
    scene_sha256: Optional[str] = None       # sim only
    scene_revision: Optional[str] = None     # sim only; required for any sim motion use

    _REQUIRED = ("identity_version", "source", "rig_layout_id", "mount_id", "calibration_id", "zoom_level",
                 "intrinsics_sha256", "rig_epoch")
    _OPTIONAL = ("scene_name", "scene_sha256", "scene_revision")

    def __post_init__(self):
        w = "RigIdentity"
        if _int(w + ".identity_version", self.identity_version) != IDENTITY_VERSION:
            _fail(w, "identity_version %r is not %d" % (self.identity_version, IDENTITY_VERSION))
        _choice(w + ".source", self.source, SOURCES)
        for name in ("rig_layout_id", "mount_id", "calibration_id", "intrinsics_sha256"):
            _str("%s.%s" % (w, name), getattr(self, name))
        # A bad zoom (unreadable, eyes disagreeing) never reaches here as a level: the builder refuses first.
        _choice(w + ".zoom_level", self.zoom_level, ZOOM_LEVELS)
        _int(w + ".rig_epoch", self.rig_epoch)
        if self.scene_name is not None:
            _str(w + ".scene_name", self.scene_name)
        if self.scene_sha256 is not None:
            _sha256(w + ".scene_sha256", self.scene_sha256)
        if self.scene_revision is not None:
            _str(w + ".scene_revision", self.scene_revision)
        if self.source == "robot":
            if any(v is not None for v in (self.scene_name, self.scene_sha256, self.scene_revision)):
                _fail(w, "scene_name, scene_sha256 and scene_revision are sim-only")
            _sha256(w + ".intrinsics_sha256", self.intrinsics_sha256)
            if not self.calibration_id.endswith("@zoom=" + self.zoom_level):
                _fail(w, "robot calibration_id %r does not carry the zoom_level %s" % (self.calibration_id,
                                                                                       self.zoom_level))

    def to_dict(self) -> Dict[str, Any]:
        out = {k: getattr(self, k) for k in self._REQUIRED}
        out.update({k: getattr(self, k) for k in self._OPTIONAL})
        return out

    @classmethod
    def from_dict(cls, data: Any) -> "RigIdentity":
        _keys("RigIdentity", data, cls._REQUIRED, cls._OPTIONAL)
        return cls(**data)


@dataclass(frozen=True)
class PipelineIdentity:
    """What produced the numbers. A pipeline change makes bounds stale; a rig change makes memory stale."""
    detector_model_sha256: str
    detector_labels: Tuple[str, ...]  # ("cube", "cylinder", "empty")
    depth_method: str                 # "sgbm-bidir-1px/inner60-median"
    surface_correction_id: str        # "none" | "class-half-extent.v1"
    bounds_model_id: str              # which BoundsModel rows produced the bounds ("none" if no model)
    code_sha: str                     # tabletop-ai commit; "" with dirty=True refuses motion use
    dirty: bool

    _FIELDS = ("detector_model_sha256", "detector_labels", "depth_method", "surface_correction_id",
               "bounds_model_id", "code_sha", "dirty")

    def __post_init__(self):
        w = "PipelineIdentity"
        _sha256(w + ".detector_model_sha256", self.detector_model_sha256)
        labels = self.detector_labels
        if not isinstance(labels, (list, tuple)) or not labels:
            _fail(w + ".detector_labels", "expected a non-empty list of strings")
        _set(self, "detector_labels", tuple(_str("%s.detector_labels[%d]" % (w, i), v)
                                            for i, v in enumerate(labels)))
        _str(w + ".depth_method", self.depth_method)
        _choice(w + ".surface_correction_id", self.surface_correction_id, SURFACE_CORRECTIONS)
        _str(w + ".bounds_model_id", self.bounds_model_id)
        _str(w + ".code_sha", self.code_sha, allow_empty=True)
        _bool(w + ".dirty", self.dirty)
        if self.code_sha == "":
            if not self.dirty:
                _fail(w, "an empty code_sha needs dirty=true")
        elif not _CODE_SHA.match(self.code_sha):
            _fail(w + ".code_sha", "expected 7 to 40 lowercase hex digits, got %r" % (self.code_sha,))

    def to_dict(self) -> Dict[str, Any]:
        out = {k: getattr(self, k) for k in self._FIELDS}
        out["detector_labels"] = list(self.detector_labels)
        return out

    @classmethod
    def from_dict(cls, data: Any) -> "PipelineIdentity":
        _keys("PipelineIdentity", data, cls._FIELDS)
        return cls(**data)


# ---------------------------------------------------------------- board and cells

@dataclass(frozen=True)
class BoardPose:
    position_m: Tuple[float, float, float]        # taped-grid centre (r2c2), torso frame
    quat_wxyz: Tuple[float, float, float, float]  # board axes: +x toward far row, +y toward c1
    bounds_m: Tuple[float, float, float]          # per-axis half-width, torso axes
    source: str                  # "head_sweep_fit" | "single_pair_fit" | "twin" | "taught"
    poses_used: int              # head poses that found the grid
    fit_rms_mm: float            # Kabsch residual
    spread_mm: float             # worst deviation from the mean over poses; 0 for one pose

    _FIELDS = ("position_m", "quat_wxyz", "bounds_m", "source", "poses_used", "fit_rms_mm", "spread_mm")

    def __post_init__(self):
        w = "BoardPose"
        _set(self, "position_m", _vec(w + ".position_m", self.position_m, 3))
        _set(self, "quat_wxyz", _quat(w + ".quat_wxyz", self.quat_wxyz))
        _set(self, "bounds_m", tuple(_non_negative("%s.bounds_m[%d]" % (w, i), v)
                                     for i, v in enumerate(_vec(w + ".bounds_m", self.bounds_m, 3))))
        _choice(w + ".source", self.source, BOARD_SOURCES)
        _int(w + ".poses_used", self.poses_used)
        _non_negative(w + ".fit_rms_mm", self.fit_rms_mm)
        _non_negative(w + ".spread_mm", self.spread_mm)
        if self.source == "head_sweep_fit" and self.poses_used < 2:
            _fail(w, "a head_sweep_fit needs at least two poses, got %d" % self.poses_used)
        if self.source == "single_pair_fit" and (self.poses_used != 1 or self.spread_mm != 0.0):
            _fail(w, "a single_pair_fit is one pose with spread_mm 0")

    def to_dict(self) -> Dict[str, Any]:
        out = {k: getattr(self, k) for k in self._FIELDS}
        for k in ("position_m", "quat_wxyz", "bounds_m"):
            out[k] = list(out[k])
        return out

    @classmethod
    def from_dict(cls, data: Any) -> "BoardPose":
        _keys("BoardPose", data, cls._FIELDS)
        return cls(**data)


@dataclass(frozen=True)
class CellReport:
    name: str                    # "r1c1" ... "r3c3"
    state: str                   # "cube" | "cylinder" | "empty" | "unknown"
    object_index: Optional[int]  # into PerceivedScene.objects when state is cube/cylinder

    _FIELDS = ("name", "state", "object_index")

    def __post_init__(self):
        w = "CellReport"
        _choice(w + ".name", self.name, CELL_NAMES)
        _choice(w + ".state", self.state, CELL_STATES)
        holds_object = self.state in KINDS
        if holds_object:
            _int("%s(%s).object_index" % (w, self.name), self.object_index)
        elif self.object_index is not None:
            _fail("%s(%s)" % (w, self.name), "object_index is only for cube or cylinder cells")

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self._FIELDS}

    @classmethod
    def from_dict(cls, data: Any) -> "CellReport":
        _keys("CellReport", data, cls._FIELDS)
        return cls(**data)


# ---------------------------------------------------------------- objects

@dataclass(frozen=True)
class SourcePair:
    pair_id: str
    bundle_sha256: str                    # the viewer's single-pair bundle hash
    left_sha256: str
    right_sha256: str
    captured_utc: str                     # when the workstation received the pair
    neck_deg: Tuple[float, float, float]  # read-back (roll, pitch, yaw), not the command

    _FIELDS = ("pair_id", "bundle_sha256", "left_sha256", "right_sha256", "captured_utc", "neck_deg")

    def __post_init__(self):
        w = "SourcePair"
        _str(w + ".pair_id", self.pair_id)
        for name in ("bundle_sha256", "left_sha256", "right_sha256"):
            _sha256("%s.%s" % (w, name), getattr(self, name))
        _utc(w + ".captured_utc", self.captured_utc)
        _set(self, "neck_deg", _vec(w + ".neck_deg", self.neck_deg, 3))

    def to_dict(self) -> Dict[str, Any]:
        out = {k: getattr(self, k) for k in self._FIELDS}
        out["neck_deg"] = list(self.neck_deg)
        return out

    @classmethod
    def from_dict(cls, data: Any) -> "SourcePair":
        _keys("SourcePair", data, cls._FIELDS)
        return cls(**data)


@dataclass(frozen=True)
class PerceivedObject:
    kind: str                                  # "cube" | "cylinder" (the detector vocabulary minus "empty")
    kind_confidence: float                     # detector score
    centre_m: Tuple[float, float, float]       # torso frame, AFTER the surface correction
    surface_point_m: Tuple[float, float, float]  # raw stereo median (visible surface), for audit
    bounds_m: Tuple[float, float, float]       # per-axis half-width on centre_m at bounds_coverage
    bounds_coverage: float                     # e.g. 0.95
    bounds_provisional: bool                   # True: may only refuse; bounds may be infinite
    quat_wxyz: Optional[Tuple[float, float, float, float]]  # None today: no yaw is estimated
    yaw_bound_deg: Optional[float]
    size_m: Optional[Tuple[float, float, float]]  # full extents; None if unknown
    size_source: str                           # "unknown" | "class_prior" | "measured"
    cell: Optional[str]                        # cell whose footprint holds centre_m +- bounds; None if ambiguous
    stereo_samples: int
    stereo_coverage: float
    depth_iqr_m: float
    source: SourcePair
    calibration_id: str                        # repeated from the rig, so an object stands alone in logs

    _FIELDS = ("kind", "kind_confidence", "centre_m", "surface_point_m", "bounds_m", "bounds_coverage",
               "bounds_provisional", "quat_wxyz", "yaw_bound_deg", "size_m", "size_source", "cell",
               "stereo_samples", "stereo_coverage", "depth_iqr_m", "source", "calibration_id")

    def __post_init__(self):
        w = "PerceivedObject"
        _choice(w + ".kind", self.kind, KINDS)
        _unit_interval(w + ".kind_confidence", self.kind_confidence)
        _set(self, "centre_m", _vec(w + ".centre_m", self.centre_m, 3))
        _set(self, "surface_point_m", _vec(w + ".surface_point_m", self.surface_point_m, 3))
        provisional = _bool(w + ".bounds_provisional", self.bounds_provisional)
        bounds = tuple(_non_negative("%s.bounds_m[%d]" % (w, i), v, allow_inf=provisional)
                       for i, v in enumerate(_vec(w + ".bounds_m", self.bounds_m, 3, allow_inf=provisional)))
        _set(self, "bounds_m", bounds)
        coverage = _float(w + ".bounds_coverage", self.bounds_coverage)
        if not 0.0 < coverage <= 1.0:
            _fail(w + ".bounds_coverage", "%r is outside (0, 1]" % (coverage,))
        if self.quat_wxyz is None:
            if self.yaw_bound_deg is not None:
                _fail(w, "yaw_bound_deg needs a quat_wxyz")
        else:
            _set(self, "quat_wxyz", _quat(w + ".quat_wxyz", self.quat_wxyz))
            if self.yaw_bound_deg is None:
                _fail(w, "a quat_wxyz needs a yaw_bound_deg")
        if self.yaw_bound_deg is not None:
            _non_negative(w + ".yaw_bound_deg", self.yaw_bound_deg)
        _choice(w + ".size_source", self.size_source, SIZE_SOURCES)
        if self.size_m is None:
            if self.size_source != "unknown":
                _fail(w, "size_source %s needs a size_m" % self.size_source)
        else:
            if self.size_source == "unknown":
                _fail(w, "size_source unknown cannot have a size_m")
            size = _vec(w + ".size_m", self.size_m, 3)
            if min(size) <= 0.0:
                _fail(w + ".size_m", "extents must be positive, got %r" % (size,))
            _set(self, "size_m", size)
        if self.cell is not None:
            _choice(w + ".cell", self.cell, CELL_NAMES)
        _int(w + ".stereo_samples", self.stereo_samples)
        _unit_interval(w + ".stereo_coverage", self.stereo_coverage)
        _non_negative(w + ".depth_iqr_m", self.depth_iqr_m)
        if not isinstance(self.source, SourcePair):
            _fail(w + ".source", "expected a SourcePair")
        _str(w + ".calibration_id", self.calibration_id)

    def to_dict(self) -> Dict[str, Any]:
        out = {k: getattr(self, k) for k in self._FIELDS}
        for k in ("centre_m", "surface_point_m"):
            out[k] = list(out[k])
        out["bounds_m"] = _out_vec(self.bounds_m)
        for k in ("quat_wxyz", "size_m"):
            out[k] = None if out[k] is None else list(out[k])
        out["source"] = self.source.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Any) -> "PerceivedObject":
        _keys("PerceivedObject", data, cls._FIELDS)
        data = dict(data)
        data["source"] = SourcePair.from_dict(data["source"])
        bounds = data["bounds_m"]
        if isinstance(bounds, list):
            # null is how an infinite bound travels; it is legal only for a provisional object.
            data["bounds_m"] = [float("inf") if b is None else b for b in bounds]
        return cls(**data)


@dataclass(frozen=True)
class PerceivedScene:
    schema_version: str                        # "perceived_scene.v1"
    scene_id: str                              # uuid4 hex
    created_utc: str
    rig: RigIdentity
    pipeline: PipelineIdentity
    board: Optional[BoardPose]                 # None: no cells, and no motion use
    cells: Tuple[CellReport, ...]              # nine, in CELL_NAMES order, or () when board is None
    objects: Tuple[PerceivedObject, ...]
    off_board: Tuple[PerceivedObject, ...]     # outside the board footprint; never obstacles or targets
    warnings: Tuple[str, ...]

    _FIELDS = ("schema_version", "scene_id", "created_utc", "rig", "pipeline", "board", "cells", "objects",
               "off_board", "warnings")

    def __post_init__(self):
        w = "PerceivedScene"
        if self.schema_version != SCHEMA_VERSION:
            _fail(w + ".schema_version", "%r is not %s" % (self.schema_version, SCHEMA_VERSION))
        if not isinstance(self.scene_id, str) or not _SCENE_ID.match(self.scene_id):
            _fail(w + ".scene_id", "expected 32 lowercase hex digits, got %r" % (self.scene_id,))
        _utc(w + ".created_utc", self.created_utc)
        if not isinstance(self.rig, RigIdentity):
            _fail(w + ".rig", "expected a RigIdentity")
        if not isinstance(self.pipeline, PipelineIdentity):
            _fail(w + ".pipeline", "expected a PipelineIdentity")
        if self.board is not None and not isinstance(self.board, BoardPose):
            _fail(w + ".board", "expected a BoardPose or None")
        for name, kind in (("cells", CellReport), ("objects", PerceivedObject), ("off_board", PerceivedObject)):
            seq = getattr(self, name)
            if not isinstance(seq, (list, tuple)) or not all(isinstance(v, kind) for v in seq):
                _fail("%s.%s" % (w, name), "expected a list of %s" % kind.__name__)
            _set(self, name, tuple(seq))
        if not isinstance(self.warnings, (list, tuple)):
            _fail(w + ".warnings", "expected a list of strings")
        _set(self, "warnings", tuple(_str("%s.warnings[%d]" % (w, i), v) for i, v in enumerate(self.warnings)))
        self._check_cells()
        self._check_objects()

    def _check_cells(self) -> None:
        w = "PerceivedScene"
        if self.board is None:
            if self.cells:
                _fail(w + ".cells", "a scene without a board has no cells")
            return
        names = tuple(c.name for c in self.cells)
        if names != CELL_NAMES:
            _fail(w + ".cells", "a board needs exactly the nine cells %s in that order, got %d: %s"
                  % (", ".join(CELL_NAMES), len(names), ", ".join(names)))
        used = set()
        for cell in self.cells:
            if cell.object_index is None:
                continue
            if cell.object_index >= len(self.objects):
                _fail("%s.cells[%s]" % (w, cell.name), "object_index %d is out of range" % cell.object_index)
            if cell.object_index in used:
                _fail("%s.cells[%s]" % (w, cell.name), "object %d is reported by two cells" % cell.object_index)
            used.add(cell.object_index)
            obj = self.objects[cell.object_index]
            if obj.kind != cell.state:
                _fail("%s.cells[%s]" % (w, cell.name), "state %s but the object is a %s" % (cell.state, obj.kind))
            if obj.cell != cell.name:
                _fail("%s.cells[%s]" % (w, cell.name), "object %d says cell %r" % (cell.object_index, obj.cell))

    def _check_objects(self) -> None:
        w = "PerceivedScene"
        for name in ("objects", "off_board"):
            for i, obj in enumerate(getattr(self, name)):
                if obj.calibration_id != self.rig.calibration_id:
                    _fail("%s.%s[%d]" % (w, name, i), "calibration_id %r differs from the rig's %r"
                          % (obj.calibration_id, self.rig.calibration_id))
        if self.board is None and any(o.cell is not None for o in self.objects):
            _fail(w + ".objects", "without a board no object can name a cell")
        for i, obj in enumerate(self.off_board):
            if obj.cell is not None:
                _fail("%s.off_board[%d]" % (w, i), "an off-board detection has no cell")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "created_utc": self.created_utc,
            "rig": self.rig.to_dict(),
            "pipeline": self.pipeline.to_dict(),
            "board": None if self.board is None else self.board.to_dict(),
            "cells": [c.to_dict() for c in self.cells],
            "objects": [o.to_dict() for o in self.objects],
            "off_board": [o.to_dict() for o in self.off_board],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "PerceivedScene":
        _keys("PerceivedScene", data, cls._FIELDS)
        for name in ("cells", "objects", "off_board", "warnings"):
            if not isinstance(data[name], list):
                _fail("PerceivedScene." + name, "expected a list")
        return cls(
            schema_version=data["schema_version"], scene_id=data["scene_id"], created_utc=data["created_utc"],
            rig=RigIdentity.from_dict(data["rig"]), pipeline=PipelineIdentity.from_dict(data["pipeline"]),
            board=_opt(data["board"], BoardPose.from_dict),
            cells=tuple(CellReport.from_dict(c) for c in data["cells"]),
            objects=tuple(PerceivedObject.from_dict(o) for o in data["objects"]),
            off_board=tuple(PerceivedObject.from_dict(o) for o in data["off_board"]),
            warnings=tuple(data["warnings"]))


# ---------------------------------------------------------------- wire form

def _reject_constant(name: str) -> None:
    raise SceneContractError("%s is not valid JSON: NaN and infinity are refused" % name)


def _no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise SceneContractError("duplicate key %r" % key)
        out[key] = value
    return out


def dumps(scene: PerceivedScene) -> str:
    """The canonical wire form: sorted keys, 2-space indent, ASCII, one trailing newline."""
    return json.dumps(scene.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"


def loads(text: str) -> PerceivedScene:
    """Parse and validate a perceived_scene.v1 document. Raises SceneContractError for anything else."""
    try:
        data = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_no_duplicates)
    except json.JSONDecodeError as exc:
        raise SceneContractError("not valid JSON: %s" % exc) from exc
    return PerceivedScene.from_dict(data)
