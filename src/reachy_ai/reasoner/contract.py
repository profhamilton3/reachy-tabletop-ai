"""disposition_decision.v1: the reasoner's wire contract (#81, part of #80). A request goes in, a decision comes out.

Pure data and validation. No cv2, numpy, yaml, SDK or network, and no motion: nothing here can move the robot.

PYTHON 3.8: the sim vendors this file (reachy-1-2-sim#216), as it vendors perception/scene_contract.py. Use
typing.Tuple and typing.Optional only: no "X | Y" and no builtin generics. tests/test_disposition_contract.py
parses this file as 3.8 and checks for both.

Strict by design, on the scene_contract.py pattern: unknown fields, missing fields, NaN/inf, bad enum values and
duplicate keys are each refused. Frozen dataclasses with validation in __post_init__, so a hand-built object is
as strict as a parsed one.

Wire form: `dumps()` is canonical (sorted keys, 2-space indent, ASCII, trailing newline), so two repos can
compare a fixture byte for byte. A request and a decision are separate documents: `loads_request()` and
`loads_decision()`.
"""
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

SCHEMA_VERSION = "disposition_decision.v1"

INPUT_SOURCES = ("perceived", "sim_truth")
SOURCES = ("robot", "sim")
SPLITS = ("development", "held_out")
KINDS = ("block", "cylinder", "unknown")
HOLE_STATES = ("present", "absent", "unknown")
ORIENTATIONS = ("upright", "lying", "rotated", "unknown")
SUFFICIENCY_FLAGS = ("blurred", "occluded", "stale", "lens_unverified", "depth_insufficient", "feature_not_visible")
COLOUR_NAMES = ("black", "white", "grey", "red", "orange", "yellow", "green", "blue", "purple", "brown", "other")

DISPOSITIONS = ("accept", "rework", "reject", "remeasure")
REFUSALS = ("cell_not_ready", "insufficient_evidence", "beyond_capability", "spec_unknown", "reasoner_invalid")
REMEASURE_ROUTES = ("reobserve", "refer")
CONFIGURATIONS = ("A", "B", "C", "D", "E", "F")
#: No AI model and no prompt: the decision must carry null for both. The others must carry both.
DETERMINISTIC_CONFIGURATIONS = ("A", "B", "E")

#: Names from planner/schema.py `Destination` (a test keeps them in step); "none" is "no destination".
PLACE_DESTINATIONS = ("left_tray", "right_tray", "rework_tray")
DESTINATIONS = PLACE_DESTINATIONS + ("point_only", "none")
#: The planner mapping: the destinations each disposition may name. A refusal allows what "refusal" lists.
ALLOWED_DESTINATIONS = {
    "accept": ("left_tray",),
    "reject": ("right_tray",),
    "rework": ("rework_tray",),
    "remeasure": ("point_only", "none"),
    "refusal": ("point_only", "none"),
}

#: Willis's draft 3, Appendix A: filled by the adjudicator later, never at decision time.
APPENDIX_A_FIELDS = ("object_id_and_lot", "expected_family_spec", "measured_dims_hole",
                     "camera_angle_confidence_occlusion", "predicted_class_and_disposition",
                     "twin_pose_and_path_feasibility", "action_approved_refused",
                     "actual_location_final_disposition", "human_adjudication_notes")

#: |norm - 1| allowed for a quaternion.
QUAT_TOLERANCE = 1e-6

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CODE_SHA = re.compile(r"^[0-9a-f]{7,40}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


class DispositionContractError(ValueError):
    """A document or value that is not a valid disposition_decision.v1."""


# ---------------------------------------------------------------- validation helpers

def _fail(where: str, why: str) -> None:
    raise DispositionContractError("%s: %s" % (where, why))


def _keys(where: str, data: Any, required: Tuple[str, ...]) -> Dict[str, Any]:
    """Every field is required (a nullable one travels as null), and no other field is allowed."""
    if not isinstance(data, dict):
        _fail(where, "expected an object, got %s" % type(data).__name__)
    unknown = sorted(set(data) - set(required))
    if unknown:
        _fail(where, "unknown field(s): %s" % ", ".join(unknown))
    missing = [k for k in required if k not in data]
    if missing:
        _fail(where, "missing field(s): %s" % ", ".join(missing))
    return data


def _str(where: str, value: Any) -> str:
    if not isinstance(value, str):
        _fail(where, "expected a string, got %r" % (value,))
    if not value:
        _fail(where, "must not be empty")
    return value


def _opt_str(where: str, value: Any) -> Optional[str]:
    return None if value is None else _str(where, value)


def _choice(where: str, value: Any, choices: Tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in choices:
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


def _float(where: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(where, "expected a number, got %r" % (value,))
    try:
        value = float(value)
    except OverflowError:
        _fail(where, "must be finite, got an integer too large for a float")
    if math.isnan(value) or math.isinf(value):
        _fail(where, "must be finite, got %r" % (value,))
    return value


def _unit_interval(where: str, value: Any) -> float:
    value = _float(where, value)
    if not 0.0 <= value <= 1.0:
        _fail(where, "%r is outside [0, 1]" % (value,))
    return value


def _vec(where: str, value: Any, size: int) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        _fail(where, "expected %d numbers, got %r" % (size, value))
    return tuple(_float("%s[%d]" % (where, i), v) for i, v in enumerate(value))


def _items(where: str, value: Any) -> Tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        _fail(where, "expected a list, got %r" % (value,))
    return tuple(value)


def _strings(where: str, value: Any) -> Tuple[str, ...]:
    out = tuple(_str("%s[%d]" % (where, i), v) for i, v in enumerate(_items(where, value)))
    if len(set(out)) != len(out):
        _fail(where, "holds a repeated value: %r" % (out,))
    return out


def _sha256(where: str, value: Any) -> str:
    if not isinstance(value, str) or not _SHA256.match(value):
        _fail(where, "expected 64 lowercase hex digits, got %r" % (value,))
    return value


def _utc(where: str, value: Any) -> str:
    if not isinstance(value, str) or not _UTC.match(value):
        _fail(where, "expected a UTC time like 2026-11-27T09:45:47Z, got %r" % (value,))
    return value


def _is(where: str, value: Any, kind: type) -> None:
    if not isinstance(value, kind):
        _fail(where, "expected a %s, got %s" % (kind.__name__, type(value).__name__))


def _set(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


def _opt(value: Any, convert) -> Any:
    return None if value is None else convert(value)


def _out(value: Any) -> Any:
    """A field value as JSON data."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [_out(v) for v in value]
    if isinstance(value, dict):
        return {k: _out(v) for k, v in value.items()}
    return value


class _Record:
    """Shared wire form for the frozen records below: `_FIELDS` is the whole field list."""
    _FIELDS = ()  # type: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {k: _out(getattr(self, k)) for k in self._FIELDS}


# ---------------------------------------------------------------- request: case and health

@dataclass(frozen=True)
class Case(_Record):
    case_id: str
    split: str                        # "development" | "held_out"
    expected_design_id: Optional[str]  # what the part is meant to be, e.g. "PART221"; null if not stated
    instance_id: str                  # e.g. "PART221-A"
    lot: str

    _FIELDS = ("case_id", "split", "expected_design_id", "instance_id", "lot")

    def __post_init__(self):
        w = "Case"
        _str(w + ".case_id", self.case_id)
        _choice(w + ".split", self.split, SPLITS)
        _opt_str(w + ".expected_design_id", self.expected_design_id)
        _str(w + ".instance_id", self.instance_id)
        _str(w + ".lot", self.lot)

    @classmethod
    def from_dict(cls, data: Any) -> "Case":
        _keys("Case", data, cls._FIELDS)
        return cls(**data)


@dataclass(frozen=True)
class PrecaptureSummary(_Record):
    """The summary of one health/precapture.py report (`HealthReport.to_dict()`)."""
    format_version: int
    passed: bool
    checked_utc: str
    failed: Tuple[str, ...]  # names of the checks that failed or did not run

    _FIELDS = ("format_version", "passed", "checked_utc", "failed")

    def __post_init__(self):
        w = "PrecaptureSummary"
        _int(w + ".format_version", self.format_version, 1)
        _bool(w + ".passed", self.passed)
        _utc(w + ".checked_utc", self.checked_utc)
        _set(self, "failed", _strings(w + ".failed", self.failed))
        if self.passed and self.failed:
            _fail(w, "passed is true but checks failed: %s" % ", ".join(self.failed))
        if not self.passed and not self.failed:
            _fail(w, "passed is false but no failed check is named")

    @classmethod
    def from_dict(cls, data: Any) -> "PrecaptureSummary":
        _keys("PrecaptureSummary", data, cls._FIELDS)
        return cls(**data)


@dataclass(frozen=True)
class Health(_Record):
    """`cell_ready` may be false although the report passed (other cell conditions); never true without a pass."""
    cell_ready: bool
    precapture: Optional[PrecaptureSummary]  # null: no report, so the cell is not ready

    _FIELDS = ("cell_ready", "precapture")

    def __post_init__(self):
        w = "Health"
        _bool(w + ".cell_ready", self.cell_ready)
        if self.precapture is not None:
            _is(w + ".precapture", self.precapture, PrecaptureSummary)
        if self.cell_ready and (self.precapture is None or not self.precapture.passed):
            _fail(w, "cell_ready needs a passed pre-capture report")

    @classmethod
    def from_dict(cls, data: Any) -> "Health":
        _keys("Health", data, cls._FIELDS)
        return cls(cell_ready=data["cell_ready"],
                   precapture=_opt(data["precapture"], PrecaptureSummary.from_dict))


def health_from_report(report: Any, cell_ready: Optional[bool] = None) -> Health:
    """The request's health block from a pre-capture report dict.

    `cell_ready` defaults to the report's `passed`. A caller may lower it; raising it above a failed report is
    refused."""
    if not isinstance(report, dict):
        _fail("health_from_report", "expected a report dict, got %s" % type(report).__name__)
    missing = [k for k in PrecaptureSummary._FIELDS if k not in report]
    if missing:
        _fail("health_from_report", "the report lacks %s" % ", ".join(missing))
    summary = PrecaptureSummary.from_dict({k: report[k] for k in PrecaptureSummary._FIELDS})
    return Health(cell_ready=summary.passed if cell_ready is None else cell_ready, precapture=summary)


# ---------------------------------------------------------------- request: observation

@dataclass(frozen=True)
class Measurement(_Record):
    """A measured length with its bound: the true value is taken to lie in value +- bound."""
    value: float
    bound: float

    _FIELDS = ("value", "bound")

    def __post_init__(self):
        value = _float("Measurement.value", self.value)
        if value <= 0.0:
            _fail("Measurement.value", "a length must be positive, got %r" % (value,))
        bound = _float("Measurement.bound", self.bound)
        if bound < 0.0:
            _fail("Measurement.bound", "%r is negative" % (bound,))
        _set(self, "value", value)
        _set(self, "bound", bound)

    @classmethod
    def from_dict(cls, data: Any) -> "Measurement":
        _keys("Measurement", data, cls._FIELDS)
        return cls(**data)


@dataclass(frozen=True)
class Hole(_Record):
    state: str                          # "present" | "absent" | "unknown": a top-face opening
    diameter_mm: Optional[Measurement]  # only for a present hole

    _FIELDS = ("state", "diameter_mm")

    def __post_init__(self):
        _choice("Hole.state", self.state, HOLE_STATES)
        if self.diameter_mm is not None:
            _is("Hole.diameter_mm", self.diameter_mm, Measurement)
            if self.state != "present":
                _fail("Hole", "diameter_mm is only for a present hole, state is %s" % self.state)

    @classmethod
    def from_dict(cls, data: Any) -> "Hole":
        _keys("Hole", data, cls._FIELDS)
        return cls(state=data["state"], diameter_mm=_opt(data["diameter_mm"], Measurement.from_dict))


@dataclass(frozen=True)
class Sufficiency(_Record):
    """Reasons the evidence may not be good enough. True means the problem is there."""
    blurred: bool
    occluded: bool
    stale: bool
    lens_unverified: bool
    depth_insufficient: bool
    feature_not_visible: bool

    _FIELDS = SUFFICIENCY_FLAGS

    def __post_init__(self):
        for name in SUFFICIENCY_FLAGS:
            _bool("Sufficiency." + name, getattr(self, name))

    def raised(self) -> Tuple[str, ...]:
        """The flags that are true, in SUFFICIENCY_FLAGS order."""
        return tuple(name for name in SUFFICIENCY_FLAGS if getattr(self, name))

    @classmethod
    def from_dict(cls, data: Any) -> "Sufficiency":
        _keys("Sufficiency", data, cls._FIELDS)
        return cls(**data)


@dataclass(frozen=True)
class Bundle(_Record):
    """The capture bundle the numbers came from: a path and a hash, never the images."""
    path: str
    sha256: str

    _FIELDS = ("path", "sha256")

    def __post_init__(self):
        _str("Bundle.path", self.path)
        _sha256("Bundle.sha256", self.sha256)

    @classmethod
    def from_dict(cls, data: Any) -> "Bundle":
        _keys("Bundle", data, cls._FIELDS)
        return cls(**data)


@dataclass(frozen=True)
class Colour(_Record):
    """Observed colour (owner, 2026-11-27): recorded and available. spec_v1's dispositions do not use it."""
    name: str                              # one of COLOUR_NAMES
    lab_median: Tuple[float, float, float]  # CIELAB median over the object mask, excluding highlights
    confidence: float
    reliable: bool                         # false when the lighting or white-balance reference check fails

    _FIELDS = ("name", "lab_median", "confidence", "reliable")

    def __post_init__(self):
        w = "Colour"
        _choice(w + ".name", self.name, COLOUR_NAMES)
        lab = _vec(w + ".lab_median", self.lab_median, 3)
        if not 0.0 <= lab[0] <= 100.0:
            _fail(w + ".lab_median", "L* %r is outside [0, 100]" % (lab[0],))
        _set(self, "lab_median", lab)
        _set(self, "confidence", _unit_interval(w + ".confidence", self.confidence))
        _bool(w + ".reliable", self.reliable)

    @classmethod
    def from_dict(cls, data: Any) -> "Colour":
        _keys("Colour", data, cls._FIELDS)
        return cls(**data)


@dataclass(frozen=True)
class Observation(_Record):
    kind: str                          # "block" | "cylinder" | "unknown"
    height_mm: Optional[Measurement]   # null: not measured
    width_mm: Optional[Measurement]
    hole: Hole
    orientation: str                   # "upright" | "lying" | "rotated" | "unknown"
    sufficiency: Sufficiency
    bundle: Optional[Bundle]           # required for perceived input
    colour: Optional[Colour]           # null: not measured

    _FIELDS = ("kind", "height_mm", "width_mm", "hole", "orientation", "sufficiency", "bundle", "colour")

    def __post_init__(self):
        w = "Observation"
        _choice(w + ".kind", self.kind, KINDS)
        for name in ("height_mm", "width_mm"):
            if getattr(self, name) is not None:
                _is("%s.%s" % (w, name), getattr(self, name), Measurement)
        _is(w + ".hole", self.hole, Hole)
        _choice(w + ".orientation", self.orientation, ORIENTATIONS)
        _is(w + ".sufficiency", self.sufficiency, Sufficiency)
        if self.bundle is not None:
            _is(w + ".bundle", self.bundle, Bundle)
        if self.colour is not None:
            _is(w + ".colour", self.colour, Colour)

    @classmethod
    def from_dict(cls, data: Any) -> "Observation":
        _keys("Observation", data, cls._FIELDS)
        return cls(kind=data["kind"],
                   height_mm=_opt(data["height_mm"], Measurement.from_dict),
                   width_mm=_opt(data["width_mm"], Measurement.from_dict),
                   hole=Hole.from_dict(data["hole"]), orientation=data["orientation"],
                   sufficiency=Sufficiency.from_dict(data["sufficiency"]),
                   bundle=_opt(data["bundle"], Bundle.from_dict),
                   colour=_opt(data["colour"], Colour.from_dict))


# ---------------------------------------------------------------- request: twin

@dataclass(frozen=True)
class PartPose(_Record):
    """Torso frame, metres, quaternion [w, x, y, z]: the perceived_scene.v1 convention."""
    position_m: Tuple[float, float, float]
    quat_wxyz: Tuple[float, float, float, float]

    _FIELDS = ("position_m", "quat_wxyz")

    def __post_init__(self):
        _set(self, "position_m", _vec("PartPose.position_m", self.position_m, 3))
        quat = _vec("PartPose.quat_wxyz", self.quat_wxyz, 4)
        norm = math.sqrt(sum(c * c for c in quat))
        if abs(norm - 1.0) > QUAT_TOLERANCE:
            _fail("PartPose.quat_wxyz", "quaternion [w, x, y, z] must have unit norm within %g, norm is %r"
                  % (QUAT_TOLERANCE, norm))
        _set(self, "quat_wxyz", quat)

    @classmethod
    def from_dict(cls, data: Any) -> "PartPose":
        _keys("PartPose", data, cls._FIELDS)
        return cls(**data)


@dataclass(frozen=True)
class Twin(_Record):
    """Filled by the sim consumer. Null on the robot until the arm gate."""
    part_pose: Optional[PartPose]
    path_feasibility: Optional[Tuple[Tuple[str, bool], ...]]  # (destination, feasible); an object on the wire

    _FIELDS = ("part_pose", "path_feasibility")

    def __post_init__(self):
        if self.part_pose is not None:
            _is("Twin.part_pose", self.part_pose, PartPose)
        if self.path_feasibility is not None:
            # stored as sorted pairs so the record stays immutable
            try:
                pairs = dict(self.path_feasibility)
            except (TypeError, ValueError):
                _fail("Twin.path_feasibility", "expected destination -> true or false")
            for destination, feasible in pairs.items():
                _choice("Twin.path_feasibility key", destination, PLACE_DESTINATIONS + ("point_only",))
                _bool("Twin.path_feasibility.%s" % destination, feasible)
            _set(self, "path_feasibility", tuple(sorted(pairs.items())))

    def to_dict(self) -> Dict[str, Any]:
        return {"part_pose": _out(self.part_pose),
                "path_feasibility": None if self.path_feasibility is None else dict(self.path_feasibility)}

    @classmethod
    def from_dict(cls, data: Any) -> "Twin":
        _keys("Twin", data, cls._FIELDS)
        feasibility = data["path_feasibility"]
        if feasibility is not None and not isinstance(feasibility, dict):
            _fail("Twin.path_feasibility", "expected an object or null")
        return cls(part_pose=_opt(data["part_pose"], PartPose.from_dict),
                   path_feasibility=None if feasibility is None else tuple(feasibility.items()))


@dataclass(frozen=True)
class Request(_Record):
    request_id: str
    created_utc: str
    input_source: str             # "perceived" | "sim_truth"
    source: str                   # "robot" | "sim"
    degradation: Tuple[str, ...]  # names of injected degradations; sim only
    reobservation_count: int      # re-observations already made of this case before this request
    case: Case
    health: Health
    observation: Observation
    twin: Optional[Twin]
    schema_version: str = SCHEMA_VERSION

    _FIELDS = ("schema_version", "request_id", "created_utc", "input_source", "source", "degradation",
               "reobservation_count", "case", "health", "observation", "twin")

    def __post_init__(self):
        w = "Request"
        if self.schema_version != SCHEMA_VERSION:
            _fail(w + ".schema_version", "%r is not %s" % (self.schema_version, SCHEMA_VERSION))
        _str(w + ".request_id", self.request_id)
        _utc(w + ".created_utc", self.created_utc)
        _choice(w + ".input_source", self.input_source, INPUT_SOURCES)
        _choice(w + ".source", self.source, SOURCES)
        _set(self, "degradation", _strings(w + ".degradation", self.degradation))
        _int(w + ".reobservation_count", self.reobservation_count)
        _is(w + ".case", self.case, Case)
        _is(w + ".health", self.health, Health)
        _is(w + ".observation", self.observation, Observation)
        if self.twin is not None:
            _is(w + ".twin", self.twin, Twin)
        if self.source == "robot":
            if self.input_source == "sim_truth":
                _fail(w, "sim_truth input cannot come from the robot")
            if self.degradation:
                _fail(w, "degradation flags are sim only")
            if self.twin is not None:
                _fail(w, "twin is null on the robot until the arm gate")
        if self.input_source == "perceived" and self.observation.bundle is None:
            _fail(w, "perceived input needs a capture bundle path and SHA-256")

    @classmethod
    def from_dict(cls, data: Any) -> "Request":
        _keys("Request", data, cls._FIELDS)
        return cls(
            schema_version=data["schema_version"], request_id=data["request_id"],
            created_utc=data["created_utc"], input_source=data["input_source"], source=data["source"],
            degradation=data["degradation"], reobservation_count=data["reobservation_count"],
            case=Case.from_dict(data["case"]), health=Health.from_dict(data["health"]),
            observation=Observation.from_dict(data["observation"]), twin=_opt(data["twin"], Twin.from_dict))


# ---------------------------------------------------------------- decision

@dataclass(frozen=True)
class Reason(_Record):
    rule_id: str
    text: str

    _FIELDS = ("rule_id", "text")

    def __post_init__(self):
        _str("Reason.rule_id", self.rule_id)
        _str("Reason.text", self.text)

    @classmethod
    def from_dict(cls, data: Any) -> "Reason":
        _keys("Reason", data, cls._FIELDS)
        return cls(**data)


@dataclass(frozen=True)
class PolicyHashes(_Record):
    spec_sha256: str
    rules_sha256: str
    prompt_sha256: Optional[str]  # null for A, B and E
    split_sha256: str

    _FIELDS = ("spec_sha256", "rules_sha256", "prompt_sha256", "split_sha256")

    def __post_init__(self):
        for name in ("spec_sha256", "rules_sha256", "split_sha256"):
            _sha256("PolicyHashes." + name, getattr(self, name))
        if self.prompt_sha256 is not None:
            _sha256("PolicyHashes.prompt_sha256", self.prompt_sha256)

    @classmethod
    def from_dict(cls, data: Any) -> "PolicyHashes":
        _keys("PolicyHashes", data, cls._FIELDS)
        return cls(**data)


@dataclass(frozen=True)
class Adjudication(_Record):
    """Willis's Appendix A record (APPENDIX_A_FIELDS) plus the operator rating and who adjudicated.

    Null at decision time. Free text; a field not yet filled is null."""
    object_id_and_lot: Optional[str]
    expected_family_spec: Optional[str]
    measured_dims_hole: Optional[str]
    camera_angle_confidence_occlusion: Optional[str]
    predicted_class_and_disposition: Optional[str]
    twin_pose_and_path_feasibility: Optional[str]
    action_approved_refused: Optional[str]
    actual_location_final_disposition: Optional[str]
    human_adjudication_notes: Optional[str]
    operator_rating: Optional[int]  # 1 to 5
    adjudicator: str

    _FIELDS = APPENDIX_A_FIELDS + ("operator_rating", "adjudicator")

    def __post_init__(self):
        w = "Adjudication"
        for name in APPENDIX_A_FIELDS:
            _opt_str("%s.%s" % (w, name), getattr(self, name))
        if self.operator_rating is not None:
            if _int(w + ".operator_rating", self.operator_rating, 1) > 5:
                _fail(w + ".operator_rating", "%d is above the maximum 5" % self.operator_rating)
        _str(w + ".adjudicator", self.adjudicator)

    @classmethod
    def from_dict(cls, data: Any) -> "Adjudication":
        _keys("Adjudication", data, cls._FIELDS)
        return cls(**data)


@dataclass(frozen=True)
class Decision(_Record):
    request_id: str
    configuration: str              # "A" ... "F"
    disposition: Optional[str]      # exactly one of disposition and refusal is set
    refusal: Optional[str]
    remeasure_route: Optional[str]  # "reobserve" | "refer", with remeasure only
    destination: str
    reasons: Tuple[Reason, ...]
    confidence: Optional[float]     # null: the configuration produces no confidence (A, B, E)
    policy: PolicyHashes
    model_id: Optional[str]         # null for A, B and E
    code_sha: str
    adjudication: Optional[Adjudication]
    schema_version: str = SCHEMA_VERSION

    _FIELDS = ("schema_version", "request_id", "configuration", "disposition", "refusal", "remeasure_route",
               "destination", "reasons", "confidence", "policy", "model_id", "code_sha", "adjudication")

    def __post_init__(self):
        w = "Decision"
        if self.schema_version != SCHEMA_VERSION:
            _fail(w + ".schema_version", "%r is not %s" % (self.schema_version, SCHEMA_VERSION))
        _str(w + ".request_id", self.request_id)
        _choice(w + ".configuration", self.configuration, CONFIGURATIONS)
        if (self.disposition is None) == (self.refusal is None):
            _fail(w, "exactly one of disposition and refusal must be set")
        if self.disposition is not None:
            _choice(w + ".disposition", self.disposition, DISPOSITIONS)
        else:
            _choice(w + ".refusal", self.refusal, REFUSALS)
        if self.disposition == "remeasure":
            _choice(w + ".remeasure_route", self.remeasure_route, REMEASURE_ROUTES)
        elif self.remeasure_route is not None:
            _fail(w, "remeasure_route is only for a remeasure disposition")
        _choice(w + ".destination", self.destination, DESTINATIONS)
        outcome = self.disposition if self.disposition is not None else "refusal"
        if self.destination not in ALLOWED_DESTINATIONS[outcome]:
            _fail(w + ".destination", "%s cannot go to %s (allowed: %s)"
                  % (outcome, self.destination, ", ".join(ALLOWED_DESTINATIONS[outcome])))
        reasons = _items(w + ".reasons", self.reasons)
        if not reasons:
            _fail(w + ".reasons", "a decision must give at least one reason")
        for reason in reasons:
            _is(w + ".reasons", reason, Reason)
        _set(self, "reasons", reasons)
        if self.confidence is not None:
            _set(self, "confidence", _unit_interval(w + ".confidence", self.confidence))
        _is(w + ".policy", self.policy, PolicyHashes)
        _opt_str(w + ".model_id", self.model_id)
        if self.configuration in DETERMINISTIC_CONFIGURATIONS:
            if self.model_id is not None or self.policy.prompt_sha256 is not None:
                _fail(w, "configuration %s uses no model or prompt" % self.configuration)
        elif self.model_id is None or self.policy.prompt_sha256 is None:
            _fail(w, "configuration %s needs a model_id and a prompt_sha256" % self.configuration)
        if not isinstance(self.code_sha, str) or not _CODE_SHA.match(self.code_sha):
            _fail(w + ".code_sha", "expected 7 to 40 lowercase hex digits, got %r" % (self.code_sha,))
        if self.adjudication is not None:
            _is(w + ".adjudication", self.adjudication, Adjudication)

    @classmethod
    def from_dict(cls, data: Any) -> "Decision":
        _keys("Decision", data, cls._FIELDS)
        return cls(
            schema_version=data["schema_version"], request_id=data["request_id"],
            configuration=data["configuration"], disposition=data["disposition"], refusal=data["refusal"],
            remeasure_route=data["remeasure_route"], destination=data["destination"],
            reasons=tuple(Reason.from_dict(r) for r in _items("Decision.reasons", data["reasons"])),
            confidence=data["confidence"], policy=PolicyHashes.from_dict(data["policy"]),
            model_id=data["model_id"], code_sha=data["code_sha"],
            adjudication=_opt(data["adjudication"], Adjudication.from_dict))


# ---------------------------------------------------------------- wire form

def _reject_constant(name: str) -> None:
    raise DispositionContractError("%s is not valid JSON: NaN and infinity are refused" % name)


def _no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise DispositionContractError("duplicate key %r" % key)
        out[key] = value
    return out


def dumps(document: Any) -> str:
    """The canonical wire form of a Request or a Decision."""
    return json.dumps(document.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"


def _parse(text: str) -> Any:
    try:
        return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_no_duplicates)
    except json.JSONDecodeError as exc:
        raise DispositionContractError("not valid JSON: %s" % exc) from exc


def loads_request(text: str) -> Request:
    """Parse and validate a request. Raises DispositionContractError for anything else."""
    return Request.from_dict(_parse(text))


def loads_decision(text: str) -> Decision:
    """Parse and validate a decision. Raises DispositionContractError for anything else."""
    return Decision.from_dict(_parse(text))
