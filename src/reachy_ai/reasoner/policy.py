"""The versioned policy files (config/disposition/) and the deterministic configurations A, B and E (#81).

Two halves:
- `load_policy()` reads spec, rules and split, validates them strictly (unknown, missing or repeated keys are
  refused) and records each file's SHA-256 over the exact bytes it parsed;
- `decide()` is a pure function: (request, policy, configuration, code_sha) -> Decision. No file, clock, SDK or
  network is read in it, and it commands no motion.

A decision is a proposal. Its destination is a name; the safety gates and the operator decide whether anything
may move. A and B skip Stage 0 (cell health) by design: they are ablation baselines.

Not covered by the #81 rules, so `decide()` does not look at them: observation.orientation, observation.width_mm,
observation.colour, case.expected_design_id and twin.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from reachy_ai.reasoner import contract as c

POLICY_DIR = Path(__file__).resolve().parents[3] / "config" / "disposition"
SPEC_FILE, RULES_FILE, SPLIT_FILE = "spec_v1.yaml", "rules_v1.yaml", "split_v0.yaml"

SHAPES = ("block", "cylinder")
HEIGHT_CLASSES = ("short", "tall")
DECIDING_FEATURES = ("shape", "height", "hole")
RULE_KINDS = ("stage0_cell_ready", "sufficiency", "height_bounds", "classify", "unmatched")


class PolicyError(ValueError):
    """A policy file that cannot be used, or a configuration this module cannot run."""


# ---------------------------------------------------------------- strict YAML

class _StrictLoader(yaml.SafeLoader):
    pass


def _mapping(loader: _StrictLoader, node, deep: bool = False) -> dict:
    keys = [loader.construct_object(key, deep=True) for key, _ in node.value]
    repeated = sorted({str(k) for k in keys if keys.count(k) > 1})
    if repeated:
        raise PolicyError(f"duplicate key(s): {', '.join(repeated)}")
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _read(path: Path) -> tuple[Any, str]:
    try:
        raw = path.read_bytes()
        data = yaml.load(raw.decode("utf-8"), Loader=_StrictLoader)  # noqa: S506 - a SafeLoader subclass
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PolicyError(f"{path.name}: {exc}") from exc
    return data, hashlib.sha256(raw).hexdigest()


def _keys(where: str, data: Any, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> dict:
    if not isinstance(data, dict):
        raise PolicyError(f"{where}: expected a mapping")
    unknown = sorted(str(k) for k in set(data) - set(required) - set(optional))
    if unknown:
        raise PolicyError(f"{where}: unknown key(s): {', '.join(unknown)}")
    missing = [k for k in required if k not in data]
    if missing:
        raise PolicyError(f"{where}: missing key(s): {', '.join(missing)}")
    return data


def _require(condition: bool, where: str, why: str) -> None:
    if not condition:
        raise PolicyError(f"{where}: {why}")


def _positive(where: str, value: Any) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value < float("inf"),
             where, f"expected a positive number, got {value!r}")
    return float(value)


def _entries(where: str, value: Any) -> list:
    _require(isinstance(value, list) and bool(value), where, "expected a non-empty list")
    return value


# ---------------------------------------------------------------- spec

@dataclass(frozen=True)
class Design:
    design: str
    shape: str
    height_class: str
    height_mm: float
    footprint_mm: float
    hole_diameter_mm: float | None
    disposition: str


@dataclass(frozen=True)
class Spec:
    version: str
    provisional: bool
    designs: tuple[Design, ...]
    height_class_boundary_mm: float
    hole_min_diameter_mm: float
    approved_rework_operation: bool
    quality_authority: str
    authority: dict
    sha256: str

    def design(self, design_id: str) -> Design:
        for row in self.designs:
            if row.design == design_id:
                return row
        raise KeyError(design_id)

    def height_class(self, height_mm: float) -> str:
        return "short" if height_mm < self.height_class_boundary_mm else "tall"

    def candidates(self, shape: str | None = None, height_class: str | None = None,
                   hole: str | None = None) -> tuple[Design, ...]:
        """The rows consistent with what is known. None means unknown; `hole` is "present" or "absent"."""
        return tuple(
            row for row in self.designs
            if (shape is None or row.shape == shape)
            and (height_class is None or row.height_class == height_class)
            and (hole is None or (row.hole_diameter_mm is not None) == (hole == "present")))


def _load_spec(path: Path) -> Spec:
    data, sha = _read(path)
    w = path.name
    _keys(w, data, ("version", "provisional", "colour_requirement", "instances_per_design", "designs",
                    "tolerances", "authority"))
    _require(isinstance(data["provisional"], bool), w, "provisional must be true or false")
    # Colour dispositions need a rule that reads colour; none exists, so a spec asking for one is refused.
    _require(data["colour_requirement"] is None, w, "this code applies no colour requirement")
    designs = []
    for row in _entries(w + ".designs", data["designs"]):
        _keys(w + ".designs", row, ("design", "shape", "height_class", "height_mm", "footprint_mm",
                                    "hole_diameter_mm", "disposition"))
        where = f"{w}.designs.{row['design']}"
        _require(isinstance(row["design"], str) and bool(row["design"]), where, "design must be a name")
        _require(row["shape"] in SHAPES, where, f"unknown shape {row['shape']!r}")
        _require(row["height_class"] in HEIGHT_CLASSES, where, f"unknown height_class {row['height_class']!r}")
        _require(row["disposition"] in c.DISPOSITIONS, where, f"unknown disposition {row['disposition']!r}")
        _positive(where + ".height_mm", row["height_mm"])
        _positive(where + ".footprint_mm", row["footprint_mm"])
        if row["hole_diameter_mm"] is not None:
            _positive(where + ".hole_diameter_mm", row["hole_diameter_mm"])
        designs.append(Design(**row))
    _require(len({d.design for d in designs}) == len(designs), w, "a design is listed twice")
    families = [(d.shape, d.height_class, d.hole_diameter_mm is not None) for d in designs]
    _require(len(set(families)) == len(families), w, "two designs have the same shape, height class and hole")

    tolerances = _keys(w + ".tolerances", data["tolerances"], ("height_class_boundary_mm", "hole_min_diameter_mm"))
    values = {}
    for name, entry in tolerances.items():
        _keys(f"{w}.tolerances.{name}", entry, ("value", "provisional"))
        _require(isinstance(entry["provisional"], bool), f"{w}.tolerances.{name}", "provisional must be a bool")
        values[name] = _positive(f"{w}.tolerances.{name}.value", entry["value"])

    authority = _keys(w + ".authority", data["authority"],
                      ("provisional", "quality_authority", "roles", "approved_rework_operation", "dispositions",
                       "refusals", "scoring_costs"))
    _require(isinstance(authority["approved_rework_operation"], bool), w + ".authority",
             "approved_rework_operation must be true or false")
    spec = Spec(version=data["version"], provisional=data["provisional"], designs=tuple(designs),
                height_class_boundary_mm=values["height_class_boundary_mm"],
                hole_min_diameter_mm=values["hole_min_diameter_mm"],
                approved_rework_operation=authority["approved_rework_operation"],
                quality_authority=authority["quality_authority"], authority=authority, sha256=sha)
    for row in spec.designs:
        _require(spec.height_class(row.height_mm) == row.height_class, f"{w}.designs.{row.design}",
                 f"height {row.height_mm} mm is not {row.height_class} at the class boundary")
    return spec


# ---------------------------------------------------------------- rules

@dataclass(frozen=True)
class Rule:
    id: str
    kind: str
    text: str


@dataclass(frozen=True)
class Rules:
    version: str
    rules: dict[str, Rule]
    max_reobservations: int
    hides: dict[str, tuple[str, ...]]
    configurations: dict[str, tuple[str, ...]]
    sha256: str

    def sequence(self, configuration: str) -> tuple[Rule, ...]:
        if configuration not in self.configurations:
            raise PolicyError(f"configuration {configuration!r} is not a deterministic configuration in "
                              f"{self.version} ({', '.join(sorted(self.configurations))})")
        return tuple(self.rules[rule_id] for rule_id in self.configurations[configuration])


def _load_rules(path: Path) -> Rules:
    data, sha = _read(path)
    w = path.name
    _keys(w, data, ("version", "rules", "max_reobservations", "sufficiency", "configurations"))
    rules: dict[str, Rule] = {}
    for row in _entries(w + ".rules", data["rules"]):
        _keys(w + ".rules", row, ("id", "kind", "text"))
        _require(row["kind"] in RULE_KINDS, w + ".rules", f"unknown rule kind {row['kind']!r}")
        _require(row["id"] not in rules, w + ".rules", f"rule {row['id']} is listed twice")
        rules[row["id"]] = Rule(**row)
    limit = data["max_reobservations"]
    _require(isinstance(limit, int) and not isinstance(limit, bool) and limit >= 0, w,
             "max_reobservations must be a whole number, 0 or more")
    sufficiency = _keys(w + ".sufficiency", data["sufficiency"], ("provisional", "hides"))
    hides = _keys(w + ".sufficiency.hides", sufficiency["hides"], c.SUFFICIENCY_FLAGS)
    for flag, features in hides.items():
        _require(isinstance(features, list) and bool(features) and set(features) <= set(DECIDING_FEATURES),
                 f"{w}.sufficiency.hides.{flag}", f"expected some of {', '.join(DECIDING_FEATURES)}")
    configurations = _keys(w + ".configurations", data["configurations"], c.DETERMINISTIC_CONFIGURATIONS)
    for name, ids in configurations.items():
        where = f"{w}.configurations.{name}"
        _entries(where, ids)
        _require(set(ids) <= set(rules), where, "names a rule that is not listed")
        _require(len(set(ids)) == len(ids), where, "names a rule twice")
        _require(rules[ids[-1]].kind == "unmatched", where, "must end with the unmatched rule")
    return Rules(version=data["version"], rules=rules, max_reobservations=limit,
                 hides={k: tuple(v) for k, v in hides.items()},
                 configurations={k: tuple(v) for k, v in configurations.items()}, sha256=sha)


# ---------------------------------------------------------------- split

_SELECTORS = {
    "designs": None,  # checked against the spec
    "parts": ("lab_pool", "terrance_own"),
    "presentation": ("upright", "lying"),
    "view": ("clear", "partial_occlusion", "defocused"),
    "twin": ("consistent", "inconsistent"),
    "cell_health": ("injected_motor_power_outage", "stale_clock"),
}


@dataclass(frozen=True)
class Split:
    version: str
    frozen: bool
    status: str
    development: tuple[dict, ...]
    held_out: tuple[dict, ...]
    sha256: str


def load_split(path: Path, design_ids: tuple[str, ...]) -> Split:
    data, sha = _read(path)
    w = path.name
    _keys(w, data, ("version", "frozen", "status", "development", "held_out"))
    _require(isinstance(data["frozen"], bool), w, "frozen must be true or false")
    _require(data["status"] in ("PROPOSED", "CONFIRMED"), w, f"unknown status {data['status']!r}")
    _require(not (data["frozen"] and data["status"] == "PROPOSED"), w, "a proposed split cannot be frozen")
    ids: list[str] = []
    for split in c.SPLITS:
        for entry in _entries(f"{w}.{split}", data[split]):
            optional = tuple(_SELECTORS) + (("development_only",) if split == "development" else ())
            _keys(f"{w}.{split}", entry, ("id", "description"), optional)
            where = f"{w}.{split}.{entry['id']}"
            for key, value in entry.items():
                if key == "designs":
                    _require(isinstance(value, list) and bool(value) and set(value) <= set(design_ids), where,
                             f"designs must be spec designs, got {value!r}")
                elif key in _SELECTORS:
                    _require(value in _SELECTORS[key], where, f"unknown {key} {value!r}")
                elif key == "development_only":
                    _require(isinstance(value, bool), where, "development_only must be true or false")
            _require(any(key in _SELECTORS for key in entry), where, "selects nothing")
            ids.append(entry["id"])
    _require(len(set(ids)) == len(ids), w, "an entry id is used twice")
    return Split(version=data["version"], frozen=data["frozen"], status=data["status"],
                 development=tuple(data["development"]), held_out=tuple(data["held_out"]), sha256=sha)


# ---------------------------------------------------------------- policy

@dataclass(frozen=True)
class Policy:
    spec: Spec
    rules: Rules
    split: Split

    def hashes(self) -> c.PolicyHashes:
        """The policy block of a deterministic decision (no prompt)."""
        return c.PolicyHashes(spec_sha256=self.spec.sha256, rules_sha256=self.rules.sha256, prompt_sha256=None,
                              split_sha256=self.split.sha256)


def load_policy(directory: Path | str = POLICY_DIR) -> Policy:
    directory = Path(directory)
    spec = _load_spec(directory / SPEC_FILE)
    return Policy(spec=spec, rules=_load_rules(directory / RULES_FILE),
                  split=load_split(directory / SPLIT_FILE, tuple(d.design for d in spec.designs)))


# ---------------------------------------------------------------- the deterministic policy

@dataclass(frozen=True)
class _Outcome:
    disposition: str | None
    refusal: str | None
    route: str | None
    text: str


def _effective_hole(spec: Spec, hole: c.Hole) -> str:
    """The hole state under the spec's tolerance. "present" needs a measured opening at least the minimum wide;
    a "present" with a smaller or unmeasured opening is unknown, never absent and never present."""
    if hole.state == "present":
        wide_enough = hole.diameter_mm is not None and hole.diameter_mm.value >= spec.hole_min_diameter_mm
        return "present" if wide_enough else "unknown"
    return hole.state


def _family(spec: Spec, observation: c.Observation) -> tuple[str | None, str | None]:
    shape = None if observation.kind == "unknown" else observation.kind
    height = observation.height_mm
    return shape, None if height is None else spec.height_class(height.value)


def _hole_is_deciding(spec: Spec, observation: c.Observation) -> bool:
    """True unless every design the observed family could be has the same disposition."""
    return len({row.disposition for row in spec.candidates(*_family(spec, observation))}) != 1


def _remeasure(policy: Policy, request: c.Request, why: str, can_reobserve: bool) -> _Outcome:
    limit, done = policy.rules.max_reobservations, request.reobservation_count
    if not can_reobserve:
        return _Outcome("remeasure", None, "refer", f"{why} This configuration cannot re-observe: refer.")
    if done < limit:
        return _Outcome("remeasure", None, "reobserve", f"{why} Re-observation {done + 1} of {limit}.")
    return _Outcome("remeasure", None, "refer",
                    f"{why} Already re-observed {done} times (limit {limit}): refer for human measurement.")


def _stage0(policy: Policy, request: c.Request, can_reobserve: bool) -> _Outcome | None:
    if request.health.cell_ready:
        return None
    report = request.health.precapture
    if report is None:
        why = "there is no pre-capture report"
    elif report.failed:
        why = "failed pre-capture checks: " + ", ".join(report.failed)
    else:
        why = "the pre-capture report passed but the cell was marked not ready"
    return _Outcome(None, "cell_not_ready", None, f"cell_ready is false; {why}.")


def _sufficiency(policy: Policy, request: c.Request, can_reobserve: bool) -> _Outcome | None:
    deciding = {"shape", "height"}
    if _hole_is_deciding(policy.spec, request.observation):
        deciding.add("hole")
    hiding = [flag for flag in request.observation.sufficiency.raised()
              if deciding & set(policy.rules.hides[flag])]
    if not hiding:
        return None
    return _remeasure(policy, request, f"A deciding feature is hidden ({', '.join(hiding)}).", can_reobserve)


def _height_bounds(policy: Policy, request: c.Request, can_reobserve: bool) -> _Outcome | None:
    height, boundary = request.observation.height_mm, policy.spec.height_class_boundary_mm
    if height is None or not height.value - height.bound <= boundary <= height.value + height.bound:
        return None
    why = (f"Height {height.value:g} +- {height.bound:g} mm straddles the class boundary {boundary:g} mm.")
    return _remeasure(policy, request, why, can_reobserve)


def _classify(policy: Policy, request: c.Request, can_reobserve: bool) -> _Outcome | None:
    spec, observation = policy.spec, request.observation
    shape, height_class = _family(spec, observation)
    if shape is None:
        return None
    hole = _effective_hole(spec, observation.hole)
    rows = spec.candidates(shape, height_class, None if hole == "unknown" else hole)
    dispositions = {row.disposition for row in rows}
    if len(dispositions) != 1:
        return None
    disposition = dispositions.pop()
    text = (f"{shape}, height class {height_class or 'unknown'}, hole {hole}: spec {spec.version} gives "
            f"{disposition} for {', '.join(row.design for row in rows)}.")
    if disposition == "rework" and not spec.approved_rework_operation:
        text += (" Rework recommended: no approved rework operation exists, so segregate the part and notify "
                 f"quality_authority ({spec.quality_authority}).")
    if disposition == "remeasure":
        return _remeasure(policy, request, text, can_reobserve)
    return _Outcome(disposition, None, None, text)


def _unmatched(policy: Policy, request: c.Request, can_reobserve: bool) -> _Outcome:
    observation = request.observation
    height = observation.height_mm
    return _Outcome(None, "spec_unknown", None,
                    f"No single spec disposition for kind {observation.kind}, height "
                    f"{'unmeasured' if height is None else format(height.value, 'g') + ' mm'}, hole "
                    f"{_effective_hole(policy.spec, observation.hole)}.")


_APPLY = {
    "stage0_cell_ready": _stage0,
    "sufficiency": _sufficiency,
    "height_bounds": _height_bounds,
    "classify": _classify,
    "unmatched": _unmatched,
}

#: The planner mapping. A remeasure or a refusal asks for no motion.
_DESTINATION = {"accept": "left_tray", "reject": "right_tray", "rework": "rework_tray"}


def decide(request: c.Request, policy: Policy, configuration: str, code_sha: str) -> c.Decision:
    """The decision of a deterministic configuration (A, B or E). The first rule that fires wins.

    Re-observation belongs to the configurations that contain the sufficiency rule (B and E). In A, a spec
    disposition of remeasure goes straight to refer."""
    sequence = policy.rules.sequence(configuration)
    can_reobserve = any(rule.kind == "sufficiency" for rule in sequence)
    for rule in sequence:
        outcome = _APPLY[rule.kind](policy, request, can_reobserve)
        if outcome is not None:
            break
    else:  # unreachable with a loaded policy: every sequence ends in the unmatched rule
        raise PolicyError(f"configuration {configuration}: no rule fired")
    return c.Decision(
        request_id=request.request_id, configuration=configuration, disposition=outcome.disposition,
        refusal=outcome.refusal, remeasure_route=outcome.route,
        destination=_DESTINATION.get(outcome.disposition, "none"),
        reasons=(c.Reason(rule_id=rule.id, text=outcome.text),), confidence=None, policy=policy.hashes(),
        model_id=None, code_sha=code_sha, adjudication=None)
