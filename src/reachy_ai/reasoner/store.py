"""DecisionStore: the reasoner's persistent memory of requests, decisions, outcomes and refusals (#82, part of #80).

On the pattern of reachy-1-2-sim experience/store.py (SQLite, versioned schema in PRAGMA user_version, WAL,
every write in one transaction, `open()` as a context manager that always closes), with two differences:
- it fails closed on a schema it does not know (no silent migration, and nothing is written to such a file);
- it is append-only. Database triggers refuse UPDATE and DELETE on every table, so a decision is never edited:
  a correction is a new outcome row and the original stays.

Holds each request and decision as validated disposition_decision.v1 JSON, with the policy hashes,
configuration, input_source and split. Images are never stored: only the bundle's path and SHA-256.

Leakage guard: `cases_for_tuning()` is the one accessor that may feed policy changes, and it returns
development cases only. A case_id keeps one split for ever, so a held-out case cannot come back as a
development one. `freeze()` and `unfreeze_for_next_version()` record which policy the trials ran against; both
are logged rows. The reports and the CSV export include held-out results: they are for the paper, not for tuning.

A case_id is one presentation of a part: its re-observations share it, and a repeated trial needs a new one.
Its source, sim or robot, is fixed by its first request: record() refuses a later request from the other one.
The latest recorded decision of a case under one policy version is that case's final decision.

Correct answers (owner, 2026-11-27) come only from sim truth, for a case whose source is sim, or from an
operator with an adjudication record that fills REQUIRED_ADJUDICATION_FIELDS. add_outcome() refuses anything else,
and nothing is inferred.

Scoring (owner decisions on #81, 2026-11-27):
- a pending case, awaiting another look or a correct answer, counts as not decided autonomously: it stays in
  coverage's denominator and is listed separately;
- acting on a cell that is not ready is its own error, acted_when_not_ready, not a false accept: accept, reject,
  rework or remeasure/reobserve (re-looking moves the head) while the request's cell_ready was false, in any
  configuration. Every decision of a case is looked at, not only the final one, and the case is charged once. It
  needs no correct answer. The confusion table is about part quality only.

No motion, SDK or network.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from reachy_ai.config import DECISION_STORE_PATH
from reachy_ai.reasoner import contract as c

SCHEMA_VERSION = 1
BUSY_TIMEOUT_S = 10.0

OUTCOME_SOURCES = ("sim_truth", "operator")
#: What an operator's correct answer must fill with more than blanks. The other Appendix A fields may stay empty.
REQUIRED_ADJUDICATION_FIELDS = ("adjudicator", "actual_location_final_disposition", "human_adjudication_notes")
#: A final decision in one of these commits to an answer; a refusal or a referral abstains.
COMMITTED = ("accept", "rework", "reject")
#: The scoring_costs keys of spec_v1.yaml's authority block.
COST_KEYS = ("false_accept", "false_reject_or_scrap_of_good_part", "wrong_rework_or_remeasure_routing",
             "unnecessary_referral", "correct_decision_or_justified_abstention", "acted_when_not_ready")
CORRECT = "correct_decision_or_justified_abstention"
ACTED_WHEN_NOT_READY = "acted_when_not_ready"

#: What makes two decisions the work of one policy version.
POLICY_COLUMNS = ("configuration", "spec_sha256", "rules_sha256", "prompt_sha256", "split_sha256", "model_id")
#: Results are never pooled across these.
GROUP_COLUMNS = POLICY_COLUMNS + ("split", "source", "input_source")

_TABLES = ("requests", "decisions", "outcomes", "policy_events")

_SCHEMA = (
    """CREATE TABLE requests (
        request_id          TEXT PRIMARY KEY,
        case_id             TEXT NOT NULL,
        split               TEXT NOT NULL CHECK (split IN ('development', 'held_out')),
        source              TEXT NOT NULL,
        input_source        TEXT NOT NULL,
        instance_id         TEXT NOT NULL,
        lot                 TEXT NOT NULL,
        expected_design_id  TEXT,
        reobservation_count INTEGER NOT NULL,
        bundle_path         TEXT,
        bundle_sha256       TEXT,
        request_json        TEXT NOT NULL,
        recorded_utc        TEXT NOT NULL
    )""",
    """CREATE TABLE decisions (
        decision_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id      TEXT NOT NULL REFERENCES requests(request_id),
        configuration   TEXT NOT NULL,
        disposition     TEXT,
        refusal         TEXT,
        remeasure_route TEXT,
        destination     TEXT NOT NULL,
        confidence      REAL,
        spec_sha256     TEXT NOT NULL,
        rules_sha256    TEXT NOT NULL,
        prompt_sha256   TEXT,
        split_sha256    TEXT NOT NULL,
        model_id        TEXT,
        code_sha        TEXT NOT NULL,
        decision_json   TEXT NOT NULL,
        recorded_utc    TEXT NOT NULL,
        CHECK ((disposition IS NULL) != (refusal IS NULL))
    )""",
    """CREATE TABLE outcomes (
        outcome_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_id           INTEGER NOT NULL REFERENCES decisions(decision_id),
        source                TEXT NOT NULL CHECK (source IN ('sim_truth', 'operator')),
        reference_disposition TEXT,
        reference_refusal     TEXT,
        adjudication_json     TEXT,
        operator_rating       INTEGER CHECK (operator_rating BETWEEN 1 AND 5),
        twin_json             TEXT,
        recorded_utc          TEXT NOT NULL
    )""",
    """CREATE TABLE policy_events (
        event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        event         TEXT NOT NULL CHECK (event IN ('freeze', 'unfreeze')),
        policy_json   TEXT NOT NULL,
        note          TEXT NOT NULL,
        recorded_utc  TEXT NOT NULL
    )""",
    "CREATE INDEX idx_requests_case ON requests(case_id)",
    "CREATE INDEX idx_decisions_request ON decisions(request_id)",
    "CREATE INDEX idx_outcomes_decision ON outcomes(decision_id)",
) + tuple(
    f"CREATE TRIGGER {table}_no_{verb.lower()} BEFORE {verb} ON {table} "
    f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END"
    for table in _TABLES for verb in ("UPDATE", "DELETE"))

#: Every object a version-1 store must have.
_OBJECTS = frozenset(statement.split()[2] for statement in _SCHEMA)


class DecisionStoreError(RuntimeError):
    """A store that cannot be used, or a write that is refused."""


class LeakageError(DecisionStoreError):
    """Held-out cases were asked for as tuning data."""


# ---------------------------------------------------------------- values

@dataclass(frozen=True)
class TwinResult:
    """What happened in the twin. None means not measured; empty contacts means none were seen."""
    feasible: bool | None = None
    placed: bool | None = None
    contacts: tuple[str, ...] | None = None

    def __post_init__(self):
        for name in ("feasible", "placed"):
            if getattr(self, name) not in (True, False, None):
                raise DecisionStoreError(f"TwinResult.{name} must be true, false or None")
        if self.contacts is not None:
            if isinstance(self.contacts, str) or not all(isinstance(v, str) and v for v in self.contacts):
                raise DecisionStoreError("TwinResult.contacts must be a list of names")
            object.__setattr__(self, "contacts", tuple(self.contacts))

    def to_dict(self) -> dict:
        return {"feasible": self.feasible, "placed": self.placed,
                "contacts": None if self.contacts is None else list(self.contacts)}


@dataclass(frozen=True)
class Outcome:
    outcome_id: int
    decision_id: int
    source: str
    reference_disposition: str | None  # the right answer, if this row states one: a disposition ...
    reference_refusal: str | None      # ... or the refusal that was right
    adjudication: c.Adjudication | None
    twin: TwinResult | None
    recorded_utc: str

    @property
    def reference(self) -> str | None:
        return _label(self.reference_disposition, self.reference_refusal)


@dataclass(frozen=True)
class TuningCase:
    case_id: str
    split: str
    requests: tuple[c.Request, ...]
    decisions: tuple[tuple[int, c.Decision], ...]  # (decision_id, decision)
    outcomes: tuple[Outcome, ...]


@dataclass(frozen=True)
class FrozenPolicy:
    policy: c.PolicyHashes
    note: str
    frozen_utc: str


# ---------------------------------------------------------------- scoring

def _label(disposition: str | None, refusal: str | None, route: str | None = None) -> str | None:
    """One word per answer: "accept", "remeasure:refer", "refusal:spec_unknown"."""
    if refusal is not None:
        return f"refusal:{refusal}"
    if disposition is None:
        return None
    return disposition if route is None else f"{disposition}:{route}"


def _acted_when_not_ready(row: Mapping[str, Any]) -> bool:
    """A decision that acts, on a cell that was not ready. A referral and a refusal move nothing."""
    acts = row["disposition"] in COMMITTED or row["remeasure_route"] == "reobserve"
    return acts and not row["cell_ready"]


def cost_key(disposition: str | None, refusal: str | None, reference_disposition: str | None,
             reference_refusal: str | None) -> str:
    """Which row of the scoring-cost table a final decision falls in on part quality, against the reference
    answer. acted_when_not_ready is not decided here: it needs no reference.

    An abstention is a refusal or a remeasure. It is justified when the reference answer is also an abstention
    (a refusal, or a part the spec says to remeasure); the two need not name the same reason."""
    if refusal is not None or disposition == "remeasure":
        justified = reference_refusal is not None or reference_disposition == "remeasure"
        return CORRECT if justified else "unnecessary_referral"
    if disposition == reference_disposition:
        return CORRECT
    if disposition == "accept":
        return "false_accept"
    if disposition == "reject" and reference_disposition == "accept":
        return "false_reject_or_scrap_of_good_part"
    return "wrong_rework_or_remeasure_routing"


def _check_costs(costs: Mapping[str, Any]) -> dict[str, float]:
    missing = [k for k in COST_KEYS if k not in costs]
    if missing:
        raise DecisionStoreError(f"scoring costs lack {', '.join(missing)}")
    return {k: float(costs[k]) for k in COST_KEYS}


def _ratio(numerator: float, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _pending(awaiting_look: list[str], awaiting_reference: list[str]) -> dict:
    return {"pending": len(awaiting_look) + len(awaiting_reference),
            "pending_cases": {"awaiting_look": sorted(awaiting_look),
                              "awaiting_reference": sorted(awaiting_reference)}}


def _selective(committed: list, cases: int) -> dict:
    """Coverage of all `cases`, and risk over `committed`, a list of (confidence, cost_key, cost)."""
    return {"committed": len(committed), "coverage": _ratio(len(committed), cases),
            "selective_error_rate": _ratio(sum(1 for _, key, _ in committed if key != CORRECT), len(committed)),
            "selective_cost": _ratio(sum(cost for _, _, cost in committed), len(committed))}


# ---------------------------------------------------------------- helpers

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, allow_nan=False)


def _rows(cursor: sqlite3.Cursor) -> list[dict]:
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _outcome(row: Mapping[str, Any]) -> Outcome:
    adjudication = row["adjudication_json"]
    twin = row["twin_json"]
    return Outcome(
        outcome_id=row["outcome_id"], decision_id=row["decision_id"], source=row["source"],
        reference_disposition=row["reference_disposition"], reference_refusal=row["reference_refusal"],
        adjudication=None if adjudication is None else c.Adjudication.from_dict(json.loads(adjudication)),
        twin=None if twin is None else TwinResult(**json.loads(twin)), recorded_utc=row["recorded_utc"])


def _inspect(conn: sqlite3.Connection, path: Path) -> bool:
    """True for a new, empty database; False for a good version-1 store. Anything else raises. Reads only."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    objects = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")}
    if version == 0 and not objects:
        return True
    if version != SCHEMA_VERSION:
        raise DecisionStoreError(f"{path}: schema version {version}, this code reads only {SCHEMA_VERSION}")
    missing = sorted(_OBJECTS - objects)
    if missing:
        raise DecisionStoreError(f"{path}: not a decision store, it lacks {', '.join(missing)}")
    return False


def _use_wal(conn: sqlite3.Connection) -> None:
    """SQLite does not wait out another connection when it changes the journal mode, so wait here."""
    deadline = time.monotonic() + BUSY_TIMEOUT_S
    while True:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


# ---------------------------------------------------------------- the store

class DecisionStore:
    """Use `with DecisionStore.open(path) as store:`."""

    def __init__(self, conn: sqlite3.Connection, path: Path) -> None:
        self._conn = conn
        self.path = path

    @classmethod
    @contextlib.contextmanager
    def open(cls, path: str | Path = DECISION_STORE_PATH) -> Iterator["DecisionStore"]:
        """Open the store, creating it if the file does not exist or is empty. Always closes."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # isolation_level=None: transactions are only the explicit ones in _write()
            conn = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_S, isolation_level=None)
        except sqlite3.Error as exc:
            raise DecisionStoreError(f"{path}: {exc}") from exc
        try:
            try:
                # Refuse before anything, even the journal mode, is written. In one read transaction, so that another
                # connection creating the schema cannot be seen half way.
                conn.execute("BEGIN")
                _inspect(conn, path)
                conn.execute("COMMIT")
                _use_wal(conn)
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("PRAGMA recursive_triggers=ON")  # so REPLACE cannot get round the delete triggers
                store = cls(conn, path)
                with store._write():
                    if _inspect(conn, path):  # again, under the write lock
                        for statement in _SCHEMA:
                            conn.execute(statement)
                        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            except sqlite3.DatabaseError as exc:
                raise DecisionStoreError(f"{path}: {exc}") from exc
            yield store
        finally:
            conn.close()

    @contextlib.contextmanager
    def _write(self) -> Iterator[None]:
        """One transaction, holding the write lock from the start so that checks and inserts are atomic."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._conn.execute("COMMIT")
        except BaseException:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK")
            raise

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        return _rows(self._conn.execute(sql, params))

    def _insert(self, table: str, values: Mapping[str, Any]) -> int:
        columns = ", ".join(values)
        marks = ", ".join("?" * len(values))
        return self._conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})",
                                  tuple(values.values())).lastrowid

    # ------------------------------------------------------------ requests and decisions

    def record(self, request: c.Request, decision: c.Decision) -> int:
        """Store a request (once) and a decision about it; returns the decision_id.

        Several decisions may share one request (configurations A, B and E), and every one is kept."""
        if not isinstance(request, c.Request) or not isinstance(decision, c.Decision):
            raise DecisionStoreError("record() takes a validated Request and Decision")
        if decision.request_id != request.request_id:
            raise DecisionStoreError(f"the decision is for request {decision.request_id}, "
                                     f"not {request.request_id}")
        if decision.adjudication is not None:
            raise DecisionStoreError("a decision is recorded without adjudication: add it as an outcome")
        request_json = _canonical(request.to_dict())
        now = _now()
        with self._write():
            if not self._stored(request):
                case = request.case
                bundle = request.observation.bundle
                self._insert("requests", {
                    "request_id": request.request_id, "case_id": case.case_id, "split": case.split,
                    "source": request.source, "input_source": request.input_source,
                    "instance_id": case.instance_id, "lot": case.lot,
                    "expected_design_id": case.expected_design_id,
                    "reobservation_count": request.reobservation_count,
                    "bundle_path": None if bundle is None else bundle.path,
                    "bundle_sha256": None if bundle is None else bundle.sha256,
                    "request_json": request_json, "recorded_utc": now})
            policy = decision.policy
            return self._insert("decisions", {
                "request_id": request.request_id, "configuration": decision.configuration,
                "disposition": decision.disposition, "refusal": decision.refusal,
                "remeasure_route": decision.remeasure_route, "destination": decision.destination,
                "confidence": decision.confidence, "spec_sha256": policy.spec_sha256,
                "rules_sha256": policy.rules_sha256, "prompt_sha256": policy.prompt_sha256,
                "split_sha256": policy.split_sha256, "model_id": decision.model_id,
                "code_sha": decision.code_sha, "decision_json": _canonical(decision.to_dict()),
                "recorded_utc": now})

    def _stored(self, request: c.Request) -> bool:
        """True if this request is already stored, False if record() may add it. Anything else raises."""
        stored = self._query("SELECT request_json FROM requests WHERE request_id=?", (request.request_id,))
        if stored:
            if stored[0]["request_json"] != _canonical(request.to_dict()):
                raise DecisionStoreError(f"request {request.request_id} is already stored with other content")
            return True
        case = request.case
        other = self._query("SELECT split FROM requests WHERE case_id=? AND split!=? LIMIT 1",
                            (case.case_id, case.split))
        if other:
            raise LeakageError(f"case {case.case_id} is already {other[0]['split']}; it cannot also be {case.split}")
        other = self._query("SELECT source FROM requests WHERE case_id=? AND source!=? LIMIT 1",
                            (case.case_id, request.source))
        if other:
            raise DecisionStoreError(f"case {case.case_id} is a {other[0]['source']} case, fixed by its "
                                     f"first request: a {request.source} request needs another case_id")
        return False

    def check_request(self, request: c.Request) -> None:
        """Raise what record() would raise for this request. Writes nothing. For a caller that is about to spend
        something it cannot take back, such as the AI reasoner's one call per case."""
        self._stored(request)

    def case_requests(self, case_id: str) -> list[c.Request]:
        """The case's requests, oldest first."""
        return [c.Request.from_dict(json.loads(row["request_json"])) for row in
                self._query("SELECT request_json FROM requests WHERE case_id=? ORDER BY rowid", (case_id,))]

    def request(self, request_id: str) -> c.Request:
        rows = self._query("SELECT request_json FROM requests WHERE request_id=?", (request_id,))
        if not rows:
            raise DecisionStoreError(f"no request {request_id}")
        return c.Request.from_dict(json.loads(rows[0]["request_json"]))

    def has_decision(self, case_id: str, configuration: str) -> bool:
        return bool(self._query("SELECT 1 FROM decisions d JOIN requests r USING (request_id) "
                                "WHERE r.case_id=? AND d.configuration=? LIMIT 1", (case_id, configuration)))

    def get(self, decision_id: int) -> tuple[c.Request, c.Decision]:
        rows = self._query("SELECT r.request_json, d.decision_json FROM decisions d JOIN requests r "
                           "USING (request_id) WHERE d.decision_id=?", (decision_id,))
        if not rows:
            raise DecisionStoreError(f"no decision {decision_id}")
        return (c.Request.from_dict(json.loads(rows[0]["request_json"])),
                c.Decision.from_dict(json.loads(rows[0]["decision_json"])))

    def refusals(self) -> list[dict]:
        """Every refusal with its reasons, oldest first."""
        rows = self._query("SELECT d.decision_id, r.case_id, r.split, d.configuration, d.refusal, "
                           "d.decision_json, d.recorded_utc FROM decisions d JOIN requests r USING (request_id) "
                           "WHERE d.refusal IS NOT NULL ORDER BY d.decision_id")
        for row in rows:
            row["reasons"] = json.loads(row.pop("decision_json"))["reasons"]
        return rows

    # ------------------------------------------------------------ outcomes

    def add_outcome(self, decision_id: int, source: str, *, reference_disposition: str | None = None,
                    reference_refusal: str | None = None, adjudication: c.Adjudication | None = None,
                    twin: TwinResult | None = None) -> int:
        """Append an outcome of a decision; returns the outcome_id. A correction is another call.

        The reference is the right answer for the case: a disposition, or the refusal that was right. An
        operator's reference needs an adjudication record that fills REQUIRED_ADJUDICATION_FIELDS, even when it
        agrees with the decision; one that differs is an override, and its notes are the reason (spec_v1
        authority). A sim_truth reference is accepted only when every request of the case came from the sim. The
        operator rating travels in the adjudication."""
        if source not in OUTCOME_SOURCES:
            raise DecisionStoreError(f"outcome source {source!r} is not one of {', '.join(OUTCOME_SOURCES)}")
        if reference_disposition is not None and reference_refusal is not None:
            raise DecisionStoreError("a reference is a disposition or a refusal, not both")
        if reference_disposition is not None and reference_disposition not in c.DISPOSITIONS:
            raise DecisionStoreError(f"unknown reference disposition {reference_disposition!r}")
        if reference_refusal is not None and reference_refusal not in c.REFUSALS:
            raise DecisionStoreError(f"unknown reference refusal {reference_refusal!r}")
        if adjudication is not None and not isinstance(adjudication, c.Adjudication):
            raise DecisionStoreError("adjudication must be a contract Adjudication")
        if twin is not None and not isinstance(twin, TwinResult):
            raise DecisionStoreError("twin must be a TwinResult")
        reference = _label(reference_disposition, reference_refusal)
        if reference is None and adjudication is None and twin is None:
            raise DecisionStoreError("an empty outcome: give a reference, an adjudication or a twin result")
        with self._write():
            rows = self._query("SELECT disposition, refusal FROM decisions WHERE decision_id=?", (decision_id,))
            if not rows:
                raise DecisionStoreError(f"no decision {decision_id}")
            decided = _label(rows[0]["disposition"], rows[0]["refusal"])
            if source == "operator" and reference not in (None, decided):
                if adjudication is None or adjudication.human_adjudication_notes is None:
                    raise DecisionStoreError(f"an operator override ({decided} -> {reference}) needs an "
                                             "adjudication with human_adjudication_notes")
            if source == "operator" and reference is not None and adjudication is None:
                raise DecisionStoreError("an operator answer needs an adjudication record (the Appendix A fields "
                                         "and the adjudicator), even when it agrees with the decision")
            if source == "operator" and reference is not None:
                blank = [name for name in REQUIRED_ADJUDICATION_FIELDS
                         if not (getattr(adjudication, name) or "").strip()]
                if blank:
                    raise DecisionStoreError(f"an operator answer needs these adjudication fields filled: "
                                             f"{', '.join(blank)}")
            if source == "sim_truth" and reference is not None:
                real = self._query(
                    "SELECT other.case_id, other.source FROM decisions d JOIN requests r USING (request_id) "
                    "JOIN requests other ON other.case_id = r.case_id "
                    "WHERE d.decision_id=? AND other.source != 'sim' LIMIT 1", (decision_id,))
                if real:
                    raise DecisionStoreError(f"a sim_truth answer is accepted only for a case whose source is sim: "
                                             f"case {real[0]['case_id']} has a {real[0]['source']} request")
            return self._insert("outcomes", {
                "decision_id": decision_id, "source": source,
                "reference_disposition": reference_disposition, "reference_refusal": reference_refusal,
                "adjudication_json": None if adjudication is None else _canonical(adjudication.to_dict()),
                "operator_rating": None if adjudication is None else adjudication.operator_rating,
                "twin_json": None if twin is None else _canonical(twin.to_dict()), "recorded_utc": _now()})

    def outcomes(self, decision_id: int) -> list[Outcome]:
        """The decision's outcomes, oldest first."""
        return [_outcome(row) for row in
                self._query("SELECT * FROM outcomes WHERE decision_id=? ORDER BY outcome_id", (decision_id,))]

    # ------------------------------------------------------------ the leakage guard

    def frozen_policy(self) -> FrozenPolicy | None:
        rows = self._query("SELECT * FROM policy_events ORDER BY event_id DESC LIMIT 1")
        if not rows or rows[0]["event"] != "freeze":
            return None
        return FrozenPolicy(policy=c.PolicyHashes.from_dict(json.loads(rows[0]["policy_json"])),
                            note=rows[0]["note"], frozen_utc=rows[0]["recorded_utc"])

    def policy_log(self) -> list[dict]:
        return self._query("SELECT * FROM policy_events ORDER BY event_id")

    def freeze(self, policy_hashes: c.PolicyHashes, note: str) -> None:
        """Record the policy the held-out trials will run against."""
        if not isinstance(policy_hashes, c.PolicyHashes):
            raise DecisionStoreError("freeze() takes the contract's PolicyHashes")
        if not isinstance(note, str) or not note.strip():
            raise DecisionStoreError("freeze() needs a note")
        with self._write():
            if self.frozen_policy() is not None:
                raise DecisionStoreError("a policy is already frozen: unfreeze_for_next_version() first")
            self._insert("policy_events", {"event": "freeze", "policy_json": _canonical(policy_hashes.to_dict()),
                                           "note": note, "recorded_utc": _now()})

    def unfreeze_for_next_version(self, note: str = "") -> None:
        """After the trials: end the freeze. Logged; the freeze row stays."""
        with self._write():
            frozen = self.frozen_policy()
            if frozen is None:
                raise DecisionStoreError("no policy is frozen")
            self._insert("policy_events", {"event": "unfreeze",
                                           "policy_json": _canonical(frozen.policy.to_dict()),
                                           "note": note, "recorded_utc": _now()})

    def cases_for_tuning(self, splits: tuple[str, ...] = ("development",)) -> list[TuningCase]:
        """The cases that may feed a policy change: development only.

        Asking for held-out ones raises, whether or not a policy is frozen (owner, 2026-11-27, #81). Using them
        for a next version needs an explicit, logged unfreeze after the trials. unfreeze_for_next_version() logs
        that, but this accessor does not open afterwards: no method here releases held-out cases for tuning."""
        splits = (splits,) if isinstance(splits, str) else tuple(splits)
        unknown = [s for s in splits if s not in c.SPLITS]
        if unknown:
            raise DecisionStoreError(f"unknown split(s): {', '.join(map(repr, unknown))}")
        if "held_out" in splits:
            raise LeakageError("held-out cases are never tuning data")
        if not splits:
            return []
        requests = self._query("SELECT * FROM requests WHERE split='development' ORDER BY rowid")
        cases: dict[str, dict] = {}
        for row in requests:
            entry = cases.setdefault(row["case_id"], {"requests": [], "decisions": [], "outcomes": []})
            entry["requests"].append(c.Request.from_dict(json.loads(row["request_json"])))
        for row in self._query("SELECT d.decision_id, d.decision_json, r.case_id FROM decisions d "
                               "JOIN requests r USING (request_id) WHERE r.split='development' "
                               "ORDER BY d.decision_id"):
            cases[row["case_id"]]["decisions"].append(
                (row["decision_id"], c.Decision.from_dict(json.loads(row["decision_json"]))))
        for row in self._query("SELECT o.*, r.case_id FROM outcomes o JOIN decisions d USING (decision_id) "
                               "JOIN requests r USING (request_id) WHERE r.split='development' "
                               "ORDER BY o.outcome_id"):
            cases[row["case_id"]]["outcomes"].append(_outcome(row))
        return [TuningCase(case_id=case_id, split="development", requests=tuple(v["requests"]),
                           decisions=tuple(v["decisions"]), outcomes=tuple(v["outcomes"]))
                for case_id, v in cases.items()]

    # ------------------------------------------------------------ reports

    def _joined(self) -> list[dict]:
        """Each decision with its request's columns, in decision order. Both tables have a recorded_utc, so the
        decision's is selected under its own name. cell_ready is read from the stored request."""
        rows = self._query("SELECT r.*, d.*, d.recorded_utc AS decision_recorded_utc FROM decisions d "
                           "JOIN requests r USING (request_id) ORDER BY d.decision_id")
        for row in rows:
            row["cell_ready"] = json.loads(row["request_json"])["health"]["cell_ready"]
        return rows

    def _references(self, reference_source: str | None) -> tuple[dict, set]:
        """decision_id -> (reference_disposition, reference_refusal) from the latest outcome that states one,
        and the decision_ids whose sources' latest references disagree."""
        if reference_source is not None and reference_source not in OUTCOME_SOURCES:
            raise DecisionStoreError(f"unknown reference source {reference_source!r}")
        latest: dict[int, tuple] = {}
        by_source: dict[int, dict] = {}
        for row in self._query("SELECT * FROM outcomes WHERE reference_disposition IS NOT NULL "
                               "OR reference_refusal IS NOT NULL ORDER BY outcome_id"):
            reference = (row["reference_disposition"], row["reference_refusal"])
            by_source.setdefault(row["decision_id"], {})[row["source"]] = reference
            if reference_source in (None, row["source"]):
                latest[row["decision_id"]] = reference
        conflicts = {key for key, values in by_source.items() if len(set(values.values())) > 1}
        return latest, conflicts

    def _final_by_group(self) -> list[tuple[dict, list[dict], list[dict]]]:
        """(group, the final decision row of each of its cases, every decision row of the group)."""
        groups: dict[tuple, dict] = {}
        every: dict[tuple, list] = {}
        for row in self._joined():
            key = tuple(row[column] for column in GROUP_COLUMNS)
            groups.setdefault(key, {})[row["case_id"]] = row  # rows are in decision_id order: the last wins
            every.setdefault(key, []).append(row)
        ordered = sorted(groups, key=lambda k: tuple("" if v is None else v for v in k))
        return [(dict(zip(GROUP_COLUMNS, key)), list(groups[key].values()), every[key]) for key in ordered]

    def confusion_table(self, *, reference_source: str | None = None) -> list[dict]:
        """One entry per policy version and split: counts[reference answer][final decision].

        It is about part quality only: acting on a cell that was not ready is reported by risk_coverage().
        `pending` cases still await a re-observation or a reference; they are listed in `pending_cases` and are
        not in `counts`."""
        references, conflicts = self._references(reference_source)
        report = []
        for group, finals, rows in self._final_by_group():
            counts: dict[str, dict[str, int]] = {}
            awaiting_look: list[str] = []
            awaiting_reference: list[str] = []
            graded = 0
            for row in finals:
                if row["remeasure_route"] == "reobserve":
                    awaiting_look.append(row["case_id"])
                    continue
                if row["decision_id"] not in references:
                    awaiting_reference.append(row["case_id"])
                    continue
                graded += 1
                answer = _label(row["disposition"], row["refusal"], row["remeasure_route"])
                line = counts.setdefault(_label(*references[row["decision_id"]]), {})
                line[answer] = line.get(answer, 0) + 1
            report.append({**group, "decisions": len(rows), "cases": len(finals), "graded": graded,
                           **_pending(awaiting_look, awaiting_reference),
                           "reference_conflicts": sum(1 for r in finals if r["decision_id"] in conflicts),
                           "counts": counts})
        return report

    def risk_coverage(self, costs: Mapping[str, Any], *, reference_source: str | None = None) -> list[dict]:
        """Selective risk against coverage, per policy version and split.

        One result per case, from its final decision, except that a case in which any decision acted on a cell
        that was not ready is charged acted_when_not_ready instead of a part-quality cost; the block of that
        name lists those cases and counts the decisions that acted. A case is `graded` when it is charged that
        or its final decision has a reference; every other case is `pending`, awaiting a re-observation or a
        reference. `committed` are the graded cases whose final decision is accept, rework or reject, and
        `abstained` those ending in a referral or a refusal (a graded case still awaiting a re-look is neither).
        Coverage is the committed share of ALL cases, so a pending case counts as not decided autonomously. The
        selective risk is the error rate, and the mean cost, over the committed cases. `mean_cost` is over the
        graded cases, so it also charges unnecessary referrals. `costs` is the spec's authority.scoring_costs. A
        figure with nothing to divide by is None.

        `curve` lowers a confidence threshold over the committed decisions. It is empty unless every committed
        decision has a confidence (A, B and E have none, so each is one point)."""
        table = _check_costs(costs)
        references, conflicts = self._references(reference_source)
        report = []
        for group, finals, rows in self._final_by_group():
            acted = [row for row in rows if _acted_when_not_ready(row)]
            acted_cases = {row["case_id"] for row in acted}
            awaiting_look: list[str] = []
            awaiting_reference: list[str] = []
            scored = []  # (final decision row, cost key)
            for row in finals:
                if row["case_id"] in acted_cases:
                    scored.append((row, ACTED_WHEN_NOT_READY))
                elif row["remeasure_route"] == "reobserve":
                    awaiting_look.append(row["case_id"])
                elif row["decision_id"] not in references:
                    awaiting_reference.append(row["case_id"])
                else:
                    scored.append((row, cost_key(row["disposition"], row["refusal"], *references[row["decision_id"]])))
            committed = [(row["confidence"], key, table[key]) for row, key in scored if row["disposition"] in COMMITTED]
            abstained = sum(1 for row, _ in scored
                            if row["refusal"] is not None or row["remeasure_route"] == "refer")
            curve = []
            if committed and all(confidence is not None for confidence, _, _ in committed):
                for threshold in sorted({confidence for confidence, _, _ in committed}, reverse=True):
                    above = [item for item in committed if item[0] >= threshold]
                    curve.append({"threshold": threshold, **_selective(above, len(finals))})
            by_cost_key: dict[str, int] = {}
            for _, key in scored:
                by_cost_key[key] = by_cost_key.get(key, 0) + 1
            report.append({**group, "decisions": len(rows), "cases": len(finals), "graded": len(scored),
                           **_pending(awaiting_look, awaiting_reference), "abstained": abstained,
                           ACTED_WHEN_NOT_READY: {"cases": sorted(acted_cases), "decisions": len(acted)},
                           "reference_conflicts": sum(1 for r in finals if r["decision_id"] in conflicts),
                           **_selective(committed, len(finals)),
                           "mean_cost": _ratio(sum(table[key] for _, key in scored), len(scored)),
                           "by_cost_key": by_cost_key, "curve": curve})
        return report

    def refusal_counts(self) -> list[dict]:
        """Every recorded refusal (not only final ones), counted by policy version, split and reason."""
        counts: dict[tuple, int] = {}
        for row in self._joined():
            if row["refusal"] is not None:
                key = tuple(row[column] for column in GROUP_COLUMNS) + (row["refusal"],)
                counts[key] = counts.get(key, 0) + 1
        ordered = sorted(counts, key=lambda k: tuple("" if v is None else v for v in k))
        return [{**dict(zip(GROUP_COLUMNS + ("refusal",), key)), "count": counts[key]} for key in ordered]

    # ------------------------------------------------------------ CSV

    def export_csv(self, path: str | Path | None = None) -> str:
        """The whole history as CSV, written to `path` if given and returned.

        One row per outcome, and one row for a decision with none. `final` marks each case's final decision
        under its policy version."""
        finals = {row["decision_id"] for _, rows, _ in self._final_by_group() for row in rows}
        outcomes: dict[int, list[dict]] = {}
        for row in self._query("SELECT * FROM outcomes ORDER BY outcome_id"):
            outcomes.setdefault(row["decision_id"], []).append(row)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in self._joined():
            decision = json.loads(row["decision_json"])
            base = {column: row[column] for column in CSV_COLUMNS if column in row}
            base.update(final=int(row["decision_id"] in finals),
                        reasons=" | ".join(f"{r['rule_id']}: {r['text']}" for r in decision["reasons"]))
            for outcome in outcomes.get(row["decision_id"], [None]):
                line = dict(base)
                if outcome is not None:
                    line.update(outcome_id=outcome["outcome_id"], outcome_source=outcome["source"],
                                reference_disposition=outcome["reference_disposition"],
                                reference_refusal=outcome["reference_refusal"],
                                operator_rating=outcome["operator_rating"],
                                outcome_recorded_utc=outcome["recorded_utc"])
                    if outcome["adjudication_json"] is not None:
                        adjudication = json.loads(outcome["adjudication_json"])
                        line.update({k: adjudication[k] for k in c.APPENDIX_A_FIELDS + ("adjudicator",)})
                    if outcome["twin_json"] is not None:
                        twin = json.loads(outcome["twin_json"])
                        line.update(twin_feasible=twin["feasible"], twin_placed=twin["placed"],
                                    twin_contacts=None if twin["contacts"] is None
                                    else json.dumps(twin["contacts"]))
                writer.writerow(line)
        text = buffer.getvalue()
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
        return text


CSV_COLUMNS = (
    "decision_id", "request_id", "case_id", "split", "source", "input_source", "instance_id", "lot",
    "expected_design_id", "reobservation_count", "bundle_path", "bundle_sha256",
    "configuration", "disposition", "refusal", "remeasure_route", "destination", "confidence", "final", "reasons",
    "spec_sha256", "rules_sha256", "prompt_sha256", "split_sha256", "model_id", "code_sha",
    "decision_recorded_utc",
    "outcome_id", "outcome_source", "reference_disposition", "reference_refusal", "operator_rating",
    "adjudicator", *c.APPENDIX_A_FIELDS, "twin_feasible", "twin_placed", "twin_contacts", "outcome_recorded_utc",
)
