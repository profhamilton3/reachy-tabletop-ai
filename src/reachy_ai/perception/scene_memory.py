"""SceneMemoryStore: what has been seen on the board, and where the board is (#85, part of #80 and #13).

docs/design/perceived-scene.md section 4, build step P4. SQLite, with the schema version in PRAGMA user_version,
WAL, every write in one transaction, and `open()` as a context manager that always closes. The scene documents
stay in files beside the database, which holds their paths and SHA-256s.

SIM ONLY. The owner lifted the P4 hold for the sim (2026-11-27). Robot-side scene memory waits for the robot head
sweep, so a robot rig is refused.

MEMORY IS NEVER A MOTION INPUT. It is a prior, for association and for "has the rig moved?". A target or an
obstacle for motion comes from a fresh PerceivedScene. Nothing here returns anything a motion primitive takes, and
nothing here moves.

FAILS CLOSED
- A file that is missing (without create=True), empty, not a scene memory, of another schema version, or holding a
  record that cannot be read raises SceneMemoryError. It never reads as an empty memory.
- Entries are served only to a rig equal to theirs in every MATCH_FIELD. The others are kept, and inactive()
  reports each as "inactive: rig mismatch (<field>)" or "inactive: rig moved".
- The store owns rig_epoch, one count per physical rig (source, rig_layout_id, mount_id). open() takes the
  caller's rig at that epoch (a lower one is raised to it; a higher one is refused). `store.rig` is the rig to
  build the next scene with, and add_scene() refuses a scene of any other epoch.

COUNTING, NOT OVERWRITING (board poses; section 4.3)
Only a head_sweep_fit board pose counts: a single pair, the twin or a taught pose is kept as an observation and
changes no entry. THE SAME FIT COUNTS ONCE: one head sweep may build many scenes, and must not confirm itself.
In each rig, one live entry is the primary and any others are its competitors.
- Agree (|difference| <= entry bound + observation bound on every axis, of the board centre): the entry's count
  goes up and its estimate is the count-weighted mean. Its bound is the largest single-observation bound that
  went into it: it never shrinks with n.
- Disagree: no entry is edited. The observation opens a competing candidate.
- A candidate primary becomes confirmed after CONFIRMATIONS (K, OD-6) agreeing observations. A competitor
  supersedes the primary after K confirmations with none for the primary over the same span; both rows stay.
- OD-8: under another PipelineIdentity (any field, the code commit included) an entry keeps its row and state,
  and its effective count, estimate and bound restart from the new observation. total_count keeps the audit.
Agreement does not judge orientation: the design gives no bound for it. The estimate keeps the mean of the
quaternions. The rig check below judges the yaw.

Object sightings are episodic: each is an immutable observation, never confirmed into a fact.

THE RIG CHECK (section 4.4)
A new head-sweep pose is compared with the rig's confirmed board entry, and with the twin's grid when given. From
either one, more than RIG_MOVED_THRESHOLD_M (10 mm, OD-7) on the board centre, or more than RIG_TURNED_THRESHOLD_DEG
(2 degrees, PROVISIONAL; owner, 2026-11-27) of board yaw, fails: rig_epoch goes up, every entry of the older epochs
is inactive, the failing pose starts a candidate in the new epoch, and the caller is given a message for the
operator. A pass does not by itself edit an entry: counting then follows the rules above. With neither reference
there is no check ("no_reference").

scene_builder does not measure the board's orientation yet: its poses carry the identity quaternion (BOARD_NOTE).
So the yaw check cannot fail on a builder-made scene until the builder measures yaw.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import sqlite3
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from reachy_ai.perception import scene_contract as sc

LOG = logging.getLogger(__name__)

SCHEMA_VERSION = 1
BUSY_TIMEOUT_S = 10.0
RIG_MOVED_THRESHOLD_M = 0.010  # OD-7, on the board centre; re-check after robot head-sweep bundles
RIG_TURNED_THRESHOLD_DEG = 2.0  # PROVISIONAL (owner, 2026-11-27), on the board's yaw about the torso z axis
CONFIRMATIONS = 3              # OD-6: K
ALLOWED_SOURCES = ("sim",)     # robot-side memory waits for the robot head sweep
PHYSICAL_RIG_FIELDS = ("source", "rig_layout_id", "mount_id")

STATES = ("candidate", "confirmed", "superseded", "invalidated")
LIVE = ("candidate", "confirmed")
OBSERVATION_KINDS = ("board_pose", "object_sighting", "rig_check")

_IMMUTABLE = ("scenes", "observations", "confirmations")
_NEVER_DELETED = ("rigs", "entries")

_SCHEMA = (
    """CREATE TABLE rigs (
        rig_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        physical_key    TEXT NOT NULL,
        scope_key       TEXT NOT NULL,
        match_key       TEXT NOT NULL UNIQUE,
        rig_epoch       INTEGER NOT NULL,
        source          TEXT NOT NULL,
        rig_layout_id   TEXT NOT NULL,
        mount_id        TEXT NOT NULL,
        calibration_id  TEXT NOT NULL,
        zoom_level      TEXT NOT NULL,
        rig_json        TEXT NOT NULL,
        first_seen_utc  TEXT NOT NULL,
        last_seen_utc   TEXT NOT NULL
    )""",
    """CREATE TABLE scenes (
        scene_id         TEXT PRIMARY KEY,
        rig_id           INTEGER NOT NULL REFERENCES rigs(rig_id),
        pipeline_json    TEXT NOT NULL,
        document_sha256  TEXT NOT NULL,
        document_path    TEXT NOT NULL,
        created_utc      TEXT NOT NULL,
        recorded_utc     TEXT NOT NULL
    )""",
    """CREATE TABLE observations (
        observation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        scene_id        TEXT NOT NULL REFERENCES scenes(scene_id),
        rig_id          INTEGER NOT NULL REFERENCES rigs(rig_id),
        kind            TEXT NOT NULL CHECK (kind IN ('board_pose', 'object_sighting', 'rig_check')),
        payload_json    TEXT NOT NULL,
        fingerprint     TEXT,
        recorded_utc    TEXT NOT NULL
    )""",
    """CREATE TABLE entries (
        entry_id           INTEGER PRIMARY KEY AUTOINCREMENT,
        rig_id             INTEGER NOT NULL REFERENCES rigs(rig_id),
        kind               TEXT NOT NULL CHECK (kind IN ('board_pose')),
        estimate_json      TEXT NOT NULL,
        bounds_json        TEXT NOT NULL,
        state              TEXT NOT NULL CHECK (state IN ('candidate', 'confirmed', 'superseded', 'invalidated')),
        effective_count    INTEGER NOT NULL,
        total_count        INTEGER NOT NULL,
        span_count         INTEGER NOT NULL,
        pipeline_json      TEXT NOT NULL,
        challenges         INTEGER REFERENCES entries(entry_id),
        superseded_by      INTEGER REFERENCES entries(entry_id),
        invalid_reason     TEXT,
        created_utc        TEXT NOT NULL,
        last_confirmed_utc TEXT NOT NULL
    )""",
    """CREATE TABLE confirmations (
        confirmation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id         INTEGER NOT NULL REFERENCES entries(entry_id),
        observation_id   INTEGER NOT NULL REFERENCES observations(observation_id),
        residual_json    TEXT NOT NULL,
        verdict          TEXT NOT NULL CHECK (verdict IN ('agree', 'disagree')),
        recorded_utc     TEXT NOT NULL
    )""",
    "CREATE INDEX idx_observations_rig ON observations(rig_id, kind)",
    "CREATE INDEX idx_entries_rig ON entries(rig_id)",
) + tuple(
    f"CREATE TRIGGER {table}_no_{verb.lower()} BEFORE {verb} ON {table} "
    f"BEGIN SELECT RAISE(ABORT, '{table} rows are immutable'); END"
    for table in _IMMUTABLE for verb in ("UPDATE", "DELETE")
) + tuple(
    f"CREATE TRIGGER {table}_no_delete BEFORE DELETE ON {table} "
    f"BEGIN SELECT RAISE(ABORT, '{table} rows are never deleted'); END"
    for table in _NEVER_DELETED
)

_OBJECTS = frozenset(statement.split()[2] for statement in _SCHEMA)
#: schema version n -> the statements that take a store from n to n + 1. None yet.
_MIGRATIONS: dict[int, tuple[str, ...]] = {}


class SceneMemoryError(RuntimeError):
    """A store that cannot be used, or a read or write that is refused."""


# ---------------------------------------------------------------- what the store gives back

@dataclass(frozen=True)
class MemoryEntry:
    """A remembered board pose. A prior only: never a motion input."""
    entry_id: int
    kind: str                         # "board_pose"
    position_m: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    bounds_m: tuple[float, float, float]
    state: str                        # one of STATES
    effective_count: int              # sightings under the entry's present pipeline
    total_count: int                  # every sighting, for audit
    challenges: int | None            # the primary this candidate competes with
    superseded_by: int | None
    invalid_reason: str | None
    rig: sc.RigIdentity
    pipeline: sc.PipelineIdentity
    created_utc: str
    last_confirmed_utc: str
    inactive: str | None = None       # why it is not served to the current rig


@dataclass(frozen=True)
class Sighting:
    """One object as one scene saw it. Episodic: objects move. A prior only."""
    observation_id: int
    scene_id: str
    created_utc: str
    object: sc.PerceivedObject


@dataclass(frozen=True)
class RigCheck:
    verdict: str                      # "pass" | "fail" | "no_reference"
    distance_m: float | None          # from the confirmed board entry
    twin_distance_m: float | None     # from the twin's grid
    epoch_before: int
    epoch_after: int
    message: str                      # for the operator
    yaw_deg: float | None = None      # turn from the confirmed board entry, 0 to 180
    twin_yaw_deg: float | None = None  # turn from the twin's grid


@dataclass(frozen=True)
class BoardUpdate:
    verdict: str   # "no_board" | "not_counted" | "new_candidate" | "agree" | "competing" | "superseded"
    entry_id: int | None
    note: str


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    rig: sc.RigIdentity               # the current rig after this scene: its epoch may have gone up
    rig_check: RigCheck | None
    board: BoardUpdate
    sightings: int


# ---------------------------------------------------------------- helpers

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, allow_nan=False)


def _key(rig: sc.RigIdentity, fields: tuple[str, ...]) -> str:
    return _canonical({name: getattr(rig, name) for name in fields})


def _match_key(rig: sc.RigIdentity) -> str:
    return _key(rig, sc.MATCH_FIELDS)


def _scope_key(rig: sc.RigIdentity) -> str:
    """The rig at any epoch."""
    return _key(rig, tuple(f for f in sc.MATCH_FIELDS if f != "rig_epoch"))


def _rows(cursor: sqlite3.Cursor) -> list[dict]:
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _check_rig(rig: Any) -> None:
    if not isinstance(rig, sc.RigIdentity):
        raise SceneMemoryError(f"expected a RigIdentity, got {type(rig).__name__}")
    if rig.source not in ALLOWED_SOURCES:
        raise SceneMemoryError(f"scene memory is sim only: a {rig.source} rig is refused until the robot head sweep "
                               "gives a torso-frame board pose (owner, 2026-11-27)")


def _when(utc: str) -> tuple[str, float]:
    """A sort key for the contract's UTC text, with or without a fraction of a second."""
    return utc[:19], float("0" + utc[19:-1] if len(utc) > 20 else 0)


def _distance(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _yaw_deg(quat) -> float:
    """The heading of the board's +x axis about the torso z axis, in degrees. The same for q and -q."""
    w, x, y, z = quat
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _turn_deg(a, b) -> float:
    """The smallest yaw between two board orientations, 0 to 180."""
    return abs((_yaw_deg(a) - _yaw_deg(b) + 180.0) % 360.0 - 180.0)


def _mean_quat(kept, new, weight: int) -> tuple:
    """The count-weighted mean of two unit quaternions, taken on the same side of the sphere."""
    sign = 1.0 if sum(a * b for a, b in zip(kept, new)) >= 0 else -1.0
    mean = [a * weight + sign * b for a, b in zip(kept, new)]
    norm = math.sqrt(sum(v * v for v in mean))
    return tuple(v / norm for v in mean)


def _create(path: Path) -> None:
    """Make a complete, empty store, then put it at `path` in one step: no half-made store is ever there to open."""
    path.parent.mkdir(parents=True, exist_ok=True)
    draft = path.parent / f".{path.name}.{uuid.uuid4().hex}.new"
    try:
        conn = sqlite3.connect(str(draft), isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN")
            for statement in _SCHEMA:
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.execute("COMMIT")
        finally:
            conn.close()
        with contextlib.suppress(FileExistsError):  # another process made it first: theirs is as good
            os.link(draft, path)
    finally:
        for suffix in ("", "-wal", "-shm"):
            Path(f"{draft}{suffix}").unlink(missing_ok=True)


# ---------------------------------------------------------------- the store

class SceneMemoryStore:
    """Use `with SceneMemoryStore.open(path, current_rig) as memory:`."""

    def __init__(self, conn: sqlite3.Connection, path: Path, rig: sc.RigIdentity) -> None:
        self._conn = conn
        self.path = path
        self._caller_rig = rig
        self.documents = path.parent / f"{path.stem}_scenes"

    @classmethod
    @contextlib.contextmanager
    def open(cls, path: str | Path, current_rig: sc.RigIdentity, *,
             create: bool = False) -> Iterator["SceneMemoryStore"]:
        """Open the memory for `current_rig`. A new store is made only with create=True, where no file exists."""
        _check_rig(current_rig)
        path = Path(path)
        if not path.exists():
            if not create:
                raise SceneMemoryError(f"{path}: no scene memory here (create=True makes a new one)")
            try:
                _create(path)
            except (OSError, sqlite3.Error) as exc:
                raise SceneMemoryError(f"{path}: {exc}") from exc
        try:
            if not path.is_file() or path.stat().st_size == 0:
                raise SceneMemoryError(f"{path}: empty or not a file, so not a scene memory")
            conn = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_S, isolation_level=None)
        except (OSError, sqlite3.Error) as exc:
            raise SceneMemoryError(f"{path}: {exc}") from exc
        try:
            try:
                store = cls(conn, path, current_rig)
                store._check_schema()  # before anything is written
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("PRAGMA recursive_triggers=ON")  # so REPLACE cannot get round the delete triggers
                store._migrate()
                if current_rig.rig_epoch > store._epoch(current_rig):
                    raise SceneMemoryError(
                        f"rig_epoch {current_rig.rig_epoch} is ahead of this store ({store._epoch(current_rig)}): "
                        "only the store raises it, so this is another or a replaced store")
            except sqlite3.DatabaseError as exc:
                raise SceneMemoryError(f"{path}: {exc}") from exc
            yield store
        finally:
            conn.close()

    def _check_schema(self) -> int:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if not 1 <= version <= SCHEMA_VERSION:
            raise SceneMemoryError(f"{self.path}: schema version {version}, this code reads 1 to {SCHEMA_VERSION}")
        objects = {r[0] for r in self._conn.execute("SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")}
        missing = sorted(_OBJECTS - objects)
        if missing:
            raise SceneMemoryError(f"{self.path}: not a scene memory, it lacks {', '.join(missing)}")
        return version

    def _migrate(self) -> None:
        with self._write():
            for version in range(self._check_schema(), SCHEMA_VERSION):
                for statement in _MIGRATIONS[version]:
                    self._conn.execute(statement)
                self._conn.execute(f"PRAGMA user_version = {version + 1}")

    @contextlib.contextmanager
    def _write(self) -> Iterator[None]:
        """One transaction, holding the write lock from the start so that checks and writes are atomic."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._conn.execute("COMMIT")
        except BaseException:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK")
            raise

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        try:
            return _rows(self._conn.execute(sql, params))
        except sqlite3.DatabaseError as exc:
            raise SceneMemoryError(f"{self.path}: {exc}") from exc

    def _insert(self, table: str, values: Mapping[str, Any]) -> int:
        marks = ", ".join("?" * len(values))
        return self._conn.execute(f"INSERT INTO {table} ({', '.join(values)}) VALUES ({marks})",
                                  tuple(values.values())).lastrowid

    def _update(self, entry_id: int, **values: Any) -> None:
        assignments = ", ".join(f"{name}=?" for name in values)
        self._conn.execute(f"UPDATE entries SET {assignments} WHERE entry_id=?", (*values.values(), entry_id))

    # ------------------------------------------------------------ rigs

    def _epoch(self, rig: sc.RigIdentity) -> int:
        (row,) = self._query("SELECT MAX(rig_epoch) AS epoch FROM rigs WHERE physical_key=?",
                             (_key(rig, PHYSICAL_RIG_FIELDS),))
        return row["epoch"] or 0

    @property
    def rig(self) -> sc.RigIdentity:
        """The current rig, at the store's rig_epoch: build the next scene with this epoch."""
        return replace(self._caller_rig, rig_epoch=self._epoch(self._caller_rig))

    def _rig_row(self, rig: sc.RigIdentity) -> int:
        """The rig's row, made or marked as seen. Only inside a write."""
        now = _now()
        found = self._query("SELECT rig_id FROM rigs WHERE match_key=?", (_match_key(rig),))
        if found:
            self._conn.execute("UPDATE rigs SET last_seen_utc=? WHERE rig_id=?", (now, found[0]["rig_id"]))
            return found[0]["rig_id"]
        return self._insert("rigs", {
            "physical_key": _key(rig, PHYSICAL_RIG_FIELDS), "scope_key": _scope_key(rig),
            "match_key": _match_key(rig), "rig_epoch": rig.rig_epoch, "source": rig.source,
            "rig_layout_id": rig.rig_layout_id, "mount_id": rig.mount_id, "calibration_id": rig.calibration_id,
            "zoom_level": rig.zoom_level, "rig_json": _canonical(rig.to_dict()), "first_seen_utc": now,
            "last_seen_utc": now})

    def rigs(self) -> list[dict]:
        """Every rig seen, with first and last seen."""
        return [{"rig": self._parse(sc.RigIdentity.from_dict, row["rig_json"]),
                 "first_seen_utc": row["first_seen_utc"], "last_seen_utc": row["last_seen_utc"]}
                for row in self._query("SELECT * FROM rigs ORDER BY rig_id")]

    # ------------------------------------------------------------ reading entries

    def _parse(self, build, text: str):
        try:
            return build(json.loads(text))
        except (ValueError, TypeError, KeyError) as exc:
            raise SceneMemoryError(f"{self.path}: a stored record cannot be read: {exc}") from exc

    def _numbers(self, row: Mapping[str, Any]) -> tuple[tuple, tuple, tuple]:
        """(position, quaternion, bounds) of an entry row, or SceneMemoryError."""
        try:
            estimate = json.loads(row["estimate_json"])
            values = (estimate["position_m"], estimate["quat_wxyz"], json.loads(row["bounds_json"]))
            position, quat, bounds = (tuple(float(v) for v in value) for value in values)
            if (len(position), len(quat), len(bounds)) != (3, 4, 3):
                raise ValueError("wrong number of values")
            if not all(math.isfinite(v) for v in position + quat + bounds):
                raise ValueError("a value is not finite")
        except (ValueError, TypeError, KeyError) as exc:
            raise SceneMemoryError(f"{self.path}: entry {row['entry_id']} cannot be read: {exc}") from exc
        return position, quat, bounds

    def _entry(self, row: Mapping[str, Any], current: sc.RigIdentity) -> MemoryEntry:
        rig = self._parse(sc.RigIdentity.from_dict, row["rig_json"])
        position, quat, bounds = self._numbers(row)
        return MemoryEntry(
            entry_id=row["entry_id"], kind=row["kind"], position_m=position, quat_wxyz=quat, bounds_m=bounds,
            state=row["state"], effective_count=row["effective_count"], total_count=row["total_count"],
            challenges=row["challenges"], superseded_by=row["superseded_by"], invalid_reason=row["invalid_reason"],
            rig=rig, pipeline=self._parse(sc.PipelineIdentity.from_dict, row["pipeline_json"]),
            created_utc=row["created_utc"], last_confirmed_utc=row["last_confirmed_utc"],
            inactive=self._why_inactive(rig, current))

    def _why_inactive(self, rig: sc.RigIdentity, current: sc.RigIdentity) -> str | None:
        same_physical = all(getattr(rig, f) == getattr(current, f) for f in PHYSICAL_RIG_FIELDS)
        if same_physical and rig.rig_epoch < current.rig_epoch:
            return "inactive: rig moved"
        for name in sc.MATCH_FIELDS:
            if getattr(rig, name) != getattr(current, name):
                return f"inactive: rig mismatch ({name})"
        return None

    def _all_entries(self) -> list[MemoryEntry]:
        current = self.rig
        rows = self._query("SELECT e.*, r.rig_json FROM entries e JOIN rigs r USING (rig_id) ORDER BY e.entry_id")
        return [self._entry(row, current) for row in rows]

    def entries(self) -> list[MemoryEntry]:
        """The live entries (candidate and confirmed) of the current rig."""
        return [e for e in self._all_entries() if e.inactive is None and e.state in LIVE]

    def inactive(self) -> list[MemoryEntry]:
        """Every entry of another rig or an older epoch, each with why it is not served."""
        return [e for e in self._all_entries() if e.inactive is not None]

    def history(self) -> list[MemoryEntry]:
        """Every entry, superseded and inactive ones included. For audit."""
        return self._all_entries()

    def board_pose(self) -> MemoryEntry | None:
        """The current rig's confirmed board entry, or None. A prior: never a motion input."""
        confirmed = [e for e in self.entries() if e.state == "confirmed"]
        return confirmed[0] if confirmed else None

    def sightings(self) -> list[Sighting]:
        """The objects of the newest scene stored for the current rig. Episodic, and never merged."""
        scenes = self._query("SELECT s.scene_id, s.created_utc FROM scenes s JOIN rigs r USING (rig_id) "
                             "WHERE r.match_key=? ORDER BY s.rowid", (_match_key(self.rig),))
        if not scenes:
            return []
        newest = max(reversed(scenes), key=lambda row: _when(row["created_utc"]))  # a tie: the later stored
        rows = self._query("SELECT observation_id, payload_json FROM observations WHERE scene_id=? AND "
                           "kind='object_sighting' ORDER BY observation_id", (newest["scene_id"],))
        return [Sighting(row["observation_id"], newest["scene_id"], newest["created_utc"],
                         self._parse(sc.PerceivedObject.from_dict, row["payload_json"])) for row in rows]

    def observations(self, kind: str | None = None) -> list[dict]:
        """The immutable observation rows, oldest first, with their payloads. For audit."""
        if kind is not None and kind not in OBSERVATION_KINDS:
            raise SceneMemoryError(f"unknown observation kind {kind!r}")
        rows = self._query("SELECT * FROM observations ORDER BY observation_id")
        for row in rows:
            row["payload"] = self._parse(dict, row.pop("payload_json"))
        return [row for row in rows if kind is None or row["kind"] == kind]

    def confirmations(self, entry_id: int) -> list[dict]:
        rows = self._query("SELECT * FROM confirmations WHERE entry_id=? ORDER BY confirmation_id", (entry_id,))
        for row in rows:
            row["residual_m"] = self._parse(tuple, row.pop("residual_json"))
        return rows

    def scene_document(self, scene_id: str) -> sc.PerceivedScene:
        """The stored scene, only if its file is still there and still has the recorded SHA-256."""
        rows = self._query("SELECT document_path, document_sha256 FROM scenes WHERE scene_id=?", (scene_id,))
        if not rows:
            raise SceneMemoryError(f"no scene {scene_id}")
        path = self.path.parent / rows[0]["document_path"]
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise SceneMemoryError(f"scene {scene_id}: its document cannot be read: {exc}") from exc
        if hashlib.sha256(raw).hexdigest() != rows[0]["document_sha256"]:
            raise SceneMemoryError(f"scene {scene_id}: its document no longer matches its SHA-256")
        try:
            return sc.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, sc.SceneContractError) as exc:
            raise SceneMemoryError(f"scene {scene_id}: {exc}") from exc

    # ------------------------------------------------------------ writing

    def invalidate(self, entry_id: int, reason: str) -> None:
        """Take an entry out of use, with the operator's reason. The row stays."""
        if not isinstance(reason, str) or not reason.strip():
            raise SceneMemoryError("invalidate() needs a reason")
        with self._write():
            rows = self._query("SELECT state, rig_id FROM entries WHERE entry_id=?", (entry_id,))
            if not rows or rows[0]["state"] not in LIVE:
                raise SceneMemoryError(f"no live entry {entry_id}")
            self._update(entry_id, state="invalidated", invalid_reason=reason.strip(), challenges=None, span_count=0)
            self._choose_primary(rows[0]["rig_id"])

    def add_scene(self, scene: sc.PerceivedScene, twin_board: sc.BoardPose | None = None) -> SceneRecord:
        """Store a scene of the current rig: its document, its observations, the rig check and the board count.
        All of it happens, or none of it. `twin_board` (source "twin") is the twin's grid, also checked."""
        if not isinstance(scene, sc.PerceivedScene):
            raise SceneMemoryError("add_scene() takes a validated PerceivedScene")
        _check_rig(scene.rig)
        if twin_board is not None and (not isinstance(twin_board, sc.BoardPose) or twin_board.source != "twin"):
            raise SceneMemoryError('twin_board must be a BoardPose with source "twin"')
        raw = sc.dumps(scene).encode("utf-8")
        relative = f"{self.documents.name}/{scene.scene_id}.json"
        written = None
        try:
            with self._write():
                current = self.rig
                if scene.rig.rig_epoch != current.rig_epoch:
                    raise SceneMemoryError(
                        f"the scene was built at rig_epoch {scene.rig.rig_epoch}, the rig is now at "
                        f"{current.rig_epoch}: build it with store.rig")
                why = self._why_inactive(scene.rig, current)
                if why is not None:
                    raise SceneMemoryError(f"the scene is of another rig: {why}")
                if self._query("SELECT 1 FROM scenes WHERE scene_id=?", (scene.scene_id,)):
                    raise SceneMemoryError(f"scene {scene.scene_id} is already stored")
                self.documents.mkdir(parents=True, exist_ok=True)
                written = self._write_document(scene.scene_id, raw)
                rig_id = self._rig_row(scene.rig)

                duplicate = None
                if scene.board is not None and scene.board.source == "head_sweep_fit":
                    duplicate = self._same_fit(scene)
                check = None
                if scene.board is not None and scene.board.source == "head_sweep_fit" and duplicate is None:
                    check = self._rig_check(scene.board, twin_board, rig_id, current.rig_epoch)
                    if check.verdict == "fail":
                        rig_id = self._rig_row(replace(scene.rig, rig_epoch=check.epoch_after))

                self._insert("scenes", {
                    "scene_id": scene.scene_id, "rig_id": rig_id,
                    "pipeline_json": _canonical(scene.pipeline.to_dict()),
                    "document_sha256": hashlib.sha256(raw).hexdigest(), "document_path": relative,
                    "created_utc": scene.created_utc, "recorded_utc": _now()})
                if check is not None:
                    self._observe(scene, rig_id, "rig_check", {
                        "verdict": check.verdict, "distance_m": check.distance_m,
                        "twin_distance_m": check.twin_distance_m, "yaw_deg": check.yaw_deg,
                        "twin_yaw_deg": check.twin_yaw_deg, "epoch_before": check.epoch_before,
                        "epoch_after": check.epoch_after, "threshold_m": RIG_MOVED_THRESHOLD_M,
                        "yaw_threshold_deg": RIG_TURNED_THRESHOLD_DEG,
                        "message": check.message})
                board = self._board(scene, rig_id, duplicate)
                for seen in scene.objects:
                    self._observe(scene, rig_id, "object_sighting", seen.to_dict())
        except BaseException as exc:
            if written is not None:
                written.unlink(missing_ok=True)
            if isinstance(exc, (sqlite3.Error, OSError)):
                raise SceneMemoryError(f"{self.path}: {exc}") from exc
            raise
        if check is not None and check.verdict == "fail":
            LOG.warning("%s", check.message)
        return SceneRecord(scene.scene_id, self.rig, check, board, len(scene.objects))

    def _write_document(self, scene_id: str, raw: bytes) -> Path:
        path = self.documents / f"{scene_id}.json"
        draft = self.documents / f".{scene_id}.{uuid.uuid4().hex}.new"
        try:
            draft.write_bytes(raw)
            os.replace(draft, path)
        finally:
            draft.unlink(missing_ok=True)
        return path

    def _observe(self, scene: sc.PerceivedScene, rig_id: int, kind: str, payload: dict,
                 fingerprint: str | None = None) -> int:
        return self._insert("observations", {
            "scene_id": scene.scene_id, "rig_id": rig_id, "kind": kind, "payload_json": _canonical(payload),
            "fingerprint": fingerprint, "recorded_utc": _now()})

    def _same_fit(self, scene: sc.PerceivedScene) -> int | None:
        """The observation that already holds this very board fit, in this rig at any epoch."""
        rows = self._query(
            "SELECT o.observation_id FROM observations o JOIN rigs r USING (rig_id) WHERE o.kind='board_pose' AND "
            "o.fingerprint=? AND r.scope_key=? LIMIT 1", (_fingerprint(scene.board), _scope_key(scene.rig)))
        return rows[0]["observation_id"] if rows else None

    def _rig_check(self, board: sc.BoardPose, twin: sc.BoardPose | None, rig_id: int, epoch: int) -> RigCheck:
        rows = self._query("SELECT * FROM entries WHERE rig_id=? AND state='confirmed'", (rig_id,))
        distance = yaw = twin_distance = twin_yaw = None
        if rows:
            position, quat, _ = self._numbers(rows[0])
            distance, yaw = _distance(board.position_m, position), _turn_deg(board.quat_wxyz, quat)
        if twin is not None:
            twin_distance = _distance(board.position_m, twin.position_m)
            twin_yaw = _turn_deg(board.quat_wxyz, twin.quat_wxyz)
        measured = [(name, moved, turned) for name, moved, turned in (
            ("the remembered board", distance, yaw), ("the twin's grid", twin_distance, twin_yaw)) if moved is not None]
        if not measured:
            return RigCheck("no_reference", None, None, epoch, epoch,
                            "Rig not checked: no confirmed board pose is remembered and no twin grid was given.")
        text = "; ".join(f"{moved * 1000:.1f} mm and {turned:.1f} deg from {name}" for name, moved, turned in measured)
        limit = (f"limits {RIG_MOVED_THRESHOLD_M * 1000:g} mm and {RIG_TURNED_THRESHOLD_DEG:g} deg of yaw, "
                 "the angle provisional")
        found = {"distance_m": distance, "twin_distance_m": twin_distance, "yaw_deg": yaw, "twin_yaw_deg": twin_yaw}
        if all(moved <= RIG_MOVED_THRESHOLD_M + 1e-12 and turned <= RIG_TURNED_THRESHOLD_DEG + 1e-9
               for _, moved, turned in measured):
            return RigCheck("pass", epoch_before=epoch, epoch_after=epoch,
                            message=f"Rig check passed: the board is {text} ({limit}).", **found)
        return RigCheck("fail", epoch_before=epoch, epoch_after=epoch + 1,
                        message=f"RIG CHECK FAILED: the board is {text} ({limit}). The rig has moved or does not "
                                f"match: rig_epoch is now {epoch + 1}, and what was remembered before is inactive.",
                        **found)

    def _board(self, scene: sc.PerceivedScene, rig_id: int, duplicate: int | None) -> BoardUpdate:
        board = scene.board
        if board is None:
            return BoardUpdate("no_board", None, "The scene has no board pose.")
        observation = self._observe(scene, rig_id, "board_pose", board.to_dict(), _fingerprint(board))
        if board.source != "head_sweep_fit":
            return BoardUpdate("not_counted", None,
                               f"A {board.source} board pose is kept as an observation; only a head sweep counts.")
        if duplicate is not None:
            return BoardUpdate("not_counted", None,
                               f"The same board fit as observation {duplicate}: one measurement counts once.")
        live = self._query("SELECT * FROM entries WHERE rig_id=? AND state IN ('candidate', 'confirmed') "
                           "ORDER BY challenges IS NOT NULL, entry_id", (rig_id,))  # the primary first
        for row in live:
            position, quat, bounds = self._numbers(row)
            residual = [new - kept for new, kept in zip(board.position_m, position)]
            if all(abs(d) <= kept + new + 1e-12 for d, kept, new in zip(residual, bounds, board.bounds_m)):
                self._confirmation(row["entry_id"], observation, residual, "agree")
                return self._agree(row, scene, position, quat, bounds, rig_id)
        for row in live:  # no entry is edited by a sighting that disagrees; the audit row says so
            residual = [new - kept for new, kept in zip(board.position_m, self._numbers(row)[0])]
            self._confirmation(row["entry_id"], observation, residual, "disagree")
        primary = next((row["entry_id"] for row in live if row["challenges"] is None), None)
        entry = self._insert("entries", {
            "rig_id": rig_id, "kind": "board_pose",
            "estimate_json": _canonical({"position_m": list(board.position_m), "quat_wxyz": list(board.quat_wxyz)}),
            "bounds_json": _canonical(list(board.bounds_m)), "state": "candidate", "effective_count": 1,
            "total_count": 1, "span_count": 0, "pipeline_json": _canonical(scene.pipeline.to_dict()),
            "challenges": primary, "superseded_by": None, "invalid_reason": None, "created_utc": scene.created_utc,
            "last_confirmed_utc": scene.created_utc})
        if primary is None:
            return BoardUpdate("new_candidate", entry, "A new candidate board pose.")
        return BoardUpdate("competing", entry, f"Disagrees with entry {primary}, which is unchanged: a competing "
                                               "candidate.")

    def _confirmation(self, entry_id: int, observation_id: int, residual: list, verdict: str) -> None:
        self._insert("confirmations", {"entry_id": entry_id, "observation_id": observation_id,
                                       "residual_json": _canonical(residual), "verdict": verdict,
                                       "recorded_utc": _now()})

    def _agree(self, row: Mapping[str, Any], scene: sc.PerceivedScene, kept_position: tuple, kept_quat: tuple,
               kept_bounds: tuple, rig_id: int) -> BoardUpdate:
        board, entry_id = scene.board, row["entry_id"]
        pipeline = _canonical(scene.pipeline.to_dict())
        restarted = pipeline != row["pipeline_json"]
        if restarted:  # OD-8: the old pipeline's numbers and count do not carry over
            count, span = 1, 0
            position, quat, bounds = list(board.position_m), list(board.quat_wxyz), list(board.bounds_m)
        else:
            weight = row["effective_count"]
            count, span = weight + 1, row["span_count"] + 1
            position = [(kept * weight + new) / count for kept, new in zip(kept_position, board.position_m)]
            quat = list(_mean_quat(kept_quat, board.quat_wxyz, weight))
            bounds = [max(kept, new) for kept, new in zip(kept_bounds, board.bounds_m)]  # never shrinks with n
        competitor = row["challenges"] is not None
        self._update(entry_id, estimate_json=_canonical({"position_m": position, "quat_wxyz": quat}),
                     bounds_json=_canonical(bounds), effective_count=count, total_count=row["total_count"] + 1,
                     span_count=span if competitor else 0, pipeline_json=pipeline,
                     last_confirmed_utc=scene.created_utc)
        note = f"Agrees with entry {entry_id} ({count} sighting(s) under this pipeline)."
        if restarted:
            note += " Another pipeline: the effective count restarted."
        if not competitor:  # the primary was confirmed, so every competitor's span starts again
            self._conn.execute("UPDATE entries SET span_count=0 WHERE challenges=?", (entry_id,))
            if row["state"] == "candidate" and count - 1 >= CONFIRMATIONS:
                self._update(entry_id, state="confirmed")
                note += " It is now confirmed."
            return BoardUpdate("agree", entry_id, note)
        if span < CONFIRMATIONS:
            return BoardUpdate("agree", entry_id, note)
        original = row["challenges"]
        self._update(original, state="superseded", superseded_by=entry_id, span_count=0)
        self._update(entry_id, state="confirmed", challenges=None, span_count=0)
        self._conn.execute("UPDATE entries SET challenges=?, span_count=0 WHERE rig_id=? AND challenges=? AND "
                           "state IN ('candidate', 'confirmed')", (entry_id, rig_id, original))
        return BoardUpdate("superseded", entry_id, f"Entry {entry_id} supersedes entry {original} after "
                                                   f"{CONFIRMATIONS} confirmations with none for it.")

    def _choose_primary(self, rig_id: int) -> None:
        """After an entry leaves use: if no primary is left, the oldest competitor becomes it."""
        live = self._query("SELECT entry_id, challenges FROM entries WHERE rig_id=? AND state IN "
                           "('candidate', 'confirmed') ORDER BY entry_id", (rig_id,))
        primaries = {row["entry_id"] for row in live if row["challenges"] is None}
        orphans = [row for row in live if row["challenges"] is not None and row["challenges"] not in primaries]
        if not orphans:
            return
        head = next(iter(primaries), orphans[0]["entry_id"])
        for row in orphans:
            self._update(row["entry_id"], challenges=None if row["entry_id"] == head else head, span_count=0)


def _fingerprint(board: sc.BoardPose) -> str:
    return hashlib.sha256(_canonical(board.to_dict()).encode()).hexdigest()
