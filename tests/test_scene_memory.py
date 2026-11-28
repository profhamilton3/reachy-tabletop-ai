"""#85: SceneMemoryStore (docs/design/perceived-scene.md section 4, exit gate P4). Offline: SQLite in tmp_path.

The PART### scenes here are SYNTHETIC documents with the printed parts' dimensions (reachy-1-2-sim#212), placed on
FWDCenterLabSivaPool's grid. No perception produced them. The last tests run the real builder on the committed sim
renders of that scene (before the PART parts).
"""
import dataclasses
import hashlib
import logging
import math
import sqlite3
import sys
import threading
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pytest

from reachy_ai.perception import scene_builder as builder
from reachy_ai.perception import scene_contract as sc
from reachy_ai.perception import scene_memory as m

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = "FWDCenterLabSivaPool"
GRID = (0.2258, 0.0, -0.4095)  # the sim grid centre in the torso frame (ADR-0010)
BOUND = 0.002
INF = float("inf")


def rig(**changes):
    values = dict(identity_version=1, source="sim", rig_layout_id=LAYOUT, mount_id="october-2026",
                  calibration_id="771392183ddc6c62", zoom_level="ZERO", intrinsics_sha256="771392183ddc6c62",
                  rig_epoch=0, scene_name=LAYOUT, scene_sha256="7" * 64, scene_revision="initial")
    values.update(changes)
    return sc.RigIdentity(**values)


def pipeline(code="abcdef1"):
    return sc.PipelineIdentity("6" * 64, ("cube", "cylinder", "empty"), "sgbm-bidir-1px/inner60-median", "none",
                               "none", code, False)


def board(dx=0.0, dy=0.0, dz=0.0, bound=BOUND, source="head_sweep_fit", nth=0, quat=(1.0, 0.0, 0.0, 0.0)):
    """A board pose `dx, dy, dz` metres from the grid. `nth` makes two otherwise equal fits two measurements."""
    poses, spread = (5, 1.0) if source == "head_sweep_fit" else (1, 0.0)
    return sc.BoardPose((GRID[0] + dx, GRID[1] + dy, GRID[2] + dz), quat, (bound,) * 3, source, poses,
                        1.0 + nth * 1e-6, spread)


def turned(deg, axis="z"):
    """The quaternion of a board turned `deg` degrees about a torso axis (z: yaw, in place)."""
    half = math.radians(deg) / 2
    return (math.cos(half), *(math.sin(half) if name == axis else 0.0 for name in "xyz"))


def part(kind="cylinder", size=(0.049212, 0.049212, 0.06985), cell="r2c2", x=0.2258, bounds=(INF, INF, INF),
         calibration="771392183ddc6c62"):
    source = sc.SourcePair("pair-1", "a" * 64, "b" * 64, "c" * 64, "2026-11-27T12:00:00Z", (0.0, 64.0, 0.0))
    return sc.PerceivedObject(
        kind, 0.8, (x, 0.0, -0.375), (x - 0.02, 0.0, -0.37), bounds, 0.95, True, None, None, size, "measured", cell,
        500, 0.8, 0.01, source, calibration)


PART221 = part("cylinder", (0.049212, 0.049212, 0.06985), "r2c2", 0.2258)   # tall cylinder with a hole
PART110 = part("cube", (0.04921, 0.04921, 0.02540), "r1c1", 0.17)           # short block, solid


def scene(pose=None, on=None, objects=(), code="abcdef1", at="2026-11-27T12:00:00Z", scene_id=None):
    on = on or rig()
    cells = ()
    if pose is not None:
        held = {o.cell: (o.kind, i) for i, o in enumerate(objects) if o.cell}
        cells = tuple(sc.CellReport(n, *(held.get(n, ("unknown", None)))) for n in sc.CELL_NAMES)
    else:
        objects = tuple(dataclasses.replace(o, cell=None) for o in objects)
    objects = tuple(dataclasses.replace(o, calibration_id=on.calibration_id) for o in objects)
    return sc.PerceivedScene("perceived_scene.v1", scene_id or uuid.uuid4().hex, at, on, pipeline(code), pose, cells,
                             objects, (), ())


@pytest.fixture
def path(tmp_path):
    return tmp_path / "memory" / "scene_memory.sqlite"


@pytest.fixture
def memory(path):
    with m.SceneMemoryStore.open(path, rig(), create=True) as opened:
        yield opened


def add(memory, pose, **kwargs):
    """Add a scene built, as a caller must, at the store's current epoch."""
    twin = kwargs.pop("twin", None)
    on = dataclasses.replace(kwargs.pop("on", rig()), rig_epoch=memory.rig.rig_epoch)
    return memory.add_scene(scene(pose, on=on, **kwargs), twin_board=twin)


def confirm(memory, **kwargs):
    """A confirmed entry at the grid: the first sighting and CONFIRMATIONS more, each its own measurement."""
    records = [add(memory, board(nth=i), **kwargs) for i in range(m.CONFIRMATIONS + 1)]
    return records[0].board.entry_id


def raw(memory, entry_id):
    (row,) = memory._query("SELECT * FROM entries WHERE entry_id=?", (entry_id,))
    return row


def files(memory):
    return sorted(p.name for p in memory.documents.glob("*")) if memory.documents.exists() else []


# ---------------------------------------------------------------- exit gate: survives restart

def test_everything_survives_a_restart(path):
    with m.SceneMemoryStore.open(path, rig(), create=True) as first:
        entry_id = confirm(first, objects=(PART221, PART110))
        last = add(first, board(nth=99), objects=(PART221, PART110), at="2026-11-27T13:00:00Z")
        kept = (first.board_pose(), first.sightings(), first.entries(), first.rigs(), first.observations(),
                first.confirmations(entry_id), first.scene_document(last.scene_id))
    with m.SceneMemoryStore.open(path, rig()) as again:
        assert (again.board_pose(), again.sightings(), again.entries(), again.rigs(), again.observations(),
                again.confirmations(entry_id), again.scene_document(last.scene_id)) == kept
        assert again.board_pose().entry_id == entry_id and again.board_pose().effective_count == 5
        assert [s.object.kind for s in again.sightings()] == ["cylinder", "cube"]
        assert again.sightings()[0].object.bounds_m == (INF,) * 3  # an unbounded sighting comes back unbounded


def test_the_store_uses_wal_records_its_version_and_always_closes(path):
    with pytest.raises(KeyError):
        with m.SceneMemoryStore.open(path, rig(), create=True) as opened:
            assert opened._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert opened._conn.execute("PRAGMA user_version").fetchone()[0] == m.SCHEMA_VERSION
            raise KeyError("boom")
    with pytest.raises(m.SceneMemoryError):
        opened.entries()
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]  # no draft left behind


def test_documents_and_bundle_hashes_stay_in_files_not_blobs(memory):
    record = add(memory, board(), objects=(PART221,))
    (row,) = memory._query("SELECT document_path, document_sha256 FROM scenes")
    stored = memory.path.parent / row["document_path"]
    assert hashlib.sha256(stored.read_bytes()).hexdigest() == row["document_sha256"]
    assert memory.scene_document(record.scene_id).objects[0].source.bundle_sha256 == "a" * 64
    types = {col[2].upper() for t in ("rigs", "scenes", "observations", "entries", "confirmations")
             for col in memory._conn.execute(f"PRAGMA table_info({t})")}
    assert "BLOB" not in types


@pytest.mark.parametrize("damage", ["altered", "removed"])
def test_a_scene_document_that_changed_or_vanished_is_not_served(memory, damage):
    record = add(memory, board())
    target = memory.documents / f"{record.scene_id}.json"
    if damage == "altered":
        target.write_text(target.read_text().replace("abcdef1", "abcdef2"))
    else:
        target.unlink()
    with pytest.raises(m.SceneMemoryError, match="document"):
        memory.scene_document(record.scene_id)
    with pytest.raises(m.SceneMemoryError, match="no scene"):
        memory.scene_document("0" * 32)


def test_connections_creating_one_store_together_all_succeed(tmp_path):
    failures = []
    for attempt in range(30):
        target = tmp_path / str(attempt) / "memory.sqlite"
        start = threading.Barrier(4)

        def work():
            start.wait()
            try:
                with m.SceneMemoryStore.open(target, rig(), create=True) as opened:
                    opened.entries()
            except Exception as exc:  # noqa: BLE001 - reported by the assert below
                failures.append(exc)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert failures == []


# ---------------------------------------------------------------- fails closed

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refused(path, phrase, **kwargs):
    with pytest.raises(m.SceneMemoryError, match=phrase):
        with m.SceneMemoryStore.open(path, rig(), **kwargs):
            pass


def test_a_missing_store_is_an_error_not_an_empty_memory(path):
    refused(path, "no scene memory here")
    assert not path.exists()


@pytest.mark.parametrize("create", [False, True])
def test_an_empty_file_is_refused_even_when_creating(path, create):
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    refused(path, "empty or not a file", create=create)
    assert path.read_bytes() == b""


def test_a_file_that_is_not_a_database_is_refused_and_left_alone(path):
    path.parent.mkdir(parents=True)
    path.write_bytes(b"this is not a database" * 100)
    before = digest(path)
    refused(path, "scene_memory.sqlite", create=True)
    assert digest(path) == before


@pytest.mark.parametrize("statements, phrase", [
    (["CREATE TABLE trials (id TEXT)"], "schema version 0"),
    (["CREATE TABLE trials (id TEXT)", "PRAGMA user_version = 1"], "not a scene memory"),
    (["PRAGMA user_version = 3"], "schema version 3"),
    (["PRAGMA user_version = 2"], "not a scene memory"),
    (["PRAGMA user_version = 99"], "schema version 99"),
])
def test_another_database_or_version_is_refused_and_not_written(tmp_path, statements, phrase):
    other = tmp_path / "other.sqlite"
    conn = sqlite3.connect(other)
    for statement in statements:
        conn.execute(statement)
    conn.commit()
    conn.close()
    before = digest(other)
    refused(other, phrase, create=True)
    assert digest(other) == before


def test_a_store_that_lost_a_trigger_is_refused(path):
    with m.SceneMemoryStore.open(path, rig(), create=True):
        pass
    conn = sqlite3.connect(path)
    conn.execute("DROP TRIGGER observations_no_update")
    conn.commit()
    conn.close()
    refused(path, "observations_no_update")


@pytest.mark.parametrize("column, value", [
    ("estimate_json", "not json"), ("estimate_json", "{}"), ("estimate_json", "[1]"),
    ("bounds_json", "[0.1]"), ("bounds_json", "[NaN, 1, 1]"), ("pipeline_json", "{}")])
def test_a_record_that_cannot_be_read_raises_and_is_never_skipped(memory, column, value):
    entry_id = confirm(memory)
    memory._conn.execute(f"UPDATE entries SET {column}=? WHERE entry_id=?", (value, entry_id))
    for read in (memory.entries, memory.board_pose, memory.inactive, memory.history):
        with pytest.raises(m.SceneMemoryError, match="cannot be read"):
            read()
    if column != "pipeline_json":
        with pytest.raises(m.SceneMemoryError, match="cannot be read"):
            add(memory, board(nth=50))


def test_a_rig_that_cannot_be_built_never_opens_a_memory(path):
    with pytest.raises(sc.SceneContractError):
        rig(zoom_level="disagree")  # a bad zoom: there is no RigIdentity to open with
    for bad in (None, rig().to_dict(), "sim"):
        with pytest.raises(m.SceneMemoryError, match="RigIdentity"):
            with m.SceneMemoryStore.open(path, bad, create=True):
                pass
    assert not path.exists()


def test_robot_scene_memory_waits_for_the_robot_head_sweep(path, memory):
    robot = sc.RigIdentity(1, "robot", "lab", "october-2026", "p@zoom=OUT", "OUT", "a" * 64, 0)
    with pytest.raises(m.SceneMemoryError, match="sim only"):
        with m.SceneMemoryStore.open(path, robot):
            pass
    with pytest.raises(m.SceneMemoryError, match="sim only"):
        memory.add_scene(scene(None, on=robot))
    assert memory.history() == [] and memory.rigs() == [] and files(memory) == []


def test_only_validated_documents_are_stored(memory):
    with pytest.raises(m.SceneMemoryError, match="validated PerceivedScene"):
        memory.add_scene(scene().to_dict())
    for bad in (board(), GRID):
        with pytest.raises(m.SceneMemoryError, match="twin"):
            memory.add_scene(scene(board()), twin_board=bad)
    assert files(memory) == []


@pytest.mark.parametrize("table", ["scenes", "observations", "confirmations"])
def test_history_rows_are_immutable(memory, table):
    confirm(memory)
    column = {"scenes": "created_utc", "observations": "kind", "confirmations": "verdict"}[table]
    before = memory._query(f"SELECT * FROM {table}")
    assert before
    for statement in (f"UPDATE {table} SET {column}={column}", f"DELETE FROM {table}",
                      f"INSERT OR REPLACE INTO {table} SELECT * FROM {table}"):
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            memory._conn.execute(statement)
    assert memory._query(f"SELECT * FROM {table}") == before


@pytest.mark.parametrize("table", ["rigs", "entries"])
def test_rigs_and_entries_are_never_deleted(memory, table):
    confirm(memory)
    with pytest.raises(sqlite3.DatabaseError, match="never deleted"):
        memory._conn.execute(f"DELETE FROM {table}")


def test_a_scene_is_stored_whole_or_not_at_all(memory, monkeypatch):
    entry_id = confirm(memory)
    before = (raw(memory, entry_id), memory.observations(), files(memory))
    observe = memory._observe

    def fail_on_the_object(sc_, rig_id, kind, payload, fingerprint=None):
        if kind == "object_sighting":
            raise OSError("disk gone")
        return observe(sc_, rig_id, kind, payload, fingerprint)

    monkeypatch.setattr(memory, "_observe", fail_on_the_object)
    for pose in (board(nth=20), board(dx=0.05)):  # one that would count, and one that would move the epoch
        with pytest.raises(m.SceneMemoryError, match="disk gone"):
            add(memory, pose, objects=(PART221,))
    assert (raw(memory, entry_id), memory.observations(), files(memory)) == before
    assert memory.rig.rig_epoch == 0 and not memory._conn.in_transaction


def test_a_scene_id_is_stored_once_and_the_first_document_stays(memory):
    first = scene(board(), objects=(PART221,))
    memory.add_scene(first)
    with pytest.raises(m.SceneMemoryError, match="already stored"):
        memory.add_scene(scene(board(nth=1), scene_id=first.scene_id))
    assert memory.scene_document(first.scene_id) == first


# ---------------------------------------------------------------- exit gate: rig mismatch -> inactive

OTHER_RIG = {"rig_layout_id": "FWDCenterLabMCC", "mount_id": "august-2026", "calibration_id": "other",
             "zoom_level": "OUT", "intrinsics_sha256": "other", "scene_sha256": "8" * 64,
             "scene_revision": "after-reset"}


def test_every_match_field_is_covered():
    # source cannot differ here (robot rigs are refused) and rig_epoch is the store's own
    assert set(OTHER_RIG) == set(sc.MATCH_FIELDS) - {"source", "rig_epoch"}


@pytest.mark.parametrize("field", OTHER_RIG)
def test_a_rig_that_differs_in_one_match_field_is_served_nothing(path, field):
    with m.SceneMemoryStore.open(path, rig(), create=True) as opened:
        entry_id = confirm(opened, objects=(PART221,))
    with m.SceneMemoryStore.open(path, rig(**{field: OTHER_RIG[field]})) as other:
        assert (other.entries(), other.board_pose(), other.sightings()) == ([], None, [])
        assert [(e.entry_id, e.inactive) for e in other.inactive()] == [
            (entry_id, f"inactive: rig mismatch ({field})")]
        with pytest.raises(m.SceneMemoryError, match=f"another rig: inactive: rig mismatch \\({field}\\)"):
            other.add_scene(scene(board(nth=9)))
    with m.SceneMemoryStore.open(path, rig()) as again:  # not deleted
        assert again.board_pose().entry_id == entry_id and again.inactive() == [] and again.sightings()


def test_a_scene_name_alone_is_not_a_mismatch(path):
    with m.SceneMemoryStore.open(path, rig(), create=True) as opened:
        confirm(opened)
    with m.SceneMemoryStore.open(path, rig(scene_name="renamed")) as renamed:
        assert renamed.board_pose() is not None


def test_an_epoch_ahead_of_the_store_is_refused(path):
    with m.SceneMemoryStore.open(path, rig(), create=True):
        pass
    with pytest.raises(m.SceneMemoryError, match="ahead of this store"):
        with m.SceneMemoryStore.open(path, rig(rig_epoch=1)):
            pass


# ---------------------------------------------------------------- counting

def test_a_board_is_confirmed_after_k_confirmations_by_the_weighted_mean(memory):
    assert m.CONFIRMATIONS == 3
    states = []
    for i, dx in enumerate((0.0, 0.001, 0.002, 0.001)):
        record = add(memory, board(dx=dx, nth=i))
        (entry,) = memory.entries()
        states.append((record.board.verdict, entry.state, entry.effective_count))
    assert states == [("new_candidate", "candidate", 1), ("agree", "candidate", 2), ("agree", "candidate", 3),
                      ("agree", "confirmed", 4)]
    assert memory.board_pose().position_m == pytest.approx((GRID[0] + 0.001, GRID[1], GRID[2]))
    assert [c["verdict"] for c in memory.confirmations(entry.entry_id)] == ["agree"] * 3


def test_the_bound_never_shrinks_with_more_sightings(memory):
    for i, bound in enumerate((0.002, 0.003, 0.001) + (0.001,) * 10):
        add(memory, board(bound=bound, nth=i))
    assert memory.board_pose().bounds_m == (0.003,) * 3 and memory.board_pose().effective_count == 13


def test_one_head_sweep_does_not_confirm_itself(memory):
    one_fit = board()
    verdicts = [add(memory, one_fit, objects=(PART221,)).board.verdict for _ in range(10)]
    assert verdicts == ["new_candidate"] + ["not_counted"] * 9
    (entry,) = memory.entries()
    assert (entry.state, entry.effective_count, entry.total_count) == ("candidate", 1, 1)
    assert len(memory.observations("board_pose")) == 10  # all kept, one counted


@pytest.mark.parametrize("source", ["single_pair_fit", "twin", "taught"])
def test_only_a_head_sweep_counts_or_checks_the_rig(memory, source):
    entry_id = confirm(memory)
    before = raw(memory, entry_id)
    for dx in (0.0, 0.05):
        record = add(memory, board(dx=dx, source=source), twin=board(source="twin"))
        assert (record.board.verdict, record.rig_check, record.rig.rig_epoch) == ("not_counted", None, 0)
    assert raw(memory, entry_id) == before and len(memory.history()) == 1


def test_a_scene_without_a_board_keeps_its_sightings_only(memory):
    record = add(memory, None, objects=(PART221,))
    assert (record.board.verdict, record.sightings, record.rig_check) == ("no_board", 1, None)
    assert memory.history() == [] and len(memory.sightings()) == 1


@pytest.mark.parametrize("axis", ["dx", "dy", "dz"])
def test_agreement_is_judged_on_every_axis_and_the_limit_itself_agrees(memory, axis):
    add(memory, board())
    assert add(memory, board(**{axis: 2 * BOUND}, nth=1)).board.verdict == "agree"
    (entry,) = memory.entries()
    beyond = entry.position_m["dx dy dz".split().index(axis)] - GRID["dx dy dz".split().index(axis)] + 2 * BOUND + 1e-4
    assert add(memory, board(**{axis: beyond}, nth=2)).board.verdict == "competing"


def test_quaternions_average_on_one_side_of_the_sphere(memory):
    add(memory, board())
    add(memory, board(nth=1, quat=(-1.0, 0.0, 0.0, 0.0)))
    assert memory.entries()[0].quat_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))


# ---------------------------------------------------------------- exit gate: a disagreeing sighting never edits

def test_a_disagreeing_sighting_never_edits_an_entry(memory):
    entry_id = confirm(memory)
    before = raw(memory, entry_id)
    record = add(memory, board(dx=0.007))  # within the 10 mm rig check, outside the bounds
    assert (record.rig_check.verdict, record.board.verdict) == ("pass", "competing")
    assert raw(memory, entry_id) == before  # every column
    other = memory.entries()[1]
    assert (other.state, other.challenges, other.effective_count) == ("candidate", entry_id, 1)
    assert memory.board_pose().entry_id == entry_id
    assert memory.confirmations(entry_id)[-1]["verdict"] == "disagree"
    assert memory.confirmations(entry_id)[-1]["residual_m"] == pytest.approx((0.007, 0.0, 0.0))


def test_no_run_of_disagreements_ever_edits_the_original(memory):
    entry_id = confirm(memory)
    before = raw(memory, entry_id)
    for i, (dx, dy) in enumerate([(0.007, 0), (-0.007, 0), (0, 0.007), (0, -0.007), (0.005, 0.005)]):
        add(memory, board(dx=dx, dy=dy, nth=100 + i))
        assert raw(memory, entry_id) == before


def test_a_competitor_supersedes_only_after_k_confirmations_with_none_for_the_original(memory):
    original = confirm(memory)
    new = add(memory, board(dx=0.007, nth=10)).board.entry_id
    for i in (11, 12):
        assert add(memory, board(dx=0.007, nth=i)).board.entry_id == new
    assert add(memory, board(nth=13)).board.entry_id == original   # the original is seen again: the span restarts
    for i in (14, 15):
        assert add(memory, board(dx=0.007, nth=i)).board.verdict == "agree"
        assert memory.board_pose().entry_id == original
    last = add(memory, board(dx=0.007, nth=16))
    assert (last.board.verdict, last.board.entry_id) == ("superseded", new)
    by_id = {e.entry_id: e for e in memory.history()}
    assert (by_id[original].state, by_id[original].superseded_by) == ("superseded", new)
    assert (by_id[new].state, by_id[new].challenges) == ("confirmed", None)
    assert memory.board_pose().entry_id == new and [e.entry_id for e in memory.entries()] == [new]
    assert len(memory.history()) == 2  # both rows stay


def test_other_competitors_move_on_to_challenge_the_new_primary(memory):
    original = confirm(memory)
    first = add(memory, board(dx=0.007, nth=10)).board.entry_id
    bystander = add(memory, board(dy=0.007, nth=11)).board.entry_id
    add(memory, board(dy=0.007, nth=12))
    for i in range(13, 13 + m.CONFIRMATIONS):
        add(memory, board(dx=0.007, nth=i))
    by_id = {e.entry_id: e for e in memory.history()}
    assert by_id[original].state == "superseded" and by_id[bystander].challenges == first
    assert raw(memory, bystander)["span_count"] == 0


# ---------------------------------------------------------------- OD-8: another pipeline

def test_another_pipeline_restarts_the_effective_count_and_keeps_the_old_one(memory):
    entry_id = confirm(memory)
    record = add(memory, board(dx=0.001, bound=0.001, nth=30), code="fedcba9")
    entry = memory.board_pose()
    assert (entry.entry_id, entry.state) == (entry_id, "confirmed")
    assert (entry.effective_count, entry.total_count, entry.pipeline.code_sha) == (1, 5, "fedcba9")
    assert entry.position_m == pytest.approx((GRID[0] + 0.001, GRID[1], GRID[2])) and entry.bounds_m == (0.001,) * 3
    assert "restarted" in record.board.note


def test_a_competitors_span_restarts_with_its_pipeline(memory):
    original = confirm(memory)
    for i in range(10, 13):
        add(memory, board(dx=0.007, nth=i))
    add(memory, board(dx=0.007, nth=13), code="fedcba9")  # would have been the third confirmation
    assert memory.board_pose().entry_id == original


# ---------------------------------------------------------------- exit gate: rig-check failure -> epoch bump

def test_a_moved_board_bumps_the_epoch_and_deactivates_everything_before(memory, caplog):
    entry_id = confirm(memory, objects=(PART221,))
    with caplog.at_level(logging.WARNING):
        record = add(memory, board(dx=0.085, nth=40), objects=(PART110,))
    check = record.rig_check
    assert (check.verdict, check.epoch_before, check.epoch_after) == ("fail", 0, 1)
    assert check.distance_m == pytest.approx(0.085) and "RIG CHECK FAILED" in check.message
    assert "85.0 mm" in check.message and check.message in caplog.text  # the operator is told
    assert record.rig.rig_epoch == memory.rig.rig_epoch == 1
    assert [(e.entry_id, e.inactive) for e in memory.inactive()] == [(entry_id, "inactive: rig moved")]
    assert memory.board_pose() is None
    (fresh,) = memory.entries()
    assert (fresh.state, fresh.effective_count, fresh.rig.rig_epoch, fresh.challenges) == ("candidate", 1, 1, None)
    assert [s.object.kind for s in memory.sightings()] == ["cube"]  # the old epoch's sighting is not served
    assert memory.observations("rig_check")[-1]["payload"]["verdict"] == "fail"


def test_the_new_epoch_survives_a_restart_and_a_caller_that_still_says_zero(path):
    with m.SceneMemoryStore.open(path, rig(), create=True) as opened:
        old = confirm(opened)
        add(opened, board(dx=0.085, nth=40))
    with m.SceneMemoryStore.open(path, rig(rig_epoch=0)) as again:
        assert again.rig.rig_epoch == 1 and again.board_pose() is None
        assert [(e.entry_id, e.inactive) for e in again.inactive()] == [(old, "inactive: rig moved")]
        with pytest.raises(m.SceneMemoryError, match="built at rig_epoch 0"):
            again.add_scene(scene(board(nth=41), on=rig(rig_epoch=0)))
        add(again, board(dx=0.085, nth=42))
    with m.SceneMemoryStore.open(path, rig(rig_epoch=1)) as current:
        assert current.entries()[0].effective_count == 2


def test_a_moved_rig_deactivates_every_calibration_of_it_and_no_other_rig(path):
    zoomed, elsewhere = rig(zoom_level="OUT"), rig(mount_id="bench")
    for other in (zoomed, elsewhere):
        with m.SceneMemoryStore.open(path, other, create=True) as opened:
            confirm(opened, on=other)
    with m.SceneMemoryStore.open(path, rig()) as opened:
        confirm(opened)
        add(opened, board(dx=0.085, nth=40))
    with m.SceneMemoryStore.open(path, zoomed) as opened:
        assert opened.rig.rig_epoch == 1 and opened.board_pose() is None
        assert {e.inactive for e in opened.inactive() if e.rig.zoom_level == "OUT"} == {"inactive: rig moved"}
    with m.SceneMemoryStore.open(path, elsewhere) as opened:
        assert opened.rig.rig_epoch == 0 and opened.board_pose() is not None


@pytest.mark.parametrize("offset, verdict", [
    ({"dx": 0.0099}, "pass"), ({"dx": 0.010}, "pass"), ({"dx": 0.0101}, "fail"),
    ({"dx": 0.0075, "dy": 0.0075}, "fail"),                      # 10.6 mm on the centre, each axis under 10
])
def test_the_rig_check_is_ten_millimetres_on_the_board_centre(memory, offset, verdict):
    assert m.RIG_MOVED_THRESHOLD_M == 0.010
    confirm(memory)
    record = add(memory, board(**offset, nth=60))
    assert record.rig_check.verdict == verdict and record.rig.rig_epoch == (1 if verdict == "fail" else 0)


@pytest.mark.parametrize("dx, verdict", [(0.003, "pass"), (0.030, "fail")])
def test_in_the_sim_the_twins_grid_is_a_reference_too(memory, dx, verdict):
    record = add(memory, board(dx=dx), twin=board(source="twin"))
    assert (record.rig_check.verdict, record.rig_check.distance_m) == (verdict, None)
    assert record.rig_check.twin_distance_m == pytest.approx(dx)


def test_either_reference_can_fail_the_check(memory):
    confirm(memory)
    far_twin = dataclasses.replace(board(source="twin"), position_m=(GRID[0] + 0.05, GRID[1], GRID[2]))
    record = add(memory, board(nth=70), twin=far_twin)
    assert record.rig_check.verdict == "fail" and record.rig_check.distance_m == pytest.approx(0.0)
    assert memory.rig.rig_epoch == 1


def test_with_no_reference_there_is_no_check_and_a_candidate_is_not_one(memory):
    assert add(memory, board()).rig_check.verdict == "no_reference"
    record = add(memory, board(dx=0.085, nth=1))  # only a candidate is remembered
    assert (record.rig_check.verdict, record.board.verdict, record.rig.rig_epoch) == ("no_reference", "competing", 0)


def test_the_same_fit_again_does_not_move_the_epoch_again(memory):
    confirm(memory)
    moved = board(dx=0.085, nth=80)
    add(memory, moved, twin=board(source="twin"))
    again = add(memory, moved, twin=board(source="twin"))
    assert (again.rig_check, again.board.verdict, memory.rig.rig_epoch) == (None, "not_counted", 1)


# ---------------------------------------------------------------- the rig check: the board's yaw

def test_a_board_turned_in_place_by_three_degrees_bumps_the_epoch_like_a_move(memory, caplog):
    entry_id = confirm(memory, objects=(PART221,))
    before = raw(memory, entry_id)
    with caplog.at_level(logging.WARNING):
        record = add(memory, board(nth=40, quat=turned(3.0)), objects=(PART110,))
    check = record.rig_check
    assert (check.verdict, check.epoch_before, check.epoch_after) == ("fail", 0, 1)
    assert check.distance_m == pytest.approx(0.0) and check.yaw_deg == pytest.approx(3.0)
    assert "RIG CHECK FAILED" in check.message and "3.0 deg" in check.message and check.message in caplog.text
    assert record.rig.rig_epoch == memory.rig.rig_epoch == 1
    assert [(e.entry_id, e.inactive) for e in memory.inactive()] == [(entry_id, "inactive: rig moved")]
    assert raw(memory, entry_id) == before and memory.board_pose() is None
    (fresh,) = memory.entries()
    assert (fresh.state, fresh.effective_count, fresh.rig.rig_epoch, fresh.challenges) == ("candidate", 1, 1, None)
    assert fresh.quat_wxyz == pytest.approx(turned(3.0))
    assert [s.object.kind for s in memory.sightings()] == ["cube"]
    payload = memory.observations("rig_check")[-1]["payload"]
    assert (payload["verdict"], payload["yaw_threshold_deg"]) == ("fail", 2.0)
    assert payload["yaw_deg"] == pytest.approx(3.0) and payload["twin_yaw_deg"] is None


def test_a_board_turned_in_place_by_one_degree_does_not_bump_the_epoch(memory):
    entry_id = confirm(memory)
    record = add(memory, board(nth=40, quat=turned(1.0)))
    assert (record.rig_check.verdict, record.rig.rig_epoch) == ("pass", 0)
    assert record.rig_check.yaw_deg == pytest.approx(1.0)
    assert memory.board_pose().entry_id == entry_id and memory.inactive() == []


@pytest.mark.parametrize("deg, verdict", [(2.0, "pass"), (2.01, "fail"), (-1.0, "pass"), (-3.0, "fail"),
                                          (180.0, "fail")])
def test_the_rig_check_is_two_degrees_of_yaw_either_way(memory, deg, verdict):
    assert m.RIG_TURNED_THRESHOLD_DEG == 2.0
    confirm(memory)
    record = add(memory, board(nth=60, quat=turned(deg)))
    assert record.rig_check.verdict == verdict and record.rig.rig_epoch == (1 if verdict == "fail" else 0)


@pytest.mark.parametrize("deg, verdict", [(1.0, "pass"), (3.0, "fail")])
def test_the_twins_grid_is_a_yaw_reference_too(memory, deg, verdict):
    record = add(memory, board(quat=turned(deg)), twin=board(source="twin"))
    assert (record.rig_check.verdict, record.rig_check.yaw_deg) == (verdict, None)
    assert record.rig_check.twin_yaw_deg == pytest.approx(deg)


def test_either_reference_can_fail_the_check_on_yaw(memory):
    confirm(memory)
    record = add(memory, board(nth=70), twin=board(source="twin", quat=turned(3.0)))
    assert record.rig_check.verdict == "fail" and record.rig_check.yaw_deg == pytest.approx(0.0)
    assert memory.rig.rig_epoch == 1


@pytest.mark.parametrize("new", [turned(-179.5), tuple(-v for v in turned(179.5))], ids=["across 180", "q and -q"])
def test_yaw_is_the_smallest_turn_between_the_two_boards(memory, new):
    for i in range(m.CONFIRMATIONS + 1):
        add(memory, board(nth=i, quat=turned(179.5)))
    record = add(memory, board(nth=90, quat=new))
    assert record.rig_check.verdict == "pass" and record.rig_check.yaw_deg <= 1.0 + 1e-6


@pytest.mark.parametrize("axis", ["x", "y"])
def test_only_yaw_is_judged_not_tilt(memory, axis):
    confirm(memory)
    record = add(memory, board(nth=95, quat=turned(3.0, axis)))
    assert record.rig_check.verdict == "pass" and record.rig_check.yaw_deg == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------- sightings are episodic

def test_sightings_are_the_newest_scenes_and_never_become_facts(memory):
    add(memory, board(), objects=(PART221, PART110), at="2026-11-27T12:00:00.5Z")
    add(memory, board(nth=1), objects=(part(x=0.31, cell="r3c3"),), at="2026-11-27T12:00:00Z")  # older, stored later
    assert [s.object.cell for s in memory.sightings()] == ["r2c2", "r1c1"]
    add(memory, board(nth=2), objects=(), at="2026-11-27T12:00:01Z")
    assert memory.sightings() == []  # a newer, empty board: nothing is carried over
    for i in range(3, 10):
        add(memory, board(nth=i), objects=(PART221,), at="2026-11-27T12:01:00Z")
    assert {e.kind for e in memory.history()} == {"board_pose"}
    assert len(memory.observations("object_sighting")) == 10


# ---------------------------------------------------------------- invalidate

def test_an_invalidated_entry_stays_and_a_competitor_takes_its_place(memory):
    original = confirm(memory)
    first = add(memory, board(dx=0.007, nth=10)).board.entry_id
    second = add(memory, board(dy=0.007, nth=11)).board.entry_id
    for reason in ("", "  ", None):
        with pytest.raises(m.SceneMemoryError, match="needs a reason"):
            memory.invalidate(original, reason)
    memory.invalidate(original, "the grid tape was replaced")
    by_id = {e.entry_id: e for e in memory.history()}
    assert (by_id[original].state, by_id[original].invalid_reason) == ("invalidated", "the grid tape was replaced")
    assert (by_id[first].challenges, by_id[second].challenges, by_id[first].state) == (None, first, "candidate")
    assert memory.board_pose() is None
    for gone in (original, 99):
        with pytest.raises(m.SceneMemoryError, match="no live entry"):
            memory.invalidate(gone, "again")


# ---------------------------------------------------------------- never a motion input

def test_the_memory_imports_no_motion_sdk_camera_or_network_module():
    text = (ROOT / "src" / "reachy_ai" / "perception" / "scene_memory.py").read_text()
    imports = [line for line in text.splitlines() if line.lstrip().startswith(("import ", "from "))]
    for banned in ("motion", "reachy_sdk", "reachy_client", "stereo", "anthropic", "grpc", "socket", "urllib",
                   "requests", "cv2"):
        assert not [line for line in imports if banned in line], banned


# ---------------------------------------------------------------- the real builder on the sim's renders

sys.path.insert(0, str(ROOT / "scripts"))
SWEEP = ROOT / "tests" / "fixtures" / "sim_head_sweep_v2"
NAMES = ("sweep_yaw_right_5", "sweep_yaw_right_8", "sweep_yaw_left_8", "sweep_roll_right_5", "sweep_roll_left_5")


def sweep_pose(tmp_path, names):
    from tests.test_scene_builder import sweep_zip

    target = tmp_path / ("%s.zip" % "-".join(n.split("sweep_")[1] for n in names))
    target.write_bytes(sweep_zip([("head_" + n, SWEEP / n) for n in names]))
    pose, _ = builder.board_from_head_sweep(target)
    assert pose is not None and pose.source == "head_sweep_fit"
    return pose


def test_builder_scenes_of_sivapool_are_checked_counted_and_remembered(tmp_path):
    """Real sim renders and the real builder. The subsets share images, so they are not independent measurements:
    this exercises the software path, not the confirmation evidence."""
    from tests.test_scene_builder import make_bundle

    poses = [sweep_pose(tmp_path, names) for names in
             (NAMES, NAMES[:4], NAMES[1:], (NAMES[0], NAMES[2], NAMES[3], NAMES[4]))]
    assert len({p.position_m for p in poses}) == 4
    twin = sc.BoardPose(GRID, (1.0, 0.0, 0.0, 0.0), (0.003,) * 3, "twin", 0, 0.0, 0.0)
    options = builder.RigOptions(rig_layout_id=LAYOUT, scene_name=LAYOUT, scene_sha256="7" * 64,
                                 scene_revision="initial")
    bundle = builder.load_pair_bundle(make_bundle(tmp_path))
    current = builder.build_rig(bundle, options)  # torso-frame sim capture: how a caller gets the rig
    path = tmp_path / "memory.sqlite"
    records = []
    for i, pose in enumerate(poses):
        with m.SceneMemoryStore.open(path, current, create=i == 0) as memory:  # a restart between every scene
            built = builder.build_scene(bundle, board=pose, code=("abcdef1", False),
                                        options=dataclasses.replace(options, rig_epoch=memory.rig.rig_epoch))
            records.append(memory.add_scene(built, twin_board=twin))
    assert [r.rig_check.verdict for r in records] == ["pass"] * 4, [r.rig_check.message for r in records]
    assert records[-1].rig.rig_epoch == 0
    with m.SceneMemoryStore.open(path, current) as memory:
        assert len(memory.sightings()) == len(memory.scene_document(records[-1].scene_id).objects) > 0
        counted = sum(e.total_count for e in memory.entries())
        assert counted == 4, [r.board.note for r in records]
        for entry in memory.entries():
            assert np.linalg.norm(np.array(entry.position_m) - GRID) < 0.010


# ---------------------------------------------------------------- #96: one source counts once

SHA_A, SHA_B = "a" * 64, "b" * 64
POSE_NAMES = ("board_view", "sweep_yaw_right_5", "sweep_yaw_left_5", "sweep_roll_right_5", "sweep_roll_left_5")


def keys(sha, names=POSE_NAMES):
    return tuple((sha, name) for name in names)


def add_sourced(memory, pose, sources, **kwargs):
    on = dataclasses.replace(rig(), rig_epoch=memory.rig.rig_epoch)
    return memory.add_scene(scene(pose, on=on, **kwargs), board_sources=sources)


def counted(memory):
    return sum(e.total_count for e in memory.entries())


def test_a_refit_of_the_same_bundle_counts_once_whatever_its_numbers(memory):
    add_sourced(memory, board(), keys(SHA_A))
    refit = add_sourced(memory, board(dx=0.0004, nth=7), keys(SHA_A), code="1234567")  # changed fitting code
    assert refit.board.verdict == "not_counted" and "same source" in refit.board.note
    assert (counted(memory), refit.rig_check) == (1, None)
    other = add_sourced(memory, board(dx=0.0004, nth=7), keys(SHA_B))  # another bundle with those very numbers
    assert other.board.verdict == "agree" and counted(memory) == 2


def test_sharing_any_pose_of_a_bundle_is_the_same_measurement(memory):
    add_sourced(memory, board(), keys(SHA_A))
    fewer = sc.BoardPose(GRID, (1.0, 0.0, 0.0, 0.0), (BOUND,) * 3, "head_sweep_fit", 2, 1.1, 0.5)
    assert add_sourced(memory, fewer, keys(SHA_A, POSE_NAMES[3:])).board.verdict == "not_counted"
    assert counted(memory) == 1


def test_the_fit_values_are_compared_only_where_a_source_is_missing(memory):
    add(memory, board())                                            # no source
    assert add(memory, board()).board.verdict == "not_counted"      # the same values, no source
    assert add_sourced(memory, board(), keys(SHA_A)).board.verdict == "not_counted"   # the earlier one unsourced
    assert add_sourced(memory, board(nth=1), keys(SHA_B)).board.verdict == "agree"    # new values, new source
    assert add(memory, board(nth=1)).board.verdict == "not_counted"  # an unsourced repeat of a sourced fit
    # an unsourced observation with these very numbers is on record, so a new source cannot claim them either
    assert add_sourced(memory, board(nth=1), keys("c" * 64)).board.verdict == "not_counted"
    assert counted(memory) == 2


def test_a_source_survives_a_restart_and_cannot_be_edited(path):
    with m.SceneMemoryStore.open(path, rig(), create=True) as first:
        add_sourced(first, board(), keys(SHA_A))
    with m.SceneMemoryStore.open(path, rig()) as again:
        assert add_sourced(again, board(nth=3), keys(SHA_A)).board.verdict == "not_counted"
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            again._conn.execute("DELETE FROM fit_sources")


@pytest.mark.parametrize("sources, why", [
    (keys("A" * 64), "SHA-256"), ((("a" * 64, " "),) * 5, "name"), (keys(SHA_A, POSE_NAMES[:4]), "each of"),
    ((("a" * 64, "board_view"),) * 5, "each of")])
def test_bad_sources_are_refused_and_nothing_is_stored(memory, sources, why):
    with pytest.raises(m.SceneMemoryError, match=why):
        add_sourced(memory, board(), sources)
    assert memory.observations() == [] and files(memory) == []


def test_only_a_head_sweep_fit_has_sources(memory):
    with pytest.raises(m.SceneMemoryError, match="head_sweep_fit"):
        add_sourced(memory, board(source="single_pair_fit"), keys(SHA_A, POSE_NAMES[:1]))


def test_a_version_1_store_is_upgraded_and_keeps_what_it_holds(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)  # a store as #95 made it: the version 1 tables alone
    for statement in m._SCHEMA:
        conn.execute(statement)
    conn.execute("PRAGMA user_version = 1")
    conn.close()
    with m.SceneMemoryStore.open(path, rig()) as upgraded:
        assert upgraded._conn.execute("PRAGMA user_version").fetchone()[0] == 2
        add_sourced(upgraded, board(), keys(SHA_A))
    with m.SceneMemoryStore.open(path, rig()) as again:
        assert len(again.entries()) == 1
        assert again._query("SELECT COUNT(*) AS n FROM fit_sources")[0]["n"] == 5


# ---------------------------------------------------------------- #96: the builder measures yaw on sim renders

TURNED = ROOT / "tests" / "fixtures" / "sim_board_yaw"
TURNED_POSES = ("board_view", "sweep_yaw_right_5", "sweep_yaw_left_5")
#: The full sweep and its three pairs: four bundles. They share images, so they are not independent evidence of
#: anything: they exercise the counting.
SUBSETS = (TURNED_POSES, TURNED_POSES[:2], TURNED_POSES[1:], (TURNED_POSES[0], TURNED_POSES[2]))


def turned_fit(tmp_path, yaw, poses):
    from tests.test_scene_builder import sweep_zip

    target = tmp_path / ("yaw%s-%s.zip" % (yaw, "-".join(p.replace("sweep_", "") for p in poses)))
    target.write_bytes(sweep_zip([("head_" + n, TURNED / ("yaw_%s" % yaw) / n) for n in poses]))
    return builder.head_sweep_fit(target), target


def heading(quat):
    return math.degrees(2.0 * math.atan2(quat[3], quat[0]))


def test_the_builder_measures_the_boards_yaw_with_a_bound(tmp_path):
    flat, _ = turned_fit(tmp_path, 0, TURNED_POSES)
    three, target = turned_fit(tmp_path, 3, TURNED_POSES)
    for fit, truth in ((flat, 0.0), (three, 3.0)):
        assert abs(fit.yaw_deg - truth) <= fit.yaw_bound_deg <= 1.5
        assert heading(fit.pose.quat_wxyz) == pytest.approx(fit.yaw_deg) and fit.pose.quat_wxyz[1:3] == (0.0, 0.0)
        assert "board yaw %+.2f deg, bound +-%.2f deg" % (fit.yaw_deg, fit.yaw_bound_deg) in fit.warnings
    assert three.yaw_deg - flat.yaw_deg == pytest.approx(3.0, abs=0.3)
    assert np.linalg.norm(np.array(three.pose.position_m) - flat.pose.position_m) < 0.003  # turned in place
    with zipfile.ZipFile(target) as archive:
        expected = hashlib.sha256(archive.read("manifest.json")).hexdigest()
    assert three.sources == tuple((expected, name) for name in TURNED_POSES)
    assert dataclasses.replace(three, manifest_sha256=None).sources == ()  # no manifest: the value fallback


def test_a_sim_board_turned_3_degrees_bumps_the_rig_epoch(tmp_path):
    from tests.test_scene_builder import make_bundle

    options = builder.RigOptions(rig_layout_id=LAYOUT, scene_name=LAYOUT, scene_sha256="7" * 64,
                                 scene_revision="initial")
    pair = builder.load_pair_bundle(make_bundle(tmp_path))
    current = builder.build_rig(pair, options)
    path = tmp_path / "memory.sqlite"

    def feed(memory, fit):
        built = builder.build_scene(pair, board=fit.pose, board_warnings=fit.warnings, code=("abcdef1", False),
                                    options=dataclasses.replace(options, rig_epoch=memory.rig.rig_epoch))
        return memory.add_scene(built, board_sources=fit.sources, board_yaw_bound_deg=fit.yaw_bound_deg)

    with m.SceneMemoryStore.open(path, current, create=True) as memory:
        records = [feed(memory, turned_fit(tmp_path, 0, poses)[0]) for poses in SUBSETS]
        assert memory.board_pose() is not None and memory.rig.rig_epoch == 0, [r.board.note for r in records]
        assert [r.rig_check.verdict for r in records] == ["no_reference"] * 4
        repeat = feed(memory, turned_fit(tmp_path, 0, TURNED_POSES)[0])
        assert repeat.board.verdict == "not_counted" and memory.board_pose().total_count == 4
        moved = feed(memory, turned_fit(tmp_path, 3, TURNED_POSES)[0])
        check = moved.rig_check
        assert (check.verdict, check.epoch_after, moved.rig.rig_epoch) == ("fail", 1, 1)
        assert check.distance_m < m.RIG_MOVED_THRESHOLD_M and check.yaw_deg == pytest.approx(3.0, abs=0.3)
        assert memory.board_pose() is None and all(e.inactive == "inactive: rig moved" for e in memory.inactive())
        payload = [o for o in memory.observations("rig_check") if o["payload"]["verdict"] == "fail"][0]["payload"]
        assert 0.3 <= payload["board_yaw_bound_deg"] <= 1.5
