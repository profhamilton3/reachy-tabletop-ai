"""#82: the DecisionStore. Offline: SQLite in a temporary directory."""
import csv
import dataclasses
import hashlib
import io
import sqlite3
import threading
from pathlib import Path

import pytest

from reachy_ai import config
from reachy_ai.reasoner import contract as c
from reachy_ai.reasoner import policy as p
from reachy_ai.reasoner import store as s

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "disposition" / "valid" / "request"
POLICY = p.load_policy()
COSTS = POLICY.spec.authority["scoring_costs"]


def request(case_id, split="development", count=0, design="PART221", request_id=None):
    base = c.loads_request((FIXTURES / ("spec_%s.json" % design.lower())).read_text())
    return dataclasses.replace(base, request_id=request_id or f"req-{case_id}-{count}", reobservation_count=count,
                               case=dataclasses.replace(base.case, case_id=case_id, split=split))


def decision(req, answer="accept", configuration="E", confidence=None, rules="2" * 64):
    """A hand-made decision. `answer` is "accept", "remeasure:refer", "refusal:spec_unknown" ..."""
    head, _, tail = answer.partition(":")
    disposition = refusal = route = None
    if head == "refusal":
        refusal = tail
    else:
        disposition, route = head, tail or None
    destination = {"accept": "left_tray", "reject": "right_tray", "rework": "rework_tray"}.get(disposition, "none")
    ai = configuration not in c.DETERMINISTIC_CONFIGURATIONS
    return c.Decision(
        request_id=req.request_id, configuration=configuration, disposition=disposition, refusal=refusal,
        remeasure_route=route, destination=destination, reasons=(c.Reason("R4", f"test: {answer}"),),
        confidence=confidence,
        policy=c.PolicyHashes("1" * 64, rules, "4" * 64 if ai else None, "3" * 64),
        model_id="model-test" if ai else None, code_sha="abcdef1", adjudication=None)


def adjudication(notes="re-measured with calipers", rating=4, **changes):
    values = {name: None for name in c.APPENDIX_A_FIELDS}
    values.update(object_id_and_lot="PART221-A, lot 1", actual_location_final_disposition="left tray, accepted",
                  human_adjudication_notes=notes, operator_rating=rating, adjudicator="adjudicator-test")
    values.update(changes)
    return c.Adjudication(**values)


@pytest.fixture
def path(tmp_path):
    return tmp_path / "decisions" / "decisions.sqlite"


@pytest.fixture
def store(path):
    with s.DecisionStore.open(path) as opened:
        yield opened


def add(store, case_id, answer="accept", reference=None, split="development", **kwargs):
    count = kwargs.pop("count", 0)
    req = request(case_id, split=split, count=count)
    decision_id = store.record(req, decision(req, answer, **kwargs))
    if reference is not None:
        head, _, tail = reference.partition(":")
        store.add_outcome(decision_id, "sim_truth",
                          **({"reference_refusal": tail} if head == "refusal" else {"reference_disposition": head}))
    return decision_id


# ---------------------------------------------------------------- location, restart, schema

def test_the_default_location_is_under_the_uncommitted_outputs_directory(tmp_path, monkeypatch):
    assert config.DECISION_STORE_PATH == "outputs/decisions/decisions.sqlite"
    assert "outputs/" in (ROOT / ".gitignore").read_text().splitlines()
    monkeypatch.chdir(tmp_path)
    with s.DecisionStore.open() as opened:
        assert opened.path == Path(config.DECISION_STORE_PATH)
    assert (tmp_path / "outputs" / "decisions" / "decisions.sqlite").is_file()


def test_everything_survives_a_restart(path):
    req = request("case-1", split="held_out")
    made = p.decide(req, POLICY, "E", "abcdef1")
    with s.DecisionStore.open(path) as first:
        decision_id = first.record(req, made)
        first.add_outcome(decision_id, "operator", reference_disposition="accept",
                          adjudication=adjudication(), twin=s.TwinResult(True, False, ["pad-part"]))
        first.freeze(POLICY.hashes(), "trials")
    with s.DecisionStore.open(path) as again:
        assert again.get(decision_id) == (req, made)
        (outcome,) = again.outcomes(decision_id)
        assert (outcome.source, outcome.reference, outcome.adjudication) == ("operator", "accept", adjudication())
        assert outcome.twin == s.TwinResult(True, False, ("pad-part",))
        assert again.frozen_policy().policy == POLICY.hashes()


def test_the_store_uses_wal_and_records_its_schema_version(store, path):
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == s.SCHEMA_VERSION


def test_open_always_closes(path):
    with pytest.raises(KeyError):
        with s.DecisionStore.open(path) as opened:
            raise KeyError("boom")
    with pytest.raises(sqlite3.ProgrammingError):
        opened.refusals()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("version", [0, 2, 99])
def test_a_schema_version_mismatch_refuses_and_leaves_the_file_alone(path, version):
    with s.DecisionStore.open(path):
        pass
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()
    conn.close()
    before = digest(path)
    with pytest.raises(s.DecisionStoreError, match=f"schema version {version}"):
        with s.DecisionStore.open(path):
            pass
    assert digest(path) == before


@pytest.mark.parametrize("statements, phrase", [
    (["CREATE TABLE trials (id TEXT)"], "schema version 0"),
    (["CREATE TABLE trials (id TEXT)", "PRAGMA user_version = 1"], "not a decision store"),
    (["PRAGMA user_version = 1"], "not a decision store"),
])
def test_another_database_is_refused_and_not_written(tmp_path, statements, phrase):
    other = tmp_path / "other.sqlite"
    conn = sqlite3.connect(other)
    for statement in statements:
        conn.execute(statement)
    conn.commit()
    conn.close()
    before = digest(other)
    with pytest.raises(s.DecisionStoreError, match=phrase):
        with s.DecisionStore.open(other):
            pass
    assert digest(other) == before


def test_a_file_that_is_not_a_database_is_refused(tmp_path):
    other = tmp_path / "notes.sqlite"
    other.write_bytes(b"this is not a database" * 100)
    before = digest(other)
    with pytest.raises(s.DecisionStoreError):
        with s.DecisionStore.open(other):
            pass
    assert digest(other) == before


def test_a_store_that_lost_an_append_only_trigger_is_refused(path):
    with s.DecisionStore.open(path):
        pass
    conn = sqlite3.connect(path)
    conn.execute("DROP TRIGGER decisions_no_update")
    conn.commit()
    conn.close()
    with pytest.raises(s.DecisionStoreError, match="decisions_no_update"):
        with s.DecisionStore.open(path):
            pass


# ---------------------------------------------------------------- append-only

@pytest.mark.parametrize("table", ["requests", "decisions", "outcomes", "policy_events"])
def test_the_database_refuses_update_delete_and_replace(store, table):
    add(store, "case-1", reference="accept")
    store.freeze(POLICY.hashes(), "trials")
    column = {"requests": "lot", "decisions": "code_sha", "outcomes": "source", "policy_events": "note"}[table]
    before = store._query(f"SELECT * FROM {table}")
    assert before
    for statement in (f"UPDATE {table} SET {column}={column}", f"DELETE FROM {table}",
                      f"INSERT OR REPLACE INTO {table} SELECT * FROM {table}"):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            store._conn.execute(statement)
    assert store._query(f"SELECT * FROM {table}") == before


def test_a_correction_is_a_new_outcome_and_the_original_stays(store):
    decision_id = add(store, "case-1", "accept", reference="accept")
    original_decision, (original,) = store.get(decision_id), store.outcomes(decision_id)
    store.add_outcome(decision_id, "sim_truth", reference_disposition="reject")
    assert store.outcomes(decision_id)[0] == original
    assert [o.reference for o in store.outcomes(decision_id)] == ["accept", "reject"]
    assert store.get(decision_id) == original_decision
    (row,) = store.confusion_table()
    assert row["counts"] == {"reject": {"accept": 1}}  # the latest reference grades


def test_an_outcome_that_states_no_reference_does_not_erase_one(store):
    decision_id = add(store, "case-1", reference="accept")
    store.add_outcome(decision_id, "sim_truth", twin=s.TwinResult(feasible=True))
    assert store.confusion_table()[0]["counts"] == {"accept": {"accept": 1}}


def test_disagreeing_sources_are_reported_and_can_be_graded_separately(store):
    decision_id = add(store, "case-1", reference="accept")
    store.add_outcome(decision_id, "operator", reference_disposition="reject", adjudication=adjudication())
    (row,) = store.confusion_table()
    assert row["reference_conflicts"] == 1 and row["counts"] == {"reject": {"accept": 1}}
    assert store.confusion_table(reference_source="sim_truth")[0]["counts"] == {"accept": {"accept": 1}}
    assert store.confusion_table(reference_source="operator")[0]["counts"] == {"reject": {"accept": 1}}
    with pytest.raises(s.DecisionStoreError, match="reference source"):
        store.confusion_table(reference_source="guess")


@pytest.mark.parametrize("given", [None, adjudication(notes=None)])
def test_an_operator_override_needs_an_adjudication_with_a_reason(store, given):
    decision_id = add(store, "case-1", "accept")
    with pytest.raises(s.DecisionStoreError, match="override"):
        store.add_outcome(decision_id, "operator", reference_disposition="reject", adjudication=given)
    assert store.outcomes(decision_id) == []


@pytest.mark.parametrize("field, blank", [
    ("adjudicator", "   \n"),  # a null adjudicator is already refused by the contract
    ("actual_location_final_disposition", None), ("actual_location_final_disposition", "   \n"),
    ("human_adjudication_notes", None), ("human_adjudication_notes", "   \n")])
def test_an_operator_answer_needs_the_required_adjudication_fields_even_when_it_agrees(store, field, blank):
    decision_id = add(store, "case-1", "accept")
    with pytest.raises(s.DecisionStoreError, match=field):
        store.add_outcome(decision_id, "operator", reference_disposition="accept",
                          adjudication=adjudication(**{field: blank}))
    assert store.outcomes(decision_id) == []


def test_the_other_appendix_a_fields_may_stay_empty(store):
    assert s.REQUIRED_ADJUDICATION_FIELDS == (
        "adjudicator", "actual_location_final_disposition", "human_adjudication_notes")
    least = adjudication(rating=None, **{name: None for name in c.APPENDIX_A_FIELDS
                                         if name not in s.REQUIRED_ADJUDICATION_FIELDS})
    decision_id = add(store, "case-1", "accept")
    store.add_outcome(decision_id, "operator", reference_disposition="accept", adjudication=least)
    assert [o.adjudication for o in store.outcomes(decision_id)] == [least]


def test_an_outcome_that_states_no_answer_needs_no_required_fields(store):
    decision_id = add(store, "case-1", "accept")
    rating_only = adjudication(notes=None, actual_location_final_disposition=None)
    store.add_outcome(decision_id, "operator", adjudication=rating_only)
    assert [(o.reference, o.adjudication) for o in store.outcomes(decision_id)] == [(None, rating_only)]


@pytest.mark.parametrize("answer", [{"reference_disposition": "accept"}, {"reference_refusal": "cell_not_ready"}])
def test_an_operator_answer_without_an_adjudication_record_is_refused(store, answer):
    key, value = next(iter(answer.items()))
    decided = value if key == "reference_disposition" else f"refusal:{value}"
    decision_id = add(store, "case-1", decided)                    # the answer agrees with the decision
    with pytest.raises(s.DecisionStoreError, match="needs an adjudication record"):
        store.add_outcome(decision_id, "operator", **answer)
    assert store.outcomes(decision_id) == []
    store.add_outcome(decision_id, "operator", adjudication=adjudication(), **answer)
    assert [o.reference for o in store.outcomes(decision_id)] == [decided]


def test_an_operator_may_record_a_twin_result_that_states_no_answer(store):
    decision_id = add(store, "case-1")
    store.add_outcome(decision_id, "operator", twin=s.TwinResult(placed=True))
    assert [o.reference for o in store.outcomes(decision_id)] == [None]


def robot_request(case_id="robot-1", count=0):
    base = c.loads_request((FIXTURES / "robot_perceived_with_colour.json").read_text())
    return dataclasses.replace(base, request_id=f"req-{case_id}-{count}", reobservation_count=count,
                               case=dataclasses.replace(base.case, case_id=case_id, split="development"))


def test_sim_truth_on_a_robot_case_is_refused(store):
    req = robot_request()
    decision_id = store.record(req, decision(req))
    with pytest.raises(s.DecisionStoreError, match="sim_truth answer is accepted only for a case whose source is sim"):
        store.add_outcome(decision_id, "sim_truth", reference_disposition="accept")
    assert store.outcomes(decision_id) == []
    store.add_outcome(decision_id, "operator", reference_disposition="accept", adjudication=adjudication())


@pytest.mark.parametrize("first, second", [("sim", "robot"), ("robot", "sim")])
def test_a_cases_source_is_fixed_by_its_first_request(path, first, second):
    make = {"sim": lambda count: request("case-1", count=count),
            "robot": lambda count: robot_request("case-1", count=count)}
    with s.DecisionStore.open(path) as opened:
        req = make[first](0)
        opened.record(req, decision(req))
    with s.DecisionStore.open(path) as opened:  # fixed on disk, not only in this process
        other = make[second](1)
        with pytest.raises(s.DecisionStoreError, match=f"case case-1 is a {first} case"):
            opened.record(other, decision(other))
        assert [r["source"] for r in opened._query("SELECT source FROM requests")] == [first]
        assert len(opened._query("SELECT * FROM decisions")) == 1
        again = make[first](1)  # the same source is still welcome
        opened.record(again, decision(again))


def test_a_sim_truth_answer_never_gains_a_robot_request(store):
    decision_id = add(store, "case-1", reference="accept")
    real = robot_request("case-1", count=1)
    with pytest.raises(s.DecisionStoreError, match="is a sim case"):
        store.record(real, decision(real))
    assert [o.source for o in store.outcomes(decision_id)] == ["sim_truth"]


def test_a_mixed_case_written_by_earlier_code_takes_no_sim_truth(store):
    sim = request("case-1")
    sim_id = store.record(sim, decision(sim, "remeasure:reobserve"))
    # record() now refuses this, so the row is written directly, as the earlier code left it
    (row,) = store._query("SELECT * FROM requests")
    store._insert("requests", {**row, "request_id": "old-robot-request", "source": "robot"})
    with pytest.raises(s.DecisionStoreError, match="source is sim"):
        store.add_outcome(sim_id, "sim_truth", reference_disposition="accept")
    assert store.outcomes(sim_id) == []


@pytest.mark.parametrize("kwargs, phrase", [
    ({}, "empty outcome"),
    ({"reference_disposition": "ship"}, "unknown reference disposition"),
    ({"reference_refusal": "tired"}, "unknown reference refusal"),
    ({"reference_disposition": "accept", "reference_refusal": "spec_unknown"}, "not both"),
    ({"adjudication": {"adjudicator": "x"}}, "contract Adjudication"),
    ({"twin": {"feasible": True}}, "TwinResult"),
])
def test_a_bad_outcome_is_refused(store, kwargs, phrase):
    decision_id = add(store, "case-1")
    with pytest.raises(s.DecisionStoreError, match=phrase):
        store.add_outcome(decision_id, "sim_truth", **kwargs)
    assert store.outcomes(decision_id) == []


def test_a_bad_source_an_unknown_decision_and_a_bad_twin_result_are_refused(store):
    with pytest.raises(s.DecisionStoreError, match="outcome source"):
        store.add_outcome(add(store, "case-1"), "guess", reference_disposition="accept")
    with pytest.raises(s.DecisionStoreError, match="no decision 99"):
        store.add_outcome(99, "sim_truth", reference_disposition="accept")
    with pytest.raises(s.DecisionStoreError, match="no decision 99"):
        store.get(99)
    for bad in ({"feasible": "yes"}, {"placed": 2}, {"contacts": "pad"}, {"contacts": [1]}):
        with pytest.raises(s.DecisionStoreError, match="TwinResult"):
            s.TwinResult(**bad)


# ---------------------------------------------------------------- recording

def test_several_configurations_may_share_one_request_and_all_are_kept(store):
    req = request("case-1")
    ids = [store.record(req, p.decide(req, POLICY, name, "abcdef1")) for name in "ABE"]
    assert len(set(ids)) == 3
    assert len(store._query("SELECT * FROM requests")) == 1
    assert [row["configuration"] for row in store.confusion_table()] == ["A", "B", "E"]


def test_a_request_id_cannot_be_reused_for_other_content(store):
    req = request("case-1")
    store.record(req, decision(req))
    changed = dataclasses.replace(req, created_utc="2026-11-28T00:00:00Z")
    with pytest.raises(s.DecisionStoreError, match="already stored with other content"):
        store.record(changed, decision(changed))
    assert len(store._query("SELECT * FROM decisions")) == 1


def test_record_refuses_a_mismatched_adjudicated_or_unvalidated_decision(store):
    req = request("case-1")
    for bad, phrase in ((decision(request("case-2")), "is for request"),
                        (dataclasses.replace(decision(req), adjudication=adjudication()), "without adjudication"),
                        (decision(req).to_dict(), "validated Request and Decision")):
        with pytest.raises(s.DecisionStoreError, match=phrase):
            store.record(req, bad)
    assert store._query("SELECT * FROM requests") == []


def test_a_failed_write_leaves_nothing_behind(store, monkeypatch):
    insert = store._insert

    def fail_on_decisions(table, values):
        if table == "decisions":
            raise RuntimeError("disk gone")
        return insert(table, values)

    monkeypatch.setattr(store, "_insert", fail_on_decisions)
    req = request("case-1")
    with pytest.raises(RuntimeError):
        store.record(req, decision(req))
    assert store._query("SELECT * FROM requests") == []
    assert not store._conn.in_transaction


def test_only_the_bundle_path_and_hash_are_stored(store):
    base = request("case-1")
    bundle = c.Bundle("outputs/bundles/one.zip", "a" * 64)
    req = dataclasses.replace(base, observation=dataclasses.replace(base.observation, bundle=bundle))
    store.record(req, decision(req))
    (row,) = store._query("SELECT bundle_path, bundle_sha256 FROM requests")
    assert row == {"bundle_path": "outputs/bundles/one.zip", "bundle_sha256": "a" * 64}
    types = {col[2].upper() for t in s._TABLES for col in store._conn.execute(f"PRAGMA table_info({t})")}
    assert "BLOB" not in types


def test_refusals_are_kept_with_their_reasons_and_counted(store):
    add(store, "case-1", "refusal:cell_not_ready")
    add(store, "case-2", "refusal:cell_not_ready")
    add(store, "case-3", "refusal:spec_unknown", split="held_out")
    add(store, "case-4", "accept")
    assert [(r["case_id"], r["refusal"]) for r in store.refusals()] == [
        ("case-1", "cell_not_ready"), ("case-2", "cell_not_ready"), ("case-3", "spec_unknown")]
    assert store.refusals()[0]["reasons"] == [{"rule_id": "R4", "text": "test: refusal:cell_not_ready"}]
    assert [(r["split"], r["refusal"], r["count"]) for r in store.refusal_counts()] == [
        ("development", "cell_not_ready", 2), ("held_out", "spec_unknown", 1)]


# ---------------------------------------------------------------- the leakage guard

def test_tuning_gets_development_cases_only(store):
    add(store, "dev-1", reference="accept")
    add(store, "held-1", split="held_out", reference="reject")
    (case,) = store.cases_for_tuning()
    assert (case.case_id, case.split) == ("dev-1", "development")
    assert [r.case.split for r in case.requests] == ["development"]
    assert len(case.decisions) == 1 and [o.reference for o in case.outcomes] == ["accept"]
    assert store.cases_for_tuning(()) == []


@pytest.mark.parametrize("splits", [("held_out",), ("development", "held_out"), "held_out"])
def test_asking_for_held_out_cases_raises_before_during_and_after_a_freeze(store, splits):
    add(store, "held-1", split="held_out")
    with pytest.raises(s.LeakageError):
        store.cases_for_tuning(splits)
    store.freeze(POLICY.hashes(), "trials")
    with pytest.raises(s.LeakageError):
        store.cases_for_tuning(splits)
    store.unfreeze_for_next_version()
    with pytest.raises(s.LeakageError):
        store.cases_for_tuning(splits)
    assert store.cases_for_tuning() == []


def test_an_unknown_split_is_refused(store):
    with pytest.raises(s.DecisionStoreError, match="unknown split"):
        store.cases_for_tuning(("test",))


@pytest.mark.parametrize("first, second", [("held_out", "development"), ("development", "held_out")])
def test_a_case_keeps_one_split_for_ever(path, first, second):
    with s.DecisionStore.open(path) as opened:
        add(opened, "case-1", split=first)
    with s.DecisionStore.open(path) as opened:
        with pytest.raises(s.LeakageError, match=f"already {first}"):
            add(opened, "case-1", split=second, count=1)
        assert len(opened._query("SELECT * FROM decisions")) == 1


def test_freeze_and_unfreeze_are_explicit_and_logged(store):
    assert store.frozen_policy() is None
    with pytest.raises(s.DecisionStoreError, match="no policy is frozen"):
        store.unfreeze_for_next_version()
    for bad_hashes, note, phrase in (({"spec_sha256": "1" * 64}, "x", "PolicyHashes"),
                                     (POLICY.hashes(), "  ", "needs a note")):
        with pytest.raises(s.DecisionStoreError, match=phrase):
            store.freeze(bad_hashes, note)
    store.freeze(POLICY.hashes(), "held-out trials, round 1")
    assert store.frozen_policy().note == "held-out trials, round 1"
    with pytest.raises(s.DecisionStoreError, match="already frozen"):
        store.freeze(POLICY.hashes(), "again")
    store.unfreeze_for_next_version("trials done")
    assert store.frozen_policy() is None
    store.freeze(POLICY.hashes(), "round 2")
    assert [(e["event"], e["note"]) for e in store.policy_log()] == [
        ("freeze", "held-out trials, round 1"), ("unfreeze", "trials done"), ("freeze", "round 2")]
    assert all(e["recorded_utc"] for e in store.policy_log())


# ---------------------------------------------------------------- reports: a hand-checked fixture

#: case, final decision, reference answer, cost row and cost (spec_v1 authority.scoring_costs)
HAND = [
    ("c01", "accept", "accept", "correct", 0),
    ("c02", "accept", "reject", "false_accept", 10),
    ("c03", "reject", "reject", "correct", 0),
    ("c04", "reject", "accept", "false_reject", 3),
    ("c05", "rework", "reject", "wrong_routing", 2),
    ("c06", "remeasure:refer", "remeasure", "correct", 0),              # a block, rightly referred
    ("c07", "remeasure:refer", "accept", "unnecessary_referral", 1),
    ("c08", "refusal:cell_not_ready", "refusal:cell_not_ready", "correct", 0),
    ("c09", "refusal:spec_unknown", "rework", "unnecessary_referral", 1),
    ("c10", "rework", "rework", "correct", 0),
]
# cases 12, of which pending 2 (c11, c12); graded 10; committed c01-c05 and c10 = 6 of 12; abstained 4
# committed errors c02, c04, c05 = 3 of 6; committed cost 10 + 3 + 2 = 15 over 6
# all cost 15 + 1 (c07) + 1 (c09) = 17 over 10


def fill_hand(store, split="development"):
    for case_id, answer, reference, _, _ in HAND:
        add(store, case_id, answer, reference, split=split)
    add(store, "c11", "accept")                                  # pending: awaiting a correct answer
    add(store, "c12", "remeasure:reobserve", "accept")           # pending: awaiting another look


def test_coverage_and_risk_match_the_hand_check(store):
    fill_hand(store)
    (row,) = store.risk_coverage(COSTS)
    assert (row["cases"], row["graded"], row["pending"]) == (12, 10, 2)
    assert row["pending_cases"] == {"awaiting_look": ["c12"], "awaiting_reference": ["c11"]}
    assert (row["committed"], row["abstained"]) == (6, 4)
    assert row["committed"] + row["abstained"] + row["pending"] == row["cases"]
    assert row["coverage"] == pytest.approx(0.5)  # 6 of all 12 cases
    assert row["acted_when_not_ready"] == {"cases": [], "decisions": 0}
    assert row["selective_error_rate"] == pytest.approx(0.5)
    assert row["selective_cost"] == pytest.approx(2.5)
    assert row["mean_cost"] == pytest.approx(1.7)
    assert row["by_cost_key"] == {s.CORRECT: 5, "false_accept": 1, "false_reject_or_scrap_of_good_part": 1,
                                  "wrong_rework_or_remeasure_routing": 1, "unnecessary_referral": 2}
    assert row["curve"] == []  # no confidence from a deterministic configuration: one point


@pytest.mark.parametrize("case_id, answer, reference, kind, cost", HAND, ids=[h[0] for h in HAND])
def test_each_hand_case_costs_what_the_spec_table_says(store, case_id, answer, reference, kind, cost):
    add(store, case_id, answer, reference)
    assert store.risk_coverage(COSTS)[0]["mean_cost"] == pytest.approx(cost)


@pytest.mark.parametrize("args, key", [
    (("accept", None, None, "cell_not_ready"), "false_accept"),
    (("accept", None, "remeasure", None), "false_accept"),
    (("reject", None, "rework", None), "wrong_rework_or_remeasure_routing"),
    (("reject", None, "remeasure", None), "wrong_rework_or_remeasure_routing"),
    (("rework", None, "accept", None), "wrong_rework_or_remeasure_routing"),
    (("remeasure", None, None, "cell_not_ready"), s.CORRECT),
    ((None, "insufficient_evidence", "remeasure", None), s.CORRECT),
    ((None, "spec_unknown", None, "cell_not_ready"), s.CORRECT),
])
def test_cost_rows_outside_the_hand_fixture(args, key):
    assert s.cost_key(*args) == key


def not_ready_request(case_id="case-1", design="PART221", count=0):
    base = request(case_id, design=design, count=count)
    return dataclasses.replace(base, health=c.Health(cell_ready=False, precapture=None))


def add_made(store, req, configuration):
    """Record the real decision of a configuration about `req`."""
    made = p.decide(req, POLICY, configuration, "abcdef1")
    return store.record(req, made), made


def say(store, decision_id, reference):
    head, _, tail = reference.partition(":")
    store.add_outcome(decision_id, "sim_truth",
                      **({"reference_refusal": tail} if head == "refusal" else {"reference_disposition": head}))


def test_the_cost_is_listed_in_the_spec_with_the_others():
    assert s.ACTED_WHEN_NOT_READY == "acted_when_not_ready"
    assert s.ACTED_WHEN_NOT_READY in s.COST_KEYS and COSTS[s.ACTED_WHEN_NOT_READY] == 10


@pytest.mark.parametrize("reference", ["reject", "accept", None])
def test_a_reject_by_b_on_a_not_ready_cell_is_acted_when_not_ready(store, reference):
    decision_id, made = add_made(store, not_ready_request(design="PART210"), "B")
    assert made.disposition == "reject"
    if reference is not None:
        say(store, decision_id, reference)
    (row,) = store.risk_coverage(COSTS)
    assert row["by_cost_key"] == {"acted_when_not_ready": 1}        # not a false accept, nor a correct reject
    assert row["acted_when_not_ready"] == {"cases": ["case-1"], "decisions": 1}
    assert (row["graded"], row["pending"], row["committed"], row["abstained"]) == (1, 0, 1, 0)
    assert (row["selective_error_rate"], row["selective_cost"], row["mean_cost"]) == (1.0, 10.0, 10.0)


@pytest.mark.parametrize("configuration, design, answer, committed", [
    ("A", "PART221", "accept", 1), ("A", "PART210", "reject", 1), ("A", "PART220", "rework", 1),
    ("B", "PART221", "accept", 1), ("B", "PART220", "rework", 1), ("B", "PART110", "remeasure:reobserve", 0)])
def test_every_action_on_a_not_ready_cell_is_charged_in_a_and_b(store, configuration, design, answer, committed):
    _, made = add_made(store, not_ready_request(design=design), configuration)
    assert s._label(made.disposition, made.refusal, made.remeasure_route) == answer
    (row,) = store.risk_coverage(COSTS)
    assert (row["by_cost_key"], row["mean_cost"]) == ({"acted_when_not_ready": 1}, 10.0)
    assert (row["committed"], row["abstained"], row["pending"]) == (committed, 0, 0)


@pytest.mark.parametrize("configuration", ["C", "D", "E", "F"])
@pytest.mark.parametrize("answer", ["accept", "reject", "rework", "remeasure:reobserve"])
def test_it_applies_to_every_configuration(store, configuration, answer):
    req = not_ready_request()
    store.record(req, decision(req, answer, configuration=configuration))
    assert store.risk_coverage(COSTS)[0]["by_cost_key"] == {"acted_when_not_ready": 1}


@pytest.mark.parametrize("answer", ["remeasure:refer", "refusal:cell_not_ready", "refusal:spec_unknown",
                                    "refusal:insufficient_evidence"])
def test_a_referral_or_a_refusal_on_a_not_ready_cell_is_not_acting(store, answer):
    req = not_ready_request()
    decision_id = store.record(req, decision(req, answer))
    say(store, decision_id, "refusal:cell_not_ready")
    (row,) = store.risk_coverage(COSTS)
    assert row["acted_when_not_ready"] == {"cases": [], "decisions": 0}
    assert (row["by_cost_key"], row["abstained"]) == ({s.CORRECT: 1}, 1)


def test_an_action_on_a_ready_cell_is_not_charged(store):
    for case_id, answer in (("a", "accept"), ("b", "remeasure:reobserve")):
        add(store, case_id, answer)
    assert store.risk_coverage(COSTS)[0]["acted_when_not_ready"] == {"cases": [], "decisions": 0}


def test_an_earlier_relook_on_a_not_ready_cell_does_not_vanish_with_a_later_decision(store):
    _, made = add_made(store, not_ready_request(design="PART110"), "B")
    assert made.remeasure_route == "reobserve"
    later = request("case-1", design="PART210", count=1)            # the cell is ready on the second look
    second, _ = add_made(store, later, "B")
    say(store, second, "reject")
    (row,) = store.risk_coverage(COSTS)
    assert row["acted_when_not_ready"] == {"cases": ["case-1"], "decisions": 1}
    assert (row["cases"], row["by_cost_key"], row["mean_cost"]) == (1, {"acted_when_not_ready": 1}, 10.0)
    assert (row["committed"], row["selective_error_rate"]) == (1, 1.0)


def test_a_case_that_acted_twice_is_charged_once_and_both_decisions_are_counted(store):
    for count in (0, 1):
        add_made(store, not_ready_request(design="PART110", count=count), "B")
    (row,) = store.risk_coverage(COSTS)
    assert row["acted_when_not_ready"] == {"cases": ["case-1"], "decisions": 2}
    assert (row["graded"], row["mean_cost"], row["by_cost_key"]) == (1, 10.0, {"acted_when_not_ready": 1})


def test_the_confusion_table_stays_about_part_quality(store):
    decision_id, _ = add_made(store, not_ready_request(), "A")
    (row,) = store.confusion_table()
    assert "acted_when_not_ready" not in row and "accept_while_cell_not_ready" not in row
    assert row["counts"] == {} and row["pending_cases"]["awaiting_reference"] == ["case-1"]  # still pending
    assert store.outcomes(decision_id) == []                        # scoring writes no answer
    say(store, decision_id, "accept")
    assert store.confusion_table()[0]["counts"] == {"accept": {"accept": 1}}  # the part was judged rightly
    assert store.risk_coverage(COSTS)[0]["by_cost_key"] == {"acted_when_not_ready": 1}


def test_stage_0_refusing_the_same_cell_is_not_that_error(store):
    decision_id, made = add_made(store, not_ready_request(), "E")
    assert made.refusal == "cell_not_ready"
    say(store, decision_id, "refusal:cell_not_ready")
    (row,) = store.risk_coverage(COSTS)
    assert row["acted_when_not_ready"] == {"cases": [], "decisions": 0}
    assert (row["mean_cost"], row["by_cost_key"]) == (0.0, {s.CORRECT: 1})


def test_a_skipped_check_is_kept_in_the_store_and_the_csv(store):
    req = request("case-1")  # twin is null
    made = p.decide(req, POLICY, "E", "abcdef1")
    decision_id = store.record(req, made)
    assert store.get(decision_id)[1] == made
    (row,) = csv.DictReader(io.StringIO(store.export_csv()))
    assert "R9: The twin block is null: the feasibility check for left_tray was skipped." in row["reasons"]


def test_the_confusion_table_matches_the_hand_check(store):
    fill_hand(store)
    (row,) = store.confusion_table()
    assert row["counts"] == {
        "accept": {"accept": 1, "reject": 1, "remeasure:refer": 1},
        "reject": {"accept": 1, "reject": 1, "rework": 1},
        "rework": {"refusal:spec_unknown": 1, "rework": 1},
        "remeasure": {"remeasure:refer": 1},
        "refusal:cell_not_ready": {"refusal:cell_not_ready": 1}}
    assert sum(sum(line.values()) for line in row["counts"].values()) == row["graded"] == 10
    assert row["pending"] == 2
    assert row["pending_cases"] == {"awaiting_look": ["c12"], "awaiting_reference": ["c11"]}


def test_a_cases_final_decision_is_its_latest(store):
    add(store, "case-1", "remeasure:reobserve", "accept")
    assert store.risk_coverage(COSTS)[0]["pending"] == 1
    add(store, "case-1", "accept", "accept", count=1)
    (row,) = store.risk_coverage(COSTS)
    assert (row["decisions"], row["cases"], row["pending"], row["graded"]) == (2, 1, 0, 1)
    assert row["coverage"] == 1.0 and row["selective_error_rate"] == 0.0


def test_results_are_not_pooled_across_policy_versions_splits_or_input(store):
    add(store, "case-1", reference="accept")
    add(store, "case-2", reference="accept", split="held_out")
    add(store, "case-3", reference="accept", rules="9" * 64)
    truth = request("case-4")
    bundle = c.Bundle("outputs/bundles/one.zip", "a" * 64)
    seen = dataclasses.replace(truth, input_source="perceived",
                               observation=dataclasses.replace(truth.observation, bundle=bundle))
    store.record(seen, decision(seen))
    rows = store.risk_coverage(COSTS)
    assert len(rows) == 4 and all(row["cases"] == 1 for row in rows)
    assert {(r["split"], r["rules_sha256"][0], r["input_source"]) for r in rows} == {
        ("development", "2", "sim_truth"), ("held_out", "2", "sim_truth"), ("development", "9", "sim_truth"),
        ("development", "2", "perceived")}


def test_pending_cases_lower_coverage_and_are_listed_separately(store):
    add(store, "done-1", "accept", "accept")
    add(store, "done-2", "reject", "reject")
    assert store.risk_coverage(COSTS)[0]["coverage"] == 1.0
    add(store, "waiting-look", "remeasure:reobserve")
    assert store.risk_coverage(COSTS)[0]["coverage"] == pytest.approx(2 / 3)
    unanswered = add(store, "waiting-answer", "accept")             # committed, but not yet graded
    (row,) = store.risk_coverage(COSTS)
    assert (row["coverage"], row["committed"], row["graded"], row["pending"]) == (0.5, 2, 2, 2)
    assert row["pending_cases"] == {"awaiting_look": ["waiting-look"], "awaiting_reference": ["waiting-answer"]}
    assert (row["selective_error_rate"], row["selective_cost"], row["mean_cost"]) == (0.0, 0.0, 0.0)
    store.add_outcome(unanswered, "sim_truth", reference_disposition="accept")
    (row,) = store.risk_coverage(COSTS)
    assert (row["coverage"], row["pending"]) == (0.75, 1)


def test_the_curve_counts_pending_cases_too(store):
    add(store, "a", "accept", "accept", configuration="C", confidence=0.9)
    add(store, "b", "remeasure:reobserve", configuration="C", confidence=0.5)
    (row,) = store.risk_coverage(COSTS)
    assert [(point["threshold"], point["coverage"]) for point in row["curve"]] == [(0.9, 0.5)]


def test_a_figure_with_nothing_to_divide_by_is_none(store):
    add(store, "case-1")                                            # pending only
    (row,) = store.risk_coverage(COSTS)
    assert row["coverage"] == 0.0  # none of the one case was decided autonomously
    assert [row[k] for k in ("selective_error_rate", "selective_cost", "mean_cost")] == [None] * 3
    add(store, "case-2", "remeasure:refer", "accept")               # graded, nothing committed
    assert store.risk_coverage(COSTS)[0]["mean_cost"] == 1.0
    (row,) = store.risk_coverage(COSTS)
    assert row["coverage"] == 0.0 and row["selective_error_rate"] is None and row["selective_cost"] is None


def test_with_confidences_the_curve_lowers_the_threshold(store):
    for case_id, answer, reference, confidence in (("a", "accept", "accept", 0.9), ("b", "reject", "reject", 0.7),
                                                   ("c", "accept", "reject", 0.7), ("d", "accept", "accept", 0.4),
                                                   ("e", "remeasure:refer", "accept", 0.2)):
        add(store, case_id, answer, reference, configuration="C", confidence=confidence)
    (row,) = store.risk_coverage(COSTS)
    assert [(pt["threshold"], pt["committed"], pt["coverage"], pt["selective_error_rate"], pt["selective_cost"])
            for pt in row["curve"]] == [
        (0.9, 1, pytest.approx(0.2), 0.0, 0.0),
        (0.7, 3, pytest.approx(0.6), pytest.approx(1 / 3), pytest.approx(10 / 3)),
        (0.4, 4, pytest.approx(0.8), pytest.approx(0.25), pytest.approx(2.5))]


def test_incomplete_costs_are_refused(store):
    with pytest.raises(s.DecisionStoreError, match="unnecessary_referral"):
        store.risk_coverage({k: v for k, v in COSTS.items() if k != "unnecessary_referral"})


# ---------------------------------------------------------------- CSV

def test_the_csv_keeps_the_whole_history(store, tmp_path):
    first = add(store, "case-1", "remeasure:reobserve")
    second = add(store, "case-1", "accept", "accept", count=1)
    store.add_outcome(second, "operator", reference_disposition="reject",
                      adjudication=adjudication(notes='cracked, "badly"\non two lines'),
                      twin=s.TwinResult(True, True, []))
    target = tmp_path / "export" / "decisions.csv"
    text = store.export_csv(target)
    assert target.read_text(encoding="utf-8") == text
    rows = list(csv.DictReader(io.StringIO(text)))
    assert list(rows[0]) == list(s.CSV_COLUMNS)
    assert [(int(r["decision_id"]), r["final"], r["outcome_source"]) for r in rows] == [
        (first, "0", ""), (second, "1", "sim_truth"), (second, "1", "operator")]
    last = rows[-1]
    assert last["human_adjudication_notes"] == 'cracked, "badly"\non two lines'
    assert (last["operator_rating"], last["adjudicator"], last["twin_feasible"], last["twin_contacts"]) == (
        "4", "adjudicator-test", "True", "[]")
    assert last["reasons"] == "R4: test: accept" and last["case_id"] == "case-1"


def test_the_csv_times_are_the_decisions_not_the_requests(store, monkeypatch):
    times = iter(["2026-11-27T10:00:00Z", "2026-11-27T11:00:00Z"])
    req = request("case-1")
    monkeypatch.setattr(s, "_now", lambda: next(times))
    store.record(req, decision(req, configuration="A"))
    store.record(req, decision(req, configuration="E"))
    rows = list(csv.DictReader(io.StringIO(store.export_csv())))
    assert [r["decision_recorded_utc"] for r in rows] == ["2026-11-27T10:00:00Z", "2026-11-27T11:00:00Z"]


def test_an_empty_store_reports_nothing(store):
    assert store.risk_coverage(COSTS) == store.confusion_table() == store.refusal_counts() == store.refusals() == []
    assert store.export_csv().strip().split(",") == list(s.CSV_COLUMNS)


def test_the_store_does_not_import_the_sim_motion_or_network():
    lines = [line for line in (ROOT / "src" / "reachy_ai" / "reasoner" / "store.py").read_text().splitlines()
             if line.lstrip().startswith(("import ", "from "))]
    for banned in ("experience", "reachy_sdk", "motion", "anthropic", "grpc", "socket", "urllib", "requests"):
        assert not [line for line in lines if banned in line], banned


def test_connections_opening_a_new_store_together_all_succeed(tmp_path):
    failures = []
    for attempt in range(40):
        target = tmp_path / f"{attempt}.sqlite"
        start = threading.Barrier(4)

        def work():
            start.wait()
            try:
                with s.DecisionStore.open(target) as opened:
                    opened.refusals()
            except Exception as exc:  # noqa: BLE001 - reported by the assert below
                failures.append(exc)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert failures == []
