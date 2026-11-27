"""#83: the viewer's decision panel. Offline: a fake camera service, a stub reasoner and a temporary store."""
import hashlib
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from reachy_ai.audio.audio_client import AudioClient
from reachy_ai.pair_bundle import read_bundle
from reachy_ai.reasoner import ai, panel
from reachy_ai.reasoner import contract as c
from reachy_ai.reasoner import policy as p
from reachy_ai.reasoner import store as s
from reachy_ai.stereo import CommandBusy, StereoService, StereoSettings
from reachy_ai.stereo_app import create_app

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "config" / "disposition"
HEALTHY = {"format_version": 1, "passed": True, "checked_utc": "2026-11-27T12:00:00Z", "failed": [], "checks": []}
SICK = {**HEALTHY, "passed": False, "failed": ["clock_offset"]}


class Stub(ai.Provider):
    model = "model-stub"

    def __init__(self):
        self.answer, self.calls = "accept", []

    def complete(self, system, user):
        self.calls.append(json.loads(user))
        return ai.Reply(json.dumps({"answer": self.answer, "confidence": 0.9, "rationale": "MODEL-PROSE"}),
                        "end_turn")


class Detector:
    def status(self):
        return {"available": True}


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(autouse=True)
def reasoner_env_off(monkeypatch):
    for name in (ai.ENABLE_ENV, ai.MODEL_ENV, ai.KEY_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def make_viewer(tmp_path):
    opened = []

    def make(source="sim", with_ai=True, **kwargs):
        settings = StereoSettings(source=source, robot_host="robot.invalid",
                                  t3_history_dir=str(tmp_path / "history"))
        service = StereoService(settings)
        service._new_session()
        for eye in ("left", "right"):
            service._accept({"kind": "frame", "eye": eye, "sequence": 1, "observed_ns": time.monotonic_ns(),
                             "pixels": np.full((48, 64, 3), 90, np.uint8)})
        service.start = service.stop = lambda: None
        service.health = dict(HEALTHY)
        service.health_checks = 0

        def request_health():
            service.health_checks += 1
            return service.health

        def no_motion(*args, **kw):
            raise AssertionError("the decision panel tried to command the robot")

        service.request_health = request_health
        for name in ("request_head", "request_head_stop", "request_gripper", "request_gripper_stop",
                     "request_zoom", "request_lens"):
            setattr(service, name, no_motion)
        stub = Stub()
        kwargs.setdefault("reasoner", ai.AIReasoner(stub) if with_ai else None)
        kwargs.setdefault("decision_store_path", tmp_path / "decisions" / "decisions.sqlite")
        app = create_app(settings, service, detector=Detector(),
                         audio_client=AudioClient(f"http://127.0.0.1:{free_port()}"), **kwargs)
        client = TestClient(app)
        client.__enter__()
        opened.append(client)
        client.service, client.stub, client.store = service, stub, Path(kwargs["decision_store_path"])
        return client

    yield make
    for client in opened:
        client.__exit__(None, None, None)


@pytest.fixture
def viewer(make_viewer):
    return make_viewer()


def truth(**changes):
    observation = {"kind": "cylinder", "orientation": "upright", "height_mm": {"value": 69.85, "bound": 0.5},
                   "width_mm": {"value": 49.2, "bound": 0.5},
                   "hole": {"state": "present", "diameter_mm": {"value": 25.4, "bound": 0.5}}}
    observation.update(changes.pop("observation", {}))
    return {"mode": "sim_truth", "observation": observation, "degradation": [], "path_feasibility": None, **changes}


def case(case_id="case-1", split="development", **changes):
    return {"case_id": case_id, "split": split, "instance_id": "PART221-A", "lot": "lot-1",
            "expected_design_id": "PART221", **changes}


def decide(client, configuration="E", case_data=None, input_data=None, **body):
    if "request_id" not in body:
        body = {"case": case_data or case(), "input": input_data or truth(), **body}
    return client.post("/api/disposition/decide", json={"configuration": configuration, **body})


def adjudication(**changes):
    return {"adjudicator": "operator-test", "actual_location_final_disposition": "left tray, accepted",
            "human_adjudication_notes": "checked with calipers", "operator_rating": 4, **changes}


def adjudicate(client, decision_id, record=None, answer="accept"):
    return client.post(f"/api/disposition/decisions/{decision_id}/adjudication",
                       json={"record": adjudication() if record is None else record, "answer": answer})


def stored(client, sql):
    with s.DecisionStore.open(client.store) as store:
        return store._query(sql)


# ---------------------------------------------------------------- the page and the state

def test_the_page_has_the_panel_and_its_routes_name_no_motion(viewer):
    page = viewer.get("/").text
    for text in ("DECISION PANEL", "Learning mode", "Propose policy change", "ADJUDICATION (Appendix A)",
                 "Run configuration on the same presentation", "Policy files", "reachyCapturedPair",
                 *c.APPENDIX_A_FIELDS):
        assert text in page, text
    paths = {route.path for route in viewer.app.routes if route.path.startswith("/api/disposition")}
    assert paths == {"/api/disposition/state", "/api/disposition/policy", "/api/disposition/learning",
                     "/api/disposition/decide", "/api/disposition/decisions/{decision_id}",
                     "/api/disposition/decisions/{decision_id}/bundle.zip",
                     "/api/disposition/decisions/{decision_id}/adjudication", "/api/disposition/learning/report"}


def test_by_default_the_ai_reasoner_is_off_and_the_state_writes_nothing(make_viewer, tmp_path):
    settings = StereoSettings(t3_history_dir=str(tmp_path / "h"))
    app = create_app(settings, StereoService(settings), detector=Detector(),
                     decision_store_path=tmp_path / "never" / "d.sqlite")  # reasoner not given: from the environment
    state = TestClient(app).get("/api/disposition/state").json()
    assert state["ai"]["enabled"] is False and ai.ENABLE_ENV in state["ai"]["note"]
    assert {x["id"]: x["available"] for x in state["configurations"]} == {
        "A": True, "B": True, "E": True, "C": False, "D": False, "F": False}
    assert state["learning"] is False and state["awaiting_rating"] == []
    assert state["required_adjudication_fields"] == list(s.REQUIRED_ADJUDICATION_FIELDS)
    assert state["appendix_a_fields"] == list(c.APPENDIX_A_FIELDS)
    for kind, name in (("spec", "spec_v1.yaml"), ("rules", "rules_v2.yaml"), ("prompt", "prompt_v1.md"),
                       ("split", "split_v0.yaml")):
        entry = state["policy"][kind]
        assert entry["file"] == name and entry["version"] == Path(name).stem
        assert entry["sha256"] == hashlib.sha256((POLICY_DIR / name).read_bytes()).hexdigest()
    assert {f["file"] for f in state["policy"]["rules"]["available"]} == {"rules_v1.yaml", "rules_v2.yaml"}
    assert not (tmp_path / "never").exists()


def test_with_the_ai_off_an_ai_decision_is_refused_before_anything_happens(make_viewer):
    viewer = make_viewer(with_ai=False)
    for configuration in ai.AI_CONFIGURATIONS:
        response = decide(viewer, configuration)
        assert response.status_code == 409 and "is off" in response.json()["detail"]
    assert viewer.service.health_checks == 0 and not viewer.store.exists()


# ---------------------------------------------------------------- decide

def test_a_deterministic_decision_is_shown_with_its_reasons_and_policy_versions(viewer):
    response = decide(viewer, "E")
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["decision"]["disposition"], body["decision"]["destination"]) == ("accept", "left_tray")
    assert body["decision"]["reasons"][0]["rule_id"] == "R4" and body["decision"]["model_id"] is None
    assert set(body["policy_files"]) == {"spec", "rules", "split"}  # E uses no prompt
    for kind, entry in body["policy_files"].items():
        assert entry["sha256"] == body["decision"]["policy"][f"{kind}_sha256"]
    request = body["request"]
    assert (request["input_source"], request["source"], request["cell_ready"], request["reobservation_count"]) == (
        "sim_truth", "sim", True, 0)
    assert "moves the robot" in body["motion_note"] and viewer.service.health_checks == 1
    with s.DecisionStore.open(viewer.store) as store:
        saved_request, saved = store.get(body["decision_id"])
    assert saved.disposition == "accept" and saved_request.case.instance_id == "PART221-A"
    assert len(saved.code_sha) == 40 and saved_request.observation.bundle is None


def test_the_health_block_is_the_viewers_own_check(viewer):
    viewer.service.health = dict(SICK)
    body = decide(viewer, "E").json()
    assert body["decision"]["refusal"] == "cell_not_ready"
    assert (body["request"]["cell_ready"], body["request"]["failed_checks"]) == (False, ["clock_offset"])


def test_a_busy_robot_connection_records_nothing(viewer):
    def busy():
        raise CommandBusy("another command is running")

    viewer.service.request_health = busy
    assert decide(viewer).status_code == 409 and not viewer.store.exists()


def test_injected_degradation_raises_the_flag_and_is_recorded_as_sim_truth(viewer):
    body = decide(viewer, "E", input_data=truth(degradation=["stale", "blurred"])).json()
    assert body["decision"]["disposition"] == "remeasure" and body["request"]["degradation"] == ["blurred", "stale"]
    with s.DecisionStore.open(viewer.store) as store:
        request, _ = store.get(body["decision_id"])
    assert request.observation.sufficiency.raised() == ("blurred", "stale") and request.input_source == "sim_truth"


def test_a_twin_block_from_the_form_reaches_the_gate(viewer):
    body = decide(viewer, "E", input_data=truth(path_feasibility={"left_tray": False})).json()
    assert body["decision"]["refusal"] == "beyond_capability"


def test_an_ai_decision_keeps_the_raw_reply_and_does_not_show_the_models_words(viewer):
    body = decide(viewer, "C").json()
    assert body["decision"]["disposition"] == "accept" and body["decision"]["raw_reply_kept"] is True
    assert "MODEL-PROSE" not in json.dumps(body)
    assert "MODEL-PROSE" not in json.dumps(viewer.get(f"/api/disposition/decisions/{body['decision_id']}").json())
    assert set(body["policy_files"]) == {"spec", "rules", "split", "prompt"}
    prompt_sha = hashlib.sha256((POLICY_DIR / "prompt_v1.md").read_bytes()).hexdigest()
    assert body["decision"]["policy"]["prompt_sha256"] == prompt_sha and body["decision"]["model_id"] == "model-stub"
    (row,) = stored(viewer, "SELECT decision_json FROM decisions")
    assert "MODEL-PROSE" in row["decision_json"]


def test_one_ai_call_per_case_and_other_configurations_share_the_presentation(viewer):
    first = decide(viewer, "C").json()
    again = decide(viewer, "C", request_id=first["request"]["request_id"])
    assert again.status_code == 409 and "new case_id" in again.json()["detail"]
    relook = decide(viewer, "C")  # a new request of the same case
    assert relook.status_code == 409 and len(viewer.stub.calls) == 1
    checks = viewer.service.health_checks
    second = decide(viewer, "D", request_id=first["request"]["request_id"]).json()
    assert second["request"]["request_id"] == first["request"]["request_id"] and len(viewer.stub.calls) == 2
    assert viewer.service.health_checks == checks  # the stored request is used unchanged
    assert len(stored(viewer, "SELECT * FROM requests")) == 1
    assert decide(viewer, "E", request_id="req-unknown").status_code == 409


def test_a_new_look_at_a_case_is_counted_and_keeps_the_case(viewer):
    counts = [decide(viewer, "B").json()["request"]["reobservation_count"] for _ in range(3)]
    assert counts == [0, 1, 2]
    for changes, field in (({"instance_id": "PART221-B"}, "instance_id"), ({"lot": "other"}, "lot"),
                           ({"split": "held_out"}, "split"), ({"expected_design_id": None}, "expected_design_id")):
        response = decide(viewer, "B", case_data=case(**changes))
        assert response.status_code == 409 and field in response.json()["detail"]
    assert len(stored(viewer, "SELECT * FROM requests")) == 3


def test_a_blank_case_id_makes_a_new_case(viewer):
    ids = {decide(viewer, "A", case_data=case(case_id="")).json()["request"]["case_id"] for _ in range(2)}
    assert len(ids) == 2 and all(i.startswith("case-") for i in ids)


@pytest.mark.parametrize("body, status, phrase", [
    ({"configuration": "Q", "case": case(), "input": truth()}, 422, "Unknown configuration"),
    ({"configuration": "E"}, 422, "either a stored request_id"),
    ({"configuration": "E", "request_id": "x", "case": case()}, 422, "either a stored request_id"),
    ({"configuration": "E", "case": case()}, 422, "input must be an object"),
    ({"configuration": "E", "case": {"instance_id": "a", "lot": "b"}, "input": truth()}, 422, "split"),
    ({"configuration": "E", "case": case(split=""), "input": truth()}, 422, "split"),
    ({"configuration": "E", "case": case(instance_id="  "), "input": truth()}, 422, "instance_id"),
    ({"configuration": "E", "case": {**case(), "colour": "red"}, "input": truth()}, 422, "unknown field"),
    ({"configuration": "E", "case": case(), "input": truth(mode="guess")}, 422, "input.mode"),
    ({"configuration": "E", "case": case(), "input": truth(degradation=["foggy"])}, 422, "degradation"),
    ({"configuration": "E", "case": case(), "input": truth(observation={"kind": "sphere"})}, 422, "kind"),
    ({"configuration": "E", "case": case(), "input": truth(observation={"sufficiency": {}})}, 422, "unknown field"),
    ({"configuration": "E", "case": case(), "input": truth(path_feasibility={"bin": True})}, 422, "path_feasibility"),
    ({"configuration": "E", "case": case(), "input": {"mode": "pair"}}, 422, "Capture a pair"),
    ({"configuration": "E", "case": case(), "input": truth(mode="pair")}, 422, "no typed observation"),
    ({"configuration": "E", "case": case(), "input": {"mode": "pair", "pair_id": "nope"}}, 410, "expired"),
])
def test_a_bad_decide_request_is_refused_and_records_no_decision(viewer, body, status, phrase):
    response = viewer.post("/api/disposition/decide", json=body)
    assert response.status_code == status and phrase in response.json()["detail"], response.text
    assert not viewer.store.exists() or stored(viewer, "SELECT * FROM decisions") == []


def test_without_a_readable_commit_nothing_is_decided(viewer, monkeypatch):
    monkeypatch.setattr(panel, "code_sha", lambda *a: None)
    assert decide(viewer).status_code == 503 and not viewer.store.exists()


# ---------------------------------------------------------------- a captured pair, sim and robot

@pytest.mark.parametrize("source", ["sim", "robot"])
def test_a_captured_pair_is_evidence_with_nothing_measured(make_viewer, source):
    viewer = make_viewer(source=source)
    pair = viewer.post("/api/stereo/pairs").json()
    body = decide(viewer, "E", input_data={"mode": "pair", "pair_id": pair["pair_id"]}).json()
    assert (body["request"]["source"], body["request"]["input_source"]) == (source, "perceived")
    assert (body["decision"]["disposition"], body["decision"]["remeasure_route"]) == ("remeasure", "reobserve")
    assert "Missing measurement(s): width, height, orientation" in body["decision"]["reasons"][0]["text"]
    data = viewer.get(body["bundle_url"]).content
    with s.DecisionStore.open(viewer.store) as store:
        request, _ = store.get(body["decision_id"])
    assert request.observation.bundle == c.Bundle(f"reachy-pair-{pair['pair_id'][:8]}.zip",
                                                  hashlib.sha256(data).hexdigest())
    assert {"left.jpg", "right.jpg", "pair.json"} <= set(read_bundle(data)[1])
    assert request.twin is None and request.observation.height_mm is None and request.observation.kind == "unknown"
    assert [f.name for f in viewer.store.parent.iterdir() if f.suffix in (".zip", ".jpg")] == []  # images unwritten


def test_sim_truth_is_refused_on_the_robot(make_viewer):
    viewer = make_viewer(source="robot")
    response = decide(viewer)
    assert response.status_code == 409 and "--source sim only" in response.json()["detail"]
    assert viewer.get("/api/disposition/state").json()["source"] == "robot"


def test_a_stored_sim_request_is_not_run_by_a_robot_viewer(make_viewer):
    request_id = decide(make_viewer(), "E").json()["request"]["request_id"]
    response = decide(make_viewer(source="robot"), "A", request_id=request_id)
    assert response.status_code == 409 and "--source sim" in response.json()["detail"]


def test_only_a_few_bundles_are_held(viewer):
    pair = viewer.post("/api/stereo/pairs").json()
    urls = [decide(viewer, "A", case_data=case(f"case-{i}"),
                   input_data={"mode": "pair", "pair_id": pair["pair_id"]}).json()["bundle_url"]
            for i in range(panel.MAX_BUNDLES + 1)]
    assert viewer.get(urls[0]).status_code == 410 and viewer.get(urls[-1]).status_code == 200


# ---------------------------------------------------------------- adjudication

def test_an_adjudication_is_saved_as_an_operator_outcome(viewer):
    decision_id = decide(viewer).json()["decision_id"]
    record = adjudication(object_id_and_lot=" PART221-A, lot 1 ", measured_dims_hole="")
    response = adjudicate(viewer, decision_id, record, "reject")
    assert response.status_code == 200, response.text
    with s.DecisionStore.open(viewer.store) as store:
        (outcome,) = store.outcomes(decision_id)
    assert (outcome.source, outcome.reference, outcome.adjudication.operator_rating) == ("operator", "reject", 4)
    assert outcome.adjudication.object_id_and_lot == "PART221-A, lot 1"
    assert outcome.adjudication.measured_dims_hole is None
    assert adjudicate(viewer, decision_id, answer="refusal:spec_unknown").status_code == 200
    assert viewer.get(f"/api/disposition/decisions/{decision_id}").json()["outcomes"] == 2


@pytest.mark.parametrize("field", s.REQUIRED_ADJUDICATION_FIELDS)
@pytest.mark.parametrize("blank", ["", "   ", None])
@pytest.mark.parametrize("answer", ["accept", None])
def test_the_form_enforces_the_required_fields_with_or_without_an_answer(viewer, field, blank, answer):
    decision_id = decide(viewer).json()["decision_id"]
    response = adjudicate(viewer, decision_id, adjudication(**{field: blank}), answer)
    assert response.status_code == 422 and field in response.json()["detail"]
    assert stored(viewer, "SELECT * FROM outcomes") == []


def test_the_form_may_leave_the_other_fields_the_rating_and_the_answer_empty(viewer):
    decision_id = decide(viewer).json()["decision_id"]
    least = {name: "x" for name in s.REQUIRED_ADJUDICATION_FIELDS}
    assert adjudicate(viewer, decision_id, least, None).status_code == 200
    with s.DecisionStore.open(viewer.store) as store:
        (outcome,) = store.outcomes(decision_id)
    assert outcome.reference is None and outcome.adjudication.operator_rating is None


@pytest.mark.parametrize("record, answer, phrase", [
    (adjudication(operator_rating=0), "accept", "operator_rating"),
    (adjudication(operator_rating=6), "accept", "operator_rating"),
    (adjudication(operator_rating="4"), "accept", "operator_rating"),
    (adjudication(colour="red"), "accept", "unknown field"),
    (adjudication(human_adjudication_notes="x" * 4001), "accept", "at most"),
    (adjudication(), "ship_it", "Unknown correct answer"),
    (adjudication(), "refusal:tired", "Unknown correct answer"),
])
def test_a_bad_adjudication_is_refused(viewer, record, answer, phrase):
    decision_id = decide(viewer).json()["decision_id"]
    response = adjudicate(viewer, decision_id, record, answer)
    assert response.status_code == 422 and phrase in response.json()["detail"], response.text
    assert stored(viewer, "SELECT * FROM outcomes") == []


def test_adjudicating_an_unknown_decision_is_refused(viewer):
    assert adjudicate(viewer, 99).status_code == 409


# ---------------------------------------------------------------- learning mode and its report

def set_learning(viewer, on):
    return viewer.post("/api/disposition/learning", json={"on": on}).json()


def test_learning_mode_tags_new_decisions_and_asks_for_each_rating(make_viewer):
    viewer = make_viewer()
    before = decide(viewer, "A", case_data=case("before")).json()
    assert before["learning"] is False and before["rating_requested"] is False
    assert set_learning(viewer, True)["learning"] is True
    plain = decide(viewer, "E", case_data=case("tagged-e")).json()
    by_ai = decide(viewer, "C", case_data=case("tagged-c")).json()
    for body in (plain, by_ai):
        assert body["learning"] is True and body["rating_requested"] is True
        assert body["decision"]["reasons"][-1]["rule_id"] == panel.LEARNING_RULE_ID
    waiting = viewer.get("/api/disposition/state").json()["awaiting_rating"]
    assert [w["case_id"] for w in waiting] == ["tagged-e", "tagged-c"]
    unrated = adjudicate(viewer, plain["decision_id"], adjudication(operator_rating=None))
    assert unrated.status_code == 422 and "rating" in unrated.json()["detail"]
    assert adjudicate(viewer, plain["decision_id"]).status_code == 200
    waiting = viewer.get("/api/disposition/state").json()["awaiting_rating"]
    assert [w["case_id"] for w in waiting] == ["tagged-c"]
    set_learning(viewer, False)
    assert decide(viewer, "A", case_data=case("after")).json()["learning"] is False
    assert make_viewer().get("/api/disposition/state").json()["learning"] is False  # off after a restart


def policy_hashes():
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in POLICY_DIR.iterdir()}


def test_the_report_lists_development_disagreements_only_and_changes_no_file(viewer):
    set_learning(viewer, True)

    def answered(case_id, answer, split="development"):
        decision_id = decide(viewer, "E", case_data=case(case_id, split=split)).json()["decision_id"]
        assert adjudicate(viewer, decision_id, answer=answer).status_code == 200
        return decision_id

    differs = answered("dev-differs", "reject")
    answered("dev-agrees", "accept")
    corrected = answered("dev-corrected", "reject")
    assert adjudicate(viewer, corrected, answer="accept").status_code == 200  # the latest answer counts
    answered("held-differs", "reject", split="held_out")
    before = policy_hashes()
    report = viewer.get("/api/disposition/learning/report").json()
    assert policy_hashes() == before
    assert [(d["case_id"], d["decision_id"], d["decided"], d["correct"]) for d in report["disagreements"]] == [
        ("dev-differs", differs, "accept", "reject")]
    assert "held" not in json.dumps(report["disagreements"])
    assert (report["development_cases"], report["decisions_with_an_answer"]) == (3, 3)
    assert report["by_rule"] == {"E R4: accept -> reject": 1} and "No file is changed" in report["note"]
    row = report["disagreements"][0]
    assert (row["learning"], row["operator_rating"], row["answer_source"]) == (True, 4, "operator")
    assert row["notes"] == "checked with calipers" and row["rules_sha256"]


def test_the_report_refuses_a_held_out_case_and_creates_no_store(viewer):
    assert viewer.get("/api/disposition/learning/report").json()["disagreements"] == []
    assert not viewer.store.exists()
    with pytest.raises(s.DecisionStoreError, match="may not feed"):
        panel.learning_report([s.TuningCase("x", "held_out", (), (), ())])


# ---------------------------------------------------------------- the policy files

def git(directory, *args):
    subprocess.run(["git", "-C", str(directory), "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)


@pytest.fixture
def policy_repo(tmp_path):
    directory = tmp_path / "policy"
    shutil.copytree(POLICY_DIR, directory)
    git(directory, "init", "-q")
    git(directory, "add", ".")
    git(directory, "commit", "-qm", "policy")
    return directory


def choose(viewer, **files):
    return viewer.post("/api/disposition/policy", json={"files": files})


def test_a_committed_version_can_be_chosen_and_is_what_decisions_record(make_viewer, policy_repo):
    viewer = make_viewer(policy_dir=policy_repo)
    assert viewer.get("/api/disposition/state").json()["policy"]["rules"]["state"] == "committed"
    state = choose(viewer, rules="rules_v1.yaml").json()
    assert state["policy"]["rules"]["version"] == "rules_v1"
    body = decide(viewer).json()
    assert body["policy_files"]["rules"]["file"] == "rules_v1.yaml"
    assert body["decision"]["policy"]["rules_sha256"] == p.load_policy(policy_repo, p.RULES_V1_FILE).rules.sha256


def test_editing_the_active_file_is_the_other_update_path(make_viewer, policy_repo):
    viewer = make_viewer(policy_dir=policy_repo)
    first = decide(viewer, "C", case_data=case("one")).json()
    (policy_repo / "prompt_v1.md").write_text((policy_repo / "prompt_v1.md").read_text() + "\nEdited.\n")
    second = decide(viewer, "C", case_data=case("two")).json()
    assert second["decision"]["policy"]["prompt_sha256"] != first["decision"]["policy"]["prompt_sha256"]
    assert second["policy_files"]["prompt"]["state"] == "modified"
    assert viewer.get("/api/disposition/state").json()["policy"]["prompt"]["state"] == "modified"
    assert choose(viewer, rules="rules_v1.yaml", prompt="prompt_v1.md").status_code == 200  # unchanged is allowed


def test_only_committed_listed_and_loadable_versions_can_be_chosen(make_viewer, policy_repo):
    viewer = make_viewer(policy_dir=policy_repo)
    (policy_repo / "prompt_v2.md").write_text("A second prompt.\n")
    (policy_repo / "rules_v1.yaml").write_text((policy_repo / "rules_v1.yaml").read_text() + "\n# edited\n")
    for files, phrase in (({"prompt": "prompt_v2.md"}, "untracked"), ({"rules": "rules_v1.yaml"}, "modified")):
        response = choose(viewer, **files)
        assert response.status_code == 409 and phrase in response.json()["detail"]
    (policy_repo / "spec_v2.yaml").write_text("version: spec_v2\n")
    git(policy_repo, "add", "prompt_v2.md", "spec_v2.yaml")
    git(policy_repo, "commit", "-qm", "more")
    broken = choose(viewer, prompt="prompt_v2.md", spec="spec_v2.yaml")
    assert broken.status_code == 422 and "Not applied" in broken.json()["detail"]
    for files in ({"rules": "../disposition/rules_v2.yaml"}, {"rules": "spec_v1.yaml"}, {"model": "x"}):
        assert choose(viewer, **files).status_code == 422
    state = viewer.get("/api/disposition/state").json()
    assert {k: v["file"] for k, v in state["policy"].items()} == {
        "spec": "spec_v1.yaml", "rules": "rules_v2.yaml", "prompt": "prompt_v1.md", "split": "split_v0.yaml"}
    assert choose(viewer, prompt="prompt_v2.md").json()["policy"]["prompt"]["version"] == "prompt_v2"


def test_outside_a_git_checkout_the_commit_state_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    (tmp_path / "prompt_v1.md").write_text("x")
    assert panel.commit_state(tmp_path / "prompt_v1.md") == "unknown"
    assert panel.code_sha(tmp_path) is None
    assert panel.policy_entry(tmp_path, "prompt_v9.md")["state"] == "missing"


def test_a_policy_file_broken_by_an_edit_stops_decisions_with_a_message(make_viewer, policy_repo):
    viewer = make_viewer(policy_dir=policy_repo)
    (policy_repo / "rules_v2.yaml").write_text("version: [")
    response = decide(viewer)
    assert response.status_code == 422 and "Policy:" in response.json()["detail"]
    assert "rules_v2" in viewer.get("/api/disposition/state").json()["policy_error"]


def test_reading_an_unknown_decision_or_bundle_gives_a_plain_error(viewer):
    assert viewer.get("/api/disposition/decisions/1").status_code == 409
    assert viewer.get("/api/disposition/decisions/1/bundle.zip").status_code == 410
