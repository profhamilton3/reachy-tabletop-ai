"""#83: the AI reasoner (configurations C, D and F). Offline: the provider and the SDK client are stubs."""
import dataclasses
import hashlib
import inspect
import json
import logging
import socket
import threading
from pathlib import Path

import pytest

from reachy_ai.reasoner import ai
from reachy_ai.reasoner import contract as c
from reachy_ai.reasoner import policy as p
from reachy_ai.reasoner import store as s

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "disposition" / "valid" / "request"
POLICY = p.load_policy()
PROMPT = ai.load_prompt()
SHA = "abcdef1"
SECRET = "sk-ant-test-SECRET-0123456789"
COMMITTING = ("accept", "reject", "rework")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    for name in (ai.ENABLE_ENV, ai.MODEL_ENV, ai.KEY_ENV):
        monkeypatch.delenv(name, raising=False)


def reply(answer="accept", confidence=0.8, rationale="the model's own words", **extra):
    return json.dumps({"answer": answer, "confidence": confidence, "rationale": rationale, **extra})


class Stub(ai.Provider):
    model = "model-stub"

    def __init__(self, text=None, stop_reason="end_turn", error=None):
        self.text, self.stop_reason, self.error = reply() if text is None else text, stop_reason, error
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        if self.error is not None:
            raise self.error
        return ai.Reply(self.text, self.stop_reason)


def request(design="PART221", case_id="case-1", count=0, **observation):
    base = c.loads_request((FIXTURES / ("spec_%s.json" % design.lower())).read_text())
    return dataclasses.replace(
        base, request_id=f"req-{case_id}-{count}", reobservation_count=count,
        case=dataclasses.replace(base.case, case_id=case_id),
        observation=dataclasses.replace(base.observation, **observation))


def blurred(req):
    flags = dataclasses.replace(req.observation.sufficiency, blurred=True)
    return dataclasses.replace(req, observation=dataclasses.replace(req.observation, sufficiency=flags))


def with_twin(req, **feasible):
    return dataclasses.replace(req, twin=c.Twin(part_pose=None, path_feasibility=tuple(feasible.items())))


def not_ready(req):
    return dataclasses.replace(req, health=c.Health(cell_ready=False, precapture=None))


ALL_PATHS = {"left_tray": True, "right_tray": True, "rework_tray": True}


def run(req, configuration, stub):
    return ai.AIReasoner(stub).decide(req, POLICY, PROMPT, configuration, SHA)


def rules_of(decision):
    return [r.rule_id for r in decision.reasons]


def words_shown(decision):
    return " ".join(r.text for r in decision.reasons if r.rule_id != ai.RAW_REPLY_RULE_ID)


# ---------------------------------------------------------------- good replies

@pytest.mark.parametrize("configuration", ai.AI_CONFIGURATIONS)
@pytest.mark.parametrize("answer", ai.ANSWERS)
def test_every_allowed_answer_makes_a_valid_decision(configuration, answer):
    stub = Stub(reply(answer))
    decision = run(with_twin(request(), **ALL_PATHS), configuration, stub)
    assert c.loads_decision(c.dumps(decision)) == decision  # the R1 schema accepts it, through the wire form
    if answer in c.DISPOSITIONS:
        assert (decision.disposition, decision.refusal) == (answer, None)
    else:
        assert (decision.disposition, decision.refusal) == (None, answer)
    assert decision.remeasure_route == ("refer" if answer == "remeasure" else None)  # no second AI look
    assert decision.destination == {"accept": "left_tray", "reject": "right_tray",
                                    "rework": "rework_tray"}.get(answer, "none")
    assert (decision.configuration, decision.model_id, decision.confidence) == (configuration, "model-stub", 0.8)
    assert decision.policy == dataclasses.replace(POLICY.hashes(), prompt_sha256=PROMPT.sha256)
    assert rules_of(decision)[0] == "AI" and decision.reasons[-1] == c.Reason(ai.RAW_REPLY_RULE_ID, stub.text)
    assert "the model's own words" not in words_shown(decision)
    assert len(stub.calls) == 1


def test_reasoner_invalid_is_not_an_answer_the_model_may_give():
    assert "reasoner_invalid" not in ai.ANSWERS
    assert set(ai.ANSWERS) == set(c.DISPOSITIONS) | set(c.REFUSALS) - {"reasoner_invalid"}
    assert ai.REPLY_SCHEMA["properties"]["answer"]["enum"] == list(ai.ANSWERS)
    assert ai.REPLY_SCHEMA["additionalProperties"] is False


@pytest.mark.parametrize("configuration", ["A", "B", "E", "Z"])
def test_a_deterministic_configuration_is_not_run_here(configuration):
    stub = Stub()
    with pytest.raises(ai.ReasonerError, match="not an AI configuration"):
        run(request(), configuration, stub)
    assert stub.calls == []


# ---------------------------------------------------------------- bad replies

BAD = {
    "empty": "",
    "prose": "I think we should accept it.",
    "prose around the object": "Here it is: " + reply() + " Done.",
    "fenced": "```json\n" + reply() + "\n```",
    "a list": "[]",
    "a string": '"accept"',
    "a missing field": json.dumps({"answer": "accept", "confidence": 0.8}),
    "an extra field": reply(destination="left_tray"),
    "joint values": reply(joints=[0.1, 0.2, 0.3]),
    "an unknown answer": reply("ship_it"),
    "a motion word": reply("move_arm"),
    "the code's own refusal": reply("reasoner_invalid"),
    "an answer that is a list": reply(["accept"]),
    "confidence above 1": reply(confidence=1.5),
    "confidence below 0": reply(confidence=-0.1),
    "confidence as text": reply(confidence="0.8"),
    "confidence as a bool": reply(confidence=True),
    "NaN confidence": '{"answer": "accept", "confidence": NaN, "rationale": "x"}',
    "a repeated key": '{"answer": "reject", "answer": "accept", "confidence": 0.8, "rationale": "x"}',
    "a rationale that is not text": reply(rationale={"joint": 1}),
    "deep nesting": "[" * 100000,
}


@pytest.mark.parametrize("configuration", ai.AI_CONFIGURATIONS)
@pytest.mark.parametrize("text", BAD.values(), ids=BAD.keys())
def test_an_invalid_reply_is_the_refusal_reasoner_invalid_never_a_disposition(configuration, text):
    decision = run(with_twin(request(), **ALL_PATHS), configuration, Stub(text))
    assert (decision.disposition, decision.refusal, decision.destination) == (None, "reasoner_invalid", "none")
    assert decision.confidence is None and rules_of(decision) == ["AI", ai.RAW_REPLY_RULE_ID]
    assert decision.reasons[1].text == (text or ai.EMPTY_REPLY)  # every reply is kept
    assert c.loads_decision(c.dumps(decision)) == decision


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal", "pause_turn", None, "something_new"])
def test_a_reply_that_did_not_end_normally_is_invalid_even_if_it_parses(stop_reason):
    decision = run(request(), "C", Stub(reply(), stop_reason=stop_reason))
    assert decision.refusal == "reasoner_invalid"
    assert "stop reason" in decision.reasons[0].text and "something_new" not in decision.reasons[0].text


def test_a_lone_surrogate_in_the_reply_can_still_be_stored_and_exported(tmp_path):
    with s.DecisionStore.open(tmp_path / "d.sqlite") as store:
        ai.decide_once(ai.AIReasoner(Stub("\udcff broken")), store, request(), POLICY, PROMPT, "C", SHA)
        store.export_csv(tmp_path / "d.csv")


class StatusError(Exception):
    status_code = 401


@pytest.mark.parametrize("error, label", [
    (StatusError(f"401 invalid x-api-key {SECRET} at https://example.invalid"), "StatusError 401"),
    (ConnectionError(f"host unreachable, header {SECRET}"), "ConnectionError"),
    (TimeoutError(), "TimeoutError"),
    (ai.ProviderError("no API key in the environment"), "no API key in the environment"),
])
def test_a_failed_call_is_reasoner_invalid_and_never_logs_the_message(caplog, error, label):
    stub = Stub(error=error)
    with caplog.at_level(logging.DEBUG):
        decision = run(request(), "C", stub)
    assert decision.refusal == "reasoner_invalid" and rules_of(decision) == ["AI"]
    assert label in decision.reasons[0].text and label in caplog.text
    for text in (caplog.text, c.dumps(decision)):
        assert SECRET not in text and "example.invalid" not in text


# ---------------------------------------------------------------- what the model sees

def test_the_model_never_sees_names_the_split_or_the_injected_degradation():
    base = request()
    secretive = dataclasses.replace(
        blurred(base), degradation=("blurred",), request_id="req-SECRET-REQUEST",
        case=c.Case(case_id="case-SECRET-CASE", split="held_out", expected_design_id=None,
                    instance_id="PART221-SECRET-INSTANCE", lot="SECRET-LOT"),
        observation=dataclasses.replace(blurred(base).observation,
                                        bundle=c.Bundle("outputs/SECRET-BUNDLE.zip", "a" * 64)))
    for configuration in ai.AI_CONFIGURATIONS:
        document = ai.model_input(secretive, configuration)
        text = json.dumps(document)
        for hidden in ("SECRET", "PART221", "held_out", "sim_truth", "degradation", "a" * 64, "2026-"):
            assert hidden not in text, hidden
        assert set(document["observation"]) == {"kind", "height_mm", "width_mm", "hole", "orientation",
                                                "sufficiency"}
        assert document["observation"]["sufficiency"]["blurred"] is True


def test_only_d_and_f_see_the_twin_and_only_its_path_feasibility():
    req = dataclasses.replace(request(), twin=c.Twin(
        part_pose=c.PartPose((0.2, 0.0, -0.3), (1.0, 0.0, 0.0, 0.0)), path_feasibility=(("left_tray", False),)))
    assert "twin" not in ai.model_input(req, "C")
    for configuration in ("D", "F"):
        assert ai.model_input(req, configuration)["twin"] == {"path_feasibility": {"left_tray": False}}
        assert ai.model_input(request(), configuration)["twin"] is None


def test_the_system_prompt_is_the_file_then_the_designs_without_costs_or_rules():
    text = ai.system_prompt(PROMPT, POLICY.spec)
    assert text.startswith(PROMPT.text.rstrip())
    document = json.loads(text.split("## Specification\n", 1)[1])
    assert [d["design"] for d in document["designs"]] == [d.design for d in POLICY.spec.designs]
    assert set(document["tolerances_mm"]) == set(p.TOLERANCES)
    for hidden in ("scoring_costs", "false_accept", "quality_authority", "R10", "held_out"):
        assert hidden not in text


def test_the_prompt_hash_is_of_the_file_as_it_is_read(tmp_path):
    path = ROOT / "config" / "disposition" / "prompt_v1.md"
    assert (PROMPT.version, PROMPT.sha256) == ("prompt_v1", hashlib.sha256(path.read_bytes()).hexdigest())
    (tmp_path / ai.PROMPT_FILE).write_text("edited")
    assert ai.load_prompt(tmp_path).sha256 != PROMPT.sha256
    (tmp_path / ai.PROMPT_FILE).write_text("  \n")
    with pytest.raises(p.PolicyError, match="empty"):
        ai.load_prompt(tmp_path)
    with pytest.raises(p.PolicyError, match="prompt_v9"):
        ai.load_prompt(tmp_path, "prompt_v9.md")


# ---------------------------------------------------------------- one call per case

@pytest.fixture
def store(tmp_path):
    with s.DecisionStore.open(tmp_path / "decisions.sqlite") as opened:
        yield opened


@pytest.mark.parametrize("stub", [Stub(), Stub("nonsense"), Stub(error=TimeoutError())],
                         ids=["answered", "invalid", "failed"])
def test_one_call_per_case_and_configuration_whatever_came_back(stub):
    stub.calls = []
    reasoner = ai.AIReasoner(stub)
    reasoner.decide(request(), POLICY, PROMPT, "C", SHA)
    for again in (request(), request(count=1)):  # the same look, and a later one of the same case
        with pytest.raises(ai.AlreadyCalled, match="a repeated trial needs a new case_id"):
            reasoner.decide(again, POLICY, PROMPT, "C", SHA)
    assert len(stub.calls) == 1
    reasoner.decide(request(), POLICY, PROMPT, "D", SHA)               # another configuration of the ablation
    reasoner.decide(request(case_id="case-2"), POLICY, PROMPT, "C", SHA)  # another case
    assert len(stub.calls) == 3


def test_the_one_call_survives_a_restart_and_records_what_the_issue_asks_for(tmp_path):
    path, first = tmp_path / "decisions.sqlite", Stub()
    with s.DecisionStore.open(path) as store:
        decision_id, made = ai.decide_once(ai.AIReasoner(first), store, request(), POLICY, PROMPT, "C", SHA)
    with s.DecisionStore.open(path) as store:
        second = Stub()
        with pytest.raises(ai.AlreadyCalled):
            ai.decide_once(ai.AIReasoner(second), store, request(count=1), POLICY, PROMPT, "C", SHA)
        assert second.calls == [] and store.get(decision_id)[1] == made
        (row,) = store._query("SELECT * FROM decisions")
        assert (row["model_id"], row["prompt_sha256"]) == ("model-stub", PROMPT.sha256)
        assert json.loads(row["decision_json"])["reasons"][-1]["text"] == first.text


@pytest.mark.parametrize("change, error", [
    (lambda r: dataclasses.replace(r, case=dataclasses.replace(r.case, split="held_out")), s.LeakageError),
    (lambda r: dataclasses.replace(r, source="robot", input_source="perceived", observation=dataclasses.replace(
        r.observation, bundle=c.Bundle("a.zip", "a" * 64))), s.DecisionStoreError),
])
def test_no_call_is_spent_on_a_request_the_store_would_refuse(store, change, error):
    first = request()
    store.record(first, p.decide(first, POLICY, "E", SHA))
    stub = Stub()
    with pytest.raises(error):
        ai.decide_once(ai.AIReasoner(stub), store, change(request(count=1)), POLICY, PROMPT, "C", SHA)
    assert stub.calls == []


def test_a_call_whose_record_failed_is_still_spent(store, monkeypatch):
    stub = Stub()
    reasoner = ai.AIReasoner(stub)
    monkeypatch.setattr(store, "record", lambda *a: (_ for _ in ()).throw(s.DecisionStoreError("disk full")))
    with pytest.raises(s.DecisionStoreError, match="disk full"):
        ai.decide_once(reasoner, store, request(), POLICY, PROMPT, "C", SHA)
    with pytest.raises(ai.AlreadyCalled):
        ai.decide_once(reasoner, store, request(), POLICY, PROMPT, "C", SHA)
    assert len(stub.calls) == 1


def test_two_threads_on_one_case_make_one_call(tmp_path):
    stub, refused = Stub(), []
    reasoner = ai.AIReasoner(stub)
    start = threading.Barrier(4)

    def work():
        start.wait()
        with s.DecisionStore.open(tmp_path / "decisions.sqlite") as store:
            try:
                ai.decide_once(reasoner, store, request(), POLICY, PROMPT, "C", SHA)
            except ai.AlreadyCalled:
                refused.append(True)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert (len(stub.calls), len(refused)) == (1, 3)


def test_extra_reasons_are_recorded_with_the_decision(store):
    note = c.Reason("LEARNING", "tagged")
    decision_id, made = ai.decide_once(ai.AIReasoner(Stub()), store, request(), POLICY, PROMPT, "C", SHA, (note,))
    assert made.reasons[-1] == note and store.get(decision_id)[1] == made


# ---------------------------------------------------------------- F: E is a refusal-only veto

@pytest.mark.parametrize("req, refusal, rule", [
    (not_ready(request()), "cell_not_ready", "R1"),
    (request(height_mm=c.Measurement(90.0, 0.5)), "spec_unknown", "R7"),
    (with_twin(request(), left_tray=False), "beyond_capability", "R9"),
    (with_twin(request(), right_tray=True), "insufficient_evidence", "R9"),
])
def test_when_e_refuses_f_is_that_refusal_and_no_call_is_made(req, refusal, rule):
    stub = Stub(reply("accept"))
    decision = run(req, "F", stub)
    assert (decision.refusal, rules_of(decision)) == (refusal, [rule])
    assert stub.calls == [] and "the AI reasoner was not called" in decision.reasons[0].text
    assert (decision.model_id, decision.policy.prompt_sha256, decision.confidence) == (
        "model-stub", PROMPT.sha256, None)


@pytest.mark.parametrize("answer", COMMITTING)
def test_when_e_does_not_commit_f_refuses_what_the_ai_would_do(answer):
    req = blurred(request())
    assert p.decide(req, POLICY, "E", SHA).disposition == "remeasure"
    assert run(req, "D", Stub(reply(answer))).disposition == answer
    decision = run(req, "F", Stub(reply(answer)))
    assert (decision.disposition, decision.refusal, decision.confidence) == (None, "insufficient_evidence", None)
    assert rules_of(decision) == ["R2", "AI", ai.RAW_REPLY_RULE_ID] and "E veto" in decision.reasons[0].text


@pytest.mark.parametrize("answer", ["remeasure", "insufficient_evidence", "spec_unknown"])
def test_when_e_does_not_commit_an_ai_abstention_stands(answer):
    d, f = (run(blurred(request()), name, Stub(reply(answer))) for name in ("D", "F"))
    assert (f.disposition, f.refusal, f.remeasure_route) == (d.disposition, d.refusal, d.remeasure_route)


def test_a_disagreement_between_committed_dispositions_is_not_a_veto():
    assert p.decide(request(), POLICY, "E", SHA).disposition == "accept"
    assert run(request(), "F", Stub(reply("reject"))).disposition == "reject"


REQUESTS = [c.loads_request(path.read_text()) for path in sorted(FIXTURES.glob("*.json"))]
REQUESTS += [blurred(request()), not_ready(request()), with_twin(request(), right_tray=False, left_tray=True)]


@pytest.mark.parametrize("answer", list(ai.ANSWERS) + ["nonsense"])
def test_across_every_fixture_f_is_either_d_or_a_refusal(answer):
    text = reply(answer)
    for req in REQUESTS:
        d, f = (run(req, name, Stub(text)) for name in ("D", "F"))
        same = (f.disposition, f.refusal, f.remeasure_route) == (d.disposition, d.refusal, d.remeasure_route)
        assert same or (f.disposition is None and f.refusal is not None), req.request_id


# ---------------------------------------------------------------- the twin gate in D and F

@pytest.mark.parametrize("configuration", ["D", "F"])
@pytest.mark.parametrize("paths, refusal", [({"right_tray": False, "left_tray": True}, "beyond_capability"),
                                            ({"left_tray": True}, "insufficient_evidence")])
def test_the_twin_gate_checks_the_ais_destination(configuration, paths, refusal):
    decision = run(with_twin(request(), **paths), configuration, Stub(reply("reject")))
    assert (decision.disposition, decision.refusal, decision.destination) == (None, refusal, "none")
    assert rules_of(decision) == ["R9", "AI", ai.RAW_REPLY_RULE_ID] and decision.confidence is None


@pytest.mark.parametrize("configuration", ["D", "F"])
def test_without_a_twin_block_the_check_is_skipped_and_recorded(configuration):
    decision = run(request(), configuration, Stub(reply("accept")))
    assert decision.disposition == "accept" and rules_of(decision) == ["AI", "R9", ai.RAW_REPLY_RULE_ID]
    assert "skipped" in decision.reasons[1].text


def test_c_has_no_twin_gate():
    decision = run(with_twin(request(), right_tray=False), "C", Stub(reply("reject")))
    assert (decision.disposition, rules_of(decision)) == ("reject", ["AI", ai.RAW_REPLY_RULE_ID])


# ---------------------------------------------------------------- off by default, and the key

def test_the_reasoner_is_off_unless_switched_on_and_given_a_key(monkeypatch, caplog):
    assert ai.build_reasoner() is None and "is off" in ai.off_reason()
    monkeypatch.setenv(ai.KEY_ENV, SECRET)
    assert ai.build_reasoner() is None                                  # a key alone switches nothing on
    for value in ("0", "false", "", "maybe"):
        monkeypatch.setenv(ai.ENABLE_ENV, value)
        assert ai.build_reasoner() is None
    monkeypatch.setenv(ai.ENABLE_ENV, "1")
    monkeypatch.delenv(ai.KEY_ENV)
    with caplog.at_level(logging.DEBUG):
        assert ai.build_reasoner() is None
    assert ai.KEY_ENV in ai.off_reason()
    monkeypatch.setenv(ai.KEY_ENV, SECRET)
    built = ai.build_reasoner()
    assert built.model == ai.DEFAULT_MODEL == "claude-opus-5-5"
    monkeypatch.setenv(ai.MODEL_ENV, "claude-sonnet-5-5")
    assert ai.build_reasoner().model == "claude-sonnet-5-5"
    assert SECRET not in caplog.text and SECRET not in repr(built.provider) + repr(vars(built.provider))


class _Block:
    def __init__(self, type, text=""):
        self.type, self.text = type, text


class _Client:
    """Stands in for anthropic.Anthropic: records what it was given."""
    made = []

    def __init__(self, **kwargs):
        self.options, self.requests, self.closed = kwargs, [], False
        self.messages = self
        _Client.made.append(self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        response = type("Response", (), {})()
        response.content = [_Block("thinking", "not part of the reply"), _Block("text", reply())]
        response.stop_reason = "end_turn"
        return response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


def test_the_provider_sends_one_request_with_no_retry_and_holds_no_key(monkeypatch, caplog):
    import anthropic

    _Client.made = []
    monkeypatch.setattr(anthropic, "Anthropic", _Client)
    monkeypatch.setenv(ai.KEY_ENV, SECRET)
    provider = ai.AnthropicProvider()
    with caplog.at_level(logging.DEBUG):
        decision = ai.AIReasoner(provider).decide(request(), POLICY, PROMPT, "C", SHA)
    (client,) = _Client.made
    assert client.options == {"api_key": SECRET, "max_retries": 0, "timeout": ai.DEFAULT_TIMEOUT_S}
    (sent,) = client.requests
    assert (sent["model"], sent["max_tokens"]) == ("claude-opus-5-5", ai.MAX_TOKENS)
    assert sent["output_config"] == {"effort": "medium",
                                     "format": {"type": "json_schema", "schema": ai.REPLY_SCHEMA}}
    assert set(sent) == {"model", "max_tokens", "system", "messages", "output_config"}  # no sampling, no budget
    assert json.loads(sent["messages"][0]["content"]) == ai.model_input(request(), "C")
    assert client.closed and decision.disposition == "accept" and decision.model_id == "claude-opus-5-5"
    for text in (caplog.text, c.dumps(decision), repr(vars(provider)), sent["system"], str(sent["messages"])):
        assert SECRET not in text


def test_without_a_key_the_provider_makes_no_client(monkeypatch):
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: pytest.fail("a client was made"))
    with pytest.raises(ai.ProviderError, match="no API key"):
        ai.AnthropicProvider().complete("system", "user")


def test_the_installed_sdk_takes_the_parameters_the_provider_sends():
    from anthropic.resources.messages import Messages

    assert {"model", "max_tokens", "system", "messages", "output_config"} <= set(
        inspect.signature(Messages.create).parameters)
