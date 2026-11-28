"""#83: scripts/reasoner_smoke.py. Offline: the provider is a stub and sockets are blocked. CI never runs the
script against the real API."""
import dataclasses
import importlib.util
import json
import socket
from pathlib import Path

import pytest

from reachy_ai.reasoner import ai
from reachy_ai.reasoner import contract as c
from reachy_ai.reasoner import policy as p
from reachy_ai.reasoner import store as s

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "disposition" / "valid" / "request"
SECRET = "sk-ant-test-SECRET-0123456789"

_spec = importlib.util.spec_from_file_location("reasoner_smoke", ROOT / "scripts" / "reasoner_smoke.py")
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


class Stub(ai.Provider):
    model = "model-stub"
    calls: list = []
    answer = "accept"

    def complete(self, system, user):
        Stub.calls.append(json.loads(user))
        return ai.Reply(json.dumps({"answer": Stub.answer, "confidence": 0.9, "rationale": "MODEL-PROSE"}),
                        "end_turn")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(ai, "AnthropicProvider", Stub)
    for name in (ai.ENABLE_ENV, ai.MODEL_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ai.KEY_ENV, SECRET)
    Stub.calls, Stub.answer = [], "accept"


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "decisions.sqlite"


def run(store_path, *args):
    return smoke.main(["--store", str(store_path), *map(str, args)])


def rows(store_path):
    with s.DecisionStore.open(store_path) as store:
        return store._query("SELECT r.case_id, r.split, d.configuration, d.disposition, d.refusal FROM decisions d "
                            "JOIN requests r USING (request_id) ORDER BY d.decision_id")


def held_out_file(tmp_path, name="spec_part220.json"):
    request = c.loads_request((FIXTURES / name).read_text())
    held = dataclasses.replace(request, case=dataclasses.replace(request.case, split="held_out"))
    path = tmp_path / "held.json"
    path.write_text(c.dumps(held))
    return path


def test_by_default_one_development_case_runs_through_c_and_is_recorded(store_path, capsys):
    assert run(store_path) == 0
    (row,) = rows(store_path)
    assert (row["split"], row["configuration"], row["disposition"]) == ("development", "C", "accept")
    assert row["case_id"].startswith("smoke-") and len(Stub.calls) == 1
    out = capsys.readouterr()
    assert '"disposition": "accept"' in out.out and "model-stub" in out.out
    assert SECRET not in out.out + out.err and "MODEL-PROSE" not in out.out


def test_two_cases_through_f_and_a_second_run_makes_new_cases(store_path):
    files = (FIXTURES / "spec_part221.json", FIXTURES / "spec_part110.json")
    for _ in range(2):  # a repeated trial needs a new case_id, and gets one
        assert run(store_path, "--configuration", "F", *files) == 0
    made = rows(store_path)
    assert [r["configuration"] for r in made] == ["F"] * 4 and len({r["case_id"] for r in made}) == 4
    assert len(Stub.calls) == 4


def test_a_held_out_case_is_refused_before_any_call_even_beside_a_development_one(store_path, tmp_path, capsys):
    assert run(store_path, FIXTURES / "spec_part221.json", held_out_file(tmp_path)) == 2
    assert Stub.calls == [] and not store_path.exists()
    assert "held_out" in capsys.readouterr().err


def test_a_case_the_store_holds_as_held_out_cannot_be_relabelled_for_the_smoke_run(store_path, tmp_path):
    held = c.loads_request(held_out_file(tmp_path).read_text())
    with s.DecisionStore.open(store_path) as store:
        store.record(held, p.decide(held, p.load_policy(), "E", "abcdef1"))
    assert run(store_path, FIXTURES / "spec_part220.json") == 2  # the same case_id, marked development
    assert Stub.calls == [] and len(rows(store_path)) == 1


@pytest.mark.parametrize("args, phrase", [
    ([FIXTURES / "spec_part110.json"] * 3, "1 or 2"),
    ([FIXTURES / "no_such_file.json"], "no_such_file"),
])
def test_bad_arguments_make_no_call(store_path, capsys, args, phrase):
    assert run(store_path, *args) == 2
    assert Stub.calls == [] and not store_path.exists() and phrase in capsys.readouterr().err


@pytest.mark.parametrize("configuration", ["A", "B", "D", "E"])
def test_only_c_and_f_are_offered(store_path, configuration):
    with pytest.raises(SystemExit):
        run(store_path, "--configuration", configuration)
    assert Stub.calls == []


def test_without_a_key_nothing_is_called_or_written(store_path, monkeypatch, capsys):
    monkeypatch.delenv(ai.KEY_ENV)
    assert run(store_path) == 2
    assert Stub.calls == [] and not store_path.exists() and ai.KEY_ENV in capsys.readouterr().err


def test_without_the_package_nothing_is_called_or_written(store_path, monkeypatch, capsys):
    monkeypatch.setattr(ai, "sdk_installed", lambda: False)
    assert run(store_path) == 2
    assert Stub.calls == [] and not store_path.exists() and "pip install -e .[ai]" in capsys.readouterr().err


def test_an_unusable_reply_is_recorded_and_fails_the_smoke_run(store_path):
    Stub.answer = "ship_it"
    assert run(store_path) == 1
    assert rows(store_path)[0]["refusal"] == "reasoner_invalid"


def test_the_script_reads_the_policy_and_changes_no_file(store_path):
    def hashes():
        return {f.name: f.read_bytes() for f in p.POLICY_DIR.iterdir()}

    before = hashes()
    assert run(store_path) == 0
    assert hashes() == before
    source = (ROOT / "scripts" / "reasoner_smoke.py").read_text()
    assert "cases_for_tuning" not in source and "freeze(" not in source


def test_ci_does_not_run_the_script():
    assert "reasoner_smoke" not in (ROOT / ".github" / "workflows" / "ci.yml").read_text()
