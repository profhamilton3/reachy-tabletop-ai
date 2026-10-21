"""Voice -> sim panel bridge, against a fake panel on loopback (issue #29). No external network."""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from reachy_ai.audio.panel_bridge import PanelClient, PanelError, VoicePanelBridge, panel_url
from reachy_ai.audio.voice_commands import build_grammar, degraded_grammar

COOKIE = "reachy_panel_session"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class FakePanel:
    """Enforces the real panel's Origin and cookie rules and records every call."""

    def __init__(self):
        self.calls = []
        self.task = None
        self.n = 0

    def task_dict(self, state, **extra):
        d = {"task_id": "t1", "state": state, "version": self.n, "detail": "", "question_id": "",
             "events": [], "proposal": None, "mode": "planning_only", "cancel_requested": False}
        d.update(extra)
        return d


def make_handler(panel):
    class Handler(BaseHTTPRequestHandler):
        log_message = lambda *a: None  # noqa: E731

        def _send(self, status, payload, cookie=False):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if cookie:
                self.send_header("Set-Cookie", f"{COOKIE}=abc123; Path=/; HttpOnly; SameSite=Strict")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            panel.calls.append(("GET", self.path, None))
            if self.path == "/":
                self._send(200, {}, cookie=True)
            elif self.path == "/capabilities":
                self._send(200, {"abilities": [{"name": "lift_object", "summary": "pick up an object"}]})
            elif self.path == "/scene":
                self._send(200, {"objects": ["red_cube"], "cells": [{"name": "r1c1"}]})
            elif self.path == "/tasks/t1":
                self._send(200, panel.task)
            else:
                self._send(404, {"error": "nope"})

        def do_POST(self):  # noqa: N802
            host, origin = self.headers.get("Host"), self.headers.get("Origin", "")
            if origin.split("://", 1)[-1] != host:
                return self._send(403, {"error": "Cross-origin request refused.", "code": "bad_origin"})
            if COOKIE + "=abc123" not in self.headers.get("Cookie", ""):
                return self._send(403, {"error": "No panel session.", "code": "no_session"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            panel.calls.append(("POST", self.path, body))
            panel.n += 1
            if self.path == "/tasks":
                text = body["text"]
                if "cube" in text:
                    panel.task = panel.task_dict(
                        "awaiting_confirmation",
                        proposal={"plan_id": "p1", "plan_version": 3, "task_type": "lift_object",
                                  "summary": "lift the red cube and put it back", "brief_reason": "reachable"})
                elif "joint" in text:
                    panel.task = panel.task_dict("failed", detail="I will not move joints.")
                elif "which" in text:
                    panel.task = panel.task_dict(
                        "needs_clarification", question_id="q1",
                        events=[{"role": "reachy", "text": "Which cube?", "question_id": "q1",
                                 "choices": ["red", "blue"]}])
                else:
                    panel.task = panel.task_dict("failed", detail="not understood")
                return self._send(202, panel.task)
            if self.path == "/tasks/t1/reply":
                panel.task = panel.task_dict("awaiting_confirmation", proposal={
                    "plan_id": "p2", "plan_version": 1, "task_type": "lift_object", "summary": "lift red"})
                return self._send(202, panel.task)
            if self.path == "/tasks/t1/confirm":
                panel.task = panel.task_dict("executing")
                return self._send(200, panel.task)
            if self.path == "/tasks/t1/cancel":
                panel.task = panel.task_dict("cancelled", detail="Cancelled.")
                return self._send(200, panel.task)
            self._send(404, {"error": "nope"})

    return Handler


@pytest.fixture
def panel():
    fake = FakePanel()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(fake))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    fake.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield fake
    server.shutdown()
    server.server_close()


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def make_bridge(panel, **kw):
    lines = []
    clock = Clock()
    kw.setdefault("sleep", lambda s: None)
    bridge = VoicePanelBridge(PanelClient(panel.url, timeout_s=2), out=lines.append, clock=clock, **kw)
    return bridge, lines, clock


def posts(panel):
    return [(p, b) for m, p, b in panel.calls if m == "POST"]


def test_panel_bridge_url_default_and_env(monkeypatch):
    monkeypatch.delenv("REACHY_PANEL_URL", raising=False)
    assert panel_url() == "http://127.0.0.1:8080"
    monkeypatch.setenv("REACHY_PANEL_URL", "http://example.test:9/")
    assert panel_url() == "http://example.test:9"


def test_panel_bridge_vocabulary_builds_grammar(panel):
    abilities, objects, cells = PanelClient(panel.url).vocabulary()
    grammar = build_grammar(abilities, objects, cells)
    assert "lift" in grammar and "cube" in grammar and "two" in grammar


def test_panel_bridge_no_wake_word_sends_nothing(panel):
    bridge, lines, _ = make_bridge(panel)
    assert bridge.handle("pick up the cube and put it back", 0.95) == "ignored"
    assert bridge.handle("move the cube", 0.95) == "ignored"
    assert bridge.handle("yes", 0.95) == "ignored"
    assert panel.calls == []


def test_panel_bridge_low_confidence_sends_nothing(panel):
    bridge, lines, _ = make_bridge(panel)
    assert bridge.handle("reachy pick up the cube", 0.59) == "low_confidence"
    assert bridge.handle("reachy [unk] cube", 0.99) == "low_confidence"
    assert lines == ["didn't catch that"] * 2
    assert panel.calls == []


def test_panel_bridge_motion_request_needs_yes(panel):
    bridge, lines, _ = make_bridge(panel)
    assert bridge.handle("reachy pick up the cube and put it back", 0.9) == "sent"
    # sent exactly as typed text, with the wake word removed
    assert posts(panel)[0][0] == "/tasks"
    assert posts(panel)[0][1]["text"] == "pick up the cube and put it back"
    assert bridge.pending
    assert not any(p.endswith("/confirm") for p, _ in posts(panel))
    assert any("lift the red cube" in line for line in lines)
    assert bridge.handle("reachy yes", 0.9) == "confirmed"
    assert posts(panel)[-1] == ("/tasks/t1/confirm", {"plan_id": "p1", "plan_version": 3})
    assert any("executing" in line for line in lines)


def test_panel_bridge_yes_without_proposal_confirms_nothing(panel):
    bridge, lines, _ = make_bridge(panel)
    assert bridge.handle("reachy yes", 0.9) == "nothing_pending"
    assert panel.calls == []


def test_panel_bridge_no_declines(panel):
    bridge, _, _ = make_bridge(panel)
    bridge.handle("reachy pick up the cube", 0.9)
    assert bridge.handle("reachy no", 0.9) == "declined"
    assert posts(panel)[-1][0] == "/tasks/t1/cancel"
    assert not bridge.pending


def test_panel_bridge_silence_declines_after_timeout(panel):
    bridge, _, clock = make_bridge(panel, confirm_timeout_s=15)
    bridge.handle("reachy pick up the cube", 0.9)
    clock.t = 14.9
    assert bridge.expire() is None and bridge.pending
    clock.t = 15.0
    assert bridge.expire() == "expired"
    assert posts(panel)[-1][0] == "/tasks/t1/cancel"
    assert bridge.handle("reachy yes", 0.9) == "nothing_pending"


def test_panel_bridge_changed_plan_is_not_confirmed(panel):
    bridge, lines, _ = make_bridge(panel)
    bridge.handle("reachy pick up the cube", 0.9)
    panel.task["proposal"]["plan_version"] = 4  # board changed under the proposal
    assert bridge.handle("reachy yes", 0.9) == "stale"
    assert not any(p.endswith("/confirm") for p, _ in posts(panel))


@pytest.mark.parametrize("phrase", ["stop", "reachy stop"])
def test_panel_bridge_stop_cancels_with_no_confirmation_while_pending(panel, phrase):
    bridge, _, _ = make_bridge(panel)
    bridge.handle("reachy pick up the cube", 0.9)
    assert bridge.pending
    assert bridge.handle(phrase, 0.1) == "stop"  # not held back by the confidence setting
    assert posts(panel)[-1][0] == "/tasks/t1/cancel"
    assert not any(p.endswith("/confirm") or p == "/tasks" and "stop" in b["text"] for p, b in posts(panel))
    assert not bridge.pending


def test_panel_bridge_stop_is_never_a_request(panel):
    bridge, lines, _ = make_bridge(panel)
    assert bridge.handle("stop", 0.9) == "nothing_to_stop"
    assert posts(panel) == []


def test_panel_bridge_refusal_and_clarification_are_shown(panel):
    bridge, lines, _ = make_bridge(panel)
    bridge.handle("reachy rotate the joint", 0.9)
    assert any(line.startswith("refused: I will not move joints") for line in lines)
    assert not bridge.pending
    bridge.handle("reachy which one", 0.9)
    assert any("needs clarification: Which cube?" in line for line in lines)
    assert bridge.handle("reachy the red cube", 0.9) == "replied"
    assert posts(panel)[-1] == ("/tasks/t1/reply", {"question_id": "q1", "text": "the red cube"})


def test_panel_bridge_dry_run_sends_nothing(panel):
    bridge, lines, _ = make_bridge(panel, dry_run=True)
    assert bridge.handle("reachy pick up the cube and put it back", 0.9) == "dry_run"
    assert bridge.handle("stop", 0.9) == "dry_run"
    assert panel.calls == []
    assert '[dry run] would POST /tasks {"text": "pick up the cube and put it back"}' in lines


def test_panel_bridge_unreachable_is_explicit_and_only_stop(panel):
    client = PanelClient("http://127.0.0.1:9", timeout_s=1)  # nothing listens on the discard port
    with pytest.raises(PanelError, match="unreachable"):
        client.vocabulary()
    assert "stop" in degraded_grammar() and "cube" not in degraded_grammar()
    lines = []
    bridge = VoicePanelBridge(client, degraded=True, out=lines.append)
    assert bridge.handle("reachy pick up the cube", 0.9) == "degraded"
    assert bridge.handle("reachy yes", 0.9) == "degraded"
    assert lines == ["panel unreachable: only 'stop' is accepted"] * 2
    assert bridge.handle("stop", 0.9) == "nothing_to_stop"


def test_panel_bridge_error_has_no_retry(panel):
    bridge, lines, _ = make_bridge(panel)
    panel.url = "http://127.0.0.1:9"
    bridge.client = PanelClient(panel.url, timeout_s=1)
    assert bridge.handle("reachy pick up the cube", 0.9) == "error"
    assert lines[-1].startswith("panel error: ") and "unreachable" in lines[-1]


def test_panel_bridge_origin_and_cookie_are_sent(panel):
    # The fake enforces the real panel's rules; a bare POST without a session is refused.
    import urllib.error
    import urllib.request
    req = urllib.request.Request(panel.url + "/tasks", data=b'{"text":"x"}', method="POST",
                                 headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req, timeout=2)
    assert err.value.code == 403
    assert PanelClient(panel.url).submit("pick up the cube")["state"] == "awaiting_confirmation"


# --- Reachy speaks the replies (#59 part 2): --speak and the self-hearing filter -----------------------------------

class FakeSpeaker:
    """Records what would be said; `talking` is the self-hearing flag. `hush` notes how many panel POSTs came first."""

    def __init__(self, panel):
        self.panel, self.said, self.hushes, self.talking = panel, [], [], False

    def say(self, text):
        self.said.append(text)

    def speaking(self):
        return self.talking

    def hush(self):
        self.hushes.append(len(posts(self.panel)))
        return {"ok": True}


def test_panel_bridge_voice_speaks_the_proposal_and_the_refusal(panel):
    speaker = FakeSpeaker(panel)
    bridge, lines, _ = make_bridge(panel, speaker=speaker)
    bridge.handle("reachy pick up the cube and put it back", 0.9)
    assert speaker.said == ["I'd lift the red cube and put it back. Say Reachy, yes to go ahead."]
    assert not any(p.endswith("/confirm") for p, _ in posts(panel))  # speaking never confirms
    bridge.handle("reachy rotate the joint", 0.9)
    assert speaker.said[-1] == "I can't do that: I will not move joints."


def test_panel_bridge_voice_ignores_all_but_stop_while_speaking(panel, caplog):
    speaker = FakeSpeaker(panel)
    bridge, _, _ = make_bridge(panel, speaker=speaker)
    bridge.handle("reachy pick up the cube", 0.9)
    speaker.talking = True  # Reachy is saying "... Say Reachy, yes to go ahead."
    with caplog.at_level("INFO", logger="reachy_ai.audio.panel_bridge"):
        assert bridge.handle("reachy yes", 0.95) == "ignored_while_speaking"
        assert bridge.handle("reachy no", 0.95) == "ignored_while_speaking"
        assert bridge.handle("reachy pick up the cube", 0.95) == "ignored_while_speaking"
    assert "ignored while speaking: 'reachy yes'" in caplog.text
    assert bridge.pending and posts(panel)[-1][0] == "/tasks"  # no confirm, no cancel, no new request
    before = len(posts(panel))
    assert bridge.handle("stop", 0.2) == "stop"
    assert speaker.hushes == [before]  # hush first ...
    assert posts(panel)[-1][0] == "/tasks/t1/cancel"  # ... then the panel's cancel
    assert speaker.said[-1] == "Stopping."
    speaker.talking = False
    assert bridge.handle("reachy yes", 0.95) == "nothing_pending"  # heard again once Reachy is quiet


def test_panel_bridge_voice_stop_hushes_even_with_nothing_to_stop(panel):
    speaker = FakeSpeaker(panel)
    speaker.talking = True
    bridge, _, _ = make_bridge(panel, speaker=speaker)
    assert bridge.handle("reachy stop", 0.9) == "nothing_to_stop"
    assert speaker.hushes == [0] and speaker.said == []


def test_panel_bridge_voice_speak_is_off_by_default(panel):
    import voice_panel
    assert voice_panel.build_parser().parse_args(["--mic"]).speak is False
    assert voice_panel.build_parser().parse_args(["--mic", "--speak"]).speak is True
    bridge, _, _ = make_bridge(panel)
    assert bridge.speaker is None
    bridge.handle("reachy pick up the cube", 0.9)
    assert bridge.handle("reachy yes", 0.9) == "confirmed"  # no speaker, no filter


def test_panel_bridge_voice_stop_cuts_real_speech_through_the_helper(panel):
    import reachy_audio_helper as helper

    from reachy_ai.audio.audio_client import AudioClient
    from reachy_ai.audio.speech import Speaker

    backend = helper.FakeBackend(playback_s=30)  # a long (silent) playback subprocess
    server = helper.make_server(0, backend)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        audio = AudioClient(f"http://127.0.0.1:{server.server_address[1]}")
        speaker = Speaker(audio, tail_s=0)
        bridge, _, _ = make_bridge(panel, speaker=speaker)
        bridge.handle("reachy pick up the cube", 0.9)
        deadline = time.monotonic() + 5
        while not audio.status()["speaking"]:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert bridge.handle("reachy yes", 0.95) == "ignored_while_speaking"
        backend.playback_s = None  # the proposal's 30 s playback is running; "Stopping." itself is short
        started = time.monotonic()
        assert bridge.handle("stop", 0.9) == "stop"
        assert posts(panel)[-1][0] == "/tasks/t1/cancel" and time.monotonic() - started < 3
        speaker._thread.join(5)
        assert speaker.last["text"] == "Stopping." and time.monotonic() - started < 5
        assert ("say", "I'd lift the red cube and put it back. Say Reachy, yes to go ahead.") in backend.calls
    finally:
        server.shutdown()
        server.server_close()
