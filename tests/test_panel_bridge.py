"""Voice -> sim panel bridge, against a fake panel on loopback (issue #29). No external network."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from reachy_ai.audio.panel_bridge import PanelClient, PanelError, VoicePanelBridge, panel_url
from reachy_ai.audio.voice_commands import build_grammar, degraded_grammar

COOKIE = "reachy_panel_session"


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
