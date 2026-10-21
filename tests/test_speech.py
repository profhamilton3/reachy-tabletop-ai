"""What Reachy says (reachy_ai.audio.speech, #59 part 2). Offline: the helper is the fake backend on 127.0.0.1, or
a stub client; nothing plays."""
import ast
import dataclasses
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from reachy_ai.audio import speech
from reachy_ai.audio.audio_client import AudioClient, AudioHelperError
from reachy_ai.audio.speech import Speaker, speak, spoken_proposal, spoken_refusal, spoken_t3
from reachy_ai.games import tictactoe as t3

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import reachy_audio_helper as helper  # noqa: E402


def said(board):
    return spoken_t3(t3.recommend(t3.parse(board)))


# --- spoken_t3: every result kind -----------------------------------------------------------------------------------

@pytest.mark.parametrize("board, rule, text", [
    ("XX./OO./...", 1, "It's X's turn. Place a cube on row one, column three, to win on row one."),
    ("O../O.X/XX.", 2, "It's O's turn. Place a cylinder on row three, column three, to block X's row three."),
    ("XXO/O../X..", 3, "It's O's turn. Place a cylinder on row two, column three, to make two threats at once."),
    (".../.../.OX", 3, "It's X's turn. Place a cube on row one, column three. That forces O's reply, then sets up "
                       "a fork."),
    (".../..O/.OX", 4, "It's X's turn. Place a cube in the centre, row two, column two, to stop O setting up a "
                       "fork."),
    (".../.../...", 5, "It's X's turn. Place a cube in the centre, row two, column two. That's the centre rule."),
    ("O../.X./...", 6, "It's X's turn. Place a cube on row three, column three. That's the opposite corner rule."),
    ("X../.O./...", 7, "It's X's turn. Place a cube on row one, column three. That's the corner rule."),
    ("XXO/OOX/X.O", 8, "It's X's turn. Place a cube on row three, column two. That's the side rule."),
    (".../..X/O..", 9, "It's X's turn. Place a cube on row three, column three. I checked every move ahead: the usual "
                       "rule would miss the win."),
])
def test_speech_t3_every_rule(board, rule, text):
    result = t3.recommend(t3.parse(board))
    assert result.rule_number == rule
    assert spoken_t3(result) == text


def test_speech_t3_rule_9_give_up_the_draw(monkeypatch):
    # as in the engine's own test: no legal board reaches it, so force a losing rule-of-thumb move
    monkeypatch.setattr(t3, "_candidates", lambda board, me: [(t3.CELLS.index("r1c3"), 5, "centre", (), "x")])
    result = t3.recommend(t3.parse("X../.O./..X"))
    assert result.search == ("draw", "r1c3", 5)
    assert spoken_t3(result) == ("It's O's turn. Place a cylinder on row one, column two. I checked every move "
                                 "ahead: the usual rule would give up the draw.")


@pytest.mark.parametrize("board, text", [
    ("XXX/OO./...", "X has won, on row one."),
    ("OX./OX./O.X", "O has won, on column one."),
    ("X.O/.X./O.X", "X has won, on the diagonal."),
    ("XOX/XOO/OXX", "It's a draw."),
])
def test_speech_t3_win_and_draw(board, text):
    assert said(board) == text


@pytest.mark.parametrize("board, text", [
    ("XXX/XO./O..", "I can't analyse this board: there are four cubes and two cylinders."),
    ("XX./X../...", "I can't analyse this board: there are three cubes and no cylinders."),
    ("#../.../...", "I can't analyse this board: there's something that isn't a game piece on row one, column one."),
    ("#.#/.../...", "I can't analyse this board: there are things that aren't game pieces on two squares."),
    ("?../.../...", "I can't analyse this board: row one, column one is still unknown."),
    ("XXX/OOO/...", "I can't analyse this board: both X and O have three in a row."),
    ("XXX/OO./OO.", "I can't analyse this board: play went on after X won."),
    ("#XX/XX./...", "I can't analyse this board: there's something that isn't a game piece on row one, column one, "
                    "and there are four cubes and no cylinders."),
])
def test_speech_t3_invalid_boards_say_why(board, text):
    assert said(board) == text


def test_speech_t3_is_built_from_the_fields_never_from_the_printed_text():
    for board in ("O../O.X/XX.", "XXX/OO./...", "XXX/XO./O..", ".../..X/O..", "XOX/XOO/OXX"):
        result = t3.recommend(t3.parse(board))
        garbled = dataclasses.replace(result, text="nonsense", reasons=("nonsense",))
        assert spoken_t3(garbled) == spoken_t3(result)
        assert "nonsense" not in spoken_t3(garbled)


def test_speech_t3_engine_reports_problems_and_search_as_data():
    invalid = t3.recommend(t3.parse("#XX/XX./..."))
    assert invalid.problems == (("other", ("r1c1",)), ("counts", 4, 0))
    assert invalid.reasons == ("non-game object on r1c1", "counts: 4 X and 0 O differ by more than 1")
    assert invalid.to_dict()["problems"] == [["other", ["r1c1"]], ["counts", 4, 0]]
    search = t3.recommend(t3.parse(".../..X/O.."))
    assert search.search == ("win", "r2c2", 5)
    assert search.to_dict()["search"] == {"verdict": "win", "overridden_cell": "r2c2", "overridden_rule": 5}
    assert t3.recommend(t3.parse(".../.../...")).to_dict()["search"] is None


def test_speech_t3_every_reachable_position_fits_the_helper_limit():
    seen, stack = set(), [t3.parse(".../.../...")]
    while stack:
        board = stack.pop()
        if board in seen:
            continue
        seen.add(board)
        result = t3.recommend(board)
        text = spoken_t3(result)
        assert 0 < len(text) <= speech.MAX_SPOKEN and text.isprintable()
        if result.outcome == "in_progress":
            stack += [t3._play(board, i, result.turn) for i in t3._empties(board)]
    assert len(seen) > 5000
    # the longest invalid case still fits
    assert len(said("?#?/#?#/XXX")) <= speech.MAX_SPOKEN


# --- the panel's replies -------------------------------------------------------------------------------------------

def test_speech_panel_short_forms():
    assert spoken_proposal("pick up the cube and put it back") == ("I'd pick up the cube and put it back. "
                                                                   "Say Reachy, yes to go ahead.")
    assert spoken_refusal("I will not move joints.") == "I can't do that: I will not move joints."
    assert speech.STOPPING == "Stopping."
    long = spoken_refusal("the cell r3c3 is out of reach " * 20)
    assert len(long) <= speech.MAX_SPOKEN and long.endswith(".") and long.startswith("I can't do that: the cell")
    assert len(spoken_proposal("lift\nthe   cube " * 40)) <= speech.MAX_SPOKEN


# --- speak(): through the helper, never raising --------------------------------------------------------------------

@pytest.fixture
def helper_client():
    servers = []

    def start(backend, **kwargs):
        server = helper.make_server(0, backend, **kwargs)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return AudioClient(f"http://127.0.0.1:{server.server_address[1]}")

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def test_speech_speak_says_the_text_through_the_helper(helper_client, caplog):
    backend = helper.FakeBackend()
    client = helper_client(backend)
    with caplog.at_level("INFO", logger="reachy_ai.audio.speech"):
        outcome = speak("It's a draw.", client=client)
    assert outcome["ok"] and outcome["spoken"] and not outcome["hushed"] and outcome["error"] is None
    assert backend.calls == [("say", "It's a draw.")]
    assert "speak: \"It's a draw.\"" in caplog.text


def test_speech_speak_reports_mute(helper_client):
    outcome = speak("It's a draw.", client=helper_client(helper.FakeBackend(), muted=True))
    assert outcome["ok"] and outcome["muted"] and not outcome["spoken"]


def test_speech_speak_never_raises_and_returns_the_reason(helper_client):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    outcome = speak("It's a draw.", client=AudioClient(f"http://127.0.0.1:{port}"))
    assert not outcome["ok"] and outcome["error"] == "unreachable" and "REACHY_AUDIO_URL" in outcome["detail"]
    outcome = speak("It's a draw.", client=helper_client(helper.FakeBackend(tts=None)))
    assert outcome["error"] == "unavailable"

    class Broken:
        def say(self, text, timeout_s):
            raise ValueError("boom")

    assert speak("x", client=Broken())["error"] == "failed"


def test_speech_speak_has_a_timeout():
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(1)
        started = time.monotonic()
        outcome = speak("It's a draw.", client=AudioClient(f"http://127.0.0.1:{silent.getsockname()[1]}"),
                        timeout_s=0.3)
    assert outcome["error"] == "timeout" and time.monotonic() - started < 3


# --- Speaker: the self-hearing flag --------------------------------------------------------------------------------

class StubClient:
    def __init__(self):
        self.calls = []
        self.release = threading.Event()

    def say(self, text, timeout_s):
        self.calls.append(("say", text))
        self.release.wait(5)
        return {"ok": True, "hushed": False}

    def hush(self, timeout_s=1.0):
        self.calls.append(("hush",))
        self.release.set()
        return {"ok": True, "stopped": True}


def test_speech_speaker_flag_covers_the_speech_and_its_tail():
    now = [100.0]
    client = StubClient()
    speaker = Speaker(client, tail_s=1.5, clock=lambda: now[0])
    assert not speaker.speaking()
    thread = speaker.say("I'd lift the cube. Say Reachy, yes to go ahead.")
    assert speaker.speaking()  # up at once, before the helper answers
    client.release.set()
    thread.join(5)
    assert speaker.speaking()  # the tail: the recogniser's late result for its own words
    now[0] += 1.6
    assert not speaker.speaking()
    assert speaker.last["ok"]


def test_speech_speaker_new_phrase_hushes_the_old_one():
    client = StubClient()
    speaker = Speaker(client, tail_s=0)
    first = speaker.say("I'd lift the cube. Say Reachy, yes to go ahead.")
    time.sleep(0.05)
    second = speaker.say("Stopping.")
    second.join(5)
    first.join(5)
    assert client.calls[0] == ("say", "I'd lift the cube. Say Reachy, yes to go ahead.")
    assert ("hush",) in client.calls and client.calls[-1] == ("say", "Stopping.")


def test_speech_speaker_hush_never_raises():
    class Down:
        def hush(self, timeout_s=1.0):
            raise AudioHelperError("unreachable", "unreachable")

    assert Speaker(Down()).hush()["error"] == "unreachable"


# --- import guard ---------------------------------------------------------------------------------------------------

def test_speech_import_guard_no_robot_imports():
    tree = ast.parse((ROOT / "src" / "reachy_ai" / "audio" / "speech.py").read_text())
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    forbidden = ("reachy_sdk", "reachy2_sdk", "reachy_ai.motion", "reachy_ai.reachy_client", "reachy_ai.stereo")
    assert names and not [name for name in names if name.startswith(forbidden)]
