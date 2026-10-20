"""Voice commands: grammar, WAV recognition and the no-motion import guard (issue #29)."""
import ast
import wave
from pathlib import Path

import pytest

from reachy_ai.audio import voice_commands
from reachy_ai.audio.voice_commands import (
    DEFAULT_MODEL_PATH,
    build_grammar,
    degraded_grammar,
    normalize_utterance,
)

TESTS = Path(__file__).parent
SRC = TESTS.parent / "src" / "reachy_ai" / "audio"

ABILITIES = [
    {"name": "lift_object", "summary": "pick up an object, hold it about 5 cm up, and put it back "
                                       "(evaluated in simulation for the red cube)"},
    {"name": "place_on_cell", "summary": "pick an object up and set it down on another grid cell"},
    {"name": "stow_arm", "summary": "store your arm"},
]
OBJECTS = ["red_cube", "pool_cylinder_01"]
CELLS = ["r1c1", "r2c2"]


def test_voice_grammar_comes_from_the_served_lists():
    grammar = build_grammar(ABILITIES, OBJECTS, CELLS)
    for word in ("reachy", "yes", "no", "stop", "lift", "object", "stow", "cube", "red", "cylinder",
                 "pool", "grid", "cell", "r", "two", "[unk]"):
        assert word in grammar
    assert "simulation" not in grammar  # parenthetical operator detail is left out
    assert len(grammar) == len(set(grammar))


def test_voice_grammar_without_cells_has_no_cell_words():
    grammar = build_grammar(ABILITIES, OBJECTS, [])
    assert "two" not in grammar and "see" not in grammar


def test_voice_degraded_grammar_is_stop_only():
    grammar = degraded_grammar()
    assert "stop" in grammar
    assert not {"yes", "no", "cube", "move", "pick"} & set(grammar)


def test_voice_normalize_wake_alias_and_cells():
    assert normalize_utterance("richie move the cube to are two see three") == "reachy move the cube to r2c3"
    assert normalize_utterance("move the cube") == "move the cube"
    assert normalize_utterance("reach stop") == "reachy stop"


@pytest.fixture(scope="module")
def vosk_ready():
    pytest.importorskip("vosk")
    if not DEFAULT_MODEL_PATH.is_dir():
        pytest.skip("Vosk model not present")


def test_voice_command_wav_gives_expected_text(vosk_ready):
    grammar = build_grammar(ABILITIES, OBJECTS, CELLS)
    heard = list(voice_commands.recognize_wav(TESTS / "command.wav", grammar))
    assert [t for t, _ in heard] == ["move the cube"]
    assert 0.0 < heard[0][1] <= 1.0


def test_voice_multichannel_wav_uses_channel_zero(vosk_ready):
    grammar = build_grammar(ABILITIES, OBJECTS, CELLS)
    with wave.open(str(TESTS / "reachy_test.wav")) as wf:
        assert wf.getnchannels() == 6
    heard = list(voice_commands.recognize_wav(TESTS / "reachy_test.wav", grammar))
    assert not any(t.startswith("reachy ") for t, _ in heard)  # runs; says no command


def test_voice_missing_model_is_explicit(tmp_path):
    with pytest.raises(voice_commands.VoiceError, match="model not found"):
        voice_commands._model_path(tmp_path / "nope")


def _imports(path):
    tree = ast.parse(path.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


@pytest.mark.parametrize("module", ["voice_commands.py", "panel_bridge.py"])
def test_voice_import_guard_no_robot_imports(module):
    forbidden = ("reachy_sdk", "reachy2_sdk", "reachy_ai.motion", "reachy_ai.reachy_client")
    for name in _imports(SRC / module):
        assert not name.startswith(forbidden), f"{module} imports {name}"

