"""Offline voice commands: Vosk speech recognition with a constrained grammar.

Yields ``(text, confidence)`` for each recognised utterance, from a WAV file or
the local microphone. It produces TEXT only. It never imports ``reachy_sdk`` or
``reachy_ai.motion`` and has no way to move the robot; what happens to the text
is decided by ``panel_bridge`` and, behind it, the sim panel's own checks.

The Vosk logic is the notebook's (tests/TestingMicrophoneAudio.ipynb): a
``KaldiRecognizer`` given a JSON word list, fed 4000-frame chunks.

The grammar is built from what the panel serves (its ability list and the scene's
object and cell names), not from a hand-written second list. The few words that
are not served (the wake word, yes, no, stop, and the glue words of a spoken
sentence) are named constants below.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import wave
from array import array
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

WAKE_WORD = "reachy"
STOP_WORD = "stop"
YES_WORD = "yes"
NO_WORD = "no"

#: The small English model has no entry for "reachy" (checked with
#: Model.vosk_model_find_word), so a grammar containing it silently drops it and
#: the wake word can never be heard. These in-vocabulary sound-alikes go in the
#: grammar instead, and ``normalize_utterance`` writes them back as "reachy" so
#: the text the rest of the system sees always starts with the real wake word.
WAKE_ALIASES: Tuple[str, ...] = ("richie", "ritchie", "reach", "ricky")

#: Connectives of a spoken request. The panel does not serve these (its patterns
#: do), so they are listed here and nowhere else.
GLUE_WORDS: Tuple[str, ...] = (
    "the", "a", "and", "it", "to", "on", "at", "in", "into", "onto", "up",
    "down", "back", "put", "place", "pick", "move", "lift", "point", "cell",
)

_DIGIT_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
#: How "r" and "c" of a cell name come out of the recogniser.
_ROW_WORDS = ("r", "are")
_COL_WORDS = ("c", "see", "sea")

UNKNOWN = "[unk]"
SAMPLE_RATE = 16000
CHUNK_FRAMES = 4000

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = _REPO_ROOT / "tests" / "vosk" / "vosk-model-small-en-us-0.15"


class VoiceError(RuntimeError):
    """Recognition could not run (missing model, microphone, or dependency)."""


# -- grammar ----------------------------------------------------------------

def _words(text: str) -> List[str]:
    return re.findall(r"[a-z]{2,}", text.lower())


def build_grammar(abilities: Iterable[Dict[str, Any]], objects: Iterable[str] = (),
                  cells: Iterable[str] = ()) -> List[str]:
    """Word list for the recogniser, from the panel's served data.

    ``abilities`` is the panel's ``/capabilities`` ``abilities`` list (name and
    summary are used), ``objects`` and ``cells`` come from ``/scene``. Object
    ids such as ``red_cube`` contribute their parts ("red", "cube"). Anything
    in parentheses in a summary is detail for the operator and is left out.
    """
    words: List[str] = [WAKE_WORD, *WAKE_ALIASES, YES_WORD, NO_WORD, STOP_WORD]
    words.extend(GLUE_WORDS)
    for ability in abilities:
        words.extend(_words(str(ability.get("name", "")).replace("_", " ")))
        words.extend(_words(re.sub(r"\([^)]*\)", " ", str(ability.get("summary", "")))))
    for name in objects:
        words.extend(_words(str(name).replace("_", " ")))
    if any(True for _ in cells):
        words.extend(_ROW_WORDS)
        words.extend(_COL_WORDS)
        words.extend(_DIGIT_WORDS)
    seen = set()
    grammar = [w for w in words if not (w in seen or seen.add(w))]
    grammar.append(UNKNOWN)
    return grammar


def degraded_grammar() -> List[str]:
    """Used when the panel cannot be reached: "stop" and nothing else."""
    return [*WAKE_ALIASES, WAKE_WORD, STOP_WORD, UNKNOWN]


def normalize_utterance(text: str) -> str:
    """Wake-word sound-alikes back to "reachy"; spoken cell names to ``r2c2``."""
    tokens = text.lower().split()
    if tokens and tokens[0] in WAKE_ALIASES:
        tokens[0] = WAKE_WORD
    out: List[str] = []
    i = 0
    while i < len(tokens):
        if (i + 3 < len(tokens) and tokens[i] in _ROW_WORDS and tokens[i + 1] in _DIGIT_WORDS
                and tokens[i + 2] in _COL_WORDS and tokens[i + 3] in _DIGIT_WORDS):
            out.append(f"r{_DIGIT_WORDS[tokens[i + 1]]}c{_DIGIT_WORDS[tokens[i + 3]]}")
            i += 4
            continue
        out.append(tokens[i])
        i += 1
    return " ".join(out)


# -- recognition ------------------------------------------------------------

def _load_vosk():
    try:
        import vosk
    except ImportError as exc:
        raise VoiceError('vosk is not installed. Run: pip install -e ".[voice]"') from exc
    vosk.SetLogLevel(-1)
    return vosk


def _model_path(model_path: Optional[os.PathLike]) -> Path:
    path = Path(model_path or os.environ.get("REACHY_VOSK_MODEL") or DEFAULT_MODEL_PATH)
    if not path.is_dir():
        raise VoiceError(f"Vosk model not found at {path}. Set REACHY_VOSK_MODEL or pass --model.")
    return path


def _make_recognizer(grammar: Sequence[str], sample_rate: int, model_path: Optional[os.PathLike]):
    vosk = _load_vosk()
    model = vosk.Model(str(_model_path(model_path)))
    # A word the model does not know is dropped by Kaldi without a word; say so.
    known = [w for w in grammar if w == UNKNOWN or model.vosk_model_find_word(w) >= 0]
    dropped = [w for w in grammar if w not in known and w != WAKE_WORD]  # "reachy": see WAKE_ALIASES
    if dropped:
        log.warning("grammar words not in the Vosk model, dropped: %s", ", ".join(dropped))
    rec = vosk.KaldiRecognizer(model, sample_rate, json.dumps(known))
    rec.SetWords(True)
    return rec


def _parse(result_json: str) -> Optional[Tuple[str, float]]:
    """(normalised text, mean word confidence), or None for silence/[unk]-only."""
    result = json.loads(result_json)
    text = str(result.get("text", "")).strip()
    if not text:
        return None
    confs = [float(w.get("conf", 0.0)) for w in result.get("result", [])]
    conf = sum(confs) / len(confs) if confs else 0.0
    return normalize_utterance(text), conf


def recognize_wav(path: os.PathLike, grammar: Sequence[str], *,
                  model_path: Optional[os.PathLike] = None) -> Iterator[Tuple[str, float]]:
    """Yield ``(text, confidence)`` for each utterance in a PCM16 WAV file.

    Multi-channel files use channel 0, as the notebook does for the ReSpeaker.
    """
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise VoiceError(f"{path}: expected 16-bit PCM, got {8 * wf.getsampwidth()}-bit")
        channels = wf.getnchannels()
        rec = _make_recognizer(grammar, wf.getframerate(), model_path)
        while True:
            data = wf.readframes(CHUNK_FRAMES)
            if not data:
                break
            if channels > 1:
                samples = array("h")
                samples.frombytes(data)
                data = samples[::channels].tobytes()
            if rec.AcceptWaveform(data):
                parsed = _parse(rec.Result())
                if parsed:
                    yield parsed
        parsed = _parse(rec.FinalResult())
        if parsed:
            yield parsed


def recognize_mic(grammar: Sequence[str], *, model_path: Optional[os.PathLike] = None,
                  device: Optional[Any] = None, stop_event: Optional[Any] = None
                  ) -> Iterator[Tuple[str, float]]:
    """Yield ``(text, confidence)`` from the default (or given) microphone.

    Runs until ``stop_event`` is set. Audio is passed to the recogniser and
    discarded; it is never written or logged.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise VoiceError('sounddevice is not installed. Run: pip install -e ".[voice]"') from exc
    rec = _make_recognizer(grammar, SAMPLE_RATE, model_path)
    blocks: "queue.Queue[bytes]" = queue.Queue()

    def on_audio(indata, frames, time_info, status):  # noqa: ARG001
        blocks.put(bytes(indata))

    try:
        stream = sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_FRAMES, dtype="int16",
                                   channels=1, device=device, callback=on_audio)
        stream.start()
    except Exception as exc:  # PortAudioError and friends
        raise VoiceError(
            f"cannot open the microphone: {exc}. On macOS, allow microphone access for your "
            "terminal in System Settings > Privacy & Security > Microphone."
        ) from exc
    try:
        while stop_event is None or not stop_event.is_set():
            try:
                data = blocks.get(timeout=0.25)
            except queue.Empty:
                continue
            if rec.AcceptWaveform(data):
                parsed = _parse(rec.Result())
                if parsed:
                    yield parsed
    finally:
        stream.stop()
        stream.close()
