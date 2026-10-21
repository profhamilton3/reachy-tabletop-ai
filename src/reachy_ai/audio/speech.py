"""What Reachy says, and the call that says it (#59, part 2).

Reachy speaks only text built here from the system's own results: the tic-tac-toe engine's Result fields
(``spoken_t3``) and the sim panel's replies (``spoken_proposal``, ``spoken_refusal``, ``STOPPING``). Nothing is
generated free-form, and a Result is never re-read from its printed ``text``.

Speech is output only. It never triggers an action and never replaces a confirm. ``speak`` sends the text to the
audio helper (REACHY_AUDIO_URL) and never raises into the caller: a failure is logged and returned. The spoken
text is logged; audio never is. This module imports nothing that can move the robot.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from reachy_ai.audio.audio_client import AudioClient, AudioHelperError
from reachy_ai.games import tictactoe as t3

log = logging.getLogger(__name__)

MAX_SPOKEN = 200  # the helper's /say limit
SPEAK_TIMEOUT_S = 40.0  # longer than the helper's TTS timeout (30 s); /say answers when the speech ends
#: After Reachy stops speaking, results are still treated as self-heard for this long: the recogniser reports an
#: utterance only after its trailing silence, so its own "Say Reachy, yes" would otherwise arrive just after.
SPEECH_TAIL_S = 1.5
STOPPING = "Stopping."

_NUMBERS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_PIECES = {t3.X: ("cube", "cubes"), t3.O: ("cylinder", "cylinders")}
_RULE_PHRASES = {5: "That's the centre rule.", 6: "That's the opposite corner rule.", 7: "That's the corner rule.",
                 8: "That's the side rule."}


def _number(n: int) -> str:
    return _NUMBERS[n] if 0 <= n < len(_NUMBERS) else str(n)


def _square(cell: str) -> str:
    """r3c3 -> "row three, column three"; the centre is "the centre, row two, column two"."""
    spoken = f"row {_number(int(cell[1]))}, column {_number(int(cell[3]))}"
    return f"the centre, {spoken}" if cell == "r2c2" else spoken


def _on(cell: str) -> str:
    return f"in {_square(cell)}" if cell == "r2c2" else f"on {_square(cell)}"


def _line(cells) -> str:
    """A line of three cell names as spoken: "row three", "column one" or "the diagonal"."""
    rows, cols = {c[1] for c in cells}, {c[3] for c in cells}
    if len(rows) == 1:
        return f"row {_number(int(next(iter(rows))))}"
    if len(cols) == 1:
        return f"column {_number(int(next(iter(cols))))}"
    return "the diagonal"


def _pieces(piece: str, n: int) -> str:
    one, many = _PIECES[piece]
    return f"{_number(n)} {one if n == 1 else many}"


def _problem(problem: tuple) -> str:
    kind = problem[0]
    if kind == "counts":
        return f"there are {_pieces(t3.X, problem[1])} and {_pieces(t3.O, problem[2])}"
    if kind == "unknown":
        cells = problem[1]
        return (f"{_square(cells[0])} is still unknown" if len(cells) == 1
                else f"{_number(len(cells))} squares are still unknown")
    if kind == "other":
        cells = problem[1]
        return (f"there's something that isn't a game piece on {_square(cells[0])}" if len(cells) == 1
                else f"there are things that aren't game pieces on {_number(len(cells))} squares")
    if kind == "two_winners":
        return "both X and O have three in a row"
    if kind == "after_win":
        return f"play went on after {problem[1]} won"
    return "the position is not legal"


def _move(result: t3.Result) -> str:
    me, cell, rule = result.turn, result.move_cell, result.rule_number
    opp = t3.O if me == t3.X else t3.X
    place = f"It's {me}'s turn. Place a {_PIECES[me][0]} {_on(cell)}"
    line = _line(result.lines[0]) if result.lines else None
    if rule == 1:
        return f"{place}, to win on {line}." if line else f"{place}, to win."
    if rule == 2:
        return f"{place}, to block {opp}'s {line}." if line else f"{place}, to block {opp}."
    if rule == 3 and result.rule_name == t3.FORK_IN_TWO:
        return f"{place}. That forces {opp}'s reply, then sets up a fork."
    if rule == 3:
        return f"{place}, to make two threats at once."
    if rule == 4:
        return f"{place}, to stop {opp} setting up a fork."
    if rule == 9 and result.search is not None:
        verb = "miss the win" if result.search[0] == "win" else "give up the draw"
        return f"{place}. I checked every move ahead: the usual rule would {verb}."
    return f"{place}. {_RULE_PHRASES.get(rule, '')}".strip()


def spoken_t3(result: t3.Result) -> str:
    """The T3-Analysis result as Reachy says it, built from the Result's fields (never from ``result.text``)."""
    if result.outcome == "win":
        return f"{result.winner} has won, on {_line(result.winner_line)}."
    if result.outcome == "draw":
        return "It's a draw."
    if result.outcome == "invalid":
        said = [_problem(p) for p in result.problems] or ["the position is not legal"]
        text = f"I can't analyse this board: {', and '.join(said)}."
        return text if len(text) <= MAX_SPOKEN else f"I can't analyse this board: {said[0]}."
    return _move(result)


def _fit(prefix: str, middle: str, suffix: str) -> str:
    """prefix + middle + suffix within MAX_SPOKEN, cutting `middle` at a word if it is too long."""
    middle = " ".join(str(middle).split()).rstrip(" .")
    room = MAX_SPOKEN - len(prefix) - len(suffix)
    if len(middle) > room:
        middle = middle[:room].rsplit(" ", 1)[0].rstrip(" ,;:")
    return prefix + middle + suffix


def spoken_proposal(summary: str) -> str:
    """The panel's proposal: "I'd pick up the cube and put it back. Say Reachy, yes to go ahead." """
    return _fit("I'd ", summary, ". Say Reachy, yes to go ahead.")


def spoken_refusal(reason: str) -> str:
    """A refusal with the panel's reason: "I can't do that: <reason>." """
    return _fit("I can't do that: ", reason or "no reason given", ".")


def speak(text: str, client: Optional[AudioClient] = None, timeout_s: float = SPEAK_TIMEOUT_S) -> dict:
    """Say `text` through the audio helper. Never raises: returns {"ok", "text", "spoken", "hushed", "muted",
    "error", "detail"}, and logs the text (never audio) and any failure."""
    reply = {"ok": False, "text": text, "spoken": False, "hushed": False, "muted": False, "error": None,
             "detail": None}
    log.info("speak: %r", text)
    try:
        said = (client or AudioClient()).say(text, timeout_s=timeout_s)
    except AudioHelperError as exc:
        log.warning("speech unavailable (%s): %s", exc.code, exc)
        return dict(reply, error=exc.code, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - speech must never break the caller
        log.warning("speech failed: %s", type(exc).__name__)
        return dict(reply, error="failed", detail=f"speech failed: {type(exc).__name__}")
    hushed, muted = bool(said.get("hushed")), bool(said.get("muted"))
    return dict(reply, ok=True, spoken=not muted, hushed=hushed, muted=muted)


class Speaker:
    """Speaks in the background, so a listener keeps hearing "stop" while Reachy talks.

    ``speaking()`` is the self-hearing flag: true from ``say`` until SPEECH_TAIL_S after the helper answers. A new
    phrase replaces the one in progress (it is hushed first): never a backlog. Nothing here raises."""

    def __init__(self, client: Optional[AudioClient] = None, *, tail_s: float = SPEECH_TAIL_S,
                 clock: Callable[[], float] = time.monotonic,
                 speak_fn: Callable[..., dict] = speak):
        self.client = client or AudioClient()
        self.tail_s = tail_s
        self._clock = clock
        self._speak = speak_fn
        self._lock = threading.Lock()
        self._active = 0
        self._ended_at: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self.last: Optional[dict] = None  # the latest speak() outcome

    def speaking(self) -> bool:
        with self._lock:
            if self._active:
                return True
            return self._ended_at is not None and self._clock() < self._ended_at + self.tail_s

    def say(self, text: str) -> threading.Thread:
        with self._lock:
            self._active += 1  # up before the thread starts, so no result slips through
            previous = self._thread
            thread = threading.Thread(target=self._run, args=(text, previous), daemon=True)
            self._thread = thread
        thread.start()
        return thread

    def _run(self, text: str, previous: Optional[threading.Thread]) -> None:
        try:
            if previous is not None and previous.is_alive():
                self.hush()
                previous.join(5.0)
            self.last = self._speak(text, client=self.client)
        finally:
            with self._lock:
                self._active -= 1
                self._ended_at = self._clock()

    def hush(self) -> dict:
        """The helper's /hush, with a short timeout. Never raises."""
        try:
            return self.client.hush()
        except AudioHelperError as exc:
            log.warning("hush failed (%s): %s", exc.code, exc)
            return {"ok": False, "error": exc.code, "detail": str(exc)}
        except Exception as exc:  # noqa: BLE001
            log.warning("hush failed: %s", type(exc).__name__)
            return {"ok": False, "error": "failed", "detail": type(exc).__name__}
