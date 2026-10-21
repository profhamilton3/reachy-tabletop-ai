"""Tic-tac-toe engine: board model, validation, turn, outcome and a rule-based move with its hint (issue #50, epic #49).

Pure logic: no robot, camera, SDK or motion code. X is a cube, O is a cylinder.

Squares are r1c1..r3c3: r1 is nearest Reachy, c1 is on Reachy's left. The compact string lists rows r1..r3 joined by
"/", with X, O, "." (empty), "?" (unknown) and "#" (a non-game object): "X.O/.X./..O".

Turn: the side with fewer pieces; X when the counts are equal (Phase 1 does not remember who went first).

recommend() applies the rules in order and the first that applies decides:
  1 win; 2 block; 3 fork; 4 block a fork; 5 centre; 6 opposite corner; 7 empty corner; 8 empty side.
Rule 4 is the standard refinement: when the opponent has one fork square, take it; when it has several, make two in a
row whose forced reply gives the opponent neither a win nor a fork (this is why a side beats a corner when the
opponent holds two opposite corners and you hold the centre).
Owner ruling (PR #54): the first applicable rule's move can give up value after an opponent error. So the rules are
tried in order and the first whose candidate cell is minimax-optimal is used, with that rule's label. Only when none
is optimal is rule 9 (search) used; its text names the overridden rule-of-thumb move and what it would cost. Rule 3
also covers "fork in two moves": a forcing two in a row after which a fork is certain.
Ties between equal cells are broken in the fixed order r1c1, r1c2, r1c3, r2c1 ... r3c3, and lines in LINES order.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional

X, O, EMPTY, UNKNOWN, OTHER = "X", "O", "empty", "unknown", "other"  # noqa: E741
_CHARS = {"X": X, "O": O, ".": EMPTY, "?": UNKNOWN, "#": OTHER}
_SYMBOLS = {v: k for k, v in _CHARS.items()}
PIECE_NAMES = {X: "cube", O: "cylinder"}

CELLS = tuple(f"r{r}c{c}" for r in (1, 2, 3) for c in (1, 2, 3))
_INDEX = {name: i for i, name in enumerate(CELLS)}
LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)
CENTRE = 4
CORNERS = (0, 2, 6, 8)
SIDES = (1, 3, 5, 7)
_OPPOSITE_CORNER = {0: 8, 2: 6, 6: 2, 8: 0}

RULE_NAMES = {1: "win", 2: "block", 3: "fork", 4: "block a fork", 5: "centre", 6: "opposite corner",
              7: "corner", 8: "side", 9: "search"}
FORK_IN_TWO = "fork in two moves"  # rule 3 when the fork needs a forcing move first

Board = tuple  # nine of X, O, EMPTY, UNKNOWN, OTHER in CELLS order


def parse(text: str) -> Board:
    rows = text.strip().split("/")
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError(f"board must be three rows of three joined by '/': {text!r}")
    try:
        return tuple(_CHARS[ch] for row in rows for ch in row)
    except KeyError as bad:
        raise ValueError(f"unknown board character {bad.args[0]!r} in {text!r}") from None


def symbol(state: str) -> str:
    """The one-character form of a square state: X, O, ".", "?" or "#"."""
    return _SYMBOLS[state]


def state_of(char: str) -> str:
    """The square state a board-string character stands for."""
    return _CHARS[char]


def to_string(board: Board) -> str:
    chars = [_SYMBOLS[v] for v in board]
    return "/".join("".join(chars[i:i + 3]) for i in (0, 3, 6))


def line_name(line: tuple) -> str:
    return f"{CELLS[line[0]]}-{CELLS[line[2]]}"


def _winners(board: Board) -> dict:
    """{piece: [winning lines]} for the pieces with three in a row."""
    found = {}
    for line in LINES:
        a, b, c = (board[i] for i in line)
        if a == b == c and a in (X, O):
            found.setdefault(a, []).append(line)
    return found


def _count(board: Board, piece: str) -> int:
    return sum(1 for v in board if v == piece)


def validate(board: Board) -> list:
    """Reasons the board is not a legal game position; empty when valid."""
    reasons = []
    unknown = [CELLS[i] for i, v in enumerate(board) if v == UNKNOWN]
    if unknown:
        reasons.append(f"unknown square(s): {', '.join(unknown)}")
    other = [CELLS[i] for i, v in enumerate(board) if v == OTHER]
    if other:
        reasons.append(f"non-game object on {', '.join(other)}")
    nx, no = _count(board, X), _count(board, O)
    counts_ok = abs(nx - no) <= 1
    if not counts_ok:
        reasons.append(f"counts: {nx} X and {no} O differ by more than 1")
    won = _winners(board)
    if len(won) == 2:
        reasons.append("both X and O have three in a row")
    elif len(won) == 1 and counts_ok:
        winner = next(iter(won))
        mine, theirs = (nx, no) if winner == X else (no, nx)
        # the winner made the last move, so it has as many pieces as the other side or one more
        if mine < theirs:
            reasons.append(f"play continued after {winner} won")
    return reasons


def turn(board: Board) -> str:
    """The side with fewer pieces; X when equal."""
    return O if _count(board, O) < _count(board, X) else X


def _other(piece: str) -> str:
    return O if piece == X else X


@dataclass(frozen=True)
class Outcome:
    kind: str  # "win", "draw" or "in_progress"
    winner: Optional[str] = None
    line: Optional[tuple] = None  # cell indexes


def outcome(board: Board) -> Outcome:
    won = _winners(board)
    if won:
        winner = next(iter(won))
        return Outcome("win", winner, won[winner][0])
    if all(v in (X, O) for v in board):
        return Outcome("draw")
    return Outcome("in_progress")


def _threat_cells(board: Board, piece: str) -> list:
    """[(cell, line)] where piece has two in a line and the third square is empty, in LINES order."""
    found = []
    for line in LINES:
        values = [board[i] for i in line]
        if values.count(piece) == 2 and values.count(EMPTY) == 1:
            found.append((line[values.index(EMPTY)], line))
    return found


def _play(board: Board, cell: int, piece: str) -> Board:
    return board[:cell] + (piece,) + board[cell + 1:]


def _empties(board: Board) -> list:
    return [i for i, v in enumerate(board) if v == EMPTY]


def _fork_lines(board: Board, cell: int, piece: str) -> list:
    """The threat lines piece would have after playing cell, when that is two or more distinct winning squares."""
    threats = _threat_cells(_play(board, cell, piece), piece)
    return [line for _, line in threats] if len({c for c, _ in threats}) >= 2 else []


def _forcing_moves(board: Board, me: str) -> list:
    """[(cell, reply, line, replied)]: a single two in a row that forces the opponent's reply, where that reply does not
    win; replied is the board after the reply. Cells in r1c1..r3c3 order."""
    opp = _other(me)
    found = []
    for cell in _empties(board):
        after = _play(board, cell, me)
        threats = _threat_cells(after, me)
        if len(threats) != 1:
            continue
        reply, line = threats[0]
        replied = _play(after, reply, opp)
        if not _winners(replied):
            found.append((cell, reply, line, replied))
    return found


@dataclass(frozen=True)
class Result:
    valid: bool
    reasons: tuple
    turn: Optional[str]
    outcome: str  # "win", "draw", "in_progress" or "invalid"
    winner: Optional[str]
    winner_line: Optional[tuple]  # cell names
    move_cell: Optional[str]
    rule_number: Optional[int]
    rule_name: Optional[str]
    lines: tuple  # the line(s) involved in the rule, each a tuple of cell names
    text: str

    def to_dict(self) -> dict:
        return {
            "valid": self.valid, "reasons": list(self.reasons), "turn": self.turn, "outcome": self.outcome,
            "winner": self.winner, "winner_line": list(self.winner_line) if self.winner_line else None,
            "move_cell": self.move_cell, "rule_number": self.rule_number, "rule_name": self.rule_name,
            "lines": [list(line) for line in self.lines], "text": self.text,
        }


def _candidates(board: Board, me: str) -> list:
    """[(cell, rule, name, lines, detail)]: each rule's candidate move, in rule order (rule 3 has two forms)."""
    opp = _other(me)
    empties = _empties(board)
    found = []

    for cell, line in _threat_cells(board, me)[:1]:
        found.append((cell, 1, RULE_NAMES[1], (line,), f"completes {line_name(line)}."))
    for cell, line in _threat_cells(board, opp)[:1]:
        found.append((cell, 2, RULE_NAMES[2], (line,), f"{opp} threatens {line_name(line)}."))
    for cell in empties:
        lines = _fork_lines(board, cell, me)
        if lines:
            found.append((cell, 3, RULE_NAMES[3], tuple(lines),
                          f"makes two threats, {' and '.join(line_name(ln) for ln in lines)}."))
            break
    for cell, reply, line, replied in _forcing_moves(board, me):
        if not _threat_cells(replied, opp) and any(_fork_lines(replied, c, me) for c in _empties(replied)):
            found.append((cell, 3, FORK_IN_TWO, (line,), f"forces {opp} to {CELLS[reply]}, then forks."))
            break

    opp_forks = [c for c in empties if _fork_lines(board, c, opp)]
    if opp_forks:
        names = ", ".join(CELLS[c] for c in opp_forks)
        chosen = None
        if len(opp_forks) > 1:  # several fork squares: force a reply that gives the opponent no fork
            for cell, reply, line, replied in _forcing_moves(board, me):
                if len({c for c, _ in _threat_cells(replied, opp)}) < 2:
                    chosen = (cell, (line,), f"{opp} could fork at {names}; this forces {opp} to {CELLS[reply]}.")
                    break
        if chosen is None:
            cell = opp_forks[0]
            chosen = (cell, tuple(_fork_lines(board, cell, opp)), f"{opp} could fork at {names}; take {CELLS[cell]}.")
        found.append((chosen[0], 4, RULE_NAMES[4], chosen[1], chosen[2]))

    if board[CENTRE] == EMPTY:
        found.append((CENTRE, 5, RULE_NAMES[5], (), "take the centre."))
    for corner in CORNERS:
        if board[corner] == opp and board[_OPPOSITE_CORNER[corner]] == EMPTY:
            found.append((_OPPOSITE_CORNER[corner], 6, RULE_NAMES[6], (), f"opposite {opp} at {CELLS[corner]}."))
            break
    for corner in CORNERS:
        if board[corner] == EMPTY:
            found.append((corner, 7, RULE_NAMES[7], (), "take an empty corner."))
            break
    for side in SIDES:
        if board[side] == EMPTY:
            found.append((side, 8, RULE_NAMES[8], (), "take an empty side."))
            break
    return found


@functools.lru_cache(maxsize=None)
def _value(board: Board, mover: str) -> int:
    """Minimax value for X (+1 X wins, 0 draw, -1 O wins) with mover to play, both sides playing best."""
    won = _winners(board)
    if won:
        return 1 if X in won else -1
    empties = _empties(board)
    if not empties:
        return 0
    values = [_value(_play(board, c, mover), _other(mover)) for c in empties]
    return max(values) if mover == X else min(values)


def _pick(board: Board, me: str) -> tuple:
    """(cell, rule, name, lines, detail): the first rule whose candidate is minimax-optimal; rule 9 if none is.

    The first applicable rule's move can give up value after an opponent error, so the later rules are tried before
    searching. Rule 9 names the rule-of-thumb move it overrides and says what it would have cost."""
    best = max if me == X else min
    scores = {c: _value(_play(board, c, me), _other(me)) for c in _empties(board)}
    top = best(scores.values())
    candidates = _candidates(board, me)
    for candidate in candidates:
        if scores[candidate[0]] == top:
            return candidate
    first = candidates[0]  # a rule always applies on an in-progress board
    sign = 1 if me == X else -1
    verdict = "win" if sign * top == 1 else "draw"
    verb = "miss the win" if verdict == "win" else "give up the draw"
    cell = next(c for c in _empties(board) if scores[c] == top)
    detail = (f"the rule-of-thumb move {CELLS[first[0]]} (rule {first[1]}, {first[2]}) would {verb}; "
              f"{CELLS[cell]} {'wins' if verdict == 'win' else 'draws'}.")
    return cell, 9, RULE_NAMES[9], (), detail


def recommend(board: Board) -> Result:
    reasons = validate(board)
    if reasons:
        return Result(False, tuple(reasons), None, "invalid", None, None, None, None, None, (),
                      f"Invalid board: {'; '.join(reasons)}.")
    result = outcome(board)
    if result.kind == "win":
        line = tuple(CELLS[i] for i in result.line)
        return Result(True, (), None, "win", result.winner, line, None, None, None, (line,),
                      f"{result.winner} wins: {line_name(result.line)}. No move.")
    if result.kind == "draw":
        return Result(True, (), None, "draw", None, None, None, None, None, (),
                      "Draw: the board is full with no winner. No move.")
    me = turn(board)
    cell, rule, name, lines, detail = _pick(board, me)
    text = (f"{me}'s turn: place a {PIECE_NAMES[me]} ({me}) at {CELLS[cell]}. "
            f"Rule {rule}, {name}: {detail}")
    return Result(True, (), me, "in_progress", None, None, CELLS[cell], rule, name,
                  tuple(tuple(CELLS[i] for i in line) for line in lines), text)
