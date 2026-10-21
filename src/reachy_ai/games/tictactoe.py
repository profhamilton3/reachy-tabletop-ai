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
Rule 9 (search) is a guard, not a strategy: the eight rules never lose a game played from the start, but after an
opponent error they can miss a win or throw away a draw. When the rules' move is not minimax-optimal the first optimal
cell is recommended instead and labelled rule 9, so the hint stays honest. (Decision flagged on the PR for the owner.)
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

Board = tuple  # nine of X, O, EMPTY, UNKNOWN, OTHER in CELLS order


def parse(text: str) -> Board:
    rows = text.strip().split("/")
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError(f"board must be three rows of three joined by '/': {text!r}")
    try:
        return tuple(_CHARS[ch] for row in rows for ch in row)
    except KeyError as bad:
        raise ValueError(f"unknown board character {bad.args[0]!r} in {text!r}") from None


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


def _pick(board: Board, me: str) -> tuple:
    """(cell, rule, lines, detail) for the side to move on a valid in-progress board."""
    opp = _other(me)
    empties = _empties(board)

    for cell, line in _threat_cells(board, me):  # 1 win
        return cell, 1, (line,), f"completes {line_name(line)}."
    for cell, line in _threat_cells(board, opp):  # 2 block
        return cell, 2, (line,), f"{opp} threatens {line_name(line)}."
    for cell in empties:  # 3 fork
        lines = _fork_lines(board, cell, me)
        if lines:
            return cell, 3, tuple(lines), f"makes two threats, {' and '.join(line_name(ln) for ln in lines)}."

    for cell, reply, line, replied in _forcing_moves(board, me):  # 3, forced: reply leaves no threat, then fork
        if not _threat_cells(replied, opp) and any(_fork_lines(replied, c, me) for c in _empties(replied)):
            return cell, 3, (line,), f"forces {opp} to {CELLS[reply]}, then forks."

    opp_forks = [c for c in empties if _fork_lines(board, c, opp)]  # 4 block a fork
    if opp_forks:
        names = ", ".join(CELLS[c] for c in opp_forks)
        if len(opp_forks) > 1:  # several fork squares: force a reply that gives the opponent no fork
            for cell, reply, line, replied in _forcing_moves(board, me):
                if len({c for c, _ in _threat_cells(replied, opp)}) < 2:
                    return cell, 4, (line,), f"{opp} could fork at {names}; this forces {opp} to {CELLS[reply]}."
        cell = opp_forks[0]
        return cell, 4, tuple(_fork_lines(board, cell, opp)), f"{opp} could fork at {names}; take {CELLS[cell]}."

    if board[CENTRE] == EMPTY:  # 5 centre
        return CENTRE, 5, (), "take the centre."
    for corner in CORNERS:  # 6 opposite corner
        if board[corner] == opp and board[_OPPOSITE_CORNER[corner]] == EMPTY:
            return _OPPOSITE_CORNER[corner], 6, (), f"opposite {opp} at {CELLS[corner]}."
    for corner in CORNERS:  # 7 empty corner
        if board[corner] == EMPTY:
            return corner, 7, (), "take an empty corner."
    for side in SIDES:  # 8 empty side
        if board[side] == EMPTY:
            return side, 8, (), "take an empty side."
    raise AssertionError("no empty square on an in-progress board")


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


def _search_fallback(board: Board, me: str, cell: int) -> Optional[int]:
    """Rule 9: the first optimal cell when the rules' move gives up value (the eight rules never lose a game played
    from the start, but after an opponent error they can miss a win, or a draw when the position is already worse)."""
    best = max if me == X else min
    scores = {c: _value(_play(board, c, me), _other(me)) for c in _empties(board)}
    return next((c for c in _empties(board) if scores[c] == best(scores.values())), None) \
        if scores[cell] != best(scores.values()) else None


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
    cell, rule, lines, detail = _pick(board, me)
    better = _search_fallback(board, me, cell)
    if better is not None:
        cell, rule, lines = better, 9, ()
        detail = "the move the rules 1-8 would pick gives up a win or a draw; this one does not."
    text = (f"{me}'s turn: place a {PIECE_NAMES[me]} ({me}) at {CELLS[cell]}. "
            f"Rule {rule}, {RULE_NAMES[rule]}: {detail}")
    return Result(True, (), me, "in_progress", None, None, CELLS[cell], rule, RULE_NAMES[rule],
                  tuple(tuple(CELLS[i] for i in line) for line in lines), text)
