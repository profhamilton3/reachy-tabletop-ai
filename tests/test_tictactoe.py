"""Tic-tac-toe engine (issue #50): parsing, validation, turn, outcome, one case per rule, and a minimax oracle."""
import ast
import functools
import json
from pathlib import Path

import pytest

from reachy_ai.games import tictactoe as t

ROOT = Path(__file__).resolve().parents[1]


def rec(text):
    return t.recommend(t.parse(text))


# --- parsing and printing -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["X.O/.X./..O", ".../.../...", "XO?/#../XOX", "XXX/OO./..."])
def test_parse_print_round_trip(text):
    assert t.to_string(t.parse(text)) == text


def test_parse_maps_squares_in_order():
    board = t.parse("X.O/.?./..#")
    assert board[t.CELLS.index("r1c1")] == t.X
    assert board[t.CELLS.index("r1c3")] == t.O
    assert board[t.CELLS.index("r2c2")] == t.UNKNOWN
    assert board[t.CELLS.index("r3c3")] == t.OTHER
    assert board[t.CELLS.index("r3c1")] == t.EMPTY


@pytest.mark.parametrize("text", ["", "X.O/.X.", "X.O/.X./..OO", "X.A/.X./..O"])
def test_parse_rejects_malformed(text):
    with pytest.raises(ValueError):
        t.parse(text)


# --- validation -----------------------------------------------------------------------------------------------------

def test_valid_boards_have_no_reasons():
    assert t.validate(t.parse(".../.../...")) == []
    assert t.validate(t.parse("X.O/.X./..O")) == []
    assert t.validate(t.parse("XXX/OO./...")) == []


def test_reason_unknown():
    assert t.validate(t.parse("X?O/.X./..?")) == ["unknown square(s): r1c2, r3c3"]


def test_reason_non_game_object():
    assert t.validate(t.parse("X../.#./...")) == ["non-game object on r2c2"]


def test_reason_counts():
    assert t.validate(t.parse("XXX/X../OO.")) == ["counts: 4 X and 2 O differ by more than 1"]


def test_reason_both_win():
    reasons = t.validate(t.parse("XXX/OOO/..."))
    assert "both X and O have three in a row" in reasons


def test_reason_play_continued_after_win():
    # X has three in a row but fewer pieces than O: X could not have moved last
    assert t.validate(t.parse("OO./XXX/O.O")) == ["play continued after X won"]
    # O has three in a row but fewer pieces than X
    assert t.validate(t.parse("OOO/XX./XX.")) == ["play continued after O won"]


def test_winner_with_equal_or_one_more_is_fine():
    assert t.validate(t.parse("XXX/OO./...")) == []  # X moved first and last
    assert t.validate(t.parse("OXX/XOX/OOO")) == []  # 5 O, 4 X: O moved first and last


def test_several_reasons_reported_together():
    reasons = t.validate(t.parse("XXX/X?#/..."))
    assert len(reasons) == 3


# --- turn and outcome -----------------------------------------------------------------------------------------------

def test_turn():
    assert t.turn(t.parse(".../.../...")) == t.X
    assert t.turn(t.parse("X../.../...")) == t.O
    assert t.turn(t.parse("XO./.../...")) == t.X
    assert t.turn(t.parse("O../.../...")) == t.X


def test_outcome_win_draw_in_progress():
    win = t.outcome(t.parse("OO./XXX/O.."))
    assert (win.kind, win.winner, win.line) == ("win", t.X, (3, 4, 5))
    assert t.outcome(t.parse("XOX/XOO/OXX")).kind == "draw"
    assert t.outcome(t.parse("X../.../...")).kind == "in_progress"


def test_win_diagonal_and_column():
    assert t.outcome(t.parse("XO./OX./..X")).line == (0, 4, 8)
    assert t.outcome(t.parse("OX./OX./.XO")).winner == t.X


def test_finished_boards_have_no_move():
    won = rec("XXX/OO./...")
    assert (won.valid, won.outcome, won.winner, won.move_cell) == (True, "win", "X", None)
    assert won.winner_line == ("r1c1", "r1c2", "r1c3")
    assert "X wins" in won.text
    drawn = rec("XOX/XOO/OXX")
    assert (drawn.outcome, drawn.move_cell, drawn.winner_line) == ("draw", None, None)


def test_invalid_board_has_no_move_and_says_why():
    result = rec("XXX/X../OO.")
    assert not result.valid and result.move_cell is None and result.rule_number is None
    assert result.text == "Invalid board: counts: 4 X and 2 O differ by more than 1."
    assert result.turn is None and result.outcome == "invalid"


# --- the owner's example, then one case per rule -------------------------------------------------------------------

def test_owner_example_three_x_two_o():
    result = rec("O../O.X/XX.")
    assert result.turn == "O"
    assert (result.move_cell, result.rule_number, result.rule_name) == ("r3c3", 2, "block")
    assert result.lines == (("r3c1", "r3c2", "r3c3"),)
    assert result.text == "O's turn: place a cylinder (O) at r3c3. Rule 2, block: X threatens r3c1-r3c3."


def test_rule_1_win_beats_block():
    result = rec("XX./OO./X..")  # X to move? 3 X 2 O: O's turn; O completes r2c3 though X threatens r1c3
    assert result.turn == "O"
    assert (result.move_cell, result.rule_number) == ("r2c3", 1)
    assert rec("XX./OO./...").text.startswith("X's turn: place a cube (X) at r1c3. Rule 1, win")


def test_rule_2_block():
    assert (rec("XX./.O./...").move_cell, rec("XX./.O./...").rule_number) == ("r1c3", 2)


def test_rule_3_fork():
    result = rec("XXO/O../X..")
    assert (result.turn, result.move_cell, result.rule_number) == ("O", "r2c3", 3)
    assert len(result.lines) == 2


def test_rule_4_single_fork_square_is_taken():
    # O has a lone fork square at r2c2; X has no win, block or fork of its own
    result = rec(".../..O/.OX")
    assert result.turn == "X"
    assert result.rule_number == 4
    assert result.move_cell == "r2c2"


def test_rule_4_block_a_fork_refinement_matters():
    # X holds opposite corners, O the centre. Taking a corner loses to a fork; the side forces X and holds.
    board = t.parse("X../.O./..X")
    assert [t.CELLS[c] for c in range(9) if board[c] == t.EMPTY and t._fork_lines(board, c, t.X)] == ["r1c3", "r3c1"]
    result = rec("X../.O./..X")
    assert result.rule_number == 4
    assert result.move_cell in ("r1c2", "r2c1", "r2c3", "r3c2")  # a side, never a corner
    assert result.move_cell == "r1c2"  # fixed tie-break order
    assert value_after(board, t.CELLS.index(result.move_cell)) == 0
    assert value_after(board, t.CELLS.index("r1c3")) == 1  # the corner loses (X wins)


def test_rule_5_centre():
    assert (rec(".../.../...").move_cell, rec(".../.../...").rule_number) == ("r2c2", 5)
    assert rec("X../.../...").move_cell == "r2c2"


def test_rule_6_opposite_corner():
    result = rec("O../.X./...")
    assert (result.move_cell, result.rule_number) == ("r3c3", 6)


def test_rule_7_corner_first_in_order():
    result = rec("X../.O./...")
    assert (result.move_cell, result.rule_number) == ("r1c3", 7)


def test_rule_8_side():
    result = rec("XXO/OOX/X.O")
    assert (result.move_cell, result.rule_number, result.rule_name) == ("r3c2", 8, "side")


# --- minimax oracle -------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def value(board, mover):
    """+1 X wins, 0 draw, -1 O wins, with `mover` to play on `board` and best play from both sides."""
    won = t._winners(board)
    if won:
        return 1 if t.X in won else -1
    empties = t._empties(board)
    if not empties:
        return 0
    results = [value(t._play(board, c, mover), t._other(mover)) for c in empties]
    return max(results) if mover == t.X else min(results)


def value_after(board, cell):
    mover = t.turn(board)
    return value(t._play(board, cell, mover), t._other(mover))


def reachable(first):
    seen, stack = set(), [((t.EMPTY,) * 9, first)]
    while stack:
        board, mover = stack.pop()
        if board in seen:
            continue
        seen.add(board)
        if t._winners(board) or not t._empties(board):
            continue
        for cell in t._empties(board):
            stack.append((t._play(board, cell, mover), t._other(mover)))
    return seen


def test_oracle_every_reachable_position():
    positions = reachable(t.X) | reachable(t.O)
    checked = guarded = 0
    for board in positions:
        assert t.validate(board) == [], t.to_string(board)
        if t.outcome(board).kind != "in_progress":
            continue
        mover = t.turn(board)  # equal counts mean X, whichever side started
        result = t.recommend(board)
        assert result.move_cell is not None and result.turn == mover
        best = max if mover == t.X else min
        optimal = best(value_after(board, c) for c in t._empties(board))
        assert value_after(board, t.CELLS.index(result.move_cell)) == optimal, t.to_string(board)
        checked += 1
        guarded += result.rule_number == 9
    print(f"oracle: {len(positions)} reachable positions, {checked} in progress checked, {guarded} by rule 9")
    assert (len(positions), checked, guarded) == (8533, 6617, 34)  # rule 9 only after an opponent error


# --- the CLI and the import guard -----------------------------------------------------------------------------------

def test_cli_json(capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("t3_analyze", ROOT / "scripts" / "t3_analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--board", "XX./OO./...", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert (data["move_cell"], data["rule_number"], data["turn"]) == ("r1c3", 1, "X")
    assert module.main(["--board", "XXX/X../OO."]) == 1
    assert module.main(["--board", "nonsense"]) == 2


def test_import_guard_no_robot_code():
    tree = ast.parse((ROOT / "src" / "reachy_ai" / "games" / "tictactoe.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    banned = ("reachy_sdk", "reachy2_sdk", "reachy_ai.motion", "reachy_ai.stereo")
    assert not [name for name in imported if name.startswith(banned) or name.startswith(".")], imported
