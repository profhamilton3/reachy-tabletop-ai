"""Analyse a tic-tac-toe board offline: validity, turn, outcome, and the recommended move with its rule.

    python scripts/t3_analyze.py --board "XX./OO./..."      # rows r1..r3: X . O ? #
    python scripts/t3_analyze.py --board "XX./OO./..." --json

Pure logic: moves nothing. Exit status: 0 the board is valid; 1 it is invalid; 2 the board string could not be parsed.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachy_ai.games import tictactoe  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--board", required=True, help='compact board, e.g. "X.O/.X./..O"')
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)
    try:
        board = tictactoe.parse(args.board)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    result = tictactoe.recommend(board)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(tictactoe.to_string(board))
        print(result.text)
    return 0 if result.valid else 1


if __name__ == "__main__":
    sys.exit(main())
