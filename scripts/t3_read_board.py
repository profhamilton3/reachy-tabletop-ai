"""Read a tic-tac-toe board from a stereo pair against an empty-board reference. Offline: reads files, moves nothing.

    python scripts/t3_read_board.py --reference empty_pair.zip --pair pair.zip [--analyze] [--json]

Both are viewer pair bundles (.zip or unpacked folder) at the same head pose, calibration, zoom and source. The pair
needs analysis.json (analyse it in the viewer). Prints the board string (X cube, O cylinder, . empty, ? unknown), the
per-square evidence and, with --analyze, the engine's turn and move.

Exit status: 0 the board was read; 2 the reference or the pair was refused.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachy_ai.games import t3_board_reader as reader  # noqa: E402
from reachy_ai.games import tictactoe  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", required=True, help="empty-board pair bundle at the session's head pose")
    parser.add_argument("--pair", required=True, help="pair bundle to read")
    parser.add_argument("--analyze", action="store_true", help="also run the 1a engine on the board read")
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)
    try:
        read = reader.read_board(args.pair, reader.build_reference(args.reference))
    except reader.BoardReadRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    result = tictactoe.recommend(read.board) if args.analyze else None
    if args.json:
        document = read.to_dict()
        if result is not None:
            document["analysis"] = result.to_dict()
        print(json.dumps(document, indent=2))
        return 0
    print(read.board_string())
    for square in read.squares:
        seen = ", ".join(f"{d['label']} {d['confidence']:.2f}" for d in square.detections) or "none"
        score = "n/a" if square.change_score is None else f"{square.change_score:.2f}"
        hint = f"  (suggest {square.suggested} by {square.suggested_by})" if square.suggested else ""
        print(f"  {square.cell}  {square.state:<7} change {score:>5}  seen: {seen}  - {square.reason}{hint}")
    if result is not None:
        print(result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
