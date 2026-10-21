"""Print every T3 demo layout's board and expected result, computed by the engine from the boards in docs/T3_DEMO.md.

    python scripts/t3_demo_expected.py            # print each layout and its expected text
    python scripts/t3_demo_expected.py --check    # exit 1 if the expected text in the doc no longer matches

Pure logic: moves nothing. Exit status: 0 all match (or printing), 1 a mismatch, 2 the doc could not be read.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachy_ai.games import tictactoe  # noqa: E402

DOC = Path(__file__).resolve().parents[1] / "docs" / "T3_DEMO.md"
LAYOUT = re.compile(
    r"### Layout (\d+): (.+?)\n\nBoard: `([^`]+)`.*?Expected result \(exit status (\d)\):\n\n```text\n(.*?)\n```",
    re.S,
)


def doc_layouts(text: str) -> list:
    """(number, name, board, exit status, expected text) for every layout in the doc."""
    return [(int(n), name, board, int(rc), expected) for n, name, board, rc, expected in LAYOUT.findall(text)]


def expected_text(board: str) -> tuple:
    """What scripts/t3_analyze.py prints for a board, and its exit status."""
    parsed = tictactoe.parse(board)
    result = tictactoe.recommend(parsed)
    return f"{tictactoe.to_string(parsed)}\n{result.text}", 0 if result.valid else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with the expected text in docs/T3_DEMO.md")
    parser.add_argument("--doc", type=Path, default=DOC, help="the demo doc (default docs/T3_DEMO.md)")
    args = parser.parse_args(argv)
    try:
        layouts = doc_layouts(args.doc.read_text())
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not layouts:
        print(f"error: no layouts found in {args.doc}", file=sys.stderr)
        return 2
    mismatches = 0
    for number, name, board, status, documented in layouts:
        text, expected_status = expected_text(board)
        print(f"Layout {number}: {name}\n{text}\n(exit status {expected_status})\n")
        if (text, expected_status) != (documented, status):
            mismatches += 1
            print(f"MISMATCH in layout {number}: the doc says\n{documented}\n(exit status {status})\n", file=sys.stderr)
    if args.check and mismatches:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
