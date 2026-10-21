"""The expected results in docs/T3_DEMO.md match the engine (issue #53, epic #49)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("t3_demo_expected", ROOT / "scripts" / "t3_demo_expected.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_doc_has_the_eight_layouts_the_issue_names():
    layouts = demo.doc_layouts((ROOT / "docs" / "T3_DEMO.md").read_text())
    assert [n for n, *_ in layouts] == list(range(1, 9))
    rules = [text.split("Rule ")[1][0] if "Rule " in text else "-" for *_, text in layouts]
    assert rules == ["5", "1", "2", "3", "4", "9", "-", "-"]
    assert "XXX/OO./..." in [board for _, _, board, *_ in layouts]


def test_expected_text_in_doc_matches_engine(capsys):
    assert demo.main(["--check"]) == 0
    out = capsys.readouterr().out
    for _, _, _, _, documented in demo.doc_layouts((ROOT / "docs" / "T3_DEMO.md").read_text()):
        assert documented in out


def test_check_fails_when_the_doc_is_stale(tmp_path, capsys):
    text = (ROOT / "docs" / "T3_DEMO.md").read_text().replace("Rule 1, win", "Rule 1, wins", 1)
    stale = tmp_path / "T3_DEMO.md"
    stale.write_text(text)
    assert demo.main(["--check", "--doc", str(stale)]) == 0  # headings are not checked
    stale.write_text(text.replace("completes r1c1-r1c3", "completes the row", 1))
    assert demo.main(["--check", "--doc", str(stale)]) == 1
    assert "MISMATCH in layout 2" in capsys.readouterr().err


def test_boards_fit_the_sim_pool():
    for _, _, board, *_ in demo.doc_layouts((ROOT / "docs" / "T3_DEMO.md").read_text()):
        assert board.count("X") <= 6 and board.count("O") <= 6
