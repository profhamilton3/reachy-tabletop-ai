# T3-Analysis: tic-tac-toe in the stereo viewer

Reachy reads a tic-tac-toe board with its cameras, you confirm or correct what it read, and the viewer tells you whose
turn it is, the next move and why. Nothing is placed for you: this is Phase 1 of epic #49 (issues #50, #51, #52).
**X is a cube, O is a cylinder.**

**Safety.** The only motion is the head, to `board_view`, and only through the viewer's existing gated neck path: you
tick the confirm box for each move, the safety gate runs, the neck sign check applies, and **Stop head** works. The head
moves for real only when the viewer runs with `--allow-head-motion` and `REACHY_ENABLE_MOTION=true` (the launcher does
both); otherwise a move is a logged dry run and the head stays where it is. No arm or gripper route exists.

## Operator steps

1. Start the viewer (`./StereoLauncher.sh sim` or `robot`) and open the page. Press **Open T3-Analysis**.
2. **Clear the board.** Remove every piece, can and foam block.
3. Tick **I confirm: move the head to board_view**, press **Capture empty board**. The head goes to `board_view`, a pair
   is captured and analysed, and the reference is built. The panel lists the squares it found (all nine are needed).
   If it refuses, nothing is held: check the head is at `board_view`, the board is in view and lit, and try again.
4. Place the pieces. Tick the confirm box again, press **Analyze board**. The head moves to `board_view` again (a no-op
   when it is already there), a pair is captured, the detector runs and the board is read.
5. Check the board shown twice: **Reachy's side** (r1 nearest Reachy at the bottom, c1 on Reachy's left) and the
   **player's side** (the same board turned 180 degrees: r1c1 is at the player's far right).
6. Fix the squares: click one to cycle **X, O, empty, #**. **Accept suggestions** fills the unknown squares from their
   suggestions, but only where there is one. **Confirm board** is enabled only when no square is unknown.
7. Optionally tick quick tags (`demo`, `misread`, `illegal-setup`), add free text, then **Confirm board**. The viewer
   shows whose turn it is, the move square (green outline) and the line involved (green fill) on both views, and the
   rule hint, for example *O's turn: place a cylinder (O) at r3c3. Rule 2, block: X threatens r3c1-r3c3.* A finished
   board shows the win or draw; an illegal one shows why (counts, two winners, play after a win, a non-game object).
8. **Save labelled pair** downloads the pair zip with a `labels.json` of the confirmed squares.

Between games you only repeat steps 4 to 8. Capture the empty board again whenever the panel says the reference is stale.

## What the squares mean

| Square | Meaning |
|---|---|
| **X** (blue) | Cube. The detector's label was confident (>= 0.75) **and** the object's shape was square. |
| **O** (orange) | Cylinder. Detector confident and shape round. |
| **·** (grey) | Empty. The square did not change from the empty board and no piece was detected. |
| **#** (purple) | Other (a can, foam block, anything not a game piece). You set this by clicking; the reader never guesses it. |
| **?** (yellow, dashed) | Unknown. The reader will not guess: something changed but the detector and the shape did not agree, or nothing was detected. **Unknown is never empty.** You must set it. |
| faint letter | A *suggestion* for an unknown square (from the shape, or from a confident detector when the shape was unsure). It is not an answer until you accept or click it. |
| ✎ | You changed this square from what Reachy read. |

Hover a square for the reason it was read that way. The history keeps the detector label and confidence, the shape and
the change score for every square.

## When the reference goes stale

The empty-board reference is held in the viewer's memory. It is marked **stale**, and Analyze is refused, when the
camera session reconnects, the calibration, zoom or source changes, or the head is not at `board_view` (the last one
clears itself when Analyze moves the head). Capture the empty board again.

## History and labels

Each **Confirm board** appends one JSON line to `<t3-history-dir>/<session id>.jsonl` (default `outputs/t3_history`,
set with `--t3-history-dir`; `outputs/` is git-ignored and must never be committed). **This history log is the viewer's
only server-side write**; everything else (pairs, references, reads) stays in memory. A line records: the time (UTC),
session id, reference id, pair id and the pair bundle's SHA-256, the board as read, every square's evidence (state,
suggestion, shape, detector label and confidence, change score), the **corrections** (`square`, `read`, `confirmed`),
the confirmed board, the engine's result, your tags and note, and the reader settings.

**How corrections become labels.** Every square you change is a case where Reachy was wrong or unsure, with the picture
(by pair id and SHA-256) and the truth you confirmed: real-lab training labels for the detector and the shape rule. A
`misread` tag marks the readings worth re-checking. **Save labelled pair** gives the same pair as a zip (both images,
the analysis, `pair.json`) with `labels.json`: `{"board": "XOX/OXO/XOX", "squares": [{"cell", "label": cube|cylinder|empty|other,
"state", "read", "corrected"}, ...]}`. Keep these zips in `outputs/` and attach them to issues by hand; do not commit them.

## Under the hood

Routes: `GET /api/t3/state`, `POST /api/t3/reference`, `POST /api/t3/analyze`, `POST /api/t3/accept-suggestions`,
`POST /api/t3/confirm`, `GET /api/t3/reads/{id}/labelled-pair.zip`. Engine: `src/reachy_ai/games/tictactoe.py`; reader:
`src/reachy_ai/games/t3_board_reader.py`; routes: `src/reachy_ai/games/t3_viewer.py`. Offline check without the viewer:
`python scripts/t3_read_board.py --reference <empty pair zip> --pair <pair zip> --analyze`.
