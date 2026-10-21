# T3-Analysis demo script (Phase 1)

Step-by-step demo and checklist for issue #53 (epic #49, Phase 1). It takes the T3-Analysis panel through eight board
layouts, first in the simulator, then on the robot with the lab's pieces, and records what Reachy read and what it
recommended. How the panel works is in [T3_ANALYSIS.md](T3_ANALYSIS.md). **X is a cube, O is a cylinder.** Squares
are named from Reachy's side: r1 is nearest Reachy, c1 is on Reachy's left.

This page is a script, not a result. The results tables are blank until the runs are done.

**Safety.** Nothing here moves an arm. The only motion is the head, to `board_view`, through the viewer's gated neck
path (confirm tick, safety gate, sign check, **Stop head**). With the launcher's flags the head moves for real; in
dry-run it stays put and you must have it at `board_view` first.

## 1. Setup

**Simulator**

1. Start the simulator: `~/reachy-1-2-sim/SivaLabLauncher.sh`.
2. Start the viewer: `./StereoLauncher.sh` (simulator mode) and open the page.
3. Press **Open T3-Analysis**.
4. Clear the board with the panel's Scene control: every `pool_box_*`, `pool_cyl_*`, `red_cube`, `blue_cylinder`,
   `soda_can` and `foam_block` off the cells.
5. Tick the head confirm box, press **Capture empty board**. All nine squares must be listed as found.

**Robot**

1. Close every notebook or other program connected to the robot. Two clients must never command the neck at once.
2. `./StereoLauncher.sh robot`, open the page, and check the page shows `neck_sign_check: passed`. If it doesn't,
   stop and report it.
3. Press **Open T3-Analysis**, clear the board by hand (Siva), tick the head confirm box, press **Capture empty
   board**. Watch the board-view approach: at the 20 degree step the head must tilt **down**; if it tilts up or
   sideways, press **Stop head**.

## 2. The eight layouts

Sim limits: at most 5 `pool_box_*` plus `red_cube` (6 X) and 5 `pool_cyl_*` plus `blue_cylinder` (6 O). Every layout
is inside them. Place the pieces in the order listed, then tick the confirm box and press **Analyze board**.
Correct any wrong squares, then **Confirm board**. On the robot, Siva places a lab cube for each X and a lab cylinder
for each O on the same squares.

The expected text under each board is the exact output of `python scripts/t3_analyze.py --board "<board>"` for the
confirmed board. `scripts/t3_demo_expected.py` reprints it from the boards below (see section 6).

### Layout 1: Opening: O went first

Board: `O../.../...`

```text
  Reachy's side (r1 nearest Reachy, c1 on Reachy's left):
              c1 c2 c3
  r3 (far)    .  .  .
  r2          .  .  .
  r1 (near)   O  .  .
```

Panel placements (sim):

| Scene-control object | Cell |
|---|---|
| pool_cyl_1 | r1c1 |

Expected result (exit status 0):

```text
O../.../...
X's turn: place a cube (X) at r2c2. Rule 5, centre: take the centre.
```

### Layout 2: Rule 1, win

Board: `XX./OO./...`

```text
  Reachy's side (r1 nearest Reachy, c1 on Reachy's left):
              c1 c2 c3
  r3 (far)    .  .  .
  r2          O  O  .
  r1 (near)   X  X  .
```

Panel placements (sim):

| Scene-control object | Cell |
|---|---|
| pool_box_1 | r1c1 |
| pool_box_2 | r1c2 |
| pool_cyl_1 | r2c1 |
| pool_cyl_2 | r2c2 |

Expected result (exit status 0):

```text
XX./OO./...
X's turn: place a cube (X) at r1c3. Rule 1, win: completes r1c1-r1c3.
```

### Layout 3: Rule 2, block

Board: `XX./.O./...`

```text
  Reachy's side (r1 nearest Reachy, c1 on Reachy's left):
              c1 c2 c3
  r3 (far)    .  .  .
  r2          .  O  .
  r1 (near)   X  X  .
```

Panel placements (sim):

| Scene-control object | Cell |
|---|---|
| pool_box_1 | r1c1 |
| pool_box_2 | r1c2 |
| pool_cyl_1 | r2c2 |

Expected result (exit status 0):

```text
XX./.O./...
O's turn: place a cylinder (O) at r1c3. Rule 2, block: X threatens r1c1-r1c3.
```

### Layout 4: Rule 3, fork

Board: `XOX/O../...`

```text
  Reachy's side (r1 nearest Reachy, c1 on Reachy's left):
              c1 c2 c3
  r3 (far)    .  .  .
  r2          O  .  .
  r1 (near)   X  O  X
```

Panel placements (sim):

| Scene-control object | Cell |
|---|---|
| pool_box_1 | r1c1 |
| pool_cyl_1 | r1c2 |
| pool_box_2 | r1c3 |
| pool_cyl_2 | r2c1 |

Expected result (exit status 0):

```text
XOX/O../...
X's turn: place a cube (X) at r2c2. Rule 3, fork: makes two threats, r1c1-r3c3 and r1c3-r3c1.
```

### Layout 5: Rule 4, block a fork

Board: `..O/.X./O..`

```text
  Reachy's side (r1 nearest Reachy, c1 on Reachy's left):
              c1 c2 c3
  r3 (far)    O  .  .
  r2          .  X  .
  r1 (near)   .  .  O
```

Panel placements (sim):

| Scene-control object | Cell |
|---|---|
| pool_cyl_1 | r1c3 |
| pool_box_1 | r2c2 |
| pool_cyl_2 | r3c1 |

Expected result (exit status 0):

```text
..O/.X./O..
X's turn: place a cube (X) at r1c2. Rule 4, block a fork: O could fork at r1c1, r3c3; this forces O to r3c2.
```

### Layout 6: Rule 9, search

Board: `XO./..O/...`

```text
  Reachy's side (r1 nearest Reachy, c1 on Reachy's left):
              c1 c2 c3
  r3 (far)    .  .  .
  r2          .  .  O
  r1 (near)   X  O  .
```

Panel placements (sim):

| Scene-control object | Cell |
|---|---|
| pool_box_1 | r1c1 |
| pool_cyl_1 | r1c2 |
| pool_cyl_2 | r2c3 |

Expected result (exit status 0):

```text
XO./..O/...
X's turn: place a cube (X) at r3c1. Rule 9, search: the rule-of-thumb move r2c2 (rule 4, block a fork) would miss the win; r3c1 wins.
```

### Layout 7: Invalid counts

Board: `XX./X../...`

```text
  Reachy's side (r1 nearest Reachy, c1 on Reachy's left):
              c1 c2 c3
  r3 (far)    .  .  .
  r2          X  .  .
  r1 (near)   X  X  .
```

Panel placements (sim):

| Scene-control object | Cell |
|---|---|
| pool_box_1 | r1c1 |
| pool_box_2 | r1c2 |
| pool_box_3 | r2c1 |

Expected result (exit status 1):

```text
XX./X../...
Invalid board: counts: 3 X and 0 O differ by more than 1.
```

### Layout 8: A won game

Board: `XXX/OO./...`

```text
  Reachy's side (r1 nearest Reachy, c1 on Reachy's left):
              c1 c2 c3
  r3 (far)    .  .  .
  r2          O  O  .
  r1 (near)   X  X  X
```

Panel placements (sim):

| Scene-control object | Cell |
|---|---|
| pool_box_1 | r1c1 |
| pool_box_2 | r1c2 |
| pool_box_3 | r1c3 |
| pool_cyl_1 | r2c1 |
| pool_cyl_2 | r2c2 |

Expected result (exit status 0):

```text
XXX/OO./...
X wins: r1c1-r1c3. No move.
```

## 3. Results

Fill in one row per layout per run. **Read correct** is `yes` if Analyze read every square right with no corrections,
otherwise list the corrected squares (for example `r2c1 O->X`). **Move as expected** compares the result after
Confirm with the expected text above.

### Simulator run

| # | Layout | Read correct (yes / corrected squares) | Move as expected (yes / no) | Notes |
|---|---|---|---|---|

| 1 | Opening: O went first | | | |
| 2 | Rule 1, win | | | |
| 3 | Rule 2, block | | | |
| 4 | Rule 3, fork | | | |
| 5 | Rule 4, block a fork | | | |
| 6 | Rule 9, search | | | |
| 7 | Invalid counts | | | |
| 8 | A won game | | | |

### Robot run

| # | Layout | Read correct (yes / corrected squares) | Move as expected (yes / no) | Notes |
|---|---|---|---|---|

| 1 | Opening: O went first | | | |
| 2 | Rule 1, win | | | |
| 3 | Rule 2, block | | | |
| 4 | Rule 3, fork | | | |
| 5 | Rule 4, block a fork | | | |
| 6 | Rule 9, search | | | |
| 7 | Invalid counts | | | |
| 8 | A won game | | | |

The exit gate for Phase 1 is a correct move on every confirmed board, with the read accuracy reported: squares read
right first time, out of 9 per layout.

## 4. The robot run

Same eight layouts, with the lab's pieces and Siva placing them.

**Check first**

- No other client is connected to the robot.
- `neck_sign_check: passed` on the page. If not, stop and report; do not move the head.
- The board-view approach tilts the head **down** at the 20 degree step. Anything else: **Stop head**.
- The board is lit and fully in view. If the Capture step cannot find all nine squares (the chest can hide part of
  the near row, r1, at board_view), stop and report it; do not continue with a partial reference.
- The empty-board reference was captured in this session and the panel does not say it is stale.

**Record, for each layout**

- The squares Reachy read wrongly or left unknown (the corrections), and what each was corrected to.
- Whether the move after Confirm matches the expected text.
- A tag: `demo` for a normal run, `misread` for a reading worth re-checking, `illegal-setup` for layouts 7 and 8.
- **Save labelled pair** for every layout where a square was corrected, or was unknown, or looked wrong. Keep the zips
  under `outputs/` and attach them to the issue by hand. Do not commit them.
- The history file: `outputs/t3_history/<session id>.jsonl` (or the `--t3-history-dir` you launched with). Note its
  path and keep it with the run notes.

## 5. What the history file and the labelled pairs are for

Each **Confirm board** appends one line to the history file: the read, every square's evidence, the corrections, the
confirmed board, the engine's result and the tags. The history file is the viewer's only server-side write.

Every correction is a real-lab training label: the picture (pair id and SHA-256), what Reachy read, and the truth you
confirmed. The labelled pair zip carries the same pair with `labels.json`. Together they are the first real-lab labels
for the detector and the shape rule (perceived-scene design, P5). The history also shows how often Reachy needed a
correction on each kind of piece, which is the read accuracy to report. `outputs/` is git-ignored: do not commit these
files.

## 6. Re-checking the expected text

```bash
PYTHONPATH=src python scripts/t3_demo_expected.py          # print every layout and its expected result
PYTHONPATH=src python scripts/t3_demo_expected.py --check  # exit 1 if docs/T3_DEMO.md no longer matches the engine
```

`tests/test_t3_demo_expected.py` runs the same check, so a change to the engine's rules or wording fails a test until
the expected text above is regenerated.
