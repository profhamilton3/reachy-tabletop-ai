# T3 board-reader fixtures

Sim renders (reachy-1-2-sim, board_view, ZERO zoom, calibration 771392183ddc6c62), safe to commit. Each folder is the
first pair of a perception-186 layout, kept to pair.json, left.jpg, right.jpg and analysis.json (the right-eye detector
output), plus the matching truth_before.json (sim truth export).

| folder | layout | truth |
|---|---|---|
| A_default | empty board | `.../.../...` |
| D_full | 5 cubes, 4 cylinders | `XOX/OXO/XOX` |
| E_full_swapped | 4 cubes, 5 cylinders | `OXO/XOX/OXO` |

The reference for the tests is `tests/fixtures/sim_head_sweep` (an empty board at board_view). The detector mislabels some
cubes as cylinders on these renders; tests/test_t3_board_reader.py pins what it does.
