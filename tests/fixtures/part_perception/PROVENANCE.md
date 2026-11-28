# Part-presentation test frames (#84)

Rendered offline by `scripts/sim_render_part_cases.py` from reachy-1-2-sim `c46f4e4` (main after sim PR #219), with
the server's own world, renderer, lens distortion and zoom model, the measured calibration
(`calibration_measured_2026_08_27`), `gaze.BOARD_VIEW`, MuJoCo 3.11.0 on macOS arm64. Scenes:
`FWDCenterLabSivaPoolBlack` (black_*, empty) and `FWDCenterLabSivaPool` (family_210). These are development cases
only: none is a held-out combination of `config/disposition/split_v0.yaml`.

Each folder holds the four frames (zoom OUT and IN, left and right, 640 x 480, the server's JPEG) and `case.json`: the
placement, the neck read-back and, for the part's own object, the sim's `truth.v1` entry.

| folder | case |
|---|---|
| empty | the empty board (the reference) |
| black_221 | PART221 upright on r2c2 |
| black_111 | PART111 upright on r2c2 |
| family_210 | PART210 upright on r2c2, family colours |
| black_221_far | PART221 upright on r3c2: its top reaches the far tape, black on black |
| black_220_lying | PART220 lying on r2c2, turned 45 deg |
