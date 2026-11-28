# Head-sweep pairs with the sim's grid turned (#96)

Rendered offline by `scripts/sim_render_board_yaw.py` from reachy-1-2-sim `c46f4e4` with the server's own world and
renderer classes, the measured calibration and lens distortion at the calibrated zoom, MuJoCo 3.11.0, macOS arm64:

    python3 scripts/sim_render_board_yaw.py --sim-repo ../reachy-1-2-sim --out OUT --yaw-deg 0 3

The scene is a child of `FWDCenterLabSivaPool` that turns only the 17 `grid_*` and `cell_*` objects about the centre
square, by 0 and by +3 degrees (counter-clockwise seen from above, the torso's +z). Poses: `board_view`,
`sweep_yaw_right_5`, `sweep_yaw_left_5`. `pair.json` holds the neck read-back and the lens, the fields the analysers
read.
