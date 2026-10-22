# Step 1 evidence for `docs/design/arm-motion.md` (#66)

Offline only: no robot, no live sim, no server, SDK or Docker. The sim repo was only read.

- **Sim:** reachy-1-2-sim @ `1948823`, MuJoCo 3.11.0, `native_mujoco/model/reachy_1_2.xml` compiled with
  `scenes/FWDCenterLabSivaPool.yaml` (the October layout, ADR-0010: torso at (0.206, 0, 1.15)).
- **Poses:** this repo @ `ef6e4ec`. SHA-256 of the notebooks as read is in `poses.json` → `provenance`
  (`CalibrateBoard.ipynb` is byte-identical to the copy #185 analysed, `dbd6ae66…`).

| File | What |
|---|---|
| `extract.py` → `poses.json`, `extract.txt` | Every `NAME = {reachy.r_arm.<joint>: <number>, …}` literal in `tests/MovePieceLogic.ipynb` and `tests/CalibrateBoard.ipynb`, with its cell, plus `BOARD_POSES`. Parses literals; executes nothing. |
| `replay.py` → `replay.txt` | Per pose: joint limits, `r_tip` (= SDK FK, checked against #185 to 0.00 mm), nearest cell (October model, the taught grid, two August boards), worst distance to the board, rig, torso/head and phantom left arm. Per segment (the joint-space line an SDK `goto` follows): the same. Physics flights of every chain. |
| `replay2.py` → `replay2.txt` | Per-geom board height at the low poses (analytic); entry and exit legs between REST and TRANSPORT with pieces on the board, kinematic and physics; a lab cube on r2c2 under `B2_PLACE`. |
| `replay3.py` → `replay3.txt` | A 50.8 mm lab cube at each taught grasp point: the descent, a physics close and lift, and REST's routes against a cube standing there. |
| `replay4.py` → `replay4.txt` | The candidate Part B route `B2_GRIP_TUNE`, end to end, kinematic and physics, in four layouts. |
| `board_check.py` → `board_check.txt` | The board clearance of every leg of every chain, analytic against `mj_geomDistance`: the box-box reads of "+0.0" are the known artefact. |

```bash
cd docs/design/arm-motion-step1
python3 -I extract.py ../../.. poses.json > extract.txt
for f in replay replay2 replay3 replay4 board_check; do python3 -I $f.py <reachy-1-2-sim> poses.json > $f.txt; done
```

The outputs are deterministic: a second run from this folder reproduced the first byte for byte.

**Reading the numbers.**
- Distances are signed, in cm: `+` clear, `-` inside. `+9.0+` means at least 9 cm. The read is capped there, because
  `mj_geomDistance` gives a spurious 0 for box pairs at large `distmax`.
- **Board clearances come from the analytic check** (`replay2.txt` (a), `board_check.txt`). The "+0.0" box-box readings
  in `replay.txt`'s board column are the artefact.
- Kinematic distances to box pieces can read too close, never too far. Every object result is also flown in physics.
- `self` is the forearm and hand against the model's torso and head boxes. The model does not collide the arm with
  them, and Siva flew the poses that read negative on the robot. So the torso box is larger than the real chest
  there, and `self` is reported, not judged.
