# Part perception for the PART set (#84)

Status: **sim only.** Every number here was measured on rendered frames. Nothing was measured on the robot or on a
printed part. The Part presentation procedure is in `docs/DATA_CAPTURE.md`.

## What it produces

The `observation` block of a `disposition_decision.v1` request (#81): `kind`, `height_mm` and `width_mm` (each with
a bound), `hole` (state, and diameter with a bound), `orientation`, six `sufficiency` flags, the capture bundle's
name and SHA-256, and `colour`.

Code: `perception/part_geometry.py` (lens, board pose, outline, solid fit), `part_hole.py`, `part_colour.py`,
`part_perception.py` (the whole analysis) and `part_viewer.py` (the viewer's routes). It runs offline
(`scripts/analyze_capture_bundle.py <part>.zip --reference <empty>.zip`) and in the viewer (*Analyze part*, then
the decision panel's input *part analysis*). Nothing in it can command the robot.

## Method

1. **Board pose per eye and zoom**, from the grid in the **empty-board run** at the same zoom. No neck angle, camera
   mount or stereo calibration enters a part's size. The near row is often clipped by the robot's chest, so corners
   that do not fit are dropped; a fit worse than 1 px is refused. The squares are 127 mm in the sim and 129 mm
   (5.08 in) on the lab board: the wrong one costs about 1 px.
2. **Finding the part:** what changed on the board against that run (largest difference over the three channels),
   with cast shadows removed. No detector, so no PART classes are needed.
3. **Size and shape:** a box, an upright cylinder and a lying cylinder, each standing on the board, are fitted to
   the outlines of both eyes and both captures at zoom IN. The best must explain the outline, and the other shape
   must fit clearly worse; otherwise kind and orientation are `unknown`. Zoom OUT is only a cross-check.
4. **Hole:** the fitted top face is rectified to millimetres and tested for a circle with an edge around most of it
   and the face just outside. Top face only: through and blind cannot be told apart from one view.
5. **Colour:** CIELAB median over the part's pixels, clipped ones left out, white-balanced to the board's unclipped
   white. `reliable` is false when that white is not visible, the lighting check fails, or more than half of the
   part's pixels are clipped.

### A difference from the issue's formula

#84 names *height = object pixel height × stereo depth ÷ focal length*. At `board_view` the cameras look about
16° off straight down at r2c2, so a part's pixel height is mostly its **top face**; its height adds only about 0.3 px
per mm at zoom IN. The fit is the same pinhole relation with the view's angle included. Depth comes from the board:
its pose in each eye, and the part standing on it. Stereo matching is not used, because the flat faces have little
or no texture to match.

### Colour is never a geometry input

Every geometry step uses only the largest difference, or gradient, over the three channels, and nothing reads which
channel was largest. A test reorders the channels of every picture: the colour changes, while kind, orientation, hole,
height and width stay the same.

### Orientation

`upright`: a square or round footprint with its axis vertical, at any turn about it. `lying`: a side face is down.
`unknown`: no solid explains the outline. **`rotated` is never reported:** no tilted solid is fitted. The contract
does not define the word, so this definition needs the owner's confirmation. A block with three equal sizes could
not be told lying from upright; none of the PART designs is a cube.

The operator's orientation label is **not** copied into a perceived observation (the decision panel's rule: typed
values are not perceived). It goes into `ground_truth.json` for grading.

### Sufficiency flags

| flag | raised when |
|---|---|
| `blurred` | the outline's edge is over 2.2 px wide, or the board is under half as sharp as in the reference |
| `occluded` | no solid explains the outline, other things on the board changed, the part moved between captures, or it is cut off by the frame |
| `stale` | the eyes' frames are over 500 ms apart, or two captures carry the same frames |
| `lens_unverified` | a capture's zoom read-back is not the zoom asked for, or the viewer's lens warning |
| `depth_insufficient` | no lens model, no board pose, no outline in both eyes, a moved head, over 30% of the outline against dark tape, OUT only, or a height bound over 12 mm |
| `feature_not_visible` | orientation is not `upright`, or the top face is not in the picture |

## Sim grading (2026-11-28)

**How:** `scripts/sim_render_part_cases.py` renders each case with the sim's own world, renderer, lens distortion and
zoom model (reachy-1-2-sim `c46f4e4`, MuJoCo 3.11.0, macOS arm64, measured calibration, `gaze.BOARD_VIEW`) and
writes the sim's `truth.v1`. `scripts/grade_part_perception.py` builds each bundle with the viewer's own
`CaptureRun`, analyses it and compares. No live server was involved.

```sh
python3 scripts/sim_render_part_cases.py --sim-repo ../reachy-1-2-sim --out /tmp/part_cases   # the sim's Python
python scripts/grade_part_perception.py --cases /tmp/part_cases --json results.json            # this project's
```

**These are development numbers, not held-out ones.** The thresholds and both bound floors were set on these cases:
all eight designs upright, clear view, plus turned, lying, occluded and artificially blurred ones. The proposed
held-out combinations of `split_v0.yaml` (a lying holed block, a tall cylinder partly occluded, a defocused short
part) were never rendered or graded. The same frames repeat for the two captures, because the sim has no noise, so
the "between captures" part of a bound is zero here and will not be on the robot.

### Height, hole and orientation: black print and two randomised draws (seeds 2026 and 7)

240 upright cases: each design on all nine squares, plus the four blocks turned 20° and 45° on r2c2, in each scene.

| measure | result |
|---|---|
| kind (block or cylinder) | 240 / 240 right |
| orientation | 240 / 240 `upright` |
| height class (short or tall) | 240 / 240 right |
| height error, mm | mean −2.0 (reads short); median 1.8, 95th percentile 4.2, largest 6.0 |
| height bound | holds the truth in 228 / 240 (95.0 %); median ± 4.3 mm, largest ± 4.9 |
| width error, mm | median 0.2, 95th percentile 1.0, largest 1.9; bound (median ± 2.3) holds 240 / 240 |
| holed parts | `present` 120 / 120 |
| solid parts | `absent` 119 / 120, `unknown` 1, never `present` |
| hole diameter (true 25.4 mm) | reads 1.2 mm wide on average, largest error 1.8; bound holds 120 / 120 |
| flags raised on these clear views | none |

The height bias is similar on all three rows (−1.9, −2.3, −1.6 mm). It was **not** corrected: a correction fitted to
the sim would say nothing about the real lens.

### The other development cases (black and family scenes)

| cases | result |
|---|---|
| lying (36: six designs, three turns, two scenes) | `lying` 24, `unknown` 12, **never `upright`**; `feature_not_visible` on all 36. Kind wrong in 2 (PART211 end-on, called a block). Height, when measured (24): median error 0.8 mm, largest 5.5; bound holds 23 |
| a pool cylinder in front of the part (24, no tall cylinder) | `occluded` 24 / 24 |
| tall parts blurred with a Gaussian, σ 1 to 3 px (48) | `blurred` 48 / 48 |

The blur cases are synthetic: the sim renders no defocus.

### Colour: family and randomised scenes

80 upright and turned cases per scene. The expected name is the family's (yellow, orange, blue for the dusty-blue
grey, green).

| scene | name right | `reliable` | right among `reliable` |
|---|---|---|---|
| family | 68 / 80 | 53 | 49 / 53 (92 %) |
| random, seed 2026 | 69 / 80 | 58 | 55 / 58 (95 %) |
| random, seed 7 | 64 / 80 | 58 | 50 / 58 (86 %) |

- Most misses are **orange called brown** and **dusty blue called grey**. A tall part's median is mostly its unlit
  sides once the clipped top is left out. The sim's own reference puts the blue's side faces at chroma 10, under
  the grey limit of 11. The naming limits are generic and were not moved to fit the family colours.
- About 60 % of the yellow and orange top pixels are clipped in the sim; over half clipped means not `reliable`.
- Against the sim's rendered reference table (clipped pixels kept, no white balance, so like with like), the median
  distance is ΔE76 13.6 to 15.1 per scene, with single placements up to 51. The reference mixes lit tops and dim sides over
  every square, so one placement can sit far from it. On black it is 2.2 (largest 5.0).

### What configuration E does with these measurements

Run on each upright observation as a first look at a healthy cell with the expected design given, **E decided 1 of
72 in each scene** (the family, black and both random ones). Each of those decisions matched the spec. Every other
case was `remeasure/reobserve`, almost all by R7.

This follows from the numbers, not from a fault in either part. R7 passes a size only when value ± bound lies wholly
inside nominal ± tolerance. The height tolerance is ± 5 mm (provisional) and the height bound is about ± 4.3 mm, so
only a value within about 0.7 mm of nominal passes, and heights read 2 mm short. **An owner decision is needed**
before E can accept anything from this analyser: a looser provisional height tolerance, a better measurement, or a
height bias calibrated on the robot against measured parts. Nothing was changed in the policy files.

## Limits that hold today

- **Robot:** there is no lens calibration at zoom OUT or IN (#4), so nothing metric is reported from a robot
  capture (`depth_insufficient`, with the reason). The captures can still be collected.
- **The empty-board run must match:** same head pose (within 1°), same zoom, same lighting, board not moved.
- **Black on black:** a black part over black tape shows no edge there. Those outline points count only as a lower
  limit. On the marked spot r2c2 the PART parts do not reach the tape.
- **Real black PLA** under lab light, a pale part on the white board, and real shadows are untested. The sim has no
  auto-exposure and no noise.
