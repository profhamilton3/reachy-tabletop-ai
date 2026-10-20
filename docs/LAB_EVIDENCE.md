# Lab evidence — provenance

Primary source material for the physical FWD Center (MCC) tabletop setup. The
simulator scene `scenes/FWDCenterLabMCC.yaml` in the companion repo
`reachy-1-2-sim` derives its geometry from these files and cites them by path,
so they are committed here to keep those citations resolvable.

**Do not edit the primary records.** `ActualLabSetupNotes.txt` contains numbers
that later measurement showed to be wrong (see below). It is kept verbatim as
the original observation; corrections belong in this file and in the scene, not
in the note.

## Files

| File | SHA-256 | Notes |
|---|---|---|
| `ActualLabSetupNotes.txt` | `b403e340511af0c4856149052c07af671306f1593be0789e9f0e7210ae02a575` | Hand-written setup notes |
| `pics/80903693586__73910DA6-76BD-4BDE-A21A-DB4BBB873787.JPG` | `d212e1f73a07604de43b3a438d0eac1e9676af27dfb7fe0b4dc4c893fa65e35e` | Arm resting on the board; frame and mast visible |
| `pics/80903694842__9F4E2977-4CA3-4678-A038-8BBAC9E04824.JPG` | `a1030629482dfa7d6b1574808e4cbd1b765029c1041d0eaf29dacad09fcb47df` | Frontal view of the full grid; best for rectification |
| `pics/80903696209__91C25E33-BC7D-4CD1-ADE3-9CB7DCF7E767.JPG` | `9db5c78c7a8c62a85df402959544569a82d3d9be12649457da78714760cdaf13` | Head, Orbita neck and the mast-to-frame clamp |

Photos: Siva, 2026-08-21 16:28–16:29, iPhone 16 main camera (26 mm equivalent,
4032×3024). Kept at full resolution deliberately — they are measurement input,
and downscaling would destroy the ability to re-derive the numbers below.

The three companion Live Photo `.mov` clips are **not** committed. They are
1–3 second motion clips of these same three frames, ~7.8 MB, and add nothing
the stills do not; the open questions they might have answered need a tape
measure, not more pixels. They remain on the original device if wanted.

Related: the head-camera captures in `tests/Pictures/` (20 stereo pairs,
2026-08-20) are the other half of the evidence base.

## What was derived from these

Calibration used the taped board as its own target: nine 5 in cells on a 6 in
pitch give 36 metric points per view. 20 stereo pairs solved at 0.97 / 0.84 px
reprojection RMS; the phone photos were then rectified through the same pattern
as an independent cross-check. Both methods put the board's far edge at
12.4–12.5 in from grid centre.

## Corrections to the setup notes

| The note says | Measurement shows |
|---|---|
| Grid squares 5 in with 1 in borders | Confirmed — cells 5.06/5.12/5.12 in, tape 0.85–0.91 in |
| Table 18 × 18 in | Board is ~27.5 in wide × ~23.1 in deep; an 18 in board cannot carry the pattern |
| (implied) pattern has interior borders only | There is a **full outer border**; the pattern measures 18.90 in overall |
| "Table distance from table 8 inches" | 8 in is pedestal axis → near edge of the **taped pattern** (7.5 in), not the board edge |
| "railing system … opening 9 inches by 19 inches on both sides" | Confirmed, orientation resolved: 9 in is **lateral**, 19 in **fore-aft**, frame extending back behind the robot |
| (implied) rails flush with the table surface | **Wrong.** The board rests **on top of** the frame; rail tops meet its **underside** |
| (implied) a cross rail in front of the board | **No such rail.** The board's robot-side edge overhangs open air |

The railing orientation was settled by the frame's overall width. Two 19 in
lateral openings would make the frame 47 in wide against a 27.5 in board, with
rails hanging well past the table on both sides, which photo `…9F4E2977…` plainly
contradicts. At 9 in lateral the frame is 27.0 in — just inside the board.

## Rail height and the missing front rail — corrected 2026-08-28

Two assumptions baked into the first version of `FWDCenterLabMCC.yaml` were
wrong, and together they were putting a wall across the arm's only route to the
board.

**The board sits on top of the rails.** Photo `…91C25E33…` shows its laminate
edge standing proud of the aluminium beneath it. The rails had been modelled
with their tops flush with the board's *surface*, which put 25 mm of phantom
aluminium in exactly the plane the arm has to cross. Corrected: rail tops now
meet the board's underside. Board thickness ~1 in, operator-confirmed.

**There is no cross rail in front of the board.** Photo `…73910DA6…` shows that
edge finished with a wooden trim strip overhanging open air above the skirt; the
only aluminium nearby runs perpendicular, away from the camera. The operator
confirms nothing there blocks the path in or out of the arm pocket. A
`rig_rail_front` had been modelled at x ∈ [0.114, 0.152] — squarely across the
route, and the single largest obstruction in the scene. Removed.

Measured effect on the arm's clearance during the swing from the pocket onto the
board, worst point per segment:

| segment | before | after |
|---|---|---|
| fold → first swing | 3.0 mm | 26.8 mm |
| swing 1 → 2 | 1.4 mm | 23.4 mm |
| swing 2 → 3 | 1.4 mm | 25.9 mm |
| swing 3 → hover | 0.3 mm | 4.8 mm |

The one tight spot left is the **elbow crossing the board's robot-side edge**,
and it is gated on the last unmeasured number below rather than on the rig.

A side effect worth noting: lowering the rails moved the **back** rail into the
path of the backward extension, which went from 19.7 mm of clearance to 1.8 mm.
The route's backward reach has to be re-tuned against the new height.

## Still unmeasured

These need a tape measure; photographs cannot supply them.

- ~~**Shoulder height to camera height on the robot.**~~ **Resolved — corrected
  2026-10-20.** The 0.26 m torso-above-board figure here is the superseded August
  model. reachy-1-2-sim #185 (camera PnP over 17 views, and Siva's taught poses)
  and its ADR-0010 put the board surface about **0.41 m** below the shoulder line
  (0.40–0.42 m; shoulder line about 1.15 m above the floor). With the URDF camera
  about 0.155 m above the torso, that is about 0.565 m camera-above-board,
  consistent with the calibration's 0.553 m to about 1 cm, so no non-stock head
  mount is needed to explain it.
  The October layout also put the board about 0.24 m closer to the robot (torso
  at (0.206, 0, 1.15) in the scene frame).
- **Rig rail profile size and the frame's outer footprint.** Still assumed at
  1.5 in square. The rails' *height* is now resolved (tops at the board's
  underside), as is the absence of a blocking front member.
- **Board depth on the robot side.** White board on a white glossy floor gives no
  edge signal, and removing the phantom front rail removed its lower bound too —
  so this is now a genuinely open number, not a bounded one. All it has to do is
  carry the tape pattern, whose near edge is at x = 0.1905; it sits at 0.1600 in
  the scene.

  **This is the highest-value measurement outstanding.** It is the only tight
  spot left on the arm's route: the elbow crosses this edge at z = 0.770 with a
  35 mm collision radius, so at x = 0.160 it interferes by 5 mm, and at x = 0.190
  it clears by 20 mm. Tape from the pedestal axis to the board's near edge.
- **Where the opening's 19 in comes from.** With no front cross member, the
  pocket's fore-aft extent runs from the back rail to the board's near edge —
  ~20.8 in, not the 19 in the notes record. Either a front member sits further
  forward than the photos show, or the board's near edge is closer to the robot
  than 0.160. The measurement above resolves this too.
- **Table surface friction**, currently inherited from the demo scene.

## Stereo depth error on the 6 in cells (#28 C, offline)

**Only one distance band is covered: about 0.5–0.85 m (median 0.71 m).** The 2026-08-20 pictures in
`tests/Pictures/` were all taken with the board at that distance. There is no evidence here for 0.3, 0.5 (as a
centred target), 0.8 or 1.2 m; those need the ruler test (`docs/PERCEPTION_STAGE1.md`, `scripts/ruler_depth_test.py`).
Profile: `calibration/reachy12_stereo_2026_08_27.json`. Reproduce with
`PYTHONPATH=src python scripts/check_depth_error.py` (6 pairs have the grid in both eyes).

Method: the reference for each grid corner is the PnP board plane of **one eye alone** (known 5.08 in cells on a
6 in pitch, that eye's intrinsics; it does not use the stereo baseline). Stereo depth is compared with it two ways:
(a) both eyes' corners triangulated with the profile's `R`/`T`, forward depth Z in the left camera frame; (b) the
viewer's dense SGBM depth on boxes over the four tape junctions, forward Z in the right camera frame.

| Set | Corners n | Z error mean | Z error RMS | median \|error\| | max \|error\| |
|---|---|---|---|---|---|
| All 6 pairs, triangulated corners | 168 | −2.5 mm | **14.8 mm** | 6.0 mm | 76.4 mm |
| 4 pairs used to derive `R`/`T` | 108 | −3.2 mm | 12.3 mm | 5.7 mm | 72.8 mm |
| 2 held-out pairs (`_18`, `_2`) | 60 | −1.3 mm | 18.3 mm | 6.2 mm | 76.4 mm |
| All pairs, dense SGBM at the junctions | 15 | −4.8 mm | 11.7 mm | 10.4 mm | 20.5 mm |

Range (|X|) error over all corners: mean −2.6 mm, RMS 16.6 mm, max 99.8 mm. Per pair (Z error, reference Z):

| Pair | Used for pose | Reference Z | Corners | Mean | RMS | Max |
|---|---|---|---|---|---|---|
| 20.08.2026 | yes | 0.57–0.82 m | 24 | +2.1 mm | 10.5 mm | 23.3 mm |
| 20.08.2026_4 | yes | 0.57–0.85 m | 24 | −4.6 mm | 11.3 mm | 24.0 mm |
| 20.08.2026_5 | yes | 0.59–0.84 m | 24 | −4.4 mm | 10.3 mm | 31.6 mm |
| 20.08.2026_6 | yes | 0.51–0.84 m | 36 | −5.1 mm | 15.1 mm | 72.8 mm |
| 20.08.2026_18 | no | 0.53–0.83 m | 24 | −3.4 mm | 23.8 mm | 76.4 mm |
| 20.08.2026_2 | no | 0.48–0.85 m | 36 | +0.2 mm | 13.5 mm | 50.8 mm |

Reading it:

- Typical error is small (median 6 mm) and unbiased (mean −2.5 mm), but the RMS (14.8 mm) sits at the 15 mm
  target and the tail is long (up to 76 mm). The two worst pairs are the ones where row-error checks already
  flagged corners as probably mislabelled (PR #34), so part of the tail is the test reference, not the stereo.
- This is a **consistency check, not an independent bound**: the pose in the profile was derived from four of these
  six pairs. The two held-out pairs are the only fairly independent evidence, and they are the weakest (RMS 18 mm,
  n = 60 corners from two pictures of the same scene).
- It says nothing about distance dependence. By `ΔZ = Z² / (f·B) · Δd`, one pixel of disparity error is about
  2.9 / 8.0 / 20.5 / 46 mm at 0.3 / 0.5 / 0.8 / 1.2 m (f ≈ 400 px, B = 0.078 m): analytic, not measured.
- The zoom of these pictures was never recorded; the bounds apply to the profile as it is, at that zoom.
