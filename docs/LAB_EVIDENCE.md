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

The railing orientation was settled by the frame's overall width. Two 19 in
lateral openings would make the frame 47 in wide against a 27.5 in board, with
rails hanging well past the table on both sides, which photo `…9F4E2977…` plainly
contradicts. At 9 in lateral the frame is 27.0 in — just inside the board.

## Still unmeasured

These need a tape measure; photographs cannot supply them.

- **Shoulder height to camera height on the robot.** Calibration puts the camera
  0.553 m above the board, but the official URDF puts it only 0.155 m above the
  torso, and the reach cross-check (which matches Siva's physical check of the
  two unreachable cells) pins the torso at 0.26 m above the board. Those three
  cannot all be right — most likely this rig's head mount is non-stock.
- **Rig frame rail height above the board, and its outer footprint.** The frame
  is in the sim as a collision obstacle, but its profile size and footprint are
  assumed; only the opening dimensions are pinned down.
- **Board depth on the robot side.** White board on a white glossy floor gives no
  edge signal. It is currently bounded, not measured: it must clear the frame's
  front rail and still carry the tape pattern.
- **Table surface friction**, currently inherited from the demo scene.
