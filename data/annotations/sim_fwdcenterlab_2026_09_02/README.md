# Simulator detection + classification set — FWD Center lab board

Generated 2026-09-02 from the `FWDCenterLabSiva` scene in `reachy-1-2-sim`.
Boxes come from the renderer's segmentation buffer, so they are exact rather
than drawn by hand. Nothing here was annotated by a person.

Move it wherever your pipeline wants it — nothing in this folder is referenced
by path from either repo.

```
detection/          YOLO
  images/           480x640 JPEG, 300 frames
  labels/           <class_index> <cx> <cy> <w> <h>, normalised, one per frame
  classes.txt       can, cube, cylinder, foam  (index = line number, 0-based)
  annotations/      per-frame JSON: head pose, zoom, fov, object world poses,
                    and the same boxes in pixels
classification/     one folder per class, the imprinting layout
  cube/ cylinder/ can/ foam/ empty/     80 frames each
summary.json        counts and settings, machine-readable
```

`empty/` is a real class, not a failure: the board with nothing on it.

## What matches the real captures, and how it was checked

Measured against your captures in `data/calibration/annotation/`.

| | real | sim | |
|---|---|---|---|
| frame | 480x640 portrait | 480x640 portrait | rendered 640x480 and rotated -90, the same way the SDK hands you the frame |
| board luminance | 130 | 130 | |
| object luminance | 65 | 59 | |
| object / board ratio | 0.50 | 0.47 | |
| clipped pixels | — | 0% | |
| lens | measured barrel | same coefficients | k1 -0.3163, k2 0.1027 |
| field | 61.1° over the 480 axis | same | from your August calibration |
| camera to board | ~0.5-0.6 m | 0.55 m | |

Two things drove those numbers. **Colour**: your objects are achromatic — R, G
and B agree to within a few counts on every sample I took — so the simulator's
red cube and blue cylinder would have taught a detector a feature your camera
never produces. Both are now neutral grey at the measured object/board ratio.
**Exposure**: MuJoCo has no auto-exposure and rendered the board clipped at 255,
where your camera, stopped down by the bright floor, photographs it at 130.
Lights are scaled to land on that.

Every frame is at the **calibrated zoom**. The barrel profile is measured at one
zoom and nowhere else, and applying it at the wide setting folds the frame back
on itself — three boxes in an earlier run covered 15x their object. Fixing that
properly needs the per-zoom captures in point 2b of my last email.

## What does NOT match — read this before scoring anything

**1. The objects are the wrong size, and this is the big one.** Measured against
a grid cell, which is a known 5 in / 12.7 cm: your cube spans about a third of a
cell, so roughly **4 cm**. The simulator models **6 cm** — about half a cell —
and the scene file claims those are the real dimensions. They are not. Same for
the cylinder: 7 cm modelled diameter against something nearer 4 cm.

This is eyeballed off a distorted image against a known reference, so treat it
as "clearly wrong, magnitude approximate". **A tape measure on the real cube and
cylinder settles it in ten seconds and I will re-render.** It matters beyond
vision — object width sets gripper aperture and approach clearance, so the
grasping work is carrying the same error.

**2. The background is not your lab.** Your frames show benches, monitors, a
column and a glass wall. The simulator shows a flat pale room. That is most of
every image and it is the largest domain gap after object size. Nothing short of
modelling the room or compositing real backgrounds closes it.

**3. The can and the foam block have no real reference at all.** Nobody has
photographed them, so their colour and finish here are guesses. A transfer
number for those two classes means nothing yet. **Ten frames each on the real
board would fix that** — that is the blocker on your "train with can and foam"
question, not the annotation volume.

**4. Object placement is on-cell.** Objects sit on grid cell centres with ±2 cm
jitter and random yaw, never between cells, never overlapping, never occluded by
the arm. Real clutter will be harder than this.

**5. Frames are noise-free.** No sensor noise, no motion blur, no JPEG artifacts
beyond the save. Your captures have all three.

Taken together: **treat a score on this set as an upper bound.** It is useful for
"does the model fire on the right pixels at all", not for a number you would put
in a paper.

## Regenerating

```bash
cd reachy-1-2-sim
python3 scripts/build_detection_handoff.py --out <dir> --detection 300 --per-class 80
```

Deterministic given the seed (2026 by default; `summary.json` records it). No
Docker, no running simulator, no mjpython — it renders offscreen and exits.
Roughly four minutes for the set above.

To vary it: `--detection` / `--per-class` for volume, `--seed` for a different
draw. The generator underneath
(`native_mujoco/cli/generate_dataset.py`) also takes `--zoom`, `--pitch-range`,
`--exposure` and `--drop`, which is how the single-class folders are built.

## On the classification / detection question

The set ships in both layouts so the existing classifier can be pointed at
simulator frames today, without waiting on the detector. But the reason to move
to detection stands: a class label with no pixel location cannot drive a grasp,
and grounding would otherwise fall entirely to depth. The annotation cost that
made detection unattractive is gone — this folder was produced with no manual
annotation at all, and volume is now set by render time.
