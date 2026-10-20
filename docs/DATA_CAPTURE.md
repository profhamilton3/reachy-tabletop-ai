# Data capture — operator guide (#28)

The stereo viewer can walk you through the two physical checks of `docs/PERCEPTION_STAGE1.md` and hand
you one zip to send back. **Camera only: nothing here moves the arm, head or gripper.** The only robot
command is an optional camera **zoom level**, sent after you tick a confirm box, and only if the viewer
was started with `--allow-zoom-commands`. Without that flag a zoom button is a dry run: it is logged and
the lens does not move. Nothing is written on the server; the zip is a browser download.

## Start the viewer

```sh
export REACHY_IP=<ip>                       # never type the address into code or the page
reachy-stereo --source robot                # zoom buttons are dry-run
reachy-stereo --source robot --allow-zoom-commands   # zoom buttons really command the lens
```

Open http://localhost:8081. The simulator has no zoom, so zoom steps are refused there.

## Run a procedure

1. In **Data capture**, pick a procedure. The numbered instructions and one row per step appear.
2. For each step: do what the instruction says, then press **Capture** on that step's row.
   A step can be captured again; the zip keeps every pair (up to 5 per step, 24 per run).
3. Press **Download capture bundle**. Choosing "None" or **Discard run** forgets the run.

### Zoom field-of-view check (v1)

Robot at rest, taped 3x3 board fully in view of both cameras. For each of **OUT, INTER, IN**:
tick *I confirm: command zoom X*, press *Set zoom X*, wait for the pause to finish, press *Capture*.
(The tick must be repeated for every command.) If the viewer is dry-run, the zoom will not change; the
analyser says so. This also measures whether OUT is really the widest view.

### Depth ruler test (v1)

Robot at rest at the scene zoom (OUT); do not change zoom. Hold the flat, textured target (the 60 mm
cube face or a checker card) square to the right camera, centred in the image, at **0.3, 0.5, 0.8 and
1.2 m** measured from the right camera's lens face along its axis. Press *Capture* three times at each
distance, keeping the target still.

## Send the bundle

Attach the downloaded `reachy-capture-<procedure>.zip` to the #28 thread (or send it to Terrance). Do not
unzip or rename files inside it: the manifest holds a SHA-256 for every file and a changed file is rejected.
Each bundle holds, per step: `left.jpg`, `right.jpg`, `pair.json`; plus `capture.json` (procedure name and
version, step labels, requested and read-back zoom per step, calibration id) and `manifest.json`.

## Analyse a returned bundle (offline)

```sh
python scripts/analyze_capture_bundle.py reachy-capture-zoom_fov.zip
python scripts/analyze_capture_bundle.py reachy-capture-depth_ruler.zip --json depth.json
```

- **zoom**: the apparent cell pitch in px per level (raw frames) and its ratio to the widest level.
- **depth**: forward-depth error per distance against the ruler, shown next to the analytic sensitivity
  `dZ = Z² / (f·B)` (f ≈ 400 px, B = 78 mm: 2.9 / 8.0 / 20.5 / 46 mm per pixel of disparity at
  0.3 / 0.5 / 0.8 / 1.2 m) and the disparity accuracy the 15 mm target would need there. A miss at 1.2 m
  that equals about one pixel of disparity is reported as "record the measured bound", not as a calibration
  fault; a miss at 0.3–0.5 m is reported as a real problem. The 0.5 px figure behind that wording is a
  typical SGBM limit, not a measured property of this robot.

## No-viewer fallback

`scripts/zoom_fov_check.py` and `scripts/ruler_depth_test.py` run the same procedures from a terminal
(`REACHY_IP` set, zoom commands only with `--live-zoom` plus a typed `y` per command) and also write
`capture-bundle.zip` in the same format.
