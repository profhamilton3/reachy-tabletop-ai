# Data capture — operator guide (#28)

The stereo viewer walks you through the physical checks of `docs/PERCEPTION_STAGE1.md` and hands you one zip
to send back. **Camera and neck only: nothing here moves an arm or the gripper.** Owner decision, 2026-10-20:

> "Neck motion is required for vision data capture. Capture procedures may articulate the head through the
> neck for viewing and perception. Arm motion stays forbidden in capture procedures."

So you never aim the cameras by hand: a procedure points the head at a named pose. A procedure can send two
kinds of robot command, each only after you tick a confirm box:

- a camera **zoom level**: real only if the viewer was started with `--allow-zoom-commands`;
- a **head pose** (neck only): real only if `REACHY_ENABLE_MOTION=true` **and** the viewer was started with
  `--allow-head-motion`.

Without those switches the buttons are a dry run: the command is logged, nothing moves. Nothing is written on
the server; the zip is a browser download.

## Safety locks on a head move

A head move is live only when **all** of these hold. Otherwise it is a logged dry run, or a refusal, and
nothing is sent.

1. `REACHY_ENABLE_MOTION=true` in the viewer's environment.
2. The viewer was started with `--allow-head-motion`.
3. A fresh tick of *I confirm: move head to …* for this one move (the tick clears after every press).
4. The head safety gate (`motion/safety.py`, `head_gate_check`) passes:
   - neck angles inside roll ±15°, pitch −20…+64°, yaw ±45° (the URDF travel is ±45.8° / −45.8…+64.7° / ±159.9°);
   - duration at least 1.5 s, and at least 3 s for a move to a pitch above 55°;
   - average speed (largest joint change from the neck angles read just before the move, divided by the
     duration) at most 30°/s; if the present angles cannot be read, the duration must be at least 3 s;
   - for a look-at point (not used by the procedures), the point is inside the head look-at box.
5. No other head command is running (one at a time). **Stop head** interrupts a move and holds the head where it is.
6. On `--source robot` the neck sign check passed (see *First robot session*).

Close any notebook or other program connected to the robot before moving the head from the viewer: two
clients must never command the neck at once.

Moves run in the viewer's SDK process through `motion/primitives.py`, which touches the neck only. Every move
is logged with its preset, target angles, duration and live/dry-run, and recorded in the run and the bundle,
with the neck read-back after the move.

## Start the viewer

```sh
export REACHY_IP=<ip>                       # never type the address into code or the page
reachy-stereo --source robot                # zoom and head buttons are dry-run
reachy-stereo --source robot --allow-zoom-commands   # zoom buttons really command the lens

# head moves really move the neck: both switches, with Siva at the robot
REACHY_ENABLE_MOTION=true reachy-stereo --source robot --allow-zoom-commands --allow-head-motion
```

Open http://localhost:8081. The simulator has no zoom, so zoom steps are refused there; head steps work on
the simulator's neck.

## What you see

- Under **Data capture**, next to the procedure list:
  - a red **Stop head** button, always available, with no confirm needed;
  - **Release head**, with its own confirm box;
  - `head motion: live` or `head motion: dry-run (needs REACHY_ENABLE_MOTION=true and --allow-head-motion)`.
- Each step is one row.
  - **Head step:** a tick box *I confirm: move head to board_view (0, 64, 0°)* and a **Move head** button.
  - **Zoom step:** *I confirm: command zoom X* and **Set zoom X**.
  - **Capture** on every step that takes pictures.
  - At the end of the row: the pairs captured and the last head result (`reached`, `stopped`, `dry-run` or
    `refused`) with the neck read-back in degrees.
- The line under the steps tells you what happened, or why something was refused.

A capture is refused (with the reason shown) until it is safe to take:
- **The command has not settled.** The frames must be observed at least 0.5 s after the last zoom or head
  command finished. Wait a moment and press Capture again.
- **The neck is off target.** After a live head move, each neck joint must read back within 3° of its command.
- **The head was stopped mid-move.** Move it to a preset again first.
- **A head move is still running.**

At the end of a run the last step, *End: head to neutral*, brings the head back to (0, 0, 0). The head stays
powered (stiff) there. **Release head** turns its motors off smoothly; the head may sag, so do it only with a
hand ready and the run finished.

## First robot session: approach the board view in steps

`board_view` is neck pitch +64°, 0.7° inside the neck's stop: the only pose that shows all nine cells
(reachy-1-2-sim ADR-0010). Before any head sweep or zoom check (which starts at `board_view`), run
**Board-view approach** once with Siva watching the head throughout and a hand near **Stop head**:

neutral → 20° → 55° → 60° → 64° (`board_view`) → back to neutral.

**Neck sign check.** When the viewer connects it computes the SDK head IK (`head._look_at`; nothing moves) for
two points and requires `neck_pitch > 0` for (0.3, 0.0, −0.4), i.e. looking down, and `neck_yaw > 0` for
(0.5, 0.3, 0.0), i.e. turning left. Check the result on the page under *Connection & camera diagnostics*
(`neck_sign_check`, also from `/api/stereo/status`); every bundle records it. If it failed, or the IK call
failed on `--source robot`, every live head move and release is refused with
`neck sign check failed: <detail>` (**Stop head** still works). On `--source sim` a server without head IK
records `skipped (sim)` instead.

**The 20° step.** The first move is only 20° of pitch. The head must tilt **down**; if it tilts up or
sideways, press **Stop head** and report.

Capture at each pose. Stop and report if the neck labours, sounds wrong, or reads back more than 3° short.

## Procedures

1. In **Data capture**, pick a procedure. The numbered instructions and one row per step appear.
2. For each step: move the head or set the zoom if the row asks for it, do what the instruction says, then
   press **Capture**. A step can be captured again; the zip keeps every pair (up to 5 per step, 24 per run).
3. Run the last step (head to neutral), then press **Download capture bundle**. Choosing "None" or
   **Discard run** forgets the run.

### Zoom field-of-view check (v2)

Arm at rest on the table, taped 3x3 board clear.
1. Step 0: point the head at `board_view`.
2. For each of **OUT, INTER, IN**: tick *I confirm: command zoom X*, press *Set zoom X*, wait for the pause to
   finish, press *Capture*. The tick must be repeated for every command.
3. Last step: head to neutral.

If the viewer is dry-run, the zoom does not change, and the analyser says so. This also measures whether OUT
is really the widest view.

### Depth ruler test (v2)

Arm at rest at the scene zoom (OUT); do not change the zoom.
1. Step 0 sets the head to **neutral**; it stays there for the whole test.
2. Hold the flat, textured target (the 60 mm cube face or a checker card) square to the right camera, centred
   in the image, at **0.3, 0.5, 0.8 and 1.2 m**, measured from the right camera's lens face along its axis.
3. Press *Capture* three times at each distance, keeping the target still.

### Board-view approach (v2)

See *First robot session* above. Capture once per pose.

### Head sweep (v1)

Arm at rest, the taped board clear and still, nobody near the head. Capture twice at each pose:

| Pose | roll, pitch, yaw (deg) |
|---|---|
| neutral | 0, 0, 0 |
| board_view | 0, 64, 0 |
| sweep_64_right | 0, 64, −15 |
| sweep_64_left | 0, 64, +15 |
| sweep_56_right | 0, 56, −15 |
| sweep_56_centre | 0, 56, 0 |
| sweep_56_left | 0, 56, +15 |

Then head to neutral. The five sweep poses are **not yet validated**: a head-sweep run on the simulator must
first confirm that the board stays in view at each of them. The board is not expected in view at neutral.

## Send the bundle

Attach the downloaded `reachy-capture-<procedure>.zip` to the #28 thread (or send it to Terrance). Do not
unzip or rename files inside it: the manifest holds a SHA-256 for every file, and a changed file is rejected.

Each bundle holds, per step, `left.jpg`, `right.jpg` and `pair.json`, plus:
- `capture.json`:
  - procedure name and version, step labels and the calibration id;
  - the requested and read-back zoom per step;
  - the requested head pose per step, every head command attempt (the commanded preset and angles,
    live/dry-run, the outcome, the gate's reasons, the neck read-back after the move);
  - the neck read-back frozen with each capture;
  - the run's Stop and Release events;
- `manifest.json`.

## Analyse a returned bundle (offline)

```sh
python scripts/analyze_capture_bundle.py reachy-capture-zoom_fov.zip
python scripts/analyze_capture_bundle.py reachy-capture-depth_ruler.zip --json depth.json
python scripts/analyze_capture_bundle.py reachy-capture-head_sweep.zip
```

- **zoom**: the apparent cell pitch in px per level (raw frames) and its ratio to the widest level.
  - The level is the cameras' read-back, never the requested one.
  - A capture with no read-back, or mixed levels within a step, is reported as a finding.
- **depth**: forward-depth error per distance against the ruler, next to the analytic sensitivity
  `dZ = Z² / (f·B)` and the disparity accuracy the 15 mm target would need there.
  - f (rectified, px) and B come from the calibration profile; `--focal-px` / `--baseline-m` override them.
    For the 2026-08-27 profile, f ≈ 433 px and B = 78 mm: about 2.7 / 7.4 / 19 / 43 mm per pixel of
    disparity at 0.3 / 0.5 / 0.8 / 1.2 m.
  - A miss at 1.2 m that equals about one pixel of disparity is reported as "record the measured bound",
    not as a calibration fault. A miss at 0.3–0.5 m is reported as a real problem.
  - The 0.5 px figure behind that wording is a typical SGBM limit, not a measured property of this robot.
- **head** (head sweep, board-view approach): the taped grid is found in both eyes and its centre
  triangulated with the profile. The **recorded** neck angles and the URDF head transform
  (`stereo_depth.camera_to_torso`) then carry it into the torso frame.
  - The board does not move, so the spread of that centre across head poses measures the consistency of the
    head/neck calibration.
  - A deviation over 20 mm is a finding. #185's camera check spread 6 / 4 / 9 mm.
  - So is a run where every pose has the same neck read-back (dry run).

## Robot-session checklist

Before the session:
- [ ] The viewer started with the switches you intend (see *Start the viewer*); the page shows the head and
      zoom modes you expect.
- [ ] Arm at rest on the table, board clear, nobody near the head.
- [ ] No notebook or other program is connected to the robot (two clients must never command the neck).
- [ ] The page shows `neck_sign_check: passed`.
- [ ] First session only: **Board-view approach** (starts with the 20° step) with Siva watching, before anything else.

**Neck IMU (#6): read-only, for Siva to run on the NUC.** It sends no command. Paste the output into #6.

```sh
ros2 topic list                              # find the IMU topic (likely /head/imu or /neck/imu)
ros2 topic info <imu_topic>                  # message type
ros2 topic echo <imu_topic> --once           # one message
```

Per-capture IMU recording follows once #6 has found the topic. There is no IMU code yet.

## No-viewer fallback

`scripts/zoom_fov_check.py` and `scripts/ruler_depth_test.py` run the zoom and depth procedures from a terminal
(`REACHY_IP` set; zoom commands only with `--live-zoom` plus a typed `y` per command). They also write
`capture-bundle.zip` in the same format. They never move the head: aim it with the viewer first.
