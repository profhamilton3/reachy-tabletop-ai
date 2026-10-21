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

`StereoLauncher.sh` passes both switches and sets `REACHY_ENABLE_MOTION=true` on the viewer process only
(owner, 2026-10-20); `./StereoLauncher.sh --dry-run` starts without them to rehearse. Without those switches
the buttons are a dry run: the command is logged, nothing moves. Nothing is written on
the server; the zip is a browser download.

## Safety locks on a head move

A head move is live only when **all** of these hold. Otherwise it is a logged dry run, or a refusal, and
nothing is sent.

1. `REACHY_ENABLE_MOTION=true` in the viewer's environment (the launcher sets it; not with `--dry-run`).
2. The viewer was started with `--allow-head-motion` (the launcher passes it; not with `--dry-run`).
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
./StereoLauncher.sh robot                   # head and zoom LIVE (each command needs a confirm tick); Siva at the robot
./StereoLauncher.sh robot --dry-run         # rehearsal: head and zoom buttons only log
./StereoLauncher.sh --dry-run               # the same against the simulator
```

The launcher passes `--allow-head-motion --allow-zoom-commands` and sets `REACHY_ENABLE_MOTION=true` on the
viewer's command line only (it is never exported). Starting `reachy-stereo` directly stays dry-run unless you
pass those flags and the variable yourself.

Open http://localhost:8081. Zoom steps work on the simulator under the same rules as on the robot: a confirm
tick per command and `--allow-zoom-commands` (otherwise a logged dry run). Head steps work on the simulator's
neck. See *The simulator's zoom model* below.

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

**Order of a robot session:** Board-view approach, then Head sweep, then Depth ruler test, then **Zoom FOV
check last**. The zoom check can leave the lens at a level the SDK cannot restore (see the restore step
below), and every other test assumes the scene zoom.

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

**Sound check (v1)** (speaker and microphones, no motion; it needs the audio helper and a consent tick per recording): see `docs/SOUND_CHECK.md`.

### Robot inventory (v1)

One bundle of read-only facts that replaces most of the commands typed for #6, #7, #59 and #28 (item C).
**No motion and no writes of any kind.** The SDK is only read, and the NUC runs only the audio helper's fixed
probes. Nothing typed in the page is ever run.

Before you start, on the NUC: start the audio helper **from a terminal with ROS sourced**. Add `--imu-topic`
if the IMU topic is known. Then open the SSH tunnel (`docs/SOUND_CHECK.md` §1 and §5).

| Step | Button | What it reads |
|---|---|---|
| 1. SDK reads | **Read robot** | The SDK version; every joint's position and compliance; the gripper; the force sensor; the fans; both cameras' zoom and focus; the neck sign check (IK only). |
| 2. NUC probes | **Run NUC probes** | `audio_devices`, `audio_hw_params`, `usb`, `system`, `ros2_topics`, `imu`. Each probe has its own result: ok, failed, or not run, with the reason. |
| 3. Operator answers | **Save answers** | "Is the arm resting on the table?" (for the record only), plus a free note. |

- **Each SDK read stands alone.** A failed or stuck read (5 s limit) records its error type, and the other
  reads continue.
- **Helper unreachable.** Step 1 still works, and step 2 marks every probe "not run (helper unreachable)".
  The analyser then reports the answers as unknown, never as absent.
- **Redaction.** The NUC output is redacted twice, once by the helper and again by the viewer before
  bundling. That removes host names, IP and MAC addresses, the user name and home paths, and serial-looking
  fields. ROS topic names are kept.
- **Download.** Press **Download capture bundle** (`reachy-capture-robot_inventory.zip`). It holds
  `capture.json`, one `probes/<name>.json` per probe and `manifest.json`. Attach it to #28.

### Zoom field-of-view check (v4)

Run this last in a robot session. Arm at rest on the table, taped 3x3 board clear.
1. Step 0, the starting zoom, untouched: press *Read zoom* (read-back only; no zoom command). The level can be
   `ZERO`, the SDK's unset level (the simulator starts there); it is recorded as `ZERO`, not mapped to another
   level. The **first complete read** of the run is the starting level; later reads are kept as extra
   read-backs and never change it, and it is fixed once any zoom command has been sent. Then tick the confirm
   box, press *Move head* (to `board_view`), wait for the read-back and press *Capture*: this pair is at the
   starting zoom.
2. For each of **OUT, INTER, IN**: tick *I confirm: command zoom X*, press *Set zoom X*, wait for the pause to
   finish, press *Capture*. Capture is refused until that step's zoom command has completed. The tick must be
   repeated for every command.
3. Head to neutral.
4. Last step: tick *I confirm: command zoom restore* and press *Restore zoom*. This commands the starting
   level from step 0 and records the read-back. If the starting level is unknown, nothing is sent. If it
   cannot be commanded, the failure is recorded. `ZERO` cannot be commanded through the SDK; restarting the
   robot's camera service is expected to re-home the lens (unconfirmed). Any other level: set it by hand.

If the viewer is dry-run, the zoom does not change, and the analyser says so. This also measures whether OUT
is really the widest view. The analyser reports the starting level, the cell pitch of the starting pair and
which tested level it matches within 3% (e.g. "start ZERO matches INTER (ratio 1.00)", or "matches none"), and
flags a restore that failed, was a dry run, was not run, or had no known starting level. It groups pairs by
each capture's own zoom read-back, not by step.

### Depth ruler test (v3)

Arm at rest at the scene zoom (OUT); do not change the zoom.
1. Step 0: press *Read zoom* (no command). A warning shows if the zoom differs from the level the calibration
   profile is valid at. While the profile's `zoom_level` is null, the warning says the profile's zoom is
   unknown, so the result applies only at the zoom recorded in the bundle.
2. Step 1 sets the head to **neutral**; it stays there for the whole test.
3. Hold the flat, textured target (the 60 mm cube face or a checker card) square to the right camera, centred
   in the image, at **0.3, 0.5, 0.8 and 1.2 m**, measured from the right camera's lens face along its axis.
4. Press *Capture* three times at each distance, keeping the target still.

### The simulator's zoom model

Under `mujoco-remote` (what `SivaLabLauncher.sh` runs) the simulator models zoom as a **field-of-view change
only**:
- a zoom command changes both cameras' field of view; `INTER` is the measured lab calibration, `OUT` and `IN`
  come from Pollen's 65-125° range read as diagonal (an inference from one data point);
- lens distortion is fixed, not re-modelled per level;
- the change is instant.

The `kinematic` and `fixture` backends only store the level (the analyser then reports that the cell pitch
did not change). A zoom bundle from the simulator is marked `source: sim` in `capture.json`, and the analyser
prints "Field of view from the simulator's zoom model, not the real lens."

### Board-view approach (v2)

See *First robot session* above. Capture once per pose.

### Head sweep (v2)

Arm at rest, the taped board clear and still (no objects on it), nobody near the head. Capture twice at each pose:

| Pose | roll, pitch, yaw (deg) |
|---|---|
| neutral | 0, 0, 0 |
| board_view | 0, 64, 0 |
| sweep_yaw_right_5 | 0, 64, −5 |
| sweep_yaw_left_5 | 0, 64, +5 |
| sweep_yaw_right_8 | 0, 64, −8 |
| sweep_yaw_left_8 | 0, 64, +8 |
| sweep_roll_right_5 | −5, 64, 0 |
| sweep_roll_left_5 | +5, 64, 0 |

Then head to neutral. Sim re-shoot, 2026-10-20: with the whole-rows finder the grid was found in both eyes only
at board_view and sweep_yaw_left_5; every other sweep pose lost it in one eye, because the chest hides part of the
near row (#41). The analyser now accepts a partly hidden near row: it labels the far and middle rows only
(6 cells, 24 corners, enough for the rigid fit) and reports "6 of 9 cells (near row hidden)" for that pose. With
that, the grid is found in both eyes at every sweep pose of the sim re-shoot and each pose's board centre is
within 10 mm of the scene grid centre (spread across the 7 poses: 3.2 mm worst). The robot's bundle is
re-analysed with no new robot time. The board is not expected in view at neutral.

Why these poses (render findings, sim head sweep of 2026-10-20, #28 / #39). The v1 poses (yaw ±15° at pitch 64
and 56, plus 56° centre) were not analysable:
- **The chest clips the near row.** The robot's own body is a dark mass across the bottom of the frame, so
  below pitch 64 the near row of cells is cut off. The grid finder needs whole rows of 3 (or a cell touching the
  image edge), so a clipped row is neither. Pitch 64 is the neck stop, so the sweep varies yaw and roll only.
- **Frame margin.** At pitch 64, without the chest, the pinhole margin to the frame edge is 66 px at yaw 0,
  49 at ±5°, 38 at ±8° and only 14 at ±15°.
- **Floor checker squares and the arm.** The floor's checker squares show up as extra quads, and the arm's
  fingertip overlaps the right-hand cells in the yaw frames; both disturb the row grouping.
- Objects left on the board break the finder too (a cell with a cube or cylinder is not a clean white quad).
  Only `board_view` on a cleared board was found in both eyes.
- The analyser picks the camera geometry (robot or sim and its lens) from the bundle; sim frames are 640 × 480.

## Send the bundle

Attach the downloaded `reachy-capture-<procedure>.zip` to the #28 thread (or send it to Terrance). Do not
unzip or rename files inside it: the manifest holds a SHA-256 for every file, and a changed file is rejected.

Each bundle holds, per step, `left.jpg`, `right.jpg` and `pair.json`, plus:
- `capture.json`:
  - procedure name and version, step labels and the calibration id;
  - the requested and read-back zoom per step, with the camera source (`sim` or `robot`);
  - for the zoom check, the starting zoom and every read, command and restore record;
  - the requested head pose per step, every head command attempt (the commanded preset and angles,
    live/dry-run, the outcome, the gate's reasons, the neck read-back after the move);
  - the neck read-back frozen with each capture;
  - the run's Stop and Release events;
- `manifest.json`.

A Robot inventory bundle has no images. It holds `capture.json` (the SDK reads, each probe's outcome and
reason, the operator's answers, the helper status, and the redaction record), `probes/<name>.json` (each
probe's redacted commands and output), and `manifest.json`.

## Analyse a returned bundle (offline)

```sh
python scripts/analyze_capture_bundle.py reachy-capture-zoom_fov.zip
python scripts/analyze_capture_bundle.py reachy-capture-depth_ruler.zip --json depth.json
python scripts/analyze_capture_bundle.py reachy-capture-head_sweep.zip
python scripts/analyze_capture_bundle.py reachy-capture-robot_inventory.zip
```

- **inventory**: answers the open questions directly.
  - **#7**: whether the ReSpeaker was found (`arecord -l`, `lsusb`), its channel count, and so its firmware:
    6 channels is the 6-channel firmware, 1 is the 1-channel one.
  - **#59**: the playback devices, and which one the helper's `--play-device` points at.
  - **#6**: the IMU topic, its message type and publisher count, and one sample's fields.
  - **System**: the OS, kernel and Python versions.
  - **SDK**: the version, the joint count and every read error.
  - A probe that did not run is reported as unknown, never as absent.

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
  - When the near row is not whole (the chest hides part of it), only the far and middle rows are labelled and
    the pose is reported as "6 of 9 cells (near row hidden)". A quad that does not fit the known 3 × 3 layout
    (floor checker squares, the arm: size, spacing, alignment) is left out, and a near-row cell must be seen
    to show that the hidden row is the near one. This is opt-in for the head analyser only: the calibration
    derivation and the zoom check keep the whole-rows finder.
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

**Neck IMU (#6): read-only, for Siva to run on the NUC.** It sends no command. The **Robot inventory** runs
these for you (its `ros2_topics` and `imu` probes; start the helper with `--imu-topic` once the topic is known).
Typing them by hand is the fallback. Paste the output into #6.

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
