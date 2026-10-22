# Gripper range — operator guide (#27 Part A)

The **Gripper range (Part A)** procedure in the stereo viewer's *Data capture* menu measures the right gripper
on the robot: its true range, where it stops, whether the sim's −68° release is reachable, and the force sensor's
zero and drift, with a stereo pair at every step. The goal is calibration data for reachy-1-2-sim #179, not a demo.

Owner decision, 2026-10-22:

> "Gripper-only motion is allowed in capture procedures for gripper calibration: only `r_gripper` is powered, the arm
> stays at REST, every move needs a confirm and passes a gripper gate (travel limits, low speed and torque, a
> force-limited close, `REACHY_ENABLE_MOTION`), and the gripper returns to compliant at the end."

**Part B** (the arm carries the gripper to a taught square and closes on a piece, #66) is a different procedure with
its own owner decision. **Nothing in Part A moves an arm.**

## Before you start

- The arm rests on the table (REST), as on the lab photos. The gripper is shut or nearly.
- **Hands, cables and objects clear of the gripper and the hand.** The finger swings through about 90°.
- One person at the robot with a hand near **Stop gripper** (the page) and the hardware e-stop.
- Close every notebook or other program connected to the robot.

## Safety locks on a gripper move

A gripper move is live only when **all** hold; otherwise it is a logged dry run or a refusal, and nothing is sent.

1. `REACHY_ENABLE_MOTION=true` in the viewer's environment (the launcher sets it, not with `--dry-run`).
2. The viewer was started with `--allow-gripper-motion` (the launcher passes it, not with `--dry-run`).
3. A fresh tick of *I confirm: hands clear, move the gripper to …°* for this one move (the tick clears after every press).
4. `safety.gripper_gate_check` passes:
   - the target is inside −68.8 … +20.05° (the sim's travel), and one move changes the position by at most 10° (plus 0.5° of read-back noise: a shut stop at +20.05° still reaches the first target, +10°; if Stage 0 reads further from it, the first move is refused: report the reading);
   - the speed limit is above 0 and at most 20°/s (the sweep uses 10°/s), the torque limit above 0 and at most 30 %
     (the sweep uses 20 %). **These caps are proposals; Siva confirms them before the first robot run.** The repo has
     no datasheet; `reachy_sdk` documents `speed_limit` in degrees per second and `torque_limit` in percent;
   - the present position was read (the step size cannot be checked without it).
5. Stage 0 has read the gripper (it sets the force zero), and no earlier move aborted the run.
6. No other robot command is running (one at a time; never while the neck moves).

Only the `r_gripper` joint is powered, through `joint.compliant = False`. `turn_on('r_arm')` is never called, so no
arm joint is ever stiffened. The torque and speed limits are written **before** the joint is powered.

## What stops a move

Each move is polled at 20 Hz. It ends **reached**, or one of these, and then the gripper is released
(`joint.compliant = True`):

| Rule | Limit |
|---|---|
| position not following (an end stop) | no 0.5° of progress in 1 s, short of the target — **not** an abort, the run goes on |
| temperature | above 55 °C |
| load | \|`present_load`\| above 40 (SDK units) |
| force | \|reading − Stage 0 zero\| above 3.0 (the empty sweep touched something) |
| Stop gripper | any time, no confirm needed |
| timeout | 8 s without reaching or stalling |
| error | the position cannot be read, or the SDK raises |

All but the end stop abort the run: later moves are refused ("start a new run"). The limits are in
`reachy_ai/config.py`. An unreadable load, force or temperature cannot trip its rule and is noted; the position
and the end-stop rule still protect.

## The run

1. Start the viewer: `./StereoLauncher.sh robot` (or `./StereoLauncher.sh` against the simulator). Check the page
   reads `gripper motion: live`.
2. *Data capture* → **Gripper range (Part A)**.
3. **Stage 0 — Read gripper.** Read-only. Note the position, compliance, limits, temperature, load and force. The
   force reading is the zero for the run. It also records whether the SDK has a `gripper.open()` / `close()`.
4. **Head: aim at the hand.** Confirm and *Move head* (the gated neck path; preset `gripper_view`, roll 0, pitch
   64°, yaw −38°). Both eyes should show the hand; if not, press **Stop head** and report. Capture a pair.
5. **Open sweep**, nine steps: 10, 0, −10, −20, −30, −40, −50, −60, −68°. For each: tick the confirm box, press
   *Move gripper*, wait for the read-back, press *Capture*.
6. **Shut sweep**, nine steps: −60, −50 … 0, 10, 20°. Same.
7. **End: Release gripper** (r_gripper compliant). Then *Download capture bundle*.

If a step ends `end_stop`, the gripper has reached a mechanical end short of the target: that is a result, not a
fault. The following steps toward that end will also stop immediately; carry on or skip them.

After a release the torque and speed limits stay at the sweep's low values; set them from a notebook if you need
them back.

## `gripper.open()` and `close()`

`reachy_sdk` 0.7.0 has no gripper object: `robot.r_arm` has no `gripper`, and `ReachyClient.open_gripper()` /
`close_gripper()` used to call attributes that do not exist; they now refuse live (#68). The capture therefore does not command them; the run record
says `sdk_open_close: skipped` with that reason. If a future SDK has them, Stage 0 notices and records `not_exercised`.

## Send the bundle

The zip is a browser download; keep it in `outputs/` (never commit it) and attach it to #27 and reachy-1-2-sim #179.
Analyse it yourself:

```bash
.venv/bin/python scripts/analyze_capture_bundle.py outputs/reachy-capture-gripper_range.zip
```

The analyser reports the range found, where the ends were, whether −68° was reached, where `open()` / `close()` go (or
why skipped), the force sensor's zero and drift, peak load and temperature, and every abort.

## The offline sim check that gated Part A

Run before this code existed (reachy-1-2-sim @ `1948823`, MuJoCo 3.11, scene `FWDCenterLabSivaPool`, no server, no
Docker; the sim was only read). Its numbers are in every run record's `step0.sim_check`.

- **Gripper sweep at REST** (October model, `home` keyframe, arm motors off, only `r_gripper` actuated, slowly to
  −68° and back): the moving finger and the fixed pad touch nothing at any angle. The finger stays at least 37 mm
  (MuJoCo `geomDistance`; 33 mm by an independent point-to-box distance) from the board, which it sits beside, not
  over. The hand's back and palm rest on the board's margin throughout, as ADR-0010 says.
- **Arm movement:** the sim's motors-off arm settles 0.85° in 10 s with no gripper command at all. The sweep adds at
  most 0.5° to that (0.43–0.73° measured from the settled arm). Measured from the keyframe and including that
  settling, a sweep run reaches 1.26°: above the 1° stop rule read literally, below it as arm motion caused by the
  sweep. This was a judgement call, flagged in the PR.
- **Visibility:** the gripper tip at REST is at (0.141, −0.397, −0.397) m in the torso frame. The sim's neck chain
  applies yaw last, and `gaze.project` applies it first, so the aim was computed with the model's own cameras. Inside
  the neck limits the best aim is roll 0, pitch +64°, yaw −38°: the tip is 7.5° off the optical axis, and all finger
  points are in both frames at every gripper angle (167 px margin). About half of the finger's surface is hidden by the
  hand's own palm when shut, 80 % is visible from 0° to −68°. `board_view` does not show the hand.

The real arm rests where it rests; check the hand in the live view.
