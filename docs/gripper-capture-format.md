# Gripper capture format — `format_version: 1`

The raw capture of the **Gripper range** viewer procedure (#27 Part A). This repo owns the format; the analysis
and the calibration profile live in reachy-1-2-sim #179. **Any change to this document's rules bumps
`format_version`.** An analyser accepts only versions it knows, so a mismatch is reported with a reason rather than
misread. When the telemetry logger ("50 Hz servo/sensor → HDF5") is built it either adopts this format or defines
`format_version: 2` with a converter.

## The bundle

The viewer's *Download capture bundle* zip (the same container as every capture, `manifest.json` lists each file with
its SHA-256):

| File | What |
|---|---|
| `manifest.json` | the container manifest: procedure `gripper_range` v1, versions, every file's size and SHA-256 |
| `capture.json` | the run summary: `kind: "gripper"`, `format_version`, one row per step (its gripper records **without** samples, and its captured pairs), `gripper_aborted`, `gripper_events` |
| `gripper/run_record.json` | the JSON run record (below) |
| `gripper/samples.csv` | every sample of every move (below) |
| `steps/NN_<step id>/capK/{left,right}.jpg`, `pair.json` | the stereo pair per step (the container's standard pair files) |

Nothing is written on the viewer's server: the bundle is built in memory and downloaded.

## Units and sign conventions

- **Angles are degrees, exactly as `reachy_sdk` 0.7.0 reports them** for `joints.r_gripper` (`present_position`,
  `goal_position`). **The sign is not assumed:** the simulator runs −68.8° (open) to +20.05° (shut), and the capture
  exists to find out what the robot does. The sweep targets are the sim's convention (open = more negative); a robot
  whose sign differs shows up as an end stop, a refusal, or a range on the other side of 0.
- `load` is the SDK's `present_load`, unscaled (the SDK documents it as unreliable); `force` is the gripper force
  sensor's reading (`force_sensors.r_force_gripper.force`), unscaled, units unconfirmed on the robot. Both are
  empty when unreadable.
- `temperature_c` is degrees Celsius; `torque_limit_pct` is percent; `speed_limit_deg_s` is degrees per second.
- Time is seconds on the viewer SDK process's monotonic clock, shifted so the first sample of the run is 0. The wall
  clock of the run's start is `utc_time` in the run record.

## `gripper/samples.csv`

UTF-8, `\n` line endings, one header row, one row per poll (about 20 Hz while a move runs), moves in step order.

| Column | Meaning |
|---|---|
| `step_id` | the procedure step (`gripper_open_01` … `gripper_shut_09`) |
| `attempt` | 1, 2, … when the step's move was repeated |
| `time_s` | seconds, three decimals, 0 at the run's first sample |
| `target_deg` | the goal sent for this move |
| `present_deg` | `present_position` |
| `load` | `present_load`, or empty |
| `force` | the force sensor, or empty |
| `temperature_c` | `temperature`, or empty |
| `torque_limit_pct` | the torque limit in force |
| `speed_limit_deg_s` | the speed limit in force |

The first rows of each move are taken before the gripper has moved; the analyser uses them as the rest readings for
the force sensor's drift.

## `gripper/run_record.json`

| Key | Meaning |
|---|---|
| `format_version` | `1` |
| `kind` | `"gripper"` |
| `units` | the unit text above, for a reader without this document |
| `sdk_version` | `reachy-sdk` as installed in the viewer (0.7.0) |
| `viewer_version` | the `reachy-ai` package version |
| `utc_time` | the run's start, `YYYY-MM-DDTHH:MM:SSZ` |
| `script` | `reachy_ai.perception.procedures:gripper_range_procedure` |
| `script_sha256` | SHA-256 of `procedures.py` as run |
| `code_sha` | the checkout's `git rev-parse HEAD`, or `null` |
| `procedure` | `{id, name, version}` |
| `stage_parameters` | the sweep's speed and torque limits, the caps, the sample rate, the maximum step, the travel limits, the abort limits (temperature, \|load\|, \|force − zero\|), the following tolerance, stall window and progress, the step timeout, the open and shut ladders, the head-aim neck angles |
| `step0` | `{stage0, sim_check}`: Stage 0's read-only reads from the robot (`position_deg`, `goal_deg`, `compliant`, `torque_limit_pct`, `speed_limit_deg_s`, `temperature_c`, `present_load`, `force`, `sdk_open_close`, `errors`) and the offline sim check that gated Part A (`docs/gripper-capture.md`) |
| `force_zero` | Stage 0's force reading, the zero every move was checked against (`null` if unreadable) |
| `sdk_open_close` | `{outcome, note}`: where the SDK's `gripper.open()` / `close()` go. `reachy_sdk` 0.7.0 has no gripper object, so `skipped` with the reason |
| `aborted` | the outcome that ended the run's moves, or `null` |
| `gripper_events` | Stop records between steps |
| `files` | the paths of the samples and this record |

## Move outcomes (`capture.json` → step → `gripper_commands[].outcome`)

| Outcome | Meaning | Gripper afterwards |
|---|---|---|
| `reached` | within the following tolerance of the target | powered, holding |
| `end_stop` | the position stopped following short of the target (a mechanical end) | released |
| `temperature`, `load`, `force` | a reading went over its limit | released, run aborted |
| `stop` | Stop gripper | released, run aborted |
| `timeout` | neither reached nor stalled in time | released, run aborted |
| `error` | the position could not be read, or the SDK raised | released, run aborted |
| `refused` | the gate or the run refused; nothing was sent | unchanged |
| `dry-run` | the gate passed but the viewer or `REACHY_ENABLE_MOTION` does not allow live moves; nothing was sent | unchanged |

A `refused` or `dry-run` move has no samples.
