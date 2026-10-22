# Reachy 1.2 Tabletop AI Assistant — Claude Code Guide

## CRITICAL: Hardware version

This is **Reachy 1.2** — use `reachy_sdk`, **NOT** `reachy2_sdk`.
Never import from `reachy2_sdk` or `reachy2_core`. CI will catch this.

## Hard constraints

- Never generate code that sends raw joint angles from an LLM response.
- All physical robot movement must go through functions in `src/reachy_ai/motion/primitives.py`.
- Default mode is dry-run. Only set `REACHY_ENABLE_MOTION=true` on the physical robot with a human operator present.
- Never call the cloud VLM planner inside a per-frame loop — event-level only (once per task or recovery).
- Do not log API keys, robot IPs, or credentials anywhere in the codebase.
- Do not move the robot if `safety.gate_check()` returns `False`.
- Do not skip tests, disable CI, or bypass the safety gate.
- Do not import from `reachy2_sdk` under any circumstances.
- ReachyClient sends no arm or gripper motion; it refuses until the arm gate (#66). Gripper calibration motion lives only in the capture procedure (primitives.gripper_*).
- OWNER DECISION (2026-10-20): "Neck motion is required for vision data capture. Capture procedures may articulate the head through the neck for viewing and perception. Arm motion stays forbidden in capture procedures."
  - Head moves go only through `motion/primitives.py` (head-only functions) and `safety.head_gate_check()`: neck limits, minimum duration, operator approval, `REACHY_ENABLE_MOTION`.
  - In the stereo viewer a head move is live only with ALL of: `REACHY_ENABLE_MOTION=true`, `--allow-head-motion`, a fresh confirm tick per move, and a passed gate. Anything else is a logged dry run.
  - StereoLauncher.sh enables head motion and zoom commands by default (owner, 2026-10-20); REACHY_ENABLE_MOTION=true is set only on the viewer process. Code defaults stay dry-run.
- OWNER DECISION (2026-10-22), Part A: "Gripper-only motion is allowed in capture procedures for gripper calibration: only `r_gripper` is powered, the arm stays at REST, every move needs a confirm and passes a gripper gate (travel limits, low speed and torque, a force-limited close, `REACHY_ENABLE_MOTION`), and the gripper returns to compliant at the end."
  - Gripper moves go only through `motion/primitives.py` (gripper-only functions) and `safety.gripper_gate_check()`. They touch the `r_gripper` joint and nothing else: no `turn_on('r_arm')`, no `goto`, no `look_at`. Arm motion stays forbidden (Part B, #66, needs its own owner decision).
  - In the stereo viewer a gripper move is live only with ALL of: `REACHY_ENABLE_MOTION=true`, `--allow-gripper-motion`, a fresh confirm tick per move, and a passed gate. Anything else is a logged dry run. StereoLauncher.sh passes the flag by default, like the head flag; `--dry-run` removes it.

## Test commands

```bash
pytest tests/                                            # all offline unit tests
pytest tests/ -m "not hardware"                          # same, explicit
python scripts/run_demo_dry.py                           # full pipeline, no motion
python scripts/run_perception_offline.py --image tests/fixtures/images/table_01.png
```

## Hardware tests (operator required)

```bash
export REACHY_IP=<ip>
python scripts/smoke_test_all.py                         # read-only checks
python scripts/probe_sdk1.py                             # read-only SDK probe (IK/FK only, no writes)
export REACHY_ENABLE_MOTION=true
pytest tests/ -m hardware_motion                         # motion tests — human must be present
```

## SDK version

`reachy_sdk` (v1) — `pip install reachy-sdk`
Python 3.10
ROS 2

See `scripts/probe_sdk1.py` for confirmed API surface from the physical robot.
It and `scripts/smoke_test_all.py` must stay read-only (no goals, look_at, turn_on/off,
compliance, gripper or camera setters); `tests/test_probe_read_only.py` enforces it (#61).

## Architecture

| Module | Purpose |
|---|---|
| `reachy_client.py` | `reachy_sdk` wrapper; logs all calls |
| `perception/coral_detector.py` | Google Coral TPU inference (TFLite) |
| `perception/stereo_depth.py` | Stereo disparity → depth map |
| `perception/segmenter.py` | MobileSAM mask generation |
| `planner/vlm_planner.py` | Cloud VLM reasoning — event-level only |
| `planner/schema.py` | Pydantic plan schema (no raw joints) |
| `grounding/depth_to_robot.py` | Camera frame → robot frame transform |
| `motion/primitives.py` | All deterministic motion functions |
| `motion/safety.py` | Safety gate — must pass before any motion |
| `telemetry/logger.py` | 50 Hz servo/sensor → HDF5 |
| `audio/respeaker.py` | ReSpeaker DoA + 4-channel capture |

## Good Claude Code tasks

- "Implement the planner JSON schema with Pydantic and write schema tests."
- "Write a dry-run ReachyClient mock that logs commands instead of connecting."
- "Write unit tests for safety bounds checking."
- "Implement an offline perception runner that annotates fixture images."
- "Add a GitHub Actions CI workflow that runs Tier 1 tests only."
- "Write the stereo depth calibration loader and projection functions."
- "Implement force-threshold gripper close in primitives.py."
