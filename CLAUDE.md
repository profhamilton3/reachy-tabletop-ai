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
export REACHY_ENABLE_MOTION=true
pytest tests/ -m hardware_motion                         # motion tests — human must be present
```

## SDK version

`reachy_sdk` (v1) — `pip install reachy-sdk`
Python 3.10
ROS 2

See `scripts/probe_sdk1.py` for confirmed API surface from the physical robot.

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
