# Reachy 1.2 Tabletop AI Assistant

Zero-shot tabletop task assistant for the **Reachy 1.2** robot (IITG / FWD Center).

> **Hardware note:** This project uses a Reachy 1.2 (2021/2023). Use `reachy_sdk`, **not** `reachy2_sdk`.

## What it does

A user gives Reachy a natural-language instruction such as *"Put the recyclable item in the left tray."* Reachy captures a stereo image, detects tabletop objects using the onboard Google Coral TPU, selects the target via a cloud vision-language planner, asks for human confirmation, then picks and places using safe motion primitives.

## Hardware

| Component | Spec |
|---|---|
| Platform | Reachy 1.2 — right arm, white |
| Arm | 7DoF + force gripper |
| Cameras | Dual 1080p, motorized zoom (FOV 65°–125°) |
| Compute | Intel NUC (internal) |
| Edge ML | Google Coral TPU (embedded) |
| Audio | ReSpeaker mic array + 10W speaker |
| VR teleoperation | Meta Quest 2 (included) |
| SDK | `reachy_sdk` (v1) on ROS 2 |

## Quick start

```bash
# 1. Install Python 3.10 environment
python3.10 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Run offline smoke test (no robot required)
python scripts/run_perception_offline.py --image tests/fixtures/images/table_01.png

# 4. Run dry-run demo (no robot required)
python scripts/run_demo_dry.py

# 5. Run unit tests
pytest tests/
```

## Robot connection (physical Reachy)

```bash
export REACHY_IP=<reachy-ip-address>
python scripts/smoke_test_all.py
```

Set `REACHY_ENABLE_MOTION=true` only when a human operator is present and the workspace is clear.

## Project structure

```
src/reachy_ai/
  app.py               FastAPI task endpoint
  config.py            environment config
  reachy_client.py     reachy_sdk wrapper
  perception/          Coral TPU detection + stereo depth
  planner/             cloud VLM reasoning (event-level only)
  grounding/           stereo depth → robot-frame pose
  motion/              deterministic motion primitives + safety gate
  audio/               ReSpeaker DoA + STT
  telemetry/           50 Hz servo/sensor logger → HDF5
scripts/               smoke tests, calibration, demo runners
tests/                 unit tests + fixture images
docs/                  wiring diagram, hardware notes, roadmap
```

## Team

| Person | Role |
|---|---|
| Terrance Hamilton | PI, planner/VLM integration |
| Siva Visveswaran | Hardware lead, SDK 1.2, Coral perception |
| Parul | Motion primitives, safety gate, approval UI |

## Documentation

- [Project Roadmap](docs/reachy_1_2_updated_roadmap.md)
- [Original Roadmap (Reachy 2 — superseded)](docs/reachy_zero_shot_tabletop_roadmap.md)
- [Wiring Diagram](docs/Full_Reachy_v1.2_wiring_diagram.pdf)
- [Siva's Hardware Notes](docs/SivaNotes_Reachy.pdf)
- [CLAUDE.md](CLAUDE.md) — constraints for AI-assisted development

## Safety

- All motion goes through `motion/primitives.py` — never raw joint angles from an LLM
- Default is dry-run; set `REACHY_ENABLE_MOTION=true` with an operator present
- Human approval required before every motion via the web UI
- Emergency stop button always visible in UI

## License

MIT
