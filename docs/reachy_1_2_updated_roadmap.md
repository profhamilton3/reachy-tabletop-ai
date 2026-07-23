# IITG Reachy 1.2 Project Roadmap: Real-Time AI Integration for an Untrained Tabletop Task

**Robot:** Reachy 1.2 (Reachy Starter Kit + VR Teleoperation — Color: White, Right Arm)
**SDK:** `reachy_sdk` (legacy v1 package — NOT `reachy2_sdk`)
**Team:** Terrance Hamilton · Siva Visveswaran · Parul · (FWD Center / IITG)
**Collaboration:** GitHub code sharing, CI-gated PR workflow
**Prepared:** 2026-07-22 (revised from original Reachy 2 roadmap per Siva's email, Jul 9)

> **Critical correction from Siva (Jul 9):** The physical robot is a **Reachy 1.2 (2021/2023)**,
> not Reachy 2. Per Pollen/HuggingFace: *"reachy2_sdk is only compatible with a Reachy 2 robot.
> If your Reachy is from 2021 or 2023 (Reachy 1.2), you should use the reachy_sdk package."*
> The original roadmap assumed Reachy 2 APIs, Docker simulation, and SDK 2 features — all require
> revision. The Greengrocer demo was built for SDK 2 and will not run as-is; Siva has built a
> sample to probe what is available.

---

## Hardware Inventory (Confirmed from Siva's Email)

| Component | Specification |
|---|---|
| Platform | Reachy Humanoid Platform + VR Teleoperation — v1.2 |
| Arm | 1 × 7DoF bio-inspired right arm with force gripper |
| Head cameras | Dual 1080p @ 30 fps, motorized zoom (FOV 65° to 125°) |
| Compute | NUC (internal x86 computer) |
| Edge ML accelerator | **Google Coral TPU** (embedded) |
| Audio in | ReSpeaker microphone array (multi-channel) |
| Audio out | 10W speaker |
| Software | ROS 2 + Python SDK (`reachy_sdk`) |
| VR teleoperation | Meta Quest 2 application (included) |
| Display | Arduino-driven torso display |
| Color | White |

---

## Wiring Architecture (from Full_Reachy_v1.2 Wiring Diagram)

Understanding the hardware topology is essential for writing correct SDK code and for
identifying data channels. The diagram reveals five subsystems:

### Head
- XL-320 servo motors (id30, id31) for antenna actuation — DXL-XL-320 protocol
- USB2AX board → DXL HUB → fans and antenna servos
- Hub USB 4 ports → Camera Right, Zoom controller, Camera Left
- 12V → 7.5V converter for head supply

### Neck
- **Pacman Board**: 3× magnetic encoders for pinion position + **IMU**
- **Houston Board**: Motor controller — reads Pacman encoders + hall sensors, sends PWMs to MajorTom; DXL device
- **MajorTom Board**: Powers and drives 3× BLDC motors; **temperature sensing**; fans control
- USB Prog interface for firmware access

### Torso (Central Hub)
- **NUC**: HDMI, RJ45, 12V IN, On/Off — main compute node
- **Power Supply Board**: distributes 12V throughout robot
- **Google Coral TPU**: connected via USB2AX to NUC
- **2× USB2AX boards**: Dynamixel bus bridges
- USB 4-port hub + USB 3 hub
- **Arduino**: drives torso display
- **Audio Amplifier**: Mic (ReSpeaker) → NUC; NUC → Speaker Left + Speaker Right (Audio Jack)
- Mobile base connector; USBC hub for keyboard/mouse

### Right Arm (installed)
Dynamixel servo chain: MX106 (id10,11) → MX64 (id12) → MX106 (id13) → MX28 (id14) →
MX28 (id15) → MX26 (id16) → MX28 (id17) → **Force Sensor** in gripper
- DXL HUB + DXL-Fan Module for bus and thermal management
- 12V Relay for arm power switching
- Load + DXL Module (id10) at gripper

### Left Arm (wired on robot, not in kit purchase)
Same topology as right arm: MX106 (id20,21) → MX64 (id22) → MX106 (id23) → MX28 (id24) →
MX28 (id25) → MX26 (id26) → MX28 (id27) → **Force Sensor** in gripper
- Future expansion path if second arm is procured

---

## Data Engineering Opportunities (from Wiring Diagram Analysis)

The Reachy 1.2 hardware exposes significantly richer data channels than the original roadmap
assumed. Each represents an engineering opportunity.

### 1. Google Coral TPU — Local Edge Inference
The Coral is connected directly to the NUC via USB. It can run TensorFlow Lite models at
high speed without cloud calls.

**Opportunities:**
- Run MobileNet / EfficientDet-Lite for zero-shot object detection on-device
- Run MobileNet-SSD with custom labels for tabletop task objects
- Eliminate cloud VLM calls from the fast perception loop — use Coral for detection,
  cloud VLM only for task reasoning
- Latency target: <50 ms per frame on Coral vs. 500+ ms for cloud round-trip

**Action:** Benchmark `pycoral` runtime on the Reachy NUC. Port or quantize the OWL-ViT
or DETR detection head to TFLite format for Coral deployment.

### 2. IMU (Pacman Board in Neck)
The neck Pacman board exposes a 6-axis IMU alongside the 3 magnetic encoders.

**Opportunities:**
- Track head orientation in world frame for gaze-aligned task context
- Detect head nod / shake as implicit confirmation signal
- Log head pose during manipulation for attention map analysis
- Fuse with arm pose for full-body kinematic logging

**Action:** Confirm IMU topic on ROS 2 (`/head/imu` or similar). Add to sensor logger.

### 3. Force Sensors (Both Grippers)
Both left and right grippers have dedicated force sensors (soldered wire connection visible
in wiring diagram).

**Opportunities:**
- Detect grasp success without vision (object contact = non-zero force)
- Measure grasp force to protect fragile objects
- Log contact events as ground-truth manipulation labels for dataset creation
- Implement force-controlled grasping: close gripper until threshold, not by position
- Compliance estimation: known force + known deflection → stiffness of object surface

**Action:** Identify force sensor channel in `reachy_sdk` (likely via gripper joint load
feedback or a dedicated sensor topic). Calibrate zero-offset and full-scale.

### 4. ReSpeaker Microphone Array
The multi-element ReSpeaker is connected to the NUC audio chain (via mic input to audio
amplifier, routed to USB or I2C).

**Opportunities:**
- Direction of Arrival (DoA) detection: know which direction a user is speaking from
- Beamforming: focus pickup toward detected speaker, suppress ambient noise
- Wake-word detection running locally on Coral (keyword spotting model in TFLite)
- Multi-channel audio logging synchronized with video for multimodal dataset
- NLU pipeline: ReSpeaker → local STT → task instruction → planner

**Action:** Install `respeaker` Python driver. Stream 4-channel audio. Expose DoA angle
to the task planner for user-proximity awareness.

### 5. Dual Motorized Zoom Cameras (Stereo Pair)
Two 1080p cameras with motorized zoom (FOV 65°–125°). This is a key departure from
Reachy 2's fixed cameras.

**Opportunities:**
- **Stereo depth estimation**: the two cameras form a stereo pair — compute disparity map
  for 3D object localization without a separate depth sensor
- **Active zoom**: zoom in for fine object identification; zoom out for scene context
- **Foveated attention**: wide FOV for scene survey, narrow FOV for selected target
- **Gaze-controlled zoom**: command zoom level based on head/arm pose + detected object size

**Action:** Measure stereo baseline (distance between cameras). Calibrate intrinsics +
extrinsics with a checkerboard. OpenCV `StereoSGBM` or RAFT-Stereo as depth backend.

### 6. Dynamixel Servo Telemetry (All Axes)
Every servo has a unique ID (mapped in wiring diagram). Dynamixel protocol exposes per-servo:
position, velocity, load, voltage, temperature, error flags.

**Right arm IDs:** 10, 11, 12, 13, 14, 15, 16, 17
**Head IDs:** 30, 31
**Neck:** via Houston/MajorTom BLDC controllers

**Opportunities:**
- Log full arm state at every control timestep → kinematic demonstration dataset
- Detect joint overload or position error early (safety monitoring)
- Predictive maintenance: rising temperature trend signals bearing or motor issue
- Replay logged trajectories for offline analysis and motion primitive refinement
- LeRobot-compatible dataset format (episode → observation → action)

**Action:** Add a background telemetry thread that reads all servo states at ~50 Hz and
writes to a timestamped HDF5 or MCAP file.

### 7. Temperature Sensors (MajorTom Board)
The MajorTom board in the neck monitors temperature for BLDC motor protection.

**Opportunities:**
- Continuous thermal monitoring dashboard
- Automatic motion pause when temp exceeds threshold
- Log temperature alongside task execution for thermal load profiling

### 8. Arduino + Torso Display
The Arduino drives a display panel mounted on the torso, connected via USB to the NUC.

**Opportunities:**
- Show task status, current object target, confidence score
- Display safety state (green/yellow/red)
- Show planner output in human-readable form for demo presentations
- Heartbeat indicator for operator situational awareness

**Action:** Write a simple serial protocol from NUC → Arduino to drive display state
from the task manager.

### 9. LeRobot Dataset Pipeline
With force sensors, servo telemetry, stereo video, IMU, and audio all accessible:

**Opportunities:**
- Record human teleoperation demonstrations (via Meta Quest 2) as training episodes
- Create a Reachy 1.2 dataset in the LeRobot HDF5 format
- Train behavior cloning or diffusion policy on collected demos
- Benchmark zero-shot vs. trained performance on the same tabletop tasks
- Contribute dataset to HuggingFace Hub for community use

**Note:** HuggingFace's `lerobot` library added Reachy 2 support; Reachy 1.2 requires
a custom teleop adapter, but the data format is the same.

---

## SDK Migration: `reachy_sdk` vs `reachy2_sdk`

This is the most critical code change required. **Do not use `reachy2_sdk` imports.**

### Installation
```bash
# Reachy 1.2 — correct
pip install reachy-sdk

# Reachy 2 — WRONG for this robot
# pip install reachy2-sdk
```

### Connection
```python
# Reachy 1.2 (reachy_sdk)
from reachy_sdk import ReachySDK
reachy = ReachySDK(host="<REACHY_IP>")

# Reachy 2 style (DO NOT USE)
# from reachy2_sdk import ReachySDK
```

### Arm access
```python
# Reachy 1.2
reachy.r_arm.shoulder_pitch.goal_position = 0.0
reachy.r_arm.elbow_pitch.goal_position = -90.0

# Force gripper
reachy.r_arm.gripper.open()
reachy.r_arm.gripper.close()
```

### Head / cameras
```python
# Reachy 1.2 — cameras accessed via OpenCV or ROS 2 topic, not SDK
import cv2
cap_right = cv2.VideoCapture(0)  # verify device index on NUC
cap_left  = cv2.VideoCapture(2)

# Head motors
reachy.head.look_at(x=1, y=0, z=0, duration=1.0)
```

### Simulation note
The Reachy 2 Docker simulator is **not compatible** with Reachy 1.2. For offline
development, use:
- Recorded real camera frames (fixture images)
- Mock `ReachySDK` class that logs calls instead of connecting
- ROS 2 bag replay if a ROS 2 bag was captured from the physical robot

---

## System Architecture (Revised for Reachy 1.2)

```
User task (text or voice)
  |
  +-- [Voice path] ReSpeaker array → DoA → local STT → task text
  |
  v
Task UI / API  (FastAPI or Gradio web app on NUC or workstation)
  |
  v
Scene capture
  - Camera Right + Camera Left (OpenCV, 1080p)
  - Motorized zoom control (wide for scene, narrow for target)
  - Stereo disparity → depth map  (no separate depth sensor needed)
  |
  v
Zero-shot perception  [runs on Google Coral TPU via pycoral]
  - TFLite object detection (EfficientDet-Lite / OWL-ViT quantized)
  - Segmentation mask (MobileSAM lite)
  - 3D centroid from stereo depth
  |
  v
AI planner  [cloud VLM — Claude / GPT-4V — event-level only]
  - Receives: object proposals, crops, user instruction
  - Returns: constrained JSON plan (target_id, action, confidence)
  |
  v
Safety gate
  - Workspace bounds check
  - Force sensor baseline check
  - Human approval (UI confirm button)
  - Emergency stop ready
  |
  v
Motion primitives  [reachy_sdk — deterministic functions]
  - goto_safe_posture()
  - look_at_table()
  - point_at(target_pose)
  - move_to_pregrasp(target_pose)
  - close_gripper_until_force(threshold_N)
  - lift_object()
  - move_to_place_pose(dest_pose)
  - open_gripper()
  - retract()
  - emergency_soft_stop()
  |
  v
Telemetry logger  [background thread — 50 Hz]
  - Servo states (position, velocity, load, temp) → HDF5
  - Force sensor readings
  - IMU (head orientation)
  - Camera frames (keyframes)
  - Audio events (DoA, wake word)
  - Task events (start, confirm, success, fail)
```

---

## GitHub Repository Structure

```
reachy-tabletop-ai/
  README.md
  CLAUDE.md
  pyproject.toml
  docker-compose.yml          # for workstation dev (not Reachy 2 sim)
  .github/
    workflows/
      ci.yml                  # Tier 1 unit tests — no hardware
      sim.yml                 # Tier 2 offline/mock tests
  src/
    reachy_ai/
      app.py                  # FastAPI task endpoint
      config.py               # env vars: REACHY_IP, REACHY_SDK_VERSION, etc.
      reachy_client.py        # wraps reachy_sdk (v1)
      perception/
        coral_detector.py     # pycoral TFLite inference
        stereo_depth.py       # stereo camera depth estimation
        segmenter.py          # MobileSAM lite
      planner/
        vlm_planner.py        # cloud VLM reasoning
        schema.py             # Pydantic plan schema
      grounding/
        depth_to_robot.py     # stereo → camera → robot frame transform
      motion/
        primitives.py         # all motion functions
        safety.py             # safety gate
      audio/
        respeaker.py          # DoA + multi-channel capture
        stt.py                # speech-to-text adapter
      telemetry/
        logger.py             # 50 Hz servo + sensor logger → HDF5
        display.py            # Arduino torso display driver
  tests/
    fixtures/
      images/                 # captured from physical Reachy cameras
      depth/                  # stereo disparity maps
      planner_cases/
    test_planner_schema.py
    test_safety_bounds.py
    test_motion_dry_run.py
    test_grounding_from_fixture.py
    test_coral_detector.py
    test_stereo_depth.py
    test_force_sensor.py
  scripts/
    run_camera_smoke.py       # verify both cameras, zoom control
    run_stereo_calibration.py # checkerboard intrinsics + extrinsics
    run_coral_benchmark.py    # measure Coral inference latency
    run_imu_stream.py         # verify IMU topic from neck
    run_force_calibration.py  # zero and calibrate force sensor
    run_audio_doa.py          # verify ReSpeaker DoA
    run_perception_offline.py # test with fixture images
    run_demo_dry.py           # full pipeline, no motion
    run_demo_robot.py         # full pipeline, physical robot
    record_episode.py         # LeRobot-format teleoperation recording
  data/
    episodes/                 # recorded demonstrations (gitignored)
    calibration/              # stereo calibration files
```

---

## Roadmap Phases

### Phase 0: Team Sync and Repo Bootstrap *(Week 1 — Terrance + Siva + Parul)*

**Goal:** Align on hardware reality and establish shared codebase.

- [ ] Siva shares his sample SDK probe code → merge to `main` under `scripts/probe_sdk1.py`
- [ ] Run Siva's sample; document which SDK 1 modules/APIs are available
- [ ] Create GitHub repo `reachy-tabletop-ai` — Terrance owns, Siva + Parul as collaborators
- [ ] Set up branch protection: PRs required, CI must pass, one review required
- [ ] Add `CLAUDE.md` with hard constraints (see below)
- [ ] Add `pyproject.toml` pinning `reachy-sdk`, `pycoral`, `opencv-python`, `scipy`, `pydantic`
- [ ] Document confirmed SDK 1.2 API surface in `README.md`

**Issues to open on GitHub:**
```
#1  Repo scaffold + environment (Python 3.10, pyproject.toml, CI)
#2  reachy_sdk smoke test script (connect, read joints, dry-run)
#3  Stereo camera calibration script
#4  Google Coral benchmark
#5  IMU stream verification
#6  Force sensor calibration
#7  ReSpeaker DoA test
```

---

### Phase 1: Physical Hardware Smoke Tests *(Week 1–2 — Siva leads, Terrance reviews)*

**Goal:** Confirm all data channels are accessible from the NUC.

#### 1.1 SDK connection
```python
from reachy_sdk import ReachySDK
reachy = ReachySDK(host="<REACHY_IP>")
print(reachy.r_arm.joints)        # verify right arm
print(reachy.head.joints)         # verify head
```

#### 1.2 Camera verification
```python
import cv2
cap = cv2.VideoCapture(0)         # verify index
ret, frame = cap.read()
assert frame.shape == (1080, 1920, 3)
# Repeat for second camera (left)
# Test zoom: send zoom command and confirm FOV change
```

#### 1.3 Stereo baseline measurement
Physically measure distance between camera optical centers. Record in
`data/calibration/stereo_params.json`.

#### 1.4 Google Coral verification
```bash
python3 -c "from pycoral.utils import edgetpu; print(edgetpu.list_edge_tpus())"
# Expected: [{type: 'usb', path: '/dev/bus/usb/...'}]
```

#### 1.5 IMU stream
```bash
ros2 topic list | grep imu
ros2 topic echo /head/imu --once
```

#### 1.6 Force sensor
```python
# Check gripper force reading at rest (should be near 0)
reachy.r_arm.gripper.force_sensor  # or equivalent v1 API
```

#### 1.7 ReSpeaker
```python
import pyaudio
# List devices, identify ReSpeaker (4-channel input)
```

**Deliverable:** `scripts/smoke_test_all.py` — runs all checks, prints PASS/FAIL per channel.
Captured fixture images saved to `tests/fixtures/images/`.

---

### Phase 2: Stereo Depth Pipeline *(Week 2–3 — Terrance leads)*

**Goal:** Replace the Reachy 2 ToF depth sensor with stereo depth from the two onboard cameras.

1. Run `scripts/run_stereo_calibration.py` (OpenCV checkerboard method)
2. Save `stereo_params.json` (intrinsics L/R + extrinsics)
3. Implement `perception/stereo_depth.py`:
   - Rectify stereo pair
   - Compute SGBM disparity
   - Convert disparity → depth map
   - Project to 3D point cloud
4. Validate against a ruler-measured object at known distances (0.3, 0.5, 0.8, 1.2 m)
5. Save fixture stereo pairs and depth maps to `tests/fixtures/depth/`

**Target accuracy:** <15 mm error at 0.5 m working distance.

---

### Phase 3: Google Coral Perception *(Week 3–4 — Siva leads)*

**Goal:** Run object detection locally on the embedded Coral TPU.

1. Download `EfficientDet-Lite2` TFLite model from TensorFlow Hub
2. Run `scripts/run_coral_benchmark.py` — measure latency per frame
3. Implement `perception/coral_detector.py`:
   - Load model via `pycoral.adapters.detect`
   - Run detection on RGB frame
   - Return bounding boxes + class labels + confidence scores
4. Add MobileSAM-lite (if Coral memory allows) or run SAM on NUC CPU
5. Integrate `label_expansion()` in planner: user task → candidate labels
6. Run offline on fixture images; save annotated outputs to `tests/fixtures/`

**Target latency:** <80 ms end-to-end detection on Coral (vs. 400+ ms on NUC CPU alone).

---

### Phase 4: reachy_sdk Motion Primitives *(Week 4–5 — Parul leads, Siva reviews)*

**Goal:** Build and test deterministic motion functions using the correct v1 SDK.

All primitives default to **dry-run mode** unless `REACHY_ENABLE_MOTION=true`.

```python
def goto_safe_posture(reachy, duration=2.0): ...
def look_at_table(reachy, duration=1.5): ...
def point_at(reachy, target_pose_robot_frame, duration=2.0): ...
def move_to_pregrasp(reachy, pose, duration=2.0): ...
def close_gripper_until_force(reachy, force_threshold_N=3.0): ...
def lift_object(reachy, lift_height_m=0.10, duration=1.5): ...
def move_to_place_pose(reachy, dest_pose, duration=2.0): ...
def open_gripper(reachy): ...
def retract(reachy, duration=1.5): ...
def emergency_soft_stop(reachy): ...
```

Note for Reachy 1.2: use `reachy_sdk` joint `goal_position` assignment and
`reachy.turn_on('r_arm')` / `reachy.turn_off('r_arm')` for power control.
Verify exact API against Siva's probe results.

**Test matrix** (all in dry-run first, then physical with observer):
- [ ] Each primitive executes without exception
- [ ] Joint targets are inside safe workspace bounds
- [ ] `emergency_soft_stop()` executes in <200 ms
- [ ] Force sensor stops gripper at threshold

---

### Phase 5: Cloud VLM Planner *(Week 5–6 — Terrance leads)*

**Goal:** Task reasoning via Claude API (event-level only — not in the fast loop).

Planner receives: object proposals from Coral, depth-localized centroids, user instruction.
Returns constrained JSON:

```json
{
  "task_type": "pick_place | point | handover | sort",
  "target_id": "obj_2",
  "target_description": "whiteboard eraser",
  "destination": "left_tray | right_tray | handover_zone | point_only",
  "confidence": 0.87,
  "requires_confirmation": false,
  "safety_notes": ["object appears light", "inside reachable zone"],
  "brief_reason": "The sponge-like object matches the eraser task description."
}
```

Rules enforced by safety gate (not by VLM):
- confidence < 0.7 → requires_confirmation forced true
- object outside workspace bounds → task_type forced to "point_only"
- human hand detected near object → refuse motion
- no stable depth reading → refuse grasp, request object repositioning

**Claude API usage pattern:**
```python
# Use claude-sonnet-4-6 for vision + reasoning
# Call ONCE per task initiation, not per frame
# Pass: base64-encoded image crop + object list + user instruction
# Cost guard: log token usage per session
```

---

### Phase 6: Telemetry Logger *(Week 6 — Siva leads)*

**Goal:** Record all sensor streams for dataset creation and debugging.

Background thread at 50 Hz:
```python
@dataclass
class RobotState:
    timestamp: float
    joint_positions: dict[str, float]   # all servo IDs
    joint_velocities: dict[str, float]
    joint_loads: dict[str, float]
    joint_temps: dict[str, float]
    force_right: float                   # gripper force sensor
    head_imu: np.ndarray                 # [ax, ay, az, gx, gy, gz]
    task_state: str                      # 'idle | detecting | planning | executing | done'
```

Write to HDF5 per episode. Compatible with LeRobot dataset format for future training.

---

### Phase 7: Safety Gate + Approval UI *(Week 7 — Parul leads)*

**Goal:** No motion without gate clearance and operator approval.

Safety gate checklist (all must pass):
- [ ] Target 3D pose inside allowed workspace (x, y, z bounds from calibration)
- [ ] Target reachable by right arm IK check
- [ ] Target above table plane (z > table_z + 20 mm)
- [ ] Target not too close to torso (safety margin)
- [ ] No human hand detected in robot workspace
- [ ] Force sensor at zero (gripper open, no unexpected contact)
- [ ] Robot in safe posture before motion
- [ ] Operator clicked APPROVE in UI

Web UI (Gradio or FastAPI + HTML, accessible on NUC local network):
```
┌─────────────────────────────────────┐
│  REACHY 1.2 TASK ASSISTANT          │
│─────────────────────────────────────│
│  Task: [text input]        [Submit] │
│─────────────────────────────────────│
│  [Camera frame with bounding boxes] │
│  Target: whiteboard eraser (87%)    │
│  Action: pick → left_tray           │
│  Reason: "Matches erasing task"     │
│─────────────────────────────────────│
│  [APPROVE]  [POINT ONLY]  [REJECT]  │
│─────────────────────────────────────│
│  [  EMERGENCY STOP  ]               │
└─────────────────────────────────────┘
```

---

### Phase 8: ReSpeaker Voice Command Path *(Week 8 — Optional upgrade)*

**Goal:** Enable voice-driven task input as an alternative to the text UI.

1. Capture 4-channel audio from ReSpeaker
2. Run Direction of Arrival → identify speaker direction → log alongside task
3. Run wake word detection on Coral (`hey reachy` keyword model in TFLite)
4. On wake word: record utterance → run STT (local Whisper.cpp or cloud Whisper API)
5. Pass transcribed text to task planner — same pipeline as typed input
6. Respond via 10W speaker (TTS: `pyttsx3` local or ElevenLabs API)

---

### Phase 9: Closed-Loop Execution *(Week 8–9)*

Re-observe after each motion stage:

```
Before motion:    confirm target still visible in camera frame
After pointing:   confirm human approval maintained
Before grasp:     re-check target pose from fresh stereo frame
After gripper close: check force sensor > 0 (object grasped)
After lift:       check force sensor sustained (not dropped)
After place:      confirm object appears in destination zone
After retract:    confirm force sensor near 0 (object released)
```

**Fast loop (NUC, ~30 Hz):** camera capture, servo state, force sensor, safety checks
**Slow loop (event-driven):** VLM planner calls, recovery decisions, UI updates

---

### Phase 10: LeRobot Dataset Recording *(Week 9–10 — Optional research extension)*

Use Meta Quest 2 VR teleoperation (included in kit) to record human demonstrations.

```python
# record_episode.py
# 1. Operator teleoperates via Meta Quest 2
# 2. System records: stereo video, servo states, force, IMU, audio
# 3. Saves in LeRobot HDF5 episode format
# 4. Episodes can be uploaded to HuggingFace Hub
```

Train behavior cloning or diffusion policy on collected episodes.
Compare zero-shot (Phase 5 VLM planner) vs. trained policy on same tasks.

---

### Phase 11: Evaluation and Demo *(Week 10–12)*

**Object set** (physical robot, tabletop):
- Marker, eraser, sponge, cup, cable, sticky notes, tape roll, small cardboard box

**Task types:**
```
Point:      "Point to the object used for writing."
Pick/place: "Put the recyclable item in the left tray."
Semantic:   "Hand me the thing that charges a phone."
Sort:       "Put all desk supplies on the right and everything else on the left."
```

**Success metrics:**
```
Target selection accuracy (vs. human label)
3D localization error (vs. ruler measurement)
Pointing success rate
Grasp success rate (force sensor confirmed)
Place success rate (visual confirmation)
Avg task latency (input → motion complete)
Coral inference latency (per frame)
Human confirmations per task
Safety gate refusals (zero unsafe motions = goal)
```

---

## GitHub Collaboration Workflow

### Roles
| Person | Role |
|---|---|
| Terrance Hamilton | PI / repo owner / planner + VLM integration |
| Siva Visveswaran | Hardware lead / SDK 1.2 expert / Coral perception |
| Parul | Motion primitives / safety gate / approval UI |

### Branch strategy
```
main            — protected; CI must pass; one approval required
dev             — integration branch for ongoing work
feature/ISSUE#  — one branch per GitHub issue
```

### PR rules
- Every PR references an issue (`closes #N`)
- CI Tier 1 (unit tests, schema, safety tests) must pass on every PR
- Hardware tests are manual (`REACHY_ENABLE_MOTION=true`) and never run on PRs
- Siva reviews motion/SDK PRs; Terrance reviews planner/AI PRs; Parul reviews UI/safety PRs

### CI Tiers
```
Tier 1 (always, no hardware):
  pytest tests/test_planner_schema.py
  pytest tests/test_safety_bounds.py
  pytest tests/test_motion_dry_run.py
  pytest tests/test_grounding_from_fixture.py
  pytest tests/test_coral_detector.py      # uses fixture images
  pytest tests/test_stereo_depth.py        # uses fixture depth

Tier 2 (manual, requires NUC):
  pytest tests/ -m hardware --no-motion    # SDK connect, read-only

Tier 3 (manual, operator present):
  pytest tests/ -m hardware_motion         # requires REACHY_ENABLE_MOTION=true
```

---

## CLAUDE.md (add to repo root)

```markdown
# Reachy 1.2 Tabletop AI Assistant

## CRITICAL: Hardware version
This is Reachy 1.2 — use `reachy_sdk`, NOT `reachy2_sdk`.
Do not import from `reachy2_sdk` or `reachy2_core`.

## Hard constraints
- Never generate code that sends raw joint angles from an LLM response.
- All physical robot movement must go through motion primitives in motion/primitives.py.
- Default mode is dry-run; set REACHY_ENABLE_MOTION=true only on physical robot with operator.
- Never call the cloud VLM planner inside a per-frame loop.
- Do not log API keys, robot IPs, or credentials.
- Do not move if safety.gate_check() returns False.
- Do not bypass tests or CI.

## Test commands
pytest tests/
python scripts/run_demo_dry.py
python scripts/run_perception_offline.py --image tests/fixtures/images/table_01.png

## SDK version
reachy_sdk (v1) — see scripts/probe_sdk1.py for confirmed API surface
Python 3.10

## Architecture modules
- reachy_client.py        → reachy_sdk wrapper
- perception/coral_detector.py → Coral TPU inference
- perception/stereo_depth.py   → stereo depth from dual cameras
- planner/vlm_planner.py       → cloud VLM (event-level only)
- motion/primitives.py         → deterministic motion functions
- motion/safety.py             → safety gate
- telemetry/logger.py          → 50 Hz sensor logging → HDF5
- audio/respeaker.py           → DoA + voice command
```

---

## Key Risks and Mitigations (Revised)

| Risk | Mitigation |
|---|---|
| `reachy2_sdk` imports in old code | Audit all imports; `grep -r reachy2_sdk src/` in CI |
| No compatible simulator for v1.2 | Use mock ReachySDK + fixture images for all offline testing |
| Stereo depth less accurate than ToF | Calibrate carefully; use SGBM + confidence mask; validate with ruler |
| Coral model not compatible | Use pycoral-compatible TFLite (.tflite); quantize to int8 |
| SDK 1 API differs from roadmap | Siva's probe script is authoritative; update docs as we confirm |
| Camera device indices change on reboot | Use udev rules to fix camera device paths by USB port |
| Force sensor calibration drift | Re-zero at startup; log baseline continuously |
| ReSpeaker driver version conflicts | Pin exact driver version in pyproject.toml |
| LeRobot format not yet for v1.2 | Write custom episode recorder; target HF-compatible HDF5 |
| VLM latency degrades UX | Gate VLM behind "thinking…" spinner; never block fast safety loop |
| Only one arm available | Constrain table layout to right-arm reachable zone; document clearly |

---

## Final Deliverables

```
Working demo:         zero-shot tabletop pick/place on Reachy 1.2
GitHub repo:          reachy-tabletop-ai (public, clean README)
SDK guide:            reachy_sdk v1 API reference (confirmed from probe)
Calibration:          stereo camera calibration files
Coral models:         quantized TFLite detection model for Reachy NUC
Fixture dataset:      stereo images + depth + planner test cases
Safety checklist:     operator pre-run checklist (PDF)
Web approval UI:      Gradio app, runs on NUC, accessible on LAN
Telemetry logger:     HDF5 per-episode files, 50 Hz
Teleoperation demo:   at least 10 recorded Meta Quest 2 episodes
Evaluation report:    success/fail matrix, latency table, Coral vs. CPU comparison
Demo video:           narrated screen + robot camera recording
Extension plan:       LeRobot training, left arm addition, voice command upgrade
```

---

## Reference Links

- Reachy 1.2 / reachy_sdk: <https://github.com/pollen-robotics/reachy-sdk>
- Pollen Robotics docs (Reachy 1.2 section): <https://docs.pollen-robotics.com/>
- Google Coral / pycoral: <https://coral.ai/docs/accelerator/get-started/>
- EfficientDet-Lite TFLite models: <https://coral.ai/models/object-detection/>
- ReSpeaker Python driver: <https://github.com/respeaker/usb_4_mic_array>
- OpenCV stereo calibration: <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>
- LeRobot dataset format: <https://huggingface.co/docs/lerobot/>
- Claude API (for VLM planner): <https://docs.anthropic.com/en/api/getting-started>
- Claude Code workflows: <https://docs.anthropic.com/en/docs/claude-code/common-workflows>
