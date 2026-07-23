# IITG Reachy Project Roadmap: Real-Time AI Integration for an Untrained Task

**Project concept:** Zero-shot tabletop task assistant for Reachy 2 with one arm and vision support  
**Prepared for:** IITG Reachy / FWD Center  
**Robot context:** Reachy 2 robot, one arm, vision support, microphones available but audio treated as an optional upgrade  
**Development resources:** Docker simulator, Python SDK, Claude Code for GitHub and heavier coding/testing workflows

---

## 1. Link/resource status from the attached FAQ

I can read the PDF text and the hyperlink targets in the attached Reachy orientation FAQ. The links include:

- User Manual video
- Reachy documentation
- Pollen Robotics GitHub
- SDK documentation
- Teleoperation video, documentation, and GitHub resources
- FaceTracking documentation and video
- Idle App documentation
- Older Tic Tac Toe guide

The PDF also mentions a separate “microphone test notebook for short recordings,” but that notebook is not embedded in the uploaded PDF. I cannot inspect that notebook unless it is uploaded separately.

One important note: the FAQ’s “Reachy 2023 doc” link now lands on the current Pollen documentation site, which is explicitly the Reachy 2 documentation. The same page links to older Reachy 2023, 2021, and 2019 docs.

---

## 2. Recommended use case: Zero-shot tabletop task assistant

For the IITG Reachy project, I recommend implementing a **real-time AI-guided tabletop assistant**.

The core behavior:

> A user gives Reachy a natural-language task such as “find the item that can erase a whiteboard,” “put the recyclable item in the left tray,” or “hand me the thing that charges a phone.” Reachy looks at the table, identifies unfamiliar objects using zero-shot vision and a vision-language planner, chooses the target, asks for confirmation when confidence is low, and then uses one-arm motion primitives to point, pick, place, or hand over the object.

This is stronger than simply reproducing a fruit-sorting demo, because Pollen already has a “Greengrocer” tutorial for object detection and fruit sorting. That tutorial teaches object detection, image-to-robot-frame conversion, and moving according to what Reachy sees. Your project should generalize that idea to **unseen objects and semantic tasks**, not just predefined fruit labels.

The key definition of “not trained previously” should be:

> Reachy is not fine-tuned or manually programmed for the specific object or task category. It uses general AI perception and planning at runtime, while the actual robot movements remain controlled by safe, audited primitives.

---

## 3. Why this is feasible with the available hardware

Reachy 2’s Python SDK is intended for programming behaviors that control the arms, head, gripper, mobile base, and sensors, including cameras and microphones. Reachy’s vision stack includes head RGB cameras, a ToF depth module, and a fixed torso RGB-D camera for manipulation. The SDK camera API exposes RGB frames, depth frames, camera parameters, and extrinsics, which are exactly what we need for object grounding.

The main compute caveat is that Pollen’s hardware documentation says AI processing is expected to run on the customer’s own computers at this stage, not on Reachy’s internal CPU-only computer. Therefore, the architecture should run AI on your workstation or lab server, while Reachy provides cameras, depth, pose data, and actuation.

The simulator is useful, but with one important limitation: the current Reachy 2 simulation supports the Python SDK and ROS 2 interface, but camera access is not available yet in simulation. That means the Docker simulator is excellent for testing connection code, arm/head/gripper commands, safety gates, and motion primitives, while the vision pipeline should be tested with recorded real frames, webcam frames, saved images, or the physical robot.

---

## 4. Target demo behavior

A clean first demo would look like this:

1. Place 5-8 random tabletop objects in front of Reachy: marker, eraser, cup, cable, sticky notes, tape roll, sponge, small box.
2. User enters a task in a web UI: “Put the object used for writing in the right tray.”
3. Reachy captures a torso RGB-D frame.
4. A zero-shot detector/segmenter proposes objects and masks.
5. A vision-language planner selects the target object and destination.
6. The UI shows the selected object with a bounding box and asks for approval.
7. Reachy points to the object first.
8. Reachy picks it with the single arm, lifts it, places it in a predefined tray zone, and returns to a safe posture.
9. The system logs the task, model response, image, selected object, motion result, and success/failure.

Keep the first version to **text input plus visual confirmation**, not speech. The microphones can be added later as a voice-command upgrade. Reachy 2’s audio stack supports microphones, speaker, STT/TTS-style AI use cases, and SDK audio recording/playback, but the FWD Center setup treats sound as an upgrade.

---

## 5. System architecture

Use a layered architecture so the AI never directly sends joint angles.

```text
User task
  |
  v
Task UI / API
  |
  v
Scene capture: RGB + depth + camera params
  |
  v
Zero-shot perception
  - object detection
  - segmentation
  - depth-based 3D localization
  |
  v
AI planner
  - chooses target object
  - chooses skill: point / pick / place / handover
  - returns JSON only
  |
  v
Safety gate
  - workspace limits
  - object size/weight assumptions
  - reachable pose check
  - confidence threshold
  - human approval
  |
  v
Motion primitives
  - point_at()
  - move_to_pregrasp()
  - grasp()
  - lift()
  - place()
  - retract()
  |
  v
Reachy SDK
  - arm
  - gripper
  - head
  - cameras
```

The AI planner should decide **what** to do, not **how to move motors**. Motion should come from deterministic primitives that are tested in simulation and fake mode.

This matters because Reachy currently has limited automatic collision protection. Pollen’s fake-mode documentation explicitly says Reachy does not currently have collision-avoidance restrictions such as arm-against-torso or arm-against-arm checks. Pollen’s safety guide also says operators must stay vigilant, be ready to press emergency stop, maintain clearance, and avoid risky object handling.

---

## 6. Recommended AI stack

Use a two-level AI stack.

### 6.1 Local real-time perception loop

Use Pollen’s `pollen-vision` or a similar local zero-shot vision stack. `pollen-vision` is designed as a unified interface to zero-shot vision models for robotics. Its README shows live video object detection with OWL-ViT plus segmentation with Mobile-SAM. It also supports zero-shot object detection, segmentation, and monocular depth wrappers.

### 6.2 Event-level reasoning loop

Use a vision-language model or LLM only when the task starts, the scene changes, or the robot needs recovery. Do not call a cloud model in the inner servo loop.

Example planner output:

```json
{
  "task_type": "pick_place",
  "target_object": "whiteboard marker",
  "target_id": "obj_3",
  "destination": "right_tray",
  "confidence": 0.82,
  "requires_confirmation": true,
  "reason": "The object appears to be a marker, which matches the instruction to find the item used for writing."
}
```

The planner gets object proposals, labels, cropped images, and the user instruction. It returns a constrained JSON plan. The robot controller verifies that JSON against allowed skills and workspace constraints before moving.

---

## 7. Roadmap

### Phase 1: Project setup and resource audit

Create a GitHub repo for the project, for example:

```text
reachy-zero-shot-tabletop/
  README.md
  CLAUDE.md
  pyproject.toml
  docker-compose.yml
  src/
    reachy_ai/
      app.py
      config.py
      reachy_client.py
      perception/
      planner/
      grounding/
      motion/
      safety/
      logging/
  tests/
    fixtures/
      images/
      depth/
      planner_cases/
    test_planner_schema.py
    test_safety_bounds.py
    test_motion_dry_run.py
    test_grounding_from_fixture.py
  scripts/
    run_sim_smoke.py
    run_camera_smoke.py
    run_perception_offline.py
    run_demo_dry.py
    run_demo_robot.py
```

Pin the environment to Python 3.10 initially. The Reachy 2 SDK supports Python 3.10+, and Pollen’s docs note that the perception dependency currently needs Python 3.10.

First deliverables:

```text
- Confirmed decoded FAQ links
- README with setup
- CLAUDE.md for Claude Code
- Basic Python package
- Unit-test scaffold
- Simulator smoke-test script
- Safety policy draft
```

### Phase 2: Simulator and SDK smoke tests

Use the Docker simulator to verify that your workstation can connect to Reachy’s SDK server and command basic behaviors. Pollen’s current Docker command exposes ports 8888, 6080, and 50051.

Start with these tests:

```text
- Connect to ReachySDK host
- Print reachy.info
- Move head to a neutral posture
- Open and close gripper in dry-run or fake mode
- Move right arm to safe posture
- Return to default posture
- Stop cleanly
```

Do not test camera logic in the simulator as the main path, because camera access is not currently available in simulation. Instead, build perception tests with image/depth fixtures.

### Phase 3: Real robot sensor smoke tests

On the physical Reachy, verify:

```text
- Robot and workstation are on the same network
- Reachy IP address is known
- ReachySDK connects successfully
- Right arm or available arm is detected
- Cameras are detected
- RGB frame capture works
- Depth frame capture works
- Camera intrinsics and extrinsics can be read
```

The SDK connection docs say the computer and robot must be on the same network, Ethernet is recommended for stability, and the SDK connection uses the actual IP address rather than a `*.local` hostname.

Save representative frames into:

```text
tests/fixtures/images/
tests/fixtures/depth/
```

These fixture images let Claude Code and CI test perception code without needing the physical robot.

### Phase 4: Perception MVP

Build an offline perception script first.

Input:

```text
- RGB image
- Optional depth image
- User task text
```

Output:

```text
- Object proposals
- Bounding boxes
- Masks
- Object crops
- Optional 3D centroids
- Annotated image
```

Start with a known list of broad candidate labels:

```text
marker, pen, cup, bottle, sponge, eraser, cable, phone charger,
box, paper, sticky note, tape, tool, plastic object, metal object
```

Then add a planner step that expands task text into candidate labels.

Example:

```text
User: "Find something that can erase a whiteboard."
Candidate labels:
  - eraser
  - sponge
  - cloth
  - marker eraser
  - whiteboard eraser
```

This keeps the perception model grounded and reduces hallucinated object choices.

### Phase 5: Planner MVP with JSON contract

The planner should never return executable Python or motor values. It should return structured data only.

Example schema:

```json
{
  "task_type": "point | pick_place | handover | sort",
  "target_id": "obj_3",
  "target_description": "blue marker",
  "destination": "left_tray | right_tray | handover_zone | point_only",
  "confidence": 0.0,
  "requires_confirmation": true,
  "safety_notes": ["object appears light", "object is inside reachable area"],
  "brief_reason": "The object matches the instruction because it appears to be a marker."
}
```

Planner rules:

```text
- If confidence < threshold, ask for confirmation.
- If multiple objects match, ask the user to choose.
- If object is sharp, heavy, liquid-filled, fragile, or partly occluded, point only.
- If a human hand is near the target, do not grasp.
- If the target is outside the one-arm reachable workspace, refuse the motion and explain.
- If depth is missing or unstable, point only or ask the user to move the object.
```

### Phase 6: 2D-to-3D grounding

For each target object:

```text
1. Take the mask or bounding box.
2. Select stable depth pixels inside the mask.
3. Reject outliers.
4. Compute object centroid in camera coordinates.
5. Transform camera coordinates into Reachy coordinates using SDK extrinsics.
6. Clamp to allowed workspace.
7. Estimate grasp approach pose.
```

The SDK camera docs expose frame capture, camera parameters, depth frames, and extrinsics, which is the basis for this grounding layer.

For MVP reliability, use predefined tray positions and a calibrated tabletop plane. Avoid free-form placement at first.

### Phase 7: One-arm motion primitives

Implement motion primitives in small, auditable functions:

```text
goto_safe_posture()
look_at_table()
point_at(target_pose)
move_to_pregrasp(target_pose)
descend_to_grasp(target_pose)
close_gripper_until_contact()
lift_object()
move_to_place_pose(destination_pose)
open_gripper()
retract()
emergency_soft_stop()
```

Reachy’s arm has 7 degrees of freedom plus a gripper joint, and the SDK supports joint-space and Cartesian `goto()` commands. The SDK also offers helper functions like `translate_by()` and `rotate_by()` for relative Cartesian movements, which are useful for cautious approach/lift/retract steps.

For the first physical tests, use:

```text
- Soft foam block
- Empty cup
- Sponge
- Marker
- Small cardboard box
```

Avoid glass, liquids, scissors, sharp tools, heavy objects, cables tangled with other objects, or objects near people.

### Phase 8: Safety gate and approval UI

Before any physical movement, run a safety gate:

```text
- Is target inside allowed x/y/z workspace?
- Is target reachable by the installed arm?
- Is target above the table plane?
- Is target not too close to the torso?
- Is destination inside allowed workspace?
- Is robot in safe posture?
- Is gripper open?
- Is a human hand detected near object?
- Is operator approval present?
- Is emergency stop person ready?
```

The first UI should show:

```text
- Live/last camera frame
- Detected object boxes
- Selected target
- Planner explanation
- Proposed action
- Approve / Reject / Point only
- Stop button
```

This is also the best place to handle uncertainty. If the AI says “I think this is the charger,” the human confirms before motion.

### Phase 9: Closed-loop execution

After each stage, re-observe:

```text
Before moving:
  confirm target is still visible

After pointing:
  confirm the selected target is correct

Before grasp:
  re-check target pose

After gripper closes:
  check gripper state and object movement if available

After lift:
  confirm object moved with gripper

After place:
  confirm object appears in destination zone
```

Keep the VLM out of the fast loop. A practical split is:

```text
Fast loop:
  camera capture, object tracking, safety checks, motion state

Slow/event loop:
  task interpretation, object choice, recovery explanation
```

This makes the system feel real-time without depending on cloud-model latency for every frame.

### Phase 10: Evaluation protocol

Create a formal test matrix.

Object categories:

```text
- Writing tools
- Cleaning/erasing tools
- Paper/cardboard
- Plastic containers
- Cables/chargers
- Small boxes
- Distractor objects
```

Task types:

```text
- Point to target
- Pick target
- Place target in left/right tray
- Semantic selection: "thing used for..."
- Sorting: recyclable / desk supply / electronics accessory
```

Scene variation:

```text
- Different lighting
- Object rotations
- Partial occlusion
- Clutter
- Similar-looking distractors
```

Proposed MVP success metrics:

```text
- Target selection accuracy on unseen objects
- 3D localization error against manually measured points
- Pointing success
- Pick success
- Place success
- Number of human confirmations
- Number of safety refusals
- Average task latency
- Zero unsafe motions
```

For the first public demo, optimize for **safe, explainable success** rather than autonomy. A robot that says “I found two possible erasers; please confirm” is much better than one that confidently grabs the wrong object.

---

## 8. How to use Claude Code effectively

Use Claude Code as a development accelerator, not as the runtime robot brain. Claude Code can read a codebase, edit files, run commands, integrate with development tools, write tests, fix lint errors, resolve merge conflicts, create branches, and open pull requests.

Create a `CLAUDE.md` file like this:

```markdown
# Reachy zero-shot tabletop assistant

## Hard constraints
- Never generate code that sends raw joint angles from an LLM response.
- All physical robot movement must go through motion primitives.
- Default mode is dry-run unless REACHY_ENABLE_MOTION=true.
- Hardware tests require explicit operator approval.
- Do not log API keys, robot credentials, or private network details.
- Do not move the robot if safety.allowed_motion() returns false.
- Do not bypass tests.

## Test commands
- pytest tests
- python scripts/run_demo_dry.py
- python scripts/run_perception_offline.py --image tests/fixtures/images/table_01.png

## Architecture
- reachy_client.py wraps ReachySDK.
- perception/ creates object proposals.
- planner/ returns JSON plans.
- grounding/ converts image/depth to robot-frame poses.
- motion/ contains deterministic skills.
- safety/ gates all actions.
```

Good Claude Code tasks:

```text
- “Create the project skeleton and pytest configuration.”
- “Implement the planner JSON schema with pydantic and tests.”
- “Write a dry-run ReachyClient mock that logs commands instead of moving hardware.”
- “Write tests for safety bounds.”
- “Implement an offline perception runner that annotates fixture images.”
- “Create a GitHub Actions workflow that runs unit tests but skips hardware tests.”
- “Add a hardware test marker that only runs when REACHY_HOST and REACHY_ENABLE_MOTION are set.”
```

Claude Code is especially useful for test scaffolding, generating tests, adding edge cases, and running/fixing tests.

---

## 9. GitHub workflow

Use issues and PRs from the start:

```text
Issue 1: Repo scaffold + environment
Issue 2: Simulator SDK smoke test
Issue 3: ReachyClient dry-run wrapper
Issue 4: Camera fixture capture script
Issue 5: Offline zero-shot perception
Issue 6: Planner JSON schema
Issue 7: Safety gate
Issue 8: 2D-to-3D grounding
Issue 9: Pointing primitive
Issue 10: Pick/place primitive
Issue 11: Web approval UI
Issue 12: Demo script and evaluation report
```

CI should have three tiers:

```text
Tier 1: Always run
  - unit tests
  - schema tests
  - safety tests
  - offline perception tests with fixtures

Tier 2: Optional simulator
  - requires Docker
  - no camera assumptions
  - SDK/motion dry-run and fake-mode tests

Tier 3: Manual hardware
  - requires REACHY_HOST
  - requires REACHY_ENABLE_MOTION=true
  - requires human operator
  - never runs on normal PRs
```

---

## 10. Minimum viable demo

The smallest successful MVP is:

```text
Input:
  typed instruction: "Point to the object that can write on the board."

Robot:
  captures image
  detects tabletop objects
  selects marker
  displays selected marker for approval
  points at marker
  returns to safe posture

No grasping yet.
```

The next MVP is:

```text
Input:
  "Place the object that can erase the board in the left tray."

Robot:
  identifies eraser/sponge
  asks for confirmation
  picks it
  places it in a fixed tray position
  returns to safe posture
```

This staged approach is important:

> Pointing proves perception and planning; pick/place proves manipulation.

---

## 11. Key risks and mitigations

| Risk | Mitigation |
|---|---|
| Simulator has no camera support | Use simulator for motion and SDK tests; use saved real images/depth or webcam frames for perception tests. |
| LLM/VLM latency | Use AI for event-level decisions only; keep tracking and motion local. |
| Object detection ambiguity | Show annotated image and require confirmation below confidence threshold. |
| Depth noise | Use mask median depth, tabletop plane calibration, and reject unstable poses. |
| One-arm reach limits | Constrain table layout to a reachable zone and use fixed tray/drop poses. |
| No collision avoidance | Use fake mode, bounded primitives, slow movements, manual e-stop, and physical workspace clearance. |
| Unsafe object handling | Forbid sharp, heavy, liquid, fragile, hot, or human-adjacent objects. |
| Claude Code over-edits risky code | Use PR review, tests, dry-run defaults, and strict `CLAUDE.md` constraints. |

---

## 12. Final deliverables for IITG

The project should end with:

```text
- Working demo: zero-shot tabletop assistant
- GitHub repo with clean README
- Docker/simulator setup notes
- Reachy physical setup notes
- Safety checklist
- Web approval UI
- Offline perception test suite
- Dry-run motion test suite
- Manual hardware test scripts
- Demo video
- Evaluation report with success/failure cases
- Extension plan for audio and LeRobot data collection
```

For a later research extension, add LeRobot recording and training. Hugging Face’s LeRobot documentation includes Reachy 2 support, recording, replay, training, and evaluation workflows. That should be phase two, not the first milestone, because the immediate project requirement is real-time AI integration for a task Reachy has not been specifically trained on.

---

## 13. Suggested reference links

- Pollen Robotics documentation: <https://docs.pollen-robotics.com/>
- Reachy 2 SDK introduction: <https://docs.pollen-robotics.com/developing-with-reachy-2/sdk-introduction/discover-sdk/>
- Reachy 2 camera/image documentation: <https://docs.pollen-robotics.com/developing-with-reachy-2/basics/8-get-images-from-cameras/>
- Reachy 2 simulation introduction: <https://docs.pollen-robotics.com/developing-with-reachy-2/simulation/simulation-introduction/>
- Reachy 2 simulation installation: <https://docs.pollen-robotics.com/developing-with-reachy-2/simulation/simulation-installation/>
- Pollen Vision GitHub: <https://github.com/pollen-robotics/pollen-vision>
- Claude Code overview: <https://docs.anthropic.com/en/docs/claude-code/overview>
- Claude Code common workflows: <https://docs.anthropic.com/en/docs/claude-code/common-workflows>
- LeRobot Reachy 2 documentation: <https://huggingface.co/docs/lerobot/reachy2>
