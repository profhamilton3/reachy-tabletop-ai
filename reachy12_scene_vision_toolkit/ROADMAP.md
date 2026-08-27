# Toolkit Roadmap for Claude Code

This backlog is organized as mergeable vertical slices. Do not open one giant “add MuJoCo” change. Each epic has explicit entry conditions, tasks, and exit gates.

## Global rules

- Preserve Reachy v1 SDK behavior and port `50051`.
- Keep `kinematic` mode operational throughout.
- Never mutate state from a streaming RPC.
- One component owns simulation time and state advancement.
- Every cross-process message has `protocol_version`, `sequence`, `sim_time`, and `wall_time`.
- Scene files are validated before a backend starts.
- New browser-facing ports bind to `127.0.0.1` in examples/default Compose.
- Do not overwrite existing notebook work.
- Pin all new dependencies and external repository revisions.

## EPIC 0 — Baseline, reproducibility, and safety

### R12-000 Inventory and baseline manifest

**Tasks**

- Record archive hash, Git commit, worktree status, Docker base image digest, cloned Git SHAs, pip lock, host architecture, and Docker Desktop configuration.
- Add `scripts/collect_environment.py`.
- Correct README architecture language: the current server is a custom Python v1 gRPC fake, not Pollen `reachy_sdk_server` fake mode.

**Exit gate**

- `artifacts/environment.json` can be regenerated and contains no secrets.

### R12-001 Pin external inputs

**Tasks**

- Replace moving shallow-clone heads with build arguments and default immutable SHAs.
- Add a constraints/lock file for Python dependencies.
- Record ROS package versions.

**Exit gate**

- Two clean builds resolve the same revisions.

### R12-002 CLI smoke tests

**Tasks**

- Add a non-notebook test that connects to `localhost:50051`, enumerates joints, stiffens a part, sends a bounded command, observes convergence, and exits nonzero on failure.
- Keep the notebook as a user demo, not the only acceptance test.

**Exit gate**

- Smoke test can run in CI or from the host without Jupyter.

### R12-003 Local-only port defaults

**Tasks**

- Change Compose mappings to `127.0.0.1:8888:8888`, `127.0.0.1:6080:6080`, and `127.0.0.1:50051:50051`.
- Document opt-in remote access and authentication.

**Exit gate**

- Default services are not exposed on all host interfaces.

## EPIC 1 — Authoritative backend and state snapshots

### R12-100 Introduce domain models

**Tasks**

- Create typed `JointCommand`, `JointStateSnapshot`, `CameraFrame`, `ObjectState`, `SimulationStatus`, and `BackendCapabilities` models.
- Include sequence IDs and timestamps.

**Exit gate**

- Models have validation and serialization tests.

### R12-101 Define `SimulatorBackend`

Use `contracts/backend_protocol.py` as the starting contract.

**Tasks**

- Create `KinematicBackend` from the existing joint-state logic.
- Move gRPC service behavior out of the backend.
- Keep service classes as adapters only.

**Exit gate**

- Existing joint smoke tests pass using `KinematicBackend`.

### R12-102 One simulation loop

**Tasks**

- Remove state mutation from `StreamJointsState`.
- Make one loop advance all joints at a configured fixed step.
- Publish immutable snapshots to readers.
- Add monotonic sequence and simulation time.

**Exit gate**

- Motion result is invariant to zero, one, or multiple state-stream subscribers.

### R12-103 Replace JSON-file coupling

Preferred: make a ROS bridge read snapshots directly in-process or from a bounded local IPC transport. Transitional option: atomic write/rename with a sequence number.

**Exit gate**

- ROS never reads partial JSON; skipped/late snapshots are observable metrics.

## EPIC 2 — Scene contract and RViz visualization

### R12-200 Scene JSON Schema and loader

**Tasks**

- Adopt `scenes/scene.schema.json`.
- Add YAML loading, JSON Schema validation, unique-ID validation, asset-root containment, unit checks, and helpful diagnostics.
- Resolve relative mesh paths only below approved scene/asset roots.

**Exit gate**

- `tabletop.example.yaml` validates.
- duplicate IDs, invalid quaternion, missing mesh, and path traversal fail clearly.

### R12-201 Scene service/API

**Tasks**

- Add read-only scene metadata and status endpoints.
- Add explicit `load_scene` and `reset_scene` commands with revision IDs.
- Reject scene changes while stepping unless backend supports atomic reload.

**Exit gate**

- Scene revisions and load errors are visible to SDK-side diagnostics and browser UI.

### R12-202 RViz marker adapter

**Tasks**

- Compile primitives/meshes to `visualization_msgs/MarkerArray`.
- Publish static objects with transient-local durability.
- Publish dynamic object poses at a bounded rate.
- Add labels, camera axes/frusta, and scene namespace.

**Exit gate**

- The sample table scene appears in RViz with correct scale and frame.

### R12-203 RViz configuration

**Tasks**

- Add MarkerArray display.
- Later add left/right image or camera displays.
- Keep RobotModel and TF displays.

**Exit gate**

- A fresh container opens with robot and sample scene visible without manual RViz configuration.

## EPIC 3 — Reachy v1 camera compatibility

### R12-300 Camera service contract tests

**Tasks**

- Import `camera_reachy_pb2` and `_grpc`.
- Test `GetImage`, `StreamImage`, cancellation, left/right selection, bad requests, and slow consumers.
- Test zoom/focus/autofocus simulated semantics.

**Exit gate**

- Contract tests prove encoded frames can be decoded by OpenCV.

### R12-301 Fixture camera backend

**Tasks**

- Generate deterministic left/right test patterns containing camera name, sequence, and timestamp.
- Add latest-frame buffers and configurable FPS/resolution.
- Register `CameraService` in the current gRPC server.

**Exit gate**

- Unmodified Reachy v1 SDK returns valid arrays from both `last_frame` properties.

### R12-302 ROS image publisher

**Tasks**

- Publish left/right `Image` or `CompressedImage` and `CameraInfo` with optical frame IDs.
- Use a single coherent backend snapshot per pair.
- Add image-age and drop metrics.

**Exit gate**

- `ros2 topic hz` and an RViz Image display show stable streams.

### R12-303 Browser stereo page

**Tasks**

- Add side-by-side MJPEG or WebSocket image display.
- Show backend, scene revision, FPS, frame age, and connection status.
- Keep it read-only in MVP.

**Exit gate**

- Left/right feeds are visible at a localhost URL independently of noVNC.

## EPIC 4 — Native MuJoCo server and remote backend

### R12-400 Native server skeleton

**Tasks**

- Create a separate host package/environment for current Python and pinned MuJoCo.
- Implement health, handshake, capabilities, model metadata, reset, pause, step, command, state, and frame messages.
- Launch interactive/viewer mode using `mjpython` where required.

**Exit gate**

- Docker client establishes a version-checked connection and receives heartbeat/state fixtures.

### R12-401 Reachy 1.2 MJCF model

**Tasks**

- Import/convert the pinned Reachy URDF.
- Create an explicit joint mapping table: SDK name, URDF joint, MuJoCo joint, axis, sign, units, limits, actuator.
- Preserve fixed camera frames.
- Add asset provenance and checksums.

**Exit gate**

- Single-joint sweeps match URDF/RViz direction and approximate pose.

### R12-402 Static scene compiler

**Tasks**

- Convert scene primitives/materials/lights/cameras to MJCF.
- Resolve mesh assets with scale and coordinate normalization.
- Cache compiled scenes by content hash.

**Exit gate**

- `tabletop.example.yaml` renders from both camera frames.

### R12-403 MuJoCo remote backend

**Tasks**

- Implement `MujocoRemoteBackend` with deadlines, heartbeat, reconnect, state-age checks, and explicit degraded mode.
- Map SDK commands to target/control messages.
- Map remote snapshots to gRPC/ROS/browser outputs.

**Exit gate**

- Existing motion smoke test drives the native model through the Docker gRPC endpoint.

### R12-404 Stereo RGB rendering

**Tasks**

- Render both eyes from one simulation step.
- Encode asynchronously.
- Separate physics and camera rates.
- Add resolution/FOV configuration and frame metadata.

**Exit gate**

- Head motion changes image geometry correctly and frame pairs share the same simulation step.

## EPIC 5 — Physics and interaction

### R12-500 Collision and inertial audit

**Tasks**

- Audit collision shapes and inertials; replace decorative/zero collision geometry with validated approximations where needed.
- Separate visual and collision assets.

**Exit gate**

- Robot does not explode, tunnel, or self-collide incorrectly in baseline poses.

### R12-501 Actuator and compliance model

**Tasks**

- Define target-position/velocity/torque semantics that match the v1 SDK as closely as practical.
- Map compliant/stiff states.
- Enforce limits and expose saturation.

**Exit gate**

- Step responses and limit behavior are stable and documented.

### R12-502 Gripper and contact model

**Tasks**

- Implement gripper coupling/mimic behavior.
- Expose grasp/contact state.
- Derive approximate force-sensor values with clear units and limitations.

**Exit gate**

- A cube can be contacted, lifted, released, and detected in a deterministic scenario.

### R12-503 Dynamic object state

**Tasks**

- Stream tracked object poses.
- Update RViz markers and browser overlays from backend state.
- Add reset and seeded placement.

**Exit gate**

- Object position remains coherent across MuJoCo, RViz, and camera frames.

## EPIC 6 — Fidelity and research instrumentation

### R12-600 Camera calibration ingestion

**Tasks**

- Define calibration file format and provenance.
- Support intrinsics, distortion, baseline/convergence, and resolution profiles.
- Keep explicit synthetic defaults.

### R12-601 Depth and segmentation

**Tasks**

- Add optional depth and object-ID/semantic segmentation products.
- Do not alter the v1 RGB API; expose extras through ROS/browser/research API.

### R12-602 Sensor-effect pipeline

**Tasks**

- Configurable latency, rate jitter, JPEG quality, blur, noise, exposure, dropped frames, and deterministic seed.
- Keep every effect independently disableable.

### R12-603 Recording and replay

**Tasks**

- Record commands, state snapshots, scene revision, seeds, and camera metadata.
- Add deterministic replay and artifact manifest.

### R12-604 Benchmark harness

**Tasks**

- Measure simulation real-time factor, render FPS, frame age, command-to-state latency, command-to-frame latency, CPU, memory, and dropped frames.
- Compare native MuJoCo, container headless MuJoCo, and kinematic fixture on target Apple-Silicon machines.

**Exit gate for Epic 6**

- A versioned benchmark report and reproducible research bundle are produced.

## EPIC 7 — Optional Gazebo adapter

Do not begin until the backend, scene, and camera contracts are stable.

### R12-700 Decision spike

- Identify the exact modern ROS/Gazebo distribution and native-vs-Linux deployment.
- Enumerate required legacy controller/plugin behavior.
- Build one head-and-camera proof before committing to full-body migration.

### R12-701 Scene-to-SDF adapter

- Compile the common scene contract into supported SDF.
- Maintain a documented feature-loss matrix.

### R12-702 Gazebo backend

- Implement the same `SimulatorBackend` behavior and acceptance suite.

**Exit gate**

- Gazebo passes the same SDK, camera, scene, and failure-mode tests as MuJoCo, or deviations are explicitly accepted for research reasons.

## Recommended issue/PR sequence

1. R12-000 through R12-003
2. R12-100 through R12-103
3. R12-200 and R12-202
4. R12-300 and R12-301
5. R12-302 and R12-303
6. R12-400 through R12-404
7. R12-201 and R12-203 refinements
8. R12-500 through R12-503
9. R12-600 through R12-604
10. R12-700 only after an explicit decision
