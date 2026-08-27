# CLAUDE.md — Reachy 1.2 Simulator

## Mission

Extend this repository into a backward-compatible Reachy 1.2 research simulator with:

- the existing Reachy v1 Python SDK and gRPC API;
- browser Jupyter notebooks;
- RViz robot and scene visualization;
- renderer-independent scene files;
- simulated left/right cameras;
- a native Apple-Silicon MuJoCo backend;
- deterministic tests and research artifacts.

Preserve the working kinematic simulator while building the new capabilities in vertical slices.

## Baseline and user work

Reviewed baseline:

- commit: `a0f02d745ce6c90e0d6e947dcdb50de82cc8e0b9`
- archive SHA-256: `af64f429e1afa75d8f98000642a15dd6f1eb6a6dcf2ae65ba6d9877f65382c1a`

At review time, these paths contained user work:

- modified `notebooks/test_motion.ipynb`
- untracked `notebooks/tlh_motion-routine.ipynb`

Never reset, overwrite, clean, reformat, or regenerate those notebooks unless explicitly assigned. Before every change, run `git status --short` and preserve unrelated modifications.

## Non-negotiable compatibility rules

1. `from reachy_sdk import ReachySDK` and `ReachySDK(host='localhost')` must remain valid.
2. Reachy v1 gRPC remains on port `50051` by default.
3. Existing motion examples must continue to work in `kinematic` mode.
4. Do not replace `reachy-sdk` with `reachy2-sdk`.
5. Do not invent a second camera API for the MVP. Implement `reachy_sdk_api.camera_reachy.CameraService`.
6. RViz is a visualization client, not the authoritative simulator.
7. A streaming RPC must never advance simulation state.
8. One authoritative loop/backend owns time and state advancement.
9. Readers consume immutable, coherent snapshots.
10. Every external dependency and model asset must be pinned and provenance-recorded.

## Architecture decision

Default target architecture:

```text
Native macOS arm64 MuJoCo server
    ⇅ versioned WebSocket protocol
Docker compatibility core (linux/amd64, ROS 2 Foxy)
    ├── Reachy v1 gRPC services
    ├── ROS publishers / TF / RViz
    ├── browser camera page
    └── JupyterLab
```

Supported backends:

- `kinematic`: fast current behavior; no rigid-body physics.
- `fixture`: deterministic state/image test backend.
- `mujoco-remote`: preferred scene, camera, and physics backend.
- `gazebo`: optional future adapter only.

Read `docs/adr/0001-hybrid-native-mujoco.md` before changing this split.

## Target source layout

Move toward this layout incrementally; do not perform a giant rename-only PR.

```text
reachy_sim/
  app.py
  config.py
  domain/
    commands.py
    snapshots.py
    camera.py
    scene.py
    status.py
  backends/
    base.py
    kinematic.py
    fixture.py
    mujoco_remote.py
  grpc/
    server.py
    joint_service.py
    sensor_service.py
    fan_service.py
    kinematics_services.py
    camera_service.py
  ros/
    bridge_node.py
    joint_publisher.py
    scene_marker_publisher.py
    camera_publisher.py
  scene/
    loader.py
    validation.py
    rviz_adapter.py
    mujoco_compiler.py
  web/
    app.py
    cameras.py
  transport/
    messages.py
    websocket_client.py
native_mujoco/
  pyproject.toml
  server.py
  model/
  assets/
scenes/
tests/
  unit/
  contract/
  integration/
  acceptance/
scripts/
docs/
```

## State and concurrency model

- The backend simulation loop is the only writer of dynamic simulation state.
- gRPC handlers submit commands through a bounded command channel and read the latest immutable snapshot.
- ROS and web publishers read the same snapshot objects.
- A stereo pair comes from one `sim_step` and one scene revision.
- Use monotonic clocks for timeouts and wall-clock UTC only for logs/artifacts.
- Never hold a global state lock while encoding an image, writing a socket, yielding gRPC data, or publishing ROS messages.
- Bound all queues. For live camera output, prefer “latest frame wins” over unbounded backlog.
- A disconnected remote backend must have an explicit status; never silently present stale data as live.

## Scene rules

The common YAML/JSON scene is authoritative. Backend-specific MJCF/SDF and RViz markers are generated views.

- Validate against `scenes/scene.schema.json` before loading.
- Enforce unique object IDs in custom validation.
- Resolve assets only below approved roots; reject `..` traversal and remote URLs by default.
- Use SI units: meters, kilograms, seconds, radians.
- Declare quaternion order explicitly as `[w, x, y, z]` in the schema and code.
- Keep visuals and collisions separate where fidelity requires it.
- Record asset source, license, checksum, scale, and axis convention.
- Scene reload is explicit and revisioned.
- Unknown fields should fail validation during development rather than being ignored.

## Camera rules

Required v1 RPCs are defined in `camera_reachy.proto`:

- `GetImage`
- `StreamImage`
- `GetZoomLevel`
- `GetZoomSpeed`
- `SendZoomCommand`
- `GetZoomFocus`
- `SetZoomFocus`
- `StartAutofocus`
- `StopAutofocus`

For MVP:

- encode RGB frames as JPEG bytes accepted by the existing SDK’s OpenCV decoder;
- return left/right frames from the correct optical frames;
- keep zoom/focus values stable and acknowledge valid operations;
- document that optical effects are not modeled yet;
- expose the same coherent frames to ROS and browser clients;
- include frame sequence, sim step, sim time, scene revision, render time, and age in internal metadata;
- never run a render operation directly in a gRPC worker.

Camera coordinate conversions require tests. ROS optical frames use +x right, +y down, +z forward. Do not “fix” mirrored or inverted frames by arbitrary image flips; fix the transform and prove it with an axis/calibration scene.

## Reachy model rules

- Maintain an explicit joint map with SDK name, UID, URDF name, MuJoCo name, axis, sign, limits, and units.
- Test one joint at a time before full-body trajectories.
- Treat current hand-entered arm geometry as approximate.
- Prefer backend/URDF model transforms for FK when MuJoCo is active.
- Do not silently change current SDK-visible limit or compliance semantics; document deliberate deviations.
- Keep physical calibration separate from synthetic defaults.

## Docker and Mac rules

- Keep the compatibility container on its known ROS 2 Foxy base until a separate migration is approved.
- Bind host ports to `127.0.0.1` by default.
- Do not assume GPU/Metal access inside a Linux container on macOS.
- Run the preferred MuJoCo renderer natively on arm64 macOS.
- Use `host.docker.internal` for Docker-to-host connectivity on Docker Desktop.
- Use `mjpython` for native interactive/viewer mode where MuJoCo’s macOS documentation requires it.
- Keep a headless/container mode only after benchmarks demonstrate its intended use.

## Development workflow

Before editing:

```bash
git status --short
git rev-parse HEAD
python3 -m py_compile fake_reachy_server.py joint_state_bridge.py
```

For every issue:

1. State the narrow behavior to add or repair.
2. Identify compatibility surfaces and failure modes.
3. Add or update tests first when practical.
4. Implement the smallest complete vertical slice.
5. Run targeted tests, then the full available suite.
6. Inspect logs for warnings, dropped frames, reconnect loops, and thread leaks.
7. Update docs and the capability matrix.
8. Report commands run, results, untested areas, and changed files.

Do not claim a Docker build or Apple-Silicon result unless it was actually run in that environment.

## Required test pyramid

### Unit

- scene validation and path safety;
- transform/coordinate conversions;
- joint mapping and unit conversion;
- command application;
- camera encoding/decoding;
- protocol serialization and version checks.

### Contract

- all Reachy v1 RPCs used by the SDK;
- left/right camera selection;
- stream cancellation and slow consumer behavior;
- remote backend handshake, timeout, reconnect, and stale-state behavior.

### Integration

- gRPC command → backend → snapshot → ROS joint state;
- backend camera → gRPC SDK array;
- backend camera → ROS image/camera info;
- scene YAML → RViz markers and MuJoCo scene;
- native server disconnect/reconnect.

### Acceptance

- original motion notebook behavior;
- scene visible in RViz and cameras;
- head motion changes view correctly;
- stereo/calibration target projection;
- deterministic reset/replay;
- target-Mac benchmark report.

Read `docs/ACCEPTANCE_TEST_PLAN.md` for exact gates.

## Quality gates

A change is not complete when:

- it only works in a notebook;
- it requires manual RViz configuration not captured in the repo;
- it changes the SDK API without approval;
- state progression depends on subscribers;
- images have no timestamp/sequence metadata internally;
- threads cannot shut down cleanly;
- queues are unbounded;
- a simulator disconnect causes an indefinite block;
- scene parsing accepts unsafe paths;
- a new dependency is unpinned;
- a test was skipped without recording why.

## Logging and metrics

Use structured fields where practical:

- backend
- protocol version
- connection state
- scene name/revision
- sim step and sim time
- state age
- camera and frame sequence
- physics step rate
- render FPS/time
- encoded frame size
- dropped/coalesced frames
- command latency
- reconnect count

Avoid per-step INFO logs. Use rate-limited warnings for recurring conditions.

## Security and research hygiene

- Do not commit secrets, tokens, private hostnames, or user paths.
- Keep notebook/VNC/API ports local by default.
- Validate all scene and protocol inputs.
- Enforce message-size limits for image and scene payloads.
- Keep a machine-readable run manifest for research results.
- Record random seeds and scene/model content hashes.
- Preserve raw benchmark output alongside summarized results.

## Commit and PR discipline

- One issue/behavior per PR when practical.
- Avoid mixing architecture refactors with visual polish.
- Include a migration note when moving modules.
- PR description must contain:
  - problem;
  - design;
  - compatibility impact;
  - tests run and results;
  - target environments actually tested;
  - screenshots or frame artifacts when visual output changes;
  - known limitations and next issue.

Suggested commit prefixes:

- `core:` backend/state architecture
- `scene:` schema/loaders/adapters
- `camera:` v1 service/render products
- `mujoco:` native server/model/bridge
- `ros:` topics/TF/RViz
- `web:` browser UI
- `test:` harnesses/fixtures
- `docs:` documentation only
- `build:` Docker/dependencies/CI

## Primary references

See `docs/RESEARCH_LINKS.md`. Prefer Pollen Robotics repositories/docs, MuJoCo docs, Gazebo docs, ROS docs, and Docker docs over third-party tutorials. When an external behavior is material, pin the source revision or record the retrieval date.
