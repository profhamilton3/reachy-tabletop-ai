# Feasibility Study: Scene Definition and Simulated Vision for Reachy 1.2

## Executive decision

The extension is **feasible**, but it is an architectural expansion rather than a small RViz feature. The current repository is a kinematic compatibility simulator: it accepts Reachy 1.x SDK commands, advances in-memory joint positions, writes a JSON snapshot, publishes `/joint_states`, and lets RViz display the URDF. It has no world state, collision model, rendering engine, image service, camera calibration model, or authoritative simulation clock.

The lowest-risk path is:

- add a renderer-independent scene definition;
- refactor the fake server behind a backend interface and one authoritative simulation loop;
- add the existing Reachy v1 `CameraService` to the gRPC server;
- publish rendered frames as ROS images so RViz can show the camera feeds;
- add a dedicated browser stereo-camera page;
- use native arm64 MuJoCo on the Mac for real rendering and physics;
- keep the present Docker image as the SDK/Jupyter/RViz compatibility layer.

A modern Gazebo port is possible, but is materially more work on Apple Silicon. The old official Reachy 1 Gazebo effort is useful as a model and controller reference, not as a durable platform: it is tied to ROS 2 Foxy, Gazebo Classic-era packages, and branch-specific dependencies.

## 1. What the uploaded repository does today

### Runtime topology

The current image is explicitly `linux/amd64` on ROS 2 Foxy (`Dockerfile:6`) and is forced to `linux/amd64` again by Compose (`docker-compose.yml:8`). Supervisord starts:

- Xvfb, Fluxbox, x11vnc, and noVNC;
- a custom Python Reachy v1 gRPC server;
- a separate JSON-to-ROS joint-state bridge;
- `robot_state_publisher` with the Reachy 1.2 URDF;
- RViz;
- JupyterLab.

The repository clones `reachy_sdk_server_2021` and `reachy_description` without pinned commits (`Dockerfile:53-54`) and intentionally skips Gazebo dependencies (`Dockerfile:59`). Python dependencies are also mostly unpinned.

### Simulation behavior

`fake_reachy_server.py` implements:

- 21 named arm, gripper, head, and antenna joints;
- joint state and command RPCs;
- force-sensor and fan stubs;
- head IK/FK;
- approximate arm IK/FK using hand-entered link lengths and L-BFGS-B optimization.

It does not import or register the Reachy camera protobuf service. The server registration at `fake_reachy_server.py:434-439` only installs joint, sensor, fan, head-kinematics, and arm-kinematics services.

The server mutates joint positions in two places:

- inside each `StreamJointsState` subscriber loop (`fake_reachy_server.py:144-157`);
- in a background interpolation loop (`fake_reachy_server.py:417-425`).

That makes motion progression dependent on the number and frequency of state subscribers and can advance a joint twice. A physics or camera extension should not be built on that timing model.

### ROS/RViz integration

The server writes `/tmp/reachy_joints.json` at 30 Hz (`fake_reachy_server.py:401-414`). `joint_state_bridge.py` reads the file and publishes `/joint_states`. Partial writes are tolerated by dropping a frame. RViz currently has only Grid, RobotModel, and TF displays (`rviz/reachy.rviz`).

This means the current VIZ is **robot-state visualization**, not world simulation.

### Operational issues to resolve before scaling

- noVNC uses `-nopw` and Jupyter disables token/password authentication;
- Compose publishes ports without a loopback-only host address;
- Git and pip inputs are not reproducibly pinned;
- the README describes the custom Python server as `reachy_sdk_server (fake mode)`, which obscures the actual implementation;
- the README references only a notebook smoke test; a repository-level CLI smoke test should be added;
- shared state does not have a clear locking/snapshot contract.

These are not blockers for research use, but they are important before adding a renderer, more threads, a remote process, and image streams.

## 2. What “scene” and “simulated vision” should mean

Use explicit capability levels so the project does not accidentally declare completion too early.

### Level S1 — scene description and visualization

A file describes a floor, table, primitive objects, mesh objects, poses, colors, lights, and semantic labels. Objects appear in RViz as `MarkerArray` messages. There is no occlusion-correct camera image and no physics.

**Feasibility:** high. This is a natural first slice.

### Level V1 — SDK-compatible camera frames

The Reachy v1 SDK returns valid left and right BGR arrays through `reachy.left_camera.last_frame` and `reachy.right_camera.last_frame`. Initially the source may be a test pattern or static fixture.

**Feasibility:** high. The v1 protobuf already transmits encoded image bytes; the client decodes them with OpenCV. No notebook API redesign is required.

### Level V2 — geometric RGB rendering

Camera images are rendered from the Reachy camera optical frames, reflect head movement, include scene occlusion, and maintain stereo geometry. The same frames are exposed through the v1 SDK, ROS image topics, and browser UI.

**Feasibility:** high with MuJoCo; medium-to-high with a custom WebGL renderer; poor fit for stock RViz alone.

### Level P1 — rigid-body interaction

Objects fall, collide, are pushed or grasped, and contact information influences gripper/force-sensor behavior.

**Feasibility:** high with MuJoCo or Gazebo, but this requires a better dynamic model, actuator mapping, collision meshes, and contact semantics.

### Level V3 — sensor fidelity

Images model measured intrinsics, stereo calibration, distortion, exposure, blur, rolling shutter, latency, compression artifacts, and noise. Depth and segmentation may be added.

**Feasibility:** technically high, but calibration-dependent. This should be last because the uploaded repository does not contain physical camera calibration.

## 3. Can RViz show what Reachy’s cameras capture?

Yes, with an important distinction.

RViz can subscribe to ROS image messages and display a camera or image panel. It can also show scene objects as markers. It does **not** automatically turn the marker/URDF view into a synthetic camera sensor. Something else must render the left and right images.

A practical UI design is:

1. Render frames in MuJoCo.
2. Expose JPEG bytes through Reachy v1 `CameraService`.
3. Publish the same frame snapshots as:
   - `/reachy/camera/left/image_raw`
   - `/reachy/camera/right/image_raw`
   - corresponding `camera_info` topics
   - optional compressed-image topics.
4. Add left and right RViz Image/Camera displays.
5. Also provide a lightweight browser page with side-by-side video, FPS, frame timestamp, scene name, and simulator status.

The browser page is recommended as the primary human camera UI because it avoids arranging image panels inside an RViz session delivered through noVNC. RViz remains valuable for TF, camera frusta, scene markers, and debugging frame alignment.

A custom RViz/OGRE render-to-texture plugin is technically possible, but it would create a second simulator-like rendering implementation without solving rigid-body physics. It also increases dependence on the Xvfb/noVNC/OpenGL stack. Do not choose it as the primary architecture.

## 4. Option comparison

Scores are relative for this project: 5 is strongest and 1 is weakest.

| Option | Scene | RGB cameras | Physics | Apple-Silicon fit | Reachy v1 integration | Maintenance risk | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| RViz markers only | 4 | 1 | 1 | 4 | 5 | 1 | Implement first for scene debugging, but not simulated vision. |
| Stock RViz Image/Camera displays plus external renderer | 5 | 5 | Depends on renderer | 4 | 5 | 2 | Required presentation layer, not the renderer itself. |
| Custom RViz/OGRE camera renderer | 3 | 4 | 1 | 2 | 4 | 5 | Technically feasible; poor strategic fit. |
| Browser WebGL/Three.js renderer | 4 | 4 | 1-2 | 5 | 3 | 3 | Good debug/education UI; avoid making browser state authoritative. |
| MuJoCo native arm64 on macOS plus Docker bridge | 5 | 5 | 5 | 5 | 5 | 3 | **Recommended.** |
| MuJoCo inside current amd64 container | 5 | 5 | 5 | 2 | 5 | 3 | Useful as headless/CI fallback; benchmark before interactive use. |
| Old Reachy 1 Gazebo/Foxy/Classic stack | 5 | 5 | 5 | 1 | 4 | 5 | Reference/archaeology only; not a new foundation. |
| Modern Gazebo native on macOS plus ROS bridge | 5 | 5 | 5 | 2-3 | 3 | 5 | Consider only when Gazebo ecosystem compatibility is a research requirement. |

## 5. MuJoCo on Apple Silicon

### Why it fits

MuJoCo has native Python bindings, loads MJCF and a supported subset of URDF, provides deterministic stepping and cameras, and can render RGB and depth. Pollen Robotics’ current Reachy 2 work is especially relevant: its native MuJoCo server can be launched with `mjpython`, while an amd64 Dockerized Reachy core connects to it over a WebSocket URL. That is nearly the same platform split needed here.

The proposed Reachy 1.2 topology is:

```text
macOS arm64 host
├── native MuJoCo process
│   ├── authoritative simulation clock
│   ├── Reachy 1.2 model and scene
│   ├── collision/contact/actuation
│   ├── left/right RGB renderers
│   └── WebSocket server :8765
│
└── Docker Desktop Linux VM, amd64 container
    ├── Reachy v1 SDK gRPC :50051
    ├── ROS bridge and robot_state_publisher
    ├── RViz/noVNC :6080
    ├── JupyterLab :8888
    └── camera browser/API :8080
```

### Benefits

- Native Apple-Silicon execution for the simulator and renderer.
- Existing Reachy notebooks and gRPC API remain unchanged.
- The Docker image can retain ROS 2 Foxy and archived Reachy 1 packages while new simulation code uses a current host Python environment.
- The native server can later be used by non-Docker clients.
- The protocol boundary makes it possible to add Gazebo or a headless backend later.

### Costs and risks

- Two processes and two environment setup paths.
- A versioned transport contract, heartbeat, reconnect behavior, and coherent snapshots are required.
- MJCF conversion and joint mapping must be verified carefully.
- Host-native viewer/rendering on macOS may require `mjpython` rather than ordinary `python` for certain interactive modes.
- The project must pin MuJoCo and asset versions and record benchmark results on the actual target Mac.

### Container-only MuJoCo

It should remain an optional mode. Docker Desktop runs Linux containers in a VM, while the current image is amd64. Do not assume Apple Metal acceleration is available to that Linux renderer. A CPU/software or headless renderer may still be adequate for low-resolution CI camera tests. Pollen’s Reachy 2 simulation documentation currently warns that its MuJoCo mode has low performance, so measure this path rather than treating it as the default interactive solution.

## 6. Gazebo comparison on Apple Silicon

### Existing Reachy 1 effort

Pollen’s `reachy_gazebo` repository confirms that Reachy 1 had a Gazebo integration. Its README targets ROS 2 Foxy and lists special branches of `reachy_description`, `reachy_controllers`, `reachy_sdk_server`, `reachy_kinematics`, `reachy-sdk`, `reachy-sdk-api`, and other packages. This is valuable evidence and a source of controller/model conventions.

It is not a drop-in for the uploaded repository. Reproducing that environment means reviving a branch matrix around an end-of-life ROS distribution and Gazebo Classic-era packages, then running the result under amd64 emulation on the Mac.

### Modern Gazebo

Modern Gazebo provides macOS installation paths, including Homebrew, but non-Ubuntu and arm combinations are generally a less-supported path than the primary Ubuntu amd64 target. Porting old Reachy 1 integration to modern Gazebo requires more than changing package names:

- migrate Classic plugins and launch assumptions;
- rebuild `ros2_control` integration against a current ROS distribution;
- reconcile the archived Reachy v1 SDK/server with the newer ROS graph;
- adapt scene/world files and sensors to current SDF/Gazebo APIs;
- bridge the result back to the v1 gRPC API;
- decide whether Gazebo runs natively on macOS or in a Linux VM/container.

### When Gazebo is justified

Choose Gazebo first only if the research requires one of these:

- direct compatibility with an existing Gazebo/SDF/ROS simulation estate;
- Gazebo-specific sensors or plugins;
- parity with a downstream team already standardized on modern Gazebo;
- experiments whose validity depends on Gazebo’s contact/sensor implementation.

For this project’s stated goal—backward-compatible Reachy 1.2 SDK, browser notebooks, browser visualization, scene definition, and camera images—MuJoCo offers a shorter and cleaner path.

## 7. Proposed architecture

### 7.1 One authoritative simulation state

Create a `SimulatorBackend` interface. The gRPC services, ROS publishers, and browser UI must read immutable snapshots from it; they must not independently integrate motion.

Initial implementations:

- `KinematicBackend`: preserves current fast fake-servo behavior.
- `MujocoRemoteBackend`: connects to the native server.
- `FixtureBackend`: deterministic test frames and states for unit/contract tests.
- future `GazeboBackend`: optional.

### 7.2 Single simulation clock

The backend owns `step()` and timestamps. State-stream subscribers never mutate state. The server publishes snapshots at requested rates, but motion progression is independent of subscriber count.

### 7.3 Renderer-independent scene contract

Author scenes in a small, validated YAML format, not directly in RViz Marker messages, MJCF, or SDF. Compile one source into:

- RViz `MarkerArray` for visualization;
- MuJoCo model additions or a generated MJCF include;
- later, SDF if Gazebo is added;
- browser metadata and semantic labels.

The initial schema should support:

- units and world frame;
- gravity, floor, background, and lights;
- unique object IDs;
- primitive and mesh geometry;
- pose, scale, material/color;
- collision and dynamics properties;
- semantic labels and tracked-object flags;
- camera resolution/FOV overrides;
- deterministic seed.

### 7.4 Reachy v1 camera service

Implement the existing protobuf service rather than creating a new notebook API. Required behavior:

- `GetImage`: return the latest encoded frame for the requested camera.
- `StreamImage`: yield encoded frames at a bounded configured rate; disconnect cleanly.
- zoom/focus/autofocus RPCs: return stable simulated values and acknowledgements; document that optical zoom/focus effects are initially not modeled.

Use a bounded latest-frame buffer, not an unbounded queue. Rendering must not block the physics loop or gRPC worker pool.

### 7.5 Camera data products

For each coherent frame snapshot provide:

- JPEG or PNG bytes for Reachy v1 gRPC;
- BGR/RGB array internally;
- ROS `Image` or `CompressedImage`;
- ROS `CameraInfo`;
- optional depth array/image;
- timestamp, sequence number, camera name, scene revision, and simulator step ID.

The Reachy description already contains left and right camera frames and ROS optical-frame rotations. Use them as the initial extrinsic source. Treat intrinsic calibration as synthetic until physical calibration data is supplied.

### 7.6 Browser UI

Add a small read-only camera page, for example `/cameras`, showing:

- left and right streams;
- frame rate and age;
- active backend and scene;
- connected/disconnected/degraded status;
- optional depth/segmentation selector later.

A simple MJPEG stream is enough for the first vertical slice. WebRTC can be deferred until latency or bandwidth measurements justify it.

## 8. Vertical slices

### Slice A — scene contract visible in RViz

Load `scenes/tabletop.yaml`, validate it, publish table/cubes as markers, and show camera frame axes/frusta. Existing motion notebook remains unchanged.

### Slice B — Reachy camera API with deterministic fixture

Register `CameraService`, generate left/right labeled test patterns, and prove the existing SDK returns valid frames. This de-risks protobuf and threading before adding graphics.

### Slice C — native MuJoCo static scene camera

Launch a native server, render the table scene from both optical frames, bridge frames to gRPC/ROS/browser, and move only the head. A head yaw/pitch command must visibly change both images.

### Slice D — full joint bridge and object physics

Map all joints, move arms and grippers, add collisions and object dynamics, and expose contact-derived sensor estimates.

### Slice E — fidelity and research instrumentation

Add calibration, depth/segmentation, noise/latency, deterministic replay, capture/export, and benchmark reporting.

## 9. Major technical risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Joint names, axes, signs, and limits differ between URDF, SDK, and MJCF | Incorrect motion and camera pose | Build a checked joint-map table; add single-joint sweep tests and golden poses. |
| Approximate IK/FK does not match the physical robot | Invalid reachability and object interaction | Use URDF/MuJoCo transforms for backend FK; retain current solver only for compatibility until validated. |
| ROS optical, OpenCV, MuJoCo, and web coordinate conventions differ | Mirrored/upside-down or stereo-inverted images | Add calibration-board scene, axis overlay, projection tests, and explicit conversion utilities. |
| Mutable state is accessed from gRPC, ROS, renderer, and transport threads | Tearing, races, nondeterminism | Single owner loop plus immutable timestamped snapshots; bounded queues. |
| Rendering slows simulation | Poor real-time factor and command latency | Separate render cadence from physics cadence; latest-frame buffers; low default resolution; metrics. |
| Host simulator disconnects | SDK hangs or stale data is mistaken for live data | Heartbeat, deadlines, state-age metadata, reconnect, explicit degraded mode, fixture/kinematic fallback. |
| Physical camera intrinsics are unknown | Geometric mismatch with real Reachy | Store configurable synthetic defaults; later ingest measured calibration and record provenance. |
| Archived/EOL dependencies drift | Build failures | Pin commits, lock Python versions, cache artifacts, and generate a software bill of materials. |
| Unauthenticated browser ports are exposed | Local network access to notebook/VNC | Bind to `127.0.0.1` by default; make remote exposure explicit and authenticated. |
| Mesh/license provenance is unclear | Distribution constraints | Keep an asset manifest with source, license, checksum, scale, and coordinate convention. |

## 10. Acceptance definition for the MVP

The scene/vision MVP is complete only when all of the following are true:

1. The original motion notebook still runs without API changes.
2. A scene YAML file is schema-validated and rejected with an actionable error when invalid.
3. The same scene objects appear in RViz markers and in rendered camera frames.
4. `reachy.left_camera.last_frame` and `reachy.right_camera.last_frame` return nonempty BGR arrays using the unmodified Reachy v1 SDK.
5. Head yaw/pitch visibly and correctly alters the camera images.
6. A known calibration target projects to expected pixels within a declared tolerance.
7. Stereo baseline/disparity direction is correct.
8. Resetting with a fixed seed reproduces the same initial state and image hash within the renderer’s declared determinism limits.
9. Loss of the native simulator produces a clear degraded/unavailable status rather than a hang.
10. Actual target-Mac benchmarks record simulation step rate, render FPS, image age, command-to-frame latency, CPU use, and memory use.

## 11. Final recommendation

Proceed, but do not start by embedding a camera renderer into RViz or reviving the old Gazebo stack.

Build in this order:

1. harden and refactor the current server around a backend/snapshot boundary;
2. add scene YAML and RViz markers;
3. implement the Reachy v1 camera service with deterministic fixture frames;
4. add ROS image topics and a browser stereo view;
5. connect a native Apple-Silicon MuJoCo server;
6. validate full-body mapping, collisions, and contacts;
7. add calibrated/noisy/depth vision last;
8. add a Gazebo adapter only when a concrete research dependency justifies it.

This produces early visible value, preserves the working Docker experience, and prevents the first camera implementation from locking the project into an obsolete simulator stack.
