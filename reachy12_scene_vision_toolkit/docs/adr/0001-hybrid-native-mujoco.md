# ADR-0001: Native Apple-Silicon MuJoCo with Docker Compatibility Core

- Status: Proposed
- Date: 2026-08-11
- Decision owners: IITG Reachy 1.2 simulation project

## Context

The existing Reachy 1.2 simulator runs an amd64 ROS 2 Foxy container on Docker Desktop. It provides JupyterLab, noVNC/RViz, a custom Reachy v1 gRPC server, and kinematic joint visualization. The next capability requires scene definition, stereo camera images, and eventually contact/physics.

Stock RViz can display scene markers and image topics but is not a world renderer. The old Reachy 1 Gazebo code targets Foxy/Gazebo Classic-era dependencies. Modern Gazebo on macOS would require a substantial controller/plugin/ROS migration. MuJoCo has a native macOS Python path, cameras, rigid-body dynamics, and a current Pollen Reachy 2 precedent in which a native server communicates with a Dockerized core.

## Decision

Use two cooperating processes:

1. **Docker compatibility core**
   - Reachy v1 SDK gRPC endpoint;
   - ROS 2 Foxy publishing and TF;
   - RViz/noVNC;
   - JupyterLab;
   - browser camera/status UI;
   - backend adapter and failure handling.

2. **Native macOS arm64 MuJoCo server**
   - authoritative simulation clock;
   - robot and scene model;
   - command application;
   - dynamics/collision/contact;
   - left/right RGB rendering and optional depth;
   - versioned WebSocket API.

Keep `KinematicBackend` and `FixtureBackend` in the Docker core. A future Gazebo implementation must conform to the same backend and scene contracts.

## Rationale

- Preserves the working Reachy v1/Jupyter/RViz interface.
- Keeps archived ROS/Foxy dependencies isolated.
- Runs the graphics/physics engine natively on Apple Silicon.
- Mirrors a current Pollen Reachy 2 architectural pattern.
- Allows early fixture-camera and scene-marker work before model conversion.
- Makes simulator choice replaceable rather than embedding MuJoCo in gRPC services.

## Consequences

### Positive

- Better expected Mac performance than amd64 emulated in-container rendering.
- Clear separation between compatibility API and simulation engine.
- Easy deterministic unit and contract testing with fixture/kinematic backends.
- Same images can feed SDK, ROS, RViz, browser, and research capture.
- Gazebo remains possible without reworking client-facing APIs.

### Negative

- Two launch environments and a network protocol.
- Disconnect/reconnect and version compatibility become product concerns.
- Native server packaging must be documented for macOS.
- Model and scene content must be compiled/mapped explicitly.

## Rejected alternatives

### RViz as the camera renderer

Rejected as the primary path. It would require a custom render-to-texture plugin, does not provide rigid-body physics, and ties the camera engine to Xvfb/noVNC/RViz internals.

### Browser renderer as authoritative simulator

Rejected as the primary path. WebGL is attractive for visualization but complicates authoritative state, frame capture, deterministic research runs, and server-side SDK image delivery. It may remain a secondary viewer.

### MuJoCo only inside the current amd64 container

Rejected as the default interactive Mac path. It remains useful for headless tests after benchmarking.

### Revive old Reachy Gazebo/Foxy/Classic stack

Rejected as the foundation because of archived/EOL components and branch-specific dependencies. Use it as a reference for joint/controller conventions.

### Port directly to modern Gazebo first

Deferred. It has value when Gazebo ecosystem compatibility is a hard requirement, but it is a larger migration than needed to deliver scene definition and camera frames.

## Protocol requirements

- explicit protocol version and capability negotiation;
- monotonic sequence and simulation time;
- bounded messages/queues;
- command acknowledgements or observable command sequence;
- coherent stereo frame pairs;
- heartbeat and deadlines;
- state-age/stale flags;
- reconnect and reset semantics;
- content hashes for model and scene;
- clean shutdown;
- no arbitrary code execution or remote asset fetch.

## Revisit conditions

Revisit this ADR when:

- measured native MuJoCo behavior cannot satisfy required physics/sensor fidelity;
- a collaborating research system requires modern Gazebo/SDF integration;
- Docker/Apple-Silicon graphics support changes enough to remove the native-process advantage;
- the project migrates completely off ROS 2 Foxy and Reachy v1 compatibility is no longer required.
