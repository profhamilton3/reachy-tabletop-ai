# Bounded Claude Code Prompts

Run these prompts in order. Each prompt deliberately avoids asking the agent to implement the entire simulator at once.

## Prompt 0 — Establish a safe baseline

```text
Study CLAUDE.md, ROADMAP.md, the current Git status, Dockerfile, Compose file,
supervisord.conf, fake_reachy_server.py, joint_state_bridge.py, RViz config, and
existing notebooks. Do not modify user notebook changes. Implement only R12-000
through R12-003. Pin external inputs without opportunistically upgrading ROS or
rewriting the server. Add a CLI smoke test, local-only port mappings, an
environment manifest, and corrected architecture documentation. Run every test
available in this environment. Report exact commands, results, and anything that
requires a real Docker/Apple-Silicon host.
```

## Prompt 1 — Create the backend seam

```text
Implement R12-100 through R12-103. Refactor current state progression into a
SimulatorBackend/KinematicBackend with one authoritative fixed-step loop. gRPC
streaming methods must only read snapshots; subscriber count must not alter
motion. Preserve all existing Reachy v1 behavior. Add tests that compare final
joint state with zero, one, and multiple state subscribers. Replace the JSON file
bridge with direct or atomic coherent snapshots, choosing the smallest safe
change. Do not add MuJoCo yet.
```

## Prompt 2 — Scene definition and RViz markers

```text
Implement R12-200 and R12-202 using scenes/scene.schema.json and the example
scene. Add safe YAML loading, JSON Schema validation, unique-ID checks, asset-root
containment, quaternion validation, and useful errors. Publish primitives and
meshes as MarkerArray with correct world-frame transforms. Add tests for valid
scene, duplicate IDs, malformed pose, missing asset, and path traversal. Update
the RViz config only enough to show the markers. Do not add camera rendering or
physics in this change.
```

## Prompt 3 — Reachy v1 camera vertical slice

```text
Implement R12-300 and R12-301. Read the installed reachy_sdk_api camera proto and
the current reachy-sdk camera client before coding. Register CameraService in the
custom server. Generate deterministic left/right test images, encode them to
bytes that the unmodified Reachy v1 SDK decodes, and use bounded latest-frame
buffers. Implement stable simulated zoom/focus/autofocus semantics. Add contract
tests for GetImage, StreamImage, cancellation, slow consumers, and both SDK
last_frame properties. Do not add MuJoCo yet.
```

## Prompt 4 — ROS and browser camera presentation

```text
Implement R12-302 and R12-303. Publish the same coherent left/right fixture frames
as ROS images plus CameraInfo using the Reachy optical frame IDs. Add RViz image
or camera displays. Add a read-only localhost browser page with side-by-side
feeds and backend/scene/FPS/frame-age status. Keep queues bounded and rendering
out of request threads. Demonstrate that SDK, ROS, and browser outputs share the
same internal frame sequence. Do not add WebRTC unless measurements prove MJPEG
insufficient.
```

## Prompt 5 — Native MuJoCo protocol skeleton

```text
Implement R12-400 and the Docker-side connection skeleton only. Use
contracts/remote_protocol.md. Create a pinned native arm64 macOS Python package
with handshake, capabilities, heartbeat, reset, pause, step, command, state, and
fixture-frame messages. Implement deadlines, message-size limits, reconnect, and
explicit degraded state. Do not convert the full robot model in this PR. Add
protocol and disconnect/reconnect tests with a fake native server.
```

## Prompt 6 — Reachy model and static stereo scene

```text
Implement R12-401, R12-402, and the static portion of R12-404. Pin the Reachy
model source. Produce an explicit joint/camera mapping manifest and asset
provenance file. Compile the shared tabletop scene into MJCF. Render coherent
left/right frames from the documented optical frames. Add axis/calibration-board
tests and visual artifacts. Do not guess physical camera intrinsics; label
synthetic defaults clearly.
```

## Prompt 7 — End-to-end MuJoCo backend

```text
Implement R12-403 and complete R12-404. Connect SDK commands through the Docker
core to the native MuJoCo server and route state plus frames back to gRPC, ROS,
RViz, and browser clients. Preserve kinematic fallback. Demonstrate head motion
changes both camera views correctly and existing arm motion tests still pass.
Measure and report physics rate, render FPS, frame age, and command-to-frame
latency on the environment actually available; do not invent target-Mac results.
```

## Prompt 8 — Physics interaction

```text
Implement R12-500 through R12-503 one sub-issue at a time. First audit collisions
and inertials, then actuator/compliance semantics, then gripper/contact behavior,
then dynamic object state. Create a deterministic cube grasp scenario. Never tune
physics solely by visual appearance; record parameters and acceptance metrics.
```

## Prompt 9 — Fidelity and research bundle

```text
Implement R12-600 through R12-604. Keep synthetic calibration defaults separate
from measured calibration. Add optional depth/segmentation and independently
configurable sensor effects. Record commands, seeds, scene/model hashes, state,
and frame metadata for replay. Produce a benchmark harness and machine-readable
run manifest. Keep the original Reachy v1 RGB camera API unchanged.
```

## Prompt 10 — Gazebo decision only

```text
Perform R12-700 as a design spike. Do not start a full Gazebo port. Compare a
specific modern ROS/Gazebo pairing against the stable backend/scene/camera
contracts and against measured MuJoCo gaps. Build only a head-and-camera proof if
the environment supports it. Return a go/no-go decision, migration inventory,
and evidence. Treat old reachy_gazebo as a reference, not a dependency to revive
without justification.
```
