# Acceptance Test Plan

## A. Baseline compatibility

### A1 — SDK connection

- Start the Docker core in `kinematic` mode.
- Connect with the unmodified Reachy v1 SDK.
- Assert both arms, head, grippers, fans, and sensors expected by the current demo are discoverable.
- Assert no Reachy 2 package is required.

### A2 — Existing motion

- Run the CLI equivalent of the existing motion notebook.
- Assert commands converge within declared tolerance.
- Assert RViz reflects the same joint snapshot.

### A3 — Subscriber invariance

Repeat the same command trajectory with zero, one, and three joint-state stream subscribers. Final joint sequences and timings must remain within the declared deterministic tolerance. This catches the current subscriber-dependent integration defect.

## B. Scene contract

### B1 — valid scene

Load `scenes/tabletop.example.yaml`. Assert schema version, world frame, object count, object IDs, and content hash.

### B2 — invalid inputs

Each must fail before backend load with a useful field/path error:

- duplicate object ID;
- NaN/Infinity;
- invalid quaternion norm;
- unsupported primitive;
- negative mass;
- missing mesh;
- `../` path traversal;
- remote mesh URL;
- unknown schema version.

### B3 — RViz parity

Assert every visible scene object has a corresponding marker with expected frame, pose, scale, and color. For dynamic objects, assert marker sequence follows backend object sequence.

## C. Reachy v1 camera API

### C1 — fixture frames

- Connect with unmodified Reachy v1 SDK.
- Access `left_camera.last_frame` and `right_camera.last_frame`.
- Assert each is a nonempty 3-channel array.
- Assert left/right identifiers embedded in fixture pixels differ as expected.

### C2 — stream lifecycle

- Start and cancel streams repeatedly.
- Disconnect clients mid-stream.
- Run a deliberately slow client.
- Assert no unbounded memory growth, orphan thread, or server-wide blocking.

### C3 — zoom/focus semantics

- Exercise every proto RPC.
- Assert stable values and documented acknowledgements.
- Assert unsupported optical effects are reported/documented, not silently misrepresented.

## D. ROS and browser presentation

### D1 — ROS image products

For both cameras assert:

- image topic exists;
- camera-info topic exists;
- optical frame ID is correct;
- width/height/encoding are consistent;
- timestamps and sequence correspond to backend frame metadata.

### D2 — coherent fan-out

For one captured frame sequence, verify SDK, ROS, and browser outputs derive from the same backend camera frame, allowing for encoding differences.

### D3 — stale/degraded state

Stop the native server. The web UI and logs must report unavailable/degraded state, SDK calls must fail or return policy-defined fixture/stale behavior within a bounded deadline, and nothing may hang indefinitely.

## E. Geometry and stereo

### E1 — optical-axis board

Place a board with labeled +X/+Y axes and colored quadrants in front of each camera. Assert images are neither mirrored nor upside down and obey the documented ROS/OpenCV conversion.

### E2 — head movement

At a fixed scene:

- command positive/negative yaw and pitch;
- assert projected target motion has the expected direction;
- assert both eyes use the same head state step.

### E3 — projection accuracy

Use known 3D calibration points and camera intrinsics. Compute expected pixels independently and compare rendered pixels within a declared tolerance.

### E4 — stereo baseline

Place objects at several depths. Assert disparity sign is correct and magnitude changes monotonically with inverse depth for the configured baseline/convergence.

### E5 — occlusion

Place one object behind another. Assert the rendered image and depth buffer show the correct visible surface.

## F. Physics

### F1 — deterministic reset

With fixed model, scene, version, and seed, reset repeatedly and compare initial state plus frame hashes under the renderer’s declared determinism policy.

### F2 — contact

Move a gripper or arm into a known object. Assert contact appears, object motion is physically plausible, and no numerical instability occurs.

### F3 — grasp fixture

Execute a deterministic cube grasp/lift/release scenario. Assert object height/contact/gripper state at named checkpoints.

### F4 — limits and compliance

Assert joint limits, compliant/stiff state, saturation, and reset behavior are visible and documented.

## G. Performance and reliability

Run on each declared target environment and save raw plus summarized output:

- kinematic Docker mode;
- fixture camera mode;
- native arm64 MuJoCo + Docker core;
- optional headless/container MuJoCo.

Measure:

- physics steps per second and real-time factor;
- RGB frames per second per camera;
- p50/p95/p99 render time;
- p50/p95/p99 frame age;
- command-to-state latency;
- command-to-frame latency;
- dropped/coalesced frames;
- CPU and memory;
- reconnect recovery;
- long-run thread/file-descriptor/memory stability.

Do not set arbitrary pass numbers before the first target-Mac baseline. Record initial measurements, define research-appropriate thresholds, then version those thresholds.

## H. Release evidence

Every release candidate must include:

- Git commit and dirty-state report;
- model, scene, and dependency hashes;
- target machine/OS/architecture;
- Docker and MuJoCo versions;
- test command log;
- machine-readable results;
- representative left/right frames;
- RViz screenshot showing robot, scene, TF, and camera products;
- known deviations from physical Reachy 1.2.
