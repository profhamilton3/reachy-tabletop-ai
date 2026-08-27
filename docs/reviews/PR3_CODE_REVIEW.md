# PR #3 Code Review — Interactive Control-Panel Scene

**Repository:** `profhamilton3/reachy-1-2-sim`  
**Pull request:** [#3 — Interactive control-panel scene + two-arm practice](https://github.com/profhamilton3/reachy-1-2-sim/pull/3)  
**Reviewed head:** [`e13e36138083f14574214818cda14126683b02a9`](https://github.com/profhamilton3/reachy-1-2-sim/commit/e13e36138083f14574214818cda14126683b02a9)  
**Review disposition:** **REQUEST CHANGES — do not merge as currently written**

## Executive assessment

The pull request is directionally strong. It introduces a renderer-independent interactive-scene model, native MuJoCo control mechanics, a bilateral arm abstraction, scene selection, a practical control-panel scene, and useful tests. It also documents its current live reliability honestly.

Two deterministic correctness defects make the current head unsafe to merge:

1. `MujocoRemoteBackend` does not initialize its joint-index maps or `_thread` in `__init__`; those statements were accidentally indented into `request_reset()`. A normal SDK command can fail before the first reset, and a reset can erase the live thread handle.
2. The scene schema and control-panel YAML use cylinder `length`, while the compiler reads `height`. All six cylindrical buttons therefore compile with the default 0.100 m height instead of the declared 0.028 m height. The new unit fixtures use the invalid `height` field and bypass schema validation, masking the defect.

After those blockers, the most important design issue is physical truthfulness. The console body, ledge, and slanted face are all configured as non-colliding, while the demo does not pass the scene into `CartesianPlanner` and its `reach()` routine calls IK directly. The robot can therefore pass through the console structure; hand-authored route geometry, not physics or full-link collision checking, is currently providing most of the apparent avoidance. This must be corrected before adaptive path search, because an optimizer will exploit non-colliding geometry.

## Review method and limits

I reviewed the PR conversation, all changed paths exposed by GitHub, the PR-head source, and the relevant main-branch contracts. I performed local Python syntax compilation on the available changed Python modules and ran two focused probes:

- a freshly constructed `MujocoRemoteBackend` lacks `_uid_to_idx`, and `_build_command()` raises `AttributeError`;
- compiling `scenes/control_panel.yaml` emits `btn_red__g` with MuJoCo cylinder half-length `0.050000`, proving a full 0.100 m compiled length rather than the declared 0.028 m.

I did **not** independently execute the full 345-test suite or the native Apple-Silicon MuJoCo/Docker path in this environment. The PR author reports 345 offline tests passing, but GitHub currently exposes no completed CI run for the PR.

---

# What is good in the PR

## Architecture and scope

- The base robot model becomes furniture-independent, which is the correct long-term separation between robot and scene assets.
- `articulation` and `interactive` are useful renderer-independent scene concepts rather than control-panel-specific hard coding.
- `InteractiveController` keeps native physics behavior on the native MuJoCo side.
- State messages now include interactive state, giving the Docker compatibility layer a way to observe controls.
- The left/right arm parameterization removes avoidable duplication.
- A single scene selector is intended to drive native physics and RViz.
- The PR explicitly states that Gazebo is skipped and calls out live actuation reliability rather than claiming deterministic success.

## Test additions

The new tests provide useful component coverage for:

- button toggle/debounce;
- bistable switch behavior;
- scene control discovery and preferred-arm queries;
- left-arm pose/orientation mirroring;
- named articulated joints/geometries;
- interactive specification extraction.

These tests should be retained, corrected where noted below, and supplemented with contract and end-to-end tests.

---

# Blocking findings

## B1 — `MujocoRemoteBackend` initialization is inside `request_reset()`

**Severity:** Blocker  
**Files:** [`mujoco_remote_backend.py:91-126`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/mujoco_remote_backend.py#L91-L126), [`mujoco_remote_backend.py:331-365`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/mujoco_remote_backend.py#L331-L365)

### Problem

`__init__()` ends after `_reconnect_count`. The following initialization is inside `request_reset()`:

- `_uid_to_idx`;
- `_idx_to_uid`;
- `_thread`.

`_build_command()` accesses `_uid_to_idx`, so the first SDK command received before a reset raises `AttributeError`. In the normal startup path, `start()` sets `_thread`, but a later `request_reset()` overwrites it with `None`; `stop()` then cannot join the live thread.

### Why tests missed it

[`tests/unit/test_mujoco_remote_backend.py`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/tests/unit/test_mujoco_remote_backend.py) tests snapshot conversion and bridge delegation, but it never:

- calls `_build_command()` on a freshly constructed backend;
- starts, resets, and stops the backend;
- verifies that reset preserves the thread handle;
- exercises a real command-send cycle.

### Required fix

Move joint maps and `_thread` initialization into `__init__()`. Keep `request_reset()` limited to reset request state.

### Required regression tests

```text
test_constructor_initializes_joint_maps
test_build_command_works_before_first_reset
test_request_reset_does_not_replace_thread_handle
test_start_reset_stop_joins_thread
test_sdk_command_before_reset_reaches_remote_command_builder
```

---

## B2 — Cylinder `length`/`height` mismatch compiles wrong button geometry

**Severity:** Blocker  
**Files:** [`scenes/scene.schema.json:175-250`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/scenes/scene.schema.json#L175-L250), [`native_mujoco/scene_compiler.py:210-219`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/native_mujoco/scene_compiler.py#L210-L219), [`scenes/control_panel.yaml:107-140`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/scenes/control_panel.yaml#L107-L140), [`tests/unit/test_interactive.py:24-31`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/tests/unit/test_interactive.py#L24-L31), [`tests/unit/test_scene_compiler_articulation.py:17-27`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/tests/unit/test_scene_compiler_articulation.py#L17-L27)

### Problem

The schema defines `length` for cylinders and capsules. The control-panel YAML correctly uses `length: 0.028`. The compiler reads `geo.get("height")`, falls back to `0.1`, and emits MuJoCo size `radius 0.050000`, meaning a full 0.100 m cylinder length.

The test fixtures use `height`, which is rejected by the schema because `additionalProperties` is false, but the tests invoke `compile_scene()` directly without schema validation. They therefore test a document that is invalid under the public contract and hide the production defect.

### Required fix

- Read `length`, not `height`, in the compiler.
- Validate test scene documents with the same loader/schema used in production before compiling.
- Assert the resulting `MjModel.geom_size` for the actual `control_panel.yaml` button.
- Either compile schema-supported `capsule` and `plane` primitives or reject them consistently until implemented.

### Required regression tests

```text
test_control_panel_schema_then_compile
test_cylinder_length_maps_to_mujoco_half_length
test_height_is_rejected_by_schema
test_all_schema_geometry_kinds_have_compiler_contract
```

---

# High-priority findings

## H1 — Console fixtures are non-colliding, and the demo bypasses scene collision planning

**Severity:** High  
**Files:** [`scenes/control_panel.yaml:58-105`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/scenes/control_panel.yaml#L58-L105), [`scripts/demo_control_panel.py:109-140`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/scripts/demo_control_panel.py#L109-L140), [`scripts/demo_control_panel.py:229-238`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/scripts/demo_control_panel.py#L229-L238), [`src/reachy_ai/scene/awareness.py:331-382`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/src/reachy_ai/scene/awareness.py#L331-L382)

### Problem

`console_base`, `console_ledge`, and `console_slant` have `collision: false`. Their comments say the arm avoids them through scene-aware planning. However:

- both planners are constructed without `scene=scene`;
- `reach()` calls `planner.solve()` directly;
- it does not call `plan_segment()` or a collision-checking method;
- the existing scene checker evaluates a gripper point against simple bounds, not all arm links.

The route is therefore collision avoidance by hand-authored waypoints against ghost geometry. It does not establish that the arm, elbow, forearm, wrist, and gripper avoid the console physically.

### Required fix

- Give fixture geometry a collision channel that collides with robot collision geoms but does not self-collide with its mounted controls.
- Keep control-to-fixture filtering separate from robot-to-fixture filtering.
- Treat MuJoCo contacts for all robot links as the authoritative safety result.
- Passing the scene into the Cartesian planner is useful as a fast precheck, but point-only collision awareness must not be the final safety gate.

### Required test

A headless MuJoCo integration test must fail a trajectory that passes a forearm through the slanted panel even when the gripper target itself is clear.

---

## H2 — Reset is asynchronous, but the client does not wait for `reset_ack`

**Severity:** High  
**Files:** [`mujoco_remote_backend.py:228-259`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/mujoco_remote_backend.py#L228-L259), [`native_mujoco/protocol.py:241-249`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/native_mujoco/protocol.py#L241-L249), [`native_mujoco/server.py:404-413`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/native_mujoco/server.py#L404-L413), [`scripts/demo_control_panel.py:245-251`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/scripts/demo_control_panel.py#L245-L251)

### Problem

The server emits a structured `ResetAck`, but the remote client does not handle it. The demo writes a sentinel file, sleeps 1.5 seconds, and assumes reset completed. The send loop can transmit a reset and pending joint commands in the same cycle. Because `_last_target` is cleared before a post-reset snapshot is observed, the next command can also seed unspecified joints from a stale pre-reset snapshot.

### Required fix

- Generate a unique reset request ID.
- Add `RESETTING` state to the remote backend.
- Hold commands while resetting.
- Complete `request_reset()` only after the matching `reset_ack` and the first state whose `sim_step`/`scene_revision` correspond to the reset.
- Expose a bounded timeout and an explicit failure.

This is a mandatory prerequisite for deterministic Epic 8 episodes.

---

## H3 — File-based control/reset side channels are stale-prone and suppress all failures

**Severity:** High  
**Files:** [`fake_reachy_server.py:101-139`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/fake_reachy_server.py#L101-L139), [`scripts/demo_control_panel.py:85-98`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/scripts/demo_control_panel.py#L85-L98)

### Problem

The interactive state file contains only control values. It has no:

- sequence;
- simulation step/time;
- scene revision;
- backend state;
- data age;
- reset generation.

Both writer and watcher catch `Exception` and silently `pass`. The demo also ignores reset-file write errors. A stale file can be mistaken for a fresh actuation result, and operational failures are invisible.

### Required fix

Preferred: add a typed research/control API over the existing transport with reset and interactive-state acknowledgements.

Minimum acceptable interim fix: use an atomic envelope containing `protocol_version`, `sequence`, `sim_step`, `sim_time_s`, `scene_revision`, `backend`, `generated_wall_time_ns`, and `controls`; log failures with rate limiting; require freshness and matching reset generation before accepting a state change.

---

## H4 — The demo catches every exception as “unreachable”

**Severity:** High  
**File:** [`scripts/demo_control_panel.py:121-129`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/scripts/demo_control_panel.py#L121-L129)

`except (UnreachableError, Exception)` is equivalent to `except Exception`. It hides programming errors, transport failures, malformed state, and the constructor defect described in B1, then misreports them as an unreachable target.

**Required fix:** catch only expected IK exceptions. Log contextual information and re-raise unexpected exceptions. A trial infrastructure must distinguish `UNREACHABLE`, `BACKEND_FAILED`, `INVALID`, and `ABORTED`.

---

## H5 — Physics-only behavior warns on the wrong backend but continues

**Severity:** High  
**Files:** [`scripts/demo_control_panel.py:216-226`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/scripts/demo_control_panel.py#L216-L226), [`docker-compose.yml:19-31`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/docker-compose.yml#L19-L31)

The control-panel demo merely warns when the backend is not `mujoco-remote`. Compose enables fixture fallback. This can make a research run look successful even though interactive physics is absent.

**Required fix:** physics evaluation and all Epic 8 search modes must fail closed unless the active backend is verified as native MuJoCo, `READY`, and bound to the expected model and scene identity. Fixture fallback may remain for ordinary visualization, but it must be disabled and impossible to mistake for physics during experiments.

---

## H6 — Scene schema does not enforce relationships among interactivity and articulation

**Severity:** High  
**Files:** [`scenes/scene.schema.json:331-455`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/scenes/scene.schema.json#L331-L455), [`native_mujoco/interactive.py:61-91`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/native_mujoco/interactive.py#L61-L91)

An object may define `interactive` without `articulation`. The compiler creates an interactive spec anyway; the controller silently skips it when the joint cannot be found. The schema also does not enforce:

- button → slide joint;
- switch/lever → hinge joint;
- non-zero axis;
- ordered range;
- thresholds inside the range;
- appropriate `off_threshold`/`bistable` combinations.

**Required fix:** add conditional schema constraints and compiler-level defensive validation. Missing named joints/geometries should be startup errors for declared interactive controls, not silent omissions.

---

## H7 — Runtime `scene_load` returns success without loading a scene

**Severity:** High  
**File:** [`native_mujoco/server.py:566-578`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/native_mujoco/server.py#L566-L578)

The handler hashes the supplied document, assigns a new scene revision, and returns `accepted=True`, but it never recompiles or swaps the MuJoCo model. This creates false provenance: clients can believe a new scene is active when the old physical world remains loaded.

**Required fix:** until atomic scene replacement is implemented, reject runtime `scene_load` with a clear `restart_required` error. Startup scene loading is sufficient for the current MVP.

---

## H8 — Startup scene coherence and process identity are weaker than documented

**Severity:** High  
**File:** [`scripts/start_sim.sh:31-69`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/scripts/start_sim.sh#L31-L69)

### Problems

- Absolute external scene paths are accepted for native MuJoCo, but the container is given `/opt/scenes/<basename>` without staging that external file into the mounted/container scene directory.
- The script kills by listening port with `xargs -r`, which is not portable to the default BSD userland on macOS.
- It uses `kill -9` rather than a tracked PID and graceful shutdown.
- Readiness means only “some process is listening on 8765”; it does not verify protocol, model hash, scene hash, or the newly launched PID.
- The native WebSocket binds `0.0.0.0` and has no authentication.

### Required fix

- Restrict scene selection to repository scenes or copy/stage an external scene into the container-visible mount.
- Track the native server PID and terminate gracefully, escalating only when necessary.
- Perform a hello/health check that validates protocol version, model identity, scene SHA/revision, and capabilities.
- Bind to the narrowest interface compatible with Docker Desktop, or add an explicit session token and host firewall instructions.

---

## H9 — Recording metadata does not identify the model and scene actually used

**Severity:** High for research/Epic 8  
**Files:** [`native_mujoco/server.py:326-346`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/native_mujoco/server.py#L326-L346), [`native_mujoco/recorder.py`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/native_mujoco/recorder.py)

The recorder manifest writes the default model path and `scene_path: None` even when a different model or scene was loaded. It does not capture content hashes, Git commit, compiler version, physics profile, protocol version, or MuJoCo version.

**Required fix:** record immutable identity for the actual compiled world. Epic 8 must never reuse an experience across incompatible model/scene/physics identities.

---

## H10 — Camera frame metadata is discarded at the Docker compatibility boundary

**Severity:** High for synchronized vision and Epic 8  
**Files:** [`native_mujoco/protocol.py:205-226`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/native_mujoco/protocol.py#L205-L226), [`mujoco_remote_backend.py:317-330`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/mujoco_remote_backend.py#L317-L330), [`fake_reachy_server.py:141-180`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/fake_reachy_server.py#L141-L180)

The native frame has sequence, simulation step/time, scene revision, dimensions, and render latency. The remote backend writes only JPEG bytes. The camera adapter then invents a new sequence on every file read, stamps the current wall clock, and reports dimensions as zero.

**Required fix:** carry the original frame metadata through one bounded camera-source abstraction. Stereo frames and state used by an episode must share authoritative simulation-step identity.

---

## H11 — `gate_check()` does not deny any motion path

**Severity:** High before generated trajectories can ever target hardware  
**File:** [`src/reachy_ai/motion/safety.py:20-33`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/src/reachy_ai/motion/safety.py#L20-L33)

Every branch returns `True`. Because `REACHY_SIM_BACKEND` defaults to `kinematic`, the function can also classify an unspecified environment as simulation. This is not a real hardware safety gate.

**Required fix:** Epic 8 search must be simulator-only. Physical execution must be a separate, denied-by-default command requiring an explicit hardware target, motion-enable flag, operator approval, and verified endpoint identity.

---

# Medium-priority findings

## M1 — Mesh compilation shadows the requested geom name

**File:** [`native_mujoco/scene_compiler.py:215-227`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/native_mujoco/scene_compiler.py#L215-L227)

The `name` argument intended for the geom is overwritten with the mesh asset name. An articulated/interactive mesh can therefore receive the wrong geom name, preventing `InteractiveController` from finding and recoloring it.

**Fix:** use a separate `mesh_asset_name` local and add an articulated-mesh test.

## M2 — Legal object IDs can collide after `_safe_name()` normalization

**Files:** [`scenes/scene.schema.json:426-438`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/scenes/scene.schema.json#L426-L438), [`native_mujoco/scene_compiler.py:302-306`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/native_mujoco/scene_compiler.py#L302-L306)

IDs such as `a.b` and `a_b` are both legal but normalize to the same MuJoCo prefix. Validate compiled-name uniqueness or use a reversible/hashed encoding.

## M3 — WebSocket handler can leave sibling tasks alive

**File:** [`native_mujoco/server.py:491-565`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/native_mujoco/server.py#L491-L565)

`asyncio.gather()` waits for all three loops. If receive exits cleanly on a requested disconnect or timeout, send/heartbeat loops can continue indefinitely. Use `asyncio.wait(..., return_when=FIRST_COMPLETED)`, cancel siblings, await cancellation, and clear connection state in `finally`.

## M4 — `qfrc_applied` is overwritten rather than composed

**File:** [`native_mujoco/interactive.py:143-153`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/native_mujoco/interactive.py#L143-L153)

The bistable controller assigns to `qfrc_applied[dof]`. This can erase another subsystem’s external force contribution. Clear a dedicated controller-owned force buffer each step or add the contribution after defining ownership.

## M5 — Contact-source filtering is too broad and assumes exact bit equality

**File:** [`native_mujoco/interactive.py:123-142`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/native_mujoco/interactive.py#L123-L142)

Any robot collision geom can toggle a contact-sourced button, not just gripper/finger geoms. `contype == 2` also assumes a single exact bit rather than testing a bitmask. Restrict accepted source geoms and use bitwise checks.

## M6 — Planner side values are not validated

**File:** [`src/reachy_ai/motion/kinematics.py:54-59`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/src/reachy_ai/motion/kinematics.py#L54-L59)

Any value other than exact `left` silently selects the right arm. Reject invalid side values at construction.

## M7 — Scene collision awareness ignores object orientation

**File:** [`src/reachy_ai/scene/awareness.py:331-382`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/src/reachy_ai/scene/awareness.py#L331-L382)

Static obstacle checks use axis-aligned extents even for the slanted console face. This is acceptable only as a conservative precheck if authoritative full-link contacts are evaluated in MuJoCo. Otherwise use oriented bounds or signed-distance queries.

## M8 — Scene data is parsed into two divergent representations

**Files:** [`native_mujoco/server.py:253-271`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/native_mujoco/server.py#L253-L271), [`src/reachy_ai/scene/awareness.py:176-203`](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/src/reachy_ai/scene/awareness.py#L176-L203)

The server validates through `scene_loader`, then reopens raw YAML because the typed loader does not preserve all physics/interactive fields. Prefer one validated domain model compiled into all consumers, or explicitly document the loader as validation-only and test parity.

---

# Test and CI assessment

The reported 345 passing tests are encouraging but insufficient for the changed risk surface. The new tests are primarily component tests, and the interactive tests deliberately bypass the schema. There is no checked-in evidence in PR #3 of a test covering this complete path:

```text
Reachy v1 SDK
  → Docker gRPC server
  → KinematicBridge/MujocoRemoteBackend
  → reset acknowledgement
  → native MuJoCo scene
  → robot contact with one control
  → interactive state returned with matching sim_step/scene_revision
```

## Minimum additional test matrix before merge

### Pure unit/contract

- constructor and lifecycle regression for B1;
- schema→compiler→MjModel control-panel contract for B2;
- invalid interactive/articulation combinations rejected;
- compiled-name collision rejected;
- mesh interactive geom naming;
- reset request/ack state machine;
- unexpected demo exceptions propagate;
- invalid arm side rejected.

### Native MuJoCo integration

- actual `control_panel.yaml` loads;
- button dimensions match YAML;
- all 10 declared controls bind to a joint and geom;
- reset seats every control OFF and returns a matching acknowledgement;
- forbidden robot-console contacts are detected;
- intended gripper-control contact can actuate the intended control;
- no neighboring control changes;
- head and arm commands remain stable through repeated batched commands.

### Hybrid Docker/native acceptance

- clean startup selects the same scene identity in native MuJoCo and RViz;
- one SDK command works before any reset;
- reset followed immediately by commands does not use stale targets;
- native disconnect produces bounded degraded behavior without fixture masquerade;
- original camera metadata survives to research consumers;
- repeated start/stop does not retain an old server or leak connection tasks.

### CI

Add at least one GitHub Actions workflow for syntax, unit, schema/compiler contract, and headless MuJoCo tests. Keep Apple-Silicon/Docker integration as a documented self-hosted or release-gate test if hosted CI cannot provide it.

---

# Merge gate

PR #3 should be eligible to merge only when all items below are true:

- [ ] B1 constructor/lifecycle defect fixed and covered.
- [ ] B2 cylinder contract fixed; tests validate production YAML before compile.
- [ ] Console fixture collision behavior is physically truthful and tested.
- [ ] Reset is acknowledged; commands cannot race reset or seed from stale state.
- [ ] Control state includes freshness/scene identity; silent exception swallowing removed.
- [ ] Demo fails closed on the wrong backend and propagates unexpected exceptions.
- [ ] Runtime `scene_load` is implemented atomically or truthfully rejected.
- [ ] Recorder captures actual model/scene identity.
- [ ] At least one SDK→Docker→native MuJoCo interactive-control acceptance test passes.
- [ ] GitHub CI runs the offline suite and schema/compiler contract.
- [ ] The known 4–7/10 motion reliability is labeled experimental rather than used as a deterministic acceptance claim.

The reliability limitation itself does not have to block merge after correctness is restored, provided the control-panel demo remains clearly experimental and the simulator reports failures honestly. Epic 8 can then optimize the path parameters against a physically truthful, deterministic episode interface.

---

# Recommended disposition

**Request changes.** Preserve the feature work, fix the two deterministic blockers immediately, then close the reset, collision-truthfulness, provenance, and fail-closed gaps before beginning adaptive path optimization.
