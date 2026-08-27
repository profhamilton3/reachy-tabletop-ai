# Epic 8 — Entry Gate Fix Plan (PR 8.0 / R12-800)

> **For the next Claude Code session.** This is the actionable close-out plan for the
> Epic 8 entry gate. It pairs with the assessment in
> `docs/reviews/EPIC8_ENTRY_GATE_REVIEW.md`. Read both before starting.

## Where the code lives

- **Roadmap doc (this repo):** `IITG-Reachy-Project/docs/roadmap/EPIC-8.md`
- **Code to change (separate repo):** `reachy-1-2-sim` — local path `/Users/terrancehamilton/reachy-1-2-sim`, remote `profhamilton3/reachy-1-2-sim`, branch `feat/control-panel-scene`.
- **Review baseline commit:** `9c6bca6` (confirmed the latest commit on all refs as of 2026-08-19).

## Ground rules (from EPIC-8.md §9, §12)

- **Work on PR 8.0 / R12-800 only.** Do NOT build the optimizer, experience store, Gymnasium env, RL, or browser UI. Those are PR 8.1+ and each PR must stop and report.
- **Reachy 1.2 SDK only** — `reachy_sdk`, never `reachy2_sdk`.
- No LLM-emitted raw joint commands anywhere in the loop.
- Prefer the collision *channel* fix over disabling collision. Do not alter robot geometry/mass/friction/gains/collision masks as a shortcut to make a task pass (§5.2).
- When done: report changed files, exact commands run, pass/fail evidence, what could only be run on Apple Silicon, and everything deferred to PR 8.1. Do not auto-start PR 8.1.

## Current state (what `9c6bca6` already did — do not redo)

- 8-A: `_uid_to_idx`/`_idx_to_uid` and `_thread` are initialized in `MujocoRemoteBackend.__init__` and `request_reset()` is inert to them. Unit-tested & green.
- 8-B: cylinder compiles `length` (not `height`); one unified compiler (`scene_compiler.compile_scene_body_fragment`).
- 8-D (server half): `ResetAck{request_id, sim_step, scene_revision}` type exists and the server emits it.
- 8-E (partial): WS `state`/`camera_frame` carry `seq/sim_step/sim_time_s/scene_revision`; runtime `scene_load` honestly returns `restart_required`.

Run this first to confirm the green baseline:
```bash
cd /Users/terrancehamilton/reachy-1-2-sim
python3 -m pytest tests/unit/test_mujoco_remote_backend.py \
  tests/unit/test_scene_compiler.py \
  tests/unit/test_scene_compiler_articulation.py -q
# expect: 60 passed
```

---

## Fixes, in recommended order

### FIX 1 — Gate 8-C: truthful console collision (primary blocker)

**Problem:** `scenes/control_panel.yaml` sets `collision: false` on `console_base` (`:72-74`),
`console_ledge` (`:88-92`), and `console_slant` (`:105-109`). Robot links can pass through the
console; the only avoidance is a non-authoritative point check
(`reachy_ai.scene.awareness.SceneModel`, used at `scripts/demo_control_panel.py:52`).

**Do:**
1. Make the three console fixtures collidable while their mounted controls (buttons/levers/switches)
   stay stable. Use **separate contact channels** (contype/conaffinity) so the console collides with
   robot links but NOT with its own child controls. Reference the existing object channel constants in
   `native_mujoco/scene_compiler.py:29-31` (`_OBJ_CONTYPE=4`, `_OBJ_CONAFFINITY=7`) and the robot's
   channels in `native_mujoco/model/reachy_1_2.xml` (search the R12-500 collision-model comment).
   The console should share the robot's collision channel but be masked off from the control geoms.
2. The scene compiler currently emits `contype="0" conaffinity="0"` when `collision:false`
   (`scene_compiler.py:260-261`) and the object channel otherwise (`:254-259`). You will likely need a
   scene-level way to assign a *fixture* channel distinct from the *control* channel — extend the schema
   (`scenes/scene.schema.json`) and compiler rather than hardcoding.
3. Add an integration test: drive the forearm/arm into the console and assert a forbidden full-link
   contact is present in `MjData.contact`, and separately assert that resetting/settling the scene does
   NOT spontaneously toggle or displace any mounted control (controls stay stable under their parent).

**New test:** `tests/integration/test_forbidden_console_contact.py` (or `native_mujoco/tests/...`
matching repo convention — there is currently no `tests/integration/` dir; create it).

**Acceptance:** console collides with robot links; controls remain stable; forbidden contact is
observable in raw MuJoCo contact data (not just the point check).

### FIX 2 — Gate 8-D: acknowledged reset on the client + demo

**Problem:** server emits `ResetAck`, but the client
(`mujoco_remote_backend.py`) sends a hardcoded `request_id="demo"` (`:260`), never reads `reset_ack`
(`_recv` at `:227-249` has no branch), has no `RESETTING` state, and does not hold commands during
reset. The actual demo resets via a sentinel file `/tmp/reachy_reset_request` +
`time.sleep(1.5)` (`scripts/demo_control_panel.py:100-106, 268-269`).

**Do (client — `mujoco_remote_backend.py`):**
1. Generate a **unique** `request_id` per reset (e.g. `uuid4().hex` or a monotonic counter), store it as
   `self._pending_reset_id`.
2. Add a `RESETTING` value to `ConnectionState` (`:56-61`) or a separate reset-state flag.
3. On `request_reset()`, enter `RESETTING` and **buffer/hold** any `submit_command`/`submit_commands`
   until the matching `reset_ack` arrives.
4. In `_recv`, handle `mtype == "reset_ack"`: match `request_id`, confirm post-reset state, then leave
   `RESETTING`, clear the hold, and re-seed `_last_target` from the acknowledged post-reset pose.
5. Add a bounded timeout → if no matching ack in time, go `ABORTED`/`DEGRADED` (fail closed, per §5.1),
   never silently proceed.

**Do (demo — `scripts/demo_control_panel.py`):**
6. Replace the sentinel-file `request_reset()` and the `time.sleep(1.5)` with the acknowledged WS reset;
   block on the ack instead of sleeping. Remove `_RESET_REQUEST` usage. (The file-watcher path lives in
   `fake_reachy_server.py` — decide whether to route it through the WS reset or retire it.)

**New tests:** extend `tests/unit/test_mujoco_remote_backend.py` with unique-id generation, command-hold
during `RESETTING`, ack-clears-hold, and ack-timeout → aborted. A start→reset→stop lifecycle test also
satisfies the remaining 8-A gap (see FIX 6).

**Acceptance:** reset uses a unique id, waits for the matching `reset_ack` + post-reset state, holds
commands until then, and no fixed sleep is a correctness dependency.

### FIX 3 — Gate 8-F: fail-closed research execution

**Problem:** `scripts/demo_control_panel.py:240-244` only warns when the backend is not
`mujoco-remote`, then continues.

**Do:**
1. In the control-panel/research entry path, **refuse** to run unless the backend is `mujoco-remote`
   and its `connection_state` is `READY` (see `MujocoRemoteBackend.connection_state`,
   `mujoco_remote_backend.py:160-163`). Raise/exit, do not warn-and-continue.
2. Add an explicit "search/research mode" flag (env or arg) that hard-disables fixture fallback
   regardless of `REACHY_SIM_ALLOW_FIXTURE_FALLBACK` (`:43`).
3. Classify errors: expected IK failures are handled; unexpected exceptions propagate and are tagged
   (map toward the `ABORTED` semantics in EPIC-8.md §5.1/§7.5). Narrow the broad
   `except Exception` at `mujoco_remote_backend.py:195-199` so genuine faults are not silently downgraded.
4. Physical hardware execution denied by default (this repo is sim-only; assert/guard it).

**Acceptance:** a non-`READY`/non-MuJoCo backend causes a hard refusal in research mode; fallback cannot
silently activate.

### FIX 4 — Gate 8-E: real provenance + typed control state

**Problem A (recorder):** `native_mujoco/server.py:355-372` hardcodes `model_path=_DEFAULT_MODEL`,
`scene_path=None`, and records no hashes/git sha/versions.

**Do:**
1. Pass the **actual** compiled model path + `scene_path` through to the recorder, and compute/record:
   `model_sha256`, `compiled_model_sha256`, `scene_sha256`, `scene_revision`, `scene_schema_version`,
   `scene_compiler_version`, `protocol_version`, `mujoco.__version__`, `python_version`, host os/arch,
   repo git sha + dirty flag. These map to `SimulatorIdentity` in EPIC-8.md §7.1 (do NOT build the full
   dataclass here — that is PR 8.1 — just stop recording *false* metadata).

**Problem B (control state):** demo reads a bare `/tmp/reachy_interactive_state.json`
(`demo_control_panel.py:61, 92-97`), written by `fake_reachy_server.py`.

**Do:**
2. Prefer the WS `state` message's `interactive` field (already ingested at
   `mujoco_remote_backend.py:319-327`) as the source of truth. If a file interim is unavoidable, wrap it
   in a versioned envelope with a freshness/`sim_step` check and reject stale reads.

**Acceptance:** recorder metadata reflects the real model/scene/versions; control state is read from a
typed/fresh source, not an unversioned `/tmp` blob.

### FIX 5 — Gate 8-B: schema-validated fixtures + relational enforcement

**Do:**
1. In `tests/unit/test_scene_compiler*.py`, validate fixtures through the production schema
   (`scenes/scene.schema.json` via `scene_loader.load_scene`) **before** compiling, instead of hand-built
   dicts.
2. Add assertions on **actual `MjModel` geom dimensions** for every control in `scenes/control_panel.yaml`
   (compile it, load the `MjModel`, check each control geom's size against the YAML). See
   `native_mujoco/objects.py:build_scene_model` for compiling to an `MjModel`.
3. Enforce interactive↔articulation relationships in the compiler/schema: an object with `interactive`
   must have a compatible `articulation` (button⇒slide, lever/switch⇒hinge) and consistent
   source/thresholds. Extend `scene_compiler._compile_articulation`
   (`native_mujoco/scene_compiler.py:175-200`) / `interactive_specs` (`:98-126`).

### FIX 6 — Gate 8-A: live lifecycle test

**Do:** add a test that actually `start()`s the backend thread against a stub/mock WS (or a real local
server), performs a reset round-trip, and `stop()`s cleanly, asserting the thread joins and no
maps/thread are clobbered. Extend `tests/unit/test_mujoco_remote_backend.py` or add
`tests/integration/test_backend_lifecycle.py`.

### FIX 7 — Hybrid acceptance test + evidence bundle (entry-gate acceptance, EPIC-8.md §6)

**Do:** implement the acceptance sequence end-to-end:
```
clean native-server start
 → verified model/scene handshake
 → acknowledged reset
 → Reachy SDK command
 → intended robot motion
 → one intended control state transition
 → no forbidden console contact
 → state with matching sim_step/scene_revision
 → clean stop and thread/task shutdown
```
Path: Reachy SDK → Docker gRPC → `MujocoRemoteBackend` → native MuJoCo. It need not be reliable across
*all* controls yet — it must be deterministic, physically truthful, observable, and honest about failure.
Produce an evidence bundle (logs + recorder artifacts + identity metadata) as the exit artifact.

---

## Required tests to have green before declaring PR 8.0 done (EPIC-8.md §PR 8.0)

```
unit:      backend constructor/build/reset/lifecycle          (FIX 2, FIX 6)
contract:  schema → compiler → MjModel                          (FIX 5)
native:    actual control_panel.yaml, all controls bind         (FIX 5)
native:    forbidden full-link console contact detected         (FIX 1)
hybrid:    SDK → Docker → native → control state                (FIX 7)
lifecycle: start/reset/disconnect/reconnect/stop                (FIX 2, FIX 6)
```

## Test/run commands

```bash
cd /Users/terrancehamilton/reachy-1-2-sim
python3 -m pytest tests/ -q                      # offline unit/contract
# native + hybrid tests need mjpython + Docker on the Apple-Silicon host:
#   native server:  cd native_mujoco && mjpython server.py --scene ../scenes/control_panel.yaml
#   physics stack:  scripts/start_sim.sh   (mujoco-remote backend)
```

## Stop condition

Close Gates 8-A→8-F, land the hybrid acceptance test + evidence bundle, update tests and the EPIC-8.md
status, then **stop**. Do not add experience storage, an optimizer, or UI search controls (that is PR 8.1+).
