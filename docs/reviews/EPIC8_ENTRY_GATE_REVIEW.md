# Epic 8 — Entry Gate Review

**Reviewer:** Claude Code (Opus 4.8)
**Date:** 2026-08-19
**Repository:** `profhamilton3/reachy-1-2-sim`
**Reviewed ref:** branch `feat/control-panel-scene` @ `9c6bca6` — *"fix: address PR3 code-review blockers and high-priority findings"* (2026-08-19 14:25:47 -0700)
**Baseline confirmed latest:** after `git fetch --all`, `9c6bca6` is the newest commit across all local and remote refs; working tree clean except unrelated `notebooks/` files.
**Scope:** Section 6 of `docs/roadmap/EPIC-8.md` — Gates 8-A → 8-F and the entry-gate acceptance sequence. No code was changed during this review.
**Evidence run:** `python3 -m pytest tests/unit/test_mujoco_remote_backend.py tests/unit/test_scene_compiler.py tests/unit/test_scene_compiler_articulation.py -q` → **60 passed in 0.14s**.

---

## Overall verdict: ENTRY GATE NOT PASSED

The hardest plumbing is already in place — remote-backend lifecycle, a single unified scene compiler, server-side `ResetAck`, and an honest `scene_load` rejection. But two hard blockers (8-C truthful collision, 8-D acknowledged reset on the client/demo side) plus a fail-closed gap (8-F) remain, and there is no hybrid acceptance test and no evidence bundle. PR 8.1 must not begin until these close.

| Gate | Status | One-line reason |
|---|---|---|
| 8-A Remote backend lifecycle | 🟢 Substantially met | Maps + `_thread` in `__init__`; `request_reset` inert to them; unit-tested. Missing a live start/stop test. |
| 8-B Schema/compiler unify | 🟡 Partial | Cylinder `length` fixed & single compiler path; fixtures not validated through production schema; weak interactive↔articulation enforcement. |
| 8-C Truthful collision | 🔴 Not met | Console `collision: false` by design; relies on non-authoritative point checks; no forbidden-contact test. |
| 8-D Acknowledged reset | 🔴 Not met | Server emits `ResetAck`, but client/demo never consume it; hardcoded `request_id="demo"`; demo resets via sentinel file + `sleep(1.5)`. |
| 8-E Truthful provenance | 🟡 Partial | WS state/frame identity good; `scene_load` honest. Recorder logs fake model/scene metadata; control state still a bare `/tmp` JSON file. |
| 8-F Fail-closed research | 🔴 Not met | Demo only *warns* on non-MuJoCo backend and continues; no enforced search mode; no hybrid acceptance test. |

---

## Gate-by-gate detail

### 8-A — Correct remote backend lifecycle 🟢 Substantially met
- `_uid_to_idx` / `_idx_to_uid` built in `__init__` — `mujoco_remote_backend.py:112-118`.
- `_thread` initialized to `None` in `__init__` — `mujoco_remote_backend.py:120`.
- `request_reset()` sets `_pending_reset` and clears `_last_target` only; does not touch maps or thread — `mujoco_remote_backend.py:122-127`.
- `_build_command()` works before any reset; covered by `TestConstructorLifecycle` — `tests/unit/test_mujoco_remote_backend.py:269-311` (green).

**Gap:** No test actually starts the background thread, stops it, or exercises a reset while running. Add one start→reset→stop lifecycle test to fully close 8-A.

### 8-B — Unify schema and compiler contracts 🟡 Partial
- ✅ Cylinder compiles `length`, not `height` — `native_mujoco/scene_compiler.py:231-234`; `scenes/control_panel.yaml` uses `length` throughout; regression test at `tests/unit/test_scene_compiler_articulation.py:114`.
- ✅ Single compiler path: `server.py` → `objects.build_scene_model_xml` (`native_mujoco/objects.py:135-158`) → `scene_compiler.compile_scene_body_fragment`. No competing compiler.

**Gaps:**
- Fixtures are hand-built dicts; tests do **not** validate them through `scenes/scene.schema.json` / `scene_loader.load_scene` before compiling. The gate requires "validate test fixtures through the production schema before compiling."
- No test asserts **actual `MjModel` geom dimensions** for every control in `control_panel.yaml`.
- Relational enforcement is thin: `_compile_articulation` validates joint type/axis (`scene_compiler.py:178-184`) but nothing enforces "interactive control ⇒ articulation present, with consistent source/thresholds."

### 8-C — Make collision physics truthful 🔴 Not met (primary blocker)
- All three console fixtures — `console_base`, `console_ledge`, `console_slant` — are `collision: false`: `scenes/control_panel.yaml:72-74, 88-92, 105-109`, with comments stating the arm avoids the console "via scene-awareness path planning."
- That path planning is a **point check** (`reachy_ai.scene.awareness.SceneModel`, used at `scripts/demo_control_panel.py:52`) — exactly the mechanism Section 6 calls out as "may remain a precheck but are **not** authoritative."
- No `MjModel`-level contact between robot links and the console exists, so the evaluator can never observe a forbidden full-link contact. No forbidden-contact test exists.
- Root tension: the fixtures were made non-colliding because "the console must not shove its own mounted controls around" — which is gate item 2 ("mounted controls must not be destabilized by their parent fixture"). Correct fix is **separate collision channels** (contype/conaffinity), not disabling collision.

### 8-D — Deterministic acknowledged reset 🔴 Not met (blocker)
- ✅ Server side ready: `ResetAck{request_id, sim_step, scene_revision}` (`native_mujoco/protocol.py:265-267`), emitted after the reset is applied (`native_mujoco/server.py:436-445`), request_id echoed from client (`server.py:175`).
- ❌ Client never uses it: `mujoco_remote_backend.py:260` sends `{"type":"reset","request_id":"demo"}` — hardcoded, not unique; `_recv` has no `reset_ack` branch (`:227-249`). No wait for ack, no `RESETTING` state, commands not held during reset — reset and commands drained in the same `_send` cycle (`:256-273`).
- ❌ Actual control-panel path doesn't use the WS reset at all: `scripts/demo_control_panel.py:100-106` drops a sentinel file `/tmp/reachy_reset_request` (watched by `fake_reachy_server.py`) then `time.sleep(1.5)` (`:268-269`) — a fixed sleep used as a correctness dependency, the exact anti-pattern this gate removes.

### 8-E — Truthful state, camera, and provenance 🟡 Partial
- ✅ State & `CameraFrame` preserve `seq`, `sim_step`, `sim_time_s`, `scene_revision` across the process boundary — `server.py:455-469, 419-431`; ingested at `mujoco_remote_backend.py:328-340`.
- ✅ Runtime `scene_load` is honest: returns `restart_required` rather than false success — `server.py:612-622`.
- ❌ Recorder provenance is false: metadata hardcodes `model_path = _DEFAULT_MODEL` and `scene_path = None` even when a scene is compiled, with no model/scene sha256, git sha, compiler/protocol/mujoco versions, or host arch — `server.py:355-372`. Gate requires actual paths/hashes/revision.
- ❌ Control state is still a "stale-prone bare control-state file": demo reads `/tmp/reachy_interactive_state.json` (`demo_control_panel.py:61, 92-97`), written by `fake_reachy_server.py`. Not the required typed/versioned + freshness-checked API/envelope.

### 8-F — Fail-closed research execution 🔴 Not met
- ❌ Control-panel runner does not refuse a non-MuJoCo backend — it logs a warning and proceeds — `scripts/demo_control_panel.py:240-244`.
- ⚠️ Fixture fallback is env-gated (`_ALLOW_FALLBACK`, default false — `mujoco_remote_backend.py:43`) but there is no enforced "search mode" that hard-disables it and requires a verified `READY` mujoco-remote backend.
- ⚠️ Error handling is broad-catch → `DEGRADED` (`mujoco_remote_backend.py:195-199`); unexpected exceptions are not classified into `ABORTED` vs ordinary task failure.

### Entry-gate acceptance sequence / evidence bundle 🔴 Not met
No hybrid **Reachy SDK → Docker gRPC → `MujocoRemoteBackend` → native MuJoCo → intended control transition** acceptance test exists, and no evidence bundle is produced. Both are required before PR 8.1.

---

## Recommended order to close the gate (PR 8.0)
1. **8-C** — real console collision geometry on a separate contact channel; forbidden forearm/console-contact integration test. *(Highest effort; unblocks the evaluator premise of the whole epic.)*
2. **8-D** — unique reset `request_id`; client `reset_ack` wait + `RESETTING` state that holds commands; delete sentinel-file reset and `sleep(1.5)` from the demo.
3. **8-F** — control-panel/research runner refuses non-`READY` mujoco-remote backends and hard-disables fallback in search mode.
4. **8-E** — real recorder provenance (paths, hashes, git sha, versions) + typed/versioned control-state envelope replacing the `/tmp` JSON.
5. **8-B** — validate fixtures through `scene.schema.json` before compile; assert `MjModel` dims per control; enforce interactive↔articulation relations.
6. **8-A** — add the live start/reset/stop lifecycle test.
7. Wire the **hybrid acceptance test** and emit the **evidence bundle**.

See `docs/roadmap/gate-fix.md` for the actionable per-gate fix plan.
