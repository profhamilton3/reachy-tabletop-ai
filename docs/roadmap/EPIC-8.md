# EPIC 8 — Adaptive Motion Search and Experience Memory

**Project:** Reachy 1.2 Docker + native MuJoCo simulator  
**Repository target:** `profhamilton3/reachy-1-2-sim`  
**Status:** Proposed  
**Dependency:** Epic 6 complete; PR #3 control-panel scene must pass the Epic 8 entry gate below  
**Gazebo:** Not planned. MuJoCo is the sole physics backend for this epic.

## 1. Purpose

Epic 8 adds a simulator-side capability that can automatically try bounded arm-motion variations, evaluate them in deterministic MuJoCo episodes, remember what worked, and reuse successful experience for related tasks.

The first benchmark is the interactive control-panel scene introduced by PR #3. The system should learn reliable recipes for pressing buttons and moving switches/levers without colliding with the console, actuating neighboring controls, saturating joints, or depending on hand-tuned sleeps.

The second benchmark is the existing tabletop pick-and-place task.

Epic 8 is **not** an LLM servo controller. Claude, ChatGPT, Gemini, or a VLM may help create task definitions and review results, but no language model may emit raw joint commands in the real-time control loop. Search happens over typed, bounded trajectory recipes executed by deterministic motion and simulation code.

## 2. Expected outcome

At Epic 8 completion, a researcher can:

1. select a task such as `operate_control(btn_red)` or `pick_and_place(red_cube, target_pose)`;
2. reset the exact model and scene deterministically;
3. run many headless candidate episodes faster than wall clock where the host permits;
4. reject unsafe or invalid trajectories using authoritative MuJoCo state/contact data;
5. rank successful candidates by robustness, accuracy, smoothness, effort, and duration;
6. resume a study after interruption;
7. retrieve the best compatible prior recipe for a similar task;
8. replay finalists through the normal Reachy v1 SDK → Docker gRPC → native MuJoCo path;
9. promote a validated recipe to versioned YAML without modifying Python source;
10. retain a reproducible evidence bundle for every promoted result.

## 3. Relationship to the other Reachy AI repository

The separate [`reachy-tabletop-ai`](https://github.com/profhamilton3/reachy-tabletop-ai) repository already owns application-layer concepts such as natural-language tasks, perception, VLM planning, operator approval, and eventual physical-robot execution.

Epic 8 should remain in `reachy-1-2-sim` and own only simulator research infrastructure:

- deterministic episode execution;
- trajectory recipe evaluation;
- safety/contact metrics;
- experience persistence and retrieval;
- candidate promotion;
- Docker/native compatibility validation.

A promoted recipe or read-only experience-query API may later be consumed by `reachy-tabletop-ai`. Do not duplicate VLM or perception code in the simulation core.

## 4. Non-goals

The initial Epic 8 does not include:

- Gazebo support;
- end-to-end reinforcement learning from camera pixels;
- neural-network policies in the 500 Hz control loop;
- automatic modification of robot geometry, mass, friction, actuator gains, or collision masks to make a task pass;
- automatic transfer of generated trajectories to physical Reachy hardware;
- unsupervised source-code editing by the optimizer;
- cloud-hosted search as a requirement;
- runtime hot replacement of the MuJoCo model.

A Gymnasium adapter, imitation learning, behavior cloning, diffusion policy, or camera-to-action RL may be proposed after the deterministic episode and experience contracts are stable.

## 5. Core design principles

### 5.1 Fail closed

A learning/search episode is valid only when:

- backend identity is native MuJoCo;
- connection state is `READY`;
- fixture fallback is disabled;
- model and scene hashes match the requested `SimulatorIdentity`;
- reset was acknowledged;
- state age is inside the configured bound;
- all measurements carry authoritative simulation-step identity.

Backend degradation, stale state, scene mismatch, reset timeout, or transport loss produces `ABORTED`, never `SUCCEEDED` or an ordinary task failure.

### 5.2 Search task parameters, not physics truth

Allowed early search parameters include:

- Cartesian standoff and approach offsets;
- approach direction within a bounded cone;
- transit/guard waypoints;
- waypoint count;
- trajectory duration or fixed-step schedule;
- dwell/settle steps;
- gripper angle and close duration;
- press depth;
- hinge sweep extent;
- limited orientation yaw/pitch offsets;
- bounded retry policy.

The optimizer must not vary:

- robot link geometry;
- object/control geometry;
- mass or inertia;
- friction as a path-search variable;
- actuator gains or torque limits as a shortcut;
- collision masks;
- camera calibration;
- success thresholds after a trial starts.

Physics variations may be used later for robustness validation, but they are part of the evaluation distribution, not free decision variables.

### 5.3 Hard constraints precede scoring

Candidate ranking is lexicographic:

```text
valid backend/state
  > safe
  > task success
  > robust across seeds/perturbations
  > accurate
  > smooth / low effort
  > fast
```

A faster collision path can never outrank a slower safe path.

### 5.4 Fast search, faithful validation

Use two execution paths:

1. **Direct native episode runner** — fixed simulation steps, no WebSocket, no SDK property writes, no rendering by default, and no wall-clock sleep. This is the high-throughput search path.
2. **Compatibility validation path** — replay finalists through Reachy v1 SDK → Docker gRPC → `MujocoRemoteBackend` → native MuJoCo. This catches batching, timing, unit, and transport effects before promotion.

Both paths must share the same model, scene compiler, actuator model, gripper/contact model, interactive controller, reset logic, joint mapping, and evaluator code.

### 5.5 Immutable provenance

No experience is reusable without an exact or explicitly compatible `SimulatorIdentity`.

## 6. Mandatory entry gate from PR #3

The following work must be completed on PR #3 or in a dedicated PR 8.0 before the optimizer is implemented

### Gate 8-A — Correct remote backend lifecycle

- Move UID/index maps and `_thread` initialization into `MujocoRemoteBackend.__init__()`.
- Prove `_build_command()` works before any reset.
- Prove reset does not erase the live thread handle.
- Add start/reset/stop lifecycle tests.

### Gate 8-B — Unify schema and compiler contracts

- Compile cylinder `length`, not non-schema `height`.
- Validate test fixtures through the production schema before compiling.
- Assert actual MuJoCo dimensions for every control in `control_panel.yaml`.
- Resolve all schema-supported/compiler-unsupported geometry kinds.
- Enforce interactive/articulation relationships.

### Gate 8-C — Make collision physics truthful

- The console body, ledge, and slanted face must collide with robot links.
- Mounted controls must not be destabilized by their parent fixture.
- Forbidden full-link contacts must be observable by the evaluator.
- Scene-awareness point checks may remain a precheck but are not authoritative.

### Gate 8-D — Deterministic acknowledged reset

- Use unique reset request IDs.
- Add remote `RESETTING` state.
- Wait for matching `reset_ack` and post-reset state.
- Hold/reject commands during reset.
- Remove fixed reset sleeps as correctness mechanisms.

### Gate 8-E — Truthful state, camera, and provenance

- Preserve frame/state `sequence`, `sim_step`, `sim_time_s`, and `scene_revision` across process boundaries.
- Replace stale-prone bare control-state files with a typed API, or add a fully versioned/freshness-checked envelope as an interim step.
- Record actual model path/hash, scene path/hash/revision, Git commit, protocol version, compiler version, physics profile, MuJoCo version, and host architecture.
- Runtime scene load must either be atomic or return `restart_required` rather than false success.

### Gate 8-F — Fail-closed research execution

- Control-panel and search runners must refuse non-MuJoCo backends.
- Fixture fallback must be disabled in search mode.
- Unexpected exceptions must propagate and be classified.
- Physical hardware execution must be denied by default.

### Entry-gate acceptance

The following test must pass before PR 8.1 begins:

```text
clean native-server start
→ verified model/scene handshake
→ acknowledged reset
→ Reachy SDK command before any previous reset
→ intended robot motion
→ one intended control state transition
→ no forbidden console contact
→ state result with matching sim_step/scene_revision
→ clean stop and thread/task shutdown
```

The control path does not yet need high success reliability across all controls. It must be deterministic, physically truthful, observable, and honest about failure.

---

# 7. Shared data contracts

These contracts should be immutable dataclasses or validated Pydantic/dataclass equivalents with explicit schema versions.

## 7.1 `SimulatorIdentity`

Required fields:

```text
identity_version
repository_git_sha
working_tree_dirty
model_source_path
model_sha256
compiled_model_sha256
scene_source_path
scene_sha256
scene_revision
scene_schema_version
scene_compiler_version
physics_profile_id
protocol_version
mujoco_version
python_version
host_os
host_arch
backend_name
calibration_profile_id
sensor_effect_profile_id
```

## 7.2 `TaskSpec`

Required common fields:

```text
task_id
task_type
task_schema_version
scene_revision
arm_policy
timeout_steps
success_definition
forbidden_contact_policy
randomization_profile
```

Control-panel extension:

```text
control_id
requested_final_state
allowed_contact_geoms
neighbor_control_ids
preferred_arm
```

Pick-and-place extension:

```text
object_id
initial_pose_tolerance
target_pose
target_pose_tolerance
required_lift_height
settle_velocity_thresholds
```

## 7.3 `TrajectoryRecipe`

Required fields:

```text
recipe_id
recipe_version
task_type
arm
primitive_sequence
bounded_parameters
joint/velocity/effort limits
source (baseline | search | imported | human-reviewed)
parent_recipe_id
```

Each primitive must use named, typed parameters. Do not store executable Python or arbitrary expressions.

## 7.4 `EpisodeConfig`

```text
simulator_identity
seed
fixed_timestep
max_steps
render_mode (off | rgb | research)
record_trace
contact_sampling
state_sampling
randomization_values
```

## 7.5 `EpisodeResult`

```text
episode_id
trial_id
status
termination_reason
success
start/end sim_step
sim_duration_s
wall_duration_s
metrics
hard_violations
contact_summary
control_state_changes
final object/control states
artifact_paths
warnings
```

Statuses:

```text
PENDING
RUNNING
SUCCEEDED
FAILED
INVALID
ABORTED
CANCELLED
```

## 7.6 `TrialRecord`

A trial binds:

```text
study_id
trial_id
TaskSpec
TrajectoryRecipe
EpisodeConfig
EpisodeResult
SimulatorIdentity
created/started/completed timestamps
optimizer metadata
parent/warm-start trial IDs
promotion state
review metadata
```

---

# 8. Evaluation model

## 8.1 Universal hard failures

An episode is invalid or unsafe when any of the following occurs:

- model/scene identity mismatch;
- unacknowledged or timed-out reset;
- stale state or backend degradation;
- NaN/Inf or unstable simulation;
- forbidden robot-fixture contact;
- forbidden self-collision;
- joint-limit violation outside tolerance;
- sustained actuator saturation above policy;
- object/control leaves a permitted workspace;
- non-target control changes state;
- transport interruption during compatibility replay;
- task timeout;
- evaluator or recorder failure.

## 8.2 Control-panel task success

`operate_control(control_id, desired_state)` succeeds only when:

1. the intended control reaches the desired logical state;
2. the change persists for the configured settle steps;
3. no neighboring control changes;
4. no forbidden console contact occurs;
5. the acting arm retracts to a safe guard/home state;
6. the final state is observed after the actuation step with matching scene revision;
7. backend and recording remain valid through episode completion.

Useful metrics:

```text
intended_state_change
unintended_state_changes
minimum robot-fixture clearance
forbidden contact count/impulse
peak and integrated actuator effort
saturation fraction
joint-space path length
Cartesian path length
jerk proxy
approach error
actuation margin past threshold
settle time
total simulation steps
```

## 8.3 Pick-and-place task success

Success requires:

1. intended object contact/grasp;
2. sustained grip force over a minimum interval;
3. lift above the required margin;
4. retention during transit;
5. target-pose arrival;
6. release;
7. object settling within position/orientation/velocity tolerance;
8. safe arm return;
9. no forbidden contact or non-target object displacement.

## 8.4 Ranking

Recommended lexicographic comparison:

```text
1. valid episode
2. zero hard violations
3. task success
4. robust success rate
5. unintended-state/object disturbance
6. task accuracy
7. minimum clearance margin
8. actuator saturation and effort
9. smoothness
10. duration
```

Use a scalar objective only inside a tier after the hard ordering is enforced.

---

# 9. Pull-request roadmap

Every PR below is intentionally bounded. Claude Code must stop after each PR and report changed files, commands run, test evidence, unresolved risks, and work explicitly deferred.

## PR 8.0 — Control-panel integration correctness closure

**Work item:** `R12-800`  
**May be implemented as fixes on PR #3 rather than a separate post-merge PR.**

### Scope

Complete Gates 8-A through 8-F:

- remote backend constructor/lifecycle;
- cylinder schema/compiler contract;
- schema relational validation;
- console collision channels and contact policy;
- reset acknowledgement/state machine;
- state/camera metadata preservation needed for evaluation;
- actual recorder identity;
- fail-closed demo/research mode;
- truthful scene-load response;
- shutdown/task cleanup;
- hybrid interactive-control acceptance test.

### Required tests

```text
unit: backend constructor/build/reset/lifecycle
contract: schema → compiler → MjModel
native: actual control_panel.yaml, all controls bind
native: forbidden full-link console contact detected
hybrid: SDK → Docker → native → control state
lifecycle: start/reset/disconnect/reconnect/stop
```

### Exit gate

The mandatory entry-gate acceptance sequence in Section 6 passes and produces an evidence bundle.

### Stop condition

Do not add experience storage, an optimizer, or UI search controls in PR 8.0.

---

## PR 8.1 — Experience contracts, persistence, safety, and provenance

**Work item:** `R12-801`

### Proposed files

```text
src/reachy_ai/experience/__init__.py
src/reachy_ai/experience/models.py
src/reachy_ai/experience/store.py
src/reachy_ai/experience/identity.py
src/reachy_ai/motion/recipe.py
tests/unit/test_experience_models.py
tests/unit/test_experience_store.py
tests/unit/test_simulator_identity.py
tests/unit/test_learning_safety.py
```

### Scope

- Implement the shared contracts in Section 7.
- Add SQLite `ExperienceStore` with migrations/schema versioning.
- Support create/start/complete/fail/abort/cancel trial lifecycle.
- Support artifact attachment and interrupted-trial recovery.
- Query only identity-compatible successful trials.
- Add simulator-only learning guard.
- Deny physical execution by default.
- Generate actual `SimulatorIdentity` from the running/compiled system.

### Required behavior

- Startup marks orphaned `RUNNING` trials as interrupted/aborted with reason.
- Identity mismatch prevents reuse unless an explicit compatibility policy exists and is recorded.
- Database writes are transactional.
- Large state/image traces remain in artifact files; SQLite stores metadata and indexes, not unbounded blobs.

### Exit gate

A process can create a study/trial, persist a complete result, restart, retrieve it by exact simulator identity, and refuse it under a changed model or scene hash.

### Stop condition

No episode runner and no optimizer.

---

## PR 8.2 — Shared deterministic simulation core and native episode runner

**Work item:** `R12-802`

### Proposed files

```text
native_mujoco/simulation_core.py
native_mujoco/episode_runner.py
native_mujoco/evaluation_snapshot.py
native_mujoco/cli/run_episode.py
tests/integration/test_episode_reset.py
tests/integration/test_episode_determinism.py
tests/integration/test_episode_contacts.py
```

### Scope

Extract reusable physics behavior from `native_mujoco/server.py` so the live WebSocket server and the episode runner share:

- compiled model/scene loading;
- actuator controller;
- gripper model;
- object tracker;
- interactive controller;
- joint mapping;
- reset implementation;
- control stepping;
- state/contact snapshots.

The episode runner must:

- use fixed simulation steps;
- run without wall-clock sleeping;
- disable rendering by default;
- reset deterministically by seed;
- consume typed trajectory commands;
- expose all MuJoCo contacts needed by evaluators;
- support cancellation and max-step termination;
- optionally emit the existing recording trace plus Episode 8 metadata.

### Determinism test

Same model, scene, recipe, seed, MuJoCo version, and physics profile must produce equivalent terminal state and metrics inside documented numeric tolerances over repeated runs.

### Exit gate

A known fixed recipe can operate one control and run one pick/place episode directly through the native runner, with complete step-indexed metrics and no browser, Docker, WebSocket, or `sleep()` dependency.

### Stop condition

No optimization algorithm.

---

## PR 8.3 — Parameterized baseline recipes

**Work item:** `R12-803`

### Proposed files

```text
recipes/control_panel/baseline_v1.yaml
recipes/pick_place/baseline_v1.yaml
src/reachy_ai/motion/recipe_executor.py
src/reachy_ai/tasks/control_panel.py
src/reachy_ai/tasks/pick_place.py
tests/integration/test_baseline_control_recipe.py
tests/integration/test_baseline_pick_place_recipe.py
```

### Scope

Convert existing known-good logic into serializable typed recipes without deleting the existing demos.

Control-panel recipe primitives may include:

```text
guard
look_at (optional for rendered validation)
approach_standoff
approach_control
press_or_sweep
settle
retract
return_guard
```

Pick/place recipe primitives may include:

```text
home/ready
transit_clear
hover
descend
grasp
settle_grasp
lift
carry
place
release
retreat
home
```

Every tunable parameter must have declared units, bounds, and a stable name. Existing constants become the baseline candidate.

### Required tests

- YAML round-trip;
- bounds validation;
- unknown primitive rejection;
- same baseline behavior as the existing routine within tolerance;
- impossible/through-console recipe fails honestly;
- recipe never contains executable code.

### Exit gate

Both existing tasks can be executed from versioned recipe YAML and produce an `EpisodeResult` stored by `ExperienceStore`.

### Stop condition

Do not change the baseline constants through search in this PR.

---

## PR 8.4 — Safety evaluator and task objective library

**Work item:** `R12-804`

### Proposed files

```text
src/reachy_ai/evaluation/base.py
src/reachy_ai/evaluation/contacts.py
src/reachy_ai/evaluation/control_panel.py
src/reachy_ai/evaluation/pick_place.py
src/reachy_ai/evaluation/ranking.py
tests/unit/test_lexicographic_ranking.py
tests/integration/test_forbidden_console_contact.py
tests/integration/test_unintended_control_toggle.py
```

### Scope

- Implement universal hard-failure rules.
- Implement task-specific success definitions.
- Define explicit allowed-contact sets.
- Detect unintended neighboring control changes.
- Calculate clearance, effort, saturation, path, smoothness, and time metrics.
- Produce a structured explanation for every failure and ranking decision.
- Keep evaluator thresholds versioned and immutable within a study.

### Exit gate

A test corpus of deliberately safe, failed, colliding, stale, unintended-toggle, and successful episodes is classified correctly. A colliding trajectory can never outrank a safe successful trajectory regardless of speed.

### Stop condition

No optimizer or UI.

---

## PR 8.5 — Persistent bounded search engine

**Work item:** `R12-805`

### Proposed files

```text
src/reachy_ai/search/space.py
src/reachy_ai/search/runner.py
src/reachy_ai/search/samplers.py
src/reachy_ai/search/pruning.py
src/reachy_ai/search/cli.py
tests/integration/test_search_resume.py
tests/integration/test_search_pruning.py
```

### Scope

Start with transparent algorithms:

1. baseline evaluation;
2. bounded random or low-discrepancy sampling;
3. local perturbation around the best safe candidate;
4. optional Optuna TPE/CMA-style adapter only after the native search contract is stable.

Requirements:

- persistent SQLite study state;
- reproducible sampler seed;
- bounded parallelism appropriate for available CPU/memory;
- immediate pruning on hard violations;
- cancellation and graceful resume;
- no duplicate candidate evaluation under the same context unless requested;
- no mutation of physics/model/scene files;
- periodic best-candidate checkpoints;
- complete provenance for every trial.

### Exit gate

Starting from the current baseline, a local search improves at least one declared metric or success rate on a fixed control-panel task without violating any hard constraint, and the study resumes after process termination.

### Stop condition

No automatic promotion and no browser controls yet.

---

## PR 8.6 — Experience retrieval and warm starts

**Work item:** `R12-806`

### Proposed files

```text
src/reachy_ai/experience/query.py
src/reachy_ai/experience/similarity.py
src/reachy_ai/search/warm_start.py
tests/unit/test_experience_compatibility.py
tests/integration/test_warm_start_search.py
```

### Scope

Retrieve prior successful recipes using a strict context hierarchy:

1. exact simulator identity and exact task context;
2. exact simulator identity and nearby task geometry;
3. explicitly compatible physics/profile identity and nearby task geometry;
4. otherwise no reuse.

Control-panel similarity can use:

- control type;
- arm side;
- control pose;
- actuation axis;
- joint range and threshold;
- surrounding obstacle region.

Pick/place similarity can use:

- object geometry/mass class;
- start pose;
- target pose;
- preferred arm;
- workspace region.

No neural embedding is required initially. Use deterministic normalized distances and document every warm-start decision.

### Exit gate

A new nearby task starts from a compatible successful prior recipe and reaches an equal or better valid result in fewer trials than an unseeded comparison, with the comparison recorded.

### Stop condition

No cross-identity reuse without an explicit, tested compatibility policy.

---

## PR 8.7 — Robustness curriculum and candidate promotion

**Work item:** `R12-807`

### Proposed files

```text
src/reachy_ai/validation/curriculum.py
src/reachy_ai/validation/promotion.py
src/reachy_ai/validation/report.py
recipes/promoted/
tests/integration/test_candidate_promotion.py
```

### Scope

A single successful trial is never promotable. Validate finalists through staged distributions:

```text
Stage 0: exact nominal reset, repeated runs
Stage 1: bounded initial joint-state variation
Stage 2: bounded control/object pose jitter
Stage 3: bounded mass/friction variation for robustness testing
Stage 4: transport/SDK compatibility replay
Stage 5: rendered review episode and human approval
```

Promotion writes a new immutable recipe version plus:

- parent/baseline comparison;
- trial and validation IDs;
- success/failure matrix;
- confidence interval or transparent sample counts;
- simulator identity;
- artifact manifest;
- human reviewer and timestamp.

Do not overwrite an existing promoted recipe.

### Initial control-panel targets

These are staged research targets, not guarantees:

- nominal: at least 9 successes in 10 repeats for each promoted control recipe;
- robust: at least 90% success over 20 or more declared perturbation seeds;
- zero forbidden fixture contacts;
- zero unintended neighboring-control state changes;
- 100% acknowledged resets and valid episode provenance.

Any adjusted threshold must be versioned before the study begins, not after seeing results.

### Exit gate

At least one button and one hinge control have promoted recipes that pass all stages, including SDK/Docker compatibility replay.

---

## PR 8.8 — Docker/control-panel research UI

**Work item:** `R12-808`

### Proposed capabilities

```text
select task/control/object
select baseline or promoted recipe
start bounded study
cancel study
view backend/model/scene identity
view trial counts and status
view best valid candidate
compare baseline vs candidate
replay selected trial
inspect failure/contact explanation
request promotion validation
approve/reject promotion
export evidence bundle
```

### Requirements

- UI calls a typed localhost-only research API.
- No direct shell commands or arbitrary file paths from browser input.
- No mutation of Python source.
- No physical hardware endpoint.
- Cancellation is cooperative and leaves the database consistent.
- UI must distinguish `FAILED`, `INVALID`, and `ABORTED`.
- Backend identity, scene revision, state age, and fallback-disabled status remain visible.
- Search may run headless while selected replays can render in the camera/RViz views.

### Exit gate

A researcher can start, stop, resume, inspect, replay, validate, and promote a study from the browser without losing provenance or bypassing safety gates.

---

## PR 8.9 — Hybrid compatibility benchmark and research release bundle

**Work item:** `R12-809`

### Scope

For every finalist, compare direct-runner and compatibility-path behavior:

```text
direct native episode runner
versus
Reachy SDK → Docker gRPC → remote backend → native MuJoCo
```

Measure:

- success equivalence;
- command-to-state latency;
- command batching effects;
- trajectory tracking error;
- state/frame age;
- episode real-time factor;
- CPU and memory;
- trial throughput;
- reconnect/reset reliability;
- metric differences within declared tolerances.

Produce a versioned release bundle containing:

```text
EPIC-8 implementation version
Git SHA and clean/dirty status
model/scene/compiler hashes
physics and calibration profiles
study database export
promoted recipes
raw recorder artifacts for representative trials
benchmark tables
known limitations
reproduction commands
Apple-Silicon host details
Docker image digest
```

### Epic 8 exit gate

- Control-panel and pick/place studies can run, resume, and retrieve experience.
- At least one button, one hinge control, and one pick/place recipe pass promotion policy.
- No promoted candidate has forbidden contacts or identity/provenance gaps.
- Finalists pass the normal SDK/Docker compatibility path.
- A clean-machine reproduction follows the documented commands and obtains equivalent results within stated tolerances.
- Epic 8 benchmark/report bundle is checked in or attached to a tagged release.

---

# 10. Dependency order

```text
PR 8.0  Integration correctness and entry gate
   ↓
PR 8.1  Contracts, store, identity, safety
   ↓
PR 8.2  Shared deterministic episode runner
   ↓
PR 8.3  Parameterized baseline recipes
   ↓
PR 8.4  Safety evaluator and objectives
   ↓
PR 8.5  Persistent bounded search
   ↓
PR 8.6  Experience retrieval / warm starts
   ↓
PR 8.7  Robustness and promotion
   ↓
PR 8.8  Browser research UI
   ↓
PR 8.9  Hybrid benchmark and release bundle
```

PR 8.8 may begin in parallel with late PR 8.7 only after the API and persistence contracts are frozen. PR 8.9 continuously accumulates benchmark fixtures but closes last.

# 11. Recommended branch/commit discipline

- One PR per work item.
- Keep simulator behavior and schema migrations in separate logical commits where practical.
- Do not combine dependency upgrades with behavior changes unless required.
- Preserve current demos as regression references until recipe execution replaces them with proven parity.
- Every PR updates tests and the relevant roadmap status.
- Every PR states what was actually run on Apple Silicon versus only offline/unit tested.
- No PR automatically starts the next work item.

# 12. Suggested Claude Code instruction for PR 8.0

```text
Work on PR 8.0 / R12-800 only. Do not implement an optimizer, experience
store, Gymnasium environment, reinforcement learning, or browser search UI.

Use PR #3 head e13e36138083f14574214818cda14126683b02a9 as the review baseline.
Close the mandatory Epic 8 entry gates:

1. Fix MujocoRemoteBackend initialization and lifecycle:
   - initialize UID/index maps and _thread in __init__;
   - request_reset must not create/reset those fields;
   - add pre-reset command and start/reset/stop tests.

2. Unify scene schema and compiler:
   - cylinder uses length;
   - tests validate through the production schema before compile;
   - assert actual control-panel geom dimensions in MjModel;
   - enforce interactive/articulation compatibility.

3. Make console collision behavior physically truthful:
   - robot links collide with fixture surfaces;
   - mounted controls remain stable;
   - add a forbidden forearm/console-contact integration test.

4. Implement acknowledged reset semantics:
   - unique request ID;
   - RESETTING state;
   - wait for reset_ack and post-reset state;
   - hold commands until reset completion;
   - remove fixed reset sleep as a correctness dependency.

5. Fail closed for research/control-panel execution:
   - require verified mujoco-remote READY backend;
   - disable fixture fallback for this mode;
   - catch only expected IK failures;
   - propagate unexpected exceptions.

6. Preserve state/camera/provenance identity required by future episodes:
   - sequence, sim_step, sim_time_s, scene_revision;
   - actual model and scene hashes in recorder metadata;
   - reject runtime scene_load with restart_required unless truly implemented.

7. Add one hybrid acceptance test or reproducible harness:
   Reachy SDK -> Docker gRPC -> MujocoRemoteBackend -> native MuJoCo ->
   intended control transition with matching scene revision and no forbidden
   console contact.

Stop after PR 8.0. Report files changed, exact commands run, pass/fail evidence,
Apple-Silicon tests not runnable in the current environment, and all work deferred
to PR 8.1.
```

# 13. Deferred follow-on epics

The following are intentionally outside initial Epic 8:

- Gymnasium-compatible environment;
- reinforcement learning for recovery/contact behavior;
- imitation learning from promoted trajectories or teleoperation;
- camera-observation policies;
- sim-to-real calibration and transfer;
- automatic physical-robot execution;
- multi-agent or language-model motion planning;
- Gazebo adapter.

These may be proposed only after Epic 8 produces deterministic episodes, trustworthy metrics, compatible experience identity, and promoted baseline recipes.

# 14. Source references

- [PR #3](https://github.com/profhamilton3/reachy-1-2-sim/pull/3)
- [PR #3 reviewed head](https://github.com/profhamilton3/reachy-1-2-sim/commit/e13e36138083f14574214818cda14126683b02a9)
- [Current simulator roadmap](https://github.com/profhamilton3/reachy-1-2-sim/blob/main/docs/ROADMAP.md)
- [Control-panel scene](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/scenes/control_panel.yaml)
- [Scene compiler](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/native_mujoco/scene_compiler.py)
- [Interactive controller](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/native_mujoco/interactive.py)
- [Remote backend](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/mujoco_remote_backend.py)
- [Native server](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/native_mujoco/server.py)
- [Control-panel demo](https://github.com/profhamilton3/reachy-1-2-sim/blob/e13e36138083f14574214818cda14126683b02a9/scripts/demo_control_panel.py)
- [Related application-layer AI project](https://github.com/profhamilton3/reachy-tabletop-ai)
