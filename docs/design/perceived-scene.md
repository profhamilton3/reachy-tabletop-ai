# The perceived scene: one interface for #182 stage 2 and EPIC 9

- Status: **Design note, proposed.** No code. Nothing here was run on the robot or the sim.
- Date: 2026-10-21
- Covers: reachy-tabletop-ai#13 (EPIC 9, re-scoped 2026-10-21) and reachy-1-2-sim#182 (stage 2).
- Read against: reachy-tabletop-ai `5aa2d29`, reachy-1-2-sim `7786a65`.
- Evidence:
  - sim `docs/reviews/2026-10-20-perception-stage1-186/` (the #186 results);
  - sim ADR-0010 (October layout);
  - the head-sweep analyser results on #28 and PR #44 here.

## 0. What this note decides, and what it leaves open

Both epics need the same input: **what is on the board, where it is, and how wrong that could be.**
- #182 stage 2 uses it for the object to grasp.
- EPIC 9 Phase 2 uses it for the objects to avoid.
- EPIC 9 Phases 0 and 3 store it across restarts.

This note defines that input once, as `PerceivedScene`. It also defines how each consumer uses it, and where each consumer must refuse.

Two facts set the scope:

1. **With today's evidence, every consumer below refuses.** The #186 horizontal error is 10.9 mm median and 23.3 mm p95. The crane's lateral correction cap is 8 mm. The side-on grasp's slack is 3.9 mm per side. So the contract and the refusal paths can be built and tested now, but nothing becomes permitted until the bounds shrink. The bounds are measured, not argued (section 5).
2. **The perceived scene can only add refusals.** It never removes a check: not preflight, not D1, not the exit check, not the contact rules, not `safety.gate_check()`, and not `motion/primitives.py`. The one exception is lifting the "two or more objects on the board" refusal (ADR-0010 §7). That stays an explicit owner decision with its own evidence (section 3).

## 1. The contract

### 1.1 Conventions

| | |
|---|---|
| Frame | **URDF torso:** X forward, Y left, Z up, origin at the torso. This is what `stereo_depth.camera_to_torso` and the #186 grader use. |
| Sim scene frame | torso + `ARM_BASE_IN_WORLD` = (0.206, 0, 1.15) m, no rotation (ADR-0010). Converting is the consumer's job. The contract stays in the torso frame. |
| Units | Metres, with `_m` suffixes. Angles are in degrees, with `_deg` suffixes; that covers only neck angles and yaw. Confidences are in [0, 1]. |
| Quaternions | `[w, x, y, z]`, unit norm within 1e-6. This matches `SceneObject.quat` and `truth.v1`. |
| Time | `captured_utc` is when the workstation received the pair. The v1 camera gRPC message carries no source timestamp. |
| Schema | `perceived_scene.v1`. Strict: unknown fields fail validation, as `truth.v1` does. Adding an optional field with a default is a minor change. Removing a field, or changing its meaning, bumps the version. |

All types are frozen dataclasses with `to_dict` / `from_dict` and validation in `__post_init__`. This is the style of the sim's Section 7 contracts (`experience/models.py`). The module must also run on Python 3.8, because the sim's panel container is ros:foxy (see OD-1).

### 1.2 `RigIdentity`

This is the world the perception was made in. It plays the role of `SimulatorIdentity`. A field mismatch is a **rejection, never a score**, the same rule as `experience/compatibility.py`.

```python
@dataclass(frozen=True)
class RigIdentity:
    identity_version: int            # 1
    source: str                      # "sim" | "robot"
    rig_layout_id: str               # measured scene the rig is modelled by, e.g. "FWDCenterLabMCC"
    mount_id: str                    # robot mount, e.g. "october-2026" = torso (0.206, 0, 1.15) in the twin
    calibration_id: str              # as recorded: robot "<profile>@zoom=OUT"; sim the plain profile id
    zoom_level: str                  # camera read-back: "OUT" | "INTER" | "IN" | "ZERO"; both eyes must agree
    intrinsics_sha256: str           # robot: SHA-256 of the stereo profile file; sim: StereoGeometry.identity["id"]
    rig_epoch: int                   # bumped by the store when the rig check fails (section 4.4)
    scene_name: Optional[str] = None         # sim only
    scene_sha256: Optional[str] = None       # sim only
    scene_revision: Optional[str] = None     # sim only; required for any sim motion use

MATCH_FIELDS = ("source", "rig_layout_id", "mount_id", "calibration_id", "zoom_level",
                "intrinsics_sha256", "rig_epoch", "scene_sha256", "scene_revision")
```

- **Zoom.** Zoom is carried on its own field as well as inside `calibration_id`. On the robot, `camera_zoom.calibration_id` folds the zoom in. On the sim, the viewer keeps the plain profile id (`stereo_app.py`, `tests/test_camera_zoom.py`). The separate field makes sim and robot match on the same field list without changing that code.
- **Bad zoom refuses.** No `RigIdentity` is built when `zoom_status` reports any of these: an unreadable zoom, eyes that disagree, or `matches_profile` other than `True` on the robot. A capture like that has no valid depth.

### 1.3 `PipelineIdentity`

This is what produced the numbers. It is kept apart from the rig on purpose:
- a rig change makes **memory** stale (section 4);
- a pipeline change makes **bounds** stale (section 1.6).

```python
@dataclass(frozen=True)
class PipelineIdentity:
    detector_model_sha256: str       # e.g. 60dad10f8e6a… (the #186 best.tflite)
    detector_labels: Tuple[str, ...] # ("cube", "cylinder", "empty")
    depth_method: str                # "sgbm-bidir-1px/inner60-median" (stereo_depth._analyze)
    surface_correction_id: str       # "none" | "class-half-extent.v1" (section 1.6)
    bounds_model_id: str             # which BoundsModel rows produced the bounds
    code_sha: str                    # tabletop-ai commit; "" plus dirty=True refuses motion use
    dirty: bool
```

### 1.4 `BoardPose` and the nine cells

```python
@dataclass(frozen=True)
class BoardPose:
    position_m: Tuple[float, float, float]     # taped-grid centre, torso frame
    quat_wxyz: Tuple[float, float, float, float]  # board axes: +x toward far row, +y toward c1
    bounds_m: Tuple[float, float, float]       # per-axis half-width, torso axes
    source: str                  # "head_sweep_fit" | "single_pair_fit" | "twin" | "taught"
    poses_used: int              # head poses that found the grid
    fit_rms_mm: float            # Kabsch residual (analyze_capture_bundle.board_centre_camera)
    spread_mm: float             # worst deviation from the mean over poses; 0 for one pose

@dataclass(frozen=True)
class CellReport:
    name: str                    # "r1c1" … "r3c3", always all nine, fixed order
    state: str                   # "cube" | "cylinder" | "empty" | "unknown"
    object_index: Optional[int]  # into PerceivedScene.objects when state is cube/cylinder
```

- **Unknown is never empty.** This is #182's per-square requirement. A cell the detector did not report, one the chest hides (r1c2 and r1c3 were `unknown` in 6 and 8 of their 10 sightings), or one whose detection could not be assigned is `unknown`.
- **Cells come from the board, not from sorting boxes.** Cells are projected from `BoardPose` with the known grid pitch. Sorting boxes into a 3×3 grid is the fallback, and it sets `source="single_pair_fit"` with no head sweep behind it.

### 1.5 `PerceivedObject` and `PerceivedScene`

```python
@dataclass(frozen=True)
class SourcePair:
    pair_id: str
    bundle_sha256: str                         # the viewer's single-pair zip manifest hash
    left_sha256: str
    right_sha256: str
    captured_utc: str
    neck_deg: Tuple[float, float, float]       # read-back (roll, pitch, yaw), not the command

@dataclass(frozen=True)
class PerceivedObject:
    kind: str                                  # detector vocabulary minus "empty": "cube" | "cylinder"
    kind_confidence: float                     # detector score
    centre_m: Tuple[float, float, float]       # torso frame, AFTER the surface correction
    surface_point_m: Tuple[float, float, float]  # raw stereo median (visible surface), for audit
    bounds_m: Tuple[float, float, float]       # per-axis half-width on centre_m at bounds_coverage
    bounds_coverage: float                     # e.g. 0.95
    bounds_provisional: bool                   # True: may only refuse (section 1.6)
    quat_wxyz: Optional[Tuple[float, float, float, float]]  # None today: no yaw is estimated
    yaw_bound_deg: Optional[float]
    size_m: Optional[Tuple[float, float, float]]  # full extents; None if unknown
    size_source: str                           # "unknown" | "class_prior" | "measured"
    cell: Optional[str]                        # cell whose footprint holds centre_m ± bounds; None if ambiguous
    stereo_samples: int
    stereo_coverage: float
    depth_iqr_m: float
    source: SourcePair
    calibration_id: str                        # repeated from the rig, so an object stands alone in logs

@dataclass(frozen=True)
class PerceivedScene:
    schema_version: str                        # "perceived_scene.v1"
    scene_id: str                              # uuid4 hex
    created_utc: str
    rig: RigIdentity
    pipeline: PipelineIdentity
    board: Optional[BoardPose]                 # None: no cells, and no motion use
    cells: Tuple[CellReport, ...]              # nine, or () when board is None
    objects: Tuple[PerceivedObject, ...]
    off_board: Tuple[PerceivedObject, ...]     # detections outside the board footprint; never obstacles or targets
    warnings: Tuple[str, ...]
```

- **Off-board detections are listed, not dropped.** On real photos, glass-wall panes come back as "empty" squares (#182 finding), and the parked floor pool is in view at `board_view`. These go in `off_board` so they are visible in logs. They never become targets or obstacles.
- **There are no object ids.** Perception does not know that a cube is `red_cube`, so `PerceivedObject` has no id. Associating an object with a request is the consumer's job (section 2.1). `VisualObservation` from #13 §6.2 becomes `PerceivedObject` (or `BoardPose`) plus the persistence fields in section 4. `ground_truth_error` is not in the contract: grading stays in the sim's grader.
- **Building a scene moves only the head.** Capture is event-level: once per task or recovery, never in a per-frame loop. The head goes to `board_view` only through the head-only primitives and `safety.head_gate_check()` (owner decision of 2026-10-20). No arm moves.

### 1.6 Where the bounds come from

Bounds are **looked up, never typed in.** A `BoundsModel` row is keyed by:
- `(detector_model_sha256, depth_method, surface_correction_id, calibration_id, zoom_level)`;
- the head-pose name, the kind, and the camera-distance band (the grader's bands: <0.45, 0.45–0.50, 0.50–0.55, 0.55–0.60, ≥0.60 m).

It holds:
- a per-axis bias, which is subtracted;
- a per-axis half-width at the stated coverage;
- `n_distinct`;
- the SHA-256 of the grader `result.json` it was read from.

If no row matches, the builder emits the object with `bounds_provisional=True` and infinite bounds. That makes it unusable for motion, so the system fails closed.

**The rows #186 supports today** (`result.json`: profile `771392183ddc6c62`, detector `60dad10f8e6a…`, `board_view`, all bands pooled):

| | Measured | Read as |
|---|---|---|
| Horizontal (x, y) error | median 10.9 mm, p95 23.3 mm | Per-axis x and y half-width ≤ 23.3 mm (a norm bounds each axis) |
| Centre (3D) error | median 24.5 mm, max 31.3 mm | z half-width ≤ 31.3 mm, uncorrected |
| Along-ray error | median −24.1 mm, from −9.6 to about −30 mm by cell | Systematic: the viewer measures the near surface |
| `n` | 20 matched, but pair1 = pair2 (deterministic renders) | **10 distinct**; p95 is the max |
| Per kind | cubes: 2 of 22 correct | No per-kind row is supportable |

So every #186 row is **provisional**. With OD-2's default minimum of 30 distinct sightings per row, provisional bounds can refuse a move but never permit one.

**The surface correction** (`class-half-extent.v1`) moves the stereo median away from the camera along the view ray by the class prior's half-extent along that ray. The priors are owner-confirmed: cube 6 cm; cylinder 7 cm × 10 cm (sim `docs/scene-object-dimensions.md`). At `board_view` the ray is steep but not vertical, so part of the −24 mm along-ray bias lands in x. The correction may therefore remove much of the 10.9 mm horizontal error. **That is a hypothesis.** It becomes a bounds row only after it is re-graded (P1b), and the uncorrected #186 rows do not carry over to it.

## 2. #182 stage 2: a perceived grasp target

### 2.1 Flow

1. **Request.** Lift `<object>`, or place it on `<cell>`. In the sim the object has a scene entry. On the robot, later, a request names `(kind, cell)`.
2. **Fresh scene.** Take a `PerceivedScene` captured after the last arm motion. If any arm motion happened since capture, or the sim `scene_revision` changed, the scene is stale for planning. The consumer refuses, with that reason.
3. **Associate.** There must be exactly one `PerceivedObject` whose `cell` equals the request's cell and whose `kind` equals the request's kind. In the sim, the cell comes from the scene entry's current cell (for association only). Zero matches gives "not seen". More than one gives "ambiguous". A cell reported `unknown` gives "cell not visible". A kind disagreement gives "seen as cylinder".
4. **Gate.** The skill declares its `PerceptionTolerance`, and the perceived object must fit it (2.2).
5. **Substitute.** The perceived object becomes the plan's `SceneObject`:
   - `id`: the requested id;
   - `kind`: `cube` → `box`, `cylinder` → `cylinder`;
   - `center`: `centre_m` + `ARM_BASE_IN_WORLD`;
   - `size`: from `size_m`, or the class prior with `size_source` recorded;
   - `quat`: from `quat_wxyz`.

   It is placed into a `SceneModel.copy()`. The unmodified scene is never edited.
6. **Plan and fly through the existing path.** `plan_crane_pick` / `check_crane_plan`, or the side-on and placement planners, then preflight, D1, the exit check, the contact rules, and the lease. Nothing is skipped or loosened.
7. **Refuse and fall back.** On a refusal at step 2, 3 or 4, fall back to (a) with the scene entry, where one exists. The run record shows the fallback as `target_source: "scene_entry"` with `perceived_refusal: "<reason>"`. A fallback is never silent. On the robot there is no scene entry, so a refusal is final.

### 2.2 The tolerance gate

Each skill declares its tolerance in its own module. The value is next to the constant it is derived from, and the derivation is in the docstring:

```python
@dataclass(frozen=True)
class PerceptionTolerance:
    max_bound_xy_m: float        # per-axis half-width allowed on the target centre
    max_bound_z_m: float
    requires_yaw: bool
    max_yaw_bound_deg: Optional[float]
    allows_class_prior_size: bool
```

The gate refuses when any of these holds:
- `bounds_provisional` is set;
- a bound exceeds the tolerance;
- `requires_yaw` is set and `quat_wxyz` is None;
- the size source is not allowed;
- the pipeline is `dirty`.

The refusal names the number and the limit. For example: "x bound 23.3 mm exceeds the crane's 8 mm".

**Worked example, today:**

| Skill | Natural tolerance (sim constants) | Perceived today | Verdict |
|---|---|---|---|
| Crane (box) | `CraneLimits.max_lateral_correction_m` 8 mm; needs the object's yaw (the jaw closes on two faces) | 23.3 mm, provisional, no yaw | refuse (three reasons) |
| Side-on (cylinder) | `side_slack_total_m` 7.8 mm (3.9 mm a side); cylinder yaw-free | 23.3 mm, provisional | refuse |
| Side-on (cube) | same slack; held over ±3° yaw error | no yaw | refuse |

The tolerances in this table are illustrative. The skill owner sets each one with its own derivation (OD-4).

### 2.3 What stage 2 can honestly claim in the sim

In the sim, the crane's hover and descent corrections read the object's live pose from the simulator (`Observation.position`) and correct up to the 8 mm cap. A perceived target that is 6 mm off is therefore quietly fixed by truth. That is useful for measuring how much correction perception would need. It is **not** a perception capability, and #182 says truth is for grading once a stage claims one.

The recommendation (OD-3) has two parts:
- **Stage 2 is "perceived plan input, truth-fed corrections".** Each run logs `correction_source: "sim_truth"` and the correction it needed. That correction is the perception error, measured in flight.
- **Perception-fed corrections belong with stage 3.** These are a re-perception at the hover, with close-range bounds of their own. Stage 3 is where the robot must manage without truth.

## 3. EPIC 9 Phase 2: perceived obstacles before a move

### 3.1 Turning objects into obstacles

The precheck builds a `SceneModel.copy()` and adds one `SceneObject` per perceived object, **inflated by its bounds**:
- **Box:** size + 2·bounds per axis, with yaw unknown taken as the worst case. For an unknown yaw, a box's footprint becomes the circle around its diagonal.
- **Cylinder:** radius + max(bₓ, b_y); height + 2·b_z.
- **`unknown` cell:** the envelope of the largest object the rig is known to hold, standing on that cell. The soda can, 12.2 cm, sets the height.
- **Unknown size:** the class prior, then inflated.

The existing whole-arm checks then run on that model with the target excluded: `SceneModel.clearances` over `kinematics.link_capsules`, plus each skill's own neighbour clearance (`neighbour_clearance_m`, 10 mm). The target's own uncertainty is the tolerance gate's job (2.2). Inflating the target would break the straddle geometry.

### 3.2 It only adds refusals

```
allowed_with_perception = allowed_by_existing_checks AND perceived_precheck_passes
```

This is enforced by a test over fixture scenes: the perceived precheck never turns an existing refusal into a pass. It sits before preflight and `gate_check`, never in place of them. Moves still go only through `motion/primitives.py`, with `REACHY_ENABLE_MOTION` unchanged.

### 3.3 Lifting the "two or more objects" refusal

ADR-0010 §7 refuses two or more objects for a measured reason: **objects interacting were never flown.** That is a reach that clears its target by threading past a second object. Perception does not supply that evidence. The refusal is lifted for one ability only when all four of these hold:
1. **Clearance.** The perceived precheck passes with non-provisional bounds, every cell is reported (no `unknown`, or `unknown` taken as occupied), and every inflated obstacle clears by the skill's neighbour clearance.
2. **Sim agreement.** In the sim, every perceived object associates with a scene object within its bounds. Any disagreement refuses.
3. **Flights.** Multi-object offline flights for that ability exist and are reviewed, in the style of ADR-0010 §7.
4. **Owner sign-off.** The owner approves the lift for that ability. This is a flag in `_ability_available`, off by default.

**Phase 2 exit gate (the sim):** over a fixed set of layouts with deliberately placed obstacles, the perceived precheck and the truth precheck (`SceneModel` on the live poses) are compared move by move:
- **perceived says clear, truth says collide:** must be **0**;
- **perceived refuses, truth says clear:** counted and reported. This is the cost of the bounds.

## 4. Memory: EPIC 9 Phases 0 and 3

### 4.1 The store

`SceneMemoryStore` is a SQLite file in tabletop-ai, mirroring `ExperienceStore`:
- the schema version is in `PRAGMA user_version`, and migrations run on every `open()`;
- WAL mode;
- every write in a transaction;
- `open()` is a context manager that always closes;
- large data (pair bundles) stays in files, and the store holds paths and SHA-256s.

| Table | Holds |
|---|---|
| `rigs` | each `RigIdentity` seen (JSON + match-field columns), first and last seen |
| `scenes` | one row per stored `PerceivedScene`: id, rig id, pipeline JSON, SHA-256 of the document, path |
| `observations` | one immutable row per board pose or object in a scene |
| `entries` | `SceneMemoryEntry`: kind, estimate, bounds, state, counts, rig id, `superseded_by` |
| `confirmations` | (entry, observation, per-axis residual, verdict), for audit |

### 4.2 Fail closed

- **The rig must match to read.** `open(path, current_rig)` serves entries only where all of the rig's `MATCH_FIELDS` equal the current rig. Others are not deleted. They are reported as `inactive: rig mismatch (<field>)`.
- **Bad input fails.** An unreadable file, an unknown schema version, or a `RigIdentity` that cannot be built (bad zoom) raises. It never reads as empty.
- **Memory is never a motion input.** It is a prior, used for association, for "rig moved?", and for #181 context. A target or obstacle for motion always comes from a fresh `PerceivedScene` (section 2.1, step 2).

### 4.3 Counting confirmations, not overwriting

Entries have a state: `candidate` → `confirmed` → `superseded` | `invalidated`.

- **Agree:** an observation confirms an entry when |Δ| ≤ b_entry + b_obs on every axis. The count goes up, `last_confirmed_at` moves, and the estimate becomes the count-weighted mean.
- **The bound does not shrink with n.** It stays at least the single-observation bound. Repeated sightings share the same systematic error (the #186 along-ray bias is the example), so averaging hides it rather than removing it.
- **Disagree:** a disagreeing observation never edits the entry. It opens or confirms a competing `candidate`. The competitor supersedes the original only after K confirmations while the original gets none over the same span (OD-6). Then `superseded_by` is set and both rows stay.
- **What is stored:**
  - `board_pose` is a rig fact and is confirmation-counted;
  - `object_sighting` is episodic (objects move) and never confirmed into a fact;
  - `rig_check` is the history of section 4.4.

### 4.4 "Has the rig moved?"

The head-sweep analyser fits the taped grid in both eyes at several neck poses and carries the centre into the torso frame. On the sim v2 bundle (#28, PR #44):
- 7 poses, mean centre (0.2284, 0.0014, −0.4106) m against the scene grid's (0.2258, 0, −0.4095) m, which is about 3 mm apart;
- worst pose 3.2 mm from the mean;
- the test bound is 10 mm.

On the robot, #185's 17-view camera check spread 6 / 4 / 9 mm in x / y / z. Robot head-sweep bundles are still to come (#28).

**The rig check:** compare a new head-sweep `BoardPose` with the confirmed `board_pose` entry, and in the sim with the twin's grid.
- **Pass:** it confirms the entry.
- **Fail** (beyond the threshold, OD-7: 10 mm default): the store bumps `rig_epoch`, every entry under the old epoch goes `inactive: rig moved`, and the operator is told. The board moved 8.5 cm in August and about 0.24 m by October (#185), so this is a real event, not a hypothetical.

The rig check replaces the dropped rail and table-edge detector. The rig is already in the measured scene, so a matching board pose stands in for "rig matches twin".

## 5. The training and grading loop

```
generate_dataset.py / build_detection_handoff.py   (sim: labelled renders, scene geometry)
        → train in tabletop-ai (#12 pipeline)        → best.tflite, new detector_model_sha256
        → capture graded layouts in the sim          (the #186 procedure, held-out layouts)
        → perception_grade.py (#205)                 → result.json
        → BoundsModel rows (section 1.6)             → what stage 2 and Phase 2 may use
```

- **The first targets come from #186:**
  - grey cubes read as `empty` (cube recall 0.09; `empty` precision 0.0 on the full boards);
  - `red_cube` read as `cylinder` (4 of 4);
  - `blue_cylinder` never detected (0 of 4).
- **The generator lacks `empty` boxes.** The deployed detector has an `empty` class, which is a box on an empty square. `generate_dataset.py` labels objects only. It needs per-cell `empty` boxes projected from the cells (#182 finding 4).
- **Grading layouts stay held out.** They are disjoint from training layouts and seeds, and the dataset manifest's SHA-256 is recorded next to the detector hash.
- **Retrain or fix the sim's look?** That waits on Siva's robot images (#28):

| Deployed detector on robot images with objects | Then |
|---|---|
| Works (cubes and cylinders found) | The gap is the sim's look: exposure, background, plain table (#182 findings 3–5). Fix the renders; don't train on them yet. |
| Fails as in the sim | Retrain with sim renders plus the real photos, and grade both. |
| Mixed | Both: fix the look first, then retrain, so the grade measures one change at a time. |

**The cheapest step needs no training.** Re-grade the #186 bundles with the surface correction on (P1b). That tests the section 1.6 hypothesis on data already captured. The zips are held by the owner, not committed.

## 6. #181 (b): context for the outcome model (not designed here)

#181 conditions on context (object, cell pair, scene revision). A `PerceivedScene` can supply:
- the perceived kind;
- source and target cells;
- the scene's `scene_id`;
- the target's bounds, as a quality covariate.

Two rules for the record, and nothing more:
- **Record the source.** A run records `target_source` and `scene_id`, so outcomes from perceived and scene-entry targets are never pooled by accident.
- **Context keys are `kind@cell`, not object ids.** Perceived objects have no ids. `ReuseRequest.obstacles` uses ids today, so a perceived context needs a deterministic key.

## 7. Build plan and exit gates

**The owner said building waits for the detector decision (#28).** P0, P1 and P1b do not depend on which detector is chosen, so OD-10 asks whether they may start sooner.

| # | Repo | What (built by Sonnet) | Exit gate |
|---|---|---|---|
| P0 | tabletop-ai | `perception/scene_contract.py`: the dataclasses of §1, strict JSON, `perceived_scene.v1` golden fixtures; Python 3.8-clean | Round-trip; unknown field, bad quaternion, NaN, missing cell and bad zoom each refused; fixtures committed |
| P0b | sim | A vendored, version-pinned copy of the validator, plus the same golden fixtures | The sim's validator accepts and refuses the same fixtures, byte for byte (a CI test in each repo) |
| P1 | tabletop-ai | Builder: viewer pair bundle (`analysis.json`, `depth.json`, neck read-back) + head-sweep `BoardPose` + `BoundsModel` → `PerceivedScene`; `BoundsModel` loaded from a grader `result.json` | From committed sim fixtures: nine cells always, `unknown` kept, off-board separated, every #186 row provisional |
| P1b | sim | `perception_grade.py` grades a `PerceivedScene` document; surface correction on or off | Correction off reproduces #186 (10.9 / 23.3 mm); correction on is reported as a new row, n stated |
| P2 | sim | Stage 2 adapter: association, `PerceptionTolerance` per skill, substitution into `SceneModel.copy()`, visible fallback | Offline: every #186-based target refused with a reason; fallback recorded; monotonicity test (no check removed) |
| P3 | sim | Phase 2 obstacle precheck (§3.1); multi-object refusal flag present, off | Layout set: 0 perceived-clear / truth-collide; refusal costs reported |
| P4 | tabletop-ai | `SceneMemoryStore` (§4) | Survives restart; rig mismatch → inactive; rig-check fail → epoch bump; disagreeing sighting never edits an entry |
| P5 | both | Detector retrain or sim-look fix, per §5's table | Graded by #205, with held-out layouts; new bounds rows with n ≥ OD-2's minimum |

Robot transfer (#182 stage 5, EPIC 9 Phase 4) is out of this plan. It needs the robot's torso-frame conversion: `stereo_depth` produces `xyz_torso_m` only for the sim source today.

## 8. Open decisions for the owner

| | Decision | Recommended default |
|---|---|---|
| OD-1 | How the sim consumes the contract. Both repos ship a package named `reachy_ai`, so one cannot import the other. | JSON wire contract; the sim vendors a pinned validator; shared golden fixtures keep the two in step |
| OD-2 | Bounds coverage and the minimum evidence for a non-provisional row | p95; ≥ 30 distinct sightings per (kind, distance band, head pose); provisional rows only refuse |
| OD-3 | Stage 2 scope in the sim | Perceived plan input with truth-fed corrections, labelled `correction_source`; perception-fed corrections in stage 3 |
| OD-4 | Who sets each skill's `PerceptionTolerance` | The skill's module, derived from its own constants and reviewed like a cap change |
| OD-5 | Yaw | Boxes refused while yaw is unknown; yaw estimation is a later, graded addition |
| OD-6 | When a competing memory candidate supersedes an entry | K = 3 confirmations, with none for the original over the same span |
| OD-7 | Rig-moved threshold | 10 mm on the board centre (the test bound; the robot's spread reached 9 mm in #185); re-check after robot bundles |
| OD-8 | What a pipeline change does to memory | Entries stay, their effective count restarts under the new pipeline, and the old count is kept for audit |
| OD-9 | Where Phase 2's precheck runs first | The sim, where `SceneModel` and the kinematics live; the robot port goes with stage 5 |
| OD-10 | Start P0, P0b, P1 and P1b before #28's detector decision | Yes: none of them depends on the detector, and P1b answers the bias question with data already captured |
