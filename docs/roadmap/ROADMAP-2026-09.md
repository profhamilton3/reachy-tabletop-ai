# Roadmap — September 2026

Sequencing plan for the remaining open issues in `reachy-tabletop-ai`, with the
`reachy-1-2-sim` work each one depends on.

Written 2026-08-30. Supersedes the per-issue ordering implied by the issue
numbers themselves — which no longer reflects what blocks what.

**Closed since the last plan:** #1 (repo scaffold + CI), #3 (`reachy_sdk` smoke test).

**Still open:** #2 Coral TPU · #4 Stereo calibration · #5 Force sensor ·
#6 IMU · #7 ReSpeaker · #12 Custom detector dataset · #13 EPIC 9.

---

## 1. The finding that reorders everything

Three of the open issues (#12, #2, #13) want to train or run a vision model, and
all three assume the simulator can stand in for the real cameras. **It currently
cannot.** The sim's camera model and the measured camera differ on every axis
that matters:

**Orientation is not one of them — the sim is correct and must stay that way.**
The sim's stereo cameras used to render sideways with a vertical horizon; that
was fixed on 2026-08-27 in `reachy-1-2-sim@31fe0d5`, which replaced
`euler="0 -1.5708 0"` with an explicit `xyaxes="0 -1 0  0 0 1"` on both cameras.
The current 640 × 480 landscape render is the corrected state. Nothing in this
roadmap changes it.

The measured calibration was solved on 480 × 640 files, so its `fx`/`fy` and
`cx`/`cy` are expressed in that stored frame. Read into the landscape frame the
sim renders — the convention `reachy-1-2-sim/docs/RESEARCH_LINKS.md` already
uses — the comparison is:

| | Sim (`scenes/calibration_defaults.yaml`) | Real, in landscape frame |
|---|---|---|
| Resolution | 640 × 480 landscape ✓ | 640 × 480 (4:3) ✓ |
| fx, fy | 480, 480 | **408, 407** (L) · **401, 399** (R) |
| Horizontal FOV | 67.4° | **76.2°** (L) · 77.2° (R) |
| Vertical FOV | 53.1° | **61.1°** (L) · 62.1° (R) |
| Distortion k1 | 0.0 — pinhole | **−0.32 (L) · −0.39 (R)** |
| Stereo baseline | 65 mm | **~80 mm** (URDF says 72.5 mm) |
| Provenance | `synthetic_defaults` | `stereo_calibration.npz`, RMS 0.97 / 0.84 px |

So the real error is **focal length, not orientation or aspect**. The sim's
focal length is ~18 % too long, uniformly on both axes, so it sees about **85 %
of the real linear field** in each direction — the aspect ratio is already
right. Setting `fx = fy ≈ 408` at 640 × 480 reproduces the measured
76.2° × 61.1° almost exactly.

That is a real gap and it clips the outer edge of the scene, but it is a
one-line correction, not the rebuild an orientation change would have been.
Distortion remains the larger transfer risk: the real lens bows straight lines
visibly and MuJoCo renders none.

There is also a latent bug: `native_mujoco/model/reachy_1_2.xml:342,346` declares
`fovy="1.0472"`, intending 60° in radians, but MuJoCo reads `cam_fovy` in
**degrees** regardless of `compiler angle="radian"` — so the MJCF as written is a
1.05° telephoto. It is masked at runtime only because `native_mujoco/calibration.py`
overwrites `cam_fovy` on startup. Anything loading the MJCF directly gets the
1° camera.

**Consequence:** sim camera fidelity is Phase 0. It is not part of #4's polish —
it gates #12 and #13 outright.

---

## 2. Phases

### Phase 0 — Sim camera fidelity *(blocking; issue #4, sim track)*

> **Status 2026-08-31:** steps 1, 4 and 5 are done and in review —
> `reachy-1-2-sim` PR #30. Step 2 is a standing constraint, honoured. Step 3
> remains blocked on the hardware question in §4 below.

Bring `reachy-1-2-sim`'s camera model onto the measured calibration so every
later vision phase can use it as a stand-in.

1. Replace `scenes/calibration_defaults.yaml` with a `provenance: measured_2026_08_27`
   profile carrying the real per-eye intrinsics **transposed into the landscape
   frame** (`fx ≈ 408`, `fy ≈ 407` left; `401`/`399` right) and
   `baseline_m: 0.080`. Keep `resolution: [640, 480]` — it is already correct.
   Keep the synthetic profile as a named alternative, not the default.
2. **Leave camera orientation alone.** `31fe0d5` fixed it; do not re-roll the
   cameras and do not change the render resolution. `reachy-1-2-sim/CLAUDE.md`
   already carries the standing rule: *"Do not 'fix' mirrored or inverted frames
   by arbitrary image flips; fix the transform and prove it with an
   axis/calibration scene."*
3. Resolve the `cx`/`cy` transpose direction. The measured principal point
   (253.6, 296.4 left) is in the 480 × 640 stored frame; mapping it to landscape
   is either `(cy, cx)` or `(W−cy, H−cx)` depending on which way the capture
   path rotates. A 90 mm offset either way, so settle it with an
   axis/calibration scene rather than by eye.
4. Fix the MJCF `fovy` units bug so a directly-loaded model is not a 1° camera.
5. Add a post-render barrel-distortion pass. MuJoCo renders pinhole and cannot
   be made to distort; applying `k1/k2` to the rendered RGB after the fact is
   the only way sim frames resemble real frames. **Shipped opt-in
   (`--distortion`), not opt-out as planned here** — renders are ground truth
   for the collision and evaluation paths, and silently warping them would
   corrupt consumers that never asked for it. The same warp is applied to depth
   and segmentation (nearest-neighbour) so auto-generated labels stay aligned
   with the RGB, which is what makes Phase 1's synthetic-label pipeline viable.
6. Record the baseline discrepancy as an explicit open question in the profile:
   photogrammetry says ~80 mm, the URDF says 72.5 mm, and depth accuracy scales
   directly with that number. Measure it on the robot before trusting metric depth.

**Exit criterion:** a sim render of `FWDCenterLabMCC` and a real photo from the
same nominal head pose show the same table edge and the same rail edges at
comparable image positions.

**Caveat carried forward:** the head-mount height error is still open — per-view
pose puts the real camera 0.553 m above the table, the sim says 0.408 m, and the
reach cross-check locates the 145 mm in the head/torso offset, not the table.
Phase 0 fixes the *lens*; it does not fix *where the lens is*. Both matter for
depth, so keep the shoulder-to-camera measurement on the hardware list
(`docs/LAB_EVIDENCE.md`).

---

### Phase 1 — Detection stack *(#12, #2)*

#### 1a. Resolve #12's blocker — **already resolved by inspection**

#12 opens with "neither `data/annotation` nor any `.xml` label files exist."
That is true, and the reason is now clear: **the dataset Siva described is in the
repo, but it is a classification set, not a detection set.**

- `data/calibration/annotation/{cube,cylinder,empty}/` — 8 raw `.jpg` files,
  one class per folder, no boxes.
- `scripts/CollectImages.ipynb` writes to exactly that path, pulling frames off
  the robot's right eye via `reachy_sdk`.
- `scripts/TrainModelWithSampleImages.ipynb` and `scripts/PatchModel.ipynb`
  consume it via `mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite` — the Coral
  **on-device imprinting** flow, which learns from folder names and needs no
  bounding boxes at all.
- Output: `scripts/reachy_classifier.tflite` + `scripts/reachy_labels.txt`
  (`0 empty`, `1 cube`, `2 cylinder`), exercised by `scripts/TestModel.ipynb`.

So there are no MakeSense XML labels because the pipeline Siva actually built
never needed them. **#12 as written asks to convert a classifier dataset into a
detector dataset**, and 8 images augmented to 25–30 is enough to imprint a
3-class classifier but is nowhere near enough to train a YOLO detector.

**Two decisions for Siva, and #12 should be edited to state which was taken:**

- **(A) Keep classification.** Reachy already answers "is there a cube in front
  of me." Cheap, working, on-TPU today. But it yields no pixel location, so it
  cannot drive a grasp — grounding would have to come entirely from depth.
- **(B) Move to detection (recommended).** Needed for pick-and-place, which is
  Siva's own stated next item. Requires boxes.

If (B), the annotation cost is the whole problem — and Phase 0 removes it:

#### 1b. Synthetic detection data from the simulator

`FWDCenterLabMCC.yaml` gives every object a `semantic_class` (`grid.cell`,
rails, table), and `native_mujoco/renderer.py` already emits a **body-ID
segmentation buffer** (`seg_b64`, uint16) alongside RGB. Segmentation mask +
`semantic_class` → an exact bounding box, with no human in the loop.

Add cube/cylinder objects to a scene variant, randomize their pose over the
nine grid cells plus lighting and colour, and render thousands of
perfectly-labelled frames through the Phase-0-corrected camera. Hand-annotate
only a small **real** validation set — the real images exist to measure
sim-to-real transfer, not to train.

This is the same discipline as EPIC 9 §5.1: ground truth first, real camera
second.

#### 1c. Close out #2

More is done than the issue reflects: `data/models/efficientdet_lite2_448_ptq_edgetpu.tflite`
is present (the issue names it `efficientdet_lite2.tflite`; the on-disk file is
the edgetpu-compiled 448 PTQ variant — reconcile the name in the issue).

Genuinely still open: `scripts/run_coral_benchmark.py`,
`perception/coral_detector.py`, fixture images, `tests/test_coral_detector.py`,
and the latency writeup.

**Split #2 by where the work has to run:**

- *Sim-side / offline:* fixture images can come straight from the Phase 0
  renderer, giving `tests/fixtures/images/` real lab geometry instead of stock
  photos, and letting `tests/test_coral_detector.py` run in CI on the Tier 1
  offline suite.
- *NUC-only:* **latency cannot be simulated.** The <80 ms target is a property
  of the physical TPU. Benchmark on the NUC or the number is meaningless.

`perception/coral_detector.py` should present one interface over both the
pretrained EfficientDet path (#2) and the custom-trained path (#12), so
swapping models does not ripple into the planner.

---

### Phase 2 — Depth and grounding *(#4, remaining)*

#4's last three tasks, re-planned against the corrected sim:

- **`perception/stereo_depth.py`** — rectify + SGBM + depth projection. The sim
  emits **ground-truth depth** (`depth_b64`, float16 metres per pixel), so this
  can be scored against exact truth over thousands of frames before it ever sees
  a real image.
- **The <15 mm ruler test** — currently a one-shot lab measurement at 0.3 / 0.5 /
  0.8 / 1.2 m. Run it in sim first at the same four distances. That turns a
  manual check into a repeatable regression test, and the real ruler test becomes
  confirmation rather than discovery.
- **Zoom test** — no sim analogue; the motorized zoom (65°–125°) is not modelled
  and modelling it is not worth the effort. What the sim *should* carry is the
  format: calibration keyed by zoom level, so each real per-zoom calibration has
  somewhere to land. Operationally, pick one fixed FOV and calibrate it.

Then `grounding/depth_to_robot.py` (camera frame → robot frame), which is
currently an empty stub and is the piece #13 needs to convert a detection into
something the motion planner can reason about.

**Blocked on hardware:** the ~80 mm vs 72.5 mm baseline and the head-mount
height. Metric depth error scales with both. Do not sign off on <15 mm until
they are measured.

---

### Phase 3 — Hardware sensor sweep *(#6, #5, #7 — parallel, independent)*

None of these block or are blocked by the vision work. Run them whenever robot
time is free.

- **#6 IMU** *(Siva)* — pure discovery: `ros2 topic list`, find the topic,
  echo it, wire it into telemetry. No sim work is worth building. Cheapest open
  issue; the only risk is that the IMU is not exposed on a topic at all, in
  which case the finding itself is the deliverable.
- **#5 Force sensor** *(Parul)* — the sim has `native_mujoco/gripper.py` and
  `GRIPPER_MODEL.md` with contact forces, so `close_gripper_until_force()` can be
  written and unit-tested against simulated contact before hardware. But the
  central unknown — *does `reachy_sdk` v1 expose gripper force at all, or only
  DXL `present_load` as a proxy* — is answerable only on the robot, and it
  determines the primitive's whole design. Answer that first; the sim work is
  only worth doing once the answer is "yes, in some form."
  This is also the issue that finally creates `motion/primitives.py`, which
  `CLAUDE.md` mandates and which does not yet exist.
- **#7 ReSpeaker** *(Terrance)* — **deferred by Siva to after Sept 12** per his
  Aug 26 email; he intends to test the full audio pipeline then. Device
  enumeration and 4-channel confirmation can be done any time and are worth
  doing early, since a missing driver is a lead-time problem.

---

### Phase 4 — EPIC 9, persistent visual scene memory *(#13)*

#13 is the umbrella all of the above feeds, and it is correctly last. Its
dependencies, made explicit:

- **Phase 0** — its §5.1 principle ("nothing is pointed at a real camera until
  scored against simulator ground truth") is void if the sim camera is wrong.
- **Phase 1** — its structural-edge detector is the same training loop as #12's
  synthetic-data pipeline, aimed at rails and table edges instead of cubes.
  Build #12's generator so #13 can reuse it rather than writing a second one.
- **Phase 2** — its avoidance precheck needs a detection converted to a robot-frame
  obstacle, which is `grounding/depth_to_robot.py`.

Do not start #13's `RigIdentity` / `SceneMemoryEntry` store until Phase 2 lands.
The store's whole value is fail-closed invalidation keyed on calibration
identity, and there is no stable calibration identity to key on until Phase 0
and the outstanding hardware measurements are settled.

---

## 3. Sequenced view

| Order | Work | Issue | Where it runs | Blocked by |
|---|---|---|---|---|
| 1 | Sim camera fidelity | #4 (sim) | sim repo | — |
| 2 | Detection-vs-classification decision | #12 | decision | — |
| 3 | Synthetic labelled data generator | #12 | sim repo | 1, 2 |
| 4 | `perception/coral_detector.py` + fixtures + tests | #2 | this repo | 1 |
| 5 | Coral latency benchmark | #2 | **NUC only** | 4 |
| 6 | Custom detector training + eval | #12 | either | 3, 4 |
| 7 | `perception/stereo_depth.py` + sim scoring | #4 | both | 1 |
| 8 | Ruler validation, zoom calibration | #4 | **robot only** | 7 |
| 9 | `grounding/depth_to_robot.py` | #4 → #13 | both | 7 |
| 10 | EPIC 9 store + avoidance precheck | #13 | both | 9 |
| — | IMU stream | #6 | **NUC only** | — |
| — | Force sensor API discovery | #5 | **robot only** | — |
| — | ReSpeaker enumeration | #7 | **NUC only** | — |
| — | Audio pipeline | #7 | robot | Sept 12 (Siva) |

## 4. Hardware measurements still outstanding

Several phases above are gated on lab measurements, not code. Collected here so
they can be taken in one session (see `docs/LAB_EVIDENCE.md`):

- Shoulder-to-camera height on the physical robot — resolves the 145 mm
  head-mount discrepancy (Phases 0, 2).
- Stereo baseline, physically — settles ~80 mm vs the URDF's 72.5 mm (Phase 2).
- Rail profile cross-section and board depth (Phase 4 / scene fidelity).
- Does `reachy_sdk` v1 expose gripper force (#5, Phase 3).
- **What the capture path actually does to the frame** (Phase 0, step 3). The
  sim renders landscape and that is correct, but `reachy.left_camera.last_frame`
  lands on disk as 480 × 640 with upright content and no rotation anywhere in
  our code — see `scripts/CollectImages.ipynb`, which writes the SDK frame
  straight through `cv2.imwrite`. Pollen's docs say 1280 × 720. So the robot-side
  vision service is rotating, cropping or resizing before the frame reaches
  Python. Print `last_frame.shape` on the robot and check whether the SDK exposes
  the raw sensor readout. This decides the `cx`/`cy` transpose direction and
  whether the full sensor sees wider than 76.2°. Already flagged, unresolved, in
  `docs/LAB_EVIDENCE.md`, `reachy_1_2_updated_roadmap.md:174` and
  `reachy-1-2-sim/docs/RESEARCH_LINKS.md:65`.

## 5. Related work in `reachy-1-2-sim`

- **sim #28** — Epic 8's promised browser research UI (PR 8.8 / `R12-808`) was
  never built; the work item was silently re-scoped during execution. Not on
  this critical path, but it is the natural place to *watch* the Phase 1
  training runs, so revisit it if synthetic-data generation gets heavy.
- **sim #2** — audit of test scripts and notebooks that ignore scene awareness,
  collision, and physics. Worth clearing before Phase 4, since EPIC 9 builds
  directly on `scene/awareness.py::SceneModel`.
- **Uncommitted:** the `supervisord.conf` fix for the stale `/tmp/.X1-lock` that
  broke `xvfb`/`rviz2` on container restart is still sitting in the sim repo's
  working tree on `fix/rig-rail-height-and-front-rail`. Commit it — the bug
  recurs on every restart without it.
