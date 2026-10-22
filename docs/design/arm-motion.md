# Arm motion from the workstation: taught routes, the arm gate, Part B and T3 Phase 2

- Status: **Design note, proposed.** No production code. Nothing was run on the robot or a live sim. Step 1 is one
  offline MuJoCo replay.
- Date: 2026-10-22
- Covers: reachy-tabletop-ai#66, for #27 Part B (and its 2026-10-22 re-scope) and #49 Phase 2. Feeds reachy-1-2-sim#179.
- Read against: reachy-tabletop-ai `ef6e4ec`, reachy-1-2-sim `1948823` (MuJoCo 3.11.0).
- Evidence: `docs/design/arm-motion-step1/` (scripts, raw output, how to reproduce).

## 0. What this note decides, and what it leaves open

This would be the first **arm** motion commanded from the workstation. Siva has moved the arm from notebooks
(`goto`, ungated). Nothing in this repo does, and `ReachyClient` refuses until there is an arm gate (#68).

The note proposes four things:
- **named routes** of taught joint poses, held as data;
- a **sim clearance record** tied to each route's hash and the rig;
- a **route player** in `motion/primitives.py`;
- an **arm gate** in `motion/safety.py`.

It then designs Part B and T3 Phase 2 on top of them, and drafts the owner decision text. Every number that decides a
route comes from Step 1. Every open choice is in section 11, with a recommended default.

**Step 1's verdict, in five lines:**
1. **Usable now:** Siva's MovePieceLogic poses (`PICK_SAFE`, `PICK`, `TRANSPORT`, `B2_SAFE`, `B2_PLACE`). All are
   inside the sim's joint limits and land on the October board. Joined to REST through the sim's `REST_OUT`, they make
   one Part B route (`B2_GRIP_TUNE`) that the physics flies cleanly, provided the record's **`must_be_empty` cells**
   (r1c2, r1c3, r2c3, r3c2) are empty. Its record still fails on the target square (section 2.5).
2. **Not usable as taught:** CalibrateBoard's B2/B3 chains. 12 poses have wrist roll of 45.2–60.8° (the URDF and sim
   allow ±45°). 2 have shoulder roll of 31.6–32.9° (the sim allows +20.3°).
3. **Not usable at all:** the straight line REST ↔ TRANSPORT. In physics it throws a soda can on r1c3 1.13 m.
4. **The sim does not reproduce Siva's grasps** at his joints, so it can prove the route but not the grip.
5. **Siva already taught a staging spot** on the board's right margin, beside r2c3.

## 1. Scope and principles

- **The first arm motion from the workstation, for two purposes.** Gripper calibration (#27 Part B) comes first. T3 move
  execution (#49 Phase 2) is second and needs its own owner decision. Right arm only. The left arm is a phantom in the
  sim and absent on the robot.
- **The checks can only refuse, never authorise.** This is the perceived-scene note's rule (§0, §3.2), applied to
  motion:

  ```
  arm_move_allowed = operator_confirms_this_leg AND every_check_passes
  ```

  - No record, check or perception result makes a move happen. Each can only stop one.
  - A passing sim record is necessary, never sufficient.
  - A board read that says "empty" does not stand in for the operator. A read that says "occupied" or "unknown"
    refuses.
- **The sim first, then the robot.** A route reaches the robot only after an offline clearance record and a live
  rehearsal in the sim through the viewer (section 10).
- **The hardware e-stop and an operator at the robot are mandatory** for every live arm move. Vision (the Phase 3
  freeze) is only ever an extra.
- **Live needs three things, plus the gate.** A leg is live only with:
  - `REACHY_ENABLE_MOTION=true`;
  - the viewer started with a new **`--allow-arm-motion`**;
  - a **fresh confirm tick for this one leg**.

  It also has to pass `safety.arm_gate_check`. Anything else is a logged dry run, as for the head and the gripper.
  Code defaults stay dry-run.
- **Never generated.** No LLM, planner or IK call produces a joint goal at run time.
  - Routes are joint waypoints in a data file, and every waypoint has a provenance.
  - The SDK's head IK is still used for the gaze. That is the existing gated neck path (section 6).
- **REST is the only motors-off pose.** The arm is powered at REST and released at REST, and nowhere else. A Stop or
  an abort holds the arm in place, stiff. It never goes limp mid-air (the #61 and #68 hazard).

## 2. Step 1 evidence: Siva's poses in the October model

**Method** (`arm-motion-step1/README.md`).
- The sim's MJCF with `FWDCenterLabSivaPool`: the October mount, the board as one 25 mm slab (`table_top`, top at
  z 0.740), five rig rails under its near edge, and the phantom left arm, which doesn't collide but is measured.
- **Poses:** every joint literal in `MovePieceLogic.ipynb` (10) and `CalibrateBoard.ipynb` (46): 36 distinct arm poses,
  plus the 9 `BOARD_POSES`.
- **The tip is the model's `r_tip`.** It reproduces #185's check (`B2_SAFE_JOINTS` → `BOARD_POSES["B2"]` + 0.15 m)
  to 0.00 mm and 0.00°, so it is the SDK's `forward_kinematics()` point.
- **Segments** are the joint-space straight line an SDK `goto` follows, kinematic, with 41 samples.
- **Physics:** the motors on, minimum-jerk goals at 500 Hz, the notebook's durations.
- **Board clearances are analytic.** `mj_geomDistance` returns a spurious 0 for the hand's boxes against the slab
  (`board_check.txt`).

### 2.1 The MovePieceLogic route (the one #27 names), and REST's routes

| Pose | Sim limits | Tip, torso frame (m) | Tip over board | Lands | Board (cm) | Rig, left arm | Era |
|---|---|---|---|---|---|---|---|
| PICK_SAFE | in | (0.284, −0.257, −0.287) | 12.3 cm | right margin, beside r2c3 | +7.4 | ≥ 9, ≥ 9 | Oct |
| PICK | in | (0.275, −0.252, −0.361) | 4.9 cm | right margin, beside r2c3 | **+2.2** (palm) | ≥ 9, ≥ 9 | Oct |
| TRANSPORT | in | (0.269, −0.023, −0.175) | 23.5 cm | over r2c2 (+43, −23 mm) | ≥ 9 | ≥ 9, ≥ 9 | Oct |
| B2_SAFE | in | (0.240, +0.023, −0.198) | 21.2 cm | over r2c2 (+14, +23 mm) | ≥ 9 | ≥ 9, ≥ 9 | Oct |
| B2_PLACE | in | (0.231, −0.005, −0.375) | 3.5 cm | **r2c2 (+5, −5 mm)** | **+0.2** (forearm) | ≥ 9, ≥ 9 | Oct |
| REST (sim) | in (roll +15)* | (0.141, −0.396, −0.397) | 1.3 cm | right margin, off the grid | rests on it, by design | ≥ 9, ≥ 9 | Oct model |
| REST_OUT, PRESENT_SHUT, PRESENT | in | — | 26–65 cm | out to the side | ≥ 9 | ≥ 9, ≥ 9 | Oct model |

\* above the URDF's +10°, inside the sim's +20.3° (ADR-0010 §4).

- **Era: all October.** For every MovePieceLogic and CalibrateBoard pose, the nearest October cell is closer than any
  cell of the August boards (`replay.txt`, ERA).
  - The August boards are #185's camera solve, centre (0.467, 0.015) m, and Siva's late-August board, (0.47, 0.10) m.
  - Example: B2_PLACE lands 7 mm from the October r2c2, against 86 mm and 99 mm from the August cells.
  - So both notebooks were taught on the board the October model holds.
- **Segments of the route are all clear of the board, rig and left arm.** PICK_SAFE → PICK → PICK_SAFE → TRANSPORT →
  B2_SAFE → B2_PLACE → B2_SAFE → TRANSPORT. The lowest point is B2_PLACE's forearm, 2 mm over the board.
- **Physics** follows within 1.4° and arrives within 0.6°.
  - The forearm touches the board at B2_PLACE and on the way back up: a 0.5° elbow sag against 2 mm of clearance.
  - Siva flew this pose with a cube. It is a declared contact, not a surprise (section 11, OD-A4).
- **LIFT_TO_PRESENT and LOWER_TO_REST** reproduce ADR-0010.
  - Every leg is ≥ 9 cm from the board, rig and left arm once the hand leaves REST.
  - Physics arrives within 1.0°, follows within 3.8°, and touches the board only at REST.
- **The phantom left arm** is ≥ 9 cm from the right arm everywhere except near r1c1: 5.0–5.5 cm at the C1 board pose.
  **The rig** is ≥ 8.5 cm everywhere.
- **`self`** (forearm against the torso box) reads down to −3.6 cm on poses Siva flew on the robot. The model doesn't
  collide the arm with the torso, and the box is larger than the real chest there. Reported, not judged.

### 2.2 CalibrateBoard: the B2/B3 chains are outside the sim's limits

| Group | Poses | Out of range |
|---|---|---|
| Inside the sim's limits | HOME_POSITION, RESET_POSITION_RIGHT (cell 38), B2_PICK, B2_LIFT_1CM, every SUPPLY_*, TRANSPORT_HIGH, MIDPOINT_WITH_CUBE, SUPPLY_SAFE (cell 94) | — |
| Wrist roll past ±45° | B2_APPROACH, B2_SAFE_ALIGNED (= B2_SAFE_WITH_CUBE), B2_LIFT_4CM, B3_SAFE_JOINTS (= B3_RETREAT), B3_LOWER_3CM, B2_SAFE (cell 118), DOWN_2CM, DOWN_NEXT_2CM, B2_PLACE (cell 128), UP_2CM, UP_6CM, B2_EXIT | +45.2 to **+60.8°** |
| Shoulder roll past +20.3° | B2_TRANSPORT, B2_HIGH_WITH_CUBE | +31.6, **+32.9°** |
| On the sim's stop | B2_SAFE_JOINTS | +20.28° (the pose ADR-0010 took the stop from) |

- **The real robot goes past the URDF in two joints.** It reached wrist roll 60.8° and shoulder roll 32.9°.
- **In physics the sim clamps these.** The wrist roll arrives 9–16° short and the shoulder roll 12–13° short.
- **So these chains cannot be cleared in the sim as taught.** That would take another SDK-visible sim deviation, or a
  re-teach inside the limits (OD-A6).
- **The rest of the geometry is fine.** Every CalibrateBoard leg clears the board by 1.7 cm or more (analytic,
  `board_check.txt`), and the rig and left arm by ≥ 9 cm.

### 2.3 The taught board poses (`BOARD_POSES`, tip poses, not joints)

- **All nine solve in the model** to 0.0 mm and 0.0° (an IK on `r_tip`, position and orientation, within the sim's
  limits, seeded from Siva's joints).
  - The shoulder roll needed runs from −22.6° to +17.6°.
  - As taught, the tip is 2.8–4.0 cm over the board and the hand 0.0–0.6 cm over it.
- **The tips land within 50 mm of the October cell centres:**

  | Square (sim cell) | Offset from the cell centre (mm) |
  |---|---|
  | A1 (r3c1) | (−36, −34) |
  | A2 (r3c2) | (−29, +17) |
  | A3 (r3c3) | (−22, +23) |
  | B1 (r2c1) | (+5, −7) |
  | B2 (r2c2) | (+17, +6) |
  | B3 (r2c3) | (+7, +11) |
  | C1 (r1c1) | (+30, −4) |
  | C2 (r1c2) | (+27, −7) |
  | C3 (r1c3) | (+4, −3) |

  This is #185's compressed taught spacing (0.84× / 0.92×).
- **These IK solutions are evidence, not route data.** They show that the taught grid fits the October model. A
  robot route needs taught joints (section 3.1).

### 2.4 Entering and leaving REST, with pieces on the board

| Leg (gripper shut) | Tall pieces on column 3 (can, blue cylinder, red cube) | Lab pieces on column 3 |
|---|---|---|
| REST → TRANSPORT, straight | **forearm −1.9 cm into the can**; physics: can moved 4 mm | +2.2 cm |
| TRANSPORT → REST, straight | −1.9 cm; physics: **can thrown 1134 mm** | +2.2 cm |
| REST → REST_OUT → TRANSPORT, and back | +0.7 cm; physics: nothing moved | +3.3 cm; nothing moved |
| REST → HOME (Siva's) → TRANSPORT | +0.7 cm; physics: nothing moved | +3.3 cm; nothing moved |

- **The closest approach on the good legs is at TRANSPORT itself:** the forearm over r1c3.
- **So a route between REST and the board goes out to the side first (REST_OUT), as ADR-0010's routes do.** Siva's
  HOME works too, but REST_OUT is already a waypoint of two validated sim routes.

### 2.5 The candidate Part B route, end to end

`B2_GRIP_TUNE` = REST → REST_OUT → TRANSPORT → B2_SAFE → B2_PLACE → B2_SAFE → TRANSPORT → REST_OUT → REST.
- The gripper is shut over the board, and opens to −29° only for B2_SAFE ↔ B2_PLACE.
- The durations are set by a 10°/s average cap (section 4.2).

| Layout | Physics result |
|---|---|
| Empty board | arrives ≤ 0.7°, follows ≤ 1.7°; board contacts only at REST and B2_PLACE (forearm) |
| Lab pieces on r2c3, r3c3, r2c1, r3c1 (r1c3 empty) | **nothing moved (0.0 mm)**, the same contacts |
| A lab cylinder on r1c3 | the forearm hits it during the B2_PLACE descent (moved 0.4 mm) |
| Tall pieces on column 3 | the forearm hits the can on r1c3: **thrown 1864 mm** |

- **The B2 descent's forearm sweeps r1c3.** Kinematically it reaches 3.5 cm into a lab cylinder there and 5.5 cm into
  the can.
- **So `B2_GRIP_TUNE` carries an occupancy constraint: the record's `must_be_empty` cells.** That is how the
  multi-object question enters every route (section 8.3). The clearance record (A0b, `tools/route_clearance`) puts
  each lab piece at its worst position in each cell, toward the route, not at the centre as above:
  - **`must_be_empty`: r1c2, r1c3, r2c3, r3c2**, the cells around B2. At their edge facing B2 the hand or forearm
    reaches into a piece: a cube on r2c3 is moved 21 mm, one on r1c2 7 mm. Pieces centred on r2c3 and r3c3 (the
    layout above) do not move; at their worst position the r2c3 one does.
  - **`may_be_occupied`: r1c1, r2c1, r3c1, r3c3.** Flown all four at once: nothing moved.
  - **r2c2 is the target square,** outside the map.
- **The target piece's placement tolerance.** The record also gives `target_placement_tolerance_mm` per lab piece:
  the largest offset from the centre of r2c2, in 1 mm steps toward the approach (+y, column 1's side), at which the
  approach to B2_PLACE moves the piece < 1 mm before the pads close. **In the sim there is none, for either piece:**
  the palm lands on a centred cube (moved 2.5 mm) and a centred cylinder (4.2 mm) on the B2_PLACE descent, as in
  section 2.6. So the `B2_GRIP_TUNE` record's verdict is `fail`, with that as its only reason. Siva placed real
  cubes at this pose, so this is the sim's grasp gap (section 2.6, OD-A13); until it is resolved or the owner
  decides otherwise, `B2_GRIP_TUNE` has no passing record.

**B2 is a placing pose; PICK is the grasp.** Siva's notebook (`tests/MovePieceLogic.ipynb`) opens the gripper to −29°,
moves PICK_SAFE → PICK, and closes to −10° **at PICK**. He carries the piece to B2_SAFE → B2_PLACE and opens to −29°
there: at B2_PLACE the piece arrives already held. So `B2_GRIP_TUNE` has the target piece placed at a pose where the
real sequence never closes on a loose piece, and its failure is partly a question about the route.
- **`PICK_GRIP_TUNE`** = REST → REST_OUT → PICK_SAFE → PICK → PICK_SAFE → REST_OUT → REST, closing at PICK.
  - The gripper opens to −29° on the REST_OUT → PICK_SAFE leg, as in the notebook. The legs are 4.0, 8.77 and 5.0 s
    (10°/s rule), PICK is `near_board`. The direct REST_OUT → PICK_SAFE leg is clean: board ≥ 7.4 cm, rig and left
    arm ≥ 9 cm, following ≤ 0.9°, arrival ≤ 0.7°. B2_GRIP_TUNE is unchanged.
  - **The target is not a grid cell.** PICK's tip is at world (0.481, −0.252) m, 4.9 cm over the board, 36 mm outside
    r2c3's outer edge. The clearance tool's `--target-spot` flies the target piece there, and the record has
    `target_cell: null` and `target_placement.spot_xy_m`. All nine cells are in the occupancy map.
  - **Occupancy: `must_be_empty` r1c3 and r2c3**; `may_be_occupied` r1c1, r1c2, r2c1, r2c2, r3c1, r3c2, r3c3. The
    forearm reaches into r1c3 on the PICK descent (a cylinder moved 21 mm, a cube 2.0 mm) and touches a cube on r2c3
    (1.5 mm). The seven may-be-occupied cells flown at once: nothing moved.
  - **Head at PICK:** neck pitch 57.0°, yaw −25.8°, roll 0, at no limit, the tip in view of both cameras (at PICK_SAFE
    52.4°, −29.0°).
  - **Verdict `fail`, only on the target spot:** a centred cube moves 2.0 mm and a centred cylinder 5.0 mm, from the
    palm on the PICK_SAFE → PICK descent, so the placement tolerance is none.
- **Both routes fail the target check for the same reason: the sim's hand geometry** (reachy-1-2-sim #210, which needs
  Siva's measurements), not the routes. Siva grasped real pieces at both poses. Both stay `fail` until #210; neither
  has a passing record.

### 2.6 The sim does not reproduce the taught grasps

The test: a 50.8 mm lab cube placed at each pose's grasp point (the midpoint of the two pads at −29°, turned to the
jaw), then a physics close to −10° or −7° and a lift (`replay3.txt`).

| Pose | Pad midpoint over the board | Result in the sim |
|---|---|---|
| MPL PICK (right margin) | 47 mm | the close stops short on the cube (−11.5°: "holding"); **the lift leaves it behind** |
| CB SUPPLY_PICK (right margin) | 60 mm | the same: the close stops at −10.3°; the lift slips |
| MPL B2_PLACE (r2c2) | 35 mm | the palm lands on the cube top (12 mm overlap); no grip |
| CB B2_PICK (r2c2) | 80 mm | the pads end at the cube's top; the close runs to −10.0° on nothing |

- **Siva lifted real cubes at these joints**, so the sim's pads sit higher relative to the real fingers. The candidate
  causes are the finger reach below `right_tip` and the pieces used on 2026-10-13. This is unresolved and becomes a
  sim follow-up (OD-A13).
- **Consequences.** The sim can prove a route's clearance and a procedure's logic. It cannot prove that a grip
  holds, its force (the force sensor reads 0, per #27's Part A run), or slip. Those are robot measurements.
- **The pad gap at −29° is 69 mm between pad centres.** That is about 53 mm between the faces, so a 50.8 mm cube
  has about 1 mm a side. The 60 mm red cube needs about −45° (sim numbers; Part A measures the real range).

### 2.7 Siva's supply spot

`PICK` and `SUPPLY_*` land on the board's **right margin, 3–6 cm outside the outer tape, beside r2c3.** The grasp
points are at (0.279, −0.269) and (0.216, −0.297) m in the torso frame: taught twice, 7 cm apart.
- They are inside the sim's limits and reached on the robot.
- LIFT_TO_PRESENT and LOWER_TO_REST pass a cube standing there at ≥ 5.7 cm.
- This is a **staging area that needs no reach study** (section 8.1).

## 3. Named routes as data

### 3.1 The route file

The file is `src/reachy_ai/motion/arm_routes.json` (package data). It is loaded and validated by
`motion/arm_routes.py`, and it holds strict JSON: unknown fields fail.

```json
{"schema": "arm_routes.v1",
 "poses": {"TRANSPORT": {"joints_deg": {"r_shoulder_pitch": 3.56, "...": 0},
                         "provenance": {"kind": "taught", "source": "tests/MovePieceLogic.ipynb", "cell": 2,
                                        "sha256": "a4ae14b3…", "taught_utc": "2026-10-13", "by": "Siva"}}},
 "routes": {"B2_GRIP_TUNE": {
   "purpose": "gripper_calibration",
   "start": "REST", "end": "REST",
   "legs": [{"to": "REST_OUT", "gripper_deg": 20.0, "min_duration_s": 4.0},
            {"to": "TRANSPORT", "gripper_deg": 20.0, "min_duration_s": 8.4},
            {"to": "B2_SAFE", "gripper_deg": -29.0, "min_duration_s": 6.0},
            {"to": "B2_PLACE", "gripper_deg": -29.0, "min_duration_s": 5.0, "near_board": true}],
   "close_pose": "B2_PLACE", "return": "reverse"}}}
```

- **Provenance kinds:**
  - `taught`: a robot read-back, with notebook/cell/SHA or the teach procedure's bundle;
  - `sim`: a reviewed sim pose with its evidence, such as REST and REST_OUT from ADR-0010.

  A route that contains a `sim` pose may run on the robot only after that pose has been taught on the robot and the
  route re-cleared (OD-A5).
- **`return: reverse`.** A route's way back is its own legs in reverse, so the return is cleared by the same record.
- **The route hash.** SHA-256 of the canonical JSON (sorted keys, angles rounded to 0.01°) of the route and every pose
  it names, plus `arm_routes.py`'s limits table. Any change to a pose, a leg, a duration or a limit gives a new hash.
- **Teaching is read-only.** A *Teach arm pose* viewer procedure reads the 7 joints, the gripper and `forward_kinematics()`
  while Siva moves the arm by hand.
  - The arm starts compliant at REST and ends compliant at REST.
  - The procedure sends nothing and writes a bundle with the read-backs.
  - It joins `probe_sdk1.py` and `smoke_test_all.py` under `test_probe_read_only.py`.

### 3.2 The sim clearance record

A record is produced in the sim repo by `tools/route_clearance/` (section 10) and committed here as
`src/reachy_ai/motion/arm_route_records/<route>@<hash12>.json`:

The format is `arm_route_record.v1`. `motion/arm_records.py` loads it strictly (every object has exactly its
fields) and checks it against itself (below).

```
schema: "arm_route_record.v1",
route, route_hash, verdict ("pass" | "fail"), reasons: [...]       # empty exactly when the verdict is "pass"
rig: {rig_layout_id: "fwd-center-lab", mount_id: "october-2026", scene_name, scene_sha256, scene_revision}
     (perceived-scene §1.2 RigIdentity fields, source = "sim"; the ids are config.ARM_RIG_LAYOUT_ID / ARM_MOUNT_ID),
sim: {commit, model_sha256, mujoco_version, tool_sha256},
routes_sha256,                                                     # the route document the tool read
legs: [{from, to, direction ("out" | "return"), duration_s, gripper_deg,
        kinematic: {board_cm, board_geom, board_t_s, rig_cm, left_arm_cm, self_cm, zones},
        physics: {follow_deg, arrival_deg, arrival_weak_deg,
                  contacts: [{pair, t_s, steps, at, declared, deviation_deg: {pose: deg}}]}}],
physics_settle: {at, judged, contacts},                            # the 1 s settle at the start pose, judged
declared_contacts: [{pose, pair, condition: null | {max_joint_deviation_deg}}],
occupancy: {cell: "may_be_occupied" | "must_be_empty"},            # every cell except the target square
target_cell,                                                       # the cell under close_pose, or null
target_placement_tolerance_mm: {piece: mm | null},
target_placement: {close_pose, approach_xy, pieces: {piece: {body, steps: [{offset_mm, moved_mm}],
                                                             centred_piece_contacts, tolerance_mm}}},
layouts_flown: [...],
head: {waypoint: {neck_deg, at_limit, off_axis_deg, tip_in_view_left, tip_in_view_right, tip_px_left, tip_px_right}},
method: {...}                                                      # the tool's constants, so a reader can check a rule
```

**Declared contacts** for `B2_GRIP_TUNE` (owner decision, 2026-10-22):
- `(REST, r_forearm_col/table_top)` and `(B2_PLACE, r_forearm_col/table_top)`, **unconditional**. Each covers its pair
  on a leg that starts or ends at that pose, while the arm is within the pose's near-board zone (the run of samples
  under 1 cm from that end, plus 0.25 s), or in the hold at the pose.
- `(REST, r_palm_col/table_top)` and `(REST, r_hand_back_col/table_top)`, **conditional**:
  `{max_joint_deviation_deg: 1.0}`. Each is allowed only while the arm's largest joint deviation from REST is ≤ 1°,
  on the way out and on the way back. Outside that, the contact fails. (Leaving REST, the palm and hand-back graze
  the board for about 0.2 s at ≤ 0.15° from REST.)

Each contact in the record carries `deviation_deg`, the arm's largest joint deviation from the leg's end poses over
the contact. The loader refuses a contact marked `declared` that no declaration covers, or that is outside its
declaration's bound.

**"Pass", proposed defaults:**
- every waypoint is inside the gate's limits;
- every leg's board, rig and left-arm distance is ≥ 0, and ≥ 1 cm except at `near_board` waypoints. The board is not
  judged inside an unconditionally declared pose's zone;
- physics arrival is ≤ 3° on the guard joints, following ≤ 6°, and no contact other than the declared ones, in the
  legs and in the settle;
- the target piece is not disturbed when centred (below).

**The occupancy map.** A cell is `may_be_occupied` only if a lab piece on that cell, at its worst position in the cell
(toward the route), stays ≥ 2 cm from the arm kinematically and moves < 1 mm in physics (its largest displacement in
the flight). Every other cell is `must_be_empty`. Every `may_be_occupied` cell is then flown at once, and must touch
and move nothing.

**The target square.** The cell under the route's `close_pose` (r2c2 for `B2_GRIP_TUNE`) is not in the occupancy map:
the procedure puts the target piece there. Instead, each lab piece is flown at its intended placement: centred on the
cell, then offset in 1 mm steps toward the approach (the side the hand comes in from), along the outbound legs to the
close pose and its hold, before the pads close. `target_placement_tolerance_mm` is the largest offset at which every
step moved the piece < 1 mm. If a centred piece is already disturbed, its tolerance is `null` and the verdict fails
with that reason.

**Distances.** The board is analytic (Step 1's `board_check`). Rig, left arm, self and pieces are exact for box and
capsule pairs: `mj_geomDistance` reads two boxes as 0 whenever they are closer than its distmax, so Step 1's guard
could read a box pair too close, or too far. The flights start each leg from the measured position (section 4.2).

**The robot plays a route only with a record that:**
- matches the route's current hash;
- matches the rig's `rig_layout_id` and `mount_id`;
- has `verdict: pass`.

A changed route has no record until it is re-flown.

### 3.3 Adding or changing a route

1. Siva teaches the poses (the read-only procedure), or they come from his notebooks with cell and SHA.
2. A PR edits `arm_routes.json`. CI then:
   - validates the file;
   - computes the hashes;
   - fails if a committed record names a hash that no route has (a stale record);
   - lists the routes with no passing record as "not playable".
3. Someone runs `tools/route_clearance` in the sim and commits the record in the same PR.
4. The route gets a live rehearsal in the sim through the viewer (section 10), and its bundle is attached to the PR.
5. The owner reviews the PR. After merge, the first robot run of a new or changed route uses half the speed cap and
   the empty-square rehearsal (section 7.2).

## 4. The route player (`motion/primitives.py`, in the SDK child process)

### 4.1 Where it runs

`stereo._sdk_worker` gains an `arm_commands` thread with its own queue (`ARM_QUEUE = 1`) and halt event, like the
head and the gripper. It is **the only thread that powers or commands the seven arm joints.**
- **One motion lock in the child** serialises the head, gripper and arm threads. A leg holds it from the gate to the
  read-back, so nothing else moves while the arm does.
- **Stop is an event, so it reaches a running leg.** Stop arm also sets the head and gripper halts.

### 4.2 Primitives (all logged, each returning a record like the head's and gripper's)

| Function | What it does |
|---|---|
| `arm_read(robot)` | The 7 joints and the gripper: position, goal, compliance, torque limit, temperature, load. Plus `forward_kinematics()`. Read-only. |
| `arm_power_at_rest(robot, route, …)` | Only if the read-back is within `ARM_REST_TOL_DEG` of the route's REST. Per joint: goal = present, then `compliant = False`. Never `turn_on('r_arm')` or `turn_on('reachy')`. |
| `play_leg(robot, route, record, leg, …)` | The gate (section 5); then minimum-jerk goals at `STREAM_HZ` (50) from the present read-back to the leg's pose; then a read-back. |
| `hold_arm(robot)` | Stop: goal = present on all 7 joints. Stays powered. Needs no confirm. |
| `arm_release_at_rest(robot, route)` | Only at REST (read-back within tolerance, last leg `reached` at REST): `turn_off_smoothly('r_arm')`. Refused anywhere else. |

**Limits on a leg:**
- **Speed.** Duration = max(the leg's `min_duration_s`, largest joint change ÷ `ARM_MAX_AVG_SPEED_DEG_S`).
  - Proposed: 10°/s average, so a minimum-jerk peak of 18.75°/s.
  - The first robot run of each route uses 5°/s.
  - Siva's MovePieceLogic legs ran up to 7.7°/s average; the sim's REST legs run at 13–14°/s.
- **Step.** No streamed goal may move a joint more than `ARM_MAX_GOAL_STEP_DEG` (0.5°) from the previous one. This is
  asserted, as a backstop to the speed computation.
- **Start.** The present read-back must be within `ARM_START_TOL_DEG` (3°) of the leg's start pose. A leg is never
  flown from somewhere else.

### 4.3 Outcomes, and what the arm does on each

| Outcome | Trigger (proposed defaults) | The arm afterwards |
|---|---|---|
| `reached` | read-back within 3° on the critical joints (the four gross joints and wrist pitch, the sim's `CRITICAL_JOINTS`); forearm yaw and wrist roll within 8°, reported | held at the pose, powered |
| `following` | \|present − streamed goal\| > 6° on a gross joint for 0.25 s (blocked, or sagging) | **hold** (goal = present stops the push) |
| `load` | \|present_load\| over `ARM_MAX_LOAD`, any joint | **hold** |
| `temperature` | any arm joint over `ARM_MAX_TEMP_C` (55 °C, as the gripper) | **hold**; only the retreat to REST is then allowed |
| `readback` | arrived, but outside the read-back tolerance | **hold**; forward legs refused |
| `timeout` | 2 × the leg's duration plus 3 s | **hold** |
| `stop` | Stop arm | **hold** |
| `error` | the SDK raised, or a position is unreadable | **hold** if a command can still be sent; otherwise the Dynamixels keep the last goal (stiff) |
| `refused` / `dry-run` | the gate, or the live flags | unchanged; nothing sent |

- **After any hold, the only motion offered is the retreat.** That is the flown legs in reverse back to REST: each leg
  confirmed, at 5°/s, through the same gate.
- **Nothing ever powers the arm off mid-air.**
- **Limits on what software can do:**
  - If the viewer dies, the robot holds its last goal, stiff.
  - **The hardware e-stop removes power, and the arm then falls.** To confirm with Siva (OD-A14).
  - So the operator's hand stays near the arm on the robot's right. Stop is for everything except "about to hit
    someone".

## 5. The arm gate (`safety.arm_gate_check`)

It sits next to `head_gate_check` and `gripper_gate_check`, takes plain values and returns a `GateResult`. It refuses
when any of these holds:

1. **Route not allowed.** The route is not in the file, or its `purpose` is not one the owner decisions cover:
   `gripper_calibration` after Part B's decision, `t3_move` only after T3's.
2. **Record invalid.** No record matches the route's hash; its verdict is not `pass`; or its `rig_layout_id` or
   `mount_id` differs from the rig's.
3. **Limits.**
   - A waypoint is outside `ARM_LIMITS_DEG`: the URDF, plus the sim's deliberate r_shoulder_roll +20.3°, and nothing
     wider (OD-A6).
   - The speed or step limits (section 4.2) are exceeded.
   - The start read-back is off the leg's start.
   - A joint's temperature is over the limit before the leg starts.
4. **Approval.** No fresh confirm for this leg.
5. **Motion disabled.** `REACHY_ENABLE_MOTION` is not true. That gives a dry run, not a refusal, as with
   `MOTION_DISABLED` today.
6. **Something else running.** Any other command holds the motion lock, or a head, gripper or arm Stop or abort is
   unresolved.
7. **The board, where the route touches it** (any leg with a board distance under 9 cm in the record). There must be
   all of:
   - a **rig check** this session (OD-A8);
   - every `must_be_empty` cell confirmed empty by the operator;
   - no fresh board read showing that cell `occupied` or `unknown`. A read can only refuse.
8. **Arm FK check** (once per connection, like `neck_sign_check`). `forward_kinematics()` of the REST read-back must
   put the tip within 3 cm of the record's REST tip. Otherwise the joint signs, the frame or the mount are not the
   twin's.
9. **Hand-freeze hook.** `hand_in_view: bool | None`. Today it is `None`, which means not checked; it does not mean
   allowed. Phase 3 sets it, `True` refuses, and the player polls a `freeze_event` mid-leg to hold.

## 6. The head follows the gripper

**Once per leg, after `reached` and before the next confirm, never during a leg:**
1. Read `forward_kinematics()` (a read).
2. Clamp the tip into `HEAD_LOOK_AT_*`.
3. Call `look_at_point(tip)`: the SDK head IK (`head._look_at`), then `head_gate_check`, then the streamed neck. On the
   robot this is the existing gated path.

**In the sim, the SDK's head IK is a stub** (`FakeHeadKinematicsService` turns a quaternion into roll, pitch and yaw,
which is why the viewer reports `neck_sign_check: skipped (sim)`). So the sim uses the record's `head` neck angles
instead:
- they are computed in `tools/route_clearance` with the **model's own cameras**, the method Part A used for
  `gripper_view`;
- they come with whether the tip is in both frames;
- **not with `gaze.project`**, whose yaw order is wrong away from yaw 0 (sim #208).

On the robot, a difference of more than 10° between the SDK IK and the record's angles is logged, not gated. Where
the record says the hand can't be seen inside the neck limits (pitch ≤ 64°, yaw ±45°), the head holds its last pose.

## 7. Gripper Part B: the experiment

### 7.1 What it measures

| Item | Plan |
|---|---|
| Pieces | lab cube 50.8 mm, lab cylinder 45.7 × 71.1 mm, red cube 60 mm (if present) |
| Square | B2 (r2c2), along `B2_GRIP_TUNE` (section 2.5); r1c3 and the rest of the board empty |
| Open angle | −29° (Siva's) for the cylinder; for the cubes the widest of −29 / −45° that Part A shows the real gripper reaches. The sim's −29° leaves about 1 mm a side on the 50.8 mm cube. |
| Close series | torque limits 10, 15, 20, 25, 30 % (Part A's cap), three repeats each, 10°/s toward the shut stop |
| Each close | ends `holding` when the position stalls short of the shut stop: no 0.5° in 1 s, the Part A rule. Then a 2 s hold, sampled at 20 Hz, then a gated open to the open angle. |
| Result | force against closing angle per piece: the stall angle, `present_load` and the force sensor at the stall plateau, per torque level |
| For #182 stage 3 | "closed short = holding": each piece's stall angle against Part A's empty shut stop |

- **A new primitive, `gripper_close_force_limited`.** The same primitive as #5's `close_gripper_until_force`; one, not
  two. It differs from `gripper_step` in two ways:
  - a stall on a piece is the measurement (`holding`, the gripper stays powered), not an `end_stop` release;
  - the force rule is a ceiling (`FORCE_MAX_SAFE`, 15 by default, units still unconfirmed) rather than the empty
    sweep's change of 3.0.

  The temperature and load aborts are Part A's.
- **The optional slip check needs a taught short lift.** A close at the chosen torque, then B2_PLACE →
  `B2_LIFT_2CM` → hold 3 s with a stereo pair → back down. The slip is the stall-angle change plus the piece in view.
  `B2_LIFT_2CM` doesn't exist yet: Siva teaches it, as he taught `B2_LIFT_1CM` for CalibrateBoard (OD-A11). Until then,
  no lift.

### 7.2 The procedure (in the viewer, every move confirmed)

1. **Stage 0, read-only.** `arm_read` and `gripper_read`; the force zero; the arm FK check. The operator confirms
   that the board is empty and that r1c3 is empty.
2. **Rehearsal, no piece.**
   - The route to B2_PLACE with the gripper open, at 5°/s; the head follows.
   - The operator checks the hand's height and position over B2. A stereo pair and FK are logged.
   - The route back to REST, then release.
3. **Place the piece** on B2 at the grasp mark, with the arm compliant at REST. Hands are never under a powered arm.
4. **Run.** Power at REST; the route to B2_PLACE; the close series (and the lift if taught); open; the route back to
   REST; release.
5. **Any abort** holds the arm, offers only the retreat, and ends the run's forward moves, as in Part A.

### 7.3 What is recorded

The Part A bundle, plus the arm.
- **`samples.csv` gains columns:** `piece_id`, `phase` (open, close, hold) and `arm_pose`.
- **The run record gains keys:**
  - `arm`: the route, hash, record id, and each leg's outcome and read-back;
  - `pieces`;
  - `rehearsal`.
- **This changes the format's rules, so `format_version` becomes 2** (the doc's own bump rule).
  - Version 1 captures stay readable.
  - Sim #179's analyser must accept version 2 before the first robot run.

### 7.4 What the sim run can and cannot prove

| Can prove (sim rehearsal through the viewer) | Cannot prove (robot only) |
|---|---|
| the route's clearance, and the record matching the route | **the grip**: the taught grasps don't hold a 50.8 mm cube in the sim (section 2.6) |
| every gate refusal, the dry run, the confirms, Stop → hold, the retreat | **force**: the sim's force sensor reads 0 |
| the stall logic: Step 1's closes to −10° and −7° stopped 0.3–1.7° short on the cube (a close toward the shut stop was not flown) | **slip**, and the real stall angles and loads (the sim's gripper numbers are synthetic, #179) |
| the head-follow angles, and the bundle in format version 2 | **e-stop** behaviour |

## 8. T3 Phase 2

### 8.1 Hand-over at PRESENT, or a staging area

| | Hand-over at PRESENT | Staging at Siva's supply spot |
|---|---|---|
| Where the piece comes from | a person puts it in the open jaw, 64 cm over the board, out to the right | the board's right margin beside r2c3, where Siva already picks from (section 2.7) |
| Reach study | none | **none: taught, and reached on the robot** |
| Motion needed | TRANSPORT ↔ PRESENT (clear; physics: no contact), then the cell route | a supply route (PICK_SAFE ↔ PICK), then the cell route |
| What it adds | **a person's hand at a powered, closing gripper**: the "hands clear" rule every gate relies on, and the Phase 3 freeze (a hand is always in view), both break. Needs its own owner decision. | a pick next to column 3: PICK's forearm passes 1.7 cm from a lab cylinder on r1c3, so the pick has its own `must_be_empty`; a check that the right piece is staged (operator, or a read) |
| Proven in the sim | the motion; the hand-over itself can't be simulated | the motion; **not the pick**: the taught supply grasps slip in the sim (section 2.6) |

**Recommendation: staging at Siva's taught supply spot.** Its main cost, the reach study, has gone: Siva taught the
spot and reached it. It keeps every hand clear of a moving or closing gripper. It is the flow he already ran
(`MovePieceLogic`: the supply pick, TRANSPORT, the B2 place).
- Hand-over stays the fallback, if the real supply pick can't be made to hold.
- The pick is proven on the robot in a Part B-style rehearsal, because the sim can't prove it.

### 8.2 Placing on a cell

- **On the robot: taught per-cell poses.** For each cell, Siva teaches a `<cell>_SAFE` and a `<cell>_PLACE`, read-only.
  - B2 exists (MovePieceLogic).
  - B3's exists, but its wrist roll is 0.2–0.5° past the sim's ±45°.
  - The other seven are to teach.
  - Each cell route gets its own record, with its occupancy map.
  - The taught grid lands within 50 mm of the sim cells (section 2.3). With a 152 mm cell and a 51 mm piece, that is
    inside the cell.
- **In the sim: the sim's own planners** (`place_on_cell`, the crane and side-on, with their own checks) for the sim
  demo. Their joint goals are planned at run time, so they never become robot routes unless frozen into the route file
  as `sim` poses, taught on the robot, and re-cleared (OD-A12).

### 8.3 Lifting the two-or-more-objects refusal (perceived-scene §3.3)

This note makes the refusal per route, and measurable: the record's occupancy map (section 3.2).
1. **Clearance.** A cell is `may_be_occupied` only if a lab piece flown there at its worst position clears the route.
   The gate refuses when a `must_be_empty` cell is occupied, `unknown`, or not confirmed.
   - On the robot, cells are read as states, not positions (T3 1b). So the per-cell flights use the worst position in
     the cell, not perceived bounds.
2. **Sim agreement.** In the sim, the board read must agree with the scene. Any disagreement refuses.
3. **Flights.** The occupancy map *is* the reviewed multi-object flights, for that route.
4. **Owner sign-off,** per route `purpose`, through `_ability_available` in the sim and the route `purpose` here, off by
   default.

### 8.4 Checking the result by re-reading the board

After the arm is back at REST (out of the board view, Part A), the head goes to `board_view` plus the near-row pose,
and the T3 read runs.
- **The move counts as done only if:**
  - the target cell shows the piece's kind;
  - the supply spot is empty;
  - no other cell changed.
- **Anything else stops the run.** No further moves; the operator is told (#182 stage 3).

## 9. The owner decision text, and the operator rules

**Part B (proposed for the owner's approval):**
> "Arm motion is allowed in gripper calibration procedures: the right arm plays only named routes of Siva's taught
> joint poses that have a passing sim clearance record matching the route and the rig. Every leg needs its own confirm
> and passes the arm gate (joint, speed and step limits, the record, the board as the route requires, operator
> approval, `REACHY_ENABLE_MOTION` and `--allow-arm-motion`, nothing else moving). Stop holds the arm where it is. The
> arm is powered and released only at REST. Siva is at the robot with the hardware e-stop for every run."

**T3 (a separate decision, after Part B's robot runs):**
> "Arm motion is allowed for T3 moves the operator approves: the same rules as calibration, plus the route's occupancy
> constraints, met by the operator's board confirmation and never contradicted by a fresh board read, and a re-read
> after placing that must match before anything else moves. Perception can only refuse."

**Operator rules.**
- Siva at the robot, with a hand near the arm on the robot's right and the e-stop in reach.
- Stop arm on the page for anything unexpected; the e-stop for "about to hit someone".
- **Hands only near a powered arm at REST.** Pieces are placed while the arm is compliant at REST.
- The board is clear except what the procedure lists. The operator confirms the `must_be_empty` cells before each run.
- No notebook or other program connected to the robot.
- The first run of any new or changed route uses the half speed cap and the empty-square rehearsal.
- After any abort: retreat to REST, release, and report before a new run.

## 10. Sim and robot execution paths

```
                      arm_routes.json  ──hash──▶  record (sim)  ◀── tools/route_clearance (sim repo, offline MuJoCo)
                             │                        │
              ┌──────────────┴────────────┐           │
   viewer --source sim                 viewer --source robot
   stereo._sdk_worker                  stereo._sdk_worker            same child process, same threads, same lock
   primitives.play_leg ──▶ safety.arm_gate_check ◀── record + rig + confirm + flags
        │ reachy_sdk 0.7.0                 │ reachy_sdk 0.7.0
   fake server :50051 ─▶ native MuJoCo     robot :50051
```

- **Shared:** the route file, the hash, the record, `primitives`, the gate, the procedure and the bundle. The only
  differences between the two paths are the SDK's host, and the head angles (section 6).
- **Offline proof:** `tools/route_clearance` in the sim repo. It generalises this note's Step 1 scripts, reads the
  route JSON as a wire contract (the perceived-scene OD-1 pattern: both repos ship a `reachy_ai` package) and writes
  the record. The sim owns the model; the record is committed here, where the robot reads it.
- **Live sim rehearsal:** the viewer against the sim, through the fake server and the native physics, exactly the
  robot's code path. It is the second proof, and it exercises Stop, the aborts and the retreat.
- **The sim panel** (`rig_motion.travel`, `ROUTE_COMPATIBILITY`, `_ability_available`) keeps its own routes. It does not
  fly tabletop routes, and the two never authorise each other.

## 11. Build plan, exit gates, open decisions

| # | Repo | What (Sonnet builds) | Exit gate |
|---|---|---|---|
| A0 | tabletop-ai | `arm_routes.json` + `arm_routes.py`: schema, loader, hash; the MovePieceLogic poses + REST/REST_OUT; `B2_GRIP_TUNE` | strict validation; hash stable and change-sensitive; every pose has provenance; no motion code |
| A0b | sim | `tools/route_clearance`: kinematic + physics + occupancy + head angles → record | reproduces Step 1's numbers for `B2_GRIP_TUNE`; artefact-safe board distance; record for `B2_GRIP_TUNE` committed |
| A1 | tabletop-ai | `safety.arm_gate_check` | a refusal test per item of section 5; a monotonicity test (no record or flag makes a refused leg pass) |
| A2 | tabletop-ai | `arm_read`, `arm_power_at_rest`, `play_leg`, `hold_arm`, `arm_release_at_rest`, head follow; *Teach arm pose* (read-only) | fake-SDK tests: nothing written without the gate; Stop holds; compliant only at REST; each abort holds; teach writes nothing |
| A3 | tabletop-ai | `stereo.py` arm thread + motion lock + `--allow-arm-motion`; the *Gripper tuning (Part B)* procedure; `gripper_close_force_limited`; format v2 | dry-run end to end against the sim; format v2 test |
| A3b | sim | #179's analyser accepts format v2 | v1 and v2 accepted, unknown versions refused |
| S1 | sim | **live rehearsal** of Part B through the viewer | the full procedure; Stop mid-leg holds; an injected following error holds; the retreat; a valid bundle |
| D1 | owner | Part B decision text (section 9) | approved |
| R0 | robot | Stage 0 arm read; teach REST and REST_OUT (read-only); re-clear with the taught values | the arm FK check passes; the record still passes |
| R1 | robot | empty-square rehearsal at 5°/s | the read-backs within tolerance; no abort |
| R2 | robot | Part B with the pieces | the bundle to #27 and sim #179 |
| T1–T5 | both | T3: teach the per-cell and supply poses; their records and occupancy maps; the board-read gate; the sim demo; the owner's T3 decision; then the robot | each record passes; 0 moves past a `must_be_empty` cell in the sim |

**Open decisions (recommended defaults):**

| | Decision | Recommended default |
|---|---|---|
| OD-A1 | Speed caps | 10°/s average per joint (peak 18.75°/s), 0.5° per streamed goal; the first robot run of a route at 5°/s |
| OD-A2 | Arm torque limits | Keep the robot's own, as read at Stage 0 (a low limit sags the arm into the following abort mid-air); revisit from the first logs |
| OD-A3 | Tolerances | following 6° for 0.25 s (gross joints); read-back 3° critical, 8° on the weak joints; start 3° |
| OD-A4 | B2_PLACE's forearm 2 mm over the board (it touches in physics) | Accept it as a declared contact for Part B: the real arm flew it with a cube. The rehearsal checks it on the robot. |
| OD-A5 | Sim-provenance poses (REST, REST_OUT) in robot routes | Siva teaches the real REST and REST_OUT read-only first; then re-clear |
| OD-A6 | The robot's wrist roll (60.8°) and shoulder roll (32.9°) beyond the sim | No new sim deviation now; routes stay inside URDF + 20.3°, and the CalibrateBoard B2/B3 chains are not used |
| OD-A7 | StereoLauncher default | `--allow-arm-motion` is **not** passed by default, unlike the head and gripper: an explicit `--arm` flag |
| OD-A8 | Rig check for board-touching routes | Part B: the operator attests the board is on its marks, plus the empty-square rehearsal. T3: the camera rig check (needs the robot torso-frame conversion, perceived-scene §7). |
| OD-A9 | Capture format | Bump to `format_version: 2` (section 7.3) |
| OD-A10 | Staging or hand-over | Staging at Siva's supply spot; hand-over as the fallback (section 8.1) |
| OD-A11 | Part B slip check | Only with a Siva-taught `B2_LIFT_2CM`; no lift until then |
| OD-A12 | Sim-planned waypoints as robot routes | No for Phase 2: robot routes come from robot-taught poses |
| OD-A13 | The sim's grasp gap (section 2.6) | A sim issue: compare the sim's finger reach below `right_tip` with the real gripper (photos and a ruler); not blocking Part B |
| OD-A14 | What the e-stop does | Siva confirms (power cut, so the arm falls?) before R1; the operator rules stand either way |
