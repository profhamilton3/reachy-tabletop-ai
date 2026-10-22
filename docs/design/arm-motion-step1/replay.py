"""Step 1 evidence for tabletop-ai #66: Siva's taught poses replayed in the sim's October model (offline).

usage: python3 -I replay.py <reachy-1-2-sim checkout> poses.json > replay.txt

The sim repo is only read: its model, scene and rig_routes are imported, nothing is written there.
No server, no SDK, no Docker. MuJoCo model = native_mujoco/model/reachy_1_2.xml compiled with
scenes/FWDCenterLabSivaPool.yaml (the panel's scene; the board starts empty, the pool on the floor).

Per pose (kinematic: qpos set, mj_forward):
  limits    against the MJCF joint ranges (= joint_map; r_shoulder_roll +20.3, the URDF's +10 noted)
  tip       the r_tip site (= URDF right_tip = SDK forward_kinematics) in the torso frame; height over the board top
  cells     nearest October cell (the model) and nearest taught cell (BOARD_POSES), xy offset
  clearance worst signed distance from the right arm's collision geoms to: board (table_top slab), rig (rails),
            self (torso, head), phantom left arm (non-colliding, measured anyway); plus contacts
Per segment (joint-space straight line = the SDK goto's path, 41 samples): the same worst clearances.
Per route (physics: motors on, minimum-jerk goals streamed at the notebook's durations, 500 Hz):
  contacts of any right-arm geom, arrival error per waypoint, worst following error.
"""
import json, math, pathlib, sys
import numpy as np
import mujoco

REPO = pathlib.Path(sys.argv[1])
POSES = json.load(open(sys.argv[2]))
sys.path[:0] = [str(REPO), str(REPO / "native_mujoco"), str(REPO / "src")]
from simulation_core import SimulationCore  # noqa: E402
from reachy_ai.motion import rig_routes as RR  # noqa: E402

core = SimulationCore.from_paths(str(REPO / "native_mujoco/model/reachy_1_2.xml"),
                                 str(REPO / "scenes/FWDCenterLabSivaPool.yaml"))
m, d = core.model, core.data
NAMES = ["r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw", "r_elbow_pitch", "r_forearm_yaw", "r_wrist_pitch",
         "r_wrist_roll", "r_gripper"]
ARM7 = NAMES[:7]
J = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in NAMES}
ADR = {n: m.jnt_qposadr[J[n]] for n in NAMES}
RANGE = {n: tuple(math.degrees(x) for x in m.jnt_range[J[n]]) for n in NAMES}
URDF_ROLL_MAX = 10.0
TIP = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "r_tip")
TORSO = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")


def gid(name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)


def gname(g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])


ARM = [gid(n) for n in ("r_upper_arm_col", "r_forearm_col", "r_thumb_col", "r_hand_back_col", "r_palm_col",
                        "r_finger_col")]
BOARD = [g for g in range(m.ngeom) if gname(g) == "table_top"]
RIG = [g for g in range(m.ngeom) if gname(g).startswith("rig_rail")]
SELF = [gid("torso_col"), gid("head_col")]
LEFT = [gid(n) for n in ("l_upper_arm_col", "l_forearm_col", "l_thumb_col", "l_finger_col")]
GROUPS = {"board": BOARD, "rig": RIG, "self": SELF, "left_arm": LEFT}
BOARD_TOP = 0.740

mujoco.mj_forward(m, d)
TORSO_POS = d.xpos[TORSO].copy()
TORSO_ROT = d.xmat[TORSO].reshape(3, 3).copy()
CELLS = {}
for g in range(m.ngeom):
    n = gname(g)
    if n.startswith("cell_r"):
        CELLS[n[5:]] = TORSO_ROT.T @ (d.geom_xpos[g] - TORSO_POS)
# Siva's names: A = far row (r3), C = near (r1); columns 1..3 = +y..-y (#185 finding 3)
SIVA = {f"{'CBA'[r - 1]}{c}": f"r{r}c{c}" for r in (1, 2, 3) for c in (1, 2, 3)}
AUG = {c: v + np.array([0.467 - 0.226, 0.015, 0]) for c, v in CELLS.items()}
AUG_SIVA = {c: v + np.array([0.47 - 0.226, 0.10, 0]) for c, v in CELLS.items()}
TAUGHT = {SIVA[k]: np.array(v)[:3, 3] for k, v in POSES["board_poses"].items()}


def set_q(q):
    for n in NAMES:
        d.qpos[ADR[n]] = math.radians(q.get(n, 0.0) if n != "r_gripper" else (q.get(n) if q.get(n) is not None else RR.SHUT))
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)


def tip_torso():
    return TORSO_ROT.T @ (d.site_xpos[TIP] - TORSO_POS), TORSO_ROT.T @ d.site_xmat[TIP].reshape(3, 3)


def distance(g1, g2):
    """Signed distance in metres; box-box pairs read with distmax <= 9 cm (a spurious 0 above ~10 cm)."""
    box = mujoco.mjtGeom.mjGEOM_BOX
    if m.geom_type[g1] == box and m.geom_type[g2] == box:
        for cap in (0.03, 0.06, 0.09):
            v = mujoco.mj_geomDistance(m, d, g1, g2, cap, None)
            if v < cap - 1e-9:
                if cap > 0.03 and v < 0.03:  # disagrees with the exact small-cap read: the artefact
                    break
                return v
        return 0.09
    return min(mujoco.mj_geomDistance(m, d, g1, g2, 0.3, None), 0.3)


def clearances():
    out = {}
    for k, gs in GROUPS.items():
        best = (9.0, None)
        for a in (ARM[1:] if k == "self" else ARM):
            for b in gs:
                v = distance(a, b)
                if v < best[0]:
                    best = (v, gname(a))
        out[k] = best
    return out


def contacts():
    pairs = set()
    for c in d.contact[:d.ncon]:
        a, b = gname(c.geom1), gname(c.geom2)
        if a.startswith("r_") or b.startswith("r_"):
            pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


def limits(q):
    bad = []
    for n in ARM7:
        lo, hi = RANGE[n]
        if not lo - 0.1 <= q[n] <= hi + 0.1:
            bad.append(f"{n} {q[n]:+.1f} outside [{lo:.1f}, {hi:.1f}]")
    note = q["r_shoulder_roll"] > URDF_ROLL_MAX + 0.1
    return bad, note


def cell_offset(p, table):
    name = min(table, key=lambda c: np.hypot(*(p[:2] - table[c][:2])))
    dx, dy = (p[:2] - table[name][:2]) * 1000
    return name, dx, dy


def fmt_cl(cl):
    return "  ".join(f"{k} {v[0] * 100:+.1f}" + ("" if v[0] < 0.089 or k == "self" else "+") for k, v in cl.items())


# ---- pose catalogue ---------------------------------------------------------------------------------------
catalogue = {}
for p in POSES["joint_poses"]:
    j = {n: p["joints"].get(n, 0.0) for n in ARM7}
    j["r_gripper"] = p["joints"].get("r_gripper")
    key = tuple(round(j[n], 4) for n in ARM7)
    if any(tuple(round(v[n], 4) for n in ARM7) == key for v in catalogue.values()):
        continue
    catalogue[p["name"]] = j
for name, q in (("REST", RR.REST), ("REST_OUT", RR.REST_OUT), ("PRESENT_SHUT", RR.PRESENT_SHUT),
                ("PRESENT", RR.PRESENT)):
    catalogue[name] = dict(q)

print("# model check: URDF FK of B2_SAFE_JOINTS vs BOARD_POSES[B2] + 0.15 m (#185 finding 1)")
set_q(catalogue["B2_SAFE_JOINTS"])
p, R = tip_torso()
ref = np.array(POSES["board_poses"]["B2"])
want = ref[:3, 3] + [0, 0, 0.15]
ang = math.degrees(math.acos(max(-1, min(1, (np.trace(R.T @ ref[:3, :3]) - 1) / 2))))
print(f"  MJCF r_tip {np.round(p, 4).tolist()} want {np.round(want, 4).tolist()} -> {np.linalg.norm(p - want) * 1000:.2f} mm, {ang:.2f} deg")
print(f"  torso origin in the scene {np.round(TORSO_POS, 3).tolist()}; board top z {BOARD_TOP}")
print(f"  October cells, torso frame: " + ", ".join(f"{c} ({v[0]:.3f}, {v[1]:+.3f})" for c, v in sorted(CELLS.items())))
print()

print("# POSES  (distances in cm: worst signed distance from the right arm; + clear, - inside; '+' after 8.9 = at least 9)")
print(f"{'pose':28s} {'limits':6s} {'tip torso (m)':24s} {'tip over board':>8s}  {'nearest Oct cell (dx,dy mm)':30s} {'nearest taught cell':26s} clearances; contacts")
rows = {}
for name, q in catalogue.items():
    set_q(q)
    bad, roll_note = limits(q)
    p, _ = tip_torso()
    over = (TORSO_POS[2] + p[2] - BOARD_TOP) * 100
    oc = cell_offset(p, CELLS)
    tc = cell_offset(p, TAUGHT)
    ac = cell_offset(p, AUG); asv = cell_offset(p, AUG_SIVA)
    cl = clearances()
    cons = contacts()
    rows[name] = dict(bad=bad, roll=roll_note, tip=p, over=over, oc=oc, tc=tc, cl=cl, cons=cons, ac=ac, asv=asv)
    lim = "ok" if not bad else "OUT"
    lim += "*" if roll_note else ""
    print(f"{name:28s} {lim:6s} ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})   {over:+6.1f} cm  {oc[0]} ({oc[1]:+5.0f},{oc[2]:+5.0f})            {tc[0]} ({tc[1]:+5.0f},{tc[2]:+5.0f})          {fmt_cl(cl)}; {cons if cons else '-'}")
    for b in bad:
        print(f"{'':30s}!! {b}")
print()
print("# ERA: nearest cell centre (xy, mm) under the October model, Aug 20 camera board (#185: centre 0.467, 0.015) and Siva's late-August board (0.47, 0.10)")
for name, r in rows.items():
    dist = lambda c: (c[0], round(float(np.hypot(c[1], c[2]))))
    print(f"  {name:28s} tip over board {r['over']:+6.1f} cm   October {dist(r['oc'])}   Aug-20 {dist(r['ac'])}   late-Aug {dist(r['asv'])}")
print("  * = r_shoulder_roll above the URDF's +10 deg (inside the sim's +20.3, ADR-0010 section 4)")
print()

# ---- taught board poses: IK in the model, as taught (tip just over the surface) and at +8 cm -------------
def ik(target, seeds, iters=400):
    """Damped least squares on the r_tip site, position + orientation, clamped to the MJCF ranges."""
    best = None
    for seed in seeds:
        q = {n: seed[n] for n in ARM7}
        for _ in range(iters):
            set_q({**q, "r_gripper": RR.SHUT})
            p = d.site_xpos[TIP].copy()
            R = d.site_xmat[TIP].reshape(3, 3)
            tp = TORSO_POS + TORSO_ROT @ target[:3, 3]
            tR = TORSO_ROT @ target[:3, :3]
            e_p = tp - p
            Re = tR @ R.T
            e_r = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, TIP)
            cols = [m.jnt_dofadr[J[n]] for n in ARM7]
            Jm = np.vstack([jp[:, cols], 0.3 * jr[:, cols]])
            e = np.concatenate([e_p, 0.3 * e_r])
            dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-4 * np.eye(6), e)
            for n, v in zip(ARM7, dq):
                lo, hi = RANGE[n]
                q[n] = float(np.clip(q[n] + math.degrees(v), lo, hi))
            if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3:
                break
        res = (np.linalg.norm(e_p) * 1000, math.degrees(np.linalg.norm(e_r) * 2))
        if best is None or res[0] + res[1] < best[1][0] + best[1][1]:
            best = (dict(q), res)
    return best


seeds = [catalogue[n] for n in ("B2_SAFE_JOINTS", "B2_PICK", "B2_SAFE_ALIGNED", "B3_SAFE_JOINTS", "PICK",
                                "SUPPLY_LOWER_7CM_JOINTS", "B2_TRANSPORT", "B2_PLACE", "B2_PLACE@Cali128")]
print("# TAUGHT BOARD POSES (CalibrateBoard cell 6), solved in the model: IK on r_tip, position and orientation")
print(f"{'square':12s} {'height':7s} {'IK residual':16s} {'limits':7s} {'r_shoulder_roll':>6s}  {'tip over board':>8s} {'nearest Oct cell (dx,dy mm)':30s} clearances; contacts")
board_rows = {}
for sq in sorted(POSES["board_poses"]):
    for lift in (0.0, 0.08):
        T = np.array(POSES["board_poses"][sq]); T[2, 3] += lift
        q, res = ik(T, seeds)
        set_q({**q, "r_gripper": RR.SHUT})
        bad, roll_note = limits(q)
        p, _ = tip_torso()
        over = (TORSO_POS[2] + p[2] - BOARD_TOP) * 100
        oc = cell_offset(p, CELLS)
        cl = clearances(); cons = contacts()
        board_rows[(sq, lift)] = dict(q=q, res=res, cl=cl, cons=cons, over=over, oc=oc)
        ok = res[0] < 1.0 and res[1] < 1.0
        print(f"{sq}={SIVA[sq]:6s} +{lift * 100:2.0f} cm  {res[0]:5.1f} mm {res[1]:4.1f} deg  {'ok' if ok else 'NO FIT':7s} {q['r_shoulder_roll']:+6.1f}  {over:+6.1f} cm  {oc[0]} ({oc[1]:+5.0f},{oc[2]:+5.0f})            {fmt_cl(cl)}; {cons if cons else '-'}")
print()

# ---- segments -----------------------------------------------------------------------------------------------
def segment(a, b, n=41):
    worst = {k: (9.0, None, None) for k in GROUPS}
    low = (9.0, None)
    touch = set()
    for i in range(n):
        s = i / (n - 1)
        q = {k: a[k] + s * (b[k] - a[k]) for k in ARM7}
        ga, gb = a.get("r_gripper"), b.get("r_gripper")
        q["r_gripper"] = ga if gb is None else (gb if ga is None else ga + s * (gb - ga))
        set_q(q)
        for k, v in clearances().items():
            if v[0] < worst[k][0]:
                worst[k] = (v[0], v[1], s)
        p, _ = tip_torso()
        over = TORSO_POS[2] + p[2] - BOARD_TOP
        if over < low[0]:
            low = (over, s)
        touch.update(contacts())
    return worst, low, sorted(touch)


ROUTES = {
    # tests/MovePieceLogic.ipynb cells 1-4: the route #27 Part B names. Durations as in the notebook.
    "MPL pick-and-place": [("PICK_SAFE", 6), ("PICK", 5), ("PICK_SAFE", 5), ("TRANSPORT", 6), ("B2_SAFE", 6),
                           ("B2_PLACE", 5), ("B2_SAFE", 5), ("TRANSPORT", 6)],
    # entries and exits from the only motors-off pose (REST), none of them taught
    "REST -> PICK_SAFE (direct)": [("REST", 0), ("PICK_SAFE", 6)],
    "REST -> HOME (direct)": [("REST", 0), ("HOME_POSITION", 8)],
    "HOME -> PICK_SAFE": [("HOME_POSITION", 0), ("PICK_SAFE", 6)],
    "TRANSPORT -> HOME": [("TRANSPORT", 0), ("HOME_POSITION", 6)],
    "HOME -> REST (direct)": [("HOME_POSITION", 0), ("REST", 8)],
    "TRANSPORT -> REST (direct)": [("TRANSPORT", 0), ("REST", 8)],
    "REST -> REST_OUT -> PICK_SAFE": [("REST", 0), ("REST_OUT", 3), ("PICK_SAFE", 6)],
    "PRESENT -> PICK_SAFE": [("PRESENT", 0), ("PICK_SAFE", 6)],
    "TRANSPORT -> PRESENT": [("TRANSPORT", 0), ("PRESENT", 6)],
    # CalibrateBoard.ipynb chains, in the notebook's order (recorded poses only; the IK steps between are not)
    "CB HOME -> B2_APPROACH": [("HOME_POSITION", 0), ("B2_APPROACH", 8)],
    "CB HOME -> B2_SAFE_JOINTS": [("HOME_POSITION", 0), ("B2_SAFE_JOINTS", 8)],
    "CB B2 pick -> B3 place -> HOME": [("B2_SAFE_ALIGNED", 0), ("B2_PICK", 5), ("B2_LIFT_1CM", 3), ("B2_LIFT_4CM", 4),
                                       ("B2_SAFE_WITH_CUBE", 5), ("B3_SAFE_JOINTS", 6), ("B3_LOWER_3CM_JOINTS", 5),
                                       ("B3_RETREAT", 5), ("HOME_POSITION@Cali82", 7)],
    "CB supply pick -> B2 high": [("SUPPLY_HIGH", 0), ("SUPPLY_SAFE@Cali94", 6), ("SUPPLY_LOWER_3CM_JOINTS", 7),
                                  ("SUPPLY_LOWER_4CM_JOINTS", 4), ("SUPPLY_PICK", 5), ("SUPPLY_LIFT_4CM", 4),
                                  ("SUPPLY_SAFE@Cali94", 5), ("B2_TRANSPORT", 8), ("TRANSPORT_HIGH", 6),
                                  ("MIDPOINT_WITH_CUBE", 6), ("B2_HIGH_WITH_CUBE", 6)],
    "CB TRANSPORT -> B2 place -> exit -> HOME -> supply": [
        ("TRANSPORT", 0), ("B2_SAFE@Cali118", 6), ("DOWN_2CM", 4), ("DOWN_NEXT_2CM", 4), ("B2_PLACE@Cali128", 9),
        ("UP_2CM", 9), ("UP_6CM", 5), ("B2_EXIT", 5), ("TRANSPORT_OPEN", 7), ("HOME_POSITION@Cali140", 6),
        ("SUPPLY_SAFE@Cali141", 6), ("SUPPLY_INTERMEDIATE", 5), ("SUPPLY_PICK_LOWER", 4),
        ("SUPPLY_PICK_OLD_WITH_CUBE", 4), ("SUPPLY_UP_3CM", 5), ("SUPPLY_UP_6CM", 6)],
    "LIFT_TO_PRESENT": [("REST", 0), ("REST_OUT", 3), ("PRESENT_SHUT", 3), ("PRESENT", 1.5)],
    "LOWER_TO_REST": [("PRESENT", 0), ("PRESENT_SHUT", 1.5), ("REST_OUT", 3), ("REST", 3)],
}
ALIAS = {"B2_SAFE_WITH_CUBE": "B2_SAFE_ALIGNED", "B3_RETREAT": "B3_SAFE_JOINTS",
         "HOME_POSITION@Cali82": "HOME_POSITION", "HOME_POSITION@Cali140": "HOME_POSITION",
         "SUPPLY_PICK": "SUPPLY_LOWER_7CM_JOINTS", "SUPPLY_LIFT_4CM": "SUPPLY_LOWER_4CM_JOINTS",
         "TRANSPORT_OPEN": "TRANSPORT", "SUPPLY_SAFE@Cali141": "SUPPLY_SAFE@Cali94",
         "SUPPLY_INTERMEDIATE": "SUPPLY_LOWER_4CM_JOINTS", "SUPPLY_PICK_OLD_WITH_CUBE": "SUPPLY_LOWER_7CM_JOINTS"}
GRIP = {"B2_SAFE_WITH_CUBE": -10.0, "B3_RETREAT": -29.0, "HOME_POSITION@Cali82": -29.0, "SUPPLY_LIFT_4CM": -10.0,
        "TRANSPORT_OPEN": -29.0, "HOME_POSITION@Cali140": -29.0, "SUPPLY_SAFE@Cali141": -29.0,
        "SUPPLY_INTERMEDIATE": -29.0, "SUPPLY_PICK_OLD_WITH_CUBE": -7.0}


def resolve(name):
    q = dict(catalogue[ALIAS.get(name, name)])
    if name in GRIP:
        q["r_gripper"] = GRIP[name]
    return q


print("# SEGMENTS: joint-space straight line between taught poses (the SDK goto path), kinematic, 41 samples")
print("  worst signed distance (cm) per obstacle group, at fraction s of the segment; lowest tip height over the board")
seg_rows = {}
for route, legs in ROUTES.items():
    print(f"## {route}")
    for (a, _), (b, secs) in zip(legs, legs[1:]):
        w, low, touch = segment(resolve(a), resolve(b))
        seg_rows[(route, a, b)] = (w, low, touch)
        parts = "  ".join(f"{k} {v[0] * 100:+.1f}" + ("+" if v[0] >= 0.089 and k != "self" else "") +
                          (f" ({v[1]}, s={v[2]:.2f})" if v[0] < 0.03 else "") for k, v in w.items())
        print(f"  {a:26s} -> {b:26s} {secs:>4}s  {parts}; tip low {low[0] * 100:+.1f} cm; contacts {touch if touch else '-'}")
print()

# ---- physics flights ---------------------------------------------------------------------------------------
def fly(legs, hold_s=2.0):
    core.reset()
    first = resolve(legs[0][0])
    grip = first.get("r_gripper") if first.get("r_gripper") is not None else -29.0
    set_q({**first, "r_gripper": grip})
    core.controller.sync_targets_to_current(d)
    for n in NAMES:
        core.controller.set_compliant("act_" + n, False)
    for _ in range(500):
        core.advance()
    out = []
    cur = {n: math.degrees(d.qpos[ADR[n]]) for n in NAMES}
    cur["r_gripper"] = grip
    for name, secs in legs[1:]:
        tgt = resolve(name)
        g_t = tgt.get("r_gripper") if tgt.get("r_gripper") is not None else cur["r_gripper"]
        tgt["r_gripper"] = g_t
        steps = max(1, int(secs / m.opt.timestep))
        follow = 0.0
        touch = set()
        start = dict(cur)
        for k in range(steps):
            s = (k + 1) / steps
            blend = s ** 3 * (10 - 15 * s + 6 * s * s)
            goal = {n: start[n] + (tgt[n] - start[n]) * blend for n in NAMES}
            for n in NAMES:
                core.controller.set_goal_position("act_" + n, math.radians(goal[n]))
            core.advance()
            if k % 10 == 0:
                follow = max(follow, max(abs(math.degrees(d.qpos[ADR[n]]) - goal[n]) for n in ARM7[:4]))
                touch.update(contacts())
        for _ in range(int(hold_s / m.opt.timestep)):
            core.advance()
        touch.update(contacts())
        err = {n: math.degrees(d.qpos[ADR[n]]) - tgt[n] for n in ARM7}
        cur = dict(tgt)
        out.append((name, secs, follow, err, sorted(touch)))
    return out


print("# PHYSICS FLIGHTS: motors on, minimum-jerk goals streamed, the notebook's durations, 2 s hold per waypoint")
print("  following = worst |present - streamed goal| on the four gross joints during the leg (deg)")
print("  arrival   = present - target after the hold, per joint (deg): sh_pitch sh_roll arm_yaw elbow fore_yaw wr_pitch wr_roll")
for route, legs in ROUTES.items():
    print(f"## {route}  (from {legs[0][0]})")
    for name, secs, follow, err, touch in fly(legs):
        e = " ".join(f"{err[n]:+5.1f}" for n in ARM7)
        print(f"  -> {name:26s} {secs:>4}s  following {follow:5.1f}  arrival [{e}]  contacts {touch if touch else '-'}")
