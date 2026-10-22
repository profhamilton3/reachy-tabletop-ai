"""Step 1, part 2: what decides a usable route (offline, the sim repo only read).

usage: python3 -I replay2.py <reachy-1-2-sim checkout> poses.json > replay2.txt
(a) per-geom height over the board top, analytic (box corners, capsule ends), at the low poses
(b) entry and exit legs between REST and TRANSPORT, with tall pieces on column 3 and a lab cube on r2c2:
    worst distance to each piece over the leg's interior, kinematic; plus a physics flight with the pieces free
(c) B2 at a lab cube (pool_box, 50.8 mm) on cell r2c2, gripper at Siva's open (-29): the pads against the cube
    along B2_SAFE -> B2_PLACE, and a close at B2_PLACE in physics
"""
import json, math, pathlib, sys
import numpy as np
import mujoco

exec(open(pathlib.Path(__file__).with_name("replay.py")).read().split("# ---- pose catalogue")[0])
catalogue = {}
for p in POSES["joint_poses"]:
    j = {n: p["joints"].get(n, 0.0) for n in ARM7}
    j["r_gripper"] = p["joints"].get("r_gripper")
    catalogue.setdefault(p["name"], j)
for name, q in (("REST", RR.REST), ("REST_OUT", RR.REST_OUT), ("PRESENT_SHUT", RR.PRESENT_SHUT), ("PRESENT", RR.PRESENT)):
    catalogue[name] = dict(q)

from reachy_ai.scene.awareness import SceneModel  # noqa: E402
sm = SceneModel.from_yaml(str(REPO / "scenes/FWDCenterLabSivaPool.yaml"))
HOME = {}


def body_adr(name):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    return m.jnt_qposadr[m.body_jntadr[b]]


def obj_geoms(name):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]


def place(layout):
    for oid, cell in layout.items():
        x, y, z = sm.cell_center(cell)
        a = body_adr(oid)
        d.qpos[a:a + 3] = [x, y, z + sm.get(oid).size[2] / 2 + 0.001]
        d.qpos[a + 3:a + 7] = [1, 0, 0, 0]


def park_all():
    core.reset()


def lowest_over_board():
    out = {}
    for g in ARM:
        pos, R = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
        s = m.geom_size[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            corners = [pos + R @ (np.array([sx, sy, sz]) * s) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            z = min(c[2] for c in corners)
        else:  # capsule: half-length along local z, radius s[0]
            ends = [pos + R @ np.array([0, 0, h * s[1]]) for h in (-1, 1)]
            z = min(e[2] for e in ends) - s[0]
        out[gname(g)] = (z - BOARD_TOP) * 100
    return out


def set_pose(name, grip=None):
    q = dict(catalogue[name])
    if grip is not None:
        q["r_gripper"] = grip
    set_q(q)


print("# (a) lowest point of each right-arm collision geom over the board top (cm), analytic; empty board")
for name in ("PICK", "B2_PLACE", "B2_PLACE@Cali128", "B2_PICK", "SUPPLY_SAFE", "SUPPLY_LOWER_7CM_JOINTS", "B2_APPROACH",
             "B3_LOWER_3CM_JOINTS", "REST"):
    for grip in (-29.0, -10.0):
        set_pose(name, grip)
        low = lowest_over_board()
        print(f"  {name:26s} grip {grip:+5.0f}: " + "  ".join(f"{k[2:-4]} {v:+5.1f}" for k, v in low.items()))
print()

# (b) entry/exit with pieces ----------------------------------------------------------------------------------
LAYOUTS = {
    "tall on column 3": {"soda_can": "cell_r1c3", "blue_cylinder": "cell_r2c3", "red_cube": "cell_r3c3"},
    "lab pieces on column 3": {"pool_cyl_1": "cell_r1c3", "pool_box_1": "cell_r2c3", "pool_cyl_2": "cell_r3c3"},
    "lab cube on r2c2": {"pool_box_1": "cell_r2c2"},
}
LEGS = {
    "REST -> TRANSPORT (direct)": ["REST", "TRANSPORT"],
    "REST -> HOME -> TRANSPORT": ["REST", "HOME_POSITION", "TRANSPORT"],
    "REST -> REST_OUT -> TRANSPORT": ["REST", "REST_OUT", "TRANSPORT"],
    "REST -> REST_OUT -> PRESENT_SHUT -> TRANSPORT": ["REST", "REST_OUT", "PRESENT_SHUT", "TRANSPORT"],
    "TRANSPORT -> REST (direct)": ["TRANSPORT", "REST"],
    "TRANSPORT -> REST_OUT -> REST": ["TRANSPORT", "REST_OUT", "REST"],
    "TRANSPORT -> PRESENT_SHUT -> REST_OUT -> REST": ["TRANSPORT", "PRESENT_SHUT", "REST_OUT", "REST"],
    "TRANSPORT -> B2_SAFE -> B2_PLACE (open -29)": ["TRANSPORT", "B2_SAFE", "B2_PLACE"],
    "REST -> REST_OUT -> PICK_SAFE -> PICK": ["REST", "REST_OUT", "PICK_SAFE", "PICK"],
}


def leg_worst(names, layout, grip, n=41):
    """Worst distance from the right arm to each placed piece over every leg, endpoints included (cm)."""
    worst = {oid: (9.0, None) for oid in layout}
    for a, b in zip(names, names[1:]):
        qa, qb = catalogue[a], catalogue[b]
        for i in range(n):
            s = i / (n - 1)
            q = {k: qa[k] + s * (qb[k] - qa[k]) for k in ARM7}
            q["r_gripper"] = grip
            set_q(q)
            for oid in layout:
                for og in obj_geoms(oid):
                    for ag in ARM:
                        v = distance(ag, og)
                        if v < worst[oid][0]:
                            worst[oid] = (v, f"{a}->{b} s={s:.2f} {gname(ag)}")
    return worst


print("# (b) entry and exit legs with pieces on the board (kinematic, joint-space lines; distance to each piece, cm)")
for lname, layout in LAYOUTS.items():
    print(f"## layout: {lname} {layout}")
    for leg, names in LEGS.items():
        grip = -29.0 if "open" in leg or "PICK" in leg else RR.SHUT
        park_all(); place(layout); mujoco.mj_forward(m, d)
        w = leg_worst(names, layout, grip)
        print(f"  {leg:48s} grip {grip:+4.0f}: " + "  ".join(f"{k} {v[0] * 100:+5.1f}" + (f" [{v[1]}]" if v[0] < 0.03 else "") for k, v in w.items()))
print()


def fly_with(names, secs, layout, grips, hold=1.0):
    park_all(); place(layout)
    first = dict(catalogue[names[0]]); first["r_gripper"] = grips[0]
    for n in NAMES:
        d.qpos[ADR[n]] = math.radians(first[n])
    d.qvel[:] = 0; mujoco.mj_forward(m, d); core.controller.sync_targets_to_current(d)
    for n in NAMES:
        core.controller.set_compliant("act_" + n, False)
    for _ in range(500):
        core.advance()
    start = {oid: d.qpos[body_adr(oid):body_adr(oid) + 3].copy() for oid in layout}
    cur = {n: math.degrees(d.qpos[ADR[n]]) for n in NAMES}
    touch = set()
    for name, t, g in zip(names[1:], secs, grips[1:]):
        tgt = dict(catalogue[name]); tgt["r_gripper"] = g
        st = dict(cur)
        steps = int(t / m.opt.timestep)
        for k in range(steps):
            s = (k + 1) / steps
            bl = s ** 3 * (10 - 15 * s + 6 * s * s)
            for n in NAMES:
                core.controller.set_goal_position("act_" + n, math.radians(st[n] + (tgt[n] - st[n]) * bl))
            core.advance()
            if k % 10 == 0:
                touch.update(c for c in contacts() if "table_top" not in c)
        for _ in range(int(hold / m.opt.timestep)):
            core.advance()
        cur = tgt
    moved = {oid: round(float(np.linalg.norm(d.qpos[body_adr(oid):body_adr(oid) + 3] - start[oid])) * 1000, 1)
             for oid in layout}
    return moved, sorted(touch)


print("# (b') physics: the same legs flown with the pieces free (moved = displacement of each piece, mm; contacts other than the board)")
for lname in ("tall on column 3", "lab pieces on column 3"):
    layout = LAYOUTS[lname]
    for leg, names, secs in (("REST -> REST_OUT -> TRANSPORT", ["REST", "REST_OUT", "TRANSPORT"], [3, 6]),
                             ("TRANSPORT -> REST_OUT -> REST", ["TRANSPORT", "REST_OUT", "REST"], [6, 3]),
                             ("REST -> TRANSPORT (direct)", ["REST", "TRANSPORT"], [8]),
                             ("TRANSPORT -> REST (direct)", ["TRANSPORT", "REST"], [8]),
                             ("REST -> HOME -> TRANSPORT", ["REST", "HOME_POSITION", "TRANSPORT"], [8, 6])):
        moved, touch = fly_with(names, secs, layout, [RR.SHUT] * len(names))
        print(f"  {lname:24s} {leg:34s} moved {moved}  contacts {touch if touch else '-'}")
print()

# (c) B2 at a lab cube ------------------------------------------------------------------------------------------
print("# (c) a 50.8 mm lab cube (pool_box_1) on r2c2; MovePieceLogic B2_SAFE -> B2_PLACE with the gripper at -29")
park_all(); place(LAYOUTS["lab cube on r2c2"]); mujoco.mj_forward(m, d)
cube = obj_geoms("pool_box_1")[0]
PADS = [gid("r_thumb_col"), gid("r_finger_col")]
for name in ("B2_SAFE", "B2_PLACE"):
    for grip in (-29.0, -45.0, -68.0):
        set_pose(name, grip)
        ds = {gname(g)[2:-4]: round(distance(g, cube) * 1000, 1) for g in ARM}
        thumb, finger = d.geom_xpos[PADS[0]], d.geom_xpos[PADS[1]]
        gap = float(np.linalg.norm(thumb - finger)) * 1000
        mid = (thumb + finger) / 2 - d.geom_xpos[cube]
        print(f"  {name:9s} grip {grip:+4.0f}: distance to the cube (mm) {ds}; pad-centre gap {gap:.0f} mm; pads' midpoint - cube centre (mm) {np.round(mid * 1000).tolist()}")
for grip in (-29.0, -45.0):
    w = leg_worst(["B2_SAFE", "B2_PLACE"], {"pool_box_1": "cell_r2c2"}, grip)
    print(f"  B2_SAFE -> B2_PLACE grip {grip:+4.0f}: worst distance to the cube {w['pool_box_1'][0] * 1000:+.1f} mm at {w['pool_box_1'][1]}")
for grip in (-29.0, -45.0):
    moved, touch = fly_with(["TRANSPORT", "B2_SAFE", "B2_PLACE"], [6, 5], {"pool_box_1": "cell_r2c2"}, [grip] * 3)
    print(f"  physics TRANSPORT -> B2_SAFE -> B2_PLACE, open {grip:+.0f}: cube moved {moved} mm; contacts {touch if touch else '-'}")
