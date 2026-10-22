"""Step 1, part 3: a lab cube at each taught grasp point (offline; the sim repo only read).

usage: python3 -I replay3.py <reachy-1-2-sim checkout> poses.json > replay3.txt
The "grasp point" of a pose = the midpoint of the two pad geoms (r_thumb_col, r_finger_col) at that pose with
the gripper at Siva's open (-29), dropped to the board (cube centre 25.4 mm up), the cube turned to the jaw axis.
For each: the descent (kinematic, joint line) against that cube, a physics close at the pose (does the cube end
up between the pads, does the close hold, does the lift carry it), and REST / LIFT_TO_PRESENT / LOWER_TO_REST
against a cube standing there.
"""
import math, pathlib
import numpy as np
import mujoco

exec(open(pathlib.Path(__file__).with_name("replay2.py")).read().split("print(\"# (a)")[0])

CUBE = "pool_box_1"
PADS = [gid("r_thumb_col"), gid("r_finger_col")]
cube_g = obj_geoms(CUBE)[0]
H = 0.0254


def grasp_point(name):
    set_pose(name, -29.0)
    a, b = d.geom_xpos[PADS[0]].copy(), d.geom_xpos[PADS[1]].copy()
    mid = (a + b) / 2
    axis = b - a
    yaw = math.atan2(axis[1], axis[0])
    return mid, yaw, float(np.linalg.norm(axis))


def put_cube(xy, yaw):
    park_all()
    a = body_adr(CUBE)
    d.qpos[a:a + 3] = [xy[0], xy[1], BOARD_TOP + H + 0.0005]
    d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(m, d)


def line_worst(names, grip, n=61):
    worst = (9.0, None)
    for a_, b_ in zip(names, names[1:]):
        qa, qb = catalogue[a_], catalogue[b_]
        for i in range(n):
            s = i / (n - 1)
            q = {k: qa[k] + s * (qb[k] - qa[k]) for k in ARM7}
            q["r_gripper"] = grip
            set_q(q)
            for g in ARM:
                v = distance(g, cube_g)
                if v < worst[0]:
                    worst = (v, f"{a_}->{b_} s={s:.2f} {gname(g)}")
    return worst


def fly_grasp(names, secs, grips, xy, yaw):
    put_cube(xy, yaw)
    first = dict(catalogue[names[0]]); first["r_gripper"] = grips[0]
    for nn in NAMES:
        d.qpos[ADR[nn]] = math.radians(first[nn])
    d.qvel[:] = 0; mujoco.mj_forward(m, d); core.controller.sync_targets_to_current(d)
    for nn in NAMES:
        core.controller.set_compliant("act_" + nn, False)
    for _ in range(500):
        core.advance()
    a = body_adr(CUBE)
    start = d.qpos[a:a + 3].copy()
    cur = {nn: math.degrees(d.qpos[ADR[nn]]) for nn in NAMES}
    log = []
    for name, t, g in zip(names[1:], secs, grips[1:]):
        tgt = dict(catalogue[name]); tgt["r_gripper"] = g
        st = dict(cur)
        steps = int(t / m.opt.timestep)
        for k in range(steps):
            s = (k + 1) / steps
            bl = s ** 3 * (10 - 15 * s + 6 * s * s)
            for nn in NAMES:
                core.controller.set_goal_position("act_" + nn, math.radians(st[nn] + (tgt[nn] - st[nn]) * bl))
            core.advance()
        for _ in range(int(1.0 / m.opt.timestep)):
            core.advance()
        pads = sorted({gname(c.geom1 if c.geom2 == cube_g else c.geom2)
                       for c in d.contact[:d.ncon] if cube_g in (c.geom1, c.geom2)})
        lift = (d.qpos[a + 2] - start[2]) * 1000
        log.append(f"{name}(grip {g:+.0f}): cube dz {lift:+.1f} mm, dxy {np.linalg.norm(d.qpos[a:a + 2] - start[:2]) * 1000:.1f} mm, "
                   f"gripper {math.degrees(d.qpos[ADR['r_gripper']]):+.1f}, touching {pads}")
        cur = tgt
    return log


CASES = {
    "MovePieceLogic PICK (right margin)": ("PICK_SAFE", "PICK", [6, 5], ["PICK_SAFE"]),
    "MovePieceLogic B2_PLACE (r2c2)": ("B2_SAFE", "B2_PLACE", [6, 5], ["TRANSPORT", "B2_SAFE"]),
    "CalibrateBoard B2_PICK (r2c2)": ("B2_SAFE_ALIGNED", "B2_PICK", [5, 3], ["B2_SAFE_ALIGNED"]),
    "CalibrateBoard SUPPLY_PICK (right margin)": ("SUPPLY_LOWER_4CM_JOINTS", "SUPPLY_LOWER_7CM_JOINTS", [4, 4], ["SUPPLY_LOWER_4CM_JOINTS"]),
}
print("# a 50.8 mm lab cube at each taught grasp point")
for label, (safe, low, secs, before) in CASES.items():
    mid, yaw, gap = grasp_point(low)
    xy = mid[:2]
    tp = TORSO_ROT.T @ (np.array([xy[0], xy[1], BOARD_TOP]) - TORSO_POS)
    oc = cell_offset(tp, CELLS)
    print(f"## {label}: grasp point torso ({tp[0]:+.3f}, {tp[1]:+.3f}) = {oc[0]} {oc[1]:+.0f},{oc[2]:+.0f} mm; pad midpoint {((mid[2] - BOARD_TOP) * 1000):.0f} mm over the board (cube centre at 25); pad-centre gap {gap * 1000:.0f} mm at -29; jaw yaw {math.degrees(yaw):+.0f} deg")
    put_cube(xy, yaw)
    w = line_worst([safe, low], -29.0)
    print(f"   descent {safe} -> {low}, open -29, kinematic: worst distance to the cube {w[0] * 1000:+.1f} mm at {w[1]}")
    for close in (-10.0, -7.0):
        names = [safe, low, low, safe]
        log = fly_grasp(names, [secs[0], 2.0, secs[1]], [-29.0, -29.0, close, close], xy, yaw)
        print(f"   physics {safe} -> {low} (open) -> close {close:+.0f} -> lift to {safe}:")
        for line in log:
            print("      " + line)
    put_cube(xy, yaw)
    for route in ("LIFT_TO_PRESENT", "LOWER_TO_REST"):
        names = [w_.name for w_ in getattr(RR, route)]
        names = (["REST"] if route == "LIFT_TO_PRESENT" else ["PRESENT"]) + names
        ww = line_worst(names, RR.SHUT)
        print(f"   {route} against a cube standing here: worst {ww[0] * 100:+.1f} cm at {ww[1]}")
