"""Step 1, part 4: the candidate Part B route, end to end (offline; the sim repo only read).

usage: python3 -I replay4.py <reachy-1-2-sim checkout> poses.json > replay4.txt
ROUTE B2_GRIP_TUNE = REST -> REST_OUT -> TRANSPORT -> B2_SAFE -> B2_PLACE -> B2_SAFE -> TRANSPORT -> REST_OUT -> REST
the hand shut (+20) except B2_SAFE -> B2_PLACE -> B2_SAFE, open at -29; durations at a 10 deg/s average cap.
Kinematic clearances per leg, then physics with four layouts.
"""
import math, pathlib
import numpy as np
import mujoco

exec(open(pathlib.Path(__file__).with_name("replay2.py")).read().split("print(\"# (a)")[0])

_src = open(pathlib.Path(__file__).with_name("replay.py")).read()
exec("def segment" + _src.split("def segment", 1)[1].split("\n\n\nROUTES")[0])

ROUTE = [("REST", 20.0, 0), ("REST_OUT", 20.0, 4.0), ("TRANSPORT", 20.0, 8.4), ("B2_SAFE", -29.0, 6.0),
         ("B2_PLACE", -29.0, 5.0), ("B2_SAFE", -29.0, 5.0), ("TRANSPORT", 20.0, 6.0), ("REST_OUT", 20.0, 8.4),
         ("REST", 20.0, 4.0)]

print("# B2_GRIP_TUNE, kinematic, empty board: worst signed distance per group (cm), lowest tip height over the board (cm)")
for (a, ga, _), (b, gb, t) in zip(ROUTE, ROUTE[1:]):
    qa = dict(catalogue[a], r_gripper=ga); qb = dict(catalogue[b], r_gripper=gb)
    w, low, touch = segment(qa, qb)
    interior = "  ".join(f"{k} {v[0] * 100:+.1f}" + ("+" if v[0] >= 0.089 and k != "self" else "") for k, v in w.items())
    print(f"  {a:9s} -> {b:9s} {t:4.1f} s  {interior}; tip low {low[0] * 100:+.1f}; contacts {touch if touch else '-'}")

LAYOUTS = {
    "empty board": {},
    "lab pieces on r2c3, r3c3, r2c1, r3c1 (r1c3 empty)": {"pool_box_1": "cell_r2c3", "pool_cyl_1": "cell_r3c3",
                                                         "pool_box_2": "cell_r2c1", "pool_cyl_2": "cell_r3c1"},
    "lab cylinder on r1c3": {"pool_cyl_1": "cell_r1c3"},
    "tall pieces on column 3": {"soda_can": "cell_r1c3", "blue_cylinder": "cell_r2c3", "red_cube": "cell_r3c3"},
}


def fly_route(layout):
    park_all(); place(layout)
    first = dict(catalogue["REST"], r_gripper=20.0)
    for n in NAMES:
        d.qpos[ADR[n]] = math.radians(first[n])
    d.qvel[:] = 0; mujoco.mj_forward(m, d); core.controller.sync_targets_to_current(d)
    for n in NAMES:
        core.controller.set_compliant("act_" + n, False)
    for _ in range(500):
        core.advance()
    start = {oid: d.qpos[body_adr(oid):body_adr(oid) + 3].copy() for oid in layout}
    cur = {n: math.degrees(d.qpos[ADR[n]]) for n in NAMES}
    rows = []
    for name, g, t in ROUTE[1:]:
        tgt = dict(catalogue[name], r_gripper=g)
        st = dict(cur); steps = int(t / m.opt.timestep); follow = 0.0; touch = set()
        for k in range(steps):
            s = (k + 1) / steps; bl = s ** 3 * (10 - 15 * s + 6 * s * s)
            goal = {n: st[n] + (tgt[n] - st[n]) * bl for n in NAMES}
            for n in NAMES:
                core.controller.set_goal_position("act_" + n, math.radians(goal[n]))
            core.advance()
            if k % 10 == 0:
                follow = max(follow, max(abs(math.degrees(d.qpos[ADR[n]]) - goal[n]) for n in ARM7[:4]))
                touch.update(contacts())
        for _ in range(int(1.0 / m.opt.timestep)):
            core.advance()
        err = max(abs(math.degrees(d.qpos[ADR[n]]) - tgt[n]) for n in ARM7[:4] + ["r_wrist_pitch"])
        rows.append((name, follow, err, sorted(touch)))
        cur = tgt
    moved = {oid: round(float(np.linalg.norm(d.qpos[body_adr(oid):body_adr(oid) + 3] - start[oid])) * 1000, 1)
             for oid in layout}
    return rows, moved


print()
print("# B2_GRIP_TUNE, physics: worst following (gross four), arrival error (gross four + wrist pitch), contacts; pieces moved (mm)")
for label, layout in LAYOUTS.items():
    rows, moved = fly_route(layout)
    print(f"## {label}: moved {moved if moved else '-'}")
    for name, follow, err, touch in rows:
        print(f"   -> {name:9s} following {follow:4.1f}  arrival {err:4.1f}  contacts {touch if touch else '-'}")
